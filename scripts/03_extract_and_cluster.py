#!/usr/bin/env python3
"""The GPU pod's single job: extract -> cluster -> write assignments -> discard.

DESIGN.md §A.1 (never persist raw activations; checkpoint after every layer),
§A.4 (batch size 1, no padding), §A.6 (aligned positions; k selected once on
nl_base and reused across all four realizations), §A.11 items 1-4 and 6,
prereg gates G3 (drop rate) and G4 (Nystrom approximation error), and
prereg/DEVIATIONS.md Amendment 3 (cosine affinity).

    corpus record  --hooks.extract_problem-->  aligned activations (RAM only)
                   --ling.cluster.cluster_layer-->  per-layer partition
                   --checkpoint per layer-->  results/<run_id>/layer_NNN.json
                   --assemble-->  results/<run_id>/assignments.jsonl
                   --store.close()-->  activations gone

Nothing else leaves the pod. `assignments.jsonl` is a few hundred KB and is the
only input scripts/04_analyze.py takes (configs/assignments_schema.md).

Extraction is ALL-FOUR-OR-NONE per problem: a typed `AnchorDropError` from any
one condition drops the whole problem and is tallied into the G3 drop rate.
Keeping the three conditions that resolved would leave the pooled graphs with
different node counts per condition, which is the §A.6 graph-size confound
landing directly on H3's endpoint.

Two fields beyond configs/assignments_schema.md are written on every row,
because scripts/04_analyze.py needs them and will not fabricate them:

  * `kmeans_cluster`    -- the K-means comparator at the same k, same layer,
                           same states (H1's descriptive note, Amendment 1).
  * `linguistic_label`  -- OPTIONAL PASSTHROUGH from the corpus record, `null`
                           when the corpus does not carry one. This script does
                           NOT invent labels; the label set is another agent's
                           output and an invented one would silently become the
                           thing H1's AMI is scored against.

Usage:
    python scripts/03_extract_and_cluster.py --run-id pod01
    python scripts/03_extract_and_cluster.py --run-id pod01 --max-problems 8
    python scripts/03_extract_and_cluster.py --run-id pod01 --affinity per_layer_standardized
    python scripts/03_extract_and_cluster.py --run-id pod01 --assemble-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

# MUST be the first project import: it puts CUBLAS_WORKSPACE_CONFIG into the
# environment before torch creates a CUDA context. See ling/determinism.py.
from ling.determinism import enforce_determinism  # noqa: E402
from ling import cluster as C  # noqa: E402
from ling.hooks import (  # noqa: E402
    AlignedActivationStore,
    AnchorDropError,
    LayerCheckpointWriter,
    anchor_order,
    extract_problem,
    stream_layers_with_checkpoints,
)

DEFAULT_CORPUS = os.path.join(_REPO, "data", "corpus", "problems.jsonl")
DEFAULT_RESULTS = os.path.join(_REPO, "results")
CONDITIONS = ("formal", "nl_base", "nl_break", "nl_para")


# ======================================================================================
# corpus
# ======================================================================================


def load_corpus(path, max_problems=None):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise SystemExit("%s:%d is not valid JSON: %s" % (path, lineno, exc))
            if max_problems is not None and len(records) >= max_problems:
                break
    if not records:
        raise SystemExit("%s contains no records" % path)
    return records


def linguistic_label(record, condition, anchor):
    """OPTIONAL passthrough of a corpus-supplied linguistic label. Never invented.

    Looked up in order of specificity, most specific first:

        realizations[cond]["linguistic_labels"][anchor]
        realizations[cond]["linguistic_label"]
        record["linguistic_labels"][cond][anchor]
        record["linguistic_labels"][anchor]
        record["linguistic_label"]

    and `None` if none of those exist. H1's AMI is scored against this field, so
    a fabricated fallback (POS guessed from the anchor name, the anchor name
    itself, the condition) would be the pipeline scoring its own construction —
    Amendment 1 reason 2 with the label set chosen by whoever wrote the default.
    `None` propagates to `null` in the JSON. Note that scripts/04_analyze.py:159
    gates H1 on the FIELD being PRESENT, not on it being non-null, so an all-null
    column makes it report AMI 0.0 vs 0.0 rather than "unavailable"; the assembly
    step logs a loud WARNING when that is the case.
    """
    real = (record.get("realizations") or {}).get(condition) or {}
    labels = real.get("linguistic_labels")
    if isinstance(labels, dict) and anchor in labels:
        return labels[anchor]
    if "linguistic_label" in real:
        return real["linguistic_label"]
    top = record.get("linguistic_labels")
    if isinstance(top, dict):
        per_cond = top.get(condition)
        if isinstance(per_cond, dict) and anchor in per_cond:
            return per_cond[anchor]
        if anchor in top and not isinstance(top.get(anchor), dict):
            return top[anchor]
    if "linguistic_label" in record:
        return record["linguistic_label"]
    return None


# ======================================================================================
# extraction  (§A.1, §A.4, §A.6; gate G3)
# ======================================================================================


def extract_all(records, loaded, *, conditions, dtype, spill_dir, log):
    """Fill an AlignedActivationStore, all-four-or-none, tallying G3 drops.

    Returns ``(store, row_meta, drops)`` where ``row_meta`` is one dict per store
    row carrying the row's identity and its (optional) linguistic label, in the
    SAME order as ``store.rows``.
    """
    store = AlignedActivationStore(loaded.n_layers, loaded.d_model, dtype=dtype,
                                   spill_dir=spill_dir)
    row_meta = []
    drops = []
    kept = 0
    t0 = time.time()
    for rec in records:
        pid = rec.get("problem_id")
        missing = [c for c in conditions if c not in (rec.get("realizations") or {})]
        if missing:
            drops.append({"problem_id": pid, "condition": ",".join(missing),
                          "reason": "missing_realization"})
            log("drop %s: missing realization(s) %s" % (pid, missing))
            continue
        try:
            arrays, metas, names = extract_problem(
                rec, loaded.tokenizer, loaded, conditions=list(conditions), dtype=dtype)
        except AnchorDropError as exc:
            # All four conditions go, or none (§A.6). The tally is gate G3.
            drops.append({"problem_id": exc.problem_id if exc.problem_id is not None else pid,
                          "condition": exc.condition, "reason": exc.reason,
                          "detail": str(exc)})
            log("drop %s: %s (%s)" % (pid, exc.reason, exc.condition))
            continue

        for cond in sorted(arrays):
            store.add(pid, cond, names, arrays[cond])
            for anchor in names:
                row_meta.append({
                    "problem_id": pid,
                    "condition": cond,
                    "anchor": anchor,
                    "linguistic_label": linguistic_label(rec, cond, anchor),
                })
        del arrays, metas
        kept += 1
        if kept % 25 == 0:
            log("extracted %d problems (%d states) in %.1fs"
                % (kept, store.n_states, time.time() - t0))

    if len(row_meta) != store.n_states:
        raise SystemExit("row_meta (%d) and store rows (%d) diverged"
                         % (len(row_meta), store.n_states))
    for meta, row in zip(row_meta, store.rows):
        if (meta["problem_id"], meta["condition"], meta["anchor"]) != row.as_tuple():
            raise SystemExit(
                "row_meta order does not match the store's row order at %r vs %r"
                % (meta, row.as_tuple()))
    return store, row_meta, drops


# ======================================================================================
# invariant assertions  (configs/assignments_schema.md)
# ======================================================================================


class InvariantError(RuntimeError):
    pass


def assert_schema_invariants(records):
    """Assert, on write, every invariant configs/assignments_schema.md declares.

    These are asserted here as well as in metrics.Assignments on the read side
    because the read side runs on a laptop hours later and the pod is where the
    activations still exist to re-run.
    """
    if not records:
        raise InvariantError("no layer records to write")

    keys = None
    first_layer = None
    seen_layers = set()
    for rec in records:
        layer = rec["layer"]
        if layer in seen_layers:
            raise InvariantError("duplicate record for layer %d" % layer)
        seen_layers.add(layer)
        k = rec["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 2:
            raise InvariantError("layer %s: k must be an int >= 2, got %r" % (layer, k))
        if rec.get("affinity") not in C.VALID_AFFINITIES:
            raise InvariantError(
                "layer %s: affinity %r is not one of %r (Amendment 3; 'raw' never)"
                % (layer, rec.get("affinity"), C.VALID_AFFINITIES))

        rows = rec["rows"]
        row_keys = []
        used_by_condition = {}
        for j, row in enumerate(rows):
            for fld in ("problem_id", "condition", "anchor", "cluster",
                        "kmeans_cluster", "linguistic_label"):
                if fld not in row:
                    raise InvariantError("layer %s row %d: missing field %r"
                                         % (layer, j, fld))
            for fld in ("cluster", "kmeans_cluster"):
                v = row[fld]
                if not isinstance(v, int) or isinstance(v, bool):
                    raise InvariantError("layer %s row %d: %s must be an int, got %r"
                                         % (layer, j, fld, v))
                if not (0 <= v < k):
                    # A condition clustered at a different k shows up here first.
                    raise InvariantError(
                        "layer %s row %d: %s=%d outside [0, k=%d). k is selected ONCE on "
                        "nl_base and reused across all four conditions (§A.6, prereg "
                        "decision 1); a per-condition k invalidates every cross-condition "
                        "lambda2 comparison." % (layer, j, fld, v, k))
            if "k" in row and row["k"] != k:
                raise InvariantError("layer %s row %d: row-level k=%r != layer k=%r"
                                     % (layer, j, row["k"], k))
            row_keys.append((row["problem_id"], row["condition"], row["anchor"]))
            used_by_condition.setdefault(row["condition"], set()).add(row["cluster"])

        # identical k across conditions at a layer
        for cond, used in sorted(used_by_condition.items()):
            if len(used) > k:
                raise InvariantError(
                    "layer %s: condition %r uses %d distinct clusters > k=%d"
                    % (layer, cond, len(used), k))

        if len(set(row_keys)) != len(row_keys):
            raise InvariantError("layer %s: duplicate (problem_id, condition, anchor) rows"
                                 % layer)

        # stable row order across layers
        row_keys = tuple(row_keys)
        if keys is None:
            keys, first_layer = row_keys, layer
        elif row_keys != keys:
            if len(row_keys) != len(keys):
                detail = "row count %d != %d" % (len(row_keys), len(keys))
            else:
                bad = next(i for i, (a, b) in enumerate(zip(row_keys, keys)) if a != b)
                detail = "row %d is %r here but %r at layer %s" % (
                    bad, row_keys[bad], keys[bad], first_layer)
            raise InvariantError(
                "layer %s: row order is NOT stable across layers (%s). Layer-pair "
                "transition graphs are only well-defined if row j is the same (problem, "
                "condition, anchor) at every layer." % (layer, detail))

    # same anchor set across all four conditions, for every problem
    anchors_by_pc = {}
    for pid, cond, anchor in keys:
        anchors_by_pc.setdefault((pid, cond), set()).add(anchor)
    by_problem = {}
    for (pid, cond), anchors in anchors_by_pc.items():
        by_problem.setdefault(pid, {})[cond] = frozenset(anchors)
    for pid, per_cond in sorted(by_problem.items()):
        if len(set(per_cond.values())) > 1:
            raise InvariantError(
                "problem %r has different anchor sets across conditions: %r. §A.6 requires "
                "every (problem, condition) to share one anchor set."
                % (pid, {c: sorted(s) for c, s in per_cond.items()}))
        conds = set(per_cond)
        if len(conds) != len(set(c for _, c, _ in keys)):
            raise InvariantError(
                "problem %r is present in conditions %r but the run has %r. Extraction is "
                "all-four-or-none (§A.6)." % (pid, sorted(conds),
                                              sorted(set(c for _, c, _ in keys))))
    return True


# ======================================================================================
# assembly
# ======================================================================================


def assemble_assignments(writer, row_meta, n_layers, out_path, *, log):
    """Rebuild assignments.jsonl from the per-layer checkpoints. Resumable by design."""
    done = writer.completed_layers()
    if not done:
        raise SystemExit("no layer checkpoints in %s; nothing to assemble" % writer.dir)

    records = []
    for layer in done:
        payload = writer.read_layer(layer)
        clusters = payload["clusters"]
        kmeans = payload["kmeans_clusters"]
        keys = payload["row_keys"]
        if not (len(clusters) == len(kmeans) == len(keys) == len(row_meta)):
            raise SystemExit(
                "layer %d checkpoint has %d/%d/%d entries but this run extracted %d rows. "
                "The checkpoint directory belongs to a different corpus or a different "
                "--max-problems; refusing to mix them. Use a fresh --run-id."
                % (layer, len(clusters), len(kmeans), len(keys), len(row_meta)))
        rows = []
        for j, meta in enumerate(row_meta):
            key = tuple(keys[j])
            if key != (meta["problem_id"], meta["condition"], meta["anchor"]):
                raise SystemExit(
                    "layer %d checkpoint row %d is %r but this run's row %d is %r; the "
                    "checkpoints were written for a different row set. Use a fresh "
                    "--run-id." % (layer, j, key, j,
                                   (meta["problem_id"], meta["condition"], meta["anchor"])))
            rows.append({
                "problem_id": meta["problem_id"],
                "condition": meta["condition"],
                "anchor": meta["anchor"],
                "cluster": int(clusters[j]),
                "kmeans_cluster": int(kmeans[j]),
                "linguistic_label": meta["linguistic_label"],
            })
        rec = {k: v for k, v in payload.items()
               if k not in ("clusters", "kmeans_clusters", "row_keys")}
        rec["layer"] = int(layer)
        rec["rows"] = rows
        records.append(rec)

    records.sort(key=lambda r: r["layer"])
    assert_schema_invariants(records)

    if all(m["linguistic_label"] is None for m in row_meta):
        log("WARNING: every `linguistic_label` is null — the corpus carries no labels yet. "
            "H1's AMI comparison (Amendment 1, DESCRIPTIVE) is not computable from this "
            "file: scripts/04_analyze.py gates on the FIELD being present, so it will "
            "report AMI 0.0 vs 0.0 rather than 'unavailable'. Re-run assembly with "
            "--assemble-only once the labels exist; nothing else has to be recomputed.")

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    log("wrote %d layer records (%d rows each) -> %s"
        % (len(records), len(row_meta), out_path))
    missing = [i for i in range(n_layers) if i not in set(done)]
    if missing:
        log("WARNING: layers %r have no checkpoint and are absent from assignments.jsonl"
            % missing)
    return records


# ======================================================================================
# main
# ======================================================================================


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="results/<run_id>/")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--results-root", default=DEFAULT_RESULTS)
    ap.add_argument("--config", default=None, help="configs/model.yaml override")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--base-condition", default=C.DEFAULT_BASE_CONDITION,
                    help="k is selected on THIS condition only (§A.6)")
    ap.add_argument("--affinity", default=C.COSINE, choices=list(C.VALID_AFFINITIES),
                    help="Amendment 3: cosine is primary, per_layer_standardized is the "
                         "appendix robustness run. 'raw' does not exist.")
    ap.add_argument("--kernel", default=C.KERNEL_GAUSSIAN, choices=list(C.VALID_KERNELS))
    ap.add_argument("--k-min", type=int, default=C.DEFAULT_K_MIN)
    ap.add_argument("--k-max", type=int, default=C.DEFAULT_K_MAX)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--n-landmarks", type=int, default=512,
                    help="Nystrom landmarks, sampled once from the POOLED rows and shared "
                         "by every layer and every condition (§A.11 item 6)")
    ap.add_argument("--exact-max-rows", type=int, default=C.DEFAULT_EXACT_MAX_ROWS,
                    help="at or below this many pooled rows the exact spectrum is used "
                         "and Nystrom is skipped")
    ap.add_argument("--knn-sigma", type=int, default=7)
    ap.add_argument("--n-init", type=int, default=10, help="k-means restarts")
    ap.add_argument("--ari-stride", type=int, default=8,
                    help="ARI stability across seeds on every Nth layer (0 = never)")
    ap.add_argument("--ari-seeds", type=int, default=3)
    ap.add_argument("--g4-stride", type=int, default=8,
                    help="Nystrom-vs-exact spectrum error on every Nth layer (0 = never)")
    ap.add_argument("--g4-holdout", type=int, default=200)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--spill-dir", default=None,
                    help="node-local scratch for activations; refuses results/ and data/")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild assignments.jsonl from existing checkpoints; still needs "
                         "the corpus (for row identity and labels) but loads no model")
    ap.add_argument("--strict-determinism", dest="strict", action="store_true", default=True)
    ap.add_argument("--no-strict-determinism", dest="strict", action="store_false")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if args.base_condition not in conditions:
        raise SystemExit("--base-condition %r is not in --conditions %r"
                         % (args.base_condition, conditions))
    if not C.have_numpy():
        raise SystemExit(
            "numpy is not installed. ling.cluster has a pure-python fallback but it is a "
            "TEST backend capped at %d rows; this pipeline clusters thousands. "
            "pip install -r requirements.txt" % C.PURE_PY_MAX_ROWS)

    out_dir = os.path.join(args.results_root, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    log_lines = []

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log_lines.append(line)

    records = load_corpus(args.corpus, args.max_problems)
    log("corpus: %d problems from %s" % (len(records), args.corpus))

    if args.assemble_only:
        # Rebuild row identity from the corpus WITHOUT a model. Anchor order is
        # hooks.anchor_order, the same order extraction uses.
        row_meta = []
        for rec in records:
            names = anchor_order((rec["realizations"][conditions[0]])["anchors"])
            for cond in sorted(conditions):
                for anchor in names:
                    row_meta.append({"problem_id": rec.get("problem_id"), "condition": cond,
                                     "anchor": anchor,
                                     "linguistic_label": linguistic_label(rec, cond, anchor)})
        writer = LayerCheckpointWriter(args.results_root, args.run_id,
                                       d_model=1, prefix="layer")
        n_layers = max(writer.completed_layers()) + 1 if writer.completed_layers() else 0
        assemble_assignments(writer, row_meta, n_layers,
                             os.path.join(out_dir, "assignments.jsonl"), log=log)
        return 0

    det = enforce_determinism(args.seed, strict=args.strict)
    log("determinism: seed=%d strict=%s" % (args.seed, args.strict))

    from ling.model import load_model  # noqa: E402  (torch import deferred)
    loaded = load_model(args.config)
    log("model: %s, %d layers, d_model=%d, device=%s"
        % (loaded.model_id, loaded.n_layers, loaded.d_model, loaded.device))

    store = None
    try:
        store, row_meta, drops = extract_all(
            records, loaded, conditions=conditions, dtype=args.dtype,
            spill_dir=args.spill_dir, log=log)
        kept_problems = sorted({m["problem_id"] for m in row_meta})
        drop_rate = len(drops) / float(len(records)) if records else 0.0
        log("G3 drop rate: %d/%d = %.3f (%d states kept)"
            % (len(drops), len(records), drop_rate, store.n_states))
        with open(os.path.join(out_dir, "extraction_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "gate": "G3",
                "run_id": args.run_id,
                "corpus": args.corpus,
                "conditions": list(conditions),
                "n_problems_in_corpus": len(records),
                "n_problems_kept": len(kept_problems),
                "n_problems_dropped": len(drops),
                "drop_rate": drop_rate,
                "drops": drops,
                "drops_by_reason": _tally(drops),
                "n_token_states": store.n_states,
                "n_layers": loaded.n_layers,
                "d_model": loaded.d_model,
                "model_id": loaded.model_id,
                "determinism": det,
                "note": ("all-four-or-none per problem (§A.6): a typed AnchorDropError in "
                         "any condition drops the problem in every condition"),
            }, fh, indent=2, sort_keys=True, default=str)

        if store.n_states == 0:
            raise SystemExit("every problem was dropped; nothing to cluster")

        conds = [r.condition for r in store.rows]
        row_keys = [list(r.as_tuple()) for r in store.rows]

        # ---- Nystrom landmarks: ONE sample, from the POOLED rows, shared by every
        # layer and every condition (§A.11 item 6, prereg decision 7). Sampling per
        # layer would add layer-to-layer noise to a statistic that is differenced
        # across adjacent layers; sampling per condition would build a
        # between-condition difference into the embedding before H3 measures one.
        use_nystrom = store.n_states > args.exact_max_rows
        landmarks = None
        if use_nystrom:
            landmarks = C.sample_landmarks(store.n_states, args.n_landmarks, args.seed,
                                           conditions=conds)
            log("Nystrom: %d landmarks sampled from the %d POOLED rows (%d conditions)"
                % (len(landmarks), store.n_states, len(set(conds))))
        else:
            log("exact spectrum: %d pooled rows <= --exact-max-rows=%d"
                % (store.n_states, args.exact_max_rows))

        g4_holdout = None
        if use_nystrom and args.g4_stride:
            import random as _random
            rng = _random.Random(args.seed ^ 0x64)
            pool = [i for i in range(store.n_states) if i not in set(landmarks)]
            g4_holdout = sorted(rng.sample(pool, min(args.g4_holdout, len(pool))))

        writer = LayerCheckpointWriter(args.results_root, args.run_id,
                                       d_model=loaded.d_model, prefix="layer")

        def cluster_one(batch):
            t0 = time.time()
            lc = C.cluster_layer(
                batch.states, conds, batch.layer,
                base_condition=args.base_condition, affinity=args.affinity,
                kernel=args.kernel, k_min=args.k_min, k_max=args.k_max,
                seed=args.seed, landmark_idx=landmarks,
                n_landmarks=args.n_landmarks, exact_max_rows=args.exact_max_rows,
                knn_sigma=args.knn_sigma, n_init=args.n_init, with_kmeans=True)
            payload = lc.record_fields()
            payload.pop("layer", None)
            payload["clusters"] = lc.result.labels
            payload["kmeans_clusters"] = lc.result.kmeans_labels
            payload["row_keys"] = row_keys
            payload["seconds"] = round(time.time() - t0, 2)

            if args.ari_stride and (batch.layer % args.ari_stride == 0):
                payload["ari_stability"] = C.ari_stability(
                    batch.states, lc.k,
                    seeds=[args.seed + i for i in range(max(2, args.ari_seeds))],
                    affinity=args.affinity, kernel=args.kernel, conditions=conds,
                    n_landmarks=args.n_landmarks, exact_max_rows=args.exact_max_rows,
                    knn_sigma=args.knn_sigma, n_init=args.n_init)
            if g4_holdout and (batch.layer % args.g4_stride == 0):
                X = C.prepare_states(batch.states, args.affinity)
                payload["nystrom_error"] = C.nystrom_error_report(
                    X, landmarks, g4_holdout, lc.result.sigma, kernel=args.kernel,
                    n_eigs=min(10, lc.k + 2))
            return payload

        computed = stream_layers_with_checkpoints(
            store, writer, cluster_one, resume=not args.no_resume, log=log,
            release=False)
        log("clustered %d layers this run (%d checkpoints present)"
            % (len(computed), len(writer.completed_layers())))

        out_path = os.path.join(out_dir, "assignments.jsonl")
        assemble_assignments(writer, row_meta, loaded.n_layers, out_path, log=log)
    finally:
        # §A.1 rule 1: extract -> cluster -> write assignments -> DISCARD. This runs
        # on the exception path too; a crashed run must not leave activations behind
        # in a spill directory.
        if store is not None:
            store.close()
            log("activations discarded (§A.1)")
        with open(os.path.join(out_dir, "extract_and_cluster.log"), "a",
                  encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n")
    return 0


def _tally(drops):
    out = {}
    for d in drops:
        out[d.get("reason", "unknown")] = out.get(d.get("reason", "unknown"), 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
