#!/usr/bin/env python3
"""Gate G2 -- the behavioural accuracy check (DESIGN.md §A.12 hour 1-2, §B.8;
prereg/PRIMARY_ENDPOINTS.md gate G2).

What this decides
-----------------
G2 decides WHICH PAPER gets written, and it decides it by measurement, before
any geometry is looked at:

  * `formal` accuracy ABOVE `nl_base`  -> H3 keeps its accuracy half
                                          ("formal notation restructures routing
                                          geometry AND improves accuracy").
  * otherwise                          -> the accuracy half of H3 is CUT and the
                                          paper is the dissociation result
                                          ("formal rewriting restructures routing
                                          geometry WITHOUT improving accuracy").

The gate as sealed in prereg/PRIMARY_ENDPOINTS.md is worded on the point
estimate ("if formal accuracy is not above NL-base"). That is what the verdict
line uses. A paired exact McNemar test on the per-problem correct/incorrect
pairs is reported alongside it so the reader can see whether a nominal
difference is more than noise; the gate itself is NOT re-worded to depend on it.

What this is NOT
----------------
This is a behavioural check, not the geometry measurement. Nothing here feeds a
cluster, a lambda2, or a fork layer, so NONE of gate G1's determinism regime
applies:

  * default fast attention kernels (SDPA), not eager;
  * batched, left-padded generation, not batch size 1;
  * no `enforce_determinism`, no deterministic-algorithms flag.

Those kernels are not bitwise reproducible across shapes. That is deliberate
and is recorded in the output (`regime`) so it cannot be confused with G1's.
Batch-1 eager generation of ~570 cells would take over an hour; batched SDPA
takes minutes.

Chat template
-------------
Extraction (scripts/03) never applies a chat template -- the aligned-position
contract of §A.6 forbids the role tokens. GENERATION needs one, so this script
applies the tokenizer's chat template with `add_generation_prompt=True` and
`enable_thinking=False` (via `LoadedModel.chat_template_kwargs`). The checkpoint
documents that with thinking disabled it "will still generate the tags but with
an empty thought block", so any thought block is stripped before the answer is
parsed, and the number of NON-empty thought blocks is reported (it should be 0).
That difference between G2 and extraction is stated in the output.

Scoring
-------
Every answer-preserving condition is scored against the record's
`ground_truth`. `nl_break` is scored against ITS OWN perturbed answer
(`realizations.nl_break.answer`), which the corpus records: scoring it against
the unperturbed truth would give ~0% by construction and measure nothing.

Usage
-----
    python scripts/02b_behavioral_g2.py                     # full run
    python scripts/02b_behavioral_g2.py --max-problems 4    # smoke run
    python scripts/02b_behavioral_g2.py --batch-size 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

# NOTE: importing ling.model transitively imports ling.determinism, whose only
# import-time side effect is exporting CUBLAS_WORKSPACE_CONFIG. That env var
# changes nothing unless torch.use_deterministic_algorithms(True) is also set,
# which this script never does (load_model is called with enforce=False).
from ling import model as M  # noqa: E402

DEFAULT_CORPUS = os.path.join(_REPO, "data", "corpus", "problems.jsonl")
DEFAULT_OUT = os.path.join(_REPO, "results", "g2_behavioral.json")

CONDITIONS = ("nl_base", "nl_para", "formal", "nl_break",
              "nl_minimal_orth", "nl_minimal_lex")
UNAVAILABLE = "unavailable"

#: Uniform instruction appended to EVERY condition's prompt, so that the
#: answer-format request is identical across realizations and the parser has a
#: stable anchor to look for.
DEFAULT_INSTRUCTION = (
    "Solve the problem above. Show brief working, then finish with a final line "
    "of the exact form: The answer is <number>."
)

#: Numeric tolerance for non-integer ground truths (relative, floor 1e-6).
FLOAT_RTOL = 1e-6


# ======================================================================================
# corpus
# ======================================================================================


def load_corpus(path: str, max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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


def is_declared_hole(real: Any) -> bool:
    """Amendment 6: an unbuilt style-control cell is `status: "unavailable"`
    with empty text. Skip the CELL; never drop the problem."""
    if not isinstance(real, dict):
        return True
    status = real.get("status")
    if isinstance(status, str) and status.strip().lower() == UNAVAILABLE:
        return True
    return not (real.get("text") or "").strip()


def target_answer(rec: Dict[str, Any], cond: str, real: Dict[str, Any],
                  warnings: List[str]) -> float:
    """The number this cell is scored against.

    `nl_break` -> its own perturbed answer. Everything else -> `ground_truth`,
    cross-checked against the realization's own `answer` field.
    """
    gt = rec["ground_truth"]
    if cond == "nl_break":
        if "answer" not in real:
            raise SystemExit(
                "%s/nl_break has no `answer` field; the corpus must record the "
                "perturbed answer or nl_break cannot be scored" % rec["problem_id"])
        ans = real["answer"]
        if _num_equal(float(ans), float(gt)):
            warnings.append("%s: nl_break answer equals ground_truth (%r) -- the "
                            "perturbation did not change the answer" % (rec["problem_id"], ans))
        return float(ans)
    ans = real.get("answer", gt)
    if not _num_equal(float(ans), float(gt)):
        warnings.append("%s/%s: realization answer %r != ground_truth %r; scoring "
                        "against ground_truth per G2" % (rec["problem_id"], cond, ans, gt))
    return float(gt)


def build_cells(records: Sequence[Dict[str, Any]], conditions: Sequence[str],
                warnings: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cells: List[Dict[str, Any]] = []
    holes: Dict[str, int] = {c: 0 for c in conditions}
    for rec in records:
        reals = rec.get("realizations") or {}
        for cond in conditions:
            if cond not in reals:
                raise SystemExit(
                    "%s has no realization %r at all -- a hole is written explicitly "
                    "(status unavailable), so a missing key means this is not the "
                    "corpus this script expects" % (rec["problem_id"], cond))
            real = reals[cond]
            if is_declared_hole(real):
                holes[cond] += 1
                continue
            cells.append({
                "problem_id": rec["problem_id"],
                "condition": cond,
                "prompt_text": real["text"],
                "target": target_answer(rec, cond, real, warnings),
                "ground_truth": float(rec["ground_truth"]),
            })
    return cells, holes


# ======================================================================================
# answer parsing
# ======================================================================================

# A number, tolerant of a leading currency symbol, thousands separators, a
# unicode minus, a decimal part, and a trailing % / period (both discarded).
_NUM = r"[-\u2212]?\s*[$\u00a3\u20ac]?\s*\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)

#: "The answer is X" / "Answer: X" / "\boxed{X}" / "**X**" at the very end.
#: Searched LAST-match-wins so a restated final answer beats an intermediate one.
_ANCHORED_RES = [
    re.compile(r"answer\s*(?:is|=|:)\s*(?:\*\*)?\s*(" + _NUM + r")", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*(" + _NUM + r")\s*\}"),
    re.compile(r"####\s*(" + _NUM + r")"),
]

# Thought blocks. The checkpoint emits <|think|>...<|/think|>-style tags even
# with thinking disabled (empty block). Be generous about the tag spelling.
_THINK_OPEN = r"(?:<\|think\|>|<think>|<\|thought\|>|<thought>)"
_THINK_CLOSE = r"(?:<\|/think\|>|</think>|<\|end_think\|>|<\|/thought\|>|</thought>)"
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r"(.*?)" + _THINK_CLOSE, re.DOTALL)
_THINK_TAG_RE = re.compile(_THINK_OPEN + "|" + _THINK_CLOSE)


def strip_thought(text: str) -> Tuple[str, bool, bool]:
    """Remove thought blocks. Returns (clean, saw_tags, nonempty_thought)."""
    saw = bool(_THINK_TAG_RE.search(text))
    nonempty = any(m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(text))
    clean = _THINK_BLOCK_RE.sub("", text)
    clean = _THINK_TAG_RE.sub("", clean)  # any unpaired stray tag
    return clean, saw, nonempty


def _to_number(s: str) -> Optional[float]:
    s = s.strip().replace("\u2212", "-")
    s = re.sub(r"[$\u00a3\u20ac,\s]", "", s)
    s = s.rstrip(".%")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_answer(completion: str) -> Tuple[Optional[float], str]:
    """Final numeric answer from a completion. Returns (value, how)."""
    for how, rx in zip(("answer_is", "boxed", "hashes"), _ANCHORED_RES):
        matches = rx.findall(completion)
        for cand in reversed(matches):
            val = _to_number(cand)
            if val is not None:
                return val, how
    nums = _NUM_RE.findall(completion)
    for cand in reversed(nums):
        val = _to_number(cand)
        if val is not None:
            return val, "last_number"
    return None, "none"


def _num_equal(pred: float, truth: float) -> bool:
    if float(truth).is_integer():
        return pred == truth
    return abs(pred - truth) <= max(FLOAT_RTOL, FLOAT_RTOL * abs(truth))


def is_correct(pred: Optional[float], truth: float) -> bool:
    return pred is not None and _num_equal(pred, truth)


# ======================================================================================
# model + generation
# ======================================================================================


def load_for_generation(config_path: Optional[str], attn: str,
                        device: Optional[str]) -> M.LoadedModel:
    """Load through ling.model.load_model with G1's controls switched OFF.

    `load_model` reads `attn_implementation` from configs/model.yaml and
    ASSERTS it is what ends up in force -- there is no argument to override
    it, and model.py is not to be modified. So we write a copy of the config
    with `attn_implementation: <attn>` to a temp file and load from that. The
    MoE / quantization / fast-tokenizer checks and the thinking-off plumbing
    all still run. `enforce=False` skips enforce_determinism entirely.
    """
    cfg = M.load_config(config_path)
    override = {k: v for k, v in cfg.items() if not k.startswith("_")}
    override["attn_implementation"] = attn
    lines = []
    for k, v in override.items():
        if v is None:
            lines.append("%s: null" % k)
        elif isinstance(v, bool):
            lines.append("%s: %s" % (k, "true" if v else "false"))
        elif isinstance(v, (int, float)):
            lines.append("%s: %s" % (k, v))
        else:
            lines.append('%s: "%s"' % (k, str(v).replace('"', '\\"')))
    fd, tmp = tempfile.mkstemp(prefix="g2_model_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        lm = M.load_model(tmp, device=device, enforce=False)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    lm.load_info["config_source"] = cfg["_config_path"]
    return lm


def render_prompt(tokenizer: Any, chat_kwargs: Dict[str, Any], text: str,
                  instruction: str) -> Tuple[str, bool]:
    """Chat-templated prompt string. Returns (prompt, template_applied)."""
    user = text.rstrip() + "\n\n" + instruction if instruction else text
    messages = [{"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
            ), True
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True), True
    return user + "\n", False


def generate_batched(lm: M.LoadedModel, prompts: List[str], batch_size: int,
                     max_new_tokens: int, log) -> List[str]:
    """Greedy, batched, left-padded generation. Returns completions (new tokens
    only) in the ORIGINAL prompt order."""
    import torch

    tok = lm.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation: pad on the left so every row's
    #                            last prompt token sits at the same position

    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    completions: List[Optional[str]] = [None] * len(prompts)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    eos = getattr(lm.model.generation_config, "eos_token_id", None)
    if eos is not None:
        gen_kwargs["eos_token_id"] = eos

    t0 = time.time()
    n_batches = (len(order) + batch_size - 1) // batch_size
    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        batch_prompts = [prompts[i] for i in idx]
        # The chat template already carries BOS where the model wants it.
        enc = tok(batch_prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(lm.device)
        with torch.inference_mode():
            out = lm.model.generate(**enc, **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=False)
        for i, t in zip(idx, texts):
            completions[i] = t
        log("  batch %d/%d  (%d rows, prompt width %d)  %.0fs" % (
            b + 1, n_batches, len(idx), enc["input_ids"].shape[1], time.time() - t0))
    assert all(c is not None for c in completions)
    return [c for c in completions if c is not None]


def _strip_special(text: str, tokenizer: Any) -> str:
    """Drop pad/eos/end-of-turn strings so they cannot be read as answer text."""
    for tokstr in {tokenizer.pad_token, tokenizer.eos_token, "<end_of_turn>", "<eos>"}:
        if tokstr:
            text = text.replace(tokstr, "")
    return text


# ======================================================================================
# statistics
# ======================================================================================


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> Dict[str, Any]:
    """Exact (binomial) McNemar on discordant pairs.

    b = formal correct & nl_base wrong; c = nl_base correct & formal wrong.
    Two-sided p and the one-sided p for the pre-committed direction
    (formal > nl_base) are both reported.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_two_sided": 1.0,
                "p_one_sided_formal_gt_base": 1.0,
                "note": "no discordant pairs; test undefined, p set to 1"}

    def cdf(k: int) -> float:  # P[X <= k], X ~ Bin(n, 1/2); exact, no scipy
        return sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n

    p_two = min(1.0, 2.0 * cdf(min(b, c)))
    p_one = 1.0 - cdf(b - 1) if b > 0 else 1.0  # P[X >= b]
    return {"b": b, "c": c, "n_discordant": n,
            "p_two_sided": float(p_two),
            "p_one_sided_formal_gt_base": p_one}


# ======================================================================================
# main
# ======================================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Gate G2: behavioural accuracy on every corpus realization "
                    "(DESIGN.md A.12 / B.8; prereg gate G2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--config", default=None, help="model config (configs/model.yaml)")
    ap.add_argument("--device", default=None, help="override config device")
    ap.add_argument("--max-problems", type=int, default=None,
                    help="smoke run: only the first N corpus records")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--attn", default="sdpa",
                    help="attention implementation for GENERATION (default sdpa: "
                         "fast, non-deterministic, deliberately NOT G1's eager)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                    help="uniform instruction appended to every prompt ('' for none)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        sys.stderr.write("[g2] %s\n" % msg)
        sys.stderr.flush()

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    for needed in ("formal", "nl_base"):
        if needed not in conditions:
            raise SystemExit("--conditions must include %r; G2 is the formal vs nl_base "
                             "comparison" % needed)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    records = load_corpus(args.corpus, args.max_problems)
    warnings: List[str] = []
    cells, holes = build_cells(records, conditions, warnings)
    log("%d problems, %d cells; declared holes per condition: %s" % (
        len(records), len(cells), {c: n for c, n in holes.items() if n}))
    for w in warnings:
        log("WARNING: " + w)

    lm = load_for_generation(args.config, args.attn, args.device)
    import torch
    torch.manual_seed(int(lm.config["seed"]))

    prompts: List[str] = []
    template_applied = True
    for cell in cells:
        p, applied = render_prompt(lm.tokenizer, lm.chat_template_kwargs,
                                   cell["prompt_text"], args.instruction)
        template_applied = template_applied and applied
        prompts.append(p)
    if not template_applied:
        log("WARNING: tokenizer has no chat template; generating from raw prompts")
    if cells:
        log("example prompt (%s/%s):\n%s" % (cells[0]["problem_id"], cells[0]["condition"],
                                              prompts[0]))

    t0 = time.time()
    completions = generate_batched(lm, prompts, args.batch_size, args.max_new_tokens, log)
    gen_seconds = time.time() - t0

    n_saw_tags = n_nonempty_thought = n_unparsed = 0
    for cell, raw in zip(cells, completions):
        clean, saw, nonempty = strip_thought(raw)
        clean = _strip_special(clean, lm.tokenizer)
        pred, how = parse_answer(clean)
        cell["completion"] = raw
        cell["thought_tags_seen"] = saw
        cell["thought_nonempty"] = nonempty
        cell["pred"] = pred
        cell["parse_method"] = how
        cell["correct"] = is_correct(pred, cell["target"])
        if cell["condition"] == "nl_break":
            cell["matches_unperturbed_truth"] = is_correct(pred, cell["ground_truth"])
        n_saw_tags += saw
        n_nonempty_thought += nonempty
        n_unparsed += pred is None

    # per-condition accuracy
    per_cond: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        rows = [c for c in cells if c["condition"] == cond]
        n = len(rows)
        k = sum(c["correct"] for c in rows)
        lo, hi = wilson_ci(k, n)
        entry = {"n": n, "n_correct": k, "accuracy": (k / n) if n else float("nan"),
                 "wilson95": [lo, hi], "n_declared_holes": holes[cond],
                 "n_unparsed": sum(c["pred"] is None for c in rows),
                 "scored_against": "own perturbed answer" if cond == "nl_break"
                 else "ground_truth"}
        if cond == "nl_break":
            entry["n_matches_unperturbed_truth"] = sum(
                c["matches_unperturbed_truth"] for c in rows)
        per_cond[cond] = entry

    # paired formal vs nl_base
    by_pid: Dict[str, Dict[str, bool]] = {}
    for c in cells:
        if c["condition"] in ("formal", "nl_base"):
            by_pid.setdefault(c["problem_id"], {})[c["condition"]] = c["correct"]
    pairs = {pid: d for pid, d in by_pid.items() if "formal" in d and "nl_base" in d}
    b = sum(1 for d in pairs.values() if d["formal"] and not d["nl_base"])
    c_ = sum(1 for d in pairs.values() if d["nl_base"] and not d["formal"])
    both = sum(1 for d in pairs.values() if d["formal"] and d["nl_base"])
    neither = sum(1 for d in pairs.values() if not d["formal"] and not d["nl_base"])
    mcn = mcnemar_exact(b, c_)
    acc_f = per_cond["formal"]["accuracy"]
    acc_b = per_cond["nl_base"]["accuracy"]
    formal_above = bool(acc_f > acc_b)
    significant = bool(formal_above and mcn["p_one_sided_formal_gt_base"] < 0.05)

    if formal_above:
        paper = "H3-with-accuracy"
        verdict = ("formal accuracy %.1f%% > nl_base %.1f%% (paired diff %+.1f pp, exact "
                   "McNemar one-sided p=%.3g): the accuracy half of H3 STANDS. Paper: "
                   "geometry + accuracy." % (100 * acc_f, 100 * acc_b, 100 * (acc_f - acc_b),
                                             mcn["p_one_sided_formal_gt_base"]))
        if not significant:
            verdict += (" NOTE: the difference is not significant at 0.05; the sealed gate "
                        "is on the point estimate, so the half stands, but the accuracy "
                        "claim must be reported with this p-value and the CI, not as a "
                        "detected effect.")
    else:
        paper = "dissociation"
        verdict = ("formal accuracy %.1f%% is NOT above nl_base %.1f%% (paired diff %+.1f pp, "
                   "exact McNemar two-sided p=%.3g): the accuracy half of H3 is CUT. Paper: "
                   "the dissociation result -- formal rewriting restructures routing geometry "
                   "without improving accuracy." % (100 * acc_f, 100 * acc_b,
                                                    100 * (acc_f - acc_b), mcn["p_two_sided"]))

    regime = {
        "attn_implementation_effective": lm.load_info.get("attn_implementation_effective"),
        "attn_implementation_subconfigs": lm.load_info.get("attn_implementation_subconfigs"),
        "determinism_enforced": False,
        "batched_left_padded_generation": True,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy (do_sample=False)",
        "chat_template_applied": template_applied,
        "chat_template_kwargs": lm.chat_template_kwargs,
        "thinking": "OFF (enable_thinking=False); thought blocks stripped before parsing",
        "instruction_appended": args.instruction,
        "statement": (
            "G2 is a behavioural check, not the geometry measurement. It used the "
            "default fast (SDPA) attention kernels, batched left-padded generation, and "
            "NO determinism controls. Those kernels are not bitwise reproducible across "
            "shapes; this is deliberate and is a separate regime from gate G1's eager, "
            "batch-1, deterministic extraction. G2 also applies the tokenizer's chat "
            "template (generation needs one); extraction applies none (DESIGN.md A.6 "
            "aligned-position contract)."),
    }

    result = {
        "gate": "G2",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "corpus": os.path.relpath(args.corpus, _REPO),
        "n_problems": len(records),
        "n_cells": len(cells),
        "max_problems": args.max_problems,
        "model_id": lm.model_id,
        "model_load_info": {k: v for k, v in lm.load_info.items() if k != "determinism"},
        "regime": regime,
        "generation_seconds": gen_seconds,
        "thought_tags": {"n_cells_with_tags": n_saw_tags,
                         "n_cells_with_nonempty_thought": n_nonempty_thought},
        "n_unparsed": n_unparsed,
        "per_condition": per_cond,
        "formal_vs_nl_base": {
            "n_pairs": len(pairs),
            "both_correct": both, "neither_correct": neither,
            "formal_only": b, "nl_base_only": c_,
            "accuracy_formal": acc_f, "accuracy_nl_base": acc_b,
            "paired_difference": acc_f - acc_b,
            "mcnemar_exact": mcn,
            "formal_above_nl_base": formal_above,
            "formal_above_nl_base_significant_0.05": significant,
        },
        "verdict": {"paper": paper, "h3_accuracy_half": "stands" if formal_above else "cut",
                    "text": verdict},
        "corpus_warnings": warnings,
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)

    # --- printed table -------------------------------------------------------
    print()
    print("Gate G2 -- behavioural accuracy  (%s, %d problems, %d cells, %.0fs generation)" % (
        lm.model_id, len(records), len(cells), gen_seconds))
    print("Regime: %s attention, batched (bs=%d), greedy, thinking OFF, chat template %s -- "
          "NON-deterministic fast kernels, deliberately separate from G1." % (
              regime["attn_implementation_effective"], args.batch_size,
              "applied" if template_applied else "NOT applied"))
    print()
    print("%-16s %5s %6s %8s %16s %6s %8s  %s" % (
        "condition", "n", "corr", "acc", "wilson95", "holes", "unparsed", "scored against"))
    for cond in conditions:
        e = per_cond[cond]
        extra = ""
        if cond == "nl_break":
            extra = "  (%d/%d gave the UNperturbed answer)" % (
                e["n_matches_unperturbed_truth"], e["n"])
        print("%-16s %5d %6d %7.1f%% [%5.1f%%,%5.1f%%] %6d %8d  %s%s" % (
            cond, e["n"], e["n_correct"], 100 * e["accuracy"],
            100 * e["wilson95"][0], 100 * e["wilson95"][1],
            e["n_declared_holes"], e["n_unparsed"], e["scored_against"], extra))
    print()
    print("formal vs nl_base, paired on %d problems: both %d, neither %d, formal-only %d, "
          "nl_base-only %d" % (len(pairs), both, neither, b, c_))
    print("  exact McNemar: two-sided p=%.4g, one-sided (formal>base) p=%.4g" % (
        mcn["p_two_sided"], mcn["p_one_sided_formal_gt_base"]))
    if n_nonempty_thought:
        print("  WARNING: %d cells produced a NON-empty thought block with thinking "
              "disabled -- check enable_thinking plumbing" % n_nonempty_thought)
    if n_unparsed:
        print("  note: %d completions had no parseable number (scored incorrect)" % n_unparsed)
    print()
    print("VERDICT [%s]: %s" % (paper, verdict))
    print()
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
