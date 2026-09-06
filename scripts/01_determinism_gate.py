#!/usr/bin/env python3
"""Gate G1 — determinism. DESIGN.md §A.4; prereg/PRIMARY_ENDPOINTS.md gate G1.

    "Run each prompt twice under identical batch conditions. Fork signal must be
     EXACTLY zero, not 'near zero'. If not, it is a plumbing bug and everything
     downstream is measuring numerical noise."  -- DESIGN.md §A.4

What this checks, per probe prompt:

  1. Pass A vs pass B, back to back, identical batch-1 unpadded conditions:
     every layer's residual stream must be BITWISE identical (all n_layers of
     them, full sequence, not just the anchored positions).
  2. The next-token logit vector must be bitwise identical.
  3. The greedy next token must be identical (implied by 2, reported separately
     because it is the quantity the behavioural check G2 consumes).
  4. Sweep-order invariance: the whole prompt list is run a second time in
     REVERSE order and compared by SHA-256 digest of every layer tensor. This
     catches state that leaks between prompts — a KV cache, a cuDNN autotune
     decision, an RNG draw — which a back-to-back A/B comparison cannot see.

Exits 0 only if every difference is exactly zero. Any nonzero difference exits
1 with the first differing index and its magnitude; per §A.12 the run STOPS
there and nothing downstream is reported.

Usage:
    python scripts/01_determinism_gate.py                     # defaults
    python scripts/01_determinism_gate.py --n-prompts 12
    python scripts/01_determinism_gate.py --prompts-file probes.txt
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
from ling.determinism import (  # noqa: E402
    DeterminismError,
    assert_bitwise_identical,
    determinism_controls_status,
    determinism_summary_line,
    max_abs_difference,
    tensor_digest,
)
from ling.hooks import ResidualCapture, encode, forward_once  # noqa: E402
from ling.model import load_model  # noqa: E402


# Probe prompts. Deliberately varied in length, tokenization character and
# digit density: batch-shape-dependent reduction noise shows up as a function of
# sequence length, and the corpus is number-heavy (GSM-Symbolic), so the probes
# are too. The last two mimic the `formal` condition (Python source), whose
# tokenization statistics are the most different from prose (§A.6).
DEFAULT_PROBES = [
    "The cat sat.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
    "bolts in total does it take?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half "
    "of the money she needs. Her parents decided to give her $15 for that purpose, "
    "and her grandparents twice as much as her parents. How much more money does "
    "Betty need to buy the wallet?",
    "James writes a 3-page letter to 2 different friends twice a week. How many "
    "pages does he write a year?",
    "num_clips_april = 48\nnum_clips_may = num_clips_april / 2\ntotal = "
    "num_clips_april + num_clips_may\n# total = ?",
    # NOTE: no trailing newline. A trailing "\n" is a token in its own right
    # with a real, non-zero-width offset span, so it is NOT caught by the
    # trailing-special-token guard: the probe's final position would be the
    # newline for these code-shaped probes and the sentence-final period for the
    # prose ones, which is exactly the condition-correlated FINAL drift §A.6
    # forbids. ling.hooks.encode() also normalises, so this is belt-and-braces —
    # but the literal should not model the wrong thing.
    "def solve():\n    bolts_blue = 2\n    bolts_white = bolts_blue / 2\n    "
    "return bolts_blue + bolts_white",
]


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def capture_pass(loaded, input_ids):
    """One forward pass. Returns (residuals list [n_layers], next-token logits)."""
    with ResidualCapture(loaded, cast_to=None, keep_on_gpu=True) as cap:
        logits = forward_once(loaded, input_ids)
        cap.assert_complete()
        resid = list(cap.residuals)
    # logits[0, -1] is the next-token distribution for the last prompt position.
    return resid, logits[0, -1].detach().clone()


def digests(resid, logits) -> list:
    return [tensor_digest(r) for r in resid] + [tensor_digest(logits)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to configs/model.yaml")
    ap.add_argument("--n-prompts", type=int, default=len(DEFAULT_PROBES),
                    help="number of probe prompts to use")
    ap.add_argument("--prompts-file", default=None,
                    help="newline-separated probe prompts (blank-line separated "
                         "for multi-line probes)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(_REPO, "results", "gate_g1_determinism.json"))
    ap.add_argument("--skip-order-check", action="store_true",
                    help="skip the reverse-order sweep (faster; do not use for the "
                         "recorded gate run)")
    args = ap.parse_args()

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            probes = [p.strip() for p in fh.read().split("\n\n") if p.strip()]
    else:
        probes = DEFAULT_PROBES
    probes = probes[: args.n_prompts]
    if not probes:
        log("no probe prompts")
        return 1

    t0 = time.time()
    log("=" * 78)
    log("GATE G1 - DETERMINISM  (DESIGN.md A.4)")
    log("=" * 78)

    loaded = load_model(args.config, device=args.device, enforce=True,
                        strict_determinism=True)
    det = loaded.load_info["determinism"]

    # attn_implementation is load-bearing for this gate: SDPA/flash reduction
    # order depends on sequence length and head tiling. Check the EFFECTIVE
    # value read back off model.config (and every sub-config), never the value
    # we requested — a fallback is a warning, not an exception, in some
    # transformers releases, and a wrapped checkpoint's text_config keeps its
    # own.
    attn = loaded.load_info.get("attn_implementation_effective")
    attn_subs = loaded.load_info.get("attn_implementation_subconfigs", {})
    bad_attn = ({"config": attn} if attn != "eager" else {})
    bad_attn.update({k: v for k, v in attn_subs.items() if v != "eager"})
    if bad_attn:
        log("FAIL: effective attention implementation is {} (requested {!r}), "
            "not 'eager' everywhere. SDPA and flash kernels are not bitwise "
            "reproducible across shapes; this gate cannot pass honestly with "
            "them enabled.".format(
                bad_attn, loaded.load_info.get("attn_implementation_requested")))
        return 1

    # F4: the summary line is the paper's methods claim. Verify each control it
    # names is actually in force, and make that part of the pass predicate —
    # the gate must not pass while a named control is silently inactive.
    controls = determinism_controls_status(det)
    if not controls["ok"]:
        log("FAIL: determinism controls are not all in force: {}".format(
            controls["detail"]))
        log("This gate's summary line is the paper's methods sentence. It may "
            "not claim a control that this run did not have.")

    log(determinism_summary_line(det, loaded.load_info))
    log("model: {} | {} blocks | d_model={} | layers at model.{}".format(
        loaded.model_id, loaded.n_layers, loaded.d_model, loaded.layers_path))
    if not loaded.load_info["checkpoint_verified"]:
        log("NOTE: checkpoint string is marked UNVERIFIED in configs/model.yaml "
            "(DESIGN.md A.2 requires verifying it on HuggingFace before the run).")
    log("-" * 78)

    per_prompt = []
    forward_digests = []
    failures = []

    for pi, text in enumerate(probes):
        input_ids = encode(loaded, text)
        n_tok = int(input_ids.shape[1])

        resid_a, logits_a = capture_pass(loaded, input_ids)
        resid_b, logits_b = capture_pass(loaded, input_ids)

        rec = {
            "index": pi,
            "n_tokens": n_tok,
            "preview": text.replace("\n", "\\n")[:60],
            "max_abs_residual_diff": 0.0,
            "max_abs_logit_diff": 0.0,
            "greedy_token_match": None,
            "layers_checked": loaded.n_layers,
        }

        try:
            for li in range(loaded.n_layers):
                # magnitude first, THEN the assertion, so the per-probe row
                # still reports the size of the difference when it fails
                rec["max_abs_residual_diff"] = max(
                    rec["max_abs_residual_diff"],
                    max_abs_difference(resid_a[li], resid_b[li]),
                )
                assert_bitwise_identical(
                    resid_a[li], resid_b[li],
                    name="prompt[{}] layer[{}] residual (output of block {})".format(
                        pi, li, li),
                )
            rec["max_abs_logit_diff"] = max_abs_difference(logits_a, logits_b)
            assert_bitwise_identical(
                logits_a, logits_b, name="prompt[{}] next-token logits".format(pi))
        except DeterminismError as exc:
            failures.append(str(exc))
            rec["failed"] = True

        ga = int(logits_a.argmax().item())
        gb = int(logits_b.argmax().item())
        rec["greedy_token_match"] = (ga == gb)
        rec["greedy_token_id"] = ga
        rec["greedy_token"] = loaded.tokenizer.decode([ga])
        if ga != gb:
            failures.append(
                "prompt[{}]: greedy next token differs between passes: {} vs {} "
                "({!r} vs {!r})".format(pi, ga, gb,
                                        loaded.tokenizer.decode([ga]),
                                        loaded.tokenizer.decode([gb])))

        forward_digests.append(digests(resid_a, logits_a))
        per_prompt.append(rec)

        status = "FAIL" if rec.get("failed") or not rec["greedy_token_match"] else "ok"
        log("  probe {:2d} | {:4d} tok | {:>4} | max|dR|={:.1e} max|dlogit|={:.1e} | "
            "greedy={!r}".format(
                pi, n_tok, status, rec["max_abs_residual_diff"],
                rec["max_abs_logit_diff"], rec["greedy_token"]))

        del resid_a, resid_b, logits_a, logits_b

    # ---- sweep-order invariance -------------------------------------------
    order_ok = None
    if not args.skip_order_check:
        log("-" * 78)
        log("reverse-order sweep (catches state leaking between prompts)")
        order_ok = True
        for pi in reversed(range(len(probes))):
            input_ids = encode(loaded, probes[pi])
            resid_c, logits_c = capture_pass(loaded, input_ids)
            dig_c = digests(resid_c, logits_c)
            if dig_c != forward_digests[pi]:
                order_ok = False
                bad = [i for i, (x, y) in enumerate(zip(dig_c, forward_digests[pi]))
                       if x != y]
                failures.append(
                    "prompt[{}]: results depend on the ORDER prompts are run in; "
                    "digest mismatch at layer indices {} (index {} = next-token "
                    "logits). Something is carried between forward passes: a KV "
                    "cache, a cuDNN autotune decision, or an RNG draw.".format(
                        pi, bad, loaded.n_layers))
            del resid_c, logits_c
        log("  order-invariant: {}".format(order_ok))

    elapsed = time.time() - t0
    max_r = max((r["max_abs_residual_diff"] for r in per_prompt), default=0.0)
    max_l = max((r["max_abs_logit_diff"] for r in per_prompt), default=0.0)
    n_greedy = sum(1 for r in per_prompt if r["greedy_token_match"])

    if not controls["ok"]:
        failures.append(
            "determinism controls named in the methods sentence are not all in "
            "force: {}. Every clause of that sentence is a claim about the run "
            "that produced these numbers; the gate cannot pass while one of "
            "them is inactive (DESIGN.md A.4).".format(controls["detail"]))

    passed = (not failures) and max_r == 0.0 and max_l == 0.0 and \
             n_greedy == len(per_prompt) and (order_ok is not False) and \
             controls["ok"] and attn == "eager" and not bad_attn

    summary = (
        "G1 determinism gate: {n} probe prompts x 2 passes, {L} layers each; "
        "max |delta residual| = {mr:.1f} and max |delta next-token logit| = {ml:.1f} "
        "(bitwise identical, not merely within tolerance); greedy next token "
        "identical in {g}/{n} probes{order}. {det}"
    ).format(
        n=len(per_prompt), L=loaded.n_layers, mr=max_r, ml=max_l, g=n_greedy,
        order="" if order_ok is None else
              ("; results invariant to prompt ordering" if order_ok
               else "; ORDER-DEPENDENT"),
        det=determinism_summary_line(det, loaded.load_info),
    )

    record = {
        "gate": "G1",
        "passed": bool(passed),
        "summary": summary,
        "elapsed_s": round(elapsed, 1),
        "model": loaded.load_info,
        "determinism_controls": controls,
        "attn_implementation_effective": attn,
        "attn_implementation_subconfigs": attn_subs,
        "n_probes": len(per_prompt),
        "max_abs_residual_diff": max_r,
        "max_abs_logit_diff": max_l,
        "greedy_match": "{}/{}".format(n_greedy, len(per_prompt)),
        "order_invariant": order_ok,
        "per_prompt": per_prompt,
        "failures": failures,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    log("=" * 78)
    if not passed:
        log("GATE G1 FAILED - the difference is not exactly zero.")
        log("")
        for f in failures[:5]:
            log(f)
            log("")
        if len(failures) > 5:
            log("... and {} more failures".format(len(failures) - 5))
        log("DESIGN.md A.4: this is a plumbing bug, not a result. Everything "
            "downstream would be measuring numerical noise. Per A.12 the run "
            "STOPS here.")
        log("record written to {}".format(args.out))
        return 1

    log("GATE G1 PASSED")
    log("")
    log("paper-ready one-liner:")
    log(summary)
    log("")
    log("record written to {} ({:.1f}s)".format(args.out, elapsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
