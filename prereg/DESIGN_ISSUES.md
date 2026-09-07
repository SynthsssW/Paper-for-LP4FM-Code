# Gaps found in DESIGN.md during implementation

Raised by the builder agent while implementing §A.4/§A.6. Recorded rather than silently
patched into DESIGN.md — the design doc is the reference spec, and these need a human
decision. Items 1-3 affect what the paper's Methods section can truthfully claim.

## 1. §A.4's determinism block is incomplete in a way that breaks the gate

The doc lists `torch.backends.cuda.matmul.allow_tf32 = False` but omits:

- `torch.backends.cudnn.allow_tf32` — a SEPARATE switch from the matmul one
- `CUBLAS_WORKSPACE_CONFIG` — without it, `use_deterministic_algorithms(True)` THROWS at the
  first cuBLAS GEMM. Must be set before CUDA context creation, not just before the call.
- `torch.backends.cudnn.benchmark = False` — arguably the single most likely cause of a
  run-to-run difference at fixed shape, since autotuning can select different kernels.

Implemented correctly in `src/ling/determinism.py`. **Anyone implementing §A.4 literally
gets a gate failure for a reason §A.4 does not mention.** Patch the doc so the paper's
one-sentence determinism claim is accurate.

## 2. §A.2 says "raw HF forward hooks" but never specifies eager attention

With default SDPA/flash kernels the gate can fail non-deterministically depending on
sequence length. This is a determinism CONTROL, not an implementation detail, and belongs in
§A.4 alongside the others.

## 3. §A.6's "the final token" is underspecified against trailing special tokens

If one condition's tokenization appends an end-of-turn token and another's does not, the
`FINAL` anchor compares a special-token state in one condition against a content-token state
in another — a length-alignment confound of exactly the kind §A.6 exists to prevent.

Made a hard error in the implementation. The design doc should state the rule explicitly.

## 4. §A.1's two rules are in tension and the doc does not resolve it

"Checkpoint after every layer" and "never persist raw activations" conflict with the access
pattern. Clustering layer *i* needs all 400 prompts at layer *i*, but a forward pass produces
all layers of ONE prompt. Something must hold the transpose.

Resolved in implementation as an in-RAM buffer (~4.9 GB HOST RAM, not VRAM) with an optional
self-deleting scratch spill that refuses to sit under `results/` or `data/`. **The design
should state which it intends**, because "never persist raw activations" reads as an
absolute and the implementation makes a narrower reading ("raw activations are never a run
artifact").

## 5. Consequence of pre-norm capture that must be stated in the paper

Because we capture the post-block, PRE-normalization residual stream, activation norms grow
with depth. The downstream affinity must therefore be either cosine or per-layer
standardized — and **which one must be stated, because lambda2 is not invariant to that
choice.** This is a live decision that the clustering implementation must make explicitly
rather than inherit from a default.
