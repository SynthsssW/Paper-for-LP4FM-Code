# Methods facts sheet

Everything the Methods section must state that is not recoverable from `analysis.json`.
Numbers marked TODO arrive from the GPU run.

## Model
- `google/gemma-4-12B-it`, **dense** (not MoE), 11.95B params, **48 layers**, bf16, **unquantized**.
- `attn_implementation="eager"`. SDPA/flash kernels are not bitwise reproducible across
  shapes; eager is a determinism control, not an implementation detail.
- **Thinking mode disabled.** Extraction never applies a chat template, so `<|think|>` never
  enters the prompt. Transfer to the reasoning-trace regime is explicitly open.
- Loader hard-refuses MoE configs and any `quantization_config`.

## Determinism (gate G1)
`use_deterministic_algorithms(True)`; `cudnn.deterministic=True`, `cudnn.benchmark=False`;
TF32 off for **both** `cuda.matmul` and `cudnn`; reduced-precision reductions off;
`CUBLAS_WORKSPACE_CONFIG` set before CUDA init; batch size 1, no padding, fixed dtype/device.
Each prompt run twice under identical conditions; residuals compared **bitwise at every
layer**, plus a reverse-order re-run to catch state leaking between prompts.
**Result: TODO (must be exactly zero).**

## Residual capture
Hook on the **output of block i** (`residuals[i] == hidden_states[i+1]`), i.e. the
**post-block, PRE-normalization** residual stream. Verified mechanically against HF's own
`output_hidden_states` on a 3-block model: `norm(residuals[-1]) == hidden_states[-1]` while
the un-normed version differs.
**Consequence that must be stated:** norms grow with depth, which is why the affinity is
cosine (below).

## Corpus
- **Apple GSM-Symbolic templates** (Mirzadeh et al., ICLR 2025), all 100, vendored at pinned
  commit `b6a1625`. Freshly sampled instantiations.
- **Say "fresh instantiations of published templates." NEVER "uncontaminated."** The
  templates derive from GSM8K and have been public since Dec 2024; the instantiations are
  fresh, the surface frames are not.
- **The DSL interpreter and lexical word pools are ours, not Apple's** — Apple published the
  templates but never the generator or the pools. Numeric pools, `#conditions:` and
  `#answer:` expressions are verbatim; surface noun pools are a reconstruction.
- **100 problems x 6 conditions = 600 prompts.** 569 available cells, **31 declared holes**
  (`nl_minimal_lex` 30 — no proper name to swap; `nl_minimal_orth` 1 — no digit-rendered
  integer). Holes are per-condition skips, never whole-problem drops, and are reported
  separately from the G3 drop rate.
- Build drop rate **0.0%**; 100/100 verified mechanically (arithmetic re-executed, 1600+
  anchor spans checked for exact `text[start:end] == literal`, `nl_break` answer-change
  confirmed by recomputation). Byte-reproducible under a recorded sha256.
- **G3's "different model than the generator" is trivially satisfied and should be said so.**
  No model generated `nl_base`, `formal`, `nl_break`, or the two style controls — they are
  emitted from templates. Only `nl_para` is authored, and it is the item under human review.

## Conditions

| condition | edit | answer | lex_edit mean | role |
|---|---|---|---|---|
| `nl_base` | — | — | — | reference |
| `nl_para` | prose rewrite | preserved | 0.3911 | large-edit control |
| `formal` | notation | preserved | 0.3072 | **H3 primary** |
| `nl_break` | 1 token @ anchor | **broken** | 0.0050 | **H2 primary** |
| `nl_minimal_orth` | digits->words @ anchor | preserved | 0.0486 | upper style floor |
| `nl_minimal_lex` | proper-name swap, off-anchor | preserved | 0.0240 | lower style floor |

**`formal` is NOT "a Python rewrite."** It is quantities lifted to symbolic bindings plus an
explicit GIVEN/WHERE/FIND frame, **with the situation retained in prose**. Full
formalization makes the problem unanswerable and would make G2 measure unanswerability.

**`nl_minimal_orth` vs `nl_break` is matched on edit SITE, not magnitude.** 95/99 edit the
exact slot `nl_break` perturbs, but orth is ~8x the edit size (`7775` ->
`"seven thousand seven hundred seventy five"`). Never write "matched magnitude."
This makes the pre-committed direction **conservative**: the larger surface edit would push
orth to fork earlier, against the prediction.

Adding the two style controls cut covariate/treatment collinearity from
**r = 0.883 -> 0.471** (lexical) and **0.868 -> 0.449** (syntactic), and the previously
disjoint ranges now overlap. That is why H2's residual is identified at all.

## Aligned positions (§A.6)
Final token plus each quantity-introduction site, present in every available condition.
Anchor char spans converted to token indices via the tokenizer offset mapping — never by
re-tokenizing a substring. **3-10 anchors per cell, median 5. 2,852 pooled token states per
layer.**

## Clustering
- **Cosine affinity** (Amendment 3): scale-invariant, required because the pre-normalization
  residual stream's norms grow with depth and H3 averages **across** layer-pairs, so a
  depth artifact would accumulate rather than cancel.
- **Gaussian kernel on cosine distance**, not a literal cosine gram matrix — a linear kernel
  makes normalized spectral clustering unable to recover non-convex structure, which is the
  entire justification for the front end.
- **Bandwidth sigma** = median k-th nearest reference distance on the **pooled** set,
  computed once per layer and used for both k-selection and clustering. Pre-committed
  (Amendment 5); the primary endpoint is not selected across bandwidths.
- **Normalized symmetric Laplacian.** lambda2 = second-smallest eigenvalue, trivial zero
  skipped; a disconnected graph returns a literal 0.0 from a component count, not a noisy
  eigensolve.
- **k selected once per layer on `nl_base` by eigengap, reused across all six conditions.**
  Structurally enforced: `select_k` raises on any non-base row; `apply_clustering` has no
  selection code.
- **Exact eigendecomposition — Nystrom implemented but NOT used** at 2,852 rows (threshold
  3,000). **No approximation error enters any reported spectrum; gate G4 is inapplicable
  rather than unmet.**

## Statistics
- **lambda2 is POOLED** (Amendment 2), one value per condition per layer-pair. Per-problem
  graphs are degenerate: ~5 anchors give a disconnected graph whose lambda2 is exactly 0.
- **Layer-pair p = (output of block p, output of block p+1), p in 0..46. n = 47.**
  **H3's effective sample size is layer-pairs, not problems — state this plainly**, or a
  reviewer infers it from the degrees of freedom.
- **H3:** Wilcoxon signed-rank paired across layer-pairs, `alternative="less"`.
- **H2:** discrete-time survival, person-period expansion, **complementary log-log** link.
  Right-censored pairs are neither dropped nor assigned a fake event layer — they contribute
  all non-fork rows.
- **Cluster-robust (sandwich) SEs on `problem_id`, NOT a GLMM.** At n=100 a GLMM is unstable;
  the GLMM belongs in the archival version. Say so.
- **Every bootstrap and permutation resamples PROBLEMS** and rebuilds the pooled graph.
  Token states within a problem are dependent.
- **BH FDR q=0.05** across layer-wise families. The primary endpoints are not corrected
  against each other.

## Endpoints
- **H2 primary:** coefficient on answer-survival in the survival model, WITH lexical and
  syntactic dissimilarity covariates. Predicted: breaking forks earlier.
- **H3 primary:** mean paired lambda2 difference across layer-pairs, formal - nl_base.
  **Predicted NEGATIVE** (Cheeger: algebraic connectivity is low when a good cut exists).
- **Pre-registered secondary:** hazard ratio `nl_break` vs `nl_minimal_orth`.
- **Interpretive secondary:** style-normalized displacement, reported as a range over both
  floors, and ONLY where the denominator's bootstrap CI excludes zero.
- **H1 is a DESCRIPTIVE methods note** — no test, no direction, no primary endpoint.

## Pre-registration
Endpoints sealed by sha256 **before any analysis code existed**
(`d138968e...`, 2026-09-06T20:47:45Z). Six amendments and two self-corrections, each
recorded with its reason and git commit hash rather than silently edited, **all made before
any model was loaded or any activation existed**. Includes a commitment to report a bounded
negative result rather than reframing around whichever exploratory endpoint reaches p<0.05.
Verify with `git log --format='%h %cd' --date=iso -- prereg/DEVIATIONS.md`.
