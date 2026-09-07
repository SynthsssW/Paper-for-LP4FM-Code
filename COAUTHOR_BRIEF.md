# Coauthor brief — everything needed to write the paper

**Target:** LP4FM @ NeurIPS 2026. 4 pages main text, double-blind, non-archival.
Every number below is in `results/` and reproducible from this repo.
Where a claim has a safe and an unsafe phrasing, both are given. **Use the safe one.**

---

## 1. The claim

> A widely used measure of internal divergence — the layer at which two prompts' internal
> representations separate — fires on rewrites that do **not** change the model's answer, and
> does not fire on the one edit that **does**.

Supporting, in one line: formal notation restructures the geometry *and degrades accuracy*
(56.0% vs 70.0%, p = 0.007), so geometric change and performance come apart.

## 2. Setup (Methods §1)

- **Model:** `google/gemma-4-12B-it`. Dense (not MoE), 11.95B params, **48 layers**,
  d_model 3840, **bf16, unquantized**. Thinking mode OFF. `attn_implementation="eager"`.
- **Corpus:** 100 problems from Apple's **GSM-Symbolic** templates (Mirzadeh et al., ICLR
  2025), pinned at commit `b6a1625`, freshly sampled instantiations. **6 conditions = 600
  prompts**, 569 available cells (31 declared holes, see §6).
- **States analysed:** only *aligned anchor positions* — each quantity-introduction site plus
  the final token — present in every condition. **2,852 pooled token states per layer.**
- **Residual capture:** output of block *i*, i.e. post-block **pre-normalization** residual
  stream. Verified against HF's own `output_hidden_states` on a 3-block model.
- **Clustering:** spectral, **cosine affinity** (Gaussian kernel on cosine distance),
  normalized symmetric Laplacian, *k* selected once per layer on `nl_base` by eigengap and
  reused across all six conditions. **Exact eigendecomposition — Nyström implemented but not
  required at 2,852 rows, so no approximation error enters any spectrum.**

### The six conditions

| condition | edit | answer | mean Δtokens | accuracy |
|---|---|---|---|---|
| `nl_base` | — | — | — | 70.0% |
| `nl_para` | paraphrase | preserved | 3.7 | 69.0% |
| `formal` | GIVEN/WHERE/FIND frame, quantities as symbols | preserved | 23.0 | **56.0%** |
| `nl_minimal_orth` | one number digits→words | preserved | 0.9 | 69.7% |
| `nl_minimal_lex` | one proper name swapped | preserved | 0.0 | 70.0% |
| **`nl_break`** | **one digit** | **CHANGED** | 0.0 | 67.0%* |

\* scored against its **own** recomputed answer. It returned the *original* answer **0/100**
times.

**`formal` must NOT be called "a Python rewrite."** Safe phrasing: *quantities lifted to
symbolic bindings plus an explicit given/goal frame, with the situation retained in prose.*
Full formalization makes the problem unanswerable; the situation text is deliberately kept.

## 3. The result (Results §1)

**Gate G2 (behavioural), 569 cells, greedy, thinking off:** `nl_break` returned the
unperturbed answer **0/100** times. The one-character edit changes what the model computes in
**100 of 100** cases.

**Fork detection (forced k=8):**

| condition | fork rate | answer changed? |
|---|---|---|
| `formal` | **100/100** | no |
| `nl_para` | 35/100 | no |
| `nl_minimal_orth` | 8/99 | no |
| `nl_minimal_lex` | 4/70 | no |
| **`nl_break`** | **0/100** | **yes, 100/100** |

Same at the pre-registered eigengap-*k*: `nl_break` **0/100**.

**Regression (the pre-registered §4.2 test), n = 469:** surface dissimilarity accounts for
**McFadden R² = 0.397** of fork occurrence. Adding answer-breaking status adds **0.026**, and
its coefficient is **unidentifiable** — complete separation, zero events in that arm.
**Report `b_break` = −22.1 as unidentifiable, never as an effect estimate.**

## 4. Relation to prior work — the one place to be careful

Han, Rizk & Santosh (arXiv **2604.01639**) compare flipped vs stable samples **within** a
perturbation type, with no surface-dissimilarity control.

- **DO NOT WRITE:** "we refute their finding," or that their effect is surface-driven.
  **Their setup was not rerun.**
- **SAFE:** their design cannot separate a surface-driven from an answer-driven signal; a
  design with a surface control can; in ours the signal goes entirely to surface. That is
  *consistent with* their effect being confounded and is something their design cannot rule
  out.

## 5. What we could NOT determine — state as a limitation

**Which surface property drives fork occurrence is unidentified.** Three answers were tried
and two retracted (full trail in `prereg/DEVIATIONS.md` Obs. 5b → 5c → 5d):

| single predictor | all 5 conditions | **`formal` excluded** |
|---|---|---|
| lexical edit distance | 0.251 | **0.170** |
| syntactic (dep-bag TV) | 0.395 | 0.122 |
| token-count difference | **0.525** | **0.050** |

Token-count difference collapses to 0.050 without `formal`, which is 54% longer and forks
100%. **Safe phrasing:** *among conditions matched for length, lexical, syntactic and length
contributions cannot be separated.* Do not claim a syntactic or a length mechanism.

**One oddity, one sentence, not a claim:** `nl_minimal_lex` and `nl_break` are identical on
every surface measure (Δtokens 0, length ratio 1.00) yet fork 5.7% vs 0.0%, Fisher exact
p = 0.027 — on **4 events**, uncorrected for multiplicity.

## 6. Also report (do not omit)

- **G1 determinism gate PASSED:** 8 probes × 2 passes × 48 layers, max |Δresidual| = **0.0**,
  max |Δlogit| = **0.0**, bitwise identical, order-invariant.
- **G3 drop rate 0/100.** 31 declared holes (`nl_minimal_lex` 70/100 — 30 problems have no
  proper name; `nl_minimal_orth` 99/100), reported separately from drops, never imputed.
- **Corpus caveats:** say *"fresh instantiations of published templates,"* **never
  "uncontaminated"** — GSM-Symbolic derives from GSM8K and has been public since Dec 2024.
  The **DSL interpreter and lexical pools are ours**, not Apple's. G3's "different model than
  the generator" is **trivially satisfied** because no model generated five of six
  conditions — say so rather than implying earned independence.
- **Pre-registration:** endpoints sealed by sha256 **before any analysis code existed**
  (`d138968e…`, 2026-09-06T20:47:45Z). Six pre-data amendments, then a marked post-data
  section, each entry carrying its git commit hash. Verify:
  `git log --format='%h %cd' --date=iso -- prereg/DEVIATIONS.md`.
- **Our own hypotheses failed.** H3 predicted λ₂ would *decrease*; it increased (+0.360,
  CI [0.211, 0.385]) and then shrank to +0.094 (p = 0.56) under forced k=8 — **not robust to
  k**. H2 predicted answer-breaking forks earliest; it never forks. H1 (spectral > K-means)
  is unsupported. Report all three.
- **Mechanism of the metric's failure:** eigengap selected **k=2 on 40/48 layers**; 10/48
  layers isolate <1% of states and **5 isolate a single token**. This accounts for λ₂'s
  quantization to {0,1,2}, its 2.0 ceiling, and 94.7% censoring.

  **PHRASING — the norm check was NOT run.** A residual-norm probe (`scripts/07_norm_probe.py`)
  was written to test whether these outliers are the known massive-activation phenomenon, but
  GPU access ended before it executed. The evidence we have is cluster-size degeneracy, not
  measured norms.
  - **SAFE:** "a small number of states dominate the partition; this is *consistent with*
    massive activations, which we did not measure directly."
  - **DO NOT WRITE:** "caused by massive activations," or cite a norm ratio. We have none.
  - The script is in the repo and is a ~2-minute run for anyone with GPU access.

## 7. Limitations to write explicitly

1. Single model, single scale, dense only. MoE excluded because routing is a literal
   architectural fact there (the strongest future validation).
2. Thinking mode off — transfer to the reasoning-trace regime is open.
3. `nl_break` is a one-character edit, far more minimal than prior work's flipped samples.
   The claim is about what the diagnostic tracks when surface magnitude is controlled.
4. Surface attribution unidentified (§5); needs a condition set with length orthogonal to
   syntactic distance. **The pre-registered covariate set omitted token-count difference.**
5. Paraphrase meaning-preservation is **not yet human-verified** —
   `results/human_review_sheet.md` is unread. Mechanical checks all passed.
6. λ₂/path-entropy results are exploratory beyond the sealed endpoints; the post-data section
   of `DEVIATIONS.md` makes several comparisons with no multiplicity correction.

## 8. Figures

- **`fig_headline.pdf`** — fork rate by condition against whether the answer changed. **The
  main figure.**
- **`fig2.pdf`** — discrete-time survival curves, censoring marked; the layer-resolved version
  of the same contrast.
- Appendix if space: `fig0_method.pdf` (construct schematic), `fig1.pdf` (λ₂ by layer-pair,
  shows the {0,1,2} quantization), `fig3.pdf` (eigengap, shows k=2 selection).

## 9. Files

| file | contents |
|---|---|
| `FINDINGS.md` | plain-English summary |
| `results/METHODS_FACTS.md` | pre-results methods claims |
| `prereg/PRIMARY_ENDPOINTS.md` | sealed predictions |
| `prereg/DEVIATIONS.md` | 6 pre-data amendments + post-data findings with corrections in place |
| `results/g2_behavioral.json` | per-condition accuracy |
| `results/analysis.json` / `analysis_fixedk8.json` | pre-registered / k-sensitivity |
| `results/main/assignments.jsonl` | cluster assignments, 48 layers × 2852 rows |
