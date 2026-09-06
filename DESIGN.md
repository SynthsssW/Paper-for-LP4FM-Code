# Experimental design — LP4FM @ NeurIPS 2026

**Spectral cluster-transition geometry under paraphrase and formal rewriting**

Deadline: **Sep 7, 2026, 11:59 UTC**. Part A is the executable overnight build.
Part B is the hardened protocol — written into Methods as the pre-registered design
of which we execute a subset; it is the archival extension.

Framing: **standalone project**. No lineage argument, no "prior work in this program."

---

# PART A — The overnight build

## A.0 Framing: standalone, not a migration story

Do not motivate spectral clustering by arguing against a previous K-means front end.

- Double-blind forces it anyway: cannot write "our prior work" in an anonymized submission.
- It costs a page we do not have: in a 4-page paper a motivation section arguing with an
  invisible opponent is ~25% of the budget.
- It reads stronger standalone. Motivation is three sentences: paraphrase brittleness is
  well documented behaviorally; it has been localized to a layer via logit lens; nobody has
  a representation-level account of *what changes*. Spectral clustering then justifies
  itself directly — recovers non-convex clusters, admits principled *k* via eigengap, and
  lets linguistic similarity enter the affinity so the partition is linguistically
  constituted rather than annotated.

Disclosure caveat: if the related preprint is public, cite it once in Related Work in third
person, as any other paper. Normal disclosure, not lineage; protects against an overlap
question at the archival stage.

The timestamped pre-registration file (A.9/A.10) replaces the old credibility signal (a
prior public retraction) and is stronger, because prospective rather than retrospective.

## A.1 Compute: buy hours, not FLOPs

$10 at H100 rates (~$2.50/hr) is 4 hours — wrong purchase. Activation extraction is memory-
and I/O-bound, not compute-bound.

**Rent an A40 48GB at ~$0.35–0.45/hr → 20+ hours for the same $10.**

Three operational rules:

1. **Never persist raw activations.** Extract -> cluster -> write cluster assignments ->
   discard. Assignments for 400 prompts x ~40 layers x ~20 positions are a few hundred KB.
   Raw activations are several GB, push you onto network volumes, and mean a preempted pod
   costs the whole run instead of one layer.
2. **Model download is billed time.** ~24GB onto a fresh pod is 5–15 paid minutes. Use a
   network volume or cached template if you expect to restart pods.
3. **Checkpoint after every layer.** A crash at layer 31 should cost you layer 31.

## A.2 Model — dense, current, bf16, unquantized

**Primary: Gemma 4 12B, bf16, thinking disabled.**

27B is out on a 48GB card: 27B bf16 is ~54GB and does not fit; the alternative is 4-bit
quantization, and **for a paper whose entire claim is about activation geometry,
quantization is an unforced error.** A reviewer asks whether you measured the model or the
quantization scheme, and there is no rebuttal phase to answer it.

| Model | Params | License | Note |
|---|---|---|---|
| **Gemma 4 12B** | 12B dense | Gemma Terms | primary; needs a one-line license note |
| Phi-4 14B | 14B dense | MIT | fallback; heavily synthetic training makes its NL distribution unusual — a real weakness at a *linguistics* venue |
| Qwen3.6-27B | 27B dense | Apache-2.0 | only if >=80GB |

**Verify checkpoint strings on HuggingFace before pulling.** The open-weights frontier
churned monthly through 2026.

**Thinking mode off.** The framework measures geometry in a *prompt-encoding forward pass*,
but reasoning models do the work in generated tokens. Disable it, state it, add one sentence
that transfer to the reasoning-trace regime is open — simultaneously a limitation and the
archival follow-up.

**Tooling.** TransformerLens lags new architectures by months. Use raw HF forward hooks on
`model.model.layers[i]`, or nnsight. Do not spend the night on library compatibility.

## A.3 Why MoE is disqualified tonight (and is the best archival experiment)

Llama 4 Scout, DeepSeek V4, GLM-5.2, Kimi K2.x are MoE. The central claim is **routing
modularity**. In an MoE, routing is a literal architectural fact — the router picks experts.
Measuring "routing modularity" in an MoE residual stream invites the question of whether you
measured your construct or the actual router, and you cannot answer it. **Dense only.**

Inverted, this is the strongest validity check available: in an MoE there is *ground truth*
for routing. If spectral cluster-transition lambda2 correlates with observed expert-routing
entropy, that is external validation that the construct measures what its name claims.
One sentence in Future Work timestamps it.

## A.4 Determinism gate (first 45 minutes)

Run before anything else. The naive version fails for a mechanical reason unrelated to the
hypothesis: greedy decoding at temperature 0 is **not bitwise deterministic** on GPU across
batch shapes. Float reduction order changes with batch size and padding, so identical
prompts in different batches yield different logits and a nonzero fork signal.

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False   # TF32 reduction noise
# batch size 1, no padding, fixed dtype (bf16), fixed device
```

Run each prompt twice under identical batch conditions. Fork signal must be **exactly zero**,
not "near zero." If not, it is a plumbing bug and everything downstream is measuring
numerical noise. One sentence in the paper; the cheapest credibility available.

## A.5 Corpus — 100 x 4, contamination-controlled

**100 problems x 4 realizations = 400 prompts.** Not 300 x 4. Hand-verification is the true
bottleneck.

**Contamination.** GSM8K is thoroughly contaminated by 2026 and a reviewer will say so.
Build NL-base from **GSM-Symbolic templates** (Mirzadeh et al., ICLR 2025) with freshly
sampled instantiations. Costs nothing — the templates exist — and converts a known weakness
into a named, defensible choice.

**Verification, 90 minutes, hard stop.** Generate all four realizations with a strong model,
then round-trip verify: a *different* model independently recovers the ground-truth answer
from each realization. Any item where any realization yields a different derived answer is
dropped without discussion. **Log the drop rate** — a reportable number that pre-empts
"how do you know these are paraphrases."

## A.6 Two confounds and two invalidities

**Token-position non-alignment.** A Python rewrite has different token count, tokenization
statistics, whitespace structure, and positional distribution than English prose. The graph
is built from *token states*, so a lambda2 difference between a 60-token NL prompt and a
28-token Python prompt is confounded with length before semantics enter. Worse than the
edit-distance confound, because there is no clean regression control.

Fix: **restrict to semantically anchored aligned positions** — the final token, plus the
positions where each quantity in the problem is introduced. Present in all four
realizations. This is also the compute fallback, so it is free. Report length-stratified
regression as an appendix robustness check. Say you did this in the paper: "you're comparing
prose to code, of course the geometry differs" is the most likely reviewer objection to H3.

**Fixed *k*, not per-layer eigengap *k*.** lambda2 depends on node count. If NL-base gets
k=14 at layer 12 and the formal rewrite gets k=9, the lambda2 difference is partly graph
size. **Select *k* per layer on NL-base by eigengap, then apply that same *k* to all four
realizations at that layer.** Report eigengap values and flag flat-spectrum layers as
planned.

## A.7 Hypotheses — three, with H3's sign corrected

**H1 — Purely geometric spectral clusters align with linguistic structure better than
K-means.** Stated **at lambda = 0**. Not stylistic: at lambda > 0 linguistic similarity is
injected into the affinity, so AMI against linguistic labels partly measures our own
construction and H1 is circular. lambda = 0 is the claim about the model. lambda > 0 is a
design knob reported separately as "a linguistically-informed front end improves downstream
prediction" — a contribution, but a different one.

**H2 — Answer-breaking realizations fork earlier, and it survives surface-dissimilarity
control.** Framed as **replication with a control the original lacked**. Han, Rizk & Santosh
(arXiv 2604.01639) showed flipped samples diverge at significantly earlier layers across
Mistral, Llama-3 and Qwen2.5, with no surface-dissimilarity control at all. Our contribution
is the residual plus the syntactic-vs-lexical contrast they do not have. Say this in the
intro; do not let a reviewer find it.

**H3 — Formal rewriting *decreases* lambda2.**

The sign in the original plan was wrong. Algebraic connectivity is *high* for well-mixed
expander-like graphs and *low* when a good cut exists — bottleneck, community structure,
chain. Cheeger's inequality makes this tight. "Fewer, better-separated regions visited in
sequence" is a **low**-lambda2 graph. The original predicted modularity increases and
operationalized it as lambda2 increasing, which is self-contradictory and is caught in
thirty seconds by anyone who does spectral graph theory.

The corrected direction repairs an internal tension: lambda2 and path entropy now move
**down together**, one coherent prediction instead of two metrics in opposition.

**H4 — dropped.** Not overnight work, and partly pre-empted: Han et al. already apply
steering at mechanistically identified layers (12.2% / 7.2% / 5.2% recovery for Llama-3 /
Qwen / Mistral). Future Work.

## A.8 Metrics — three, with conditioning fixed

Keep **lambda2**, **path entropy**, **fork layer**. Drop spectral entropy (redundant with
lambda2 — same eigenvalue distribution). Drop AFRC unless H4 returns; it is the causal
handle and there is no causal experiment.

**Report the correlation matrix across retained metrics.** Known analytic bounds relate
discrete Ricci curvature to the spectral gap (Lin et al., used throughout the GNN
over-squashing rewiring literature). A reviewer from that world assumes redundancy unless
shown otherwise. Three lines converts an objection into a demonstration of care.

**lambda2 conditioning.** A concatenated ~40-layer chain of bipartite graphs is structurally
path-like, so its lambda2 is very small and its variance may be numerically dominated.
Report **per-layer-pair lambda2 averaged across layers** as primary; concatenated-graph
lambda2 as secondary. Same question, far better conditioned.

## A.9 Statistical models — drop the Gaussian conditional

The residual **is** the claim. If the conditional is misspecified, the residual absorbs
functional-form error, and "residual attributable to answer-preservation" becomes partly
"signal from fitting a line to something that isn't linear-Gaussian."

Permutation nulls do not rescue this. Permutation gives a valid p-value under
exchangeability, but it tests *the statistic you computed*. A valid p-value for a
contaminated quantity is still contaminated. **Permutation fixes inference; specification is
a separate problem and needs a separate fix.**

### A.9.1 Fork layer is censored survival data, not a Gaussian outcome

Fork layer is "the first layer at which a thing happens" — discrete-time survival, layer
index as the time axis. This surfaces **right-censoring**: some pairs never fork within the
model's depth. Both naive options are indefensible:

- *Drop them* -> biases toward pairs that fork. Answer-preserving pairs are exactly the ones
  more likely never to fork, so this biases **against our own hypothesis** by an
  unquantifiable amount.
- *Code them as layer 40* -> biases the other way, understates the effect.

**Discrete-time survival handles it natively.** Expand to person-period format (one row per
pair per layer, up to fork or end-of-depth), fit a binary GLM with a **complementary
log-log link**. Censored pairs contribute all their non-fork rows without a fabricated
event time.

### A.9.2 Family per outcome

| Outcome | Support | Model |
|---|---|---|
| **Fork layer** | discrete, right-censored | discrete-time survival, cloglog link |
| **lambda2** | strictly positive, right-skewed, very small | Gamma GLM (log link), or log-transform + Gaussian |
| **Path entropy** | bounded [0, log k] | Beta regression on the normalized value |
| **AMI** | bounded [0, 1] | Beta regression, or bootstrap CIs only |

**Random intercept for problem on all of them.** Four realizations are repeated measures on
the same item. Treating them as independent inflates significance. This is a *separate*
error from the Gaussian one and holds regardless of family.

**Bootstrap and permutation resample problems, not token states.** Token states within a
problem are dependent; resampling them gives confidence intervals far too narrow. Same for
the permutation null.

### A.9.3 Tonight vs. later

At n=100 problems, do not over-engineer. Complex GLMMs go unstable at this sample size.

Tonight, cheap:
- **Wilcoxon signed-rank** for paired lambda2 and path-entropy comparisons instead of paired
  t-tests. Distribution-free, one-line change, accepted without comment.
- **Log-transform lambda2** before any regression.
- **Keep the permutation nulls.** They do real work on the inference side.

Worth the extra 30 minutes:
- **Discrete-time survival for fork layer.** Not for elegance — for the censoring. That is a
  bias, not an efficiency loss.

Part B:
- Full GLMM with the families above, plus a Bayesian version with weakly informative priors.
  The Bayesian framing suits a pre-commitment to publishing a negative result: posterior
  intervals let you say "the effect is bounded below X" rather than "p > 0.05".

### A.9.4 Report the diagnostics

State distributional diagnostics for whichever family is picked — QQ plot in the appendix.
A reviewer's version of the Gaussian question is "why should I believe your error model,"
and one figure answers it.

## A.10 Multiplicity

Three hypotheses x three metrics x ~40 layers is on the order of 300 tests. With no
correction, something will be significant.

- **Declare one primary endpoint per hypothesis before looking at results.**
  H1: AMI at the layer with the largest eigengap.
  H2: the coefficient on answer-survival status in the survival model.
  H3: mean paired lambda2 difference across layers.
- Everything else is **exploratory**, labeled as such in the table.
- **Benjamini–Hochberg FDR at q=0.05** across the layer-wise family for anything reported
  per-layer.

Write the primary endpoints into a **timestamped text file before running the analysis**.
That is a real pre-registration without a registry.

## A.11 Review-agent checklist

Generic "find bugs" review finds style issues and misses what silently corrupts this
pipeline. Point the review agent at these specifically:

- [ ] **Pre- vs post-layernorm residual.** Whether the hook grabs the residual before or
      after the layer's normalization materially changes the geometry. Most common silent
      bug in this class of work, and it produces plausible-looking results either way.
- [ ] **Layer indexing off-by-one.** Input vs output of `layers[i]`.
- [ ] **Padding leakage.** If anything gets batched, padded positions produce garbage
      activations that get clustered as real states.
- [ ] **k recomputed per condition.** Verify *k* is selected once on NL-base and reused
      across all four realizations at that layer.
- [ ] **lambda2 index.** Confirm the code skips trivial lambda1 = 0 and takes second-
      smallest, and is consistent about normalized vs combinatorial Laplacian.
- [ ] **Nystrom landmarks** sampled randomly and from the *pooled* set, not per-condition.
      Per-condition landmark sets build in a between-condition difference before you measure
      one.
- [ ] **Resampling unit is the problem, not the token state.** A code reviewer will not
      catch this unprompted — state it explicitly.
- [ ] **Censoring handled** — pairs that never fork are neither dropped nor assigned a fake
      event layer.
- [ ] **Analysis script diffed against the pre-registered endpoints file**, to catch drift
      toward whatever looked significant.

## A.12 Sequencing

| Hours | Task | Gate |
|---|---|---|
| 0–1 | Determinism check, model load, hook plumbing | **Fork signal exactly 0, or stop** |
| 1–2 | Behavioral check: accuracy on all 4 realizations | **If formal ~ NL, H3's accuracy half is dead — cut and say so** |
| 2–3.5 | Corpus generation + round-trip verification, hard stop | Log drop rate |
| 3.5–6 | Activation extraction, aligned positions, checkpoint per layer | — |
| 6–8 | Spectral partition, fixed *k*, Nystrom + exact-spectrum error on held-out subset | Report approximation error |
| 8–10 | H1 (lambda=0) + ARI stability across seeds and bootstraps | — |
| 10–12 | H3, Wilcoxon + permutation null | — |
| 12–14 | H2, discrete-time survival with dissimilarity covariates | — |
| 14–17 | Write, anonymize, submit | **Submit at 16h, not 17h** |

**Write as you go.** Every block ends with its result pasted into the draft.

## A.13 Submission mechanics

- **Short paper, 4 pages** main text. References and appendices excluded; main text
  self-contained.
- **Double-blind, fully anonymized.** No lineage language. Scrub acknowledgements.
- NeurIPS 2026 workshop LaTeX template, single PDF, OpenReview.
- Non-archival, copyright retained, under-review NeurIPS papers explicitly welcome — the
  archival extension stays open.
- **Frame to the CFP in paragraph one.** Topic vi is verbatim about how paraphrases and
  formal rewrites reshape reasoning on the same task; topic v is mechanistic analysis of
  linguistic structure in internals. We sit on both. Cite the organizers' own references —
  Yang et al. (arXiv 2605.09271) and Wang et al., *Under the Shadow of Babel* (EMNLP
  Findings 2025) — and position against them.

---

# PART B — Hardened design (archival / Methods section)

## B.1 Model grid

| Axis | Levels | Isolates |
|---|---|---|
| Scale | ~12B / ~27B dense, same family | scale-dependence of the effect |
| Architecture | dense vs MoE | routing construct validity (A.3) |
| Reasoning mode | thinking on / off, same checkpoint | whether encode geometry predicts behavior when work happens in generated tokens |

The reasoning-mode axis is the most interesting and most current. Nobody has asked whether
prompt-encoding geometry still predicts outcomes once the model thinks in tokens. Survives
-> the diagnostic is much more valuable. Does not -> a clean, publishable boundary condition
on the whole geometric-interpretability program.

## B.2 MoE validation

Correlate spectral cluster-transition lambda2 against observed expert-routing entropy.
External construct validation, and the strongest single experiment available here.

## B.3 Construct validity for "linguistic structure"

- **Multiple independent taggers** (spaCy + Stanza) with agreement reported, so
  cluster–label alignment is not one parser's idiosyncrasy.
- **A semantic-frame level** (FrameNet / PropBank) alongside syntactic levels, so you can
  say *which* level the geometry tracks. "Aligns with dependency role but not semantic
  frame" is far more interesting than a pooled AMI, and is what a linguistics audience
  wants.
- **Permutation-of-labels null** in addition to random-partition null.

## B.4 Alternative front ends

- **SAE feature basis.** Not optional at the archival stage — the obvious competing
  principled front end.
- **UMAP/HDBSCAN.** If spectral beats K-means only because K-means is convex, HDBSCAN should
  also beat K-means. This ablation distinguishes "non-convexity matters" from "spectral
  matters" — different claims.

## B.5 Fork-layer construct robustness

- **Threshold sensitivity curve.** Sweep the bootstrap-calibrated disjointness threshold;
  show H2 is not an artifact of one setting.
- **Comparison against the incumbent scalar.** Han et al.'s first-divergence-layer and
  Cascading Amplification Index (AUC up to 0.679) are the standing diagnostics. Beat them on
  held-out prediction, or concede and argue on interpretability grounds. Either is
  defensible; silence is not.

## B.6 Statistical hardening

- Full GLMMs per A.9.2 with problem random effects, plus a Bayesian version with weakly
  informative priors.
- **Power analysis using Han et al.'s effect sizes as priors** — their flip rates
  (28.8–45.1%) and divergence-layer gaps give a real basis for sample size.
- Bootstrap CIs on every reported effect.
- Held-out split declared and hashed before analysis.
- Sensitivity analysis on the censoring assumption for fork layer.

## B.7 Competing accounts to address

- **Truth as a Trajectory** (arXiv 2603.01326) — layer-wise geometric displacement,
  explicitly motivated as escaping static lexical confounds. Same problem, different
  solution. Why is the residual regression stronger?
- **Invariant Features in Language Models** (arXiv 2605.06458) — semantically equivalent
  inputs occupy structured regions, paraphrastic variation along nuisance directions. Our
  "early fork" is their "nuisance leakage into the invariant subspace." Same claim in
  different coordinates?
- **Invariant Reasoning Directions in Latent Trajectories** (arXiv 2606.29164) —
  semantically equivalent inputs induce divergent trajectories *even when the answer is
  preserved*. Bounds the expected effect size: our contrast is a difference of degree, not
  presence vs absence. Budget for a modest effect and say so.

## B.8 The behavioral premise of H3 is contested

Whether **formal rewriting helps at all** is shaky. Strong evidence for formal notation
improving reasoning (PoT, PAL) comes from *executing* code — offloading computation to an
interpreter. Our formal condition is **input notation with no execution**, a different
intervention. A 2026 study across 1,000 GSM-Symbolic problems found chain-of-thought was the
*most robust* method (1.3pp drop) and concluded the robustness problem is not solved by code
execution.

So it is live that the formal condition shows no accuracy advantage. That does not kill the
paper — "formal rewriting restructures routing geometry without improving accuracy" is a
genuine dissociation, arguably a better result, because it separates the geometric claim
from the performance claim. But decide *in advance* which paper you are writing. The A.12
hour 1–2 behavioral check makes that decision.

---

## Fix list, priority order

1. **lambda2 sign in H3** — predict a decrease. Highest-value single fix.
2. **Cite Han et al. (2604.01639)** in the intro; reframe H2 as replication-plus-control.
3. **Fixed *k* across conditions** — per-layer eigengap *k* invalidates every cross-condition
   lambda2 comparison.
4. **Token-position alignment** for the formal condition.
5. **H1 stated at lambda = 0**, or it is circular.
6. **Censoring in fork layer** — discrete-time survival, not drop-or-impute.
7. **Standalone framing** — cut all lineage language.
8. **Gemma 4 12B bf16 on an A40** — no quantization in a paper about geometry; 20 GPU-hours
   not 4.
9. **Determinism controls** before the positive control.
10. **Primary endpoints declared** in a timestamped file before analysis.
11. Cut to 100 problems, 3 hypotheses, 3 metrics.
12. **GSM-Symbolic templates**, not raw GSM8K.
