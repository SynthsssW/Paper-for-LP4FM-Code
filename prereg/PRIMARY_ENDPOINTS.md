# Pre-registered primary endpoints

Spectral cluster-transition geometry under paraphrase and formal rewriting (LP4FM 2026).

This file is written BEFORE any analysis code is run. Per DESIGN.md §A.10, exactly one
primary endpoint is declared per hypothesis. Everything else in the results tables is
EXPLORATORY and must be labeled as such.

Any change to this file after the first analysis run is a protocol deviation and must be
recorded in `prereg/DEVIATIONS.md` with a reason, not silently edited.

---

## Primary endpoints (three, one per hypothesis)

### H1 — spectral clusters align with linguistic structure better than K-means
- **Endpoint:** AMI between cluster assignment and linguistic label, at the single layer
  with the largest eigengap on NL-base.
- **Affinity:** lambda = 0 (purely geometric). Stating H1 at lambda > 0 is circular because
  linguistic similarity would be injected into the affinity we then score against
  linguistic labels.
- **Comparator:** K-means at the same k, same layer, same token states.
- **Direction (pre-committed):** AMI(spectral) > AMI(kmeans).
- **Test:** Wilcoxon signed-rank over problems; bootstrap CI resampling PROBLEMS.

### H2 — answer-breaking realizations fork earlier, surviving surface-dissimilarity control
- **Endpoint:** the coefficient on answer-survival status in a discrete-time survival model
  for fork layer, complementary log-log link, person-period expansion.
- **Covariates:** surface dissimilarity (lexical: normalized edit distance / token Jaccard;
  syntactic: dependency-tree distance), entered as controls. The residual effect of
  answer-survival status after these controls IS the contribution.
- **Direction (pre-committed):** answer-breaking pairs have a HIGHER hazard of forking at
  each layer, i.e. fork earlier.
- **Censoring:** pairs that never fork within model depth are right-censored. They are
  NEITHER dropped NOR assigned a fabricated event layer; they contribute all non-fork rows.
- **Random effect:** intercept for problem.

### H3 — formal rewriting DECREASES lambda2
- **Endpoint:** mean paired difference in per-layer-pair lambda2, averaged across layers,
  formal minus NL-base.
- **Direction (pre-committed):** NEGATIVE. Algebraic connectivity is LOW when a good cut
  exists. "Fewer, better-separated regions visited in sequence" is a low-lambda2 graph
  (Cheeger). lambda2 and path entropy are predicted to move DOWN together.
- **Test:** Wilcoxon signed-rank over problems, plus a permutation null resampling PROBLEMS.
- **Conditioning:** per-layer-pair lambda2 averaged across layers is PRIMARY. Concatenated
  ~40-layer-chain lambda2 is SECONDARY (path-like graph, very small lambda2, variance may be
  numerically dominated).

---

## Pre-committed analysis decisions

1. **Fixed k across conditions.** k is selected per layer on NL-base by eigengap, then that
   same k is applied to all four realizations at that layer. k is NEVER recomputed per
   condition; doing so confounds cross-condition lambda2 with graph size.
2. **Aligned positions only.** Token states are restricted to semantically anchored aligned
   positions: the final token, plus the position where each problem quantity is introduced.
   Present in all four realizations. Length-stratified regression is an APPENDIX robustness
   check, not the primary analysis.
3. **Resampling unit is the PROBLEM, not the token state.** All bootstraps and all
   permutation nulls resample problems. Token states within a problem are dependent.
4. **Multiplicity.** Benjamini-Hochberg FDR at q = 0.05 across the layer-wise family for
   anything reported per layer. The three primary endpoints above are not corrected against
   each other.
5. **Metrics retained:** lambda2, path entropy, fork layer. Spectral entropy is DROPPED
   (redundant with lambda2). AFRC is DROPPED (no causal experiment). The correlation matrix
   across the three retained metrics is reported.
6. **lambda2 is log-transformed** before any regression.
7. **Nystrom landmarks** are sampled randomly from the POOLED set across conditions, never
   per condition.
8. **Distributional families:** fork layer = discrete-time survival (cloglog); lambda2 =
   Gamma GLM (log link) or log-transform + Gaussian; path entropy = Beta regression on the
   normalized value; AMI = Beta regression or bootstrap CIs only.

## Pre-committed gates (stop conditions)

- **G1 Determinism.** Each prompt run twice under identical batch conditions (batch size 1,
  no padding, fixed dtype/device, deterministic algorithms, TF32 off). Fork signal must be
  EXACTLY zero. If not, it is a plumbing bug; the run STOPS and nothing downstream is
  reported.
- **G2 Behavioral.** Accuracy is measured on all four realizations before the geometry
  analysis. If formal accuracy is not above NL-base, the accuracy half of H3 is CUT and the
  paper is written as the dissociation result ("formal rewriting restructures routing
  geometry without improving accuracy"). This decision is made by the measurement, not after
  seeing the geometry.
- **G3 Corpus.** Round-trip verification by a DIFFERENT model than the generator. Any item
  where any realization yields a different derived answer is dropped. The drop rate is
  reported.
- **G4 Nystrom.** Approximation error against the exact spectrum on a held-out subset is
  computed and reported.

## Declared negative-result commitment

If the primary endpoints do not reach significance, the result is reported as a bounded
negative result, not reframed around whichever exploratory endpoint reached p < 0.05.
