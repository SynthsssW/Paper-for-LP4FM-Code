# Protocol deviations

Amendments to `prereg/PRIMARY_ENDPOINTS.md` (sealed 2026-09-06T20:47:45Z,
sha256 d138968eab9b6b57e1e27b7239722d3a9ccbe5a0e56f21347233f8cfcd5f9501).

The sealed file is never edited. Every change is recorded here with its reason and
timestamp.

**Both amendments below were made BEFORE any activation was extracted, before any model was
loaded, and before any data existed.** They are pre-data respecifications, not
post-hoc revisions in response to results. Nothing had been observed that could have
motivated them.

---

## Amendment 1 — H1 demoted from hypothesis to methods note

**Timestamp:** 2026-09-06T21:02 UTC
**Status:** H1 no longer has a primary endpoint. Two primary endpoints remain (H2, H3).

**Change.** H1 ("purely geometric spectral clusters align with linguistic structure better
than K-means") is demoted from a tested hypothesis to a methods note that establishes the
clustering front end. AMI vs K-means is still computed and reported, with bootstrap CIs
resampling problems, but is labeled DESCRIPTIVE and carries no primary endpoint and no
directional pre-commitment.

**Reasons.**

1. *Near-tautological.* Spectral clustering recovers non-convex structure and K-means does
   not. Activation manifolds are non-convex. The result is close to entailed by the choice
   of comparator, so a positive finding teaches little about the model.
2. *Label-dependent.* AMI against "linguistic structure" depends almost entirely on which
   labels are used — POS, dependency role, semantic frame. DESIGN.md §B.3 correctly defers
   the multi-tagger and semantic-frame treatment to the archival version, which leaves the
   overnight version's label choice unspecified and therefore the endpoint underdetermined.
   Declaring a primary endpoint over an underspecified label set is not a real
   pre-commitment.
3. *Page budget.* Four pages, three hypotheses, is crowded. H2 and H3 are the two claims
   that are not foregone conclusions.

**Effect on the paper.** H1 becomes a short methods paragraph justifying the front end.
DESIGN.md §A.7's requirement that it be stated at lambda = 0 still holds for the descriptive
report — reporting it at lambda > 0 would still be circular, demotion does not change that.

**Effect on the pipeline.** None. The corpus still requires deliberate linguistic variation
across templates, k is still selected on NL-base by eigengap, and the eigengap-maximizing
layer is still identified — H3 and the k-selection procedure both depend on it.

---

## Amendment 2 — H3 endpoint respecified: lambda2 is a POOLED statistic

**Timestamp:** 2026-09-06T21:02 UTC
**Status:** endpoint quantity and direction UNCHANGED. Unit of pairing and source of
uncertainty changed.

**The defect.** The sealed file specifies the H3 endpoint as "mean paired difference in
per-layer-pair lambda2, averaged across layers, formal minus NL-base," tested by "Wilcoxon
signed-rank over problems." Those two clauses are not jointly satisfiable.

Under DESIGN.md §A.6 the analysis is restricted to semantically anchored aligned positions —
the final token plus each quantity-introduction point. A GSM-Symbolic problem has roughly
three to five quantities, so a single problem contributes on the order of four to six token
states per layer. A per-problem cluster-transition graph therefore has at most ~6 edges
spread across 2k nodes, leaving the graph overwhelmingly disconnected. lambda2 of a
disconnected graph is exactly 0. Per-problem lambda2 is degenerate, not merely noisy, and a
Wilcoxon over problems would be comparing zeros to zeros.

**Respecification.**

- lambda2 is computed on the **pooled** cluster-transition graph: one value per condition,
  per layer-pair, built from the token states of all surviving problems.
- **Primary endpoint (unchanged in substance):** mean paired difference in per-layer-pair
  lambda2, formal minus nl_base, averaged across layer-pairs.
- **Direction (unchanged, still pre-committed): NEGATIVE.** Formal rewriting decreases
  lambda2. Cheeger: algebraic connectivity is low when a good cut exists.
- **Pairing unit:** layer-pairs (n ~ 40), not problems. Wilcoxon signed-rank paired across
  layer-pairs.
- **Uncertainty and the permutation null:** bootstrap resamples **PROBLEMS**, rebuilds the
  pooled graph from the resampled problems, and recomputes lambda2. This preserves the
  §A.9.2 requirement that the resampling unit is the problem and never the token state. The
  requirement was never about which axis the paired test runs along; it was about what
  counts as an independent sampling unit, and problem-level bootstrap satisfies it.
- Same respecification applies to path entropy, for the same structural reason.

**Why this is not a weakening.** The pre-committed direction is untouched, and the bootstrap
still treats problems as the independent unit, which is the substantive statistical
commitment. What changed is the recognition that lambda2 is a property of a graph built from
many problems, not a per-problem observable.

**Consequence to report in the paper.** Because lambda2 is pooled, the effective sample size
for H3's paired test is the number of layer-pairs, not the number of problems. This must be
stated plainly rather than left for a reviewer to infer from the degrees of freedom.
