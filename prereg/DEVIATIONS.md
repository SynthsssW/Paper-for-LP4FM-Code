# Protocol deviations

> **On timestamps.** The times below were originally typed by hand and several drifted from
> the real clock; one was stamped an hour in the future. They have been corrected to the
> **git commit time of the commit that introduced each amendment**, which is the
> authoritative record — a hand-typed timestamp in a pre-registration is worth nothing,
> since it is exactly the field a motivated author would backdate. Verify any entry with
> `git log --format='%h %cd' --date=iso -- prereg/DEVIATIONS.md`. The sha256 seals in
> `prereg/SEAL.txt` were machine-generated throughout and were always correct.
>
> **Every amendment below was made before any model was loaded and before any activation
> existed.** The first GPU run had not occurred when the last of them was sealed.

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

**Timestamp:** 2026-09-06T21:09 UTC (git commit `89b6967` — authoritative)
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

**Timestamp:** 2026-09-06T21:09 UTC (git commit `89b6967` — authoritative)
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

---

## Amendment 3 — affinity is COSINE; layer-pair index convention fixed

**Timestamp:** 2026-09-06T21:22 UTC (git commit `ea6615f` — authoritative)
**Status:** operationalization decision, made before any activation exists. No endpoint or
direction changes.

**Why this must be decided now.** The extraction captures the post-block, PRE-normalization
residual stream (correct, and mechanically verified). A consequence: activation norms grow
monotonically with depth. lambda2 is NOT invariant to whether the affinity is built on raw
vectors, cosine similarity, or per-layer standardized vectors.

Under Amendment 2 the H3 primary endpoint is averaged **across layer-pairs**. A
depth-dependent scale artifact therefore does not average out — it accumulates along the
exact axis the primary statistic sums over. Choosing the affinity after seeing lambda2 would
be a researcher degree of freedom on the primary endpoint.

**Decision.** The affinity is **cosine similarity**, which is scale-invariant by
construction and therefore immune to depth-dependent norm growth. This is the conventional
choice in this literature.

- **Primary:** cosine affinity.
- **Appendix robustness check:** per-layer standardization (z-score within layer), reported
  alongside. If the two disagree, that disagreement is reported, not resolved in favour of
  whichever supports H3.
- Raw-vector affinity is NOT used at all; it confounds depth with geometry.

**Layer-pair index convention (was undefined).** Pair *p* = (output of block *p*, output of
block *p+1*), for p in 0 .. L-2. With L = 48 for gemma-4-12B-it, that is **47 layer-pairs**,
which is the n for H3's Wilcoxon. This convention is stated in `src/ling/hooks.py`'s
convention block alongside the existing layer convention, because an off-by-one here is
exactly the §A.11 item-2 failure mode applied one level up.

---

## Amendment 4 — `formal` is GIVENS + GOAL, with no derivation

**Timestamp:** 2026-09-06T21:30 UTC (git commit `6c78efe` — authoritative)
**Status:** corpus definition change, made before any activation exists. H3's endpoint and
direction unchanged.

**The defect.** As first generated, the `formal` condition emitted the full program
including the answer expression (`answer = int((k*y)/(x*12)*100)`). That is not a
re-notation of the problem — it is the solution. Under it:

- H3 would compare a prompt containing the derivation against a prompt containing a word
  problem. Any lambda2 difference is then trivially attributable to one prompt containing
  the answer, not to formal notation.
- Gate G2 would be near-ceiling for a reason unrelated to notation. "Formal beats NL" would
  mean "we handed the model the program."

DESIGN.md §B.8 is explicit that our formal condition is **input notation with no execution**.
Emitting the derivation contradicts that.

**Decision.** `formal` renders **givens and goal only**, in formal notation, with NO
derivation and no answer expression. The model must still do the reasoning; only the
notation changes. That is the intervention H3 is about.

**What this costs and why it costs nothing.** The build-time check "the formal program
executes to ground truth" is lost, because givens-plus-goal is not executable. This is not a
real loss: ground truth comes from the TEMPLATE's `#answer:` expression (Apple's verbatim)
evaluated on our sampled values, and always did. The formal program was never the source of
truth, only a redundant confirmation of it.

**Retained check.** The verifier still confirms every quantity value in `formal` matches the
sampled value it claims, and that the goal names the quantity the template asks for.

---

## Recorded corpus caveats (for Methods, not amendments)

1. **The contamination claim must be weaker than DESIGN.md §A.5 implies.** GSM-Symbolic
   templates derive from GSM8K problems and have been public since Dec 2024. The
   *instantiations* are fresh; the surface frames are not uncontaminated. The defensible
   sentence is "fresh instantiations of published templates," never "uncontaminated."
2. **G3's "different model than the generator" is trivially satisfied and should be stated
   as such.** No model generated `nl_base`, `formal`, or `nl_break` — they are emitted from
   templates. Claiming independent verification we did not have to earn would be misleading.
   Only `nl_para` is model-written, and it is the item under human review.
3. **The DSL interpreter and lexical pools are OURS, not Apple's.** Apple published the
   templates but never the generator or the word pools. Numeric pools, `#conditions:` and
   `#answer:` expressions are verbatim; surface noun pools are a reconstruction. This must
   be stated.
4. **10 of 100 problems have only 2 quantities** (3 aligned positions including FINAL).
   Amendment 2 assumed 3-5. Pooled lambda2 handles the low tail, but it is real and the
   distribution should be reported.

### Amendment 4, clarification — what `formal` actually is

**Timestamp:** 2026-09-06T22:45 UTC (git commit `cb221bf` — authoritative)

The implemented `formal` condition is:

```
# GIVEN
n = 840
frac_1 = Fraction(1, 5)

# WHERE
# A juggler can juggle n balls. frac_1 of the balls are tennis balls, ...

# FIND
# How many orange tennis balls are there?
answer = ?
```

The situation description in the WHERE block **remains English**, with quantities
replaced by their symbolic identifiers. Stripping it to bare bindings
(`x=200, k=8, y=6, solve`) makes the problem literally unanswerable, which would drive the
formal arm's accuracy to ~0 and make gate G2 meaningless — it would measure unanswerability,
not notation.

**The intervention is therefore: quantities lifted to symbolic bindings, plus an explicit
given/goal frame, with the situation retained in prose.** It is NOT full formalization.

**This must be stated in the paper exactly that way.** Describing it as a "Python rewrite"
or "formal rewriting" without qualification would overclaim the manipulation. H3's result —
in either direction — is a claim about *this* intervention, and a reviewer comparing the
figure caption against the corpus example will check.

Enforced at build: every non-comment line is `<quantity> = <bare literal>` matching the
sampled value; exactly one `answer = ?`; no compound answer expression; no scratch variable
bound (three templates whose `#answer:` names a solved-for intermediate would otherwise have
leaked the answer).

---

## Amendment 5 — kernel and bandwidth pre-committed

**Timestamp:** 2026-09-06T22:54 UTC (git commit `4f23984` — authoritative)
**Status:** operationalization, pre-data. No endpoint or direction changes.

**Kernel.** Amendment 3 fixed the *representation* (cosine, scale-invariant) but not the
kernel. A literal cosine gram matrix `A = X̂X̂ᵀ` makes normalized spectral clustering a
LINEAR method, which cannot recover non-convex structure — the very capability that
justifies the front end. The affinity is therefore a **Gaussian kernel on cosine distance**,
strictly monotone in `1 - cos` and so equally scale-invariant. `kernel="cosine_linear"`
exists for robustness only.

**Bandwidth is the one free parameter, so it is pre-committed here.** `sigma` = median over
rows of the k-th nearest reference distance, computed on the **pooled** set across all
conditions. A per-condition bandwidth is as forbidden as per-condition landmarks (§A.11
item 6) — it would build a between-condition difference in before measurement.

`knn_sigma` is fixed at its default for the primary analysis. A second value is run as a
robustness check and reported. **The primary endpoint is not selected across bandwidths.**
A bandwidth wider than a between-structure gap merges that structure, so this parameter can
move lambda2; choosing it after seeing lambda2 would be a researcher degree of freedom on
the primary endpoint.

**Reported per layer** in the assignments file: `kernel`, `sigma`, `laplacian` (normalized
symmetric), and the eigengap.

---

## Amendment 6 — two style controls; style-normalized displacement

**Timestamp:** 2026-09-06T23:00 UTC (git commit `8fd8b4d` — authoritative)
**Status:** pre-data. Adds conditions and a SECONDARY endpoint. **H2's and H3's primary
endpoints are UNCHANGED.**

**The problem being fixed.** Measured on the built corpus, the surface-dissimilarity control
correlates r = 0.883 (lexical) / 0.868 (syntactic) with answer-survival status, and the
ranges are effectively disjoint (`nl_break` lexical distance in [0.0017, 0.0222]; every
`nl_para`/`formal` pair >= 0.1227). Cause: `nl_break` is the ONLY answer-breaking condition
AND is a one-character edit by construction, so "surface dissimilarity" is close to a perfect
proxy for "answer-preserving." H2's residual coefficient is therefore weakly identified —
and that residual is precisely H2's contribution over Han et al.

**Two new conditions, both answer-preserving, both minimal edits.**

| condition | edit | property | floor it gives |
|---|---|---|---|
| `nl_minimal_orth` | one quantity re-rendered as a number word (`24` -> `twenty-four`) | ON-anchor; edit site matched to `nl_break` | likely too HIGH (digits and number-words may be genuinely differently encoded) |
| `nl_minimal_lex` | one proper name swapped | semantically null, but OFF-anchor (names are not anchors) | likely too LOW |

Together they BRACKET the true surface-perturbation floor. Neither alone is trustworthy, and
this is stated as the reason both exist rather than presented as belt-and-braces.

**Why this identifies H2.** `nl_minimal_orth` vs `nl_break` is a MATCHED-MAGNITUDE,
MATCHED-SITE contrast: both change one token at a quantity anchor; they differ only in
whether the value, and hence the answer, changes. That contrast is not confounded with
surface dissimilarity the way the current design is.

**New SECONDARY endpoint (pre-registered, does NOT replace H2's primary):**
hazard ratio for fork layer, `nl_break` vs `nl_minimal_orth`, in the same cloglog model.
**Direction pre-committed: `nl_break` forks EARLIER** (higher hazard) than a
matched-magnitude style edit.

**Style-normalized displacement (SND), INTERPRETIVE secondary:**

```
SND(c, m) = [ m(c) - m(nl_base) ] / [ m(nl_minimal, m) - m(nl_base) ]
```

Reported for lambda2 and path entropy, against BOTH floors, as a range.
**Direction pre-committed: |SND(formal)| > 1 for lambda2 against the UPPER bracket**
(`nl_minimal_orth`) — formal displaces the geometry more than a semantically-null edit of
the same kind does.

**Guard against the ratio artifact.** The raw difference with a problem-level bootstrap CI
is PRIMARY. SND is reported ONLY where the denominator's bootstrap CI excludes zero; where
it does not, SND is reported as undefined, never as a large number. A ratio with a
near-zero denominator produces a spectacular artifact and this is the obvious way this
metric fails.

**Total corpus: 100 problems x 6 conditions = 600 prompts.**

### Amendment 6, correction — the contrast is matched-SITE, not matched-magnitude

**Timestamp:** 2026-09-06T23:09 UTC (git commit `f234dd6` — authoritative)

Amendment 6 as written called `nl_minimal_orth` vs `nl_break` a "matched-magnitude,
matched-site" contrast. **The matched-magnitude half is wrong and is corrected here before
any data exists.** Measured on the built corpus:

| condition | lex_edit mean | median | range |
|---|---|---|---|
| `nl_break` | 0.0050 | 0.0046 | [0.0017, 0.0222] |
| `nl_minimal_orth` | 0.0486 | 0.0354 | [0.0066, 0.1694] |
| `nl_minimal_lex` | 0.0240 | 0.0175 | [0.0041, 0.0857] |

Digits-to-words is inherently a multi-character rewrite: median per-problem char-edit ratio
orth:break is **8.0x** (9 chars vs 1). Worst cases are large values —
`7775` -> `"seven thousand seven hundred seventy five"` is 41 characters.

**Corrected description, to be used verbatim in the paper:**
`nl_minimal_orth` is matched to `nl_break` on **edit SITE** (95/99 edit the exact quantity
slot `nl_break` perturbs) but is roughly an order of magnitude LARGER in edit size.
`nl_minimal_lex` is the closer-magnitude control (~5x) but is off-anchor.

**This strengthens rather than weakens the bracket argument.** Neither control is matched on
both site and magnitude; each is matched on one and off on the other, in opposite
directions. That is precisely why BOTH are reported and why the style floor is presented as
a RANGE. Do not write "matched magnitude" anywhere.

**Pre-committed direction is unchanged:** `nl_break` forks EARLIER than `nl_minimal_orth`.
Note this is now a CONSERVATIVE prediction — `nl_minimal_orth` is the larger surface edit, so
surface-driven forking would push it to fork earlier, against the predicted direction.

**Availability (reported, not hidden):** `nl_minimal_orth` 99/100 (1 problem has no
digit-rendered integer). `nl_minimal_lex` 70/100 (**30 problems contain no proper name**).
Unavailable cells are recorded as genuine holes, never invented, and must be excluded
upstream of the survival model rather than entering it as missing covariate rows.

---

## POST-DATA RECORD — everything below this line was written AFTER seeing results

**This boundary is the point of the document.** Amendments 1-6 were all made before any
model was loaded. What follows was not. Nothing below can become a primary endpoint, and
nothing below revises a pre-committed direction.

### Observation 1 — the pre-registered analysis returned a degenerate measurement

Run `main`, executed exactly as pre-registered (eigengap k on nl_base, cosine affinity,
pooled lambda2, cloglog survival):

- **H3 primary:** mean paired d(lambda2) = **+0.3602** over 47 layer-pairs, bootstrap CI
  **[0.211, 0.385]**, Wilcoxon(less) p = 1. The pre-committed direction was NEGATIVE.
  The observed effect is confidently POSITIVE — not merely null.
- **H2 primary:** 469 pairs, **25 fork events, 444 right-censored (94.7%)**. The reported
  coefficient (+16.37, se 0.64, p = 1.5e-144) is **perfect separation on 25 events, not a
  finding.** It must not be reported as an effect.
- **H2 pre-registered secondary:** UNFITTABLE — 0 fork events; every pair right-censored.

**Diagnosed mechanism.** Eigengap selected **k = 2 on 40 of 48 layers** (k=3 on 7, k=20 on
the one flat-spectrum layer). At k=2, **457 of 569 cells occupy BOTH clusters**, so two
conditions' cluster sets can essentially never be disjoint. Fork therefore almost never
fires, which produces the 94.7% censoring directly. It also floors lambda2's resolution: a
k=2 transition graph has four nodes and lambda2 can take only a handful of values, and path
entropy is capped at log 2.

This is not an implementation bug. It is the documented failure mode of the eigengap
heuristic on a normalized Laplacian, whose largest spectral gap typically sits immediately
after lambda_1. The pre-registered procedure ran correctly and returned k=2.

### Consequence for what may be claimed

**The pre-registered H3 result stands as reported: the effect is in the opposite direction
to the pre-commitment.** It is NOT reframed as a null, and it is NOT explained away.

**H2's primary is reported as NOT IDENTIFIED** (25 events, 94.7% censoring, separation),
not as a significant coefficient. Reporting p = 1.5e-144 as an effect would be false.

### Exploratory sensitivity run (labeled BEFORE its results were seen)

Run `fixedk8`: identical pipeline, k forced to 8 at every layer, to test whether the k=2
collapse drives the degeneracy.

**This run is EXPLORATORY and can never be promoted to primary**, whatever it shows. If it
reverses H3's sign, that is a finding about the sensitivity of the construct to k, reported
as such — it does not rescue the pre-registered prediction. If it identifies H2, the
identified coefficient is exploratory and is reported alongside, never instead of, the
non-identified primary.

The honest framing either way: **the pre-registered operationalization of "routing
modularity" did not survive contact with the data, and the metric's resolution depends
critically on a k-selection step the design treated as incidental.** That is a reportable
negative result about the construct, and the pre-registration is what makes it reportable
rather than quietly re-specifiable.

### Observation 2 — EXPLORATORY positive finding: a one-layer transition at layer 12

**Post-data, exploratory, NOT pre-registered.** No primary endpoint. Reported as exploratory
in any write-up, with this label attached.

AMI between the pooled k=2 partition and the FINAL-vs-quantity anchor distinction:

| layers | mean AMI |
|---|---|
| 1-11 | **0.881** (peak **0.9795** at layer 5) |
| 12-47 | **0.051** |

Ratio 17.4x. The collapse happens **in a single layer**: 0.880 at layer 11, 0.005 at
layer 12. It is not a gradual decay.

**The eigengap does NOT collapse at layer 12** (0.204 at L11, 0.299 at L12, 0.592 at L13).
The partition remains equally strong; it stops tracking token role and begins tracking
something else. The correct claim is therefore *the dominant axis changes what it encodes*,
NOT *structure disappears*. What it encodes after layer 12 is unresolved and is the obvious
follow-up.

**Negative control passed:** AMI(cluster, condition) is ~0.00-0.03 at every layer, so the
partition is not sorting states by which realization they came from.

**H1 does NOT hold, even descriptively.** Spectral vs K-means AMI against dependency labels
is mixed across layers (L2: 0.235 vs 0.328 kmeans; L18: 0.246 vs 0.036 spectral; L42: 0.023
vs 0.059 kmeans). There is no reliable spectral advantage. H1 is reported as unsupported,
not quietly omitted.

**Methodological irony worth stating plainly:** the k=2 degeneracy that made lambda2 and
fork layer unusable is exactly what makes this finding legible. Forcing the partition onto a
single dominant axis destroys cross-condition comparison but is a good instrument for asking
what that axis encodes. The failure and the finding have the same cause.

### Observation 3 — CORRECTION to Observation 2: the transition is largely DEGENERACY, not linguistics

**Post-data, exploratory.** This corrects the reading in Observation 2, which was written
before the finer curves and the cluster-balance check were run. Observation 2's AMI numbers
stand; its interpretation does not.

**Finer linguistic structure is absent at every layer.** AMI between the partition and
dependency role, computed on quantity anchors ONLY (excluding FINAL, whose `punct` label is
essentially the token-role split itself):

| | layers 1-11 | layers 12-47 |
|---|---|---|
| token role (FINAL vs quantity) | 0.881 | 0.051 |
| **dependency role, FINAL excluded** | **0.004** | **0.008** |
| anchor identity, FINAL excluded | 0.002 | 0.030 |

Flat at zero throughout. There is no hierarchy and no handoff from morphosyntax to syntax.
The dominant axis tracks the single coarsest distinction available and nothing below it.
`dep(all)` appeared to carry signal only because it is contaminated by `punct` vs `nummod`.

**Cluster balance shows much of the post-L12 partition is DEGENERATE:**

| layers | cluster sizes | reading |
|---|---|---|
| 1-11 | ~[2280, 570] | balanced, ~20/80, token-role aligned |
| 13-15 | **[2847, 5]** | five outlier states isolated |
| 32, 34, 37-39 | **[2851, 1]** | a SINGLE token alone in its own cluster |
| 17-31, 40-47 | ~[1900, 950] | balanced, but AMI ~0.05 against every label we have |

**10 of 48 layers put under 1% of states in a cluster.** This is the massive-activation
signature: a few states of enormous norm, which spectral clustering at k=2 correctly
isolates as their own component.

**One mechanism explains every pre-registered failure:** lambda2 quantized to {0,1,2};
lambda2 reaching 2.0 (the normalized-Laplacian ceiling, produced by a near-isolated node);
fork layer never firing (everything else in one giant cluster, so cluster sets can never be
disjoint); and the AMI collapse (a singleton cluster is uninformative about any label).

**What may NOT be claimed:** that layer 12 is where linguistic organization changes. The
norm check has not been run. Until it is, the honest statement is that the dominant
bipartition after L12 is substantially driven by outlier states, not by linguistic
structure.

**What remains genuinely unexplained:** layers 17-31 and 40-47 are BALANCED (~30/70) with
AMI ~0.05 against token role, dependency role, and anchor identity alike. Those partitions
are not degenerate and are not linguistic by any label available here. That is an open
question, stated as open.

**Semantic-frame AMI was NOT computed** — the corpus carries no FrameNet/PropBank labels
(DESIGN.md §B.3 defers this to the archival version). Its absence is a limitation, not a
null result, and must not be reported as one.
