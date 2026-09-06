# Cluster assignments — the pod's only output

Per DESIGN.md §A.1: extract -> cluster -> write assignments -> discard activations. This
file is the ONLY artifact the GPU pod produces. Everything downstream (lambda2, path
entropy, fork layer, all statistics, all figures) runs on a laptop against it.

Size: 400 prompts x 48 layers x ~4-9 aligned positions. A few hundred KB.

## `results/<run_id>/assignments.jsonl` — one record per layer

```json
{
  "layer": 12,
  "k": 11,
  "eigengap": 0.0412,
  "eigengap_flat": false,
  "affinity": "cosine",
  "rows": [
    {"problem_id": "gsms_0007", "condition": "nl_base", "anchor": "num_apples", "cluster": 3},
    {"problem_id": "gsms_0007", "condition": "nl_base", "anchor": "FINAL",      "cluster": 7}
  ]
}
```

## Invariants the analysis code may assume (and must assert)

- `layer` indexes the **output of block `layer`** (src/ling/hooks.py convention).
- **Layer-pair `p` = (output of block `p`, output of block `p+1`)**, p in 0..L-2.
  L=48 -> **47 pairs** -> n=47 for H3's Wilcoxon (DEVIATIONS Amendment 3).
- `k` is selected ONCE per layer on `nl_base` by eigengap and reused across all four
  conditions at that layer (DESIGN.md §A.6, prereg decision 1). The same `k` therefore
  appears in every row of a layer regardless of condition. **Assert this**; a per-condition
  `k` invalidates every cross-condition lambda2 comparison.
- `affinity` is `"cosine"` (Amendment 3). The appendix robustness run writes
  `"per_layer_standardized"`. Never `"raw"`.
- Row order is stable across layers: row `j` of layer `i` and row `j` of layer `i+1` are the
  same (problem, condition, anchor). This is what makes a layer-pair transition graph
  well-defined. **Assert it rather than assume it.**
- Every `(problem_id, condition)` pair has the SAME set of anchors across all four
  conditions (§A.6 invariant, enforced at extraction).

## What the analysis computes

- **lambda2** — second-smallest eigenvalue of the Laplacian of the **pooled** per-condition
  cluster-transition graph for layer-pair p. Pooled over all surviving problems
  (DEVIATIONS Amendment 2). Skip the trivial lambda1 = 0; state normalized vs combinatorial.
- **Path entropy** — over cluster-transition sequences, normalized to [0, log k].
- **Fork layer** — first layer at which a pair's cluster assignments become disjoint.
  Right-censored when it never forks: neither dropped nor assigned a fake layer.

## Resampling

Every bootstrap and every permutation null resamples **PROBLEMS**, rebuilds the pooled graph
from the resampled problems, and recomputes the statistic. Never resample token states
(prereg decision 3). Row provenance is carried by `problem_id` for exactly this reason.
