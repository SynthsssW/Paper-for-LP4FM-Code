Prompt representation misalignment across layers is often taken as evidence for altered
computation within a model. Here we examine one necessary condition for such claims:
whether these misalignments track minimal interventions that change the answer, as opposed
to surface altering (but not answer-altering) rewrites. To evaluate this we generate 100
grade school mathematics questions alongside five systematically controlled variants to
each, including a single token answer altering edit, and multiple answer preserving
rewrites. We find that in Gemma-4-12B-it, a spectral cluster-trajectory fork diagnostic
tracks all formal answer-preserving rewrites (100/100) but none of the single-token
answer-altering edits (0/100), which never elicited the original answer. The formal rewrite
preserves the answer but drops accuracy from 70% to 56% (*p* = .007), while the
answer-altering edit leaves accuracy intact. The controlled comparison holds surface change
fixed: a proper-name substitution, identical to the answer-altering edit on every surface
measure we compute, forks 4/70 while the answer-altering edit forks 0/100 (Fisher exact
*p* = .027, one-sided). Across all 469 pairs, surface dissimilarity predicts forked status
(McFadden *R*² = 0.397), and fork rate is monotone in token-count change across all five
conditions. The answer-breaking coefficient is unidentifiable, since that arm has no fork
events. While this study leaves open what aspect of surface altering edits is driving forked
status, it suggests that when there's a simple answer altering control this is not a robust
measure of answer relevant computation.
