# A Minimal-Pair Test Shows a Spectral Fork Diagnostic Tracks Prompt Surface Form

Kundana Kommini (Substrate Labs), Kevin Zhou, Johnathan Chen
Poster, Workshop on Linguistic Principles for Foundation Models (LP4FM), NeurIPS 2026

## Summary

Prompt representation misalignment across layers is often taken as evidence for altered computation within a model. Here we examine one necessary condition for such claims: whether these misalignments track minimal interventions that change the answer, as opposed to surface altering (but not answer-altering) rewrites. To evaluate this we generate 100 grade school mathematics questions alongside five systematically controlled variants to each, including a single token answer altering edit, and multiple answer preserving rewrites. We find that in Gemma-4-12B-it, a spectral cluster-trajectory fork diagnostic never fires on the single-token answer-altering edit (0/100 under both cluster-count settings we ran), even though that edit never elicits the original answer, while it fires on the formal answer-preserving rewrite in 100/100 items at forced k = 8 and 19/100 at the preregistered eigengap k. The formal rewrite preserves the answer but drops accuracy from 70% to 56% (exact McNemar, p = .007), while the answer-altering edit leaves accuracy intact. The controlled comparison holds surface change fixed: a proper-name substitution, identical to the answer-altering edit on every surface measure we compute, forks 4/70 while the answer-altering edit forks 0/100 (Fisher exact p = .027; k = 8). Across all 469 pairs, surface dissimilarity predicts forked status (McFadden R² = 0.397; k = 8), and fork rate is monotone in token-count change across all five conditions. The answer-breaking coefficient is unidentifiable, since that arm has no fork events. While this study leaves open what aspect of surface altering edits is driving forked status, it suggests that when there's a simple answer altering control this is not a robust measure of answer relevant computation.

## What's here

- `src/ling/` — DSL interpreter and lexical pools (`templates.py`, `corpus.py`), residual capture and anchor alignment (`hooks.py`), spectral clustering (`cluster.py`), statistics (`stats.py`)
- `scripts/` — the pipeline, numbered in run order (`01_` … `09_`)
- `data/templates/` — the 100 GSM-Symbolic templates, vendored at commit `b6a1625`
- `data/corpus/` — `problems.jsonl` (100 problems × 6 conditions: `nl_base` plus five variants), dissimilarity covariates, paraphrases
- `results/main/`, `results/fixedk8/` — per-layer cluster assignments (eigengap k and forced k=8); every number in the paper is recomputable from these
- `results/analysis.json`, `results/analysis_fixedk8.json` — summary statistics for each run; `results/g2_behavioral.json` — per-condition accuracy
- `prereg/` — pre-registration record (sealed endpoints, `SEAL.txt`) and the pre- and post-data deviation log
- `paper/figs/` — figures (`paper/figs/clean/` is the final set)
- `FINDINGS.md` — working notes (predates the final write-up; the paper supersedes it where they differ)

## Reproduce

Model: `google/gemma-4-12B-it`, bf16, eager attention, thinking disabled. One A40; ~1h45 wall-clock.
The checkpoint is gated: accept the licence on Hugging Face and `export HF_TOKEN=...` first.

```bash
bash scripts/pod_bootstrap.sh                 # deps + determinism env, then runs the G1 determinism gate
python scripts/02_generate_corpus.py --verify-reproducible   # optional: rebuild data/corpus/ byte-for-byte
python scripts/02b_behavioral_g2.py           # accuracy per condition -> results/g2_behavioral.json
python scripts/03_extract_and_cluster.py --run-id main                           # eigengap k
python scripts/03_extract_and_cluster.py --run-id fixedk8 --k-min 8 --k-max 8    # forced k=8
python scripts/04_analyze.py --run-id main
python scripts/04_analyze.py --run-id fixedk8 --out results/analysis_fixedk8.json
python scripts/09_clean_figures.py            # paper/figs/clean/
```

Steps 04 and 09 need no GPU and run from the committed assignments.

## Citation

```bibtex
@inproceedings{kommini2026minimalpair,
  title     = {A Minimal-Pair Test Shows a Spectral Fork Diagnostic Tracks Prompt Surface Form},
  author    = {Kommini, Kundana and Zhou, Kevin and Chen, Johnathan},
  booktitle = {Workshop on Linguistic Principles for Foundation Models (LP4FM), NeurIPS},
  year      = {2026}
}
```
