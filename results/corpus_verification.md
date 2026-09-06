# Corpus verification — `data/corpus/problems.jsonl`

Mechanical checks only (execution + exact string ops). No semantic judgment made;
paraphrase adequacy is left to `results/human_review_sheet.md` for human review.

**Scope note on Amendment 4:** the legacy check "`formal` executes to ground truth" is
obsolete (Amendment 4 removed the answer expression from `formal`; it is givens + goal
with `answer = ?`, deliberately not executable) and was **not run**. In its place: every
`<quantity> = <literal>` binding in `formal` was checked against the sampled value,
exactly one `answer = ?` line was required, and no compound answer expression was
permitted.

Corpus: 100 problems × 4 conditions (`nl_base`, `nl_para`, `formal`, `nl_break`), single
seed `20260906`.

## Pass/fail table

| # | Check | Method | Result |
|---|---|---|---|
| 1 | Template arithmetic → recorded `ground_truth` | 93/100: re-executed the vendored template's `#answer:` expression directly on the quantities stored in each record. 7/100 (`gsms_0009`, `gsms_0011`, `gsms_0021`, `gsms_0022`, `gsms_0026`, `gsms_0040`, `gsms_0050`) reference a solved-for scratch variable not stored in the record — verified instead by a full deterministic rebuild from the recorded seed (`ling.corpus.build_corpus`, seed 20260906) and comparing sampled quantities, `ground_truth`, and all 4 realizations' text/answer byte-for-byte against the shipped file. | **PASS** — 0 mismatches, 0/100 build failures on rebuild |
| 2 | Anchor spans resolve: `text[start:end] == literal`, all 4 conditions × 100 problems | Direct string check against `data/corpus/problems.jsonl` only (no rebuild dependency): `nl_base`/`nl_para` checked against `quantities[].surface`, `formal` against `quantities[].formal_literal`, `nl_break` against `perturbation.to_surface` for the perturbed quantity and `quantities[].surface` for the rest | **PASS** — 0 span mismatches across 400 realizations |
| 3 | §A.6 invariant: every quantity key present in all 4 conditions' anchors | Set-equality of `anchors.keys() - {FINAL}` vs. `{q.name for q in quantities}`, per realization | **PASS** — 0 violations, 0 problems dropped |
| 4 | `nl_break` recomputed answer differs from `nl_base`'s | `realizations.nl_break.answer != realizations.nl_base.answer`, all 100 | **PASS** — 0 non-breaking items (see itemized list below — empty) |
| 5 | `nl_break` surface-similarity vs `nl_base` | Normalized Levenshtein edit distance and token-Jaccard, char-level and whitespace/punctuation-tokenized, all 100 pairs | **PASS** — distribution below; 0 outliers flagged |
| 6 | Quantity literals appear in each realization's text | Same pass as check 2 (the literal comparison *is* the appearance check) | **PASS** |
| 7 | No duplicate problems; no `nl_base` text verbatim in another condition | Exact string dedup: `nl_base` text across all 100 problems; `nl_base` vs. each of `nl_para`/`formal`/`nl_break` within a problem; `nl_base` of problem P vs. every condition's text of every other problem Q | **PASS** — 0 duplicates |
| 8 | `FINAL` sentinel present in all realizations; no realization ends in whitespace | `anchors["FINAL"] == [-1, -1]` present in all 400 realizations; `text == text.rstrip()` | **PASS** |
| Amendment 4 | `formal`: every non-comment line is `<quantity> = <literal>` matching sampled value; exactly one `answer = ?`; no compound expression | Line-by-line parse of the shipped `formal` text against `quantities[].value`, independent of the build code | **PASS** — 0 violations |

**Drop rate: 0/100 (0.0%).** No problem violates the §A.6 invariant or any other hard
check; all 100 problems are in the human review sheet.

## `nl_break` surface-distance distribution (n=100)

Normalized character edit distance (Levenshtein / len(`nl_base`)):

| min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|
| 0.0017 | 0.0034 | 0.0046 | 0.0060 | 0.0222 | 0.0050 |

Token Jaccard (`nl_break` vs `nl_base`, lowercase alnum tokens):

| min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|
| 0.875 | 0.920 | 0.938 | 0.955 | 0.980 | 0.937 |

**Outliers:** none. Flag thresholds used (normalized edit distance > 0.15, or token
Jaccard < 0.5 — i.e. an item that is far from `nl_base` on both axes, which would make
H2 trivially true) caught 0/100 problems. Every `nl_break` item is a small, single-token
numeric substitution, consistent with the corpus's own perturbation-search design
(median normalized edit distance 0.0046 — a handful of changed characters in prompts of
100–250 characters).

## Itemized failures

None. All 8 checks plus the Amendment-4 `formal` check pass on all 100 problems / 400
realizations.

## What was not (re-)checked

- The legacy "`formal` executes to ground truth" check — obsolete per Amendment 4, see
  above.
- Whether `nl_para` preserves meaning — mechanical checks confirm it is anchored, its
  quantity literals match `nl_base`'s, its recorded answer equals `ground_truth`, and it
  is not textually identical to `nl_base`; whether the paraphrase is *actually*
  equivalent is a human judgment, made in `results/human_review_sheet.md`, not here.
