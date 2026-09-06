# Corpus record schema — 100 problems x 6 realizations

The six conditions. `DESIGN.md` does not name them; this is the assignment that makes H1, H2
and H3 measurable from one corpus. The last two are added by
`prereg/DEVIATIONS.md` **Amendment 6**.

| key | condition | answer-preserving? | edit site | serves |
|---|---|---|---|---|
| `nl_base`  | GSM-Symbolic NL instantiation | yes (reference) | — | H1 (k selection, AMI), H3 baseline |
| `nl_para`  | answer-preserving paraphrase  | yes | whole text | H2 (survival=1), surface-dissimilarity control |
| `formal`   | givens + goal rewrite, NOT executed | yes | whole text | H3 (lambda2 decrease) |
| `nl_break` | minimal surface-similar perturbation that CHANGES the answer | **no** | AT a quantity anchor | H2 (survival=0) |
| `nl_minimal_orth` | one quantity re-rendered as a number word, same value | yes | AT a quantity anchor — the **same slot** `nl_break` perturbs | H2 identification; SND upper bracket |
| `nl_minimal_lex`  | one proper name swapped for another from the same pool | yes | OFF-anchor (names are not anchors) | SND lower bracket |

## Why `nl_minimal_orth` and `nl_minimal_lex` exist

Measured on the four-condition corpus, the surface-dissimilarity control correlated
r = 0.883 (lexical) / 0.868 (syntactic) with answer-survival status, with effectively
disjoint ranges. Cause: `nl_break` was the ONLY answer-breaking condition **and** the only
one-character edit, so "surface dissimilarity" was very nearly a perfect proxy for
"answer-preserving", and H2's residual coefficient — the part that is ours rather than Han
et al.'s — was weakly identified.

Both new conditions are **answer-preserving minimal edits**, so the two factors stop moving
together. Neither is trustworthy alone (`nl_minimal_orth` likely reads too HIGH, because
digits and number words may be genuinely differently encoded; `nl_minimal_lex` too LOW,
because it never touches an anchor). They are reported as a **bracket**, and Amendment 6
says so rather than presenting them as belt-and-braces.

`nl_minimal_orth` vs `nl_break` is the matched-magnitude, matched-site contrast: both change
one token at a quantity anchor, and they differ only in whether the value — and hence the
answer — changes.

### Which quantity slot `nl_minimal_orth` targets

`nl_break`'s perturbed quantity, wherever the instance allows it. The target is chosen by
`ling.corpus.orth_target_for(inst, preferred=<nl_break's quantity>)`, which tries that slot
FIRST and falls back to the remaining quantities in first-appearance order. A slot is
eligible only if its value is a non-boolean integer in `[0, 999999]` **currently surfaced as
digits** — a value drawn from the templates' own `numbers_within(...)` pool is already a
number word, so re-wording it would be a no-op, and a `Fraction` has no cardinal form.

Every record carries `minimal_edits.nl_minimal_orth.matches_nl_break_site`, so the matched-site
claim is verifiable per problem rather than assumed. The number word uses the templates' own
convention — space-separated, no hyphen, as `numbers_within(31, 35)` emits `"thirty two"` —
so a worded quantity in this condition is indistinguishable from one the template produced.

### When a condition is unavailable

A template with no digit-rendered integer quantity, or with no proper name, records that
condition as

```json
{"text": "", "anchors": {}, "answer": <ground truth>, "status": "unavailable", "note": "..."}
```

— the same honest hole `nl_para` leaves while pending. Nothing is invented; the counts are in
`corpus_summary.json` under `minimal_edits`.

## Record

```json
{
  "problem_id": "gsms_0007",
  "template_id": "<GSM-Symbolic template name>",
  "seed": 20260906,
  "ground_truth": 42,
  "quantities": [{"name": "num_apples", "value": 5}],
  "realizations": {
    "nl_base": {
      "text": "...",
      "anchors": {"num_apples": [17, 18], "FINAL": [-1, -1]}
    }
  },
  "minimal_edits": {
    "nl_minimal_orth": {"available": true, "variable": "num_apples",
                        "from_surface": "24", "to_surface": "twenty four",
                        "on_anchor": true, "matches_nl_break_site": true},
    "nl_minimal_lex":  {"available": true, "variable": "name",
                        "from_surface": "Amara", "to_surface": "Omar",
                        "on_anchor": false}
  }
}
```

`anchors` maps each quantity name to a **character span** in that realization's `text`,
marking where the quantity is first introduced. Spans are converted to token indices at
extraction time via the tokenizer's offset mapping, never by re-tokenizing a substring.
A span always covers the whole surface form, including the multi-token `"twenty four"` that
`nl_minimal_orth` emits.

`FINAL` is the sentinel for the last token of the prompt and is always present.

**Invariant (enforced, not assumed): every quantity key appears in the `anchors` of every
realization that has text.** Per DESIGN.md §A.6, only positions present in all conditions are
used, so that a lambda2 difference between a 60-token NL prompt and a 28-token Python prompt
is not confounded with length. Any problem violating this is dropped and counted in the drop
rate. A realization with `status: "unavailable"` (or `"pending"`) has no text and no anchors
at all, and is excluded from the aligned-position set rather than counted as a violation.
