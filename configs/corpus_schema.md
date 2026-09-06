# Corpus record schema — 100 problems x 4 realizations

The four conditions. `DESIGN.md` does not name them; this is the assignment that makes both
H2 and H3 measurable from one corpus.

| key | condition | answer-preserving? | serves |
|---|---|---|---|
| `nl_base`  | GSM-Symbolic NL instantiation | yes (reference) | H1 (k selection, AMI), H3 baseline |
| `nl_para`  | answer-preserving paraphrase  | yes | H2 (survival=1), surface-dissimilarity control |
| `formal`   | Python/formal rewrite, NOT executed | yes | H3 (lambda2 decrease) |
| `nl_break` | minimal surface-similar perturbation that CHANGES the answer | no | H2 (survival=0) |

`nl_break` is the H2 workhorse: it must be surface-*similar* to `nl_base` but answer-
breaking, so that the surface-dissimilarity covariates cannot explain the fork. A perturbation
that is both semantically and lexically far away makes H2 trivially true and uninteresting.

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
  }
}
```

`anchors` maps each quantity name to a **character span** in that realization's `text`,
marking where the quantity is first introduced. Spans are converted to token indices at
extraction time via the tokenizer's offset mapping, never by re-tokenizing a substring.

`FINAL` is the sentinel for the last token of the prompt and is always present.

**Invariant (enforced, not assumed): every quantity key appears in the `anchors` of all four
realizations.** Per DESIGN.md §A.6, only positions present in all four are used, so that a
lambda2 difference between a 60-token NL prompt and a 28-token Python prompt is not
confounded with length. Any problem violating this is dropped and counted in the drop rate.
