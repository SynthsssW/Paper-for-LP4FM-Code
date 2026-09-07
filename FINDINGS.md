# What we found — plain English

*Written 2026-09-07, ~00:15 EDT, after the full run. Every number here is in
`results/` and reproducible from this repo.*

---

## The one-sentence version

There is a popular way to find where a language model "goes wrong" inside — you feed it two
versions of a prompt and look for the layer where its internal representations diverge. We
built a test that separates two things that measure could be tracking, and it turns out to
be tracking **how different the text looks**, not **whether the answer changed**.

## The evidence, in three numbers

We took 100 grade-school math problems and made six versions of each.

One version, `nl_break`, changes a single digit — so the correct answer is now different.

| | |
|---|---|
| Times the model's answer actually changed because of that edit | **100 / 100** |
| Times the divergence measure detected anything | **0 / 100** |
| Times it fired on a *style rewrite* that did NOT change the answer | **100 / 100** |

The model isn't ignoring the edit. It follows it every single time — it never once gave the
old answer, and it computes the new one correctly 67% of the time, which is the same rate it
solves the original problems (70%). The change is real and it reaches the output. The measure
just doesn't see it.

Meanwhile, rewriting the same problem in formal notation — same numbers, same answer — makes
the measure fire on every single problem, at layer 3 of 48.

## Why this hasn't been shown before

You need a control that nobody had built: an edit that **breaks the answer** but is the
**same size** as an edit that doesn't. Prior work compares "prompts where the model flipped"
against "prompts where it didn't," but those groups also differ in how much the text changed.
So a surface explanation and a meaning explanation both fit their data, and their design
can't tell them apart.

Ours can, because we made the answer-breaking edit one character long.

**Important limit:** we did not re-run anyone else's experiment. We can say their design
cannot separate these two explanations, and that in a design that can, the signal is entirely
surface. We cannot say their specific result is wrong.

## Putting a number on it

Across all 469 comparisons, predicting whether the measure fires:

- Using only **how different the text is**: explains **40%** of the variation
- Adding **whether the answer broke**: adds **2.6%**, and the effect can't even be estimated,
  because the answer-breaking group has zero events

**What we could NOT determine: which surface property is responsible.** We tried three
answers in ninety minutes and retracted the first two.

- Syntactic difference looked like the driver — until token-count difference was tested and
  beat it (0.525 vs 0.395).
- Token length looked like the driver — until we refit without the `formal` condition, and
  it collapsed from 0.525 to **0.050**. `formal` is 54% longer than the original and fires
  100% of the time; it was carrying the whole result single-handedly.
- Among the four conditions that are matched for length, no surface measure explains much
  (best 0.17) and the ordering reverses.

So the honest statement is: **among length-matched conditions we cannot separate lexical,
syntactic, and length contributions.** Answering that needs a condition set designed for it,
with length varied independently of syntactic distance. Ours wasn't.

One oddity worth a sentence, not a claim: two conditions are identical on every surface
measure we have (zero token change, one-character edits) yet fire at 5.7% and 0%. Fisher
exact p = 0.027 — but on 4 events with no multiplicity correction, so it is suggestive at
most. If real, it means the measure fires *more* on the version that preserves the answer
than the one that breaks it.

**None of this touches the main result**, which is a contrast between conditions, not a
regression coefficient.

## Two supporting results

**Formal notation makes the model worse, not better.** 56% correct vs 70% for plain English
(p = 0.007). So formal rewriting changes the internal geometry *and* hurts performance — those
come apart, which is worth knowing on its own.

**Our own metric was fragile.** The main geometric measure depended on a clustering parameter
we'd treated as a detail. Changing it shrank the effect four-fold and erased its significance.
And on 10 of 48 layers, the clustering isolated a *single token* — a handful of extreme
activations dominating everything. Those layers weren't measuring language at all.

## What we set out to do, and didn't

We predicted formal rewriting would make the internal geometry more modular, and that
answer-breaking prompts would diverge earlier. **Both were wrong** — the second one backwards.

We wrote those predictions down, with their directions, hashed and timestamped, before
running anything. That's the only reason this is a result instead of an accident: two
findings that would have looked publishable and been false got caught. One of them had a
p-value of 10⁻¹⁴⁴ and was pure artifact — a statistical model with no data in one group.

## What to do with it

The paper is not "our method failed." It is:

> **A widely used measure of internal divergence tracks surface form rather than meaning, and
> here is the minimal-pair control that shows it.**

The failed hypotheses become the mechanism section: *this is why the measure behaves that
way.* The pre-registration becomes a contribution, not an apology.

## Where everything is

| | |
|---|---|
| `results/METHODS_FACTS.md` | every claim the Methods section needs |
| `prereg/PRIMARY_ENDPOINTS.md` | predictions, sealed before any code ran |
| `prereg/DEVIATIONS.md` | full trail: 6 pre-data amendments, then post-data findings, with corrections in place |
| `results/analysis.json` | pre-registered analysis |
| `results/analysis_fixedk8.json` | k-sensitivity |
| `results/g2_behavioral.json` | accuracy per condition |
| `paper/figs/` | 4 figures, real data |
| `results/human_review_sheet.md` | the 100 paraphrases, still unread by a human |

**One thing still owed:** nobody has read the 100 paraphrases to confirm they preserve
meaning. The machine checks all passed. That's the last human step.
