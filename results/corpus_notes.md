# Corpus notes — 100 problems x 4 realizations

Covers DESIGN.md §A.5 (contamination-controlled corpus), §A.6 (aligned
positions), §B.8 (formal = input notation, no execution), and
`configs/corpus_schema.md`.

Build command, and the only command needed to reproduce the file:

```
python3 scripts/02_generate_corpus.py --verify-reproducible
```

| | |
|---|---|
| output | `data/corpus/problems.jsonl` |
| summary | `data/corpus/corpus_summary.json` |
| master seed | `20260906` (read from `configs/model.yaml`) |
| sha256 of the JSONL | `28185911251c644e6c518ff4bd2eca987842f6d98da8545a2108252dc0516473` |
| problems | 100, one per template, 100 distinct templates |
| realizations generated | 300 (`nl_base`, `formal`, `nl_break`) |
| realizations pending | 100 (`nl_para`) |
| build drop rate | 0 / 100 templates |
| runtime | ~2 s, pure stdlib, no network |

---

## 1. Template route: OFFICIAL — Apple's GSM-Symbolic templates

**We used the official templates.** This is stated up front because the
alternative route (authoring 100 fresh templates) would have been a materially
different artifact and must not be substituted quietly.

- Source: <https://github.com/apple/ml-gsm-symbolic>, `templates/symbolic/0000.json`
  … `0099.json`, commit `b6a1625025fc857300203bac9f617e5d8ec99f65` (2024-12-08),
  fetched 2026-09-06.
- Paper: Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar.
  *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in
  Large Language Models.* ICLR 2025. arXiv:2410.05229.
- Licence: Apple Inc.; see the repo `LICENSE`. Recorded in the vendored bundle.
- Vendored to `data/templates/gsm_symbolic_templates.json` so that generation
  has **no network dependency at run time**.

This is exactly what DESIGN.md §A.5 asks for: *"Build NL-base from GSM-Symbolic
templates (Mirzadeh et al., ICLR 2025) with freshly sampled instantiations."*

### 1.1 What is Apple's and what is ours

The vendored bundle keeps **only** each template's `question_annotated` field —
the symbolic form with `{var, default}` slots and the `#init:` / `#conditions:` /
`#answer:` directive blocks. The `question` and `answer` fields, which are
**verbatim GSM8K instances**, are deliberately dropped. No raw GSM8K text is in
this repository. That is the whole contamination argument, and it is enforced by
a test (`tests/test_corpus.py::test_bundle_ships_no_raw_gsm8k_text`).

Two things had to be reimplemented, because Apple published the templates but
**not** their parser/data generator (their README says so explicitly):

1. **The DSL interpreter** (`src/ling/templates.py`). Ours, not Apple's. It
   parses the `{var, default}` slots, executes `#init:` bindings, rejection-samples
   against `#conditions:`, and evaluates `#answer:`.
2. **The lexical word pools.** The DSL references pools by bare name (`names`,
   `weights_med`, `fractions`, `currencies_sym`, …) whose contents were never
   published. `templates.WORD_POOLS` is a **reconstruction**, inferred from
   (a) the in-template defaults — `{unit, pound}` proves `pound ∈ weights_med` —
   and (b) the official generated instances published in the same repo, which is
   how we know `numbers_within(31, 35)` renders as `"thirty one"` (space, not
   hyphen) and that `weights_med` contains `kg`.

**The reconstruction touches surface nouns only.** Every numeric pool
(`range`, `numbers_within`, `np.arange`), every `#conditions:` clause and every
`#answer:` expression is Apple's, verbatim. Ground truth is computed by
executing *their* answer expression on *our* freshly sampled values.

### 1.2 Deviations from Apple's generator, and why

Three places where we deliberately do not reproduce whatever Apple's unpublished
generator does. All three are visible in the code with the reasoning at the call
site.

| Deviation | Why |
|---|---|
| **Lexical coherence guard** (`Template._lexically_coherent`): reject a draw in which two distinct *string* slots land on the same word. | Template 0061 draws `objects` and `fake_object` from overlapping inline lists, which yields *"they weren't coins, but actually coins"*. An incoherent prompt is a silently broken item in a 100-item corpus. Numeric collisions stay legal. |
| **Sentence-initial capitalisation** at emit time. | Template 0007 writes `{o1, hay}` at the start of a sentence with a lowercase default; the official generated data capitalises it (*"Apples are the main diet …"*). Applied **before** the surface is appended, so the recorded span always matches the final string — the text is never post-edited. |
| **Indefinite-article agreement** (`_fix_article`). | Templates hard-code `a`, producing *"a agent fee"* / *"a hour"*. The fix rewrites the article in the *preceding literal chunk*, which is emitted before any span is recorded for the slot that follows it. |

Residual English artifacts that come from Apple's own templates (*"He tripped
over root"*, *"He gets 1 hours free"*, *"three kinds of herbs: tulips,
sunflowers and lilies"*) are **left alone**. Hand-repairing them would turn the
citable artifact into an authored one. Gate G3 (round-trip verification by a
different model) is the check that they do not prevent answer recovery, and its
drop rate is reported separately.

---

## 2. The four conditions

Per `configs/corpus_schema.md`. Three are emitted programmatically; the fourth
is an explicit placeholder.

### `nl_base` — reference
Natural-language rendering, freshly sampled values, ground truth **computed** by
executing the template's `#answer:` expression. No model in the loop, so there is
nothing to hallucinate. Rejection sampling additionally requires the answer to be
a **positive, clean** number (integer, or exact to two decimals for money) —
these are counting/money/distance problems, so a negative, zero or
`6.3333333333` "ground truth" means the draw fell outside the template author's
intent even where the arithmetic is legal.

All 100 ground truths are positive integers (range 1 … 137200).

### `formal` — Python rewrite of the same instance
Emitted as bindings for each quantity plus `answer = <the template's own answer
expression>`:

```python
# gsms_0007

n4 = 4
w4 = 5
n3 = 12
w3 = 9
n1 = 34
w1 = 38
n2 = 20
w2 = 19
total = 100

answer = (n1*w1 + n2*w2 + n3*w3 + n4*w4)//total
```

Only quantities and the variables the answer expression actually reads are bound.
Names, colours and units are dropped — that is what makes this a *formal* rather
than a *mixed* rendering, and it is why `formal` is a median 94 characters
against `nl_base`'s 220.

Per DESIGN.md §B.8 this string is **input notation and is never executed during
the experiment**. It is executed exactly once, at build time, and the build fails
unless `answer == ground_truth`. That check runs for every problem
(`corpus.build_problem`) and again in the test suite.

> **Flag for the parent / G2.** Because the formal condition contains the answer
> *expression*, it makes the arithmetic explicit in a way `nl_base` does not.
> Gate G2 ("if formal accuracy is not above NL-base, cut the accuracy half of
> H3") is therefore likely to be near-ceiling on `formal`, and the honest
> reading of a large formal>NL accuracy gap is "we handed the model the
> program", not "formal notation aids reasoning". The geometric claim (H3,
> λ₂ decrease) is unaffected. See §6.

### `nl_break` — minimal answer-breaking perturbation
Exactly one quantity is changed; the answer is recomputed and asserted to differ.
Candidates are scored so the winner is as surface-similar as the template allows:
satisfying the template's own `#conditions:` dominates, then character edit
distance of the substituted surface form, then the number of places that variable
occurs, then numeric distance.

| | |
|---|---|
| differ from `nl_base` by exactly **one character** | **98 / 100** |
| edit distance 3 | 1 (`five times` → `four times`) |
| edit distance 8 | 1 (`three times` → `twice`) |
| only one slot rewritten | 100 / 100 |
| template `#conditions:` still satisfied | 97 / 100 |
| answer actually changed | 100 / 100 (asserted, fatal otherwise) |

The three items whose conditions no longer hold are recorded with
`template_conditions_still_satisfied: false`; all three still produce a positive
integer answer (25→24, 62→61, 148→147), so they are not detectably broken
arithmetic — but the flag is there so they can be excluded in a robustness check.

A few templates sample a *solved-for* scratch variable that never surfaces in the
prose (`ans`, `x`) and pin it with an equality in `#conditions:`. Perturbing a
quantity leaves those inconsistent, so `_solve_free` re-solves the scratch
variable over its declared pool before the perturbed answer is read off.

This matters for H2: `configs/corpus_schema.md` warns that a perturbation which is
*both* semantically and lexically far away makes H2 trivially true. At a median
edit distance of 1 character, the lexical and syntactic dissimilarity covariates
in the survival model have essentially nothing to explain, which is precisely the
control Han et al. lacked.

### `nl_para` — NOT WRITTEN
Deliberately left as:

```json
{"text": "", "anchors": {}, "answer": <gt>, "status": "pending", "note": "..."}
```

Anchors are **undetermined**. A separate pass writes the answer-preserving
paraphrase and must **emit** its anchor spans during construction, exactly as
`render_nl` does — not recover them by searching the finished string.

Consequence for the schema invariant: `configs/corpus_schema.md` requires every
quantity key to appear in the anchors of **all four** realizations, because
DESIGN.md §A.6 restricts extraction to positions present in all four. That
invariant currently holds for the three generated conditions and **cannot be
evaluated** for `nl_para`. `corpus.validate_record` says so rather than quietly
passing: as soon as `status` is no longer `"pending"`, the paraphrase is held to
the same anchor requirement as the rest.

---

## 3. How span exactness is guaranteed

This is the property that most needs to be right, because a span that is off by
one character maps to the wrong token at extraction time and every λ₂ / fork
layer / AMI number downstream is plausible-looking garbage with nothing in the
output to reveal it (DESIGN.md §A.11).

1. **Spans are emitted, never searched.** `render_nl` and `render_python` build
   the output by appending chunks and record `[start, end)` as the length of the
   text emitted so far and that plus the length of the chunk just appended.
   `str.find` / `re.search` are never used to locate a quantity. A quantity whose
   surface happens to repeat elsewhere in the prose therefore cannot be
   mis-anchored — and the corpus does contain such repeats, which a test asserts
   so the guarantee is not vacuous.
2. **Emitted-length bookkeeping is asserted against reality.** After the build,
   `assert pos == len(text)`.
3. **`text[start:end] == literal` is re-asserted before anything is written.**
   `Rendered.validate()` raises `SpanError` — not a warning, not a skip — for a
   bad span, a missing quantity or a missing `FINAL`. `SpanError` propagates out
   of the generator and kills the run.
4. **Every mutation that changes text is applied before the span is recorded.**
   Sentence-initial capitalisation and article agreement both happen in the
   chunk-resolution pass, ahead of the span-recording pass. Nothing post-edits a
   finished string.
5. **`quantities[].surface` and `quantities[].formal_literal` store the literals
   actually emitted**, so the record carries its own independently checkable
   witness for both notations.
6. **Anchors are checked pairwise disjoint**, since two quantities resolving into
   one token would silently duplicate a node in the graph.
7. **The validator is itself tested** against a deliberately tampered span, a
   deleted quantity and an out-of-bounds span. A validator that never rejects
   anything is worse than none.

1200 quantity spans (400 quantities x 3 generated conditions) are verified on
every build and again in the test suite.

**`FINAL` sentinel.** Present in all 300 generated realizations, encoded as
`[-1, -1]` — the encoding `configs/corpus_schema.md` specifies and
`ling.hooks.resolve_anchor_positions` reads as "the last token". Enforced, not
assumed: `validate_record` rejects any other value.

---

## 4. Linguistic variation

H1 scores cluster assignment against linguistic labels. A corpus in which every
problem shares one syntactic frame gives those labels no variance and makes H1
**unmeasurable** — and nothing else in the pipeline would report that. This is
the specific failure mode of generating a corpus programmatically, so it is
measured, recorded per problem in `linguistic_profile`, and guarded by tests.

Using the official templates is what buys the variation: each of the 100 comes
from a different GSM8K problem, so the syntactic frames are inherited from 100
independently written texts rather than from one generator's habits.

Measured on the 100 `nl_base` renderings:

| axis | distribution |
|---|---|
| question form | how many 54, how much 28, what 8, how long 2, how old 2, imperative *calculate* 3, imperative *find* 1, other 2 |
| sentences per problem | 1:2, 2:19, 3:38, 4:20, 5:11, 6:9, 7:1 |
| words per problem | min 19, median 37, max 107 |
| clause markers used | if 37, that 22, after 12, so 10, but 8, then 7, which 7, while 5, when 5, since 2 |
| distinct clause-marker signatures | **32** |
| clause markers per problem | 0:27, 1:44, 2:20, 3:5, 4:4 |
| opens with a proper name | 65 yes / 35 no (bare NP, expletive *there*, PP, conditional *if*) |
| quantities per problem | 2:10, 3:32, 4:29, 5:15, 6:9, 7:3, 9:2 |
| first quantity, relative position | quartiles 0.08 / 0.13 / 0.23 (range 0.00 – 0.47) |
| quantity-introduction pattern | enumerated list 18, currency-marked 25, word numerals 41 |
| text length (chars) | `nl_base` 96–586 (median 220); `formal` 45–298 (median 94) |

Quantity-introduction patterns present in the corpus include: post-nominal
measure phrases (*"a 200-foot dolphin with 8 6-inch remoras"*), enumerated
inventories (*"7775 biscuits … 1774 blueberry, 2793 lemon, and some vanilla"*),
rate expressions (*"148 minutes to cover every 16 miles"*), ratio expressions
(*"in the ratio of 148:75"*), multiplicative anaphora (*"triple the number he did
on Thursday"*), fractional determiners (*"a fifth of the balls"*), currency
(*"$0.5 for each minute"*), spelled-out numerals (*"thirty four 38-kg packs"*),
and quantities embedded inside quoted advertising copy (*"Buy 16 kilograms at $21
per kilogram, get 4 kilograms 1/4 off"*).

Note the intended asymmetry: `formal` is deliberately *uniform* in syntax. That
is the H3 contrast, not a defect. k is selected on `nl_base` (DESIGN.md §A.6),
and H1 is a claim about `nl_base`.

---

## 5. Determinism

- One master seed (`configs/model.yaml: seed`), read by the script with a
  one-line regex rather than a YAML dependency, so the corpus and the GPU side
  cannot silently drift apart.
- Each problem draws from its own RNG derived by SHA-256 from
  `(seed, template_id)`. Adding, dropping or reordering templates therefore
  cannot perturb any other problem's values — tested.
- JSONL is written with `sort_keys=True`, `ensure_ascii=False`, LF newlines,
  UTF-8.
- `--verify-reproducible` rebuilds the entire corpus a second time in-process and
  requires byte-identical serialisation before writing.

---

## 6. Open items, and one thing to decide

1. **`nl_para` is not written.** 100 realizations pending. The writer must emit
   anchors during construction and must re-run the schema invariant check.
2. **G3 (round-trip verification) has not been run.** The 0/100 drop rate above
   is the *build* drop rate — templates that produced no valid problem — not G3's
   drop rate. `prereg/PRIMARY_ENDPOINTS.md` G3 requires a **different model than
   the generator** to independently recover the ground truth from each
   realization. No model was involved in generating `nl_base`, `formal` or
   `nl_break` at all, which makes "different from the generator" trivially
   satisfiable, but the verification itself still has to happen.
3. **Decide what `formal` is for** (see §2). As emitted, `formal` contains the
   answer expression, so G2's accuracy comparison will likely be near-ceiling for
   reasons that have nothing to do with notation. Two options:
   (a) keep it and report G2 honestly as "the formal condition supplies the
   computation, so the accuracy comparison is not informative; the dissociation
   result of §B.8 is the one we are writing"; or
   (b) re-emit `formal` as givens + goal without the derivation, which makes G2
   meaningful but is no longer executable and so loses the build-time
   ground-truth check. This is a design decision, not an implementation one, and
   it is flagged rather than taken unilaterally.
4. **Reconstructed word pools** (§1.1) should be named as such in the paper's
   corpus paragraph, in one sentence, alongside the template citation.

---

## 7. Files

| path | role |
|---|---|
| `data/templates/gsm_symbolic_templates.json` | vendored official templates + provenance |
| `src/ling/templates.py` | template representation, DSL interpreter, sampler, both surface forms, span emission |
| `src/ling/corpus.py` | the four conditions, perturbation search, validation, linguistic profile |
| `scripts/02_generate_corpus.py` | seeded deterministic driver; writes the JSONL and the summary |
| `tests/test_corpus.py` | 39 stdlib-`unittest` tests; `python3 -m unittest discover -s tests` |
| `configs/corpus_requirements.txt` | dependency statement (there are none) |
| `data/corpus/problems.jsonl` | the corpus |
| `data/corpus/corpus_summary.json` | machine-readable counts, regenerated with the corpus |

### Record fields beyond `configs/corpus_schema.md`

The schema's required keys (`problem_id`, `template_id`, `seed`, `ground_truth`,
`quantities`, `realizations`, `anchors`, `FINAL`) are present and unchanged.
Additive keys, all documented here:

- `quantities[].surface`, `quantities[].formal_literal` — the literals actually
  emitted, so each record carries its own span witness.
- `realizations[*].answer`, `.n_chars`; `realizations.formal.notation`;
  `realizations.nl_break.{answer_preserving, ground_truth_base, perturbed_quantity}`;
  `realizations.nl_para.{status, note}`.
- `perturbation` — full record of the `nl_break` edit (which quantity, from/to
  value and surface, both answers, whether the template's conditions still hold,
  surface and whole-text edit distances).
- `answer_preserving` — per-condition survival label; `nl_break` is the only
  `false`. This is H2's outcome variable.
- `linguistic_profile` — §4's per-problem descriptors.
- `template_source` — repo, commit, file, original GSM8K id, route.
