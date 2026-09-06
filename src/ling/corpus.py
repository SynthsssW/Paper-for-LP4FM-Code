"""Corpus construction: 100 problems x 6 realizations (configs/corpus_schema.md).

Three conditions are emitted here directly from the parameterised symbolic
templates in `ling.templates`:

  * ``nl_base``  -- natural-language rendering, freshly sampled values, ground
                    truth COMPUTED by executing the template's own `#answer:`
                    expression. No model in the loop.
  * ``formal``   -- GIVENS + GOAL rewrite of the SAME instance, in formal
                    notation, with NO derivation and no answer expression
                    (prereg/DEVIATIONS.md Amendment 4; DESIGN.md B.8: our formal
                    condition is input notation with no execution). It is
                    therefore not executable, and the old build-time "the formal
                    program returns ground truth" check is gone. In its place:
                    every literal in the givens block must equal the value it
                    claims, and the goal must be the template's own question.
                    Ground truth comes from the template's `#answer:` expression
                    and always did.
  * ``nl_break`` -- exactly ONE quantity perturbed so the computed answer
                    changes, chosen to keep the surface as close to `nl_base`
                    as the template allows. Both answers are recorded and the
                    change is asserted.

  * ``nl_para``  -- answer-preserving paraphrase, rendered from the authored
                    schemas in ``data/corpus/paraphrases.json`` against the SAME
                    bindings as ``nl_base``. The schemas carry ``{var}``
                    placeholders rather than finished prose, which is what makes
                    every quantity appear with the surface form ``nl_base``
                    emitted for it and makes its span emitted rather than
                    searched. If the bundle is absent the slot falls back to the
                    honest empty placeholder (`PARA_PLACEHOLDER`).

Two further conditions (prereg/DEVIATIONS.md **Amendment 6**) are emitted here as
well. Both are answer-preserving and both are MINIMAL edits of ``nl_base``:

  * ``nl_minimal_orth`` -- exactly ONE quantity re-rendered as an English number
                    word (``24`` -> ``twenty four``), with the SAME value. The
                    slot chosen is, wherever the value allows it, the very slot
                    ``nl_break`` perturbs, so the two conditions edit the same
                    position and differ ONLY in whether the value -- and hence
                    the answer -- changes. That matched-site/matched-kind
                    contrast is what identifies H2's residual.
  * ``nl_minimal_lex``  -- exactly ONE proper name swapped for another drawn from
                    the same template pool. Semantically null, and deliberately
                    OFF-anchor: names are not quantities, so this edit lands
                    nowhere near an anchor. It is the LOWER bracket on the
                    surface-perturbation floor, as ``nl_minimal_orth`` is the
                    upper one; Amendment 6 says in as many words that neither is
                    trustworthy alone.

Neither is faked when the instance cannot support it. A template with no
integer quantity rendered in digits, or with no proper name at all, records that
condition as ``status: "unavailable"`` with empty text and NO anchors -- the same
honest hole ``PARA_PLACEHOLDER`` leaves -- rather than inventing a name or a
wording. `build_corpus` counts them.

WHY THE ANCHOR SPANS ARE TRUSTWORTHY
------------------------------------
Spans are *emitted*, never recovered. `Template.render_nl` / `render_formal`
build the string chunk by chunk and record `[start, end)` at the instant the
quantity's surface form is appended; `str.find` never runs. Every record is then
re-validated with `text[start:end] == literal` before it is written, and a
mismatch raises `SpanError` and kills the run. A silently wrong span means the
extractor clusters the wrong token and every downstream number is plausible
garbage (DESIGN.md A.11), so this is a crash, not a warning.

WHY `nl_break` IS BUILT THE WAY IT IS
-------------------------------------
`nl_break` is the H2 workhorse. It must be surface-*similar* to `nl_base` but
answer-breaking, so surface-dissimilarity covariates cannot explain the fork
(configs/corpus_schema.md). The search therefore scores candidate perturbations
by character edit distance on the substituted surface form and by how many times
that variable occurs, and prefers perturbations that still satisfy the
template's `#conditions:` so the perturbed problem stays well-formed rather than
becoming detectably broken arithmetic.

Pure standard library.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .templates import (
    WORD_POOLS,
    Instance,
    IntQ,
    Rendered,
    SpanError,
    Template,
    TemplateError,
    _wrap,
    is_numeric,
    number_to_words,
)

__all__ = [
    "CONDITIONS",
    "GENERATED_CONDITIONS",
    "MINIMAL_CONDITIONS",
    "PARA_PLACEHOLDER",
    "UNAVAILABLE",
    "CorpusError",
    "assert_formal_givens",
    "assert_formal_goal",
    "build_problem",
    "build_corpus",
    "load_paraphrases",
    "DEFAULT_PARAPHRASE_BUNDLE",
    "linguistic_profile",
    "name_swap_for",
    "orth_target_for",
    "validate_record",
]

#: All six conditions, in corpus order (prereg/DEVIATIONS.md Amendment 6).
CONDITIONS = (
    "nl_base",
    "nl_para",
    "formal",
    "nl_break",
    "nl_minimal_orth",
    "nl_minimal_lex",
)
#: Conditions every record always carries, emitted straight from the template.
GENERATED_CONDITIONS = ("nl_base", "formal", "nl_break")
#: Amendment 6's two style controls. Present when the instance supports them,
#: `status: "unavailable"` when it does not -- never faked.
MINIMAL_CONDITIONS = ("nl_minimal_orth", "nl_minimal_lex")

#: Largest value `number_to_words` renders correctly (it recurses on thousands,
#: so a million would come out as "one thousand thousand").
MAX_WORDED_VALUE = 999999

#: Cap on how many alternative values are tried per quantity when searching for
#: a minimal answer-breaking perturbation.
MAX_ALTERNATIVES_PER_QUANTITY = 240
#: Cap on the search over solved-for scratch variables when re-solving a
#: perturbed instance (only a handful of templates have any).
MAX_FREE_COMBINATIONS = 20000


class CorpusError(RuntimeError):
    """Raised when a problem cannot be built to specification."""


def PARA_PLACEHOLDER(ground_truth: float) -> Dict[str, Any]:
    """The `nl_para` slot: deliberately empty, anchors NOT yet determined.

    DESIGN.md A.5's answer-preserving paraphrase is written by a separate pass
    (it is the one condition that genuinely needs a generator model). Leaving it
    empty here, rather than faking it, keeps the schema invariant honest: the
    "every quantity appears in all realizations" check in
    configs/corpus_schema.md cannot be evaluated until this slot is filled, and
    `validate_record` says so instead of quietly passing.
    """
    return {
        "text": "",
        "anchors": {},
        "answer": ground_truth,
        "status": "pending",
        "note": (
            "Answer-preserving paraphrase not yet written. Anchors are "
            "UNDETERMINED and must be emitted during paraphrase construction, "
            "not recovered by searching the finished string."
        ),
    }


def UNAVAILABLE(ground_truth: float, reason: str) -> Dict[str, Any]:
    """An Amendment 6 style-control slot this instance cannot support.

    Empty text, NO anchors, and a reason. The alternative -- inventing a proper
    name for a template that has none, or wording a fraction as if it were an
    integer -- would put a fabricated realization into a condition whose entire
    purpose is to be a *minimal, semantically null* edit of `nl_base`. A hole
    that counts itself is worth more than a filled cell that lies.
    """
    return {
        "text": "",
        "anchors": {},
        "answer": ground_truth,
        "status": "unavailable",
        "note": reason,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _json_number(value: Any) -> Any:
    """Make a bound value JSON-safe without losing exactness silently."""
    if isinstance(value, Fraction):
        return {"num": value.numerator, "den": value.denominator}
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _occurrences(template: Template, var: str) -> int:
    return sum(1 for seg in template.segments if seg.kind == "ph" and seg.var == var)


# ---------------------------------------------------------------------------
# nl_break: minimal answer-changing perturbation
# ---------------------------------------------------------------------------


@dataclass
class Perturbation:
    quantity: str
    old_value: Any
    new_value: Any
    old_surface: str
    new_surface: str
    bindings: Dict[str, Any]
    answer: float
    conditions_satisfied: bool
    edit_distance: int
    occurrences: int


def _solve_free(
    template: Template, bindings: Dict[str, Any], free: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Re-solve the template's scratch variables after a quantity is perturbed.

    A few templates sample a variable that never surfaces in the question (`ans`,
    `x`) and pin it with an equality in `#conditions:` -- it *is* the answer.
    Perturbing a quantity leaves such a template inconsistent until that scratch
    variable is re-solved, so search its declared pool for the value that
    restores the conditions. Returns None if none does.
    """
    if not free:
        return dict(bindings) if template.conditions_hold(bindings) else None
    pools = [[_wrap(v) for v in template.pool_for(name)] for name in free]
    total = 1
    for p in pools:
        total *= max(len(p), 1)
        if total > MAX_FREE_COMBINATIONS:
            return None
    for combo in itertools.product(*pools):
        trial = dict(bindings)
        for name, value in zip(free, combo):
            trial[name] = value
        if template.conditions_hold(trial):
            return trial
    return None


def _alternatives(template: Template, var: str, current: Any) -> List[Any]:
    """Candidate replacement values for one quantity, most surface-similar first."""
    pool = template.pool_for(var)
    cur_surface = getattr(current, "surface", str(current))
    out = []
    for raw in pool:
        cand = _wrap(raw)
        if not is_numeric(cand):
            continue
        if cand == current:
            continue
        surface = getattr(cand, "surface", str(cand))
        if surface == cur_surface:
            continue
        try:
            delta = abs(float(cand) - float(current))
        except (TypeError, ValueError):
            delta = 1e9
        out.append((_levenshtein(cur_surface, surface), delta, surface, cand))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in out[:MAX_ALTERNATIVES_PER_QUANTITY]]


def find_perturbation(inst: Instance) -> Perturbation:
    """Perturb exactly ONE quantity so the COMPUTED answer changes.

    Scored so that the winner is as surface-similar to `nl_base` as the template
    permits: satisfying the template's own `#conditions:` dominates, then the
    character edit distance of the substituted surface form, then how many
    places in the prose that variable occurs, then numeric distance. A
    perturbation that is both lexically and semantically far away would make H2
    trivially true (configs/corpus_schema.md), so "one digit differs" is the
    target and the score says how close we got.
    """
    template = inst.template
    free = [v for v in template.free_vars if v in inst.bindings]
    best: Optional[Perturbation] = None
    best_score: Tuple[int, int, int, float] = (10**9,) * 4

    for var in inst.quantities:
        occ = _occurrences(template, var)
        current = inst.bindings[var]
        cur_surface = getattr(current, "surface", str(current))
        for cand in _alternatives(template, var, current):
            trial = dict(inst.bindings)
            trial[var] = cand
            solved = _solve_free(template, trial, free)
            conditions_ok = solved is not None
            if solved is None:
                solved = trial  # fall back: conditions violated, recorded as such
            answer = template.answer(solved)
            if answer is None or answer <= 0 or answer == inst.answer:
                continue
            surface = getattr(cand, "surface", str(cand))
            dist = _levenshtein(cur_surface, surface)
            try:
                delta = abs(float(cand) - float(current))
            except (TypeError, ValueError):
                delta = 1e9
            score = (0 if conditions_ok else 1, dist, occ, delta)
            if score < best_score:
                best_score = score
                best = Perturbation(
                    quantity=var,
                    old_value=current,
                    new_value=cand,
                    old_surface=cur_surface,
                    new_surface=surface,
                    bindings=solved,
                    answer=answer,
                    conditions_satisfied=conditions_ok,
                    edit_distance=dist,
                    occurrences=occ,
                )
            if best_score[:2] == (0, 1):
                return best  # single-character difference, conditions intact
    if best is None:
        raise CorpusError(
            "%s: no answer-changing single-quantity perturbation found"
            % template.template_id
        )
    return best


# ---------------------------------------------------------------------------
# Amendment 6: the two minimal answer-PRESERVING edits
# ---------------------------------------------------------------------------
#
# The design problem these fix: on the four-condition corpus, `nl_break` was the
# only answer-breaking condition AND the only one-character edit, so "surface
# dissimilarity" was very nearly a perfect proxy for "answer-preserving" and H2's
# residual coefficient -- the part that is ours rather than Han et al.'s -- was
# weakly identified. Both conditions below are answer-preserving minimal edits,
# so the two factors stop moving together.


@dataclass
class MinimalEdit:
    """One answer-preserving minimal edit and everything a reviewer would check."""

    kind: str  # "orth" | "lex"
    variable: str
    old_surface: str
    new_surface: str
    bindings: Dict[str, Any]
    occurrences: int
    on_anchor: bool
    #: For "orth": did we land on the slot `nl_break` perturbs? Amendment 6's
    #: matched-site contrast is void if we did not, so it is recorded per
    #: problem rather than assumed.
    matches_break_site: bool
    conditions_satisfied: bool


def orth_target_for(
    inst: Instance, preferred: Optional[str] = None
) -> Optional[Tuple[str, IntQ]]:
    """Pick the quantity to re-render as a number word, and build its new value.

    `preferred` is `nl_break`'s perturbed quantity. It is tried FIRST, because
    the whole point of `nl_minimal_orth` is that it edits the same site as
    `nl_break` and differs only in whether the value changes. It is tried first
    rather than required, because a slot whose value is a Fraction or is already
    spelled out cannot be re-rendered as a number word at all; the fallback walks
    the remaining quantities in first-appearance order and the record says
    whether the site matched.

    Eligible = a non-boolean integer, currently surfaced as DIGITS (a value drawn
    from `numbers_within(...)` is already a number word, so re-wording it would
    be a no-op), in the range `number_to_words` renders correctly.
    """
    order = list(inst.quantities)
    if preferred in order:
        order = [preferred] + [q for q in order if q != preferred]
    for name in order:
        value = inst.bindings.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if not (0 <= int(value) <= MAX_WORDED_VALUE):
            continue
        if getattr(value, "surface", str(value)) != str(int(value)):
            continue  # already a word, or a decorated surface
        # Same convention the templates' own `numbers_within(...)` pool uses
        # ("thirty two", space-separated, no hyphen) -- matched deliberately so
        # that a worded quantity in this condition is indistinguishable from a
        # worded quantity the template itself produced.
        worded = number_to_words(int(value))
        if worded == value.surface:
            continue
        return name, IntQ(int(value), worded)
    return None


#: Person-name pools. `cities`/`weekdays` are proper nouns too, but Amendment 6
#: says "proper name" and a swapped city can change what the situation is about.
_PERSON_NAMES = frozenset(
    WORD_POOLS["names"] + WORD_POOLS["names_male"] + WORD_POOLS["names_female"]
)


def _is_name_pool(pool: Sequence[Any]) -> bool:
    """True if this init pool is a pool of person names.

    Decided from the POOL, not from the sampled value: a template that inlines
    its own list of names is still drawing names, and a stray string that
    happens to collide with our reconstructed `names` pool is not.
    """
    if len(pool) < 2:
        return False
    return all(type(v) is str and v in _PERSON_NAMES for v in pool)


def name_swap_for(inst: Instance) -> Optional[Tuple[str, str, str]]:
    """Pick one proper-name slot and its replacement. -> (var, old, new) or None.

    The first name-valued placeholder in first-appearance order is swapped, for
    the surface-closest alternative from the SAME pool (ties broken
    alphabetically) -- deterministic, and consuming no RNG, so this cannot
    perturb any other draw. Names already used by another lexical slot are
    excluded, because `Template._lexically_coherent` rejects a draw where two
    slots land on the same word and this edit must not create one.
    """
    template = inst.template
    taken = {
        v.lower() for k, v in inst.bindings.items() if type(v) is str
    }
    for var in template.placeholder_vars:
        current = inst.bindings.get(var)
        if type(current) is not str:
            continue
        pool = template.pool_for(var)
        if not _is_name_pool(pool):
            continue
        cands = [
            c for c in pool if c != current and c.lower() not in taken
        ]
        if not cands:
            continue
        cands.sort(key=lambda c: (_levenshtein(current, c), c))
        return var, current, cands[0]
    return None


def build_minimal_orth(inst: Instance, preferred: Optional[str]) -> Optional[MinimalEdit]:
    """`nl_minimal_orth`: one quantity, same VALUE, spelled out in words."""
    picked = orth_target_for(inst, preferred)
    if picked is None:
        return None
    var, value = picked
    bindings = dict(inst.bindings)
    bindings[var] = value
    # Answer-preserving BY CONSTRUCTION -- the value is unchanged, only its
    # surface string is -- but re-executing the template's own `#answer:` is
    # cheap and catches a template that reads a quantity's surface rather than
    # its value.
    if inst.template.answer(bindings) != inst.answer:
        return None
    return MinimalEdit(
        kind="orth",
        variable=var,
        old_surface=inst.bindings[var].surface,
        new_surface=value.surface,
        bindings=bindings,
        occurrences=_occurrences(inst.template, var),
        on_anchor=True,
        matches_break_site=(var == preferred),
        conditions_satisfied=inst.template.conditions_hold(bindings),
    )


def build_minimal_lex(inst: Instance) -> Optional[MinimalEdit]:
    """`nl_minimal_lex`: one proper name swapped for another. OFF-anchor."""
    picked = name_swap_for(inst)
    if picked is None:
        return None
    var, old, new = picked
    bindings = dict(inst.bindings)
    bindings[var] = new
    if inst.template.answer(bindings) != inst.answer:
        # A name that enters the arithmetic would not be a name. Refuse rather
        # than ship an "answer-preserving" condition whose answer moved.
        return None
    return MinimalEdit(
        kind="lex",
        variable=var,
        old_surface=old,
        new_surface=new,
        bindings=bindings,
        occurrences=_occurrences(inst.template, var),
        on_anchor=False,
        matches_break_site=False,
        conditions_satisfied=inst.template.conditions_hold(bindings),
    )


_ARTICLE_ONLY_RE = re.compile(r"(?<![A-Za-z])[Aa]n?(?=\s)")


def _assert_minimal_edit(
    template: Template,
    base_chunks: Sequence[str],
    new_chunks: Sequence[str],
    edit: MinimalEdit,
) -> bool:
    """Confirm the rendering differs from `nl_base` ONLY at `edit.variable`.

    Checked over the template's segment list -- what the two strings were BUILT
    from -- rather than by diffing the finished text, so this is a statement
    about construction and not an inference from appearance.

    One adjacent literal chunk is allowed to differ, and only by its indefinite
    article: `_fix_article` makes "a 8" into "an eight" when the digit becomes a
    word, and shipping "a eight" to keep a byte count would be trading grammar
    for a metric. Returns whether that happened, so the record can say so.
    """
    tid = template.template_id
    if len(base_chunks) != len(new_chunks):
        raise CorpusError("%s/%s: chunk count changed" % (tid, edit.kind))
    diff = [i for i, (a, b) in enumerate(zip(base_chunks, new_chunks)) if a != b]
    slots = [
        i
        for i, seg in enumerate(template.segments)
        if seg.kind == "ph" and seg.var == edit.variable
    ]
    if not slots:
        raise CorpusError("%s/%s: %r has no slot in the body" % (tid, edit.kind, edit.variable))
    if not set(slots) <= set(diff):
        raise CorpusError(
            "%s/%s: not every slot of %r was rewritten" % (tid, edit.kind, edit.variable)
        )
    article_adjusted = False
    for i in diff:
        if i in slots:
            continue
        if template.segments[i].kind != "lit" or (i + 1) not in slots:
            raise CorpusError(
                "%s/%s: edit changed segment %d, which is not a slot of %r"
                % (tid, edit.kind, i, edit.variable)
            )
        if _ARTICLE_ONLY_RE.sub("a", base_chunks[i]) != _ARTICLE_ONLY_RE.sub(
            "a", new_chunks[i]
        ):
            raise CorpusError(
                "%s/%s: adjacent literal changed by more than its article"
                % (tid, edit.kind)
            )
        article_adjusted = True
    return article_adjusted


# ---------------------------------------------------------------------------
# Linguistic structure profile
# ---------------------------------------------------------------------------

_CLAUSE_MARKERS = (
    "if", "then", "so", "because", "after", "before", "while", "when",
    "since", "although", "unless", "but", "however", "which", "that",
)
_WH = ("how many", "how much", "how long", "how far", "how old", "what", "who", "which")
#: Several templates ask imperatively ("calculate the ...", "find the ..."),
#: which is a different syntactic frame, not a missing one.
_IMPERATIVE = ("calculate", "find", "determine", "work out", "compute")


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def linguistic_profile(text: str, anchors: Dict[str, List[int]]) -> Dict[str, Any]:
    """Cheap, dependency-free surface-syntactic descriptors of one NL rendering.

    Not a substitute for the taggers of DESIGN.md B.3 -- those run at analysis
    time. This exists so the *corpus* can be shown to vary in syntactic frame,
    clause order and quantity-introduction pattern. H1 scores cluster assignment
    against linguistic labels, so a corpus where every problem shares one
    syntactic frame makes H1 unmeasurable, and that failure is invisible unless
    someone measures it.
    """
    sents = _sentences(text)
    words = re.findall(r"[A-Za-z']+", text)
    low = text.lower()

    q_index = next((i for i, s in enumerate(sents) if "?" in s), -1)
    if q_index < 0:
        q_index = next(
            (
                i
                for i, s in enumerate(sents)
                if any(re.search(r"\b%s\b" % v, s.lower()) for v in _IMPERATIVE)
            ),
            -1,
        )
    q_sent = sents[q_index].lower() if q_index >= 0 else ""
    wh = next((w for w in _WH if w in q_sent), None)
    if wh is None:
        wh = next(
            (
                "imperative:" + v
                for v in _IMPERATIVE
                if re.search(r"\b%s\b" % v, q_sent)
            ),
            "none",
        )

    positions = sorted(
        span[0] / max(len(text), 1) for name, span in anchors.items() if name != "FINAL"
    )
    if positions:
        mean = sum(positions) / len(positions)
        spread = (sum((p - mean) ** 2 for p in positions) / len(positions)) ** 0.5
    else:
        mean, spread = 0.0, 0.0

    first_tok = words[0] if words else ""
    opens_with_name = bool(first_tok) and first_tok[0].isupper() and first_tok.lower() not in (
        "a", "an", "the", "if", "there", "in", "on", "at", "every", "when", "each",
    )

    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": len(sents),
        "question_form": wh,
        "question_sentence_index": q_index,
        "question_is_final_sentence": q_index == len(sents) - 1,
        "opens_with_proper_name": opens_with_name,
        "clause_markers": sorted(
            m for m in _CLAUSE_MARKERS if re.search(r"\b%s\b" % m, low)
        ),
        "n_clause_markers": sum(
            1 for m in _CLAUSE_MARKERS if re.search(r"\b%s\b" % m, low)
        ),
        "n_quantities": len(positions),
        "first_quantity_rel_pos": round(positions[0], 4) if positions else None,
        "quantity_spread": round(spread, 4),
        "has_enumeration": bool(re.search(r",\s+[^,]+,\s+and\b", text)),
        "has_currency": bool(re.search(r"[$€£¥]", text)),
        "has_word_numeral": bool(
            re.search(
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
                r"thirty|forty|fifty|sixty|seventy|eighty|ninety|half|third|quarter|"
                r"twice|double|triple)\b",
                low,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _realization(rendered: Rendered, answer: float, **extra: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "text": rendered.text,
        "anchors": {k: list(v) for k, v in rendered.anchors.items()},
        "answer": answer,
        "n_chars": len(rendered.text),
    }
    rec.update(extra)
    return rec


def build_problem(
    template: Template,
    rng: random.Random,
    seed: int,
    problem_id: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    paraphrase: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one corpus record: all six realizations.

    `paraphrase` is this problem's entry from data/corpus/paraphrases.json (see
    `load_paraphrases`). If it is None the `nl_para` slot is left as the honest
    empty placeholder; if it is present the paraphrase is rendered from the SAME
    bindings as `nl_base`, so its quantity spans are emitted rather than
    searched, and every span is re-asserted before the record is written.
    """
    inst = template.sample(rng)
    quantities = inst.quantities
    if not quantities:
        raise CorpusError("%s: instance has no anchorable quantity" % template.template_id)

    nl = template.render_nl(inst.bindings, quantities)
    formal = template.render_formal(inst.bindings, quantities)

    # Amendment 4: `formal` is givens + goal, so it is NOT executable and there
    # is no "the program returns ground truth" check to run. These two are what
    # replace it.
    assert_formal_givens(template, formal, inst.bindings, quantities)
    assert_formal_goal(template, formal, nl)

    pert = find_perturbation(inst)
    if pert.answer == inst.answer:  # belt and braces
        raise CorpusError("%s: perturbation did not change the answer" % template.template_id)
    brk = template.render_nl(pert.bindings, quantities)

    # nl_break must differ from nl_base ONLY where the perturbed variable is
    # placed. Checked structurally against the segment list, not by diffing the
    # finished strings.
    _assert_single_quantity_edit(template, inst.bindings, pert, nl.text, brk.text)

    # Amendment 6's two style controls. Built AFTER the perturbation because
    # `nl_minimal_orth` deliberately targets the slot `nl_break` perturbs.
    base_chunks = template._chunks(inst.bindings)
    minimal: Dict[str, Any] = {}
    minimal_rendered: Dict[str, Rendered] = {}
    for kind, cond, edit in (
        ("orth", "nl_minimal_orth", build_minimal_orth(inst, pert.quantity)),
        ("lex", "nl_minimal_lex", build_minimal_lex(inst)),
    ):
        if edit is None:
            minimal[cond] = {
                "available": False,
                "reason": (
                    "no integer quantity rendered in digits that a number word "
                    "could replace"
                    if kind == "orth"
                    else "the template instantiates no proper name"
                ),
            }
            continue
        new_chunks = template._chunks(edit.bindings)
        article_adjusted = _assert_minimal_edit(template, base_chunks, new_chunks, edit)
        rendered = template.render_nl(edit.bindings, quantities)
        if rendered.text == nl.text:
            raise CorpusError("%s/%s: text identical to nl_base" % (template.template_id, cond))
        for name in quantities:
            if name != edit.variable and rendered.literals[name] != nl.literals[name]:
                raise SpanError(
                    "%s/%s: quantity %r reads %r but nl_base emitted %r"
                    % (template.template_id, cond, name, rendered.literals[name],
                       nl.literals[name])
                )
        minimal_rendered[cond] = rendered
        minimal[cond] = {
            "available": True,
            "kind": edit.kind,
            "variable": edit.variable,
            "from_surface": edit.old_surface,
            "to_surface": edit.new_surface,
            "on_anchor": edit.on_anchor,
            "matches_nl_break_site": edit.matches_break_site,
            "occurrences_rewritten": edit.occurrences,
            "article_adjusted": article_adjusted,
            "template_conditions_still_satisfied": edit.conditions_satisfied,
            "text_char_edit_distance": _levenshtein(nl.text, rendered.text),
        }

    para = None
    if paraphrase is not None:
        schema = paraphrase.get("schema", "")
        if not schema:
            raise CorpusError("%s: paraphrase entry has no schema" % template.template_id)
        para = template.render_paraphrase(inst.bindings, quantities, schema)
        _assert_paraphrase(template, nl, para, quantities, paraphrase)

    pid = problem_id or template.template_id
    record: Dict[str, Any] = {
        "problem_id": pid,
        "template_id": template.template_id,
        "seed": seed,
        "ground_truth": inst.answer,
        "quantities": [
            {
                "name": name,
                "value": _json_number(inst.bindings[name]),
                # The literals actually emitted, not the raw pool surface: a
                # sentence-initial slot is capitalised at emit time, so the raw
                # pool string would not match `text[start:end]`.
                "surface": nl.literals[name],
                "formal_literal": formal.literals[name],
            }
            for name in quantities
        ],
        "realizations": {
            "nl_base": _realization(nl, inst.answer),
            "nl_para": (
                PARA_PLACEHOLDER(inst.answer)
                if para is None
                else _realization(
                    para,
                    inst.answer,
                    status="written",
                    # Which surface operation this paraphrase applies. H2's
                    # surface-dissimilarity covariates are lexical AND
                    # syntactic, so a corpus of 100 paraphrases that all apply
                    # the same transformation gives those covariates no variance
                    # to work with (configs/corpus_schema.md).
                    paraphrase_op=paraphrase.get("op", "unspecified"),
                    source="data/corpus/paraphrases.json",
                )
            ),
            "formal": _realization(
                formal,
                inst.answer,
                # Amendment 4: givens + goal, no derivation. Not executable and
                # not meant to be -- it is input notation (DESIGN.md B.8).
                notation="givens_goal",
                goal=formal.meta["goal"],
                goal_nl=formal.meta["goal_nl"],
                given=formal.meta["given"],
                unknown=formal.meta["unknown"],
            ),
            "nl_break": _realization(
                brk,
                pert.answer,
                answer_preserving=False,
                ground_truth_base=inst.answer,
                perturbed_quantity=pert.quantity,
            ),
            # Amendment 6. Answer UNCHANGED in both -- that is the point: they
            # bracket the surface-perturbation floor that `nl_break` would
            # otherwise be measured against with nothing to compare to.
            "nl_minimal_orth": (
                _realization(
                    minimal_rendered["nl_minimal_orth"],
                    inst.answer,
                    status="written",
                    edit_kind="orthographic",
                    edited_quantity=minimal["nl_minimal_orth"]["variable"],
                    on_anchor=True,
                    matches_nl_break_site=minimal["nl_minimal_orth"][
                        "matches_nl_break_site"
                    ],
                )
                if "nl_minimal_orth" in minimal_rendered
                else UNAVAILABLE(inst.answer, minimal["nl_minimal_orth"]["reason"])
            ),
            "nl_minimal_lex": (
                _realization(
                    minimal_rendered["nl_minimal_lex"],
                    inst.answer,
                    status="written",
                    edit_kind="lexical_proper_name",
                    edited_variable=minimal["nl_minimal_lex"]["variable"],
                    on_anchor=False,
                )
                if "nl_minimal_lex" in minimal_rendered
                else UNAVAILABLE(inst.answer, minimal["nl_minimal_lex"]["reason"])
            ),
        },
        "perturbation": {
            "quantity": pert.quantity,
            "from_value": _json_number(pert.old_value),
            "to_value": _json_number(pert.new_value),
            "from_surface": pert.old_surface,
            "to_surface": pert.new_surface,
            "answer_base": inst.answer,
            "answer_break": pert.answer,
            "template_conditions_still_satisfied": pert.conditions_satisfied,
            "surface_edit_distance": pert.edit_distance,
            "occurrences_rewritten": pert.occurrences,
            "text_char_edit_distance": _levenshtein(nl.text, brk.text),
        },
        "minimal_edits": minimal,
        "answer_preserving": {
            "nl_base": True,
            "nl_para": True,
            "formal": True,
            "nl_break": False,
            "nl_minimal_orth": True,
            "nl_minimal_lex": True,
        },
        "linguistic_profile": linguistic_profile(nl.text, nl.anchors),
        "template_source": {
            "repo": (provenance or {}).get("repo"),
            "commit": (provenance or {}).get("commit"),
            "file": template.source_file,
            "gsm8k_id_orig": template.id_orig,
            "route": (provenance or {}).get("route"),
        },
    }
    validate_record(record)
    # Belt and braces: re-assert `text[start:end] == literal` against the
    # SERIALISED record, using the literals the renderers emitted. Rendered
    # .validate() already did this at render time; this repeats it after the
    # record has been assembled, so a bug in record assembly cannot slip a bad
    # span through.
    literals = {
        "nl_base": nl.literals,
        "formal": formal.literals,
        "nl_break": brk.literals,
    }
    if para is not None:
        literals["nl_para"] = para.literals
    for cond, rendered in minimal_rendered.items():
        literals[cond] = rendered.literals
    assert_span_literals(record, literals)
    return record


def _assert_paraphrase(
    template: Template,
    nl: Rendered,
    para: Rendered,
    quantities: Sequence[str],
    entry: Dict[str, Any],
) -> None:
    """Hold a paraphrase to the same standard as the generated realizations.

    Failing here is a BUILD FAILURE, not a warning: a paraphrase whose spans
    cannot be verified would put the wrong token into the aligned-position set
    for one condition of one problem, and nothing downstream would report it.

    What is checked:
      * every quantity is anchored (already fatal inside `Rendered.validate`);
      * the anchored literal is the same surface form the NL rendering used, so
        the two conditions really do anchor the same quantity introduction;
      * the paraphrase is not the NL rendering with the punctuation moved --
        some actual rewriting happened;
      * a paraphrase operation is declared, for the H2 covariate spread.
    """
    tid = template.template_id
    for name in quantities:
        if para.literals[name] != nl.literals[name]:
            raise SpanError(
                "%s/nl_para: quantity %r reads %r but nl_base emitted %r"
                % (tid, name, para.literals[name], nl.literals[name])
            )
    if para.text.strip() == nl.text.strip():
        raise CorpusError("%s: paraphrase is identical to nl_base" % tid)
    if not entry.get("op"):
        raise CorpusError("%s: paraphrase declares no surface operation" % tid)


def _assert_single_quantity_edit(
    template: Template,
    base_bindings: Dict[str, Any],
    pert: Perturbation,
    base_text: str,
    break_text: str,
) -> None:
    """Confirm nl_break rewrites the perturbed variable's slots and nothing else.

    Done over the segment list rather than by diffing the two strings, so this
    is a statement about how the text was BUILT, not an inference from how it
    looks.
    """
    changed_vars = [
        v
        for v in set(list(base_bindings) + list(pert.bindings))
        if base_bindings.get(v) != pert.bindings.get(v)
        or getattr(base_bindings.get(v), "surface", None)
        != getattr(pert.bindings.get(v), "surface", None)
    ]
    surfaced = [v for v in changed_vars if any(
        seg.kind == "ph" and seg.var == v for seg in template.segments
    )]
    if surfaced != [pert.quantity]:
        raise CorpusError(
            "%s: nl_break changes surfaced variables %r, expected only %r"
            % (template.template_id, sorted(surfaced), pert.quantity)
        )
    if base_text == break_text:
        raise CorpusError("%s: nl_break text identical to nl_base" % template.template_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_realization(
    record: Dict[str, Any], cond: str, quantities: Sequence[str]
) -> None:
    """Span/anchor invariants for one realization. Applied to all four."""
    r = record["realizations"][cond]
    text = r["text"]
    anchors = r["anchors"]
    if "FINAL" not in anchors:
        raise SpanError("%s/%s: FINAL sentinel missing" % (record["problem_id"], cond))
    if anchors["FINAL"] != [-1, -1]:
        raise SpanError(
            "%s/%s: FINAL sentinel must be [-1, -1]" % (record["problem_id"], cond)
        )
    if not text:
        raise CorpusError("%s/%s: empty text" % (record["problem_id"], cond))
    missing = [q for q in quantities if q not in anchors]
    if missing:
        raise SpanError(
            "%s/%s: quantities missing from anchors: %r"
            % (record["problem_id"], cond, missing)
        )
    for name, span in anchors.items():
        if name == "FINAL":
            continue
        start, end = span
        if not (0 <= start < end <= len(text)):
            raise SpanError(
                "%s/%s: span %r for %r out of bounds"
                % (record["problem_id"], cond, span, name)
            )
    # spans must be disjoint and increasing -- overlapping anchors would map
    # two quantities onto one token at extraction time
    ordered = sorted((tuple(s) for n, s in anchors.items() if n != "FINAL"))
    for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
        if s2 < e1:
            raise SpanError(
                "%s/%s: overlapping anchor spans %r and %r"
                % (record["problem_id"], cond, (s1, e1), (s2, e2))
            )


def validate_record(record: Dict[str, Any]) -> None:
    """Hard invariants. Every failure is fatal -- never emit a bad span."""
    quantities = [q["name"] for q in record["quantities"]]
    reals = record["realizations"]

    for cond in CONDITIONS:
        if cond not in reals:
            raise CorpusError("%s: missing realization %r" % (record["problem_id"], cond))

    # configs/corpus_schema.md: "every quantity key appears in the anchors of
    # every realization that has text". While `nl_para` was pending that
    # invariant could not be evaluated, and `validate_record` said so rather than
    # quietly passing. A written paraphrase -- and each of Amendment 6's style
    # controls, where the instance supports it -- is held to exactly the same
    # standard as the three generated conditions.
    conditions = list(GENERATED_CONDITIONS)
    para = reals["nl_para"]
    status = para.get("status")
    if status == "written":
        conditions.append("nl_para")
    elif status == "pending":
        if para.get("text") or para.get("anchors"):
            raise CorpusError(
                "%s: nl_para marked pending but not empty" % record["problem_id"]
            )
    else:
        raise CorpusError(
            "%s: nl_para has unknown status %r" % (record["problem_id"], status)
        )

    # Amendment 6's style controls are held to exactly the same standard as the
    # generated conditions when they are present, and must be a genuinely empty
    # hole -- no text, no anchors -- when they are not.
    for cond in MINIMAL_CONDITIONS:
        real = reals[cond]
        st = real.get("status")
        if st == "written":
            conditions.append(cond)
            if real.get("answer") != record["ground_truth"]:
                raise CorpusError(
                    "%s: %s is answer-preserving but records %r, ground truth is %r"
                    % (record["problem_id"], cond, real.get("answer"),
                       record["ground_truth"])
                )
        elif st == "unavailable":
            if real.get("text") or real.get("anchors"):
                raise CorpusError(
                    "%s: %s marked unavailable but not empty" % (record["problem_id"], cond)
                )
            if not real.get("note"):
                raise CorpusError(
                    "%s: %s is unavailable with no reason given"
                    % (record["problem_id"], cond)
                )
        else:
            raise CorpusError(
                "%s: %s has unknown status %r" % (record["problem_id"], cond, st)
            )

    for cond in conditions:
        _validate_realization(record, cond, quantities)

    if status == "written" and para.get("answer") != record["ground_truth"]:
        raise CorpusError(
            "%s: nl_para is answer-preserving but records %r, ground truth is %r"
            % (record["problem_id"], para.get("answer"), record["ground_truth"])
        )

    if reals["nl_break"]["answer"] == record["ground_truth"]:
        raise CorpusError("%s: nl_break answer did not change" % record["problem_id"])
    if reals["formal"]["answer"] != record["ground_truth"]:
        raise CorpusError("%s: formal answer differs from ground truth" % record["problem_id"])


_FORMAL_LITERAL_RE = re.compile(r"^(?:-?\d+|-?\d+\.\d+|Fraction\(-?\d+, -?\d+\))$")


def _literal_value(literal: str) -> Fraction:
    """Exact value of a formal givens literal. Raises on anything else."""
    m = re.match(r"^Fraction\((-?\d+), (-?\d+)\)$", literal)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    if not _FORMAL_LITERAL_RE.match(literal):
        raise CorpusError("not a bare numeric literal: %r" % literal)
    return Fraction(literal)


def assert_formal_givens(
    template: Template,
    formal: Rendered,
    bindings: Dict[str, Any],
    quantities: Sequence[str],
) -> None:
    """Retained check #1 (Amendment 4): the givens are values, not a program.

    Every non-comment line of `formal` must be either `<quantity> = <bare
    numeric literal>` -- with the literal's VALUE equal to the sampled value it
    claims -- or the single `answer = ?` line declaring the unknown. Anything
    with an operator in it would be a derivation, which is exactly what
    Amendment 4 removed, so this is a crash rather than a warning.
    """
    tid = template.template_id
    bound: List[str] = []
    saw_unknown = 0
    for line in formal.text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, sep, rhs = line.partition(" = ")
        if not sep or not name.isidentifier():
            raise CorpusError("%s: formal line is not an assignment: %r" % (tid, line))
        if name == "answer":
            if rhs != "?":
                raise CorpusError(
                    "%s: formal binds the answer (%r) -- that is the derivation, "
                    "not the notation (Amendment 4)" % (tid, rhs)
                )
            saw_unknown += 1
            continue
        if name not in quantities:
            raise CorpusError(
                "%s: formal binds %r, which is not a quantity of the problem"
                % (tid, name)
            )
        try:
            value = _literal_value(rhs)
        except CorpusError as exc:
            raise CorpusError("%s: given %r: %s" % (tid, name, exc))
        bound_value = bindings[name]
        # Exact for ints and Fractions; a float is compared as the decimal it
        # was printed as, because Fraction(62.1) is not Fraction("62.1").
        same = (
            float(value) == float(bound_value)
            if isinstance(bound_value, float)
            else value == Fraction(bound_value)
        )
        if not same:
            raise CorpusError(
                "%s: given %r = %s but the sampled value is %r"
                % (tid, name, rhs, bound_value)
            )
        bound.append(name)
    if bound != list(quantities):
        raise CorpusError(
            "%s: formal givens are %r, quantities are %r" % (tid, bound, list(quantities))
        )
    if saw_unknown != 1:
        raise CorpusError("%s: formal must declare `answer = ?` exactly once" % tid)
    # The derivation must not appear as text either. Only checked for compound
    # expressions: three templates write `#answer: ans` -- a bare scratch
    # variable, which is never emitted (only quantities are bound) and whose
    # name is a substring of the word "answer".
    expr = re.sub(r"\s+", "", template.answer_expr)
    if re.search(r"[-+*/%()]", expr) and expr in re.sub(r"\s+", "", formal.text):
        raise CorpusError("%s: formal contains the template's answer expression" % tid)
    leaked = [v for v in template.free_vars if v in bindings and v in bound]
    if leaked:
        raise CorpusError("%s: formal binds solved-for scratch variables %r" % (tid, leaked))


def assert_formal_goal(template: Template, formal: Rendered, nl: Rendered) -> None:
    """Retained check #2 (Amendment 4): the goal names what the template asks for.

    Operationally: the goal `formal` emitted is, verbatim, the final sentence of
    the natural-language rendering of the SAME instance -- so the two conditions
    ask the identical question -- and the quantity it asks for is declared as the
    unbound symbol `answer`.
    """
    tid = template.template_id
    goal_nl = formal.meta.get("goal_nl", "")
    if not goal_nl:
        raise CorpusError("%s: formal recorded no goal" % tid)
    if not nl.text.endswith(goal_nl):
        raise CorpusError(
            "%s: formal goal %r is not the final sentence of nl_base" % (tid, goal_nl)
        )
    head = nl.text[: len(nl.text) - len(goal_nl)]
    if head and not re.search(r"[.!?]\s+$", head):
        raise CorpusError("%s: formal goal is not a whole sentence of nl_base" % tid)
    if not _asks_a_question(goal_nl):
        raise CorpusError("%s: formal goal asks nothing: %r" % (tid, goal_nl))
    if formal.meta.get("unknown") != "answer" or not formal.text.endswith("answer = ?"):
        raise CorpusError("%s: formal does not declare the unknown last" % tid)


def _asks_a_question(sentence: str) -> bool:
    low = sentence.lower()
    return "?" in sentence or any(
        re.search(r"\b%s\b" % v, low) for v in _IMPERATIVE
    )


def assert_span_literals(record: Dict[str, Any], literals: Dict[str, Dict[str, str]]) -> None:
    """Re-assert `text[start:end] == literal` given the emitted literals."""
    for cond, lits in literals.items():
        text = record["realizations"][cond]["text"]
        for name, literal in lits.items():
            start, end = record["realizations"][cond]["anchors"][name]
            if text[start:end] != literal:
                raise SpanError(
                    "%s/%s: anchor %r claims %r but text holds %r"
                    % (record["problem_id"], cond, name, literal, text[start:end])
                )


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------


def problem_rng(seed: int, template_id: str) -> random.Random:
    """Per-problem RNG derived deterministically from the master seed.

    Derived rather than shared so that adding, dropping or reordering templates
    cannot change the values sampled for any other problem.
    """
    digest = hashlib.sha256(("%d|%s" % (seed, template_id)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


DEFAULT_PARAPHRASE_BUNDLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "corpus", "paraphrases.json",
)


def load_paraphrases(
    path: str = DEFAULT_PARAPHRASE_BUNDLE, required: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load the vendored `nl_para` schemas. Returns (provenance, by problem_id).

    The paraphrases are a static authored asset rather than something generated
    at build time -- they are the one condition a template cannot emit, and
    DESIGN.md A.5 wants them answer-preserving, which is a judgement no
    generator of ours makes. They are stored as SCHEMAS, with `{var}`
    placeholders naming the template's own variables, not as finished strings:
    that keeps them seed-independent (the same file renders correctly under any
    `--seed`) and it is what lets the quantity spans be emitted at substitution
    time instead of searched for in finished prose.

    A missing file yields an empty mapping, and every `nl_para` slot then stays
    the honest pending placeholder. `required=True` turns that into an error for
    callers that must have the paraphrase condition.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
    except FileNotFoundError:
        if required:
            raise CorpusError("paraphrase bundle not found: %s" % path)
        return {}, {}
    return bundle.get("_provenance", {}), bundle.get("paraphrases", {})


def build_corpus(
    templates: Sequence[Template],
    seed: int,
    n_problems: int = 100,
    provenance: Optional[Dict[str, Any]] = None,
    paraphrases: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Build up to `n_problems` records, one per template, in template order.

    Returns (records, failures). Failures are reported, never silently skipped:
    DESIGN.md A.5 wants the drop rate logged.

    `paraphrases` defaults to the vendored bundle. If the bundle is present but
    has no entry for a template, that problem FAILS the build -- a corpus with
    three conditions for 99 problems and four for one is worse than a corpus
    that says which problem it could not paraphrase.
    """
    if paraphrases is None:
        _, paraphrases = load_paraphrases()
    records: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []
    for template in templates:
        if len(records) >= n_problems:
            break
        rng = problem_rng(seed, template.template_id)
        try:
            entry = paraphrases.get(template.template_id) if paraphrases else None
            if paraphrases and entry is None:
                raise CorpusError(
                    "%s: no paraphrase in the bundle" % template.template_id
                )
            records.append(
                build_problem(
                    template,
                    rng,
                    seed=seed,
                    provenance=provenance,
                    paraphrase=entry,
                )
            )
        except (CorpusError, TemplateError, SpanError) as exc:
            failures.append((template.template_id, "%s: %s" % (type(exc).__name__, exc)))
    return records, failures


def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    """Byte-deterministic JSONL write: sorted keys, LF newlines, UTF-8."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            fh.write("\n")
