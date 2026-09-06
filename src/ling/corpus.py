"""Corpus construction: 100 problems x 4 realizations (configs/corpus_schema.md).

Three of the four conditions are emitted here, directly from the parameterised
symbolic templates in `ling.templates`:

  * ``nl_base``  -- natural-language rendering, freshly sampled values, ground
                    truth COMPUTED by executing the template's own `#answer:`
                    expression. No model in the loop.
  * ``formal``   -- Python rewrite of the SAME instance. Never executed as part
                    of the experiment (DESIGN.md B.8: our formal condition is
                    input notation with no execution); executed once HERE, at
                    build time, as a check that it really does return ground
                    truth.
  * ``nl_break`` -- exactly ONE quantity perturbed so the computed answer
                    changes, chosen to keep the surface as close to `nl_base`
                    as the template allows. Both answers are recorded and the
                    change is asserted.

The fourth, ``nl_para``, is left as an explicit empty placeholder with anchors
NOT yet determined; a separate pass writes it. See `PARA_PLACEHOLDER`.

WHY THE ANCHOR SPANS ARE TRUSTWORTHY
------------------------------------
Spans are *emitted*, never recovered. `Template.render_nl` / `render_python`
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
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .templates import (
    Instance,
    Rendered,
    SpanError,
    Template,
    TemplateError,
    _wrap,
    is_numeric,
    normalise_answer,
)

__all__ = [
    "CONDITIONS",
    "GENERATED_CONDITIONS",
    "PARA_PLACEHOLDER",
    "CorpusError",
    "build_problem",
    "build_corpus",
    "linguistic_profile",
    "validate_record",
]

CONDITIONS = ("nl_base", "nl_para", "formal", "nl_break")
GENERATED_CONDITIONS = ("nl_base", "formal", "nl_break")

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
    "every quantity appears in all four realizations" check in
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
) -> Dict[str, Any]:
    """Build one corpus record: three generated realizations plus the para slot."""
    inst = template.sample(rng)
    quantities = inst.quantities
    if not quantities:
        raise CorpusError("%s: instance has no anchorable quantity" % template.template_id)

    nl = template.render_nl(inst.bindings, quantities)
    formal = template.render_python(inst.bindings, quantities)

    # formal is NEVER executed during the experiment (DESIGN.md B.8). Executing
    # it once here is the verification that the rewrite is faithful.
    ns: Dict[str, Any] = {}
    try:
        exec(compile(formal.text, "<formal:%s>" % template.template_id, "exec"), ns)  # noqa: S102
    except Exception as exc:  # pragma: no cover - build-time gate
        raise CorpusError(
            "%s: formal rewrite is not executable: %r" % (template.template_id, exc)
        )
    if "answer" not in ns or normalise_answer(ns["answer"]) != inst.answer:
        raise CorpusError(
            "%s: formal rewrite returns %r, ground truth is %r"
            % (template.template_id, ns.get("answer"), inst.answer)
        )

    pert = find_perturbation(inst)
    if pert.answer == inst.answer:  # belt and braces
        raise CorpusError("%s: perturbation did not change the answer" % template.template_id)
    brk = template.render_nl(pert.bindings, quantities)

    # nl_break must differ from nl_base ONLY where the perturbed variable is
    # placed. Checked structurally against the segment list, not by diffing the
    # finished strings.
    _assert_single_quantity_edit(template, inst.bindings, pert, nl.text, brk.text)

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
            "nl_para": PARA_PLACEHOLDER(inst.answer),
            "formal": _realization(formal, inst.answer, notation="python"),
            "nl_break": _realization(
                brk,
                pert.answer,
                answer_preserving=False,
                ground_truth_base=inst.answer,
                perturbed_quantity=pert.quantity,
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
        "answer_preserving": {
            "nl_base": True,
            "nl_para": True,
            "formal": True,
            "nl_break": False,
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
    assert_span_literals(
        record,
        {
            "nl_base": nl.literals,
            "formal": formal.literals,
            "nl_break": brk.literals,
        },
    )
    return record


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


def validate_record(record: Dict[str, Any]) -> None:
    """Hard invariants. Every failure is fatal -- never emit a bad span."""
    quantities = [q["name"] for q in record["quantities"]]
    reals = record["realizations"]

    for cond in CONDITIONS:
        if cond not in reals:
            raise CorpusError("%s: missing realization %r" % (record["problem_id"], cond))

    for cond in GENERATED_CONDITIONS:
        r = reals[cond]
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
                    "%s/%s: span %r for %r out of bounds" % (record["problem_id"], cond, span, name)
                )
        # spans must be disjoint and increasing -- overlapping anchors would map
        # two quantities onto one token at extraction time
        ordered = sorted(
            (tuple(s) for n, s in anchors.items() if n != "FINAL"),
        )
        for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
            if s2 < e1:
                raise SpanError(
                    "%s/%s: overlapping anchor spans %r and %r"
                    % (record["problem_id"], cond, (s1, e1), (s2, e2))
                )

    para = reals["nl_para"]
    if para.get("status") != "pending" or para.get("text") or para.get("anchors"):
        # If a later pass has filled it, hold it to the same standard.
        if para.get("status") == "pending":
            raise CorpusError("%s: nl_para marked pending but not empty" % record["problem_id"])
        missing = [q for q in quantities if q not in para.get("anchors", {})]
        if missing:
            raise SpanError(
                "%s/nl_para: quantities missing from anchors: %r"
                % (record["problem_id"], missing)
            )

    if reals["nl_break"]["answer"] == record["ground_truth"]:
        raise CorpusError("%s: nl_break answer did not change" % record["problem_id"])
    if reals["formal"]["answer"] != record["ground_truth"]:
        raise CorpusError("%s: formal answer differs from ground truth" % record["problem_id"])


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


def build_corpus(
    templates: Sequence[Template],
    seed: int,
    n_problems: int = 100,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Build up to `n_problems` records, one per template, in template order.

    Returns (records, failures). Failures are reported, never silently skipped:
    DESIGN.md A.5 wants the drop rate logged.
    """
    records: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []
    for template in templates:
        if len(records) >= n_problems:
            break
        rng = problem_rng(seed, template.template_id)
        try:
            records.append(
                build_problem(
                    template, rng, seed=seed, provenance=provenance
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
