"""Parameterised symbolic problem templates.

=============================================================================
TEMPLATE ROUTE: **OFFICIAL**  (not freshly authored)
=============================================================================
The 100 templates behind this corpus are the *official* GSM-Symbolic templates
of Mirzadeh et al., ICLR 2025 (arXiv:2410.05229), fetched on 2026-09-06 from

    https://github.com/apple/ml-gsm-symbolic
    templates/symbolic/0000.json .. 0099.json
    commit b6a1625025fc857300203bac9f617e5d8ec99f65 (2024-12-08)

and vendored verbatim (the `question_annotated` field only) into

    data/templates/gsm_symbolic_templates.json

so that corpus generation has **no network dependency at run time**.

This is the route DESIGN.md A.5 asks for: "Build NL-base from GSM-Symbolic
templates (Mirzadeh et al., ICLR 2025) with freshly sampled instantiations."
We did NOT author substitute templates and we did NOT ship raw GSM8K text: the
vendored bundle deliberately drops each template's `question`/`answer` fields,
which are verbatim GSM8K instances, and keeps only the symbolic form.

WHAT IS OURS, AND THEREFORE WHAT A REVIEWER SHOULD SCRUTINISE
-------------------------------------------------------------
Apple released the templates but **not** their parser/data generator (their
README says so explicitly), and the template DSL references lexical pools by
bare name (`names`, `weights_med`, `fractions`, ...) whose contents were never
published. So:

  * The DSL interpreter below (`Template.parse` / `Template.sample`) is a
    reimplementation, not Apple's code.
  * `WORD_POOLS` is **reconstructed**. Each pool's contents were inferred from
    (a) the in-template default values -- `{unit, pound}` tells you `pound` is
    in `weights_med` -- and (b) the official generated instances published
    alongside the templates. They are plausible, not identical.

The reconstruction touches **surface nouns only**. Every numeric pool
(`range(...)`, `numbers_within(...)`, `np.arange(...)`), every `#conditions:`
clause and every `#answer:` expression is taken verbatim from Apple's template,
so ground truth is computed by executing *their* answer expression on *our*
freshly sampled values. There is no model in the loop and nothing to
hallucinate.

DESIGN NOTES THAT MATTER DOWNSTREAM
-----------------------------------
* **Spans are emitted, never searched.** `render_nl` / `render_paraphrase` /
  `render_formal` build the output string by appending chunks and record
  `[start, end)` at the moment each quantity's surface form is appended.
  `str.find` is never used to recover a span. `Rendered.validate()` re-asserts
  `text[start:end] == literal`.
* **`formal` is givens + goal, never a program.** prereg/DEVIATIONS.md
  Amendment 4: `render_formal` emits the value bindings, the situation
  statements with quantities lifted to their identifiers, and the template's own
  question with the unknown declared as `answer = ?`. No derivation, no answer
  expression, not executable.
* **A "quantity"** is a template variable that (a) appears as a placeholder in
  the question body and (b) carries a numeric value. Lexical variables (names,
  colours, units) are not quantities. Solved-for internal variables that never
  surface in the question (`ans`, `x` in a couple of templates) are not
  quantities either -- they have no character position to anchor.
* Numeric values are wrapped in `IntQ`/`FloatQ`/`FracQ`, which are `int`/`float`/
  `Fraction` subclasses that additionally carry the surface string and support
  `v[0]` / `v[1]` indexing. Both are needed: several Apple templates use a
  variable arithmetically in `#conditions:` but index it as a `(surface, value)`
  pair in `#answer:`.

Pure standard library. No numpy, no network.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Template",
    "Instance",
    "Rendered",
    "SpanError",
    "TemplateError",
    "load_templates",
    "DEFAULT_TEMPLATE_BUNDLE",
    "normalise_answer",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEMPLATE_BUNDLE = os.path.join(
    _REPO_ROOT, "data", "templates", "gsm_symbolic_templates.json"
)


class TemplateError(RuntimeError):
    """Raised when a template cannot be parsed or instantiated."""


class SpanError(AssertionError):
    """Raised when a recorded anchor span does not contain the literal it claims.

    This is deliberately fatal. A silently wrong anchor span means the
    activation extractor clusters the wrong token, and every downstream number
    is plausible-looking garbage (DESIGN.md A.11).
    """


# ---------------------------------------------------------------------------
# Numeric values that also remember their surface form
# ---------------------------------------------------------------------------
#
# `{n2, five}` with `$n2 = numbers_within(5, 10)` renders as an English word but
# must behave as an integer in `#conditions:` and `#answer:`. `{sides,
# six-sided}` with `sample([("six-sided", 6), ...])` is the same problem. And
# templates 0057/0095 write `mult[1]` in `#answer:` while writing `n * mult` in
# `#conditions:`, so the same object has to support both.


class _SurfaceMixin:
    __slots__ = ()

    def __getitem__(self, i):  # (surface, value) pair emulation
        if i == 0:
            return self.surface
        if i == 1:
            return self._plain()
        raise IndexError(i)


class IntQ(int, _SurfaceMixin):

    def __new__(cls, value, surface=None):
        obj = int.__new__(cls, value)
        obj.surface = str(int(value)) if surface is None else str(surface)
        return obj

    def _plain(self):
        return int(self)


class FloatQ(float, _SurfaceMixin):

    def __new__(cls, value, surface=None):
        obj = float.__new__(cls, value)
        obj.surface = _fmt_float(float(value)) if surface is None else str(surface)
        return obj

    def _plain(self):
        return float(self)


class FracQ(Fraction, _SurfaceMixin):

    def __new__(cls, value, surface=None):
        obj = super().__new__(cls, value)
        obj.surface = str(Fraction(value)) if surface is None else str(surface)
        return obj

    def _plain(self):
        return Fraction(self.numerator, self.denominator)


def _fmt_float(x: float) -> str:
    s = "%.10f" % x
    s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _wrap(value: Any) -> Any:
    """Wrap a sampled pool element into a surface-carrying value."""
    if isinstance(value, tuple) and len(value) == 2:
        surface, num = value[0], value[1]
    else:
        surface, num = None, value
    if isinstance(num, bool) or not isinstance(num, (int, float, Fraction)):
        return value
    if isinstance(num, Fraction):
        return FracQ(num, surface)
    if isinstance(num, int):
        return IntQ(num, surface)
    return FloatQ(num, surface)


def is_numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, Fraction))


# ---------------------------------------------------------------------------
# DSL builtins
# ---------------------------------------------------------------------------

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Cardinal in words. Matches the official generated data ('thirty two')."""
    n = int(n)
    if n < 0:
        return "negative " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ((" " + _ONES[ones]) if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ((" " + number_to_words(rest)) if rest else "")
    thousands, rest = divmod(n, 1000)
    return (
        number_to_words(thousands)
        + " thousand"
        + ((" " + number_to_words(rest)) if rest else "")
    )


def _frange(start: float, stop: float, step: float = 1.0) -> List[float]:
    out: List[float] = []
    x = float(start)
    step = float(step)
    if step == 0:
        raise TemplateError("frange step 0")
    n = 0
    while (x < stop - 1e-12) if step > 0 else (x > stop + 1e-12):
        out.append(round(x, 10))
        n += 1
        x = float(start) + n * step
        if n > 100000:
            raise TemplateError("frange runaway")
    return out


def _numbers_within(lo: int, hi: int) -> List[Tuple[str, int]]:
    """Inclusive integer range rendered as English words.

    Verified against the official generated data: `numbers_within(31, 35)`
    produces surfaces 'thirty one' / 'thirty two' etc.
    """
    return [(number_to_words(n), n) for n in range(int(lo), int(hi) + 1)]


def _divides(a, b) -> bool:
    try:
        if a == 0 or b == 0:
            return False
        return (Fraction(a) % Fraction(b) == 0) or (Fraction(b) % Fraction(a) == 0)
    except Exception:
        return False


def _is_int(x) -> bool:
    try:
        f = Fraction(x)
    except Exception:
        try:
            f = Fraction(float(x)).limit_denominator(10**6)
        except Exception:
            return False
    return f.denominator == 1


def _fix_floats(xs) -> List[float]:
    return [round(float(x), 2) for x in xs]


def _shuffle_list(xs):
    # Order-preserving on purpose: the caller draws with a seeded RNG, so an
    # extra shuffle would only add a hidden RNG consumption and break
    # reproducibility across refactors.
    return list(xs)


class _NpRandomShim:
    @staticmethod
    def randint(low, high, size=None):
        # Used only as a *pool* (`$n1 = np.random.randint(1000, 3000, 50)`), so
        # the honest deterministic reading is the full candidate range.
        return list(range(int(low), int(high)))


class _NpShim:
    arange = staticmethod(_frange)
    random = _NpRandomShim


# ---------------------------------------------------------------------------
# Reconstructed lexical pools  (see module docstring -- these are OURS)
# ---------------------------------------------------------------------------

_NAMES_MALE = [
    "James", "Robert", "John", "Michael", "David", "Ali", "Jamal", "Diego",
    "Omar", "Wei", "Ravi", "Luis", "Ahmed", "Kai", "Noah", "Ethan", "Mateo",
    "Hiroshi", "Tomas", "Andre", "Marcus", "Pavel", "Sanjay", "Elias",
]
_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elise", "Olivia", "Rania",
    "Aisha", "Mei", "Priya", "Sofia", "Fatima", "Yuki", "Nadia", "Ines",
    "Amara", "Camila", "Leila", "Anya", "Zara", "Greta", "Nour", "Thandiwe",
]

WORD_POOLS: Dict[str, List[Any]] = {
    "names": _NAMES_MALE + _NAMES_FEMALE,
    "names_male": _NAMES_MALE,
    "names_female": _NAMES_FEMALE,
    "cities": ["Springfield", "Riverton", "Fairview", "Bristol", "Oakdale", "Lakeside"],
    "weekdays": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange"],
    # Singular: the templates pluralise in the literal text ("{fruit,\n    # pineapple}s per {area,hectare}"), so a plural pool yields "peachess".
    "fruits": ["apple", "banana", "orange", "pear", "plum", "peach", "pineapple"],
    "sports": ["soccer", "basketball", "tennis", "hockey", "cricket", "volleyball"],  # singular, used bare
    "currencies_sym": ["$", "€", "£", "¥"],
    # Unit pools: the members observed in the official generated data are kept
    # first so that the common case matches published instances exactly.
    "weights_med": ["pound", "kg", "kilogram"],
    "weights_sm": ["ounce", "gram"],
    "length_lg": ["mile", "km", "kilometer"],
    # (surface, value) pools. Values are exact Fractions so that the templates'
    # `is_int(n * frac)` conditions behave, rather than drifting on 1/3.
    # Surfaces carry their determiner because the templates use them bare
    # ("{frac_1, half} of the balls are ..."), where "Fifth of the balls" would
    # be ungrammatical but "A fifth of the balls" is not.
    "fractions": [
        ("half", Fraction(1, 2)),
        ("a third", Fraction(1, 3)),
        ("a quarter", Fraction(1, 4)),
        ("a fifth", Fraction(1, 5)),
        ("a sixth", Fraction(1, 6)),
    ],
    "fraction_alph": [
        ("a half", Fraction(1, 2)),
        ("a third", Fraction(1, 3)),
        ("a quarter", Fraction(1, 4)),
        ("a fifth", Fraction(1, 5)),
    ],
    "fraction_nums": [
        ("1/2", Fraction(1, 2)),
        ("1/3", Fraction(1, 3)),
        ("1/4", Fraction(1, 4)),
        ("1/5", Fraction(1, 5)),
        ("2/3", Fraction(2, 3)),
        ("3/4", Fraction(3, 4)),
    ],
    "fraction_decimals": [
        ("0.25", Fraction(1, 4)),
        ("0.5", Fraction(1, 2)),
        ("0.75", Fraction(3, 4)),
        ("0.2", Fraction(1, 5)),
    ],
    "fraction_alnum": [
        ("half", Fraction(1, 2)),
        ("1/3", Fraction(1, 3)),
        ("a quarter", Fraction(1, 4)),
        ("1/5", Fraction(1, 5)),
    ],
    "multi_times": [
        ("twice", 2),
        ("three times", 3),
        ("four times", 4),
        ("five times", 5),
    ],
    "multiple": [("double", 2), ("triple", 3), ("quadruple", 4)],
    "multiple_ice": [("double", 2), ("triple", 3)],
    "hiragana": ["hiragana"],
    "alphabet": ["alphabet"],
    "abjad": ["abjad"],
}


def _eval_namespace() -> Dict[str, Any]:
    ns: Dict[str, Any] = dict(WORD_POOLS)
    ns.update(
        {
            "range": lambda *a: list(range(*a)),
            "list": list,
            "int": int,
            "float": float,
            "str": str,
            "round": round,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "sorted": sorted,
            "Fraction": Fraction,
            "frange": _frange,
            "arange": _frange,
            "fix_floats": _fix_floats,
            "shuffle_list": _shuffle_list,
            "numbers_within": _numbers_within,
            "divides": _divides,
            "is_int": _is_int,
            "np": _NpShim,
            "it": lambda x: x,
        }
    )
    return ns


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------


def normalise_answer(value: Any) -> Optional[float]:
    """Coerce a raw `#answer:` result to a JSON-safe int/float, or None.

    Returns None for anything that is not a clean number: a non-numeric result,
    a non-finite float, or a value that is neither an integer nor exact to two
    decimal places (money). Callers resample rather than shipping a problem
    whose "ground truth" is 6.333333333333333.
    """
    if isinstance(value, (set, frozenset)):
        # Template 0017 writes `#answer: {expr}` -- a Python set literal, not a
        # brace-delimited placeholder. Unwrap a singleton.
        if len(value) != 1:
            return None
        value = next(iter(value))
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        return None
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return int(value)
        value = float(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        r2 = round(value, 2)
        if abs(value - r2) > 1e-9:
            return None
        if abs(r2 - round(r2)) < 1e-9:
            return int(round(r2))
        return r2
    return int(value)


# ---------------------------------------------------------------------------
# Template structure
# ---------------------------------------------------------------------------


_SENTENCE_END_RE = re.compile(r"(?:^|[.!?:]|\n)\s*$")
#: A full stop after one of these is an abbreviation, not a sentence boundary.
#: Three templates say "Mr./Mrs./Ms. <name>", and naive splitting there makes
#: the "final sentence" be "Johnson's budget?" -- which would silently give the
#: formal rendering a truncated goal.
_ABBREV_BEFORE_RE = re.compile(
    r"(?:^|[\s(\"'])(?:Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|vs|etc|No|Inc|Ltd|[A-Z])\.$"
)
_ARTICLE_RE = re.compile(r"(?<![A-Za-z])([Aa])n?(\s+)$")
#: Words whose written initial letter disagrees with their initial sound.
_AN_DESPITE_CONSONANT = ("hour", "honest", "honou", "honor", "heir")
_A_DESPITE_VOWEL = ("one", "once", "uni", "use", "usu", "euro", "ewe", "ubiq")


def _needs_an(word: str) -> bool:
    w = word.lstrip("\"'(").lower()
    if not w:
        return False
    if w.startswith(_AN_DESPITE_CONSONANT):
        return True
    if w.startswith(_A_DESPITE_VOWEL):
        return False
    return w[0] in "aeiou"


def _fix_article(literal: str, following: str) -> str:
    """Make a trailing indefinite article agree with the word about to follow.

    Applied while the literal is being emitted and BEFORE any span is recorded
    for the placeholder that follows it, so the recorded offsets always describe
    the final string. The templates hard-code "a", which yields "a agent fee" or
    "a hour" once a sampled noun lands there.
    """
    m = _ARTICLE_RE.search(literal)
    if not m or not following:
        return literal
    want = "an" if _needs_an(following) else "a"
    if m.group(1).isupper():
        want = want.capitalize()
    return literal[: m.start()] + want + m.group(2)


def _starts_sentence(parts: Sequence[str]) -> bool:
    """True if the text emitted so far leaves us at a sentence boundary."""
    tail = "".join(parts[-4:]) if parts else ""
    if not tail.strip():
        return True
    return bool(_SENTENCE_END_RE.search(tail))


@dataclass(frozen=True)
class Segment:
    """One chunk of the question body: literal prose or a placeholder."""

    kind: str  # "lit" | "ph"
    text: str = ""  # kind == "lit"
    var: str = ""  # kind == "ph"
    default: str = ""  # kind == "ph"


_PLACEHOLDER_RE = re.compile(r"\{\s*([A-Za-z_]\w*)\s*,([^{}]*)\}")
#: Paraphrase schemas (data/corpus/paraphrases.json) carry no defaults -- the
#: value comes from the same bindings the NL rendering used -- so their
#: placeholders are just `{var}`.
_PARA_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]\w*)\}")
_SECTION_RE = re.compile(r"\n\s*#\s*(init|conditions|answer)\s*[:=]", re.IGNORECASE)


@dataclass
class VarSpec:
    """One `#init:` binding: its targets, its pool, and how many to draw."""

    targets: List[str]
    pool: List[Any]
    draw: str  # "one" | "many" | "sequential"
    #: whether the template marked this binding with the `$` sigil. Recorded for
    #: provenance only: quantity detection deliberately goes by the bound
    #: VALUE's type instead, because several templates (0023, 0026, 0050) omit
    #: the sigil on slots that are plainly numeric and whose positions we still
    #: need to anchor.
    declared_numeric: bool


@dataclass
class Rendered:
    """A realisation's text plus the anchor spans emitted while building it."""

    text: str
    anchors: Dict[str, List[int]]
    literals: Dict[str, str]
    #: Rendering-specific facts the record builder verifies (`render_formal`
    #: records the goal it emitted, in both notations, so `corpus.py` can check
    #: it against the NL rendering's question sentence).
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self, required: Sequence[str]) -> None:
        """Re-assert every span. Crash loudly rather than emit a bad record."""
        for name, span in self.anchors.items():
            if name == "FINAL":
                if span != [-1, -1]:
                    raise SpanError("FINAL sentinel must be [-1, -1], got %r" % (span,))
                continue
            start, end = span
            if not (0 <= start < end <= len(self.text)):
                raise SpanError(
                    "span %r for %r out of bounds (len=%d)" % (span, name, len(self.text))
                )
            literal = self.literals[name]
            got = self.text[start:end]
            if got != literal:
                raise SpanError(
                    "anchor %r claims %r at %r but text holds %r" % (name, literal, span, got)
                )
        if "FINAL" not in self.anchors:
            raise SpanError("FINAL sentinel anchor missing")
        missing = [q for q in required if q not in self.anchors]
        if missing:
            raise SpanError("quantities missing from anchors: %r" % (missing,))


def _emit(
    segments: Sequence[Segment], chunks: Sequence[str], quantities: Sequence[str]
) -> Rendered:
    """Build the string and record spans AS EACH CHUNK IS APPENDED.

    `start` is the length of the text emitted so far, `end` is that plus the
    length of the surface form just appended. Nothing is ever located by
    searching the finished string, so a quantity whose surface happens to repeat
    elsewhere in the prose cannot be mis-anchored. Shared by every realisation
    that has prose in it (`render_nl`, `render_paraphrase`) so that all of them
    anchor by the same mechanism rather than by three lookalike copies of it.
    """
    parts: List[str] = []
    pos = 0
    anchors: Dict[str, List[int]] = {}
    literals: Dict[str, str] = {}
    wanted = set(quantities)

    for seg, chunk in zip(segments, chunks):
        start = pos
        parts.append(chunk)
        pos += len(chunk)
        if seg.kind == "ph" and seg.var in wanted and seg.var not in anchors:
            anchors[seg.var] = [start, pos]
            literals[seg.var] = chunk

    text = "".join(parts)
    assert pos == len(text)  # emitted-length bookkeeping must match reality
    anchors["FINAL"] = [-1, -1]
    r = Rendered(text=text, anchors=anchors, literals=literals)
    r.validate(quantities)
    return r


@dataclass
class Instance:
    """One sampled instantiation of a template."""

    template: "Template"
    bindings: Dict[str, Any]
    quantities: List[str]  # ordered by first appearance in the question body
    answer: float


@dataclass
class Template:
    """A parameterised symbolic problem.

    Carries symbolic variable names, declared sampling pools with constraints,
    an `answer()` that COMPUTES the result, and two surface forms (NL, Python).
    """

    template_id: str
    id_orig: Optional[int]
    source_file: str
    question_annotated: str
    body: str
    segments: List[Segment]
    specs: List[VarSpec]
    conditions: List[str]
    answer_expr: str
    placeholder_vars: List[str] = field(default_factory=list)  # first-appearance order

    # -- parsing ------------------------------------------------------------

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "Template":
        qa = raw["question_annotated"]
        body, init_lines, cond_lines, answer_expr = cls._split_sections(qa)
        segments, ph_order = cls._segment(body)
        specs = [cls._parse_init_line(ln) for ln in init_lines]
        specs = [s for s in specs if s is not None]
        if not answer_expr:
            raise TemplateError("%s: no #answer section" % raw.get("template_id"))
        return cls(
            template_id=raw.get("template_id", "?"),
            id_orig=raw.get("id_orig"),
            source_file=raw.get("source_file", ""),
            question_annotated=qa,
            body=body,
            segments=segments,
            specs=specs,
            conditions=cond_lines,
            answer_expr=cls._clean_expr(answer_expr),
            placeholder_vars=ph_order,
        )

    @staticmethod
    def _clean_expr(expr: str) -> str:
        expr = expr.strip()
        # `$ans` -> `ans` (template 0021 sigils the answer expression).
        expr = expr.replace("$", "")
        # `{expr}` -> `expr` (template 0017 wraps the answer in braces).
        if expr.startswith("{") and expr.endswith("}"):
            inner = expr[1:-1]
            if "{" not in inner and "}" not in inner:
                expr = inner
        # `mult[1]` -> `mult` (templates 0057/0095 index the variable as a
        # (surface, value) pair). Our bound value already IS the number, and the
        # rendered Python program binds a plain int/Fraction, which is not
        # subscriptable -- so normalise here, where both paths see it.
        expr = re.sub(r"\b([A-Za-z_]\w*)\s*\[\s*1\s*\]", r"\1", expr)
        return expr.strip()

    @staticmethod
    def _split_sections(qa: str) -> Tuple[str, List[str], List[str], str]:
        marks = [(m.start(), m.group(1).lower(), m.end()) for m in _SECTION_RE.finditer(qa)]
        if not marks:
            raise TemplateError("no directive sections")
        body = qa[: marks[0][0]].rstrip()
        chunks: Dict[str, str] = {}
        for i, (start, name, after) in enumerate(marks):
            stop = marks[i + 1][0] if i + 1 < len(marks) else len(qa)
            chunks[name] = qa[after:stop]

        def bullets(text: str) -> List[str]:
            out = []
            for ln in text.splitlines():
                ln = ln.strip()
                if ln.startswith("-"):
                    ln = ln[1:].strip()
                    if ln:
                        out.append(ln)
            return out

        return (
            body,
            bullets(chunks.get("init", "")),
            bullets(chunks.get("conditions", "")),
            chunks.get("answer", "").strip(),
        )

    @staticmethod
    def _segment(body: str) -> Tuple[List[Segment], List[str]]:
        segs: List[Segment] = []
        order: List[str] = []
        pos = 0
        for m in _PLACEHOLDER_RE.finditer(body):
            if m.start() > pos:
                segs.append(Segment("lit", text=body[pos : m.start()]))
            var = m.group(1)
            segs.append(Segment("ph", var=var, default=m.group(2).strip()))
            if var not in order:
                order.append(var)
            pos = m.end()
        if pos < len(body):
            segs.append(Segment("lit", text=body[pos:]))
        return segs, order

    @staticmethod
    def _parse_init_line(line: str) -> Optional[VarSpec]:
        if "=" not in line:
            return None
        lhs, rhs = line.split("=", 1)
        raw_targets = [t.strip() for t in lhs.split(",") if t.strip()]
        declared_numeric = any(t.startswith("$") for t in raw_targets)
        targets = [t.lstrip("$").strip() for t in raw_targets]
        if not all(t.isidentifier() for t in targets):
            raise TemplateError("bad init targets in %r" % line)

        rhs = rhs.strip()
        try:
            node = ast.parse(rhs, mode="eval").body
        except SyntaxError as exc:
            raise TemplateError("bad init RHS %r: %s" % (rhs, exc))

        ns = _eval_namespace()
        draw = "one"
        pool_node = node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in ("sample", "sample_sequential"):
                if not node.args:
                    raise TemplateError("sample() with no pool: %r" % rhs)
                pool_node = node.args[0]
                if fname == "sample_sequential":
                    draw = "sequential"
                elif len(node.args) > 1 or len(targets) > 1:
                    draw = "many"

        def _ev(nd):
            return eval(  # noqa: S307 -- fixed template text, sandboxed namespace
                compile(ast.Expression(nd), "<template>", "eval"),
                {"__builtins__": {}},
                ns,
            )

        pool_val = _ev(pool_node)
        if isinstance(pool_val, (list, tuple)):
            pool = list(pool_val)
        else:
            raise TemplateError("init RHS %r is not a pool" % rhs)
        if not pool:
            raise TemplateError("empty pool for %r" % targets)
        if draw != "one" and len(targets) == 1:
            draw = "one"
        return VarSpec(targets=targets, pool=pool, draw=draw, declared_numeric=declared_numeric)

    # -- sampling -----------------------------------------------------------

    def _draw(self, rng: random.Random) -> Dict[str, Any]:
        bindings: Dict[str, Any] = {}
        for spec in self.specs:
            if spec.draw == "one":
                bindings[spec.targets[0]] = _wrap(rng.choice(spec.pool))
            elif spec.draw == "sequential":
                k = len(spec.targets)
                if k > len(spec.pool):
                    raise TemplateError("sequential draw longer than pool")
                i = rng.randrange(len(spec.pool))
                for j, t in enumerate(spec.targets):
                    bindings[t] = _wrap(spec.pool[(i + j) % len(spec.pool)])
            else:
                k = len(spec.targets)
                if k > len(spec.pool):
                    raise TemplateError("sample(k) longer than pool")
                picked = rng.sample(spec.pool, k)
                for t, v in zip(spec.targets, picked):
                    bindings[t] = _wrap(v)
        return bindings

    def conditions_hold(self, bindings: Dict[str, Any]) -> bool:
        ns = _eval_namespace()
        ns.update(bindings)
        for cond in self.conditions:
            expr = cond.replace("$", "")
            try:
                ok = eval(  # noqa: S307
                    expr, {"__builtins__": {}}, ns
                )
            except Exception:
                return False
            if not ok:
                return False
        return True

    def answer(self, bindings: Dict[str, Any]) -> Optional[float]:
        """COMPUTE the ground truth by executing the template's own expression."""
        ns = _eval_namespace()
        ns.update(bindings)
        try:
            raw = eval(  # noqa: S307
                self.answer_expr, {"__builtins__": {}}, ns
            )
        except Exception:
            return None
        return normalise_answer(raw)

    def quantities_for(self, bindings: Dict[str, Any]) -> List[str]:
        """Placeholder variables carrying a numeric value, in first-appearance order."""
        return [
            v
            for v in self.placeholder_vars
            if v in bindings and is_numeric(bindings[v])
        ]

    @property
    def free_vars(self) -> List[str]:
        """Init variables that never surface in the question body.

        These are the templates' solved-for scratch variables (`ans`, `x`). They
        get no anchor -- there is no character position to anchor them to -- but
        they must be re-solved when a quantity is perturbed for `nl_break`.
        """
        out = []
        for spec in self.specs:
            for t in spec.targets:
                if t not in self.placeholder_vars:
                    out.append(t)
        return out

    def pool_for(self, name: str) -> List[Any]:
        for spec in self.specs:
            if name in spec.targets:
                return spec.pool
        return []

    def answer_refs(self) -> List[str]:
        """Identifiers the answer expression reads (excluding builtins)."""
        builtins = set(_eval_namespace())
        try:
            tree = ast.parse(self.answer_expr, mode="eval")
        except SyntaxError:
            return []
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in builtins and node.id not in names:
                names.append(node.id)
        return names

    @staticmethod
    def _lexically_coherent(bindings: Dict[str, Any]) -> bool:
        """Reject draws where two distinct lexical slots land on the same word.

        Several templates draw a "real" and a "decoy" noun from overlapping
        inline lists (template 0061 draws `objects` and `fake_object` both able
        to yield "coins"), which produces "they weren't coins, but actually
        coins". Apple's own generator evidently allows this; we do not, because
        an incoherent prompt is a silently broken item in a 100-item corpus.
        Only plain-string bindings are compared -- two numeric slots holding the
        same value is normal and stays legal.
        """
        seen = set()
        for value in bindings.values():
            if type(value) is str:
                key = value.lower()
                if key in seen:
                    return False
                seen.add(key)
        return True

    def sample(
        self,
        rng: random.Random,
        max_tries: int = 200000,
        require_positive: bool = True,
        require_integer: bool = False,
        require_lexical_coherence: bool = True,
    ) -> Instance:
        """Rejection-sample values until #conditions hold and the answer is clean.

        `require_positive` enforces the "results must be positive where the
        problem implies it" constraint: these are counting / money / distance
        word problems, so a negative or zero ground truth means the sample is
        outside the template author's intent even if the arithmetic is legal.
        """
        for _ in range(max_tries):
            bindings = self._draw(rng)
            if require_lexical_coherence and not self._lexically_coherent(bindings):
                continue
            if not self.conditions_hold(bindings):
                continue
            ans = self.answer(bindings)
            if ans is None:
                continue
            if require_positive and ans <= 0:
                continue
            if require_integer and not isinstance(ans, int):
                continue
            return Instance(
                template=self,
                bindings=bindings,
                quantities=self.quantities_for(bindings),
                answer=ans,
            )
        raise TemplateError(
            "%s: no satisfying instantiation in %d tries" % (self.template_id, max_tries)
        )

    # -- surface forms ------------------------------------------------------

    def _chunks(
        self,
        bindings: Dict[str, Any],
        symbolic: "frozenset[str]" = frozenset(),
        segments: Optional[Sequence[Segment]] = None,
    ) -> List[str]:
        """Resolve every segment of the question body into its final chunk.

        Placeholder surfaces depend on whether the prose so far ends a sentence;
        literals may need their trailing indefinite article adjusted to the
        surface that follows. Neither depends on the other, so one pass settles
        both.

        `symbolic` names the variables to emit as their *identifier* rather than
        their sampled surface. That is what turns the body into the symbolic
        schema `render_formal` states the givens against; with the default empty
        set this is exactly the natural-language rendering.
        """
        segments = self.segments if segments is None else segments
        chunks: List[str] = []
        for seg in segments:
            if seg.kind == "lit":
                chunks.append(seg.text)
            elif seg.var in symbolic:
                # An identifier, never capitalised: `n_1` and `N_1` are
                # different names, and the givens block binds the lowercase one.
                chunks.append(seg.var)
            else:
                chunks.append(
                    self._surface(
                        bindings[seg.var],
                        seg.default,
                        sentence_initial=_starts_sentence(chunks),
                    )
                )
        for i, seg in enumerate(segments):
            if seg.kind == "lit" and i + 1 < len(chunks):
                chunks[i] = _fix_article(chunks[i], chunks[i + 1])
        return chunks

    def render_nl(self, bindings: Dict[str, Any], quantities: Sequence[str]) -> Rendered:
        """Natural-language surface form.

        Spans are recorded AS THE STRING IS BUILT: `start` is the length of the
        text emitted so far, `end` is that plus the length of the surface form
        just appended. Nothing is ever located by searching the finished string,
        so a quantity whose surface happens to repeat elsewhere in the prose
        cannot be mis-anchored.
        """
        return _emit(self.segments, self._chunks(bindings), quantities)

    @staticmethod
    def _surface(value: Any, default: str, sentence_initial: bool = False) -> str:
        if hasattr(value, "surface"):
            s = value.surface
        elif isinstance(value, tuple) and len(value) == 2:
            s = str(value[0])
        elif isinstance(value, float):
            s = _fmt_float(value)
        else:
            s = str(value)
        # Two independent reasons to capitalise:
        #  (a) the default is capitalised -- `{o2, Oats}` vs `{o2, oats}` in the
        #      same template records that this slot is sentence-initial;
        #  (b) the prose emitted so far ends a sentence, which catches the slots
        #      whose default the template author left lowercase. The official
        #      generated data capitalises these ("Apples are the main diet ...").
        # Both are applied BEFORE the surface is appended, so the recorded span
        # always matches the final string -- the text is never post-edited.
        d = default.strip()
        if s[:1].islower() and ((d and d[:1].isupper()) or sentence_initial):
            s = s[:1].upper() + s[1:]
        return s

    def render_paraphrase(
        self, bindings: Dict[str, Any], quantities: Sequence[str], schema: str
    ) -> Rendered:
        """Answer-preserving paraphrase of the SAME instance, from a schema.

        `schema` is vendored prose (data/corpus/paraphrases.json) written once
        per problem, with `{var}` placeholders naming this template's own
        variables. It is NOT free text: the values are substituted from the same
        bindings that produced `nl_base`, so

          * every quantity necessarily appears, with the same surface form the
            NL rendering used for it (both go through `_surface`), and
          * its span is EMITTED at substitution time by the same `_emit` the NL
            rendering uses -- never recovered by searching the finished string.

        A paraphrase that omits a quantity cannot be anchored, so `Rendered
        .validate` raises `SpanError` and the problem fails the build rather
        than being written with a missing anchor.
        """
        segments, order = self._segment_paraphrase(schema)
        unknown = [v for v in order if v not in bindings]
        if unknown:
            raise TemplateError(
                "%s: paraphrase references unbound variables %r"
                % (self.template_id, unknown)
            )
        return _emit(segments, self._chunks(bindings, segments=segments), quantities)

    @staticmethod
    def _segment_paraphrase(schema: str) -> Tuple[List[Segment], List[str]]:
        """Split a paraphrase schema into literal / `{var}` segments."""
        segs: List[Segment] = []
        order: List[str] = []
        pos = 0
        for m in _PARA_PLACEHOLDER_RE.finditer(schema):
            if m.start() > pos:
                segs.append(Segment("lit", text=schema[pos : m.start()]))
            var = m.group(1)
            segs.append(Segment("ph", var=var))
            if var not in order:
                order.append(var)
            pos = m.end()
        if pos < len(schema):
            segs.append(Segment("lit", text=schema[pos:]))
        # A stray brace means a mistyped placeholder, which would otherwise be
        # emitted verbatim into a prompt and leave a quantity unsubstituted.
        for seg in segs:
            if seg.kind == "lit" and ("{" in seg.text or "}" in seg.text):
                raise TemplateError("malformed paraphrase placeholder in %r" % schema)
        return segs, order

    def render_formal(
        self, bindings: Dict[str, Any], quantities: Sequence[str]
    ) -> Rendered:
        """Formal surface form of the SAME problem instance: GIVENS + GOAL.

        prereg/DEVIATIONS.md Amendment 4. This rendering contains **no
        derivation and no answer expression**. It states, in formal notation:

          * `GIVEN`  -- one assignment per quantity, binding the identifier the
            template uses to the sampled value as a bare literal;
          * `WHERE`  -- the problem's own situation statements, with every
            quantity replaced by its identifier, so each bound symbol has a
            referent and the problem stays well posed;
          * `FIND`   -- the template's own question, and the unknown declared as
            `answer = ?`.

        The earlier version of this method emitted the full program including
        `answer = <the template's answer expression>`. That is the solution, not
        a re-notation: under it H3 would compare a prompt containing the
        derivation against a prompt containing a word problem, and any lambda2
        difference would be attributable to one prompt containing the answer.
        Amendment 4 removes it. The model must still do the reasoning; only the
        notation changes, which is the intervention H3 is about.

        Consequences, both deliberate:

        * The text is NOT executable (`answer = ?` is not Python, and no
          derivation is emitted), so the old build-time "the formal program
          returns ground truth" check is gone. Ground truth comes from the
          template's `#answer:` expression, as it always did; the program was
          only ever a redundant confirmation. `corpus.py` replaces the check
          with: every emitted literal equals the value it claims, and the goal
          is the template's own question.
        * Only *quantities* are bound. Solved-for scratch variables (`ans`, `x`
          in a few templates) are deliberately NOT emitted even when the answer
          expression reads them -- in those templates the scratch variable IS
          the answer, so binding it would hand over the result.
        """
        quants = list(quantities)
        nl_chunks = self._chunks(bindings)
        sym_chunks = self._chunks(bindings, symbolic=frozenset(quants))
        nl_text = "".join(nl_chunks)

        where_sym, goal_sym, goal_nl = self._split_goal(nl_chunks, sym_chunks, nl_text)

        parts: List[str] = []
        pos = 0
        anchors: Dict[str, List[int]] = {}
        literals: Dict[str, str] = {}

        def put(s: str) -> None:
            nonlocal pos
            parts.append(s)
            pos += len(s)

        put("# GIVEN\n")
        for name in quants:
            put("%s = " % name)
            lit = self._py_literal(bindings[name])
            start = pos
            put(lit)
            # Spans are recorded at the instant the literal is appended; the
            # literal appears exactly once, in this block, because the WHERE
            # and FIND blocks carry identifiers rather than values.
            if name not in anchors:
                anchors[name] = [start, pos]
                literals[name] = lit
            put("\n")

        if where_sym:
            put("\n# WHERE\n# %s\n" % where_sym)
        put("\n# FIND\n# %s\nanswer = ?" % goal_sym)

        text = "".join(parts)
        assert pos == len(text)
        anchors["FINAL"] = [-1, -1]
        r = Rendered(
            text=text,
            anchors=anchors,
            literals=literals,
            meta={
                "goal": goal_sym,
                # The same goal in the NL rendering's own words. `corpus.py`
                # asserts this is verbatim the final sentence of `nl_base`,
                # which is what "the goal names the quantity the template asks
                # for" means operationally.
                "goal_nl": goal_nl,
                "where": where_sym,
                "given": list(quants),
                "unknown": "answer",
            },
        )
        r.validate(quantities)
        return r

    @staticmethod
    def _split_goal(
        nl_chunks: Sequence[str], sym_chunks: Sequence[str], nl_text: str
    ) -> Tuple[str, str, str]:
        """Split the body into (situation statements, question) in both notations.

        The split point is found in the NATURAL-LANGUAGE rendering -- the last
        sentence boundary -- and then applied to the symbolic chunk list at the
        same position. Literal chunks are byte-identical between the two
        renderings (only placeholders differ), so a boundary that falls inside a
        literal chunk transfers exactly. That is why this is a split of the
        template's own segment list rather than a re-parse of the formal string:
        the formal goal is the same span of the template as the NL question, by
        construction.

        All 100 templates put the question in the final sentence; a body with no
        sentence boundary at all (a single-sentence problem) is all goal.
        """
        boundaries = [
            m
            for m in re.finditer(r"(?<=[.!?])\s+", nl_text)
            if not _ABBREV_BEFORE_RE.search(nl_text[: m.start()])
        ]
        if not boundaries:
            return "", "".join(sym_chunks).strip(), nl_text.strip()
        b = boundaries[-1].end()

        pos = 0
        for i, chunk in enumerate(nl_chunks):
            if pos <= b < pos + len(chunk) or (b == pos and chunk == ""):
                off = b - pos
                if chunk != sym_chunks[i][: len(chunk)] and off:
                    # boundary lands inside a chunk that differs between the two
                    # renderings: refuse rather than transfer a bogus offset
                    raise SpanError("goal split falls inside a substituted chunk")
                where = "".join(sym_chunks[:i]) + sym_chunks[i][:off]
                goal = sym_chunks[i][off:] + "".join(sym_chunks[i + 1 :])
                return where.strip(), goal.strip(), nl_text[b:].strip()
            pos += len(chunk)
        # b == len(nl_text): trailing whitespace only, no question left
        raise SpanError("goal split ran off the end of the rendering")

    @staticmethod
    def _py_literal(value: Any) -> str:
        if isinstance(value, Fraction):
            return "Fraction(%d, %d)" % (value.numerator, value.denominator)
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(int(value))
        if isinstance(value, float):
            return _fmt_float(float(value))
        if isinstance(value, tuple):
            return repr(tuple(value))
        return '"%s"' % str(value).replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


def load_templates(path: str = DEFAULT_TEMPLATE_BUNDLE) -> Tuple[Dict[str, Any], List[Template]]:
    """Load the vendored template bundle. Returns (provenance, templates)."""
    with open(path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    prov = bundle.get("_provenance", {})
    templates = [Template.parse(raw) for raw in bundle["templates"]]
    return prov, templates
