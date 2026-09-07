"""Surface-dissimilarity covariates for H2, and anchor dependency labels for H1.

WHY THIS FILE EXISTS
--------------------
H2 (DESIGN.md A.7, prereg/PRIMARY_ENDPOINTS.md) is framed as a *replication with a
control the original lacked*: Han, Rizk & Santosh show answer-flipped samples diverge
at earlier layers with **no surface-dissimilarity control at all**. Our contribution is
(a) the residual effect of answer-survival AFTER surface dissimilarity is entered as a
covariate, and (b) the **syntactic-vs-lexical contrast** they do not have.

That contrast is the whole point, so the two families are kept strictly separate here:

  LEXICAL   -- what the strings look like.  Pure python, no dependency, always available.
               * normalized character Levenshtein distance   (`lex_edit`)
               * token-type Jaccard distance                 (`lex_jaccard`)

  SYNTACTIC -- what the *parse* looks like.  Requires a real parser (spaCy). If no parser
               is present the syntactic measures are reported as unavailable and are
               NOT computed. A lexical quantity renamed `syn_*` would silently gut H2's
               central claim -- the residual would then be "survives lexical control",
               dressed up as "survives syntactic control". Never substituted. See
               `SyntacticAnalyzer.load`, which returns an analyzer whose `.available`
               is False rather than a fallback.

Conventions, stated once so nothing downstream has to guess:

  * All distances are DISSIMILARITIES in [0, 1]: 0 = identical, 1 = maximally different.
  * Every pair is (condition vs `nl_base`). `nl_base` is the reference realization
    (configs/corpus_schema.md); it has no covariate row of its own.
  * Normalized edit distance divides by `max(len(a), len(b))`, so it is 1.0 only when
    the two strings share nothing at any alignment. Two empty strings are distance 0.
  * Jaccard is over the SET of token types (not the multiset), lowercased, on
    `[A-Za-z0-9_]+` runs. Numerals are kept as tokens -- `nl_break` differs from
    `nl_base` by a single quantity surface form, and dropping numerals would make that
    perturbation invisible to the lexical control, which is exactly the variance H2
    needs (configs/corpus_schema.md: "nl_break ... must be surface-similar").
  * Anchor labels are aligned to the parse BY CHARACTER OFFSET, never by re-tokenizing
    a substring -- same rule the extraction side follows for token indices
    (configs/corpus_schema.md).
"""

from __future__ import annotations

import re
from collections import Counter

REFERENCE_CONDITION = "nl_base"
#: conditions compared against the reference, in corpus order. The last two are
#: prereg/DEVIATIONS.md Amendment 6's style controls: answer-PRESERVING minimal
#: edits, added precisely so that surface dissimilarity stops being a near-perfect
#: proxy for answer survival. Leaving them out here would reproduce the
#: collinearity the amendment exists to break.
COMPARED_CONDITIONS = (
    "nl_para",
    "formal",
    "nl_break",
    "nl_minimal_orth",
    "nl_minimal_lex",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------------------
# LEXICAL -- pure python, no dependency
# --------------------------------------------------------------------------------------


def levenshtein(a, b) -> int:
    """Unweighted Levenshtein distance between two sequences.

    Works on strings (character level) and on lists (e.g. POS-tag sequences), because
    it only ever compares elements with `!=`. O(len(a) * len(b)) time, O(len(b)) space.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ai = a[i - 1]
        cur = [i]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1,             # deletion
                           cur[j - 1] + 1,          # insertion
                           prev[j - 1] + (ai != b[j - 1])))  # substitution
        prev = cur
    return prev[lb]


def normalized_levenshtein(a, b) -> float:
    """Levenshtein divided by the longer length. In [0, 1]; 0.0 for two empty inputs."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    return levenshtein(a, b) / float(n)


def tokenize(text: str):
    """Lowercased `[A-Za-z0-9_]+` runs. Numerals retained (see module docstring)."""
    return _TOKEN_RE.findall(text.lower())


def jaccard_distance(set_a, set_b) -> float:
    """1 - |A n B| / |A u B|. Two empty sets are distance 0.0 (identical, not undefined)."""
    sa, sb = set(set_a), set(set_b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / float(len(union))


def token_jaccard_distance(text_a: str, text_b: str) -> float:
    return jaccard_distance(tokenize(text_a), tokenize(text_b))


def lexical_dissimilarity(text_a: str, text_b: str) -> dict:
    """Both lexical measures plus the raw counts they were normalized from.

    `lex_edit` is the H2 primary lexical control; `lex_jaccard` is the robustness
    variant. The raw `char_edit_distance` is kept because the corpus records a
    `perturbation.text_char_edit_distance` for `nl_break` and the two must agree.
    """
    dist = levenshtein(text_a, text_b)
    n = max(len(text_a), len(text_b))
    return {
        "lex_edit": (dist / float(n)) if n else 0.0,
        "lex_jaccard": token_jaccard_distance(text_a, text_b),
        "char_edit_distance": dist,
        "n_chars_a": len(text_a),
        "n_chars_b": len(text_b),
    }


# --------------------------------------------------------------------------------------
# SYNTACTIC -- requires a real parser, or reports itself unavailable
# --------------------------------------------------------------------------------------


def _bag_tv_distance(bag_a: Counter, bag_b: Counter) -> float:
    """Total-variation distance between two normalized count distributions, in [0, 1].

    TV = 0.5 * sum_k |p_a(k) - p_b(k)|. Length-normalized on purpose: a 71-char Python
    rewrite and a 150-char word problem must not score as "syntactically distant"
    merely because one has more tokens. Two empty bags are distance 0.0.
    """
    ta, tb = sum(bag_a.values()), sum(bag_b.values())
    if ta == 0 and tb == 0:
        return 0.0
    if ta == 0 or tb == 0:
        return 1.0
    keys = set(bag_a) | set(bag_b)
    return 0.5 * sum(abs(bag_a.get(k, 0) / ta - bag_b.get(k, 0) / tb) for k in keys)


class Parse:
    """The parser output this module needs, decoupled from spaCy's object graph.

    `tokens` is a list of (char_start, char_end, text, dep, pos) in document order,
    with character offsets INTO THE ORIGINAL STRING -- that is what makes anchor
    alignment exact.
    """

    __slots__ = ("text", "tokens", "dep_bag", "dep_triples", "pos_seq")

    def __init__(self, text, tokens, dep_bag, dep_triples, pos_seq):
        self.text = text
        self.tokens = tokens
        self.dep_bag = dep_bag
        self.dep_triples = dep_triples
        self.pos_seq = pos_seq

    def token_at_char(self, start: int, end: int = -1):
        """Token covering character offset `start`, or None.

        Alignment is by offset only: the token t with t.start <= start < t.end. If
        `start` falls in whitespace between tokens, the next token starting at or after
        `start` is returned. `start == -1` is the corpus FINAL sentinel
        (configs/corpus_schema.md) and returns the last non-space token.
        """
        if not self.tokens:
            return None
        if start < 0:
            for tok in reversed(self.tokens):
                if tok[2].strip():
                    return tok
            return None
        for tok in self.tokens:
            if tok[0] <= start < tok[1]:
                return tok
        for tok in self.tokens:
            if tok[0] >= start:
                return tok
        return None


class SyntacticAnalyzer:
    """spaCy-backed syntactic distances. `available` is False when no parser loaded.

    Measures (all dissimilarities in [0, 1]):

      `syn_dep_bag`     total-variation distance between the normalized distributions of
                        DEPENDENCY LABELS (nsubj, dobj, nummod, ...). This is the H2
                        primary syntactic control: it is a property of the parse, not of
                        the string, and it is invariant to length and to word choice.
      `syn_dep_triple`  same TV distance over LABELED ARC TRIPLES (head POS, dep label,
                        child POS). Strictly more structural than the label bag -- it
                        distinguishes `nummod(NOUN <- NUM)` from `nummod(VERB <- NUM)`.
      `syn_pos_seq`     normalized Levenshtein over the coarse POS-tag SEQUENCE. This is
                        an order-sensitive surface-syntax measure and is the robustness
                        variant of the syntactic control.

    NOT implemented: true tree-edit distance (Zhang-Shasha). It needs either a
    dependency-tree edit library or ~200 lines of ordered-forest DP, and the two bag
    measures above plus the POS sequence already separate the lexical family from the
    syntactic one, which is what DESIGN.md A.7 asks for. Named here so the paper reports
    what was actually computed.
    """

    def __init__(self, nlp=None, spacy_version=None, model_name=None, model_version=None,
                 reason=None):
        self._nlp = nlp
        self.available = nlp is not None
        self.spacy_version = spacy_version
        self.model_name = model_name
        self.model_version = model_version
        self.reason = reason
        self.method = (
            "spacy dependency parse: dep-label bag total-variation distance (primary), "
            "labeled-arc-triple bag TV distance, POS-sequence normalized Levenshtein"
            if self.available else "unavailable"
        )

    # -- construction ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name: str = "en_core_web_sm") -> "SyntacticAnalyzer":
        """Load spaCy, or return an unavailable analyzer. NEVER falls back to a proxy."""
        try:
            import spacy  # noqa: WPS433 (optional dependency, by design)
        except Exception as exc:  # pragma: no cover - environment dependent
            return cls(reason="spacy is not installed (%s: %s)"
                              % (type(exc).__name__, exc))
        try:
            nlp = spacy.load(model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            return cls(reason="spacy %s is installed but model %r would not load "
                              "(%s: %s)" % (spacy.__version__, model_name,
                                            type(exc).__name__, exc))
        if "parser" not in nlp.pipe_names:
            return cls(reason="model %r has no dependency parser in its pipeline (%s); "
                              "dependency-label distances are undefined without one"
                              % (model_name, nlp.pipe_names))
        meta = getattr(nlp, "meta", {}) or {}
        return cls(nlp=nlp, spacy_version=spacy.__version__, model_name=model_name,
                   model_version=meta.get("version"))

    # -- use ---------------------------------------------------------------------------

    def parse(self, text: str) -> Parse:
        if not self.available:
            raise RuntimeError("no parser available: " + (self.reason or "unknown"))
        doc = self._nlp(text)
        tokens, dep_bag, triples, pos_seq = [], Counter(), Counter(), []
        for tok in doc:
            if tok.is_space:
                continue
            tokens.append((tok.idx, tok.idx + len(tok.text), tok.text, tok.dep_, tok.pos_))
            dep_bag[tok.dep_] += 1
            pos_seq.append(tok.pos_)
            head = tok.head
            triples[(head.pos_, tok.dep_, tok.pos_)] += 1
        return Parse(text, tokens, dep_bag, triples, pos_seq)

    def dissimilarity(self, parse_a: Parse, parse_b: Parse) -> dict:
        return {
            "syn_dep_bag": _bag_tv_distance(parse_a.dep_bag, parse_b.dep_bag),
            "syn_dep_triple": _bag_tv_distance(parse_a.dep_triples, parse_b.dep_triples),
            "syn_pos_seq": normalized_levenshtein(parse_a.pos_seq, parse_b.pos_seq),
        }

    def anchor_label(self, parse: Parse, span) -> str:
        """Dependency relation of the token at `span` = [char_start, char_end).

        `[-1, -1]` is the FINAL sentinel. Returns None when nothing aligns (empty text).
        """
        start = span[0] if span else -1
        end = span[1] if span and len(span) > 1 else -1
        tok = parse.token_at_char(start, end)
        return None if tok is None else tok[3]


def load_syntactic_analyzer(model_name: str = "en_core_web_sm") -> SyntacticAnalyzer:
    """Convenience alias for `SyntacticAnalyzer.load`."""
    return SyntacticAnalyzer.load(model_name)


# --------------------------------------------------------------------------------------
# pair-level assembly
# --------------------------------------------------------------------------------------

#: keys written into the model-ready primary covariate file, in the order the
#: pre-registration names them (lexical control, then syntactic control).
PRIMARY_KEYS = ("lex_dissim", "syn_dissim")

#: which measure fills each primary key, and which fills the robustness file.
#: Both families land on the SAME two key names, by design -- the robustness file has
#: to drop into `04_analyze.py --covariates` unchanged. That also means the two files
#: are indistinguishable from their contents, so a robustness file passed by mistake
#: would be fitted and reported as H2's PRIMARY endpoint. `SOURCE_KEY` below is the
#: reserved top-level tag that makes the two tellable apart: the builder writes it and
#: the analysis echoes it under `h2.covariates_source`.
PRIMARY_SOURCE = {"lex_dissim": "lex_edit", "syn_dissim": "syn_dep_bag"}
ROBUSTNESS_SOURCE = {"lex_dissim": "lex_jaccard", "syn_dissim": "syn_pos_seq"}

#: Reserved top-level key in a covariates file. Its value is a dict, NOT a
#: {condition: {...}} block, and consumers must strip it before taking the union of
#: covariate names -- an unstripped tag would be entered as a covariate.
SOURCE_KEY = "_source"

#: `family` values written into SOURCE_KEY.
SOURCE_PRIMARY = "primary"
SOURCE_ROBUSTNESS = "robustness"


def source_tag(family: str, keys) -> dict:
    """The SOURCE_KEY payload for a covariates file of the given family."""
    table = {SOURCE_PRIMARY: PRIMARY_SOURCE, SOURCE_ROBUSTNESS: ROBUSTNESS_SOURCE}
    if family not in table:
        raise ValueError("family must be one of %r, got %r" % (sorted(table), family))
    return {
        "family": family,
        "measures": {k: table[family][k] for k in keys},
        "note": ("which measure fills each covariate key. The primary and robustness "
                 "files share key NAMES so either drops into --covariates unchanged; "
                 "this tag is the only thing that distinguishes them, and it is echoed "
                 "into analysis.json so a robustness refit can never be read as the "
                 "pre-registered primary."),
    }


def pair_measures(text_ref: str, text_other: str, analyzer: SyntacticAnalyzer = None,
                  parse_ref: Parse = None, parse_other: Parse = None) -> dict:
    """All dissimilarity measures for one (nl_base, condition) pair.

    Syntactic keys are present with value None -- explicitly null, never imputed --
    when no parser is available. `parse_ref` may be passed in to avoid re-parsing
    `nl_base` once per condition.
    """
    out = lexical_dissimilarity(text_ref, text_other)
    if analyzer is not None and analyzer.available:
        pr = parse_ref if parse_ref is not None else analyzer.parse(text_ref)
        po = parse_other if parse_other is not None else analyzer.parse(text_other)
        out.update(analyzer.dissimilarity(pr, po))
    else:
        out.update({"syn_dep_bag": None, "syn_dep_triple": None, "syn_pos_seq": None})
    return out


def select(measures: dict, source: dict) -> dict:
    """Project the full measure dict onto the two model-ready covariate keys.

    Returns None when any selected value is None, so callers can drop the pair rather
    than write a null into a file whose consumer will `float()` it.
    """
    picked = {}
    for key, src in source.items():
        val = measures.get(src)
        if val is None:
            return None
        picked[key] = float(val)
    return picked
