"""Residual-stream capture, anchored-position extraction, per-layer streaming.

Covers DESIGN.md §A.1 (never persist raw activations, checkpoint after every
layer), §A.4 (batch size 1, no padding), §A.6 (aligned positions), §A.11 items
1-3 (pre/post layernorm, layer indexing, padding leakage), and the anchor
contract in configs/corpus_schema.md.

===========================================================================
LAYER INDEXING CONVENTION  --  read this before touching anything downstream
===========================================================================
``residuals[i]`` is the **OUTPUT of decoder block i**, i.e. the residual stream
*after* block ``i`` has written into it.  ``i`` ranges over ``0 .. n_layers-1``,
so a captured stack has exactly ``n_layers`` entries.  There is no entry for the
embedding output unless you explicitly ask for one
(``ResidualCapture(..., capture_embeddings=True)``), and it is stored separately
under ``.embeddings``, never spliced into ``residuals`` — precisely so that
"layer 0" cannot come to mean two different things in two different files.

Equivalently: ``residuals[i]`` is the INPUT of block ``i+1``, and
``residuals[n_layers-1]`` is the input of the final norm + unembedding.  If a
downstream table says "layer 12", it means the output of block 12 and the input
of block 13.  §A.11 lists the input-vs-output off-by-one as a silent-corruption
bug; this is the single place the convention is defined.

===========================================================================
LAYER-PAIR INDEX CONVENTION  --  prereg/DEVIATIONS.md Amendment 3
===========================================================================
lambda2 and path entropy are computed on a cluster-TRANSITION graph, which is a
statistic of two ADJACENT layers, not of one layer.  Amendment 2 makes the H3
primary endpoint a mean paired difference across LAYER-PAIRS (the pairing unit
is the layer-pair, n ~ 40, not the problem), so the pair index is now part of
the primary endpoint's definition and an off-by-one in it is exactly the §A.11
item-2 failure mode one level up.

    Layer-pair ``p`` = (output of block ``p``, output of block ``p+1``).

    * ``p`` ranges over ``0 .. n_layers-2`` inclusive.
    * A model with ``L`` blocks therefore has exactly ``L - 1`` layer-pairs.
      For gemma-4-12B-it, ``L = 48`` gives **47 layer-pairs**, and 47 is the
      ``n`` for H3's Wilcoxon signed-rank test — NOT 48, and NOT 40.
    * The "source"/"earlier" member of pair ``p`` is ``residuals[p]``; the
      "target"/"later" member is ``residuals[p+1]``.  A cluster transition is
      read in that direction: the state at pair ``p`` moves FROM its cluster at
      the output of block ``p`` TO its cluster at the output of block ``p+1``.
    * There is no pair ``n_layers-1``: block ``n_layers-1`` has no successor
      block.  There is no pair involving the embedding output either — the
      embedding is not in ``residuals`` at all (see above), so pair 0 is
      (output of block 0, output of block 1) and never (embeddings, block 0).
    * ``k`` for BOTH members of a pair is the ``k`` selected on nl_base at that
      member's own layer (prereg decision 1), i.e. pair ``p`` uses ``k[p]`` for
      the source partition and ``k[p+1]`` for the target partition.

:func:`n_layer_pairs`, :func:`layer_pair_indices` and
:meth:`AlignedActivationStore.iter_layer_pairs` are the only sanctioned way to
enumerate pairs; do not write ``range(n_layers)`` and index ``i-1`` in a
closure.

===========================================================================
WHICH SIDE OF THE LAYERNORM  --  §A.11 item 1
===========================================================================
We take the residual stream **post-block and pre-normalization**: the raw value
returned by ``model.model.layers[i]``, with NO layernorm applied to it.

Why that is the right side, mechanically: these are pre-norm transformers.  A
block computes, schematically,

    h = h + Attn(Norm_1(h))
    h = h + MLP(Norm_2(h))          # Gemma-family variants additionally wrap
                                    # each sublayer output in a post-norm that
                                    # lives INSIDE the residual branch

The normalizations are applied to *copies* fed into the sublayers; the value
carried along the skip connection — the residual stream proper — is never
normalized in place.  The block's return value therefore IS the residual stream.
The only normalization applied to it is ``model.model.norm``, once, after the
last block, immediately before the unembedding (the logit lens applies that norm
at read-out time rather than treating it as part of the stream).

Why it matters: RMSNorm rescales every position to a common radius.  Taking the
post-norm value would project all token states onto (a scaling of) the unit
sphere, destroying exactly the norm-growth-with-depth structure that a spectral
affinity is sensitive to — and it would do so while still producing
plausible-looking clusters and a plausible-looking lambda2.  §A.11 calls this
"the most common silent bug in this class of work", and it is silent in both
directions.

Consequence you must handle downstream, not here: because we take the
un-normalized stream, the vector norms grow substantially with depth (Gemma also
scales the embedding by sqrt(d_model) at the input).  Any affinity built on raw
Euclidean distance will therefore be dominated by depth-dependent scale.  Build
the affinity on cosine similarity, or standardize per layer — and state which,
since lambda2 is not invariant to that choice.

===========================================================================
BATCH SIZE 1, NO PADDING  --  §A.4, §A.11 item 3
===========================================================================
Extraction is batch size 1 with an all-ones attention mask, always.  Asserted in
the hook itself (``hidden.shape[0] == 1``), not merely by convention, because a
padded position produces a garbage activation that clusters as a real state and
nothing downstream can tell the difference.  There is no batched code path in
this module to accidentally reach for.

===========================================================================
RAW ACTIVATIONS ARE NEVER PERSISTED  --  §A.1 rule 1
===========================================================================
Extract -> cluster -> write assignments -> discard.  ``LayerCheckpointWriter``
actively refuses to write anything shaped like a raw activation matrix.  The
optional spill directory of ``AlignedActivationStore`` is scratch only: it must
be on node-local disk, it is deleted on close, and it refuses to sit inside
``results/`` or ``data/``.
"""

from __future__ import annotations

from . import determinism as _det  # noqa: F401  (import-order: sets CUBLAS env var)

import glob
import json
import os
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

#: Sentinel key for "the final token of the prompt" (configs/corpus_schema.md).
FINAL = "FINAL"


class AnchorError(ValueError):
    """Raised when an anchor character span cannot be resolved to a token."""


class AnchorDropError(AnchorError):
    """An anchor failure that makes the whole PROBLEM undroppable-in-place.

    Every subclass names the ``problem_id`` and the offending ``condition``, so
    the extraction driver can do::

        try:
            acts = extract_aligned(...)
        except AnchorDropError as exc:
            drops.append({"problem_id": exc.problem_id,
                          "condition": exc.condition,
                          "reason": exc.reason})
            continue          # drop the PROBLEM, all four realizations

    and report the drop rate under gate G3 (prereg/PRIMARY_ENDPOINTS.md).

    A problem is dropped in ALL FOUR conditions or none: §A.6 restricts the
    analysis to positions present in all four realizations, so keeping the three
    conditions that resolved cleanly would reintroduce exactly the graph-size
    confound the restriction exists to remove.
    """

    #: Short machine-readable code, for tallying drops by cause.
    reason = "anchor_error"

    def __init__(self, message: str, *, problem_id: Optional[str] = None,
                 condition: Optional[str] = None):
        self.problem_id = problem_id
        self.condition = condition
        super().__init__("[problem_id={} condition={}] {}".format(
            problem_id if problem_id is not None else "<unset>",
            condition if condition is not None else "<unset>",
            message))


class AnchorKeyMismatch(AnchorDropError):
    """The realization's anchor keys are not exactly the expected key set.

    §A.6: if ``formal`` is missing one quantity it contributes fewer token
    states per problem than ``nl_base``.  lambda2 is strongly node-count
    dependent, so the H3 endpoint would absorb a pure graph-size difference —
    the precise confound §A.6 exists to prevent.
    """

    reason = "anchor_key_mismatch"

    def __init__(self, message: str, *, problem_id=None, condition=None,
                 missing: Optional[Sequence[str]] = None,
                 unexpected: Optional[Sequence[str]] = None):
        self.missing = list(missing or [])
        self.unexpected = list(unexpected or [])
        super().__init__(message, problem_id=problem_id, condition=condition)


class DuplicateAnchorPosition(AnchorDropError):
    """Two anchors of one realization resolved to the SAME token index.

    Asymmetric across conditions and therefore not a wash: with
    ``strategy="last"`` a Python line ending ``num_clips_may = 24`` collapses
    FINAL and the ``num_clips_may`` anchor onto one token, while the prose
    ``"... and 24 in May."`` ends on punctuation and does not.  The formal
    pooled graph would gain a cosine-affinity-1.0 node pair per affected
    problem, shifting the degree distribution, lambda2 and the node count that
    path entropy normalises by.
    """

    reason = "duplicate_anchor_position"

    def __init__(self, message: str, *, problem_id=None, condition=None,
                 duplicates: Optional[Sequence[str]] = None):
        self.duplicates = list(duplicates or [])
        super().__init__(message, problem_id=problem_id, condition=condition)


class FinalTokenError(AnchorDropError):
    """FINAL did not resolve to a real, non-whitespace, non-special token."""

    reason = "final_token_invalid"


class ExtractionError(RuntimeError):
    pass


def _require_numpy():
    if np is None:
        raise ExtractionError("numpy is required for activation extraction")
    return np


# ==========================================================================
# 0. Layer-pair index convention  (Amendment 3; see the module docstring)
# ==========================================================================
def n_layer_pairs(n_layers: int) -> int:
    """Number of adjacent layer-pairs for a model with ``n_layers`` blocks.

    Amendment 3: pair ``p`` = (output of block ``p``, output of block ``p+1``),
    ``p`` in ``0 .. n_layers-2``.  So ``n_layers - 1`` pairs; L=48 -> 47.  This
    number is the ``n`` of H3's Wilcoxon signed-rank test.
    """
    n_layers = int(n_layers)
    if n_layers < 2:
        raise ExtractionError(
            "a layer-pair statistic needs at least 2 blocks; got {}".format(n_layers))
    return n_layers - 1


def layer_pair_indices(n_layers: int) -> List[int]:
    """``[0, 1, ..., n_layers-2]`` — the sanctioned pair index enumeration."""
    return list(range(n_layer_pairs(n_layers)))


def layer_pair_members(pair: int, n_layers: int) -> Tuple[int, int]:
    """``pair`` -> ``(source_layer, target_layer)`` = ``(pair, pair + 1)``."""
    pair = int(pair)
    if pair < 0 or pair >= n_layer_pairs(n_layers):
        raise ExtractionError(
            "layer-pair index {} out of range: a {}-block model has pairs 0..{} "
            "(Amendment 3). There is no pair {} — block {} has no successor."
            .format(pair, n_layers, n_layer_pairs(n_layers) - 1,
                    n_layers - 1, n_layers - 1))
    return pair, pair + 1


# ==========================================================================
# 1. Anchor character spans -> token indices
#    configs/corpus_schema.md: "Spans are converted to token indices at
#    extraction time via the tokenizer's offset mapping, NEVER by re-tokenizing
#    a substring."
# ==========================================================================
def anchor_order(anchors: Dict[str, Sequence[int]]) -> List[str]:
    """Deterministic, condition-invariant ordering of anchor keys.

    Quantity names sorted lexicographically, then ``FINAL`` last.

    The ordering key is the *name*, deliberately not the character offset. The
    aligned-position contract of §A.6 requires column ``j`` of the extracted
    matrix to mean the same quantity in all four realizations, and a Python
    rewrite can introduce the same quantities in a different textual order than
    the English prose. Sorting by offset would silently permute the columns
    between conditions and every cross-condition comparison would be garbage.
    """
    if FINAL not in anchors:
        raise AnchorError(
            "anchors must contain the {!r} sentinel (configs/corpus_schema.md: "
            "'FINAL ... is always present'); got keys {}".format(
                FINAL, sorted(anchors)))
    names = sorted(k for k in anchors if k != FINAL)
    return names + [FINAL]


def check_anchor_keys_consistent(realizations: Dict[str, Dict[str, Any]],
                                 *, problem_id: Optional[str] = None) -> List[str]:
    """Enforce the corpus invariant across the four realizations of one problem.

    configs/corpus_schema.md: "every quantity key appears in the `anchors` of all
    four realizations." §A.6: only positions present in all four may be used, so
    a lambda2 difference between a 60-token NL prompt and a 28-token Python
    prompt is not confounded with length.

    Returns the shared ordered key list — which is exactly the ``expected_names``
    that :func:`extract_aligned` requires for every one of the four conditions.
    Raises :class:`AnchorKeyMismatch` if any condition is missing a key; the
    caller drops the whole problem and counts it in the drop rate (G3).
    """
    keysets = {cond: set(r["anchors"]) for cond, r in realizations.items()}
    if not keysets:
        raise AnchorError("no realizations given")
    shared = set.intersection(*keysets.values())
    union = set.union(*keysets.values())
    if shared != union:
        missing = {c: sorted(union - k) for c, k in keysets.items() if k != union}
        raise AnchorKeyMismatch(
            "anchor keys differ across realizations; missing per condition: {}. "
            "configs/corpus_schema.md makes this a DROP, counted in the drop "
            "rate — do not silently intersect: a condition with fewer anchors "
            "contributes fewer nodes to the pooled graph, and lambda2 is "
            "node-count dependent (§A.6).".format(missing),
            problem_id=problem_id,
            condition=",".join(sorted(missing)) or None,
            missing=sorted(union - shared),
        )
    return anchor_order(dict.fromkeys(shared, None))


def normalize_prompt_text(text: str, *, problem_id: Optional[str] = None,
                          condition: Optional[str] = None) -> str:
    """The ONE place prompt text is normalised before tokenization.

    Strips trailing whitespace, and nothing else.

    Why this exists (and why it is not optional).  ``FINAL`` is defined as the
    last token of the prompt, and it is one of only ~4-6 aligned positions per
    problem — the one most likely to carry the answer state, and the position
    H2's fork layer is read at.  A trailing ``"\\n"`` is a token in its own
    right with a REAL, non-zero-width offset span, so the "trailing special
    token" guard does not see it: FINAL would land on ``"\\n"`` for a Python
    rewrite that ends with a newline and on ``"."`` for prose that does not.
    That is a systematic, condition-correlated difference in what FINAL means,
    injected straight into the H3 endpoint and the H2 fork layer.

    Only trailing whitespace is removed, so every anchor character span stays
    valid: ``rstrip`` cannot shift any offset before the cut.  An anchor whose
    span extended INTO the stripped tail will now fail to match, loudly, which
    is the correct outcome — that span was pointing at whitespace.

    Leading whitespace is deliberately NOT stripped: it would shift every
    character offset in the record and silently invalidate every anchor span.
    A prompt with leading whitespace is a corpus bug, and is rejected here
    rather than papered over.
    """
    if text is None:
        raise AnchorError("prompt text is None")
    out = text.rstrip()
    if not out:
        raise AnchorError("prompt text is empty after stripping trailing whitespace")
    if out[0].isspace():
        raise AnchorError(
            "prompt text begins with whitespace. Leading whitespace cannot be "
            "stripped here without shifting every anchor character span in the "
            "corpus record; fix the record (problem_id={}, condition={}) "
            "instead.".format(problem_id, condition))
    return out


def _token_text(offsets: Sequence[Tuple[int, int]], text: Optional[str], idx: int) -> str:
    if text is None:
        return ""
    s, e = offsets[idx]
    return text[int(s):int(e)]


def validate_final_token(
    idx: int,
    offsets: Sequence[Tuple[int, int]],
    *,
    text: Optional[str] = None,
    input_ids: Optional[Sequence[int]] = None,
    special_ids: Optional[Sequence[int]] = None,
    allow_trailing_special: bool = False,
    problem_id: Optional[str] = None,
    condition: Optional[str] = None,
) -> None:
    """POSITIVE check that the resolved FINAL token is a real content token.

    The zero-width-span check alone is not sufficient (see
    :func:`normalize_prompt_text`): a trailing newline has a real span and would
    sail through it.  This asserts what we actually need — that the token
    FINAL landed on carries content:

    1. its offset span is non-empty (catches an appended special token), and
    2. the text it covers is not whitespace-only, and
    3. its token id is not in ``tokenizer.all_special_ids``.

    ``allow_trailing_special`` relaxes (1) and (3) only, never (2): a
    whitespace FINAL is never acceptable in any condition.
    """
    ts, te = (int(offsets[idx][0]), int(offsets[idx][1]))
    if te <= ts and not allow_trailing_special:
        raise FinalTokenError(
            "FINAL resolves to token {} which has a zero-width offset span {} — "
            "the tokenizer appended a trailing special token (EOS / "
            "end-of-turn). The final position must be a real content token, and "
            "a trailing special can differ in presence across the four "
            "realizations, silently breaking the §A.6 alignment. Tokenize with "
            "add_special_tokens=False, or strip the chat template, or pass "
            "allow_trailing_special=True if you have verified it is present "
            "identically in all four conditions.".format(idx, (ts, te)),
            problem_id=problem_id, condition=condition)

    if text is not None and te > ts:
        covered = text[ts:te]
        if covered.strip() == "":
            raise FinalTokenError(
                "FINAL resolves to token {} whose text is whitespace-only "
                "({!r}). A trailing newline is a real token with a real span, "
                "so the zero-width guard does not catch it: FINAL would mean "
                "'the newline' in a Python rewrite ending in \\n and 'the "
                "sentence-final period' in prose, which is a systematic, "
                "condition-correlated difference in the single most important "
                "aligned position (§A.6, and the position H2's fork layer is "
                "read at). Prompt text must be passed through "
                "normalize_prompt_text() before tokenization."
                .format(idx, covered),
                problem_id=problem_id, condition=condition)

    if input_ids is not None and special_ids and not allow_trailing_special:
        tid = int(input_ids[idx])
        if tid in set(int(s) for s in special_ids):
            raise FinalTokenError(
                "FINAL resolves to token {} whose id {} is one of the "
                "tokenizer's all_special_ids. Special tokens are not content "
                "positions and their presence can differ across the four "
                "realizations (§A.6).".format(idx, tid),
                problem_id=problem_id, condition=condition)


def _spans_overlap(tok: Tuple[int, int], span: Tuple[int, int]) -> bool:
    ts, te = tok
    ss, se = span
    if te <= ts:                       # zero-width token = special token
        return False
    if se <= ss:                       # zero-width anchor: treat as a point
        return ts <= ss < te
    return ts < se and te > ss


def char_span_to_token_index(
    offsets: Sequence[Tuple[int, int]],
    span: Sequence[int],
    *,
    strategy: str = "last",
    name: str = "<anchor>",
    n_tokens: Optional[int] = None,
    allow_trailing_special: bool = False,
    text: Optional[str] = None,
    input_ids: Optional[Sequence[int]] = None,
    special_ids: Optional[Sequence[int]] = None,
    problem_id: Optional[str] = None,
    condition: Optional[str] = None,
) -> int:
    """Map one anchor character span to a single token index.

    Parameters
    ----------
    offsets:
        The tokenizer's ``offset_mapping`` for the full prompt, one
        ``(char_start, char_end)`` per token, in token order. Special tokens
        carry a zero-width span (typically ``(0, 0)``) and are skipped.
    span:
        ``[start, end)`` character offsets, or ``[-1, -1]`` for the ``FINAL``
        sentinel.
    strategy:
        ``"last"`` (default) -> the LAST token overlapping the span.
        ``"first"`` -> the first.

        Judgment call, stated for the record: a quantity such as ``1,250`` or
        ``num_apples`` spans several tokens. In a causal model the hidden state
        at the *last* token of the span is the first state that has attended to
        the whole quantity; the state at the first token has seen only a
        fragment and is largely a function of the preceding context. §A.6 asks
        for "the positions where each quantity is introduced", which is
        satisfied by either, so we take the one that actually encodes the
        quantity. Whichever is used must be used for all four realizations —
        this argument is passed through from a single call site.

    Returns
    -------
    int
        Token index into the prompt's ``input_ids``.
    """
    if strategy not in ("first", "last"):
        raise AnchorError("strategy must be 'first' or 'last', got {!r}".format(strategy))
    n = len(offsets) if n_tokens is None else n_tokens
    if n == 0:
        raise AnchorError("empty tokenization")

    start, end = int(span[0]), int(span[1])

    # --- FINAL sentinel ---------------------------------------------------
    if start < 0 or end < 0:
        idx = n - 1
        # Both the zero-width (negative) guard and the POSITIVE content check
        # live in validate_final_token, so there is exactly one definition of
        # "FINAL landed somewhere legitimate".
        validate_final_token(
            idx, offsets, text=text, input_ids=input_ids, special_ids=special_ids,
            allow_trailing_special=allow_trailing_special,
            problem_id=problem_id, condition=condition,
        )
        return idx

    matches = [i for i in range(n) if _spans_overlap(tuple(offsets[i]), (start, end))]
    if not matches:
        raise AnchorError(
            "anchor {!r} span [{}, {}) matches no token. Either the span is "
            "stale relative to the text (regenerate the corpus record), or it "
            "covers only whitespace that this tokenizer folds into a "
            "neighbouring token's leading space, or it pointed into trailing "
            "whitespace that normalize_prompt_text() removed. Do NOT work "
            "around this by re-tokenizing the substring — that gives a "
            "different tokenization than the in-context one "
            "(configs/corpus_schema.md).".format(name, start, end))
    return matches[-1] if strategy == "last" else matches[0]


@dataclass
class AnchorResolution:
    """Resolved anchor positions for one realization."""

    names: List[str]                 # ordered, FINAL last
    token_indices: List[int]         # parallel to `names`
    input_ids: List[int]
    offsets: List[Tuple[int, int]]
    n_tokens: int
    strategy: str
    text: str = ""                   # the NORMALISED text actually tokenized
    problem_id: Optional[str] = None
    condition: Optional[str] = None
    text_was_rstripped: bool = False

    @property
    def n_positions(self) -> int:
        return len(self.names)

    def token_text(self, name: str) -> str:
        """The text covered by the token that anchor ``name`` resolved to."""
        return _token_text(self.offsets, self.text or None,
                           self.token_indices[self.names.index(name)])


def _check_expected_names(names: Sequence[str], expected_names: Sequence[str],
                          *, problem_id, condition) -> None:
    """§A.6 cross-condition invariant, asserted on EVERY resolution.

    ``expected_names`` is the shared, ordered key list of the problem — the
    return value of :func:`check_anchor_keys_consistent`.  It is a REQUIRED
    parameter of the extraction entry points precisely so that this cannot be
    skipped by a caller who forgot to call the checker: an invariant that is
    only enforced when someone remembers to enforce it is not an invariant.
    """
    if expected_names is None:
        raise TypeError(
            "expected_names is required (§A.6). Pass the shared ordered key "
            "list returned by check_anchor_keys_consistent(problem['realizations']) "
            "— the SAME list for all four conditions of a problem.")
    exp = list(expected_names)
    if not exp:
        raise AnchorError("expected_names is empty")
    if FINAL not in exp:
        raise AnchorError(
            "expected_names must contain the {!r} sentinel; got {}".format(FINAL, exp))
    canonical = anchor_order(dict.fromkeys(exp, None))
    if canonical != exp:
        raise AnchorError(
            "expected_names {} is not in canonical order; use "
            "anchor_order()/check_anchor_keys_consistent() so that column j "
            "means the same quantity in all four realizations (§A.6). "
            "Canonical order is {}.".format(exp, canonical))
    if list(names) != exp:
        missing = [n for n in exp if n not in names]
        unexpected = [n for n in names if n not in exp]
        raise AnchorKeyMismatch(
            "anchor keys for this realization are {} but the problem's shared "
            "key list is {} (missing here: {}; unexpected here: {}). §A.6 "
            "restricts the analysis to positions present in ALL FOUR "
            "realizations; a condition contributing fewer aligned positions "
            "contributes fewer nodes to the pooled cluster-transition graph, "
            "and lambda2 is strongly node-count dependent — the H3 endpoint "
            "would absorb a pure graph-size difference. DROP the whole problem "
            "and count it in the G3 drop rate."
            .format(list(names), exp, missing, unexpected),
            problem_id=problem_id, condition=condition,
            missing=missing, unexpected=unexpected)


def resolve_anchor_positions(
    text: str,
    anchors: Dict[str, Sequence[int]],
    tokenizer: Any,
    *,
    expected_names: Sequence[str],          # REQUIRED — §A.6, see below
    problem_id: Optional[str] = None,
    condition: Optional[str] = None,
    strategy: str = "last",
    add_special_tokens: bool = True,
    allow_trailing_special: bool = False,
    allow_duplicate_positions: bool = False,
) -> AnchorResolution:
    """Tokenize ``text`` once and resolve every anchor span to a token index.

    The prompt is tokenized exactly once, as a whole, with
    ``return_offsets_mapping=True``. No substring is ever tokenized on its own:
    subword tokenizers are context sensitive (leading-space handling, digit
    grouping, byte fallback), so a substring's tokenization does not in general
    appear in the full sequence's tokenization, and matching on it produces
    positions that are quietly wrong.

    Parameters
    ----------
    expected_names:
        **Required.** The problem's shared, ordered anchor key list, i.e. the
        return value of :func:`check_anchor_keys_consistent`, passed identically
        for all four conditions. Validated on every call — see
        :func:`_check_expected_names` for why it is not optional. Omitting it
        raises ``TypeError``.
    allow_duplicate_positions:
        Default **False**. Two anchors resolving to one token produce two
        byte-identical rows in the graph; see :class:`DuplicateAnchorPosition`
        for why that is asymmetric across conditions rather than a wash. Set to
        True only for a deliberate diagnostic, never for the primary run.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise AnchorError(
            "a fast tokenizer is required for offset mapping; got {}".format(
                type(tokenizer).__name__))

    # ONE place where prompt text is normalised, applied identically to all four
    # conditions (F3 / §A.6).
    raw = text
    text = normalize_prompt_text(text, problem_id=problem_id, condition=condition)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        return_tensors=None,
        padding=False,           # §A.11 item 3 — no padding, ever
        truncation=False,
    )
    input_ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(offsets) != len(input_ids):
        raise AnchorError("offset_mapping length != input_ids length")

    if not hasattr(tokenizer, "all_special_ids"):
        raise AnchorError(
            "tokenizer {} does not expose `all_special_ids`, so the FINAL "
            "position cannot be checked against the special-token set. Every "
            "HF fast tokenizer has it; a test double must provide it too — the "
            "check is not optional (§A.6).".format(type(tokenizer).__name__))
    special_ids = list(getattr(tokenizer, "all_special_ids") or [])

    names = anchor_order(anchors)
    _check_expected_names(names, expected_names,
                          problem_id=problem_id, condition=condition)

    idxs: List[int] = []
    for nm in names:
        idxs.append(char_span_to_token_index(
            offsets, anchors[nm], strategy=strategy, name=nm,
            n_tokens=len(input_ids),
            allow_trailing_special=allow_trailing_special,
            text=text, input_ids=input_ids, special_ids=special_ids,
            problem_id=problem_id, condition=condition,
        ))

    # Positive FINAL check, again, unconditionally: char_span_to_token_index can
    # be reached with strategy/params that skip it, this call site cannot.
    validate_final_token(
        idxs[names.index(FINAL)], offsets, text=text, input_ids=input_ids,
        special_ids=special_ids, allow_trailing_special=allow_trailing_special,
        problem_id=problem_id, condition=condition)

    if not allow_duplicate_positions and len(set(idxs)) != len(idxs):
        dup = [nm for nm, i in zip(names, idxs) if idxs.count(i) > 1]
        collided = sorted({i for i in idxs if idxs.count(i) > 1})
        raise DuplicateAnchorPosition(
            "anchors {} resolved to the same token index(es) {}; the "
            "corresponding token states would be byte-identical rows in the "
            "graph (cosine affinity exactly 1.0). This is ASYMMETRIC across "
            "conditions — with strategy='last' a Python line ending "
            "`num_clips_may = 24` collapses FINAL onto the quantity anchor, "
            "while prose ending '... and 24 in May.' ends on punctuation and "
            "does not — so it shifts the degree distribution, lambda2, and the "
            "node count path entropy normalises by, for `formal` only. DROP "
            "the whole problem and count it in the G3 drop rate."
            .format(dup, collided),
            problem_id=problem_id, condition=condition, duplicates=dup)

    return AnchorResolution(
        names=names, token_indices=idxs, input_ids=input_ids,
        offsets=offsets, n_tokens=len(input_ids), strategy=strategy,
        text=text, problem_id=problem_id, condition=condition,
        text_was_rstripped=(raw != text),
    )


# ==========================================================================
# 2. Forward hooks on the decoder blocks
# ==========================================================================
def _unpack_block_output(out: Any):
    """A decoder block returns either a tensor or a tuple whose [0] is it."""
    if isinstance(out, (tuple, list)):
        return out[0]
    hs = getattr(out, "last_hidden_state", None)
    if hs is not None:
        return hs
    return out


class ResidualCapture:
    """Context manager capturing the residual stream at the output of each block.

    Usage::

        with ResidualCapture(loaded) as cap:
            logits = forward_once(loaded, input_ids)
            resid = cap.residuals          # list, len == n_layers

    ``cap.residuals[i]`` is the OUTPUT of block ``i`` (see the module docstring
    for the full convention and the layernorm argument). Tensors are
    ``[1, T, d_model]`` — batch dimension retained and asserted equal to 1.

    Parameters
    ----------
    keep_on_gpu:
        Keep captures on the device. Default False: they are moved to host RAM
        so that a 40-layer capture does not sit next to the weights on a 48GB
        card. Set True only for the determinism gate, where you want the bits
        exactly as the kernel produced them (the H2D/D2H copy is bit-exact
        anyway, so this is paranoia, not necessity).
    cast_to:
        Optional torch dtype to cast captures to. ``None`` (default) keeps the
        model dtype — required for the G1 gate, which compares bf16 bit
        patterns. Extraction casts to float32 downstream, which is lossless from
        bf16.
    """

    def __init__(self, loaded: Any, *, keep_on_gpu: bool = False,
                 cast_to: Any = None, capture_embeddings: bool = False):
        self.loaded = loaded
        self.keep_on_gpu = keep_on_gpu
        self.cast_to = cast_to
        self.capture_embeddings = capture_embeddings
        self.n_layers = int(loaded.n_layers)
        self.residuals: List[Any] = [None] * self.n_layers
        self.embeddings: Any = None
        self._handles: List[Any] = []
        self._active = False

    # -- hook factory ------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):  # noqa: ANN001
            hidden = _unpack_block_output(output)
            # §A.4 / §A.11 item 3: batch size 1, no padding, EVER. Asserted at
            # the point of capture rather than trusted, because a padded
            # position yields a garbage activation that is indistinguishable
            # from a real token state once it is in the graph.
            if hidden.dim() != 3 or hidden.shape[0] != 1:
                raise ExtractionError(
                    "layer {}: expected residual [1, T, d] (batch size 1, no "
                    "padding) but got shape {}. Batching is forbidden during "
                    "extraction: float reduction order changes with batch shape "
                    "(§A.4) and padded positions leak into clusters "
                    "(§A.11).".format(layer_idx, tuple(hidden.shape)))
            t = hidden.detach()
            if self.cast_to is not None:
                t = t.to(self.cast_to)
            if not self.keep_on_gpu:
                t = t.to("cpu")
            self.residuals[layer_idx] = t.contiguous()
        return hook

    def _make_embed_hook(self):
        def hook(module, args, output):  # noqa: ANN001
            hidden = _unpack_block_output(output)
            self.embeddings = hidden.detach().to(
                "cpu" if not self.keep_on_gpu else hidden.device).contiguous()
        return hook

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "ResidualCapture":
        layers = self.loaded.layer_modules()
        if len(layers) != self.n_layers:
            raise ExtractionError(
                "layer_modules() has {} entries but n_layers is {}; the resolved "
                "module list is not the decoder stack".format(len(layers), self.n_layers))
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        if self.capture_embeddings:
            emb = self.loaded.model.get_input_embeddings()
            self._handles.append(emb.register_forward_hook(self._make_embed_hook()))
        self._active = True
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._active = False
        return None

    def clear(self) -> None:
        self.residuals = [None] * self.n_layers
        self.embeddings = None

    def assert_complete(self) -> None:
        missing = [i for i, r in enumerate(self.residuals) if r is None]
        if missing:
            raise ExtractionError(
                "no residual captured for blocks {}. Either the forward pass did "
                "not run, or the model skipped layers (early exit / layer "
                "dropout).".format(missing))


def forward_once(loaded: Any, input_ids: Any, *, attention_mask: Any = None) -> Any:
    """One batch-1, unpadded, no-cache forward pass. Returns the logits.

    ``use_cache=False``: a populated KV cache changes what the next call
    computes, and a cache carried between the two passes of the G1 gate is a
    classic way to get a nonzero fork signal that has nothing to do with the
    hypothesis.
    """
    import torch

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ExtractionError(
            "input_ids must be [1, T]; got {} (batch size 1, no padding — "
            "§A.4)".format(tuple(input_ids.shape)))
    if attention_mask is None:
        # All ones by construction: nothing is padded, so there is nothing to
        # mask. We pass it explicitly rather than letting it default, so that
        # the absence of padding is visible in the call.
        attention_mask = torch.ones_like(input_ids)
    elif int(attention_mask.sum().item()) != attention_mask.numel():
        raise ExtractionError(
            "attention_mask contains a zero — something is padded. Padded "
            "positions produce garbage activations that get clustered as real "
            "token states (§A.11 item 3).")

    with torch.inference_mode():
        out = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
    return out.logits


def encode(loaded: Any, text: str, *, add_special_tokens: bool = True,
           normalize: bool = True) -> Any:
    """Tokenize one prompt into a ``[1, T]`` tensor. No padding, no truncation.

    ``normalize=True`` (default) routes the text through
    :func:`normalize_prompt_text`, the same single normalisation the extraction
    path applies, so that a probe or a behavioural prompt cannot end on a
    trailing-newline token that the aligned extraction would never see.
    """
    import torch

    if normalize:
        text = normalize_prompt_text(text)
    enc = loaded.tokenizer(
        text, add_special_tokens=add_special_tokens, return_tensors=None,
        padding=False, truncation=False,
    )
    return torch.tensor([enc["input_ids"]], dtype=torch.long, device=loaded.device)


# ==========================================================================
# 3. Aligned extraction
# ==========================================================================
def extract_aligned(
    text: str,
    anchors: Dict[str, Sequence[int]],
    tokenizer: Any,
    model: Any,
    *,
    expected_names: Sequence[str],          # REQUIRED — §A.6, see below
    problem_id: Optional[str] = None,
    condition: Optional[str] = None,
    strategy: str = "last",
    add_special_tokens: bool = True,
    allow_trailing_special: bool = False,
    allow_duplicate_positions: bool = False,
    dtype: str = "float32",
    return_meta: bool = False,
):
    """Extract the residual stream at the semantically anchored positions.

    DESIGN.md §A.6: the graph is built from token states, so a lambda2
    difference between a 60-token NL prompt and a 28-token Python rewrite is
    confounded with length before semantics enter. The fix is to restrict to
    aligned positions — the final token plus the position at which each quantity
    is introduced — which are present in all four realizations.

    Parameters
    ----------
    expected_names:
        **Required.** The problem's shared, ordered anchor key list — the return
        value of ``check_anchor_keys_consistent(problem["realizations"])`` —
        passed identically for every one of the four conditions. It is required
        rather than optional because the §A.6 cross-condition invariant has to
        be asserted on every extraction, not on the ones a caller remembered to
        check: if ``formal`` silently contributes one fewer aligned position
        than ``nl_base``, its pooled graph has fewer nodes, and lambda2 is
        strongly node-count dependent, so H3's endpoint absorbs a pure
        graph-size difference in its pre-committed direction.
    problem_id, condition:
        Carried into :class:`AnchorDropError` so a driver can tally the drop
        under gate G3 instead of guessing which record failed.

    Raises
    ------
    AnchorDropError
        (:class:`AnchorKeyMismatch`, :class:`DuplicateAnchorPosition`,
        :class:`FinalTokenError`) — the problem must be dropped in ALL FOUR
        conditions and counted in the G3 drop rate.

    Returns
    -------
    numpy.ndarray
        ``[n_layers, n_anchor_positions, d_model]``, float32 by default.
        Axis 0 index ``i`` is the OUTPUT of block ``i`` (module docstring).
        Axis 1 is ordered by :func:`anchor_order` — quantity names sorted
        lexicographically, ``FINAL`` last — which is the same order in all four
        realizations of a problem.
        If ``return_meta`` is True, returns ``(array, AnchorResolution)``.

    Notes
    -----
    ``tokenizer`` is a parameter for signature clarity but must be the same
    tokenizer the model was loaded with; a mismatch is checked.
    """
    npy = _require_numpy()
    import torch

    if getattr(model, "tokenizer", None) is not None and model.tokenizer is not tokenizer:
        warnings.warn(
            "extract_aligned was given a tokenizer that is not the loaded model's "
            "tokenizer; offset mappings and input_ids would disagree.",
            RuntimeWarning, stacklevel=2)

    res = resolve_anchor_positions(
        text, anchors, tokenizer,
        expected_names=expected_names,
        problem_id=problem_id, condition=condition,
        strategy=strategy,
        add_special_tokens=add_special_tokens,
        allow_trailing_special=allow_trailing_special,
        # Explicitly False, never left to the callee's default: a duplicate
        # resolution is a loud, typed, countable DROP (§A.6), not a shrug.
        allow_duplicate_positions=allow_duplicate_positions,
    )

    input_ids = torch.tensor([res.input_ids], dtype=torch.long, device=model.device)

    with ResidualCapture(model) as cap:
        forward_once(model, input_ids)
        cap.assert_complete()
        idx = torch.tensor(res.token_indices, dtype=torch.long)
        # [L, P, d]; index on the CPU copies, then cast. bf16 -> float32 is exact.
        out = npy.empty((model.n_layers, res.n_positions, model.d_model),
                        dtype=npy.dtype(dtype))
        for i, resid in enumerate(cap.residuals):
            sel = resid[0].index_select(0, idx.to(resid.device))
            out[i] = sel.to(torch.float32).cpu().numpy().astype(dtype, copy=False)
        cap.clear()

    if return_meta:
        return out, res
    return out


# ==========================================================================
# 4. Per-layer streaming store  (§A.1: never hold every layer of every prompt)
# ==========================================================================
@dataclass
class StateRow:
    """Provenance of one token state. Kept alongside every activation row."""

    problem_id: str
    condition: str
    anchor_name: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.problem_id, self.condition, self.anchor_name)


@dataclass
class LayerBatch:
    """All token states for ONE layer, across every prompt added to the store."""

    layer: int
    states: Any                 # numpy [N, d_model]
    rows: List[StateRow]

    def __post_init__(self):
        if self.states.shape[0] != len(self.rows):
            raise ExtractionError("states/rows length mismatch")

    @property
    def n_states(self) -> int:
        return len(self.rows)

    def index_of(self, problem_id: str, condition: str, anchor_name: str) -> int:
        return self.rows.index(StateRow(problem_id, condition, anchor_name))

    def problem_index(self) -> Dict[str, List[int]]:
        """problem_id -> the row indices contributed by that problem.

        Exists so that the resampling unit can be the PROBLEM and never the
        token state (§A.9.2, §A.11, prereg decision 3). A bootstrap draws
        problem ids, unions their row indices, and rebuilds the graph from those
        rows; a bootstrap that drew rows directly would treat the ~4-6
        dependent states within a problem as independent and give intervals far
        too narrow. Note that a problem's rows span all four conditions, which
        is what keeps a resample balanced across conditions.
        """
        out: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            out.setdefault(r.problem_id, []).append(i)
        return out

    def condition_index(self) -> Dict[str, List[int]]:
        """condition -> row indices. For building the per-condition graphs.

        k is NOT selected here: §A.6 and prereg decision 1 require k to be
        chosen once per layer on nl_base and reused for all four conditions.
        """
        out: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            out.setdefault(r.condition, []).append(i)
        return out


@dataclass
class LayerPair:
    """Two ADJACENT layers, handed to a pair-consuming callback together.

    Amendment 3 convention (module docstring): pair ``p`` = (output of block
    ``p``, output of block ``p+1``).  ``source.layer == pair`` and
    ``target.layer == pair + 1``, asserted here rather than assumed, because
    lambda2 and path entropy are two-layer statistics and an off-by-one in the
    pair index is §A.11 item 2 one level up.

    This exists so that no caller has to carry "layer i-1" in a closure over
    :meth:`AlignedActivationStore.iter_layers`.  That pattern is where the
    off-by-one gets written, and it is invisible in the output: a transition
    graph built from (i-1, i) instead of (i, i+1) still produces a perfectly
    plausible lambda2.
    """

    pair: int
    source: "LayerBatch"        # output of block `pair`
    target: "LayerBatch"        # output of block `pair + 1`

    def __post_init__(self):
        if self.source.layer != self.pair or self.target.layer != self.pair + 1:
            raise ExtractionError(
                "layer-pair {} must be (output of block {}, output of block {}) "
                "but got batches for layers ({}, {}). Amendment 3."
                .format(self.pair, self.pair, self.pair + 1,
                        self.source.layer, self.target.layer))
        if len(self.source.rows) != len(self.target.rows):
            raise ExtractionError(
                "pair {}: source has {} rows, target has {} — the two members of "
                "a pair must be the SAME token states at two depths, or the "
                "transition graph is meaningless".format(
                    self.pair, len(self.source.rows), len(self.target.rows)))

    @property
    def n_states(self) -> int:
        return self.source.n_states

    @property
    def rows(self) -> List["StateRow"]:
        """Row provenance, shared by both members of the pair."""
        return self.source.rows

    def layers(self) -> Tuple[int, int]:
        return (self.pair, self.pair + 1)


class AlignedActivationStore:
    """Collect aligned activations prompt-by-prompt, hand them back layer-by-layer.

    The access pattern the pipeline needs is transposed from the one the model
    produces: a forward pass yields all layers of ONE prompt, but spectral
    clustering needs all prompts of ONE layer. This class does that transpose
    without ever materialising [all prompts x all layers] as a live array in the
    caller's hands.

    Two backends:

    * ``spill_dir=None`` (default) — per-layer buffers in host RAM.
      ``memory_estimate_bytes()`` tells you the total; for the planned corpus
      (400 prompts x ~40 layers x ~20 positions x d_model=3840, float32) that is
      about 4.9 GB, which fits comfortably in a pod's host RAM but not on the
      card. Nothing touches disk.
    * ``spill_dir=<path>`` — one append-only scratch file per layer. Use this if
      host RAM is tight. **Scratch only**: it is deleted by ``close()`` /
      context-manager exit, it must be node-local (never a network volume,
      §A.1 rule 2), and it refuses to live under ``results/`` or ``data/``,
      because §A.1 rule 1 is that raw activations are never *persisted* —
      extract, cluster, write assignments, discard.

    Either way, ``iter_layers()`` yields one ``LayerBatch`` at a time, so the
    CALLER never holds more than one layer: the spectral machinery sees
    ``[N_states, d_model]``, not ``[n_layers, N_states, d_model]``. With
    ``release=True`` the store also drops each layer's buffer after yielding it,
    so total residency falls monotonically through the sweep (do that only if
    nothing needs a second pass over the layers).

    **Two-layer statistics use** ``iter_layer_pairs()``, not ``iter_layers()``.
    lambda2 and path entropy are properties of a cluster-TRANSITION graph
    between adjacent layers, so they need both members of a pair at once;
    ``iter_layer_pairs()`` yields them together with the pair index defined by
    Amendment 3 (module docstring), instead of leaving the caller to stash the
    previous batch in a closure and get the index off by one.

    ``release=True`` on ``iter_layers()`` makes a subsequent PAIR sweep
    impossible — the earlier member of every pair has been freed.
    ``iter_layer_pairs()`` refuses up front rather than failing halfway
    through; if both sweeps are needed, run the pair sweep first, or run the
    single-layer sweep with ``release=False``.
    """

    def __init__(self, n_layers: int, d_model: int, *, dtype: str = "float32",
                 spill_dir: Optional[str] = None):
        _require_numpy()
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.dtype = np.dtype(dtype)
        self.rows: List[StateRow] = []
        self._closed = False
        self._spill_dir = None
        self._buffers: Optional[List[List[Any]]] = None
        self._released: set = set()

        if spill_dir is None:
            self._buffers = [[] for _ in range(self.n_layers)]
        else:
            spill_dir = os.path.abspath(spill_dir)
            for forbidden in ("results", "data"):
                parts = spill_dir.split(os.sep)
                if forbidden in parts:
                    raise ExtractionError(
                        "refusing to spill raw activations under {}/ — DESIGN.md "
                        "§A.1 rule 1: raw activations are NEVER persisted. Use a "
                        "node-local scratch path such as /dev/shm/ling or "
                        "/tmp/ling.".format(forbidden))
            os.makedirs(spill_dir, exist_ok=True)
            self._spill_dir = spill_dir
            self._files = [
                open(os.path.join(spill_dir, "layer_{:03d}.raw".format(i)), "wb")
                for i in range(self.n_layers)
            ]

    # -- writing -----------------------------------------------------------
    def add(self, problem_id: str, condition: str, anchor_names: Sequence[str],
            values: Any) -> None:
        """Add one prompt's aligned activations, ``values`` = ``[L, P, d]``."""
        if self._closed:
            raise ExtractionError("store is closed")
        if values.shape != (self.n_layers, len(anchor_names), self.d_model):
            raise ExtractionError(
                "expected [{}, {}, {}], got {}".format(
                    self.n_layers, len(anchor_names), self.d_model, values.shape))
        arr = values.astype(self.dtype, copy=False)
        for j, nm in enumerate(anchor_names):
            self.rows.append(StateRow(problem_id, condition, nm))
        for i in range(self.n_layers):
            block = np.ascontiguousarray(arr[i])
            if self._buffers is not None:
                self._buffers[i].append(block)
            else:
                self._files[i].write(block.tobytes())

    # -- reading -----------------------------------------------------------
    def iter_layers(self, layers: Optional[Iterable[int]] = None,
                    *, release: bool = False) -> Iterator[LayerBatch]:
        """Yield one :class:`LayerBatch` at a time.

        ``release=True`` frees each layer's stored buffer once it has been
        yielded — one-shot consumption, lower peak residency, no second pass.
        """
        if self._closed:
            raise ExtractionError("store is closed")
        if self._buffers is None:
            for fh in self._files:
                fh.flush()
        wanted = list(range(self.n_layers)) if layers is None else list(layers)
        for i in wanted:
            yield LayerBatch(layer=i, states=self._load_layer(i), rows=list(self.rows))
            if release:
                self._release_layer(i)

    # -- two-layer affordance (Amendment 3) --------------------------------
    def iter_layer_pairs(self, pairs: Optional[Iterable[int]] = None,
                         *, release: bool = False) -> Iterator[LayerPair]:
        """Yield adjacent layer-pairs, one :class:`LayerPair` at a time.

        Pair ``p`` = (output of block ``p``, output of block ``p+1``), for
        ``p`` in ``0 .. n_layers-2`` (Amendment 3).  A full sweep therefore
        yields ``n_layers - 1`` pairs — 47 for a 48-block model, which is the
        ``n`` of H3's Wilcoxon.

        This is the sanctioned way to consume a TWO-layer statistic (lambda2 of
        the cluster-transition graph, path entropy).  ``iter_layers`` hands the
        callback one layer at a time, which forces a pair-consuming caller to
        stash the previous batch in a closure — that is where the off-by-one
        gets written, and nothing downstream can see it.

        Residency: at most TWO layers are resident at once, not the whole stack.
        The source batch of pair ``p`` is the target batch of pair ``p-1``,
        loaded once and reused.

        ``release=True`` frees layer ``p`` after pair ``p`` has been yielded —
        the last point at which it is needed, since layer ``p`` appears only in
        pairs ``p-1`` (as target) and ``p`` (as source).  The final layer is
        freed when the sweep completes.  Only use it for a one-shot sweep.

        Raises
        ------
        ExtractionError
            If any layer the requested pairs need was already released by an
            earlier ``release=True`` sweep.  This is checked UP FRONT, before
            any work is done, rather than letting the sweep die halfway: a
            partially completed pair sweep is the shape of bug that gets
            "fixed" by narrowing the layer range.
        """
        if self._closed:
            raise ExtractionError("store is closed")
        if self._buffers is None:
            for fh in self._files:
                fh.flush()

        wanted = (layer_pair_indices(self.n_layers) if pairs is None
                  else [int(p) for p in pairs])
        for p in wanted:
            layer_pair_members(p, self.n_layers)      # range check, Amendment 3

        needed = sorted({l for p in wanted for l in (p, p + 1)})
        gone = [l for l in needed if l in self._released]
        if gone:
            raise ExtractionError(
                "layer-pair sweep needs layers {} but {} were already released "
                "by an earlier iter_layers(release=True)/iter_layer_pairs("
                "release=True) sweep. lambda2 and path entropy are TWO-layer "
                "statistics (Amendment 3): a release=True single-layer sweep "
                "makes the previous layer unreadable and therefore makes the "
                "pair sweep impossible. Re-run the single-layer sweep with "
                "release=False, or do the pair sweep first."
                .format(needed, gone))

        if release and wanted != sorted(wanted):
            raise ExtractionError(
                "iter_layer_pairs(release=True) requires ascending pair indices "
                "(a released layer cannot be re-read); got {}".format(wanted))

        prev: Optional[LayerBatch] = None
        for p in wanted:
            if prev is not None and prev.layer == p:
                source = prev
            else:
                source = LayerBatch(layer=p, states=self._load_layer(p),
                                    rows=list(self.rows))
            target = LayerBatch(layer=p + 1, states=self._load_layer(p + 1),
                                rows=list(self.rows))
            yield LayerPair(pair=p, source=source, target=target)
            if release:
                # layer p is needed by pairs p-1 and p only; pair p is done.
                self._release_layer(p)
            prev = target
        if release and prev is not None:
            self._release_layer(prev.layer)

    def _release_layer(self, i: int) -> None:
        self._released.add(i)
        if self._buffers is not None:
            self._buffers[i] = []
        elif self._spill_dir is not None:
            try:
                os.remove(os.path.join(self._spill_dir, "layer_{:03d}.raw".format(i)))
            except OSError:
                pass

    def layer(self, i: int) -> LayerBatch:
        return LayerBatch(layer=i, states=self._load_layer(i), rows=list(self.rows))

    def _load_layer(self, i: int) -> Any:
        if i in self._released:
            raise ExtractionError(
                "layer {} was released after being streamed (release=True) and "
                "cannot be read again. Use release=False if you need a second "
                "pass over the layers.".format(i))
        if self._buffers is not None:
            bufs = self._buffers[i]
            if not bufs:
                return np.empty((0, self.d_model), dtype=self.dtype)
            return np.concatenate(bufs, axis=0)
        path = os.path.join(self._spill_dir, "layer_{:03d}.raw".format(i))
        arr = np.fromfile(path, dtype=self.dtype)
        return arr.reshape(-1, self.d_model)

    # -- accounting / teardown --------------------------------------------
    @property
    def n_states(self) -> int:
        return len(self.rows)

    def memory_estimate_bytes(self) -> int:
        return self.n_layers * self.n_states * self.d_model * self.dtype.itemsize

    @staticmethod
    def projected_bytes(n_prompts: int, n_positions: int, n_layers: int,
                        d_model: int, itemsize: int = 4) -> int:
        return n_prompts * n_positions * n_layers * d_model * itemsize

    def close(self) -> None:
        """Discard everything. §A.1: extract -> cluster -> assignments -> discard."""
        if self._closed:
            return
        self._buffers = None
        if self._spill_dir is not None:
            for fh in getattr(self, "_files", []):
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._spill_dir = None
        self.rows = []
        self._closed = True

    def __enter__(self) -> "AlignedActivationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ==========================================================================
# 5. Per-layer checkpointing  (§A.1 rule 3)
# ==========================================================================
class LayerCheckpointWriter:
    """Write one checkpoint per layer, and resume from the last completed one.

    §A.1 rule 3: "Checkpoint after every layer. A crash at layer 31 should cost
    you layer 31." §A.1 rule 1: what gets written is *cluster assignments and
    metrics*, never raw activations — assignments for 400 prompts x ~40 layers x
    ~20 positions are a few hundred KB, raw activations are several GB.

    That rule is enforced, not documented: :meth:`write_layer` rejects any array
    whose trailing dimension looks like ``d_model``, any float or COMPLEX array
    bigger than ``max_float_elems``, and the same shapes expressed as nested
    python lists — a ``.tolist()`` is still a raw activation matrix, it is just
    a slower one. The guard RECURSES into nested dicts and lists, because the
    payload is written with ``json.dump`` which recurses too.

    ``d_model`` is REQUIRED. It used to default to ``None``, which silently
    disabled the strongest of the three guards for every caller that forgot it.

    Parameters
    ----------
    prefix:
        Checkpoint filename stem. ``"layer"`` (default) for the per-layer sweep,
        ``"pair"`` for the per-layer-PAIR sweep (Amendment 3). They must differ:
        layer ``p`` and pair ``p`` are different quantities over different
        index ranges, and writing both as ``layer_007.json`` in one run
        directory would have each silently resume from the other.
    """

    def __init__(self, root: str, run_id: str, *, d_model: int,
                 max_float_elems: int = 200_000, prefix: str = "layer"):
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        if d_model is None:
            raise ExtractionError(
                "d_model is required: it is what lets the checkpoint guard "
                "recognise a raw activation matrix (§A.1 rule 1). Passing None "
                "used to disable the check silently.")
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ExtractionError("d_model must be positive, got {}".format(d_model))
        self.max_float_elems = int(max_float_elems)
        if prefix not in ("layer", "pair"):
            raise ExtractionError(
                "prefix must be 'layer' or 'pair', got {!r}".format(prefix))
        self.prefix = prefix

    def path(self, layer: int) -> str:
        return os.path.join(self.dir, "{}_{:03d}.json".format(self.prefix, layer))

    def is_done(self, layer: int) -> bool:
        return os.path.exists(self.path(layer))

    def completed_layers(self) -> List[int]:
        out = []
        stem = self.prefix + "_"
        for p in glob.glob(os.path.join(self.dir, stem + "*.json")):
            base = os.path.basename(p)
            try:
                out.append(int(base[len(stem):-len(".json")]))
            except ValueError:
                continue
        return sorted(out)

    # -- the §A.1 rule-1 guard --------------------------------------------
    @staticmethod
    def _scan_sequence(val: Any) -> Tuple[int, bool, set, int]:
        """-> (n_leaves, has_float_leaf, innermost_lengths, depth).

        ``json.dump`` will happily serialise a nested list of floats, so the
        guard has to see the same structure json does. A ``.tolist()`` on an
        activation matrix arrives here as depth-2 nesting whose innermost
        length is ``d_model``.
        """
        n_leaves = 0
        has_float = False
        inner: set = set()
        depth = 0

        def rec(v, d):
            nonlocal n_leaves, has_float, depth
            if isinstance(v, (list, tuple)):
                depth = max(depth, d + 1)
                if v and not any(isinstance(x, (list, tuple, dict)) for x in v):
                    inner.add(len(v))
                for item in v:
                    rec(item, d + 1)
            elif isinstance(v, dict):
                depth = max(depth, d + 1)
                for item in v.values():
                    rec(item, d + 1)
            else:
                n_leaves += 1
                if isinstance(v, float) or (
                        np is not None and isinstance(v, (np.floating, np.complexfloating))):
                    has_float = True
                elif isinstance(v, complex):
                    has_float = True

        rec(val, 0)
        return n_leaves, has_float, inner, depth

    def _guard_value(self, key: str, val: Any) -> None:
        if np is not None and isinstance(val, np.ndarray):
            # complex included: np.linalg.eig on a non-symmetric transition
            # matrix returns complex128, whose .kind is "c", not "f".
            is_floatish = val.dtype.kind in ("f", "c")
            if val.ndim >= 2 and val.shape[-1] == self.d_model:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: trailing dimension is d_model="
                    "{}, i.e. this is a raw activation matrix. DESIGN.md §A.1 "
                    "rule 1 — extract, cluster, write ASSIGNMENTS, discard."
                    .format(key, self.d_model))
            if val.ndim == 1 and is_floatish and val.size == self.d_model:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: a length-{} float vector is a "
                    "single raw activation (d_model={}). §A.1 rule 1."
                    .format(key, val.size, self.d_model))
            if is_floatish and val.size > self.max_float_elems:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: {} float/complex elements "
                    "exceeds the {} guard. Checkpoints hold assignments and "
                    "metrics, not activations (§A.1 rule 1).".format(
                        key, val.size, self.max_float_elems))
            return

        if isinstance(val, dict):
            for sub, subval in val.items():
                self._guard_value("{}.{}".format(key, sub), subval)
            return

        if isinstance(val, (list, tuple)):
            n_leaves, has_float, inner, depth = self._scan_sequence(val)
            if depth >= 2 and self.d_model in inner:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: a nested list whose innermost "
                    "length is d_model={} is a raw activation matrix that has "
                    "been through .tolist(). §A.1 rule 1 — write ASSIGNMENTS, "
                    "not activations.".format(key, self.d_model))
            if depth == 1 and has_float and self.d_model in inner:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: a flat list of {} floats is a "
                    "single raw activation (d_model={}). §A.1 rule 1."
                    .format(key, self.d_model, self.d_model))
            if has_float and n_leaves > self.max_float_elems:
                raise ExtractionError(
                    "refusing to checkpoint {!r}: {} float elements exceeds the "
                    "{} guard. Checkpoints hold assignments and metrics, not "
                    "activations (§A.1 rule 1).".format(
                        key, n_leaves, self.max_float_elems))
            # a list can still hold dicts/arrays deeper down
            for i, item in enumerate(val):
                if isinstance(item, (dict,)) or (
                        np is not None and isinstance(item, np.ndarray)):
                    self._guard_value("{}[{}]".format(key, i), item)
            return

    def _guard(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ExtractionError("checkpoint payload must be a dict")
        for key, val in payload.items():
            self._guard_value(str(key), val)

    @staticmethod
    def _jsonable(val: Any) -> Any:
        if np is not None and isinstance(val, np.ndarray):
            return val.tolist()
        if np is not None and isinstance(val, (np.integer, np.floating)):
            return val.item()
        if np is not None and isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, dict):
            return {k: LayerCheckpointWriter._jsonable(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [LayerCheckpointWriter._jsonable(v) for v in val]
        return val

    def write_layer(self, layer: int, payload: Dict[str, Any]) -> str:
        """Atomically write one layer's (or pair's) checkpoint. Returns the path."""
        self._guard(payload)
        body: Dict[str, Any] = {self.prefix: layer}
        if self.prefix == "pair":
            # Amendment 3: make the convention explicit in the artifact itself,
            # so a downstream reader cannot mistake pair p for layer p.
            body["source_layer"] = layer
            body["target_layer"] = layer + 1
        for key, val in payload.items():
            body[key] = self._jsonable(val)
        tmp = self.path(layer) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(layer))   # atomic: a preempted pod never
        return self.path(layer)             # leaves a half-written checkpoint

    def read_layer(self, layer: int) -> Dict[str, Any]:
        with open(self.path(layer), "r", encoding="utf-8") as fh:
            return json.load(fh)


def stream_layers_with_checkpoints(
    store: "AlignedActivationStore",
    writer: "LayerCheckpointWriter",
    fn,
    *,
    resume: bool = True,
    log=None,
    release: bool = False,
) -> List[int]:
    """Drive the layer loop: for each layer, compute, checkpoint, free.

    ``fn(batch: LayerBatch) -> dict`` returns the per-layer payload (cluster
    assignments, k, eigengap, lambda2, ...). It is called once per layer, with
    only that layer's states resident. Layers already checkpointed are skipped
    when ``resume=True`` (§A.1 rule 3).

    The skip set is computed BEFORE the sweep and the skipped layers are never
    asked for, so a resume does not pay to concatenate (or, on the spill
    backend, to read off disk) forty layers it is about to discard. That cost
    is exactly what §A.1 rule 3 exists to avoid.

    ``release=True`` frees each layer as it is consumed. Default False, and
    deliberately so: a released layer cannot be re-read, which makes a
    subsequent :meth:`AlignedActivationStore.iter_layer_pairs` sweep — i.e.
    lambda2 and path entropy, both TWO-layer statistics (Amendment 3) —
    impossible. Only pass True when this single-layer sweep is the last thing
    that touches the store.

    Returns the list of layer indices actually computed this call.
    """
    done = set(writer.completed_layers()) if resume else set()
    wanted = [i for i in range(store.n_layers) if i not in done]
    if log:
        for i in sorted(done & set(range(store.n_layers))):
            log("layer {:3d}: checkpoint exists, skipping".format(i))

    computed: List[int] = []
    for batch in store.iter_layers(wanted, release=release):
        payload = fn(batch)
        writer.write_layer(batch.layer, payload)
        computed.append(batch.layer)
        if log:
            log("layer {:3d}: {} states -> {}".format(
                batch.layer, batch.n_states, writer.path(batch.layer)))
        del batch
    return computed


def stream_layer_pairs_with_checkpoints(
    store: "AlignedActivationStore",
    writer: "LayerCheckpointWriter",
    fn,
    *,
    resume: bool = True,
    log=None,
    release: bool = False,
) -> List[int]:
    """Drive the layer-PAIR loop: for each pair, compute, checkpoint, free.

    ``fn(pair: LayerPair) -> dict`` returns the per-pair payload (lambda2, path
    entropy, transition counts, ...). Pair ``p`` = (output of block ``p``,
    output of block ``p+1``), ``p`` in ``0 .. n_layers-2`` — Amendment 3.

    ``writer`` must have been built with ``prefix="pair"``. A pair checkpoint
    written into the layer namespace would make layer ``p`` and pair ``p``
    resume from each other's file, which is §A.11 item 2 with a filesystem
    attached.
    """
    if getattr(writer, "prefix", "layer") != "pair":
        raise ExtractionError(
            "stream_layer_pairs_with_checkpoints needs a LayerCheckpointWriter "
            "built with prefix='pair'; got prefix={!r}. Pair p and layer p are "
            "different quantities over different index ranges (pairs run 0..L-2, "
            "layers 0..L-1) and must not share checkpoint filenames."
            .format(getattr(writer, "prefix", "layer")))

    all_pairs = layer_pair_indices(store.n_layers)
    done = set(writer.completed_layers()) if resume else set()
    wanted = [p for p in all_pairs if p not in done]
    if log:
        for p in sorted(done & set(all_pairs)):
            log("pair  {:3d}: checkpoint exists, skipping".format(p))

    computed: List[int] = []
    for lp in store.iter_layer_pairs(wanted, release=release):
        payload = fn(lp)
        writer.write_layer(lp.pair, payload)
        computed.append(lp.pair)
        if log:
            log("pair  {:3d} (blocks {}->{}): {} states -> {}".format(
                lp.pair, lp.pair, lp.pair + 1, lp.n_states, writer.path(lp.pair)))
        del lp
    return computed
