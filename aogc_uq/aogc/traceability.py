"""Step 2 of AOGC: can each claim be traced back to the real observation(s)?

* entities  -> normalized substring / word-boundary match against the context.
* numbers   -> exact numeric match (format-insensitive) against numbers in the
               context, with a small relative tolerance for floats.
* assertions-> NLI entailment (max over context windows) >= threshold.

"Context" = the current observation plus prior observations (an agent may
legitimately cite something a few steps back; see Step.grounding_context).
"""

from __future__ import annotations

import re

from .claims import extract_numbers
from .nli import NLIScorer

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def _norm_entity(e: str) -> str:
    return _PUNCT_EDGE.sub("", normalize(e))


def entity_traceable(entity: str, contexts: list[str]) -> bool:
    """True if the entity appears in any context (case/whitespace-insensitive)."""
    ne = _norm_entity(entity)
    if not ne:
        return True  # empty -> don't penalize
    for c in contexts:
        nc = normalize(c)
        if len(ne) <= 3:
            # short tokens: require word-ish boundary to avoid spurious substrings
            if re.search(r"(?<![\w])" + re.escape(ne) + r"(?![\w])", nc):
                return True
        elif ne in nc:
            return True
    return False


def number_traceable(value: float, contexts: list[str], rel_tol: float = 1e-6) -> bool:
    """True if a numerically-equal value appears in any context."""
    for c in contexts:
        for cand in extract_numbers(c):
            if cand == value:
                return True
            denom = max(abs(value), abs(cand), 1e-9)
            if abs(cand - value) / denom <= rel_tol:
                return True
    return False


def assertion_supported(
    assertion: str, contexts: list[str], nli: NLIScorer, threshold: float = 0.5
) -> tuple[bool, float]:
    """True if some context entails the assertion. Returns (supported, max_prob)."""
    if not contexts:
        return False, 0.0
    probs = nli.entail_probs([(c, assertion) for c in contexts])
    best = max(probs) if probs else 0.0
    return best >= threshold, best
