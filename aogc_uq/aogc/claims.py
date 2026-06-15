"""Step 1 of AOGC: extract the agent's *claims about the observation*.

We split claims into two kinds, verified by two different mechanisms downstream:

* **referents** — hard, literal things the agent names: entities (quoted spans,
  file paths, URLs, code identifiers) and numbers. Verified by string / numeric
  matching against the observation (cheap, high-precision).
* **assertions** — paraphrastic sentences ("the file shows X", "the search
  returned Y"). Verified by NLI entailment against the observation.

This is a deliberately transparent, training-free heuristic (v0). The paper will
ablate entity-only vs +NLI vs +action-legality; this module is the entity/number
+ assertion front-end for all of that.
"""

from __future__ import annotations

import re

# ---- entities --------------------------------------------------------------

_QUOTED = re.compile(r'"([^"\n]{1,200})"|\'([^\'\n]{1,200})\'|`([^`\n]{1,200})`')
_URL = re.compile(r"https?://[^\s)\]\"'>]+")
_PATH = re.compile(r"(?:\.{0,2}/)?(?:[\w.\-]+/){1,}[\w.\-]+")           # a/b/c.txt
_FILE = re.compile(r"\b[\w\-]+\.[A-Za-z]{1,6}\b")                       # config.yaml
_DOTTED = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")        # obj.attr.x
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")               # snake_case
_CAMEL = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b")                     # camelCase
_CONST = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")           # UPPER_CONST


def extract_entities(text: str) -> list[str]:
    """Return de-duplicated literal entities the agent named (order-preserving)."""
    if not text:
        return []
    found: list[str] = []
    for m in _QUOTED.finditer(text):
        found.append(next(g for g in m.groups() if g is not None))
    for rx in (_URL, _PATH, _FILE, _DOTTED, _SNAKE, _CAMEL, _CONST):
        found.extend(rx.findall(text))
    seen, out = set(), []
    for e in found:
        e = e.strip().strip(".,;:")
        key = e.lower()
        if len(e) >= 2 and key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ---- numbers ---------------------------------------------------------------

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"   # 1,234.5  $1,000
    r"|(?<![\w.])[-+]?\$?\d+(?:\.\d+)?%?"                  # 42  3.14  50%  $9.99
)


def _to_float(token: str):
    t = token.replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """Return numeric values the agent stated (commas/%/$ normalized away)."""
    if not text:
        return []
    vals, seen = [], set()
    for tok in _NUMBER.findall(text):
        v = _to_float(tok)
        if v is not None and v not in seen:
            seen.add(v)
            vals.append(v)
    return vals


# ---- assertions ------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# cue words that bias a sentence toward being an observation-attributed claim
_OBS_CUES = (
    "show", "shows", "showed", "display", "displays", "return", "returned",
    "returns", "contain", "contains", "found", "indicates", "says", "states",
    "see", "saw", "output", "result", "results", "response", "exists", "is ",
    "are ", "lists", "reported", "according",
)


def split_assertions(text: str, min_len: int = 12, cue_filter: bool = False) -> list[str]:
    """Split reasoning into candidate assertions for NLI verification.

    cue_filter=True keeps only sentences with an observation cue word (higher
    precision, lower recall); default keeps all non-trivial sentences.
    """
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    out = []
    for s in sents:
        if len(s) < min_len:
            continue
        if cue_filter and not any(c in s.lower() for c in _OBS_CUES):
            continue
        out.append(s)
    return out


# ---- action targets (for the action-legality term) -------------------------

# verbs whose first quoted argument is an ELEMENT the observation must contain.
# free-text / navigation verbs (send_msg_to_user, report_infeasible, goto, scroll,
# stop, answer, respond, noop, go_back, ...) reference no element -> no legality check.
_ELEMENT_VERBS = {
    "click", "dblclick", "double_click", "fill", "type", "press", "hover",
    "focus", "clear", "check", "uncheck", "select_option", "set_value",
    "drag_and_drop", "tap", "scroll_to", "mouse_click",
}
_ACTION_CALL = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.S)
_ARG_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def extract_action_target(action: str) -> list[str]:
    """Return the element id(s) an action targets, or [] if it targets no element.

    Only element-targeting verbs contribute; the element is the FIRST quoted arg
    (``click('a12')`` -> ['a12']; ``fill('a12','text')`` -> ['a12']). A free-text
    action like ``send_msg_to_user('...')`` returns [] (no legality penalty).
    """
    if not action:
        return []
    m = _ACTION_CALL.search(action.strip())
    if not m:
        return []
    verb, args = m.group(1).lower(), m.group(2)
    if verb not in _ELEMENT_VERBS:
        return []
    for g in _ARG_QUOTED.findall(args):
        val = g[0] or g[1]
        if val:
            return [val]
    return []


def extract_referents(*texts: str) -> dict[str, list]:
    """Convenience: merge entities + numbers across several text fields."""
    entities, numbers = [], []
    seen_e, seen_n = set(), set()
    for t in texts:
        for e in extract_entities(t):
            if e.lower() not in seen_e:
                seen_e.add(e.lower())
                entities.append(e)
        for n in extract_numbers(t):
            if n not in seen_n:
                seen_n.add(n)
                numbers.append(n)
    return {"entities": entities, "numbers": numbers}
