"""Verbalized-confidence baseline (Tian 2023; Lin 2022).

Black-box: works on closed API models too (H3). Uncertainty = 1 - confidence.
Confidence is read from ``step.meta['verbalized_confidence']`` if present, else
parsed from the reasoning/action text.
"""

from __future__ import annotations

import re

from ..data.schema import Step

_PATTERNS = [
    re.compile(r"confidence\s*[:=]?\s*([01]?\.\d+|\d{1,3})\s*(%?)", re.I),
    re.compile(r"\b(\d{1,3})\s*%\s*(?:confiden|sure|certain|probab)", re.I),
    re.compile(r"\bI\s+am\s+(\d{1,3})\s*%", re.I),
    re.compile(r"\b(?:confident|certain|sure)\s*[:=]?\s*([01]?\.\d+)\b", re.I),
]


def parse_confidence(text: str, default: float | None = None) -> float | None:
    """Return confidence in [0,1], or ``default`` if none found."""
    if not text:
        return default
    for rx in _PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        val = float(m.group(1))
        is_pct = (m.lastindex and m.group(m.lastindex) == "%") or val > 1.0
        if is_pct:
            val /= 100.0
        return max(0.0, min(1.0, val))
    return default


def verbalized_uncertainty(step: Step, default: float = 0.5) -> float:
    conf = step.meta.get("verbalized_confidence")
    if conf is None:
        conf = parse_confidence(f"{step.reasoning} {step.action}", default=None)
    if conf is None:
        return default
    if conf > 1.0:
        conf /= 100.0
    return 1.0 - max(0.0, min(1.0, conf))
