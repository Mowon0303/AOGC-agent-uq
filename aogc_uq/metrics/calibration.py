"""Calibration metrics.

These take a **confidence** = predicted P(correct) in [0, 1] and a binary
``correct`` label. To go from an uncertainty score u to a confidence, use
``confidence = 1 - normalize(u)`` (see ``aogc_uq.fusion``); calibration is only
meaningful for a probability, not an arbitrary score.
"""

from __future__ import annotations

import numpy as np


def _clean(conf, correct):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct).astype(int)
    if conf.shape != correct.shape:
        raise ValueError(f"shape mismatch: conf {conf.shape} vs correct {correct.shape}")
    m = np.isfinite(conf)
    return np.clip(conf[m], 0.0, 1.0), correct[m]


def reliability_bins(conf, correct, n_bins: int = 10):
    """Return per-bin (avg_confidence, accuracy, count) for a reliability diagram."""
    conf, correct = _clean(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            out.append((float("nan"), float("nan"), 0))
        else:
            out.append((float(conf[in_bin].mean()), float(correct[in_bin].mean()), cnt))
    return out


def ece(conf, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins, L1)."""
    conf, correct = _clean(conf, correct)
    if conf.size == 0:
        return float("nan")
    n = conf.size
    total = 0.0
    for avg_conf, acc, cnt in reliability_bins(conf, correct, n_bins):
        if cnt == 0:
            continue
        total += (cnt / n) * abs(avg_conf - acc)
    return float(total)


def brier(conf, correct) -> float:
    """Brier score = mean squared error between confidence and correctness."""
    conf, correct = _clean(conf, correct)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - correct) ** 2))


def overconfidence(conf, correct) -> float:
    """Mean predicted confidence minus empirical accuracy.

    Positive => systematically overconfident (the agentic-UQ failure mode this
    project is about). See Agentic Overconfidence (arXiv 2602.06948).
    """
    conf, correct = _clean(conf, correct)
    if conf.size == 0:
        return float("nan")
    return float(np.mean(conf) - np.mean(correct))
