"""Selective-prediction metrics: risk-coverage curve and AURC.

The selective predictor abstains on the most-uncertain items first. We sweep a
coverage threshold from 1/n to 1 (keeping the most-confident items) and report
the error rate (risk) among the kept items. A better UQ signal pushes errors to
low coverage, giving a lower area under the risk-coverage curve (AURC).
"""

from __future__ import annotations

import numpy as np


def _trapz(y, x) -> float:
    # numpy 2.x renamed trapz -> trapezoid; support both.
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def risk_coverage_curve(uncertainty, error):
    """Return (coverages, risks).

    uncertainty : higher = abstain earlier.
    error       : 1 if the item is wrong.
    """
    uncertainty = np.asarray(uncertainty, dtype=float)
    error = np.asarray(error).astype(int)
    m = np.isfinite(uncertainty)
    uncertainty, error = uncertainty[m], error[m]
    n = uncertainty.size
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(uncertainty, kind="mergesort")  # ascending: keep most-certain
    err_sorted = error[order]
    counts = np.arange(1, n + 1)
    coverages = counts / n
    risks = np.cumsum(err_sorted) / counts
    return coverages, risks


def aurc(uncertainty, error) -> float:
    """Area under the risk-coverage curve (lower = better). NaN if empty."""
    coverages, risks = risk_coverage_curve(uncertainty, error)
    if coverages.size == 0:
        return float("nan")
    if coverages.size == 1:
        return float(risks[0])
    return _trapz(risks, coverages)
