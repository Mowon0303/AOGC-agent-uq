"""Error-detection metrics.

Convention used everywhere in this repo: a UQ *score* is an **uncertainty**
(higher = more likely the step/trajectory is an error). So the positive class
for AUROC/AUPRC is ``is_error == 1``, and a good signal ranks errors high.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _clean(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")
    finite = np.isfinite(scores)
    return scores[finite], labels[finite]


def auroc(scores, labels) -> float:
    """AUROC of uncertainty vs is_error. NaN if only one class present."""
    scores, labels = _clean(scores, labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auprc(scores, labels) -> float:
    """Average precision (area under PR curve) for the error (positive) class."""
    scores, labels = _clean(scores, labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))
