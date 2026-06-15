"""Blind-spot slicing — the core of the H1 (blind-spot exists) demonstration.

The blind-spot demo asks: *on grounding-induced failures specifically*, how well
does a signal separate them from correct steps? We build a clean slice whose
positives are errors of the target type (default GROUNDING) and whose negatives
are correct steps, dropping errors of *other* types so the contrast isolates the
target failure mode. H1 predicts existing signals score ~0.5 here while AOGC is
high.
"""

from __future__ import annotations

import numpy as np

from ..data.schema import ErrorType
from .detection import auroc


def _as_error_type_array(error_types):
    out = []
    for e in error_types:
        out.append(e.value if isinstance(e, ErrorType) else str(e))
    return np.asarray(out, dtype=object)


def slice_mask(is_error, error_types, target_type=ErrorType.GROUNDING):
    """Boolean keep-mask + binary labels for the target-type-vs-correct slice.

    Returns (mask, labels): ``mask`` selects correct steps and target-type errors;
    ``labels`` is 1 for target-type errors, 0 for correct steps (defined over the
    *full* input length; index it with ``mask``).
    """
    is_error = np.asarray(is_error).astype(bool)
    et = _as_error_type_array(error_types)
    target = target_type.value if isinstance(target_type, ErrorType) else str(target_type)

    is_target_err = is_error & (et == target)
    is_correct = ~is_error
    mask = is_target_err | is_correct
    labels = is_target_err.astype(int)
    return mask, labels


def blindspot_auroc(scores, is_error, error_types, target_type=ErrorType.GROUNDING) -> float:
    """AUROC restricted to {target-type errors vs correct steps}."""
    scores = np.asarray(scores, dtype=float)
    mask, labels = slice_mask(is_error, error_types, target_type)
    return auroc(scores[mask], labels[mask])
