"""Metrics-layer sanity tests. Run: python3 -m pytest tests/test_metrics.py
or just: python3 tests/test_metrics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.data.schema import ErrorType
from aogc_uq.metrics import (
    aurc,
    auroc,
    auprc,
    blindspot_auroc,
    brier,
    ece,
    overconfidence,
    risk_coverage_curve,
)


def test_auroc_perfect_and_random():
    # uncertainty perfectly ranks errors high -> AUROC 1.0
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert abs(auroc(scores, labels) - 1.0) < 1e-9
    # inverted -> 0.0
    assert abs(auroc([0.9, 0.8, 0.2, 0.1], labels) - 0.0) < 1e-9
    # single class -> NaN
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))


def test_auprc_basic():
    ap = auprc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert abs(ap - 1.0) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    # confidence == accuracy in every bin
    conf = [0.0, 0.0, 1.0, 1.0]
    correct = [0, 0, 1, 1]  # conf 1.0 -> always correct, conf 0.0 -> always wrong
    assert ece(conf, correct, n_bins=10) < 1e-9


def test_ece_detects_overconfidence():
    conf = [0.9, 0.9, 0.9, 0.9]
    correct = [1, 0, 0, 0]  # claims 0.9 but only 25% correct
    assert ece(conf, correct, n_bins=10) > 0.5
    assert overconfidence(conf, correct) > 0.5  # mean conf 0.9 - acc 0.25


def test_brier_bounds():
    assert abs(brier([1, 1, 0, 0], [1, 1, 0, 0])) < 1e-9       # perfect
    assert abs(brier([0, 0, 1, 1], [1, 1, 0, 0]) - 1.0) < 1e-9  # worst


def test_risk_coverage_and_aurc():
    # signal that ranks the single error as most uncertain -> low AURC
    uncertainty = [0.1, 0.2, 0.3, 0.99]
    error = [0, 0, 0, 1]
    cov, risk = risk_coverage_curve(uncertainty, error)
    assert cov[0] == 0.25 and risk[0] == 0.0   # most-certain kept first: no error
    assert risk[-1] == 0.25                      # full coverage: 1/4 error rate
    good = aurc(uncertainty, error)
    # a useless (reversed) signal abstains on correct items first -> higher AURC
    bad = aurc([0.99, 0.3, 0.2, 0.1], error)
    assert good < bad


def test_blindspot_slice_isolates_target_type():
    # 2 grounding errors, 1 planning error, 3 correct
    scores = [0.9, 0.8, 0.85, 0.1, 0.2, 0.15]
    is_error = [True, True, True, False, False, False]
    etypes = [
        ErrorType.GROUNDING,
        ErrorType.GROUNDING,
        ErrorType.PLANNING,   # excluded from the grounding slice
        ErrorType.NONE,
        ErrorType.NONE,
        ErrorType.NONE,
    ]
    a = blindspot_auroc(scores, is_error, etypes, ErrorType.GROUNDING)
    # grounding errors (0.9,0.8) all rank above correct (0.1,0.2,0.15) -> AUROC 1.0
    assert abs(a - 1.0) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} metric tests passed.")


if __name__ == "__main__":
    _run_all()
