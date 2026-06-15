"""Fusion + selective-control tests (CPU).
Run: python3 tests/test_fusion.py  or  python3 -m pytest tests/test_fusion.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.fusion import (
    LogRegFusion,
    area_under_success_budget,
    decide_action,
    mean_fuse,
    minmax,
    rank_fuse,
    success_vs_budget,
)
from aogc_uq.metrics import auroc


def test_minmax_and_ranges():
    assert np.allclose(minmax([0, 5, 10]), [0.0, 0.5, 1.0])
    out = minmax([3, 1, 2, np.nan])          # NaN passes through (imputed later by fusion)
    finite = out[np.isfinite(out)]
    assert np.all((finite >= 0) & (finite <= 1)) and np.isnan(out[-1])


def test_rank_and_mean_fuse_shapes():
    sig = {"a": [0.1, 0.9, 0.5], "b": [0.2, 0.3, 0.8]}
    for fn in (rank_fuse, mean_fuse):
        out = fn(sig)
        assert out.shape == (3,)
        assert np.all((out >= 0) & (out <= 1))


def _complementary_dataset(seed=0):
    rng = np.random.default_rng(seed)
    n_each = 80
    labels, sigA, sigB = [], [], []
    for kind in ("correct", "errA", "errB"):
        for _ in range(n_each):
            labels.append(0 if kind == "correct" else 1)
            sigA.append((1.0 if kind == "errA" else 0.0) + rng.normal(0, 0.08))
            sigB.append((1.0 if kind == "errB" else 0.0) + rng.normal(0, 0.08))
    return np.array(sigA), np.array(sigB), np.array(labels)


def test_logreg_fusion_beats_each_component():
    a, b, y = _complementary_dataset()
    fuser = LogRegFusion().fit({"aogc": a, "verbalized": b}, y)
    fused = fuser.fuse({"aogc": a, "verbalized": b})
    au_a, au_b, au_f = auroc(a, y), auroc(b, y), auroc(fused, y)
    # each signal sees only half the errors; fusion sees both
    assert au_f > max(au_a, au_b) + 0.1, (au_a, au_b, au_f)
    assert set(fuser.coef_) == {"aogc", "verbalized"}


def test_fusion_handles_nan_column():
    a = np.array([0.1, 0.9, 0.5, np.nan])
    b = np.array([0.2, 0.8, 0.4, 0.6])
    y = np.array([0, 1, 0, 1])
    fuser = LogRegFusion().fit({"a": a, "b": b}, y)
    out = fuser.fuse({"a": a, "b": b})
    assert out.shape == (4,) and np.all(np.isfinite(out))


def test_success_vs_budget_monotonic_and_better_signal_wins():
    success = np.array([1, 1, 1, 0, 0, 0])   # 3 succeed, 3 fail
    good_u = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])   # high on failures -> ask them first
    bad_u = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])    # high on successes -> wasted asks
    b, r = success_vs_budget(good_u, success, fix_prob=1.0)
    assert r[0] == 0.5 and abs(r[-1] - 1.0) < 1e-9      # base 50% -> 100% at full budget
    assert np.all(np.diff(r) >= -1e-9)                  # non-decreasing
    assert area_under_success_budget(good_u, success) > area_under_success_budget(bad_u, success)


def test_decide_action_bands():
    assert decide_action(0.1) == "continue"
    assert decide_action(0.5) == "verify"
    assert decide_action(0.8) == "escalate"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} fusion tests passed.")


if __name__ == "__main__":
    _run_all()
