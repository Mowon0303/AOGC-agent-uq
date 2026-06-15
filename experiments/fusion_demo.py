"""H2 (fusion) + H4 (selective control) demo on synthetic data.

Reuses the blind-spot synthetic set (verbalized is blind to grounding failures,
AOGC catches them). Shows:
  H2 : LogReg late-fusion of {AOGC, verbalized} beats either alone on all-error
       AUROC (fit on a dev split, reported on a held-out split).
  H4 : using fused uncertainty to triage trajectories to a human raises task
       success faster per budget than verbalized alone (area under success-budget).

SYNTHETIC plumbing demo, not paper results.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, score_trajectory
from aogc_uq.baselines import verbalized_uncertainty
from aogc_uq.fusion import (
    LogRegFusion,
    area_under_success_budget,
    rank_fuse,
)
from aogc_uq.metrics import auroc
from experiments.blindspot_demo import make_synthetic_dataset


def main():
    trajs = make_synthetic_dataset(n_tasks=160, seed=0)
    nli = LexicalNLI()
    aogc, verb, y, tix = [], [], [], []
    for ti, tr in enumerate(trajs):
        score_trajectory(tr, nli=nli)
        for s in tr.steps:
            aogc.append(s.signals["aogc"])
            verb.append(verbalized_uncertainty(s))
            y.append(int(bool(s.is_error)))
            tix.append(ti)
    aogc, verb, y, tix = map(np.array, (aogc, verb, y, tix))

    # split by trajectory: dev (fit fusion) vs test (report)
    cut = int(len(trajs) * 0.6)
    dev, test = tix < cut, tix >= cut
    fuser = LogRegFusion().fit({"aogc": aogc[dev], "verbalized": verb[dev]}, y[dev])
    fused = fuser.fuse({"aogc": aogc, "verbalized": verb})
    rankf = rank_fuse({"aogc": aogc, "verbalized": verb})

    print("=" * 70)
    print("  FUSION (H2) + SELECTIVE CONTROL (H4)   (SYNTHETIC, not paper results)")
    print(f"  steps={len(y)}  dev={int(dev.sum())}  test={int(test.sum())}")
    print("=" * 70)
    print("  step-level all-error AUROC (held-out test split):")
    for name, sc in [("verbalized", verb), ("AOGC", aogc),
                     ("rank-fuse(AOGC+verb)", rankf), ("logreg-fuse(AOGC+verb)", fused)]:
        print(f"    {name:<26}{auroc(sc[test], y[test]):.3f}")
    print(f"  logreg coefficients (orthogonal contribution): {fuser.coef_}")

    # H4: trajectory-level triage on the test split
    test_tids = [i for i in range(len(trajs)) if i >= cut]
    unc_fused, unc_verb, succ = [], [], []
    for ti in test_tids:
        m = tix == ti
        unc_fused.append(fused[m].max())
        unc_verb.append(verb[m].max())
        succ.append(0 if y[m].any() else 1)        # success = no error step
    oracle = [1.0 - s for s in succ]                # failures first = ceiling

    a_fused = area_under_success_budget(unc_fused, succ)
    a_verb = area_under_success_budget(unc_verb, succ)
    a_oracle = area_under_success_budget(oracle, succ)
    base = float(np.mean(succ))
    print("\n  selective control — area under success-vs-(ask-human)-budget:")
    print(f"    base success rate (no asking): {base:.3f}")
    print(f"    verbalized triage:  {a_verb:.3f}")
    print(f"    fused triage:       {a_fused:.3f}")
    print(f"    oracle (ceiling):   {a_oracle:.3f}")
    print("=" * 70)

    assert auroc(fused[test], y[test]) > auroc(verb[test], y[test]) + 0.05, "fusion should beat verbalized"
    assert a_fused >= a_verb, "fused triage should be >= verbalized triage"
    print("\nOK: fusion improves detection; fused uncertainty triages better than verbalized.")


if __name__ == "__main__":
    main()
