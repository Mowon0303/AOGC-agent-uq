"""Step-level late-fusion of UQ signals + selective control (H2 fusion, H4 control).

The AOGC claim is *orthogonal augmentation*: fuse u_t^g on top of existing signals
(verbalized, semantic entropy, ...) and the combination beats any single one.
Fusion options:
  - rank_fuse / mean_fuse : training-free (no labels), normalize-and-average.
  - LogRegFusion          : fit 2-4 scalars on a small dev set (no big-model training).

Selective control turns fused uncertainty into actions and measures the payoff
(success rate vs ask-human budget).
"""

from .late_fusion import (
    LogRegFusion,
    mean_fuse,
    minmax,
    rank_fuse,
    zscore,
)
from .control import (
    area_under_success_budget,
    decide_action,
    success_vs_budget,
)

__all__ = [
    "minmax",
    "zscore",
    "rank_fuse",
    "mean_fuse",
    "LogRegFusion",
    "success_vs_budget",
    "area_under_success_budget",
    "decide_action",
]
