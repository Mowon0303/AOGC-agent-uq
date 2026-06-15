"""Selective control (H4): turn fused uncertainty into actions and measure payoff.

Two pieces:
  - ``decide_action``: per-step policy mapping uncertainty -> {continue, verify,
    escalate} via two thresholds (verify = re-read/re-sample; escalate = ask
    human / abstain).
  - ``success_vs_budget``: the evaluation curve. Under a fixed "ask-human" budget
    we intervene on the most-uncertain items; a human fixes them (prob fix_prob).
    A better signal raises task success at lower budget -> larger area.
"""

from __future__ import annotations

import numpy as np


def decide_action(uncertainty: float, verify_at: float = 0.4, escalate_at: float = 0.7) -> str:
    """Map a fused uncertainty to a control action."""
    if uncertainty >= escalate_at:
        return "escalate"      # ask human / abstain
    if uncertainty >= verify_at:
        return "verify"        # re-read observation / re-sample the step
    return "continue"


def success_vs_budget(uncertainty, success, fix_prob: float = 1.0):
    """Curve of task success rate vs fraction of items handed to a human.

    uncertainty : per-item fused uncertainty (higher -> intervene first).
    success     : 1 if the item (trajectory) originally succeeded.
    fix_prob    : probability a human intervention turns a failure into success.

    Returns (budgets[n+1] in [0,1], success_rate[n+1]). budgets[0]=0 is the
    no-intervention baseline; budgets[-1]=1 asks on everything.
    """
    u = np.asarray(uncertainty, dtype=float)
    s = np.asarray(success).astype(float)
    n = s.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    u = np.where(np.isfinite(u), u, -np.inf)   # NaN uncertainty -> asked last
    order = np.argsort(-u, kind="mergesort")    # most uncertain first
    s_sorted = s[order]
    cum = float(s.sum())
    rates = [cum / n]
    for i in range(n):
        if s_sorted[i] == 0:
            cum += fix_prob                     # asked a failure -> (maybe) fixed
        rates.append(cum / n)
    budgets = np.arange(0, n + 1) / n
    return budgets, np.array(rates)


def area_under_success_budget(uncertainty, success, fix_prob: float = 1.0) -> float:
    """Area under the success-vs-budget curve (higher = better triage). NaN if empty."""
    b, r = success_vs_budget(uncertainty, success, fix_prob)
    if b.size < 2:
        return float("nan")
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(r, b))
