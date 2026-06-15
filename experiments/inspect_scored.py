"""Diagnose a scored .jsonl: did responses parse? do AOGC's differentiators fire?

Run (Mac after downloading the jsonl, or in Colab):
    python3 experiments/inspect_scored.py runs/scored_Qwen2.5-3B-Instruct.jsonl

Answers the two questions from the first real run:
  - why so few judged errors? (judge verdict distribution)
  - why AOGC == FRANQ? (are reasoning/action parsed; does action-legality /
    numbers / cross-step ever fire — the three things AOGC adds over FRANQ)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, aogc_step_score
from aogc_uq.aogc.claims import extract_action_target
from aogc_uq.baselines import franq_as_agent_uncertainty
from aogc_uq.rollout import load_scored


def main(path):
    trajs = load_scored(path)
    nli = LexicalNLI()
    steps = [s for tr in trajs for s in tr.steps]
    n = len(steps)
    n_lab = sum(1 for s in steps if s.is_error is not None)
    n_err = sum(1 for s in steps if s.is_error)

    empty_reason = 0
    action_parsed = 0          # response had a parseable action
    legality_applicable = 0    # action targets an element (legality term active)
    has_prior = 0              # cross-step context available
    aogc_neq_franq = 0
    examples = []

    for s in steps:
        a = aogc_step_score(s, nli=nli)
        f = franq_as_agent_uncertainty(s, nli=nli)
        if not s.reasoning.strip():
            empty_reason += 1
        if s.action.strip():
            action_parsed += 1
        if extract_action_target(s.action):
            legality_applicable += 1
        if s.prior_observations:
            has_prior += 1
        if abs(a["score"] - f) > 1e-9:
            aogc_neq_franq += 1
        examples.append((s, a, f))

    print("=" * 76)
    print(f"  INSPECT  {os.path.basename(path)}")
    print(f"  steps={n}  labeled={n_lab}  judged_errors={n_err} "
          f"({n_err/max(n_lab,1):.0%})")
    print("=" * 76)
    print("  --- why AOGC == FRANQ? (AOGC's 3 differentiators) ---")
    print(f"  empty reasoning (parse failed) : {empty_reason}/{n}"
          f"  <- if high, model didn't emit <think>; both signals see nothing")
    print(f"  action parsed (non-empty)      : {action_parsed}/{n}")
    print(f"  action targets an element      : {legality_applicable}/{n}"
          f"  <- AOGC's action-legality term only fires here")
    print(f"  has prior_observations         : {has_prior}/{n}"
          f"  <- AOGC's cross-step term; MIRAGE GUI snapshots = 0 (single decision point)")
    print(f"  steps where AOGC != FRANQ      : {aogc_neq_franq}/{n}")
    print("=" * 76)

    # show a few examples: prefer judged errors, then some non-errors
    errs = [e for e in examples if e[0].is_error]
    oks = [e for e in examples if e[0].is_error is False]
    show = errs[:3] + oks[:2]
    for s, a, f in show:
        print(f"\n  [{'ERROR' if s.is_error else 'ok'}] setting={s.meta.get('setting')} "
              f"env={s.meta.get('environment')}")
        print(f"    judge: {s.meta.get('judge', {})}")
        print(f"    raw_response[:260]: {s.meta.get('raw_response','')[:260]!r}")
        print(f"    parsed reasoning[:160]: {s.reasoning[:160]!r}")
        print(f"    parsed action: {s.action[:120]!r}")
        print(f"    AOGC={a['score']:.3f} FRANQ={f:.3f}  breakdown={a['breakdown']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 experiments/inspect_scored.py <scored.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
