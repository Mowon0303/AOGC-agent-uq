"""Hand-label a small GOLD set of is_error labels (no API key needed).

The local judge is unreliable; this lets you create trustworthy labels for a
sample (the plan's "人工抽检 100-200 条"), then reports AOGC vs FRANQ on the
gold-labeled steps. Resumable: labels are saved after every answer.

Run (after downloading the scored jsonl, on Mac or Colab):
    python3 experiments/label_gold.py <scored.jsonl> [gold.json] [--n 60]

For each step you see the goal, the agent's reasoning+action, and (for GUI steps)
whether the clicked element actually exists in the observation. Press:
    y = hallucinated   n = faithful   s = skip   q = quit & save
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, aogc_step_score
from aogc_uq.aogc.claims import extract_action_target
from aogc_uq.baselines import franq_as_agent_uncertainty
from aogc_uq.data.schema import ErrorType
from aogc_uq.metrics import auroc, blindspot_auroc
from aogc_uq.rollout import load_scored


def _show(tid, s):
    print("\n" + "=" * 78)
    print(f"task: {tid[:70]}")
    print(f"setting={s.meta.get('setting')}  env={s.meta.get('environment')}")
    print(f"\nGOAL: {s.meta.get('instruction','')[:200]}")
    print(f"\nAGENT REASONING:\n  {s.reasoning[:500]}")
    print(f"\nAGENT ACTION: {s.action[:160]}")
    tgt = extract_action_target(s.action)
    if tgt:
        avail = {o.lower() for o in s.available_objects}
        miss = [t for t in tgt if t.lower() not in avail]
        if miss:
            print(f"  ⚠ action targets {miss} — NOT among the {len(s.available_objects)} "
                  f"elements in the observation (likely a grounding failure)")
        else:
            print(f"  ✓ action target {tgt} exists in the observation")
    print(f"\nOBSERVATION (head):\n{s.observation[:900]}")
    print("=" * 78)


def main(scored_path, gold_path=None, max_label=60):
    gold_path = gold_path or scored_path + ".gold.json"
    gold = json.load(open(gold_path)) if os.path.exists(gold_path) else {}
    trajs = load_scored(scored_path)
    steps = [(tr.task_id, s) for tr in trajs for s in tr.steps]

    print(f"{len(steps)} steps; {len(gold)} already labeled; target {max_label}.")
    for tid, s in steps:
        if tid in gold or len(gold) >= max_label:
            continue
        _show(tid, s)
        ans = input("[y]hallucinated  [n]faithful  [s]kip  [q]uit+save > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans in ("y", "n"):
            gold[tid] = (ans == "y")
            json.dump(gold, open(gold_path, "w"), indent=1)
    print(f"\nsaved {len(gold)} gold labels -> {gold_path}")

    # evaluate on gold-labeled steps
    nli = LexicalNLI()
    aogc, franq, iserr, et = [], [], [], []
    for tid, s in steps:
        if tid not in gold:
            continue
        s.is_error = gold[tid]
        s.error_type = ErrorType(s.meta.get("setting")) if (gold[tid] and s.meta.get("setting") in
            {e.value for e in ErrorType}) else (ErrorType.GROUNDING if gold[tid] else ErrorType.NONE)
        aogc.append(aogc_step_score(s, nli=nli)["score"])
        franq.append(franq_as_agent_uncertainty(s, nli=nli))
        iserr.append(bool(gold[tid])); et.append(s.error_type)

    n_err = sum(iserr)
    print(f"\ngold: {len(iserr)} labeled, {n_err} hallucinated")
    if n_err and (len(iserr) - n_err):
        print(f"AOGC  all-error AUROC: {auroc(aogc, [int(x) for x in iserr]):.3f}")
        print(f"FRANQ all-error AUROC: {auroc(franq, [int(x) for x in iserr]):.3f}")
        print(f"AOGC  blindspot AUROC: {blindspot_auroc(aogc, iserr, et, ErrorType.GROUNDING):.3f}")
        print(f"FRANQ blindspot AUROC: {blindspot_auroc(franq, iserr, et, ErrorType.GROUNDING):.3f}")
        print("(small n — treat as directional, not final)")
    else:
        print("need both hallucinated and faithful labels for AUROC; label more.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = 60
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    if not args:
        print("usage: python3 experiments/label_gold.py <scored.jsonl> [gold.json] [--n N]")
        sys.exit(1)
    main(args[0], args[1] if len(args) > 1 else None, max_label=n)
