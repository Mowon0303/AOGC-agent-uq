"""§3.5b FRANQ漏检对比 — the anti-collision experiment (defends arXiv 2505.21072).

Goal: show that porting FRANQ's RAG-faithfulness mechanism naively to agent steps
is a STRONG baseline on single-observation grounding failures, but is
structurally blind to agent-specific failure modes that AOGC catches. The
headline number is ΔAUROC(AOGC − FRANQ-as-agent) on the agent-specific slice.

Failure subtypes (synthetic, controlled so each isolates ONE dimension):
  correct_single     : grounded in current obs, correct                  (negative)
  correct_crossstep  : correctly cites a PRIOR-obs object (not in o_t)   (negative)
                       -> FRANQ (o_t only) false-positives; AOGC uses prior context
  ground_single      : fabricates an object in the REASONING             (positive)
                       -> BOTH should catch (FRANQ is a fair, strong baseline here)
  ground_action      : reasoning fully grounded, but the ACTION targets a
                       nonexistent object                                (positive)
                       -> FRANQ reads reasoning only -> BLIND; AOGC has the action term

SYNTHETIC sanity/contrast harness, NOT paper results. Real numbers come from
MIRAGE-Bench rollouts through this same comparison.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, aogc_step_score
from aogc_uq.baselines import franq_as_agent_uncertainty
from aogc_uq.data.schema import ErrorType, Step
from aogc_uq.metrics import auroc

_BTNS = ["Submit", "Cancel", "Next", "Back", "Save", "Edit", "Delete", "Search"]
_GHOST_BTNS = ["Login", "Logout", "Profile", "Upload", "Export", "Admin"]
_FILES = ["report.pdf", "data.csv", "config.yaml", "notes.txt", "model.ckpt",
          "index.html", "log.txt", "schema.sql"]


def _page(rng):
    btns = rng.choice(_BTNS, size=3, replace=False).tolist()
    return f"The page shows buttons: {', '.join(btns)}. Status code 200.", btns


def make_dataset(n_per=90, seed=1):
    rng = np.random.default_rng(seed)
    steps = []

    def add(reasoning, action, obs, avail, is_err, etype, subtype, prior=None):
        steps.append(Step(
            index=0, observation=obs, reasoning=reasoning, action=action,
            available_objects=avail, prior_observations=prior or [],
            is_error=is_err, error_type=etype, meta={"subtype": subtype},
        ))

    for _ in range(n_per):
        # correct_single
        obs, btns = _page(rng)
        add(f"The page shows the {btns[0]} and {btns[1]} buttons.",
            f'click("{btns[0]}")', obs, btns, False, ErrorType.NONE, "correct_single")

        # correct_crossstep: cites a file that appeared in a PRIOR observation
        # (extractable entity -> exercises AOGC's prior-aware entity traceability,
        #  which FRANQ's current-obs-only port structurally lacks)
        obs, btns = _page(rng)
        f = str(rng.choice(_FILES))
        prior_obs = f"Earlier the tool returned the file {f}. Status code 200."
        add(f"I will reuse {f} from the earlier step.",
            f'click("{btns[0]}")', obs, btns, False, ErrorType.NONE,
            "correct_crossstep", prior=[prior_obs])

        # ground_single: fabricates an object in the REASONING text
        obs, btns = _page(rng)
        gb = str(rng.choice(_GHOST_BTNS))
        fake_n = int(rng.integers(100, 999))
        add(f"I can see a {gb} button and a counter showing {fake_n} items.",
            f'click("{btns[0]}")', obs, btns, True, ErrorType.GROUNDING,
            "ground_single")

        # ground_action: reasoning grounded, ACTION targets a nonexistent object
        obs, btns = _page(rng)
        gb = str(rng.choice(_GHOST_BTNS))
        add(f"The page shows the {btns[0]} and {btns[1]} buttons as expected.",
            f'click("{gb}")', obs, btns, True, ErrorType.GROUNDING, "ground_action")

    return steps


def _auroc_on(scores, steps, pos_subtypes, neg_subtypes):
    keep, labels = [], []
    sub = [s.meta["subtype"] for s in steps]
    for i, st in enumerate(sub):
        if st in pos_subtypes:
            keep.append(i); labels.append(1)
        elif st in neg_subtypes:
            keep.append(i); labels.append(0)
    keep = np.array(keep)
    return auroc(np.asarray(scores)[keep], labels)


def _fp_rate(scores, steps, subtype, thresh):
    s = np.array([sc for sc, st in zip(scores, steps) if st.meta["subtype"] == subtype])
    return float((s > thresh).mean()) if s.size else float("nan")


def main():
    nli = LexicalNLI()
    steps = make_dataset()
    aogc = [aogc_step_score(s, nli=nli)["score"] for s in steps]
    franq = [franq_as_agent_uncertainty(s, nli=nli) for s in steps]

    correct = {"correct_single", "correct_crossstep"}
    print("=" * 78)
    print("  FRANQ-as-agent  vs  AOGC   (§3.5b 漏检对比 — SYNTHETIC, not paper results)")
    print(f"  {len(steps)} steps; 4 subtypes x {len(steps)//4}")
    print("=" * 78)
    print(f"  {'slice (positives vs negatives)':<42}{'FRANQ':>9}{'AOGC':>9}{'Δ':>8}")
    print("  " + "-" * 66)

    slices = [
        ("ALL grounding errors  vs  all correct",
         {"ground_single", "ground_action"}, correct),
        ("single-obs grounding  vs  correct  (fair zone)",
         {"ground_single"}, {"correct_single"}),
        ("★ action-legality grounding  vs  correct  (FRANQ blind)",
         {"ground_action"}, {"correct_single"}),
    ]
    for name, pos, neg in slices:
        f = _auroc_on(franq, steps, pos, neg)
        a = _auroc_on(aogc, steps, pos, neg)
        print(f"  {name:<42}{f:>9.3f}{a:>9.3f}{a - f:>+8.3f}")

    print("  " + "-" * 66)
    # FRANQ false-positives on correct cross-step citations
    f_fp_cross = _fp_rate(franq, steps, "correct_crossstep", 0.2)
    a_fp_cross = _fp_rate(aogc, steps, "correct_crossstep", 0.2)
    f_fp_single = _fp_rate(franq, steps, "correct_single", 0.2)
    print(f"  false-positive rate on correct CROSS-STEP citations:"
          f"  FRANQ={f_fp_cross:.2f}  AOGC={a_fp_cross:.2f}")
    print(f"  false-positive rate on correct single-obs steps:    "
          f"  FRANQ={f_fp_single:.2f}")
    print("=" * 78)

    a_action = _auroc_on(aogc, steps, {"ground_action"}, {"correct_single"})
    f_action = _auroc_on(franq, steps, {"ground_action"}, {"correct_single"})
    print("\nReading (synthetic):")
    print(f"  - On single-obs grounding, FRANQ-as-agent is a STRONG baseline (≈AOGC).")
    print(f"  - On action-legality grounding, FRANQ is blind (AUROC={f_action:.2f}) while")
    print(f"    AOGC catches it (AUROC={a_action:.2f}); ΔAUROC=+{a_action - f_action:.2f}.")
    print(f"  - FRANQ false-positives on correct cross-step citations "
          f"({f_fp_cross:.0%}), AOGC does not ({a_fp_cross:.0%}).")
    print("  => the delta vs FRANQ is the agent-specific dimensions, exactly as §3.5b argues.")
    assert a_action > f_action + 0.2, "expected AOGC >> FRANQ on action-legality slice"
    assert f_fp_cross > a_fp_cross + 0.2, "expected FRANQ to false-positive cross-step"
    print("\nOK.")


if __name__ == "__main__":
    main()
