"""FRANQ-as-agent baseline tests + the AOGC-vs-FRANQ distinctions (§3.5b).
Run: python3 tests/test_franq.py  or  python3 -m pytest tests/test_franq.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, aogc_step_score
from aogc_uq.baselines import franq_as_agent_uncertainty
from aogc_uq.data.schema import ErrorType, Step

NLI = LexicalNLI()
PAGE = "The page shows buttons: Submit, Cancel, Next. Status code 200."


def _franq(step):
    return franq_as_agent_uncertainty(step, nli=NLI)


def _aogc(step):
    return aogc_step_score(step, nli=NLI)["score"]


def test_franq_catches_single_obs_fabrication():
    step = Step(index=0, observation=PAGE, available_objects=["Submit", "Cancel", "Next"],
                reasoning="I will reuse ghost_file.bin from the results.",
                action='click("Submit")')
    assert _franq(step) > 0.4


def test_franq_blind_to_action_only_ghost():
    # reasoning fully grounded; only the ACTION targets a nonexistent object.
    step = Step(index=0, observation=PAGE, available_objects=["Submit", "Cancel", "Next"],
                reasoning="The page shows the Submit and Cancel buttons as expected.",
                action='click("Login")', is_error=True, error_type=ErrorType.GROUNDING)
    assert _franq(step) < 0.2          # FRANQ reads reasoning only -> blind
    assert _aogc(step) > 0.3           # AOGC has the action-legality term -> catches


def test_franq_false_positives_on_crossstep_but_aogc_does_not():
    prior = "Earlier the tool returned the file config.yaml. Status code 200."
    step = Step(index=2, observation=PAGE, prior_observations=[prior],
                available_objects=["Submit", "Cancel", "Next"],
                reasoning="I will reuse config.yaml from the earlier step.",
                action='click("Submit")', is_error=False, error_type=ErrorType.NONE)
    assert _franq(step) > 0.4          # current-obs-only -> flags a CORRECT step
    assert _aogc(step) < 0.2           # prior-aware context -> stays clean


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} franq tests passed.")


if __name__ == "__main__":
    _run_all()
