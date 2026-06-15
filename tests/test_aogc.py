"""AOGC core tests (offline: LexicalNLI, no model download).
Run: python3 tests/test_aogc.py  or  python3 -m pytest tests/test_aogc.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import (
    AOGCConfig,
    LexicalNLI,
    aogc_step_score,
    entity_traceable,
    extract_entities,
    extract_numbers,
    number_traceable,
    score_trajectory,
    split_assertions,
)
from aogc_uq.data.schema import ErrorType, Step, Trajectory

OBS = ("The search returned 3 files: report.pdf, data.csv, notes.txt. "
       "Folder /home/user/docs contains them. Total size 1,024 KB.")


def test_extract_entities():
    ents = extract_entities('open "report.pdf" then read data.csv at /home/user/docs')
    low = [e.lower() for e in ents]
    assert "report.pdf" in low
    assert "data.csv" in low
    assert any("/home/user/docs" in e for e in low)


def test_extract_numbers():
    nums = extract_numbers("there are 3 files, total 1,024 KB and 50% used")
    assert 3 in nums and 1024 in nums and 50 in nums


def test_split_assertions():
    a = split_assertions("The file shows X. It returned three rows. ok")
    assert len(a) == 2  # 'ok' too short


def test_entity_and_number_traceable():
    ctx = [OBS]
    assert entity_traceable("report.pdf", ctx)
    assert entity_traceable("/home/user/docs", ctx)
    assert not entity_traceable("budget.xlsx", ctx)
    assert number_traceable(1024, ctx)      # comma-normalized
    assert number_traceable(3, ctx)
    assert not number_traceable(99, ctx)


def test_grounded_step_scores_low():
    step = Step(
        index=0, observation=OBS,
        reasoning="The search returned report.pdf and data.csv. There are 3 files.",
        action='read("report.pdf")',
    )
    r = aogc_step_score(step, nli=LexicalNLI())
    assert r["score"] < 0.2, r


def test_hallucinated_step_scores_high():
    step = Step(
        index=1, observation=OBS,
        reasoning="I see budget.xlsx in the results and there are 99 matching rows.",
        action='open("budget.xlsx")',
    )
    r = aogc_step_score(step, nli=LexicalNLI())
    assert r["score"] > 0.5, r
    assert "budget.xlsx" in [e.lower() for e in r["offending"]["entities"]]
    assert 99 in r["offending"]["numbers"]


def test_aogc_separates_grounded_from_hallucinated():
    grounded = Step(index=0, observation=OBS,
                    reasoning="report.pdf and data.csv were returned.",
                    action='read("data.csv")')
    halluc = Step(index=1, observation=OBS,
                  reasoning="The results list invoice.zip with 500 entries.",
                  action='open("invoice.zip")')
    sg = aogc_step_score(grounded, nli=LexicalNLI())["score"]
    sh = aogc_step_score(halluc, nli=LexicalNLI())["score"]
    assert sh > sg


def test_action_legality_term():
    step = Step(
        index=0, observation="A page with a Submit button.",
        reasoning="I will click the login button.",
        action='click("loginButton")',
        available_objects=["Submit", "Cancel"],
    )
    r = aogc_step_score(step, nli=LexicalNLI(),
                        cfg=AOGCConfig(use_nli=False, w_entity=0.0, w_number=0.0))
    assert r["breakdown"]["illegal_action"] == 1.0


def test_score_trajectory_caches_signal():
    traj = Trajectory(task_id="t1", steps=[
        Step(index=0, observation=OBS, reasoning="report.pdf returned.", action="noop"),
        Step(index=1, observation=OBS, reasoning="ghost.bin appeared.", action="noop"),
    ])
    scores = score_trajectory(traj, nli=LexicalNLI())
    assert len(scores) == 2
    assert traj.steps[0].signals["aogc"] == scores[0]
    assert scores[1] > scores[0]


def test_schema_roundtrip():
    traj = Trajectory(task_id="t", benchmark="mirage", steps=[
        Step(index=0, observation="o", reasoning="r", action="a",
             is_error=True, error_type=ErrorType.GROUNDING),
    ])
    d = traj.to_dict()
    assert d["steps"][0]["error_type"] == "grounding"
    back = Trajectory.from_dict(d)
    assert back.steps[0].error_type == ErrorType.GROUNDING
    assert back.task_id == "t"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} aogc tests passed.")


if __name__ == "__main__":
    _run_all()
