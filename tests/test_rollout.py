"""Rollout-harness tests (CPU, stub backends — no GPU/API).
Run: python3 tests/test_rollout.py  or  python3 -m pytest tests/test_rollout.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.data import mirage
from aogc_uq.data.schema import ErrorType
from aogc_uq.rollout import (
    EchoGenerator,
    StubJudge,
    flatten_messages,
    load_scored,
    run_rollout,
)
from aogc_uq.rollout.judge import apply_verdict, build_judge_messages

FX = os.path.join(os.path.dirname(__file__), "fixtures", "mirage")


def _traj(fn, setting, env):
    obj = json.load(open(os.path.join(FX, fn)))
    return mirage.parse_snapshot(obj, setting=setting, environment=env)


def _trajs():
    return [
        _traj("webarena_unexpected_transition.json", "unexpected_transition", "webarena"),
        _traj("osworld_popup.json", "popup", "osworld"),
        _traj("taubench_users_questions.json", "users_questions", "taubench"),
    ]


def test_flatten_messages():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                          {"type": "text", "text": "b"},
                                          {"type": "image_url"}]}]
    flat = flatten_messages(msgs)
    assert flat == [{"role": "user", "content": "a\nb"}]


def test_build_judge_messages_includes_observation():
    tr = _traj("webarena_unexpected_transition.json", "unexpected_transition", "webarena")
    mirage.attach_response(tr.steps[0], "<think>x</think><action>click('z')</action>")
    msgs = build_judge_messages(tr.steps[0])
    assert msgs[0]["role"] == "system"
    assert "Observation of current step" in msgs[1]["content"]
    assert "click('z')" in msgs[1]["content"]


def test_apply_verdict_label_mapping():
    tr = _traj("webarena_unexpected_transition.json", "unexpected_transition", "webarena")
    s = tr.steps[0]
    apply_verdict(s, {"hallucinated": True, "rationale": "r"})
    assert s.is_error is True and s.error_type == ErrorType.GROUNDING
    apply_verdict(s, {"hallucinated": False, "rationale": "r"})
    assert s.is_error is False and s.error_type == ErrorType.NONE


def test_run_rollout_dump_resume_roundtrip(tmp_path=None):
    out = "/tmp/aogc_rollout_test.jsonl"
    if os.path.exists(out):
        os.remove(out)
    gen = EchoGenerator("<think>I see a Clone button bid a999</think>\n<action>click('a999')</action>")
    judge = StubJudge(fn=lambda s: "a999" in s.action)  # deterministic -> always True

    run_rollout(_trajs(), gen, judge, out_path=out, model_name="echo")
    with open(out) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 3
    for l in lines:
        assert json.loads(l)["steps"][0]["meta"].get("messages") is None  # stripped

    scored = load_scored(out)
    assert len(scored) == 3
    for tr in scored:
        s = tr.steps[0]
        assert s.is_error is True            # judge fired
        assert s.reasoning and s.action == "click('a999')"
        assert "judge" in s.meta
        assert tr.model == "echo"

    # resume: re-run should add nothing (all task_ids already present)
    run_rollout(_trajs(), gen, judge, out_path=out)
    with open(out) as f:
        assert len([l for l in f if l.strip()]) == 3
    os.remove(out)


def test_judge_verdict_parsing():
    from aogc_uq.rollout.judge import _parse_verdict
    assert _parse_verdict("YES. the agent clicked a nonexistent button.")["hallucinated"] is True
    assert _parse_verdict("NO, the action is faithful to the page.")["hallucinated"] is False
    assert _parse_verdict('{"hallucinated": true, "rationale": "x"}')["hallucinated"] is True
    assert _parse_verdict("The agent was unfaithful to the observation.")["hallucinated"] is True


def test_run_rollout_without_judge_leaves_label_none():
    out = "/tmp/aogc_rollout_nojudge.jsonl"
    if os.path.exists(out):
        os.remove(out)
    run_rollout(_trajs()[:1], EchoGenerator(), judge=None, out_path=out)
    s = load_scored(out)[0].steps[0]
    assert s.is_error is None and s.reasoning != ""  # generated but unlabeled
    os.remove(out)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} rollout tests passed.")


if __name__ == "__main__":
    _run_all()
