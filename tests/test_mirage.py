"""MIRAGE-Bench loader tests, run against REAL snapshot fixtures (Apache-2.0,
see tests/fixtures/mirage/SOURCE.md). Offline.
Run: python3 tests/test_mirage.py  or  python3 -m pytest tests/test_mirage.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.data import mirage
from aogc_uq.data.schema import ErrorType

FX = os.path.join(os.path.dirname(__file__), "fixtures", "mirage")


def _load(fn, setting, env):
    obj = json.load(open(os.path.join(FX, fn)))
    return mirage.parse_snapshot(obj, setting=setting, environment=env)


def test_setting_taxonomy_mapping():
    assert mirage.setting_error_type("popup") == ErrorType.GROUNDING
    assert mirage.setting_error_type("unexpected_transition") == ErrorType.GROUNDING
    assert mirage.setting_error_type("misleading") == ErrorType.GROUNDING
    assert mirage.setting_error_type("error_feedback") == ErrorType.GROUNDING
    assert mirage.setting_error_type("underspecified") == ErrorType.PLANNING
    assert mirage.setting_error_type("users_questions") == ErrorType.OTHER
    assert mirage.is_grounding_setting("popup")
    assert not mirage.is_grounding_setting("users_questions")


def test_webarena_axtree_snapshot():
    tr = _load("webarena_unexpected_transition.json", "unexpected_transition", "webarena")
    assert len(tr) == 1
    s = tr.steps[0]
    assert s.error_type == ErrorType.GROUNDING
    assert s.meta["is_grounding_setting"] is True
    assert "Observation of current step" in s.observation
    assert s.meta["instruction"] and "clone" in s.meta["instruction"]
    # AXTree bids extracted, and the explanatory note '[bid]' is filtered out
    assert len(s.available_objects) > 5
    assert "bid" not in [o.lower() for o in s.available_objects]
    # reasoning/action are empty until a rollout fills them
    assert s.reasoning == "" and s.action == ""
    assert s.is_error is None


def test_osworld_a11y_snapshot():
    tr = _load("osworld_popup.json", "popup", "osworld")
    s = tr.steps[0]
    assert s.error_type == ErrorType.GROUNDING
    assert "accessibility tree" in s.observation.lower()
    assert s.meta["instruction"]


def test_taubench_toolcall_conversation():
    tr = _load("taubench_users_questions.json", "users_questions", "taubench")
    s = tr.steps[0]
    assert s.error_type == ErrorType.OTHER          # not a grounding setting
    # tool outputs become observations; at least one API output captured
    assert "API output" not in s.observation        # prefix stripped
    assert len(s.prior_observations) >= 1 or len(s.observation) > 0
    assert s.meta["n_history_assistant_turns"] >= 1
    assert s.meta["instruction"]


def test_attach_response_gui_and_toolcall():
    s = _load("webarena_unexpected_transition.json", "unexpected_transition", "webarena").steps[0]
    mirage.attach_response(s, "<think>404 means the repo page failed to load</think>\n<action>click('7')</action>")
    assert "404" in s.reasoning and s.action == "click('7')"

    s2 = _load("taubench_users_questions.json", "users_questions", "taubench").steps[0]
    mirage.attach_response(s2, "Thought:\nAnswer the question.\nAction:\n{\"name\": \"respond\"}")
    assert "Answer" in s2.reasoning and "respond" in s2.action


def test_parse_freeform_response_populates_reasoning():
    # SWE-bench-style free-form output has no <think>/<action> tags; the whole
    # text must become reasoning (else AOGC/FRANQ read nothing -> the AOGC==FRANQ bug)
    r, a = mirage.parse_agent_response(
        "The relevant code is on line 1474 in marshmallow/fields.py.\nbash-$ ls")
    assert "line 1474" in r and r != ""        # reasoning is NOT empty
    assert a == "bash-$ ls"                      # best-effort last line as action

    # <action> without <think>: pre-action text becomes reasoning
    r2, a2 = mirage.parse_agent_response("The page shows a 404.\n<action>click('7')</action>")
    assert "404" in r2 and a2 == "click('7')"


def test_load_file_infers_setting_from_path():
    # load_file uses .../dataset_all/<setting>/<env>/file.json; our fixture path
    # doesn't follow that, so setting falls back — just assert it parses to 1 step.
    tr = mirage.load_file(os.path.join(FX, "webarena_unexpected_transition.json"))
    assert len(tr) == 1 and tr.steps[0].observation


def test_missing_root_raises_with_hint():
    try:
        list(mirage.iter_dataset("/nonexistent/dataset_all"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "git clone" in str(e)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} mirage tests passed.")


if __name__ == "__main__":
    _run_all()
