"""LLM-judge: did the agent's step hallucinate, given the observation?

This produces the ``is_error`` label MIRAGE snapshots don't ship with. Backends:
- ``StubJudge``  : rule/fn-based, for CPU tests.
- ``LLMJudge``   : calls a chat model via a ``complete(messages)->str`` callable
  you wire to OpenAI / Anthropic / a local model. SDK-agnostic on purpose.

The judge is setting-aware: for grounding settings it asks specifically whether
the action/reasoning is unfaithful to the observation.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from ..data.schema import ErrorType, Step
from ..data.mirage import setting_error_type

_SETTING_QUESTION = {
    "popup": "Did the agent ignore or misread a popup/dialog actually present in the observation, or act as if the page state were different?",
    "unexpected_transition": "Did the agent assume a state/transition that the observation does NOT support (e.g. acting as if a prior page were still shown)?",
    "misleading": "Did the agent take misleading observation content at face value and make a claim/action unsupported by the true page state?",
    "error_feedback": "Did the agent ignore an error/failure reported in the observation and proceed as if it had succeeded?",
}
_DEFAULT_Q = ("Is the agent's reasoning or action UNFAITHFUL to the observation "
              "— i.e. does it reference, assume, or claim something not actually "
              "present in the observation (a grounding hallucination)?")

JUDGE_SYSTEM = (
    "You are a strict evaluator of LLM-agent steps. You are given the goal, the "
    "exact observation the agent saw, and the agent's reasoning and action. Decide "
    "whether the step hallucinated with respect to the observation. Be conservative: "
    "only say YES if the reasoning/action clearly references or assumes something the "
    "observation does not support."
)

_JSON = re.compile(r"\{.*\}", re.S)
_YESNO = re.compile(r"\b(YES|NO)\b")


def build_judge_messages(step: Step) -> list[dict]:
    setting = step.meta.get("setting", "")
    q = _SETTING_QUESTION.get(setting, _DEFAULT_Q)
    instruction = step.meta.get("instruction", "")
    user = (
        f"# Goal\n{instruction}\n\n"
        f"# Observation the agent saw\n{step.observation}\n\n"
        f"# Agent reasoning\n{step.reasoning}\n\n"
        f"# Agent action\n{step.action}\n\n"
        f"# Question\n{q}\n\n"
        "Answer on the FIRST line with exactly one word: YES (it hallucinated) or "
        "NO (faithful). Then add one short sentence of justification."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def _parse_verdict(text: str) -> dict:
    t = (text or "").strip()
    # 1) leading YES/NO (most reliable for small local judges)
    m = _YESNO.search(t[:60].upper())
    if m:
        return {"hallucinated": m.group(1) == "YES", "rationale": t[:300]}
    # 2) JSON {"hallucinated": ...} (strong API judges may still emit this)
    mj = _JSON.search(t)
    if mj:
        try:
            obj = json.loads(mj.group(0))
            return {"hallucinated": bool(obj.get("hallucinated")),
                    "rationale": str(obj.get("rationale", ""))[:300]}
        except json.JSONDecodeError:
            pass
    # 3) last-resort keyword heuristic
    low = t.lower()
    return {"hallucinated": ("hallucinat" in low or "unfaithful" in low),
            "rationale": "unparsed; heuristic fallback"}


class Judge:
    def judge(self, step: Step) -> dict:  # pragma: no cover
        raise NotImplementedError


class StubJudge(Judge):
    """Rule-based judge for tests. Default: hallucinated if action targets a
    ghost object or reasoning is empty. Or pass a custom fn(step)->bool."""

    def __init__(self, fn: Callable[[Step], bool] | None = None):
        self.fn = fn

    def judge(self, step: Step) -> dict:
        if self.fn is not None:
            h = bool(self.fn(step))
        else:
            from ..aogc.claims import extract_action_target
            avail = {o.lower() for o in step.available_objects}
            tgt = extract_action_target(step.action)
            h = bool(tgt) and any(t.lower() not in avail for t in tgt)
        return {"hallucinated": h, "rationale": "stub"}


class LLMJudge(Judge):
    """Calls a chat model via ``complete(messages: list[dict]) -> str``."""

    def __init__(self, complete: Callable[[list[dict]], str]):
        self.complete = complete

    def judge(self, step: Step) -> dict:
        text = self.complete(build_judge_messages(step))
        return _parse_verdict(text)


def apply_verdict(step: Step, verdict: dict) -> Step:
    """Write judge result onto the step: is_error + error_type + meta. Mutates."""
    halluc = bool(verdict.get("hallucinated"))
    step.is_error = halluc
    setting = step.meta.get("setting", "")
    # if hallucinated, the failure is of the elicited type; else it's a correct step
    step.error_type = setting_error_type(setting) if halluc else ErrorType.NONE
    step.meta["judge"] = verdict
    return step
