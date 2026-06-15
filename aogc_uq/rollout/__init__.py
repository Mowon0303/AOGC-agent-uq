"""Rollout + judge harness (GPU side: Colab T4 / RTX 3060).

Pipeline, all resumable:  MIRAGE Trajectory (loader, no labels)
    -> generator.generate(step.meta['messages'])   # local 4-bit model OR API
    -> attach_response(step, text)                   # parse <think>/<action>
    -> judge.judge(step) -> is_error + error_type     # LLM-judge
    -> append scored Trajectory to .jsonl (flush per line; disconnect-safe)

Then on the Mac (no GPU): ``load_scored(path)`` -> run AOGC / FRANQ / metrics.

Backends are pluggable so the orchestration is unit-tested on CPU with stubs
(EchoGenerator, StubJudge); HFGenerator/LLMJudge are the real Colab backends and
import torch/SDKs lazily.
"""

from .generate import EchoGenerator, HFGenerator, ResponseGenerator, flatten_messages
from .judge import Judge, LLMJudge, StubJudge, build_judge_messages
from .run import load_scored, run_rollout

__all__ = [
    "ResponseGenerator",
    "EchoGenerator",
    "HFGenerator",
    "flatten_messages",
    "Judge",
    "StubJudge",
    "LLMJudge",
    "build_judge_messages",
    "run_rollout",
    "load_scored",
]
