"""Resumable rollout orchestration -> scored .jsonl.

Each output line is one scored Trajectory (Trajectory.to_dict()). Writes+flushes
per line, so a Colab disconnect loses at most one trajectory; re-running skips
task_ids already present. The big ``meta['messages']`` prompt is dropped from the
output by default to keep files small (the observation is already extracted).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable

from ..data.mirage import attach_response
from ..data.schema import Trajectory
from .generate import ResponseGenerator
from .judge import Judge, apply_verdict


def _done_task_ids(out_path: str) -> set:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("task_id"))
                except json.JSONDecodeError:
                    continue
    return done


def _strip_messages(d: dict) -> dict:
    for s in d.get("steps", []):
        if isinstance(s.get("meta"), dict):
            s["meta"].pop("messages", None)
    return d


def run_rollout(
    trajectories: Iterable[Trajectory],
    generator: ResponseGenerator,
    judge: Judge | None = None,
    out_path: str = "runs/scored.jsonl",
    resume: bool = True,
    keep_messages: bool = False,
    model_name: str = "",
    on_progress: Callable[[int, Trajectory], None] | None = None,
) -> str:
    """Generate + (optionally) judge each trajectory; append scored lines.

    Returns ``out_path``. Safe to call repeatedly (resume skips finished tasks).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done_task_ids(out_path) if resume else set()

    n = 0
    with open(out_path, "a") as fout:
        for traj in trajectories:
            if resume and traj.task_id in done:
                continue
            if model_name:
                traj.model = model_name
            for step in traj.steps:
                messages = step.meta.get("messages", [])
                response = generator.generate(messages) if messages else ""
                attach_response(step, response)
                if judge is not None:
                    apply_verdict(step, judge.judge(step))
            d = traj.to_dict()
            if not keep_messages:
                d = _strip_messages(d)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if on_progress:
                on_progress(n, traj)
    return out_path


def load_scored(path: str) -> list[Trajectory]:
    """Read a scored .jsonl back into Trajectory objects (for Mac-side analysis)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Trajectory.from_dict(json.loads(line)))
    return out
