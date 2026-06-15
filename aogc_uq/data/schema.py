"""Unified trajectory schema.

This is the *contract* between three layers that must not know about each other:
  1. rollout producers  (MIRAGE-Bench / ALFWorld / tau-bench / BFCL loaders, or a
     local-model agent loop) -> produce ``Trajectory`` objects.
  2. UQ signals          (AOGC, verbalized, semantic entropy, P(True), ...)
     -> consume ``Trajectory`` / ``Step`` and emit a per-step scalar.
  3. evaluation          (metrics, fusion) -> consume scalars + labels.

Keeping this dataclass deliberately small and JSON-serializable so rollouts can
be dumped to .jsonl on the GPU box and analyzed anywhere (incl. this Mac).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """Failure taxonomy, aligned with AgentErrorTaxonomy / MIRAGE categories.

    ``GROUNDING`` is the blind-spot class this project targets: the action is
    unfaithful to the *observation* (misread / ignored / fabricated tool output).
    """

    NONE = "none"            # step is correct
    GROUNDING = "grounding"  # ★ unfaithful to observation (our target slice)
    PLANNING = "planning"
    REFLECTION = "reflection"
    MEMORY = "memory"
    ACTION = "action"        # malformed / illegal action not due to misreading
    SYSTEM = "system"        # tool error, timeout, env failure (aleatoric)
    OTHER = "other"


@dataclass
class Step:
    """One agent step: it saw ``observation`` and emitted ``reasoning`` + ``action``.

    Fields the AOGC signal reads
    ----------------------------
    observation        : the tool/environment return o_t this step should ground in.
    prior_observations : earlier o_{<t}, for cross-step traceability (an agent may
                         correctly cite something a few steps back).
    available_objects  : optional list of objects truly present in o_t (buttons,
                         fields, file paths) for the action-legality term.

    Label fields (for evaluation only; signals must NOT read these)
    ----------------------------
    is_error   : ground-truth — did this step go wrong?
    error_type : if is_error, which ErrorType (drives the blind-spot slice).
    """

    index: int
    observation: str = ""
    reasoning: str = ""
    action: str = ""
    prior_observations: list[str] = field(default_factory=list)
    available_objects: list[str] = field(default_factory=list)

    # ground-truth labels (eval only)
    is_error: bool | None = None
    error_type: ErrorType = ErrorType.NONE

    # optional signal cache: name -> scalar uncertainty (higher = more uncertain)
    signals: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def grounding_context(self) -> list[str]:
        """All observation text a claim is allowed to be traced back to."""
        ctx = list(self.prior_observations)
        if self.observation:
            ctx.append(self.observation)
        return [c for c in ctx if c]


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    success: bool | None = None          # final task success
    benchmark: str = ""
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    # ---- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for s in d["steps"]:
            s["error_type"] = (
                s["error_type"].value
                if isinstance(s["error_type"], ErrorType)
                else s["error_type"]
            )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        steps = []
        for s in d.get("steps", []):
            s = dict(s)
            et = s.get("error_type", ErrorType.NONE)
            s["error_type"] = ErrorType(et) if not isinstance(et, ErrorType) else et
            steps.append(Step(**s))
        return cls(
            task_id=d["task_id"],
            steps=steps,
            success=d.get("success"),
            benchmark=d.get("benchmark", ""),
            model=d.get("model", ""),
            meta=d.get("meta", {}),
        )
