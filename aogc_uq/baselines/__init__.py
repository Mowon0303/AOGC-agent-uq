"""Existing-UQ baselines that AOGC is fused on top of (not replaced).

Implemented now (black-box, work on any rollout):
  - verbalized: self-reported confidence -> uncertainty.

Need model sampling / logits (added in the rollout phase, run on the 3060 box):
  - semantic_entropy : cluster k sampled actions, cluster entropy.
  - p_true           : model's P(True) self-eval.
  - self_consistency : agreement across k samples.
  - token_entropy    : mean/min token logprob (open models only).
"""

from .franq import FranqConfig, franq_as_agent_score, franq_as_agent_uncertainty
from .verbalized import parse_confidence, verbalized_uncertainty

__all__ = [
    "parse_confidence",
    "verbalized_uncertainty",
    "FranqConfig",
    "franq_as_agent_score",
    "franq_as_agent_uncertainty",
]
