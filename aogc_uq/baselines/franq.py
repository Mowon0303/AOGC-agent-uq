"""FRANQ-as-agent: the strongest, most adjacent baseline (defends arXiv 2505.21072).

FRANQ (Faithfulness-Aware UQ for RAG) decomposes an answer into atomic claims and
NLI-checks each claim against the *retrieved evidence*; faithfulness drives the UQ.
A reviewer will say "AOGC is just FRANQ on agents." So we implement FRANQ's
mechanism *faithfully* and make it a first-class baseline — then show (in
experiments/franq_vs_aogc.py) where the naive port is structurally blind.

Faithful reduction of FRANQ to an agent step:
  - the "answer"   = the agent's REASONING text (declarative claims; FRANQ operates
    on natural-language statements, not on function-call actions).
  - the "evidence" = the CURRENT observation o_t only (RAG has one retrieval per
    query; the naive port has no notion of prior-step evidence).
  - score          = fraction of reasoning claims (entities + assertions) NOT
    supported by o_t.

What this port CANNOT see (and AOGC adds): cross-step observation attribution,
agent-misread vs tool-error split, and action-observation legality.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..aogc.claims import extract_entities, split_assertions
from ..aogc.nli import NLIScorer, get_nli
from ..aogc.traceability import assertion_supported, entity_traceable
from ..data.schema import Step


@dataclass
class FranqConfig:
    use_entities: bool = True
    use_nli: bool = True
    nli_threshold: float = 0.5


def franq_as_agent_score(step: Step, nli: NLIScorer | None = None,
                         cfg: FranqConfig | None = None) -> dict:
    """FRANQ-style unsupported-claim fraction for one agent step. Returns dict."""
    cfg = cfg or FranqConfig()
    if nli is None and cfg.use_nli:
        nli = get_nli(None)

    # the naive port: evidence is the CURRENT observation only (no prior_observations)
    evidence = [step.observation] if step.observation else []

    unsupported = 0
    total = 0
    bad_entities, bad_assertions = [], []

    if cfg.use_entities:
        for e in extract_entities(step.reasoning):   # reasoning only, not action
            total += 1
            if not entity_traceable(e, evidence):
                unsupported += 1
                bad_entities.append(e)

    if cfg.use_nli:
        for a in split_assertions(step.reasoning):
            total += 1
            ok, _ = assertion_supported(a, evidence, nli, cfg.nli_threshold)
            if not ok:
                unsupported += 1
                bad_assertions.append(a)

    score = unsupported / total if total > 0 else 0.0
    return {
        "score": float(score),
        "counts": {"claims": total, "unsupported": unsupported},
        "offending": {"entities": bad_entities, "assertions": bad_assertions},
    }


def franq_as_agent_uncertainty(step: Step, nli: NLIScorer | None = None,
                               cfg: FranqConfig | None = None) -> float:
    return franq_as_agent_score(step, nli=nli, cfg=cfg)["score"]
