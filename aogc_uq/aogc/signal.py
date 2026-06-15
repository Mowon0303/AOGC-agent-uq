"""Step 3 of AOGC: the action-observation grounding-consistency score u_t^g.

    u_t^g = w_e * untraceable_entity_frac
          + w_n * untraceable_number_frac
          + w_a * unentailed_assertion_frac
          + lam * illegal_action_indicator        (only if available_objects given)

normalized by the active weights so u_t^g in [0, 1]; higher = the agent more
likely misread / fabricated the observation. Returns a breakdown dict so the
paper's ablation (entity-only vs +NLI vs +action-legality) is just a re-weighting,
and so failures are inspectable per claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.schema import Step
from .claims import (
    extract_action_target,
    extract_entities,
    extract_numbers,
    split_assertions,
)
from .nli import NLIScorer, get_nli
from .traceability import assertion_supported, entity_traceable, number_traceable


@dataclass
class AOGCConfig:
    w_entity: float = 1.0
    w_number: float = 1.0
    w_assertion: float = 1.0
    lam_illegal: float = 0.5
    nli_threshold: float = 0.5
    use_nli: bool = True            # set False for the entity-only ablation
    use_action_legality: bool = True
    cue_filter: bool = False        # restrict assertions to observation-cue sentences
    # if no claims of a kind are found, contribute this uncertainty for that term
    empty_claim_uncertainty: float = 0.0
    component_weights: dict = field(default_factory=dict)  # reserved


def _frac_untraceable(items, predicate) -> tuple[float, int, list]:
    if not items:
        return None, 0, []
    bad = [it for it in items if not predicate(it)]
    return len(bad) / len(items), len(items), bad


def aogc_step_score(step: Step, nli: NLIScorer | None = None,
                    cfg: AOGCConfig | None = None) -> dict:
    """Compute u_t^g for one step. Returns {'score', 'breakdown', ...}."""
    cfg = cfg or AOGCConfig()
    if nli is None and cfg.use_nli:
        nli = get_nli(None)  # offline lexical default; pass TransformerNLI for paper runs

    ctx = step.grounding_context
    # Referents/assertions are the agent's CLAIMS about the observation -> read
    # from reasoning. The action is handled separately by the legality term, so a
    # free-text action (send_msg_to_user('...')) is not mistaken for a claim.
    claim_text = step.reasoning

    entities = extract_entities(claim_text)
    numbers = extract_numbers(claim_text)

    ent_frac, n_ent, bad_ent = _frac_untraceable(
        entities, lambda e: entity_traceable(e, ctx)
    )
    num_frac, n_num, bad_num = _frac_untraceable(
        numbers, lambda v: number_traceable(v, ctx)
    )

    assert_frac, n_assert, bad_assert = None, 0, []
    if cfg.use_nli:
        assertions = split_assertions(step.reasoning, cue_filter=cfg.cue_filter)
        if assertions:
            results = [
                (a, *assertion_supported(a, ctx, nli, cfg.nli_threshold))
                for a in assertions
            ]
            bad_assert = [a for a, ok, _ in results if not ok]
            assert_frac = len(bad_assert) / len(assertions)
            n_assert = len(assertions)

    # action legality: does an element-targeting action reference an object that
    # is NOT present in the observation? Free-text actions target no element and
    # get no penalty (illegal stays None).
    illegal = None
    missing: list[str] = []
    if cfg.use_action_legality and step.available_objects:
        targets = extract_action_target(step.action)
        if targets:
            avail = {o.lower() for o in step.available_objects}
            missing = [t for t in targets if t.lower() not in avail]
            illegal = 1.0 if missing else 0.0

    # weighted combine over *active* terms only
    terms = []  # (weight, value)
    def add(weight, frac):
        if frac is None:
            if cfg.empty_claim_uncertainty:
                terms.append((weight, cfg.empty_claim_uncertainty))
            return
        terms.append((weight, frac))

    add(cfg.w_entity, ent_frac)
    add(cfg.w_number, num_frac)
    if cfg.use_nli:
        add(cfg.w_assertion, assert_frac)
    if illegal is not None:
        terms.append((cfg.lam_illegal, illegal))

    wsum = sum(w for w, _ in terms)
    score = sum(w * v for w, v in terms) / wsum if wsum > 0 else 0.0

    return {
        "score": float(score),
        "breakdown": {
            "untraceable_entity_frac": ent_frac,
            "untraceable_number_frac": num_frac,
            "unentailed_assertion_frac": assert_frac,
            "illegal_action": illegal,
        },
        "counts": {"entities": n_ent, "numbers": n_num, "assertions": n_assert},
        "offending": {
            "entities": bad_ent,
            "numbers": bad_num,
            "assertions": bad_assert,
            "missing_objects": missing,
        },
    }


def score_trajectory(traj, nli: NLIScorer | None = None,
                     cfg: AOGCConfig | None = None,
                     signal_name: str = "aogc") -> list[float]:
    """Score every step; cache into ``step.signals[signal_name]``; return scores."""
    cfg = cfg or AOGCConfig()
    if nli is None and cfg.use_nli:
        nli = get_nli(None)
    scores = []
    for step in traj.steps:
        r = aogc_step_score(step, nli=nli, cfg=cfg)
        step.signals[signal_name] = r["score"]
        scores.append(r["score"])
    return scores
