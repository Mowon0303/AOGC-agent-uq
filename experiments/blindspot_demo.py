"""W2-3 ★ blind-spot demo harness (H1).

Runs the full pipeline — signals -> metrics -> blind-spot slice — end to end.
Here it runs on SYNTHETIC trajectories so it executes anywhere (this Mac, no GPU,
no benchmark download). The SAME harness consumes real MIRAGE-Bench rollouts once
the loader + 3060 rollouts land: swap ``make_synthetic_dataset()`` for the loader.

The synthetic generator bakes in the thesis to verify the harness can SEE it:
  * grounding-failure steps cite fabricated entities/numbers (AOGC should fire)
    but report the SAME high verbalized confidence as correct steps (verbalized
    is blind) -> on the grounding-vs-correct slice, verbalized AUROC ~ 0.5 while
    AOGC AUROC is high.
  * planning-failure steps are correctly grounded -> AOGC should NOT fire on them
    (specificity check).

This is a SANITY/PLUMBING demo, NOT a research result. Real H1 evidence requires
real rollouts where we do NOT control the confidence distribution.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, score_trajectory
from aogc_uq.baselines import verbalized_uncertainty
from aogc_uq.data.schema import ErrorType, Step, Trajectory
from aogc_uq.metrics import auroc, blindspot_auroc, overconfidence

_FILES = ["report.pdf", "data.csv", "notes.txt", "config.yaml", "summary.docx",
          "index.html", "model.ckpt", "log.txt", "image.png", "schema.sql"]
_GHOSTS = ["budget.xlsx", "invoice.zip", "secret.key", "backup.tar", "ghost.bin",
           "phantom.dat", "missing.json", "fake.db"]


def _obs(rng):
    files = rng.choice(_FILES, size=3, replace=False).tolist()
    code = int(rng.choice([200, 201, 204]))
    n = int(rng.integers(2, 9))
    return (f"The tool returned {len(files)} results: {', '.join(files)}. "
            f"Status code {code}. {n} records matched."), files, code, n


def _conf(rng, mu, sd):
    return float(np.clip(rng.normal(mu, sd), 0.0, 1.0))


def make_synthetic_dataset(n_tasks=120, seed=0):
    rng = np.random.default_rng(seed)
    trajs = []
    for t in range(n_tasks):
        steps = []
        for i in range(rng.integers(3, 6)):
            obs, files, code, n = _obs(rng)
            kind = rng.choice(["correct", "grounding", "planning"], p=[0.6, 0.25, 0.15])
            if kind == "correct":
                cited = ", ".join(rng.choice(files, size=2, replace=False))
                step = Step(index=i, observation=obs,
                            reasoning=f"The tool returned {cited}. {n} records matched.",
                            action=f'read("{files[0]}")',
                            is_error=False, error_type=ErrorType.NONE,
                            meta={"verbalized_confidence": _conf(rng, 0.88, 0.05)})
            elif kind == "grounding":
                ghost = str(rng.choice(_GHOSTS))
                fake_n = n + int(rng.integers(50, 500))
                step = Step(index=i, observation=obs,
                            reasoning=(f"The results include {ghost} with {fake_n} "
                                       f"records. I will open {ghost}."),
                            action=f'open("{ghost}")',
                            is_error=True, error_type=ErrorType.GROUNDING,
                            # blind spot: SAME high confidence as correct steps
                            meta={"verbalized_confidence": _conf(rng, 0.86, 0.06)})
            else:  # planning: grounded but wrong choice
                cited = ", ".join(rng.choice(files, size=2, replace=False))
                step = Step(index=i, observation=obs,
                            reasoning=(f"The tool returned {cited}. I will stop here "
                                       f"instead of continuing the task."),
                            action="finish()",
                            is_error=True, error_type=ErrorType.PLANNING,
                            meta={"verbalized_confidence": _conf(rng, 0.80, 0.08)})
            steps.append(step)
        trajs.append(Trajectory(task_id=f"syn-{t}", steps=steps, benchmark="synthetic"))
    return trajs


def collect(trajs):
    nli = LexicalNLI()
    aogc, verb, is_err, etypes = [], [], [], []
    for tr in trajs:
        score_trajectory(tr, nli=nli)
        for s in tr.steps:
            aogc.append(s.signals["aogc"])
            verb.append(verbalized_uncertainty(s))
            is_err.append(bool(s.is_error))
            etypes.append(s.error_type)
    return (np.array(aogc), np.array(verb), np.array(is_err), etypes)


def main():
    trajs = make_synthetic_dataset()
    aogc, verb, is_err, etypes = collect(trajs)
    n = len(is_err)
    n_ground = sum(1 for e in etypes if e == ErrorType.GROUNDING)
    conf = np.array([1 - v for v in verb])  # for overconfidence on the slice
    correct = (~is_err).astype(int)
    g_mask = np.array([e == ErrorType.GROUNDING for e in etypes]) | (~is_err)

    rows = [
        ("verbalized confidence", verb),
        ("AOGC (ours, lexical NLI)", aogc),
    ]
    print("=" * 74)
    print("  BLIND-SPOT DEMO  (SYNTHETIC sanity check — NOT paper results)")
    print(f"  steps={n}  errors={int(is_err.sum())}  grounding-failures={n_ground}")
    print("=" * 74)
    print(f"  {'signal':<26}{'AUROC(all err)':>15}{'AUROC(grounding)':>18}")
    print("  " + "-" * 59)
    for name, score in rows:
        a_all = auroc(score, is_err.astype(int))
        a_blind = blindspot_auroc(score, is_err, etypes, ErrorType.GROUNDING)
        print(f"  {name:<26}{a_all:>15.3f}{a_blind:>18.3f}")
    print("  " + "-" * 59)
    oc = overconfidence(conf[g_mask], correct[g_mask])
    print(f"  verbalized overconfidence on grounding-vs-correct slice: {oc:+.3f}")
    print("=" * 74)

    a_blind_verb = blindspot_auroc(verb, is_err, etypes, ErrorType.GROUNDING)
    a_blind_aogc = blindspot_auroc(aogc, is_err, etypes, ErrorType.GROUNDING)
    print("\nInterpretation (mechanical, on synthetic data):")
    print(f"  - verbalized is near-random on grounding failures (AUROC={a_blind_verb:.2f})")
    print(f"  - AOGC separates them                              (AUROC={a_blind_aogc:.2f})")
    print("  => the harness can MEASURE the blind spot. Real H1 needs real rollouts.")
    assert a_blind_aogc > a_blind_verb + 0.15, "demo failed to show the contrast"
    print("\nOK.")


if __name__ == "__main__":
    main()
