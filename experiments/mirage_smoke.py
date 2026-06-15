"""Smoke: real MIRAGE observation -> AOGC signal (loader closes the loop).

Loads a REAL MIRAGE-Bench snapshot, then attaches two hand-written responses
(stand-ins for a rollout we don't have yet) and shows AOGC scores them against
the REAL AXTree observation: a response grounded in actual page elements scores
low; one that fabricates a page state scores high.

This validates data->signal on real observation text. It is NOT an H1 result —
that needs real model rollouts + the LLM-judge label, scored over the full set.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aogc_uq.aogc import LexicalNLI, aogc_step_score
from aogc_uq.baselines import franq_as_agent_uncertainty
from aogc_uq.data import mirage

FX = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "mirage")


def main():
    tr = mirage.load_file(os.path.join(FX, "webarena_unexpected_transition.json"))
    step = tr.steps[0]
    print("=" * 74)
    print("  REAL MIRAGE snapshot -> AOGC   (loader smoke, not an H1 result)")
    print(f"  setting={step.error_type.value}  instruction={step.meta['instruction']!r}")
    print(f"  observation: {len(step.observation)} chars, "
          f"{len(step.available_objects)} AXTree elements")
    print("  observation head:", repr(step.observation[:120]))
    print("=" * 74)

    nli = LexicalNLI()

    # GROUNDED: the page is a 404 'Not Found'; this response reads it correctly.
    grounded = ("<think>The page title is 'Not Found' and shows a 404 image, so the "
                "repository page failed to load.</think>\n"
                "<action>send_msg_to_user('The repo page returned 404 Not Found.')</action>")
    # HALLUCINATED: invents page content that is NOT in the 404 observation.
    halluc = ("<think>The repository page shows the SSH clone box with url "
              "git@gitlab.com:byteblaze/super.git and a Clone button bid a999.</think>\n"
              "<action>click('a999')</action>")

    for label, resp in [("grounded response", grounded), ("hallucinated response", halluc)]:
        mirage.attach_response(step, resp)
        a = aogc_step_score(step, nli=nli)
        f = franq_as_agent_uncertainty(step, nli=nli)
        print(f"\n  [{label}]")
        print(f"    AOGC  = {a['score']:.3f}   FRANQ-as-agent = {f:.3f}")
        print(f"    breakdown: {a['breakdown']}")
        off = a["offending"]
        if off["entities"] or off["missing_objects"]:
            print(f"    flagged entities={off['entities'][:5]} "
                  f"missing_objects={off['missing_objects'][:5]}")

    # re-score cleanly for the assertion (attach_response mutates step in place)
    mirage.attach_response(step, grounded); a_g = aogc_step_score(step, nli=nli)["score"]
    mirage.attach_response(step, halluc);   a_h = aogc_step_score(step, nli=nli)["score"]
    print("\n  => on a real observation, AOGC(hallucinated) > AOGC(grounded):",
          f"{a_h:.3f} > {a_g:.3f}")
    assert a_h > a_g, "AOGC should score the fabricated response higher on real data"
    print("\nOK.")


if __name__ == "__main__":
    main()
