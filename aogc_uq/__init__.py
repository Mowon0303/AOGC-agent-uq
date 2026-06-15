"""AOGC: Action-Observation Grounding Consistency for LLM-Agent UQ.

A training-free, black-box-usable uncertainty signal that checks whether an
agent's reasoning/action is faithful to what the tool/environment actually
returned (the observation). Orthogonal to verbalized / entropy signals; fuses
on top of existing agent-UQ SOTA.

Layout
------
- ``aogc_uq.data``    : unified Step/Trajectory schema every signal consumes.
- ``aogc_uq.metrics`` : detection (AUROC/AUPRC), calibration (ECE/Brier),
                        selective (AURC / risk-coverage), blind-spot slicing.
- ``aogc_uq.aogc``    : the AOGC signal (claim extraction + traceability + NLI).
- ``aogc_uq.fusion``  : late-fusion of step-level signals.
"""

__version__ = "0.0.1"
