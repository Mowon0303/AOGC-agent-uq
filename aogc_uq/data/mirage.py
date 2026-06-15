"""MIRAGE-Bench loader (arXiv 2507.21017, sunblaze-ucb/mirage-bench).

What MIRAGE actually is (verified against the real repo, not the README):
each file in ``dataset_all/<setting>/<env>/*.json`` is a **decision-point
snapshot** — a chat prompt engineered to elicit a specific failure. You run a
model on it and an LLM-judge decides whether the model hallucinated. So:

  * the **directory** is the elicited failure CATEGORY (ground-truth risk type),
    NOT a confirmed error of this rollout;
  * ``is_error`` therefore requires a rollout + judge (the 3060/API phase); the
    loader leaves it None and fills ``error_type`` with the *elicited* type.

Two on-disk shapes, both handled:
  A. GUI agents (webarena / workarena / osworld): input = [system, user]; a user
     text block holds "# Observation of current step:" (AXTree) or an
     "accessibility tree" dump. One decision Step.
  B. tool-call agents (taubench / theagentcompany): input = full conversation
     [system(policy), user(task), assistant("Thought/Action"), user("API output: …"),
     …]. Tool outputs are observations; the last user turn is the current one.

Envelope keys (all files): ``task_name``, ``goal``, ``input_step``, ``input``.

Usage
-----
    from aogc_uq.data.mirage import load_dataset, find_dataset_root
    root = find_dataset_root()                 # ./data/raw/.../dataset_all or $MIRAGE_ROOT
    trajs = load_dataset(root, settings=["popup", "unexpected_transition"])
    # each traj has ONE decision Step (reasoning/action empty until a rollout fills it)
    attach_response(traj.steps[0], model_output)   # parses <think>/<action> or Thought/Action
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

from .schema import ErrorType, Step, Trajectory

# ---- setting -> our taxonomy ----------------------------------------------
# Grounding settings = the agent must read the *observation* correctly. These are
# our blind-spot (GROUNDING) slice. Others are kept as non-grounding error types
# so AOGC's specificity (NOT firing on them) can be measured. Mapping is by
# setting semantics; verify against the paper's taxonomy before final tables.
SETTING_ERRORTYPE: dict[str, ErrorType] = {
    "popup": ErrorType.GROUNDING,                 # unexpected popups in the observation
    "unexpected_transition": ErrorType.GROUNDING, # env state changed; must re-read obs
    "misleading": ErrorType.GROUNDING,            # misleading content in the observation
    "error_feedback": ErrorType.GROUNDING,        # tool/action error returned in the obs
    "repetitive_4": ErrorType.OTHER,              # looping (history faithfulness)
    "repetitive_7": ErrorType.OTHER,
    "unachievable": ErrorType.PLANNING,           # goal infeasible; should abstain
    "unachievable_easier": ErrorType.PLANNING,
    "underspecified": ErrorType.PLANNING,         # instruction faithfulness
    "users_questions": ErrorType.OTHER,           # interaction / clarification
}
GROUNDING_SETTINGS = frozenset(
    s for s, e in SETTING_ERRORTYPE.items() if e == ErrorType.GROUNDING
)

_OBS_MARKERS = ("# Observation of current step", "accessibility tree", "AXTree")
_EXAMPLE_MARKERS = ("# Concrete Example", "<think>", "concrete example of how to format")
_HEADER_MARKERS = ("# Instructions", "Review the current state of the page")
_TOOL_OUT_PREFIXES = ("API output:", "Observation:", "Tool output:")


def setting_error_type(setting: str) -> ErrorType:
    return SETTING_ERRORTYPE.get(setting, ErrorType.OTHER)


def is_grounding_setting(setting: str) -> bool:
    return setting in GROUNDING_SETTINGS


# ---- content flattening ----------------------------------------------------

def _content_blocks(content) -> list[str]:
    """Flatten a message ``content`` (str | list[part]) to a list of text blocks."""
    if isinstance(content, str):
        return [content]
    blocks = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    blocks.append(p["text"])
                elif p.get("type") in ("image_url", "image"):
                    blocks.append("[image]")
            elif isinstance(p, str):
                blocks.append(p)
    return blocks


def _has(text: str, markers) -> bool:
    t = text.lower()
    return any(m.lower() in t for m in markers)


# ---- AXTree element extraction (for the action-legality term) ---------------

_BID = re.compile(r"\[([a-zA-Z0-9_]+)\]")
_NAMED = re.compile(r"\[[a-zA-Z0-9_]+\]\s+\w+\s+'([^']{1,80})'")
# tokens that match the bid pattern but are AXTree explanatory notes, not elements
_NON_ELEMENT = {"bid"}


def extract_available_objects(observation: str, cap: int = 600) -> list[str]:
    """Best-effort set of legal action targets from an AXTree-style observation.

    Returns element bids + their quoted names. Works well for webarena/workarena
    AXTrees; degrades to whatever quoted tokens exist elsewhere.
    """
    if not observation:
        return []
    objs: list[str] = []
    seen = set()
    for rx in (_BID, _NAMED):
        for m in rx.findall(observation):
            k = m.lower()
            if m and k not in seen and k not in _NON_ELEMENT:
                seen.add(k)
                objs.append(m)
                if len(objs) >= cap:
                    return objs
    return objs


# ---- snapshot parsing ------------------------------------------------------

def _classify_user_block(text: str) -> str:
    if _has(text, _OBS_MARKERS):
        return "observation"
    if _has(text, _EXAMPLE_MARKERS):
        return "example"
    if _has(text, _HEADER_MARKERS):
        return "header"
    for pre in _TOOL_OUT_PREFIXES:
        if text.lstrip().startswith(pre):
            return "tool_output"
    return "instruction"


def parse_snapshot(obj: dict, setting: str = "", environment: str = "") -> Trajectory:
    """Turn one MIRAGE snapshot dict into a single-decision-step Trajectory."""
    messages = obj.get("input", []) or []
    task_name = obj.get("task_name", "")
    goal = (obj.get("goal") or "").strip()
    input_step = obj.get("input_step")

    instruction = goal
    observations: list[str] = []
    history_turns: list[dict] = []  # reconstructed prior reasoning/actions (tool-call)

    for msg in messages:
        role = msg.get("role")
        blocks = _content_blocks(msg.get("content"))
        if role == "system":
            continue
        if role == "assistant":
            history_turns.append({"text": "\n".join(blocks)})
            continue
        # user / tool
        for b in blocks:
            kind = _classify_user_block(b)
            if kind == "observation":
                observations.append(b)
            elif kind == "tool_output":
                for pre in _TOOL_OUT_PREFIXES:
                    if b.lstrip().startswith(pre):
                        observations.append(b.lstrip()[len(pre):].strip())
                        break
            elif kind == "instruction" and not instruction:
                instruction = b.strip()

    # Fallbacks: if nothing matched an obs marker, use the last/longest user block.
    if not observations:
        user_blocks = [
            b for m in messages if m.get("role") in ("user", "tool")
            for b in _content_blocks(m.get("content"))
            if _classify_user_block(b) not in ("example", "header")
        ]
        if user_blocks:
            observations = [max(user_blocks, key=len)]

    current_obs = observations[-1] if observations else ""
    prior_obs = observations[:-1]

    step = Step(
        index=input_step if isinstance(input_step, int) else 0,
        observation=current_obs,
        prior_observations=prior_obs,
        reasoning="",   # filled by a rollout via attach_response()
        action="",
        available_objects=extract_available_objects(current_obs),
        is_error=None,                          # needs rollout + judge
        error_type=setting_error_type(setting), # ELICITED type, not confirmed
        meta={
            "task_name": task_name,
            "setting": setting,
            "environment": environment,
            "is_grounding_setting": is_grounding_setting(setting),
            "input_step": input_step,
            "instruction": instruction,
            "n_history_assistant_turns": len(history_turns),
            "messages": messages,   # keep raw prompt so a rollout can re-run inference
        },
    )
    return Trajectory(
        task_id=task_name or "mirage",
        steps=[step],
        success=None,
        benchmark="mirage-bench",
        model="",
        meta={"setting": setting, "environment": environment},
    )


# ---- agent-response parsing (close the loop after a rollout) ----------------

_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.S | re.I)
_THOUGHT_KW = re.compile(r"Thought:\s*(.*?)(?:\nAction:|\Z)", re.S | re.I)
_ACTION_KW = re.compile(r"Action:\s*(.*)", re.S | re.I)


def parse_agent_response(text: str) -> tuple[str, str]:
    """Extract (reasoning, action) from a model response, robust to format.

    Handles <think>/<action>, Thought:/Action:, and free-form output. KEY: for
    free-form (no structured tags) the WHOLE text becomes ``reasoning`` — the
    agent's claims about the observation live there, and AOGC/FRANQ read reasoning.
    Returning empty reasoning (the old bug) made both signals blind.
    """
    if not text:
        return "", ""

    for think_rx, action_rx in ((_THINK, _ACTION_TAG), (_THOUGHT_KW, _ACTION_KW)):
        th = think_rx.search(text)
        ac = action_rx.search(text)
        if th or ac:
            action = ac.group(1).strip() if ac else ""
            if th:
                reasoning = th.group(1).strip()
            else:  # action found but no explicit thinking -> text before it is the reasoning
                reasoning = text[: ac.start()].strip()
            return reasoning, action

    # free-form: the entire output carries the agent's reasoning/claims;
    # best-effort action = last non-empty line.
    t = text.strip()
    last = t.splitlines()[-1].strip() if t else ""
    return t, last


def attach_response(step: Step, response_text: str) -> Step:
    """Fill ``step.reasoning``/``step.action`` from a model response. Mutates+returns."""
    step.reasoning, step.action = parse_agent_response(response_text)
    step.meta["raw_response"] = response_text
    return step


# ---- dataset discovery / iteration ----------------------------------------

def find_dataset_root() -> Path | None:
    """Locate ``dataset_all`` from $MIRAGE_ROOT or common local clone paths."""
    candidates = []
    env = os.environ.get("MIRAGE_ROOT")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[2]  # repo root
    candidates += [
        here / "data" / "raw" / "mirage-bench" / "dataset_all",
        here / "data" / "raw" / "dataset_all",
        here / "data" / "mirage-bench" / "dataset_all",
    ]
    for c in candidates:
        if c and c.is_dir():
            return c
    return None


def download_hint(dest: str = "data/raw/mirage-bench") -> str:
    return (f"MIRAGE-Bench data not found. Clone it (Apache-2.0):\n"
            f"  git clone https://github.com/sunblaze-ucb/mirage-bench {dest}\n"
            f"then set MIRAGE_ROOT={dest}/dataset_all or pass root= explicitly.")


def iter_dataset(root, settings=None, environments=None, limit=None) -> Iterator[Trajectory]:
    """Yield Trajectory per snapshot under ``root`` (= a dataset_all dir)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(download_hint())
    settings = set(settings) if settings else None
    environments = set(environments) if environments else None
    n = 0
    for setting_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        setting = setting_dir.name
        if settings and setting not in settings:
            continue
        for env_dir in sorted(p for p in setting_dir.iterdir() if p.is_dir()):
            env = env_dir.name
            if environments and env not in environments:
                continue
            for fp in sorted(env_dir.glob("*.json")):
                try:
                    obj = json.loads(fp.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(obj, dict) or "input" not in obj:
                    continue  # skip helper scripts / non-snapshot json
                yield parse_snapshot(obj, setting=setting, environment=env)
                n += 1
                if limit and n >= limit:
                    return


def load_dataset(root=None, settings=None, environments=None, limit=None) -> list[Trajectory]:
    if root is None:
        root = find_dataset_root()
        if root is None:
            raise FileNotFoundError(download_hint())
    return list(iter_dataset(root, settings, environments, limit))


def load_file(path) -> Trajectory:
    """Parse a single snapshot file; infer setting/env from the path if possible."""
    path = Path(path)
    obj = json.loads(path.read_text())
    setting, environment = "", ""
    parts = path.parts
    # .../dataset_all/<setting>/<env>/file.json
    if len(parts) >= 3:
        setting, environment = parts[-3], parts[-2]
    return parse_snapshot(obj, setting=setting, environment=environment)
