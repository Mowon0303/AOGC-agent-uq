"""NLI backend for verifying paraphrastic assertions against the observation.

Two interchangeable scorers behind one interface:

* ``LexicalNLI``     — zero-dependency token-containment proxy. Deterministic,
  offline, used in unit tests and as a fallback when no model is downloaded.
* ``TransformerNLI`` — DeBERTa-v3-MNLI (or any HF NLI model). Runs on MPS/CPU;
  this is what the paper's numbers use. Lazy-loads weights on first call so
  importing the package never triggers a download.

Contract: ``entail_prob(premise, hypothesis) -> float in [0,1]`` = P(premise
entails hypothesis). For AOGC the premise is the observation, the hypothesis is
the agent's assertion about it.
"""

from __future__ import annotations

import math
import re

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "and", "or", "that", "this", "these", "those",
    "it", "its", "as", "by", "with", "from", "we", "i", "there", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could", "should",
}
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _content_tokens(text: str) -> list[str]:
    return [w for w in (t.lower() for t in _WORD.findall(text)) if w not in _STOP]


class NLIScorer:
    """Interface."""

    def entail_prob(self, premise: str, hypothesis: str) -> float:  # pragma: no cover
        raise NotImplementedError

    def entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.entail_prob(p, h) for p, h in pairs]


class LexicalNLI(NLIScorer):
    """Containment proxy: fraction of hypothesis content tokens present in premise."""

    name = "lexical"

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        h = _content_tokens(hypothesis)
        if not h:
            return 1.0  # nothing asserted -> trivially supported
        p = set(_content_tokens(premise))
        covered = sum(1 for w in h if w in p)
        return covered / len(h)


class TransformerNLI(NLIScorer):
    """HF sequence-classification NLI model (DeBERTa-v3-MNLI by default)."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._tok = None
        self._model = None
        self._entail_idx = None

    @property
    def name(self) -> str:
        return self.model_name

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device).eval()
        # robustly locate the "entailment" logit index from id2label
        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        self._entail_idx = next(
            (i for i, lab in id2label.items() if "entail" in lab.lower()),
            len(id2label) - 1,  # MNLI convention: entailment is last
        )

    def entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        self._ensure_loaded()
        import torch

        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            premises = [p for p, _ in chunk]
            hyps = [h for _, h in chunk]
            enc = self._tok(
                premises, hyps, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self._entail_idx]
            out.extend(probs.detach().cpu().tolist())
        return out

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        return self.entail_probs([(premise, hypothesis)])[0]


def get_nli(name: str | None = None, **kwargs) -> NLIScorer:
    """Factory. ``None`` / ``"lexical"`` -> offline LexicalNLI; else TransformerNLI."""
    if name is None or name == "lexical":
        return LexicalNLI()
    return TransformerNLI(model_name=name, **kwargs)


def _softmax(xs):  # kept for any future logit post-processing / tests
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]
