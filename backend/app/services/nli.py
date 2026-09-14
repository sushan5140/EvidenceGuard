from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_WORD = re.compile(r"[A-Za-z0-9']+")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_NEGATIONS = {"not", "no", "never", "none", "without", "cannot", "can't", "doesn't", "isn't", "wasn't"}


@dataclass(slots=True)
class Relation:
    label: str
    confidence: float


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in _WORD.findall(text) if len(token) > 2}


class NLIEngine:
    """Pairwise NLI with a deterministic heuristic fallback."""

    def __init__(self, model_name: str, *, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._tokenizer = None
        self._model = None
        self.status = "not-loaded"

    def _load(self) -> bool:
        if not self.enabled:
            self.status = "heuristic-disabled"
            return False
        if self._model is not None and self._tokenizer is not None:
            return True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self.status = f"transformers:{self.model_name}"
            return True
        except Exception as exc:
            self.status = f"heuristic-fallback:{type(exc).__name__}"
            return False

    def _model_relation(self, premise: str, hypothesis: str) -> Relation | None:
        if not self._load():
            return None
        try:
            import torch

            encoded = self._tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=384,
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            id2label = {
                int(key): value.lower()
                for key, value in self._model.config.id2label.items()
            }
            best = int(np.argmax(probs))
            raw = id2label.get(best, "neutral")
            if "entail" in raw:
                label = "supports"
            elif "contrad" in raw:
                label = "contradicts"
            else:
                label = "neutral"
            return Relation(label=label, confidence=float(probs[best]))
        except Exception:
            return None

    def _heuristic(self, left: str, right: str) -> Relation:
        a, b = _token_set(left), _token_set(right)
        if not a or not b:
            return Relation("neutral", 0.50)

        overlap = len(a & b) / max(1, min(len(a), len(b)))
        neg_a = bool(a & _NEGATIONS)
        neg_b = bool(b & _NEGATIONS)

        nums_a = set(_NUMBER.findall(left))
        nums_b = set(_NUMBER.findall(right))
        numeric_conflict = bool(nums_a and nums_b and nums_a != nums_b)

        if overlap >= 0.48 and (neg_a != neg_b or numeric_conflict):
            return Relation("contradicts", min(0.92, 0.60 + overlap * 0.30))
        if overlap >= 0.62:
            return Relation("supports", min(0.94, 0.58 + overlap * 0.32))
        return Relation("neutral", max(0.50, 0.76 - overlap * 0.20))

    def compare(self, left: str, right: str, *, use_model: bool = True) -> Relation:
        if use_model:
            result = self._model_relation(left, right)
            if result is not None:
                return result
        if self.status == "not-loaded":
            self.status = "heuristic"
        return self._heuristic(left, right)
