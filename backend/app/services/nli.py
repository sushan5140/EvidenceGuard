from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

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
    """Pairwise NLI with process-wide model caching and batched inference."""

    _MODEL_CACHE: ClassVar[dict[str, tuple[Any, Any]]] = {}
    _RELATION_CACHE: ClassVar[dict[tuple[str, str, str], Relation]] = {}

    def __init__(self, model_name: str, *, enabled: bool = True, batch_size: int = 32):
        self.model_name = model_name
        self.enabled = enabled
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self.status = "not-loaded"

    def _load(self) -> bool:
        if not self.enabled:
            self.status = "heuristic-disabled"
            return False
        if self._model is not None and self._tokenizer is not None:
            return True

        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._tokenizer, self._model = cached
            self.status = f"transformers:{self.model_name}:cached"
            return True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._MODEL_CACHE[self.model_name] = (tokenizer, model)
            self.status = f"transformers:{self.model_name}"
            return True
        except Exception as exc:
            self.status = f"heuristic-fallback:{type(exc).__name__}"
            return False

    def _decode(self, probs: np.ndarray) -> list[Relation]:
        id2label = {
            int(key): value.lower()
            for key, value in self._model.config.id2label.items()
        }
        results: list[Relation] = []
        for row in probs:
            best = int(np.argmax(row))
            raw = id2label.get(best, "neutral")
            if "entail" in raw:
                label = "supports"
            elif "contrad" in raw:
                label = "contradicts"
            elif raw.startswith("label_") and len(row) == 3:
                # Some NLI checkpoints keep generic LABEL_0/1/2 names.
                # The selected MiniLM NLI checkpoint documents the order as
                # contradiction, entailment, neutral.
                label = {0: "contradicts", 1: "supports", 2: "neutral"}.get(best, "neutral")
            else:
                label = "neutral"
            results.append(Relation(label=label, confidence=float(row[best])))
        return results

    def _model_relations(self, pairs: list[tuple[str, str]]) -> list[Relation] | None:
        if not pairs or not self._load():
            return None
        try:
            import torch

            results: list[Relation] = []
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [left for left, _ in batch],
                    [right for _, right in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=384,
                )
                with torch.inference_mode():
                    logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                results.extend(self._decode(probs))
            return results
        except Exception as exc:
            self.status = f"heuristic-fallback:{type(exc).__name__}"
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

    def compare_many(
        self,
        pairs: list[tuple[str, str]],
        *,
        use_model: bool = True,
    ) -> list[Relation]:
        if not pairs:
            return []

        if use_model:
            keys = [(self.model_name, left, right) for left, right in pairs]
            resolved: list[Relation | None] = [
                self._RELATION_CACHE.get(key) for key in keys
            ]
            missing_indices = [
                index for index, relation in enumerate(resolved) if relation is None
            ]

            if not missing_indices:
                self.status = f"transformers:{self.model_name}:relation-cache"
                return [relation for relation in resolved if relation is not None]

            missing_pairs = [pairs[index] for index in missing_indices]
            model_results = self._model_relations(missing_pairs)
            if model_results is not None:
                for index, relation in zip(missing_indices, model_results):
                    key = keys[index]
                    if len(self._RELATION_CACHE) >= 100000:
                        self._RELATION_CACHE.pop(next(iter(self._RELATION_CACHE)))
                    self._RELATION_CACHE[key] = relation
                    resolved[index] = relation
                return [relation for relation in resolved if relation is not None]

        if self.status == "not-loaded":
            self.status = "heuristic"
        return [self._heuristic(left, right) for left, right in pairs]

    def compare(self, left: str, right: str, *, use_model: bool = True) -> Relation:
        return self.compare_many([(left, right)], use_model=use_model)[0]
