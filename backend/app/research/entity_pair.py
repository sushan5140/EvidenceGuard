from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


_WORD = re.compile(r"[A-Za-z0-9']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class EntityPairCalibration:
    threshold: float
    precision: float
    recall: float
    f2: float
    train_pairs: int
    calibration_pairs: int


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD.findall(text) if len(token) > 2}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _length_ratio(left: str, right: str) -> float:
    a, b = max(1, len(_WORD.findall(left))), max(1, len(_WORD.findall(right)))
    return min(a, b) / max(a, b)


def _document_views(text: str) -> tuple[str, str]:
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(sentences) >= 4:
        midpoint = len(sentences) // 2
        return " ".join(sentences[:midpoint]), " ".join(sentences[midpoint:])

    words = text.split()
    midpoint = max(1, len(words) // 2)
    return " ".join(words[:midpoint]), " ".join(words[midpoint:])


def _documents(record: dict[str, Any]) -> list[dict[str, str]]:
    value = record.get("documents", [])
    if isinstance(value, list):
        return [
            {
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "answer": str(item.get("answer", "")),
            }
            for item in value
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]

    # Hugging Face previews may expose a struct-of-lists representation.
    if isinstance(value, dict):
        texts = value.get("text", [])
        titles = value.get("title", [])
        answers = value.get("answer", [])
        docs = []
        for index, text in enumerate(texts if isinstance(texts, list) else []):
            if not str(text).strip():
                continue
            docs.append({
                "title": str(titles[index]) if isinstance(titles, list) and index < len(titles) else "",
                "text": str(text),
                "answer": str(answers[index]) if isinstance(answers, list) and index < len(answers) else "",
            })
        return docs
    return []


def load_ambigdocs(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "examples", "instances"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else []
    return payload if isinstance(payload, list) else []


class EntityPairClassifier:
    """Learn whether two passages describe the same entity.

    AmbigDocs provides different-entity negatives within each ambiguous-name case.
    Positive pairs are two non-overlapping views of one source document. RAMDocs is
    never used to train the classifier or choose its operating threshold.
    """

    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        )
        self.threshold = 0.50
        self._embedder = None
        self.calibration: EntityPairCalibration | None = None

    def _encoder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _pair_features(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.empty((0, 3), dtype=float)

        texts: list[str] = []
        index: dict[str, int] = {}
        for left, right in pairs:
            for text in (left, right):
                if text not in index:
                    index[text] = len(texts)
                    texts.append(text)

        vectors = np.asarray(
            self._encoder().encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            ),
            dtype=float,
        )

        rows = []
        for left, right in pairs:
            cosine = float(vectors[index[left]] @ vectors[index[right]])
            rows.append([
                cosine,
                _jaccard(left, right),
                _length_ratio(left, right),
            ])
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _examples(
        records: list[dict[str, Any]],
        *,
        max_docs_per_case: int = 4,
    ) -> tuple[list[tuple[str, str]], list[int]]:
        pairs: list[tuple[str, str]] = []
        labels: list[int] = []

        for record in records:
            docs = _documents(record)[:max_docs_per_case]
            usable: list[str] = []
            for doc in docs:
                text = doc["text"].strip()
                if len(text.split()) < 30:
                    continue
                left, right = _document_views(text)
                if len(left.split()) >= 10 and len(right.split()) >= 10:
                    pairs.append((left, right))
                    labels.append(1)
                    usable.append(text)

            # Different documents in an AmbigDocs instance are distinct entities
            # sharing the same ambiguous surface name.
            for i in range(len(usable)):
                for j in range(i + 1, min(len(usable), i + 3)):
                    pairs.append((usable[i], usable[j]))
                    labels.append(0)

        return pairs, labels

    def fit(
        self,
        records: list[dict[str, Any]],
        *,
        train_cases: int = 900,
        calibration_cases: int = 350,
    ) -> EntityPairCalibration:
        ordered = sorted(records, key=lambda item: str(item.get("qid", "")))
        train = ordered[:train_cases]
        calibration = ordered[train_cases : train_cases + calibration_cases]
        if not train or not calibration:
            raise ValueError("AmbigDocs development data is too small for the declared split")

        train_pairs, train_labels = self._examples(train)
        cal_pairs, cal_labels = self._examples(calibration)
        if len(set(train_labels)) < 2 or len(set(cal_labels)) < 2:
            raise ValueError("Entity-pair training/calibration requires both classes")

        self.model.fit(self._pair_features(train_pairs), np.asarray(train_labels))
        probabilities = self.model.predict_proba(self._pair_features(cal_pairs))[:, 1]
        y_true = np.asarray(cal_labels, dtype=int)

        best = None
        for threshold in [0.20 + 0.05 * i for i in range(13)]:
            pred = (probabilities >= threshold).astype(int)
            recall = recall_score(y_true, pred, zero_division=0)
            precision = precision_score(y_true, pred, zero_division=0)
            f2 = fbeta_score(y_true, pred, beta=2.0, zero_division=0)
            # Same-entity false negatives are dangerous because they can allow
            # misinformation to masquerade as a separate valid entity.
            candidate = (f2, recall, precision, -threshold, threshold)
            if best is None or candidate > best[0]:
                best = (candidate, precision, recall, f2, threshold)

        assert best is not None
        _, precision, recall, f2, threshold = best
        self.threshold = float(threshold)
        self.calibration = EntityPairCalibration(
            threshold=self.threshold,
            precision=float(precision),
            recall=float(recall),
            f2=float(f2),
            train_pairs=len(train_pairs),
            calibration_pairs=len(cal_pairs),
        )
        return self.calibration

    def same_entity_probability(self, left: str, right: str) -> float:
        if left.strip() == right.strip():
            return 1.0
        features = self._pair_features([(left, right)])
        return float(self.model.predict_proba(features)[0, 1])

    def same_entity(self, left: str, right: str) -> bool:
        return self.same_entity_probability(left, right) >= self.threshold
