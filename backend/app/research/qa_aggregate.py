from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar

from app.schemas import EvidenceItem


_NORMALIZE = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")


@dataclass(slots=True)
class QAPrediction:
    answer: str
    qa_score: float
    evidence_id: str
    document_id: str
    evidence_score: float


@dataclass(slots=True)
class AggregatedCandidate:
    answer: str
    score: float
    support_documents: int
    max_qa_score: float
    max_evidence_score: float
    evidence_ids: list[str]


@dataclass(slots=True)
class AggregatedAnswer:
    text: str
    candidates: list[AggregatedCandidate]
    engine: str


def _normalize_answer(text: str) -> str:
    value = _NORMALIZE.sub(" ", text.lower()).strip()
    return _SPACE.sub(" ", value)


def aggregate_predictions(
    predictions: list[QAPrediction],
    *,
    max_candidates: int = 3,
    relative_score_floor: float = 0.50,
) -> list[AggregatedCandidate]:
    grouped: dict[str, list[QAPrediction]] = {}
    for prediction in predictions:
        key = _normalize_answer(prediction.answer)
        if not key:
            continue
        grouped.setdefault(key, []).append(prediction)

    candidates: list[AggregatedCandidate] = []
    for group in grouped.values():
        # Do not count duplicate claims from the same document as extra support.
        best_by_document: dict[str, QAPrediction] = {}
        for item in group:
            old = best_by_document.get(item.document_id)
            if old is None or item.qa_score > old.qa_score:
                best_by_document[item.document_id] = item

        unique = list(best_by_document.values())
        weighted = [
            item.qa_score * (0.45 + 0.55 * item.evidence_score)
            for item in unique
        ]
        mean_weighted = sum(weighted) / len(weighted)
        support_bonus = min(0.20, 0.07 * math.log2(1 + len(unique)))
        score = mean_weighted + support_bonus

        representative = max(
            unique,
            key=lambda item: (
                item.qa_score * (0.45 + 0.55 * item.evidence_score),
                item.qa_score,
                item.evidence_score,
            ),
        )
        candidates.append(
            AggregatedCandidate(
                answer=representative.answer.strip(),
                score=round(score, 6),
                support_documents=len(unique),
                max_qa_score=round(max(item.qa_score for item in unique), 6),
                max_evidence_score=round(max(item.evidence_score for item in unique), 6),
                evidence_ids=[item.evidence_id for item in unique],
            )
        )

    candidates.sort(
        key=lambda item: (
            item.score,
            item.support_documents,
            item.max_qa_score,
            item.max_evidence_score,
        ),
        reverse=True,
    )
    if not candidates:
        return []

    threshold = candidates[0].score * relative_score_floor
    return [
        candidate
        for candidate in candidates
        if candidate.score >= threshold
    ][:max_candidates]


class ExtractiveQAAggregator:
    """Independent-document QA followed by candidate aggregation.

    This is intentionally separate from the production answer generator while
    it is being evaluated. It consumes only the question and retrieved evidence;
    RAMDocs evaluation labels never enter inference.
    """

    _PIPELINE_CACHE: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model_name: str = "deepset/minilm-uncased-squad2",
        *,
        batch_size: int = 16,
        min_qa_score: float = 0.05,
        max_evidence: int = 10,
        max_candidates: int = 3,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.min_qa_score = min_qa_score
        self.max_evidence = max_evidence
        self.max_candidates = max_candidates
        self._pipeline = None
        self.status = "not-loaded"

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        cached = self._PIPELINE_CACHE.get(self.model_name)
        if cached is not None:
            self._pipeline = cached
            self.status = f"transformers-qa:{self.model_name}:cached"
            return cached

        from transformers import pipeline

        qa = pipeline(
            "question-answering",
            model=self.model_name,
            tokenizer=self.model_name,
            device=-1,
        )
        self._PIPELINE_CACHE[self.model_name] = qa
        self._pipeline = qa
        self.status = f"transformers-qa:{self.model_name}"
        return qa

    def answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> AggregatedAnswer:
        if not evidence:
            return AggregatedAnswer(
                text="I do not have enough retrieved evidence to answer this question.",
                candidates=[],
                engine="qa-empty",
            )

        # Keep only the highest-scored evidence item for each document/context.
        best_by_document: dict[str, EvidenceItem] = {}
        for item in evidence:
            old = best_by_document.get(item.document_id)
            if old is None or item.evidence_score > old.evidence_score:
                best_by_document[item.document_id] = item

        selected = sorted(
            best_by_document.values(),
            key=lambda item: (
                item.evidence_score,
                item.retrieval_score,
                item.agreement_score,
            ),
            reverse=True,
        )[: self.max_evidence]

        qa = self._load()
        inputs = [
            {"question": question, "context": item.text}
            for item in selected
        ]
        try:
            outputs = qa(
                inputs,
                batch_size=self.batch_size,
                handle_impossible_answer=True,
                max_answer_len=40,
            )
        except TypeError:
            outputs = [
                qa(
                    item,
                    handle_impossible_answer=True,
                    max_answer_len=40,
                )
                for item in inputs
            ]

        if isinstance(outputs, dict):
            outputs = [outputs]

        predictions: list[QAPrediction] = []
        for item, output in zip(selected, outputs):
            answer = str(output.get("answer", "")).strip()
            score = float(output.get("score", 0.0))
            if not answer or score < self.min_qa_score:
                continue
            predictions.append(
                QAPrediction(
                    answer=answer,
                    qa_score=score,
                    evidence_id=item.id,
                    document_id=item.document_id,
                    evidence_score=item.evidence_score,
                )
            )

        candidates = aggregate_predictions(
            predictions,
            max_candidates=self.max_candidates,
        )
        if not candidates:
            return AggregatedAnswer(
                text="I do not have a sufficiently supported extractive answer.",
                candidates=[],
                engine=self.status,
            )

        text = "; ".join(candidate.answer for candidate in candidates)
        return AggregatedAnswer(
            text=text,
            candidates=candidates,
            engine=self.status,
        )
