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
        # Duplicate claims from the same document do not count as extra support.
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
        support_bonus = min(0.25, 0.10 * math.log2(1 + len(unique)))
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

    The implementation uses AutoModelForQuestionAnswering directly so it is
    compatible with both Transformers 4.x and 5.x, where the legacy generic
    question-answering pipeline may not be registered.
    """

    _MODEL_CACHE: ClassVar[dict[str, tuple[Any, Any]]] = {}

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
        self._tokenizer = None
        self._model = None
        self.status = "not-loaded"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._tokenizer, self._model = cached
            self.status = f"transformers-qa:{self.model_name}:cached"
            return cached

        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._MODEL_CACHE[self.model_name] = (tokenizer, model)
        self.status = f"transformers-qa:{self.model_name}"
        return tokenizer, model

    def _predict(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> list[QAPrediction]:
        import torch

        tokenizer, model = self._load()
        predictions: list[QAPrediction] = []

        for start in range(0, len(evidence), self.batch_size):
            batch = evidence[start : start + self.batch_size]
            encoded = tokenizer(
                [question] * len(batch),
                [item.text for item in batch],
                return_tensors="pt",
                padding=True,
                truncation="only_second",
                max_length=384,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")

            with torch.inference_mode():
                output = model(**encoded)

            for row, item in enumerate(batch):
                start_logits = output.start_logits[row]
                end_logits = output.end_logits[row]
                sequence_ids = encoded.sequence_ids(row)
                context_positions = [
                    index
                    for index, sequence_id in enumerate(sequence_ids)
                    if sequence_id == 1
                ]
                if not context_positions:
                    continue

                input_ids = encoded["input_ids"][row]
                cls_matches = (input_ids == tokenizer.cls_token_id).nonzero(as_tuple=False)
                cls_index = int(cls_matches[0].item()) if len(cls_matches) else 0
                null_score = float(start_logits[cls_index] + end_logits[cls_index])

                masked_start = start_logits.clone()
                masked_end = end_logits.clone()
                valid = torch.zeros_like(masked_start, dtype=torch.bool)
                valid[context_positions] = True
                masked_start[~valid] = -1e9
                masked_end[~valid] = -1e9

                top_k = min(20, len(context_positions))
                top_starts = torch.topk(masked_start, k=top_k).indices.tolist()
                top_ends = torch.topk(masked_end, k=top_k).indices.tolist()

                best_score = float("-inf")
                best_span: tuple[int, int] | None = None
                for token_start in top_starts:
                    for token_end in top_ends:
                        if token_end < token_start or token_end - token_start > 40:
                            continue
                        score = float(
                            start_logits[token_start] + end_logits[token_end]
                        )
                        if score > best_score:
                            best_score = score
                            best_span = (token_start, token_end)

                if best_span is None or best_score <= null_score:
                    continue

                token_start, token_end = best_span
                char_start = int(offsets[row, token_start, 0])
                char_end = int(offsets[row, token_end, 1])
                answer = item.text[char_start:char_end].strip()
                if not answer:
                    continue

                # SQuAD2-style confidence: how strongly the best span beats the
                # learned no-answer (CLS) alternative.
                qa_score = float(torch.sigmoid(torch.tensor(best_score - null_score)))
                if qa_score < self.min_qa_score:
                    continue

                predictions.append(
                    QAPrediction(
                        answer=answer,
                        qa_score=qa_score,
                        evidence_id=item.id,
                        document_id=item.document_id,
                        evidence_score=item.evidence_score,
                    )
                )

        return predictions

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

        # Keep only the highest-scored evidence item for each source document.
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

        predictions = self._predict(question, selected)
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
