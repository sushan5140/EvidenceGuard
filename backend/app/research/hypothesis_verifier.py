from __future__ import annotations

import math
from dataclasses import dataclass

from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import ConflictEdge, EvidenceItem


@dataclass(slots=True)
class VerifiedCandidate:
    answer: str
    verification_score: float
    support_documents: int
    support_quality: float
    support_diversity: float
    support_cohesion: float
    contradiction_pressure: float
    max_qa_score: float
    evidence_ids: list[str]


@dataclass(slots=True)
class HypothesisVerificationResult:
    text: str
    candidates: list[VerifiedCandidate]
    threshold: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _diversity_score(support_documents: int) -> float:
    if support_documents <= 0:
        return 0.0
    return min(1.0, math.log2(1 + support_documents) / 2.0)


def score_candidate(
    candidate: AggregatedCandidate,
    *,
    evidence_by_id: dict[str, EvidenceItem],
    graph: list[ConflictEdge],
    other_candidate_evidence: set[str],
) -> VerifiedCandidate:
    support_items = [
        evidence_by_id[evidence_id]
        for evidence_id in candidate.evidence_ids
        if evidence_id in evidence_by_id
    ]
    support_quality = (
        sum(item.evidence_score for item in support_items) / len(support_items)
        if support_items
        else candidate.max_evidence_score
    )
    support_ids = {item.id for item in support_items}
    diversity = _diversity_score(candidate.support_documents)

    internal_support: list[float] = []
    contradiction_terms: list[float] = []
    for edge in graph:
        left_in = edge.source in support_ids
        right_in = edge.target in support_ids
        if left_in and right_in:
            if edge.relation == "supports":
                internal_support.append(edge.confidence)
            elif edge.relation == "contradicts":
                contradiction_terms.append(edge.confidence)
            continue
        if edge.relation != "contradicts":
            continue

        other = edge.target if left_in else edge.source if right_in else None
        if other is None:
            continue
        ambiguity_discount = 0.40 if other in other_candidate_evidence else 1.0
        contradiction_terms.append(edge.confidence * ambiguity_discount)

    cohesion = (
        sum(internal_support) / len(internal_support)
        if internal_support
        else 0.50
    )
    contradiction = (
        sum(contradiction_terms) / len(contradiction_terms)
        if contradiction_terms
        else 0.0
    )

    raw = (
        0.35 * candidate.max_qa_score
        + 0.35 * support_quality
        + 0.20 * diversity
        + 0.10 * cohesion
        - 0.25 * contradiction
    )

    return VerifiedCandidate(
        answer=candidate.answer,
        verification_score=round(_clamp(raw), 6),
        support_documents=candidate.support_documents,
        support_quality=round(_clamp(support_quality), 6),
        support_diversity=round(diversity, 6),
        support_cohesion=round(_clamp(cohesion), 6),
        contradiction_pressure=round(_clamp(contradiction), 6),
        max_qa_score=candidate.max_qa_score,
        evidence_ids=list(candidate.evidence_ids),
    )


def verify_answer_hypotheses(
    candidates: list[AggregatedCandidate],
    *,
    evidence: list[EvidenceItem],
    graph: list[ConflictEdge],
    secondary_threshold: float,
    max_candidates: int = 3,
) -> HypothesisVerificationResult:
    if not candidates:
        return HypothesisVerificationResult(
            text="I do not have a sufficiently supported extractive answer.",
            candidates=[],
            threshold=secondary_threshold,
        )

    evidence_by_id = {item.id: item for item in evidence}
    support_sets = [set(candidate.evidence_ids) for candidate in candidates]

    verified: list[VerifiedCandidate] = []
    for index, candidate in enumerate(candidates):
        other_ids: set[str] = set()
        for other_index, support in enumerate(support_sets):
            if other_index != index:
                other_ids.update(support)
        verified.append(
            score_candidate(
                candidate,
                evidence_by_id=evidence_by_id,
                graph=graph,
                other_candidate_evidence=other_ids,
            )
        )

    verified.sort(
        key=lambda item: (
            item.verification_score,
            item.support_documents,
            item.max_qa_score,
            item.support_quality,
        ),
        reverse=True,
    )

    selected = [verified[0]]
    selected.extend(
        candidate
        for candidate in verified[1:]
        if candidate.verification_score >= secondary_threshold
    )
    selected = selected[:max_candidates]

    return HypothesisVerificationResult(
        text="; ".join(candidate.answer for candidate in selected),
        candidates=selected,
        threshold=secondary_threshold,
    )
