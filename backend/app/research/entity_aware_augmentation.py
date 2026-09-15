from __future__ import annotations

from dataclasses import dataclass

from app.research.baseline_augmentation import rank_candidates
from app.research.entity_pair import EntityPairClassifier
from app.research.metrics import contains_answer
from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import ConflictEdge, EvidenceItem


@dataclass(slots=True)
class EntityAwareAddition:
    answer: str
    verification_score: float
    max_same_entity_probability: float
    support_documents: int


@dataclass(slots=True)
class EntityAwareAugmentationResult:
    text: str
    additions: list[EntityAwareAddition]


def augment_with_distinct_entities(
    baseline_answer: str,
    candidates: list[AggregatedCandidate],
    *,
    evidence: list[EvidenceItem],
    graph: list[ConflictEdge],
    classifier: EntityPairClassifier,
    candidate_score_floor: float = 0.60,
    max_additions: int = 2,
) -> EntityAwareAugmentationResult:
    """Append only candidates backed by passages outside baseline entity neighborhoods."""

    if not candidates or not evidence:
        return EntityAwareAugmentationResult(text=baseline_answer, additions=[])

    evidence_by_id = {item.id: item for item in evidence}
    # Extractive canonical generation uses the top three evidence items.
    baseline_evidence = sorted(
        evidence,
        key=lambda item: item.evidence_score,
        reverse=True,
    )[:3]
    baseline_docs = {item.document_id: item.text for item in baseline_evidence}

    ranked = rank_candidates(candidates, evidence=evidence, graph=graph)
    additions: list[EntityAwareAddition] = []

    for candidate in ranked:
        if candidate.verification_score < candidate_score_floor:
            continue
        if contains_answer(baseline_answer, candidate.answer):
            continue

        support_items = [
            evidence_by_id[evidence_id]
            for evidence_id in candidate.evidence_ids
            if evidence_id in evidence_by_id
        ]
        if not support_items:
            continue

        # If a candidate comes from a document already used by the baseline,
        # it is not a new entity-level answer.
        if any(item.document_id in baseline_docs for item in support_items):
            continue

        probabilities = [
            classifier.same_entity_probability(item.text, baseline_text)
            for item in support_items
            for baseline_text in baseline_docs.values()
        ]
        max_same = max(probabilities) if probabilities else 1.0
        if max_same >= classifier.threshold:
            continue

        if any(contains_answer(existing.answer, candidate.answer) for existing in additions):
            continue

        additions.append(EntityAwareAddition(
            answer=candidate.answer,
            verification_score=candidate.verification_score,
            max_same_entity_probability=round(max_same, 6),
            support_documents=candidate.support_documents,
        ))
        if len(additions) >= max_additions:
            break

    if not additions:
        return EntityAwareAugmentationResult(text=baseline_answer, additions=[])

    suffix = "; ".join(item.answer for item in additions)
    return EntityAwareAugmentationResult(
        text=f"{baseline_answer} Additional entity-specific answer(s): {suffix}",
        additions=additions,
    )
