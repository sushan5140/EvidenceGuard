from __future__ import annotations

from dataclasses import dataclass

from app.research.hypothesis_verifier import VerifiedCandidate, score_candidate
from app.research.metrics import contains_answer
from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import ConflictEdge, EvidenceItem


@dataclass(slots=True)
class BaselineAugmentationResult:
    text: str
    additions: list[VerifiedCandidate]
    threshold: float


def rank_candidates(
    candidates: list[AggregatedCandidate],
    *,
    evidence: list[EvidenceItem],
    graph: list[ConflictEdge],
) -> list[VerifiedCandidate]:
    evidence_by_id = {item.id: item for item in evidence}
    support_sets = [set(candidate.evidence_ids) for candidate in candidates]

    ranked: list[VerifiedCandidate] = []
    for index, candidate in enumerate(candidates):
        other_ids: set[str] = set()
        for other_index, support in enumerate(support_sets):
            if other_index != index:
                other_ids.update(support)
        ranked.append(
            score_candidate(
                candidate,
                evidence_by_id=evidence_by_id,
                graph=graph,
                other_candidate_evidence=other_ids,
            )
        )

    ranked.sort(
        key=lambda item: (
            item.verification_score,
            item.support_documents,
            item.max_qa_score,
            item.support_quality,
        ),
        reverse=True,
    )
    return ranked


def augment_baseline_answer(
    baseline_answer: str,
    candidates: list[AggregatedCandidate],
    *,
    evidence: list[EvidenceItem],
    graph: list[ConflictEdge],
    threshold: float,
    max_additions: int = 2,
) -> BaselineAugmentationResult:
    ranked = rank_candidates(candidates, evidence=evidence, graph=graph)

    additions: list[VerifiedCandidate] = []
    for candidate in ranked:
        if candidate.verification_score < threshold:
            continue
        if contains_answer(baseline_answer, candidate.answer):
            continue
        if any(contains_answer(existing.answer, candidate.answer) for existing in additions):
            continue
        additions.append(candidate)
        if len(additions) >= max_additions:
            break

    if not additions:
        return BaselineAugmentationResult(
            text=baseline_answer,
            additions=[],
            threshold=threshold,
        )

    suffix = "; ".join(item.answer for item in additions)
    return BaselineAugmentationResult(
        text=f"{baseline_answer} Additional supported answer(s): {suffix}",
        additions=additions,
        threshold=threshold,
    )
