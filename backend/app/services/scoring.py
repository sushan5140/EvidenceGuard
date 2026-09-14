from __future__ import annotations

from collections import defaultdict

from app.schemas import ConflictEdge, EvidenceItem


def agreement_scores(evidence_ids: list[str], edges: list[ConflictEdge]) -> dict[str, float]:
    support = defaultdict(float)
    conflict = defaultdict(float)

    for edge in edges:
        if edge.relation == "supports":
            support[edge.source] += edge.confidence
            support[edge.target] += edge.confidence
        elif edge.relation == "contradicts":
            conflict[edge.source] += edge.confidence
            conflict[edge.target] += edge.confidence

    scores: dict[str, float] = {}
    for evidence_id in evidence_ids:
        s, c = support[evidence_id], conflict[evidence_id]
        # Neutral prior keeps isolated evidence from receiving an extreme score.
        scores[evidence_id] = (1.0 + s) / (2.0 + s + c)
    return scores


def evidence_score(
    retrieval: float,
    reliability: float,
    agreement: float,
    *,
    retrieval_weight: float = 0.45,
    reliability_weight: float = 0.25,
    agreement_weight: float = 0.30,
) -> float:
    total = retrieval_weight + reliability_weight + agreement_weight
    if total <= 0:
        return 0.0
    value = (
        retrieval * retrieval_weight
        + reliability * reliability_weight
        + agreement * agreement_weight
    ) / total
    return max(0.0, min(1.0, value))


def conflict_rate(edges: list[ConflictEdge]) -> float:
    meaningful = [edge for edge in edges if edge.relation != "neutral"]
    if not meaningful:
        return 0.0
    contradictory = [edge for edge in meaningful if edge.relation == "contradicts"]
    return len(contradictory) / len(meaningful)


def answer_confidence(scores: list[float], conflict: float) -> float:
    if not scores:
        return 0.0
    ranked = sorted(scores, reverse=True)
    top = ranked[: min(4, len(ranked))]
    mean_top = sum(top) / len(top)
    evidence_bonus = min(0.08, 0.02 * len(scores))
    conflict_penalty = 0.42 * conflict
    return max(0.0, min(1.0, mean_top + evidence_bonus - conflict_penalty))


def select_consensus_evidence(
    evidence: list[EvidenceItem],
    edges: list[ConflictEdge],
    *,
    max_items: int = 3,
    contradiction_threshold: float = 0.65,
) -> list[EvidenceItem]:
    """Select a compact, mutually consistent evidence set for final answering.

    EvidenceGuard still returns the complete ranked evidence list for auditability.
    This selector is used only by the full EvidenceGuard answer path so the
    baseline modes remain unchanged.
    """

    if not evidence or max_items <= 0:
        return []

    support_strength = defaultdict(float)
    conflict_strength = defaultdict(float)
    strong_conflicts: dict[str, set[str]] = defaultdict(set)

    for edge in edges:
        if edge.relation == "supports":
            support_strength[edge.source] += edge.confidence
            support_strength[edge.target] += edge.confidence
        elif edge.relation == "contradicts":
            conflict_strength[edge.source] += edge.confidence
            conflict_strength[edge.target] += edge.confidence
            if edge.confidence >= contradiction_threshold:
                strong_conflicts[edge.source].add(edge.target)
                strong_conflicts[edge.target].add(edge.source)

    def consensus_rank(item: EvidenceItem) -> tuple[float, float, float]:
        graph_adjustment = (
            0.08 * support_strength[item.id]
            - 0.12 * conflict_strength[item.id]
        )
        return (
            item.evidence_score + graph_adjustment,
            item.agreement_score,
            item.retrieval_score,
        )

    ranked = sorted(evidence, key=consensus_rank, reverse=True)
    selected: list[EvidenceItem] = []

    for item in ranked:
        if any(
            chosen.id in strong_conflicts[item.id]
            for chosen in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break

    return selected or ranked[:1]
