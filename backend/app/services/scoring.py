from __future__ import annotations

from collections import defaultdict

from app.schemas import ConflictEdge


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
