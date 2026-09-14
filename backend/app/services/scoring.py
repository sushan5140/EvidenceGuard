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
    """Keep the strongest evidence anchor and prune direct contradictions.

    The full ranked evidence and conflict graph remain visible for auditability.
    Consensus selection changes only the compact evidence set used to assemble
    an answer in consensus_rag / EvidenceGuard.

    Crucially, graph popularity is not allowed to replace the strongest scored
    claim. This prevents a larger but lower-quality conflicting cluster from
    displacing the evidence-score winner.
    """

    if not evidence or max_items <= 0:
        return []

    ranked = sorted(
        evidence,
        key=lambda item: (
            item.evidence_score,
            item.agreement_score,
            item.retrieval_score,
        ),
        reverse=True,
    )

    strong_conflicts: dict[str, set[str]] = defaultdict(set)
    support_strength = defaultdict(float)
    for edge in edges:
        if edge.relation == "supports":
            support_strength[edge.source] += edge.confidence
            support_strength[edge.target] += edge.confidence
        elif (
            edge.relation == "contradicts"
            and edge.confidence >= contradiction_threshold
        ):
            strong_conflicts[edge.source].add(edge.target)
            strong_conflicts[edge.target].add(edge.source)

    anchor = ranked[0]
    selected = [anchor]

    # Among non-conflicting candidates, prefer claims that support the anchor,
    # then fall back to their original evidence score ordering.
    remaining = sorted(
        ranked[1:],
        key=lambda item: (
            1.0 if item.id not in strong_conflicts[anchor.id] else 0.0,
            support_strength[item.id],
            item.evidence_score,
            item.agreement_score,
            item.retrieval_score,
        ),
        reverse=True,
    )

    for item in remaining:
        if any(
            chosen.id in strong_conflicts[item.id]
            for chosen in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break

    return selected
