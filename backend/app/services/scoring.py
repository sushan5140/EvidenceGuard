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



def select_answer_cluster_representatives(
    evidence: list[EvidenceItem],
    edges: list[ConflictEdge],
    *,
    max_items: int = 3,
    support_threshold: float = 0.65,
) -> list[EvidenceItem]:
    """Return one strong representative from each evidence-support cluster.

    RAMDocs-style ambiguity can make mutually contradictory answers legitimate
    for different entities sharing the same surface name. Instead of deleting
    contradictions, this exploratory selector groups claims connected by
    strong support edges and ranks those clusters by accumulated evidence mass.

    The selector uses no benchmark labels and never treats contradiction as
    proof that one side is false.
    """

    if not evidence or max_items <= 0:
        return []

    by_id = {item.id: item for item in evidence}
    parent = {item.id: item.id for item in evidence}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        if (
            edge.relation == "supports"
            and edge.confidence >= support_threshold
            and edge.source in by_id
            and edge.target in by_id
        ):
            union(edge.source, edge.target)

    clusters: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence:
        clusters[find(item.id)].append(item)

    ranked_clusters: list[tuple[float, EvidenceItem]] = []
    for members in clusters.values():
        representative = max(
            members,
            key=lambda item: (
                item.evidence_score,
                item.agreement_score,
                item.retrieval_score,
            ),
        )
        # Sublinear support-mass score: repeated support can strengthen a
        # hypothesis without letting cluster size dominate linearly.
        support_mass = sum(item.evidence_score for item in members)
        cluster_score = support_mass / (len(members) ** 0.5)
        ranked_clusters.append((cluster_score, representative))

    ranked_clusters.sort(
        key=lambda pair: (
            pair[0],
            pair[1].evidence_score,
            pair[1].agreement_score,
            pair[1].retrieval_score,
        ),
        reverse=True,
    )
    return [representative for _, representative in ranked_clusters[:max_items]]
