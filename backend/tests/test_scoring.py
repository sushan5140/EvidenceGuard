from app.schemas import ConflictEdge, EvidenceItem
from app.services.scoring import (
    agreement_scores,
    answer_confidence,
    conflict_rate,
    select_consensus_evidence,
)


def item(
    evidence_id: str,
    *,
    evidence_score: float,
    agreement_score: float = 0.5,
    retrieval_score: float = 0.5,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        document_id=f"doc-{evidence_id}",
        document_title=evidence_id,
        text=f"Evidence text for {evidence_id}.",
        claim=f"Claim {evidence_id}.",
        retrieval_score=retrieval_score,
        source_reliability=0.7,
        agreement_score=agreement_score,
        evidence_score=evidence_score,
    )


def test_conflict_reduces_confidence():
    clean = answer_confidence([0.8, 0.75], 0.0)
    conflicted = answer_confidence([0.8, 0.75], 0.8)
    assert conflicted < clean


def test_conflict_rate_and_agreement():
    edges = [
        ConflictEdge(source="a", target="b", relation="supports", confidence=0.9),
        ConflictEdge(source="a", target="c", relation="contradicts", confidence=0.9),
    ]
    assert conflict_rate(edges) == 0.5
    scores = agreement_scores(["a", "b", "c"], edges)
    assert scores["b"] > scores["c"]


def test_consensus_selector_drops_strongly_contradictory_claims():
    evidence = [
        item("a", evidence_score=0.90, agreement_score=0.80),
        item("b", evidence_score=0.86, agreement_score=0.75),
        item("c", evidence_score=0.82, agreement_score=0.40),
    ]
    edges = [
        ConflictEdge(source="a", target="b", relation="supports", confidence=0.90),
        ConflictEdge(source="a", target="c", relation="contradicts", confidence=0.91),
    ]

    selected = select_consensus_evidence(evidence, edges, max_items=3)
    ids = [entry.id for entry in selected]

    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids


def test_consensus_selector_keeps_a_fallback_when_everything_conflicts():
    evidence = [
        item("a", evidence_score=0.90),
        item("b", evidence_score=0.80),
    ]
    edges = [
        ConflictEdge(source="a", target="b", relation="contradicts", confidence=0.95),
    ]

    selected = select_consensus_evidence(evidence, edges, max_items=2)

    assert len(selected) == 1
