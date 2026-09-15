from app.research.hypothesis_verifier import (
    score_candidate,
    verify_answer_hypotheses,
)
from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import ConflictEdge, EvidenceItem


def evidence(evidence_id: str, *, document_id: str, evidence_score: float = 0.8) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        document_id=document_id,
        document_title=document_id,
        text=f"Evidence text for {evidence_id}",
        claim=f"Claim for {evidence_id}",
        retrieval_score=evidence_score,
        source_reliability=0.7,
        agreement_score=0.7,
        evidence_score=evidence_score,
    )


def candidate(
    answer: str,
    *,
    support_documents: int = 1,
    max_qa_score: float = 0.85,
    max_evidence_score: float = 0.8,
    evidence_ids: list[str],
) -> AggregatedCandidate:
    return AggregatedCandidate(
        answer=answer,
        score=0.8,
        support_documents=support_documents,
        max_qa_score=max_qa_score,
        max_evidence_score=max_evidence_score,
        evidence_ids=evidence_ids,
    )


def test_more_independent_support_increases_score():
    items = [evidence("a1", document_id="d1"), evidence("a2", document_id="d2")]
    single = candidate("Alpha", support_documents=1, evidence_ids=["a1"])
    double = candidate("Alpha", support_documents=2, evidence_ids=["a1", "a2"])
    lookup = {item.id: item for item in items}

    s1 = score_candidate(single, evidence_by_id=lookup, graph=[], other_candidate_evidence=set())
    s2 = score_candidate(double, evidence_by_id=lookup, graph=[], other_candidate_evidence=set())
    assert s2.verification_score > s1.verification_score


def test_unsupported_contradiction_reduces_candidate_score():
    items = [evidence("a1", document_id="d1"), evidence("x1", document_id="d2")]
    item = candidate("Alpha", evidence_ids=["a1"])
    graph = [ConflictEdge(source="a1", target="x1", relation="contradicts", confidence=0.9)]
    lookup = {x.id: x for x in items}

    clean = score_candidate(item, evidence_by_id=lookup, graph=[], other_candidate_evidence=set())
    conflicted = score_candidate(item, evidence_by_id=lookup, graph=graph, other_candidate_evidence=set())
    assert conflicted.verification_score < clean.verification_score


def test_alternative_hypothesis_discounts_cross_candidate_conflict():
    items = [evidence("a1", document_id="d1"), evidence("b1", document_id="d2")]
    item = candidate("Alpha", evidence_ids=["a1"])
    graph = [ConflictEdge(source="a1", target="b1", relation="contradicts", confidence=0.9)]
    lookup = {x.id: x for x in items}

    normal = score_candidate(item, evidence_by_id=lookup, graph=graph, other_candidate_evidence=set())
    ambiguous = score_candidate(item, evidence_by_id=lookup, graph=graph, other_candidate_evidence={"b1"})
    assert ambiguous.verification_score > normal.verification_score


def test_primary_survives_but_weak_secondary_is_filtered():
    items = [
        evidence("a1", document_id="d1", evidence_score=0.85),
        evidence("b1", document_id="d2", evidence_score=0.25),
    ]
    candidates = [
        candidate("Alpha", max_qa_score=0.9, evidence_ids=["a1"]),
        candidate("Beta", max_qa_score=0.2, max_evidence_score=0.25, evidence_ids=["b1"]),
    ]

    result = verify_answer_hypotheses(
        candidates,
        evidence=items,
        graph=[],
        secondary_threshold=0.60,
    )
    assert [item.answer for item in result.candidates] == ["Alpha"]
