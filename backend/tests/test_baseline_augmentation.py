from app.research.baseline_augmentation import augment_baseline_answer
from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import EvidenceItem


def evidence(evidence_id: str, document_id: str, score: float = 0.8) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        document_id=document_id,
        document_title=document_id,
        text=f"Evidence for {evidence_id}",
        claim=f"Claim for {evidence_id}",
        retrieval_score=score,
        source_reliability=0.8,
        agreement_score=0.8,
        evidence_score=score,
    )


def candidate(answer: str, evidence_id: str, *, qa: float = 0.9, score: float = 0.8) -> AggregatedCandidate:
    return AggregatedCandidate(
        answer=answer,
        score=score,
        support_documents=1,
        max_qa_score=qa,
        max_evidence_score=score,
        evidence_ids=[evidence_id],
    )


def test_baseline_is_preserved_when_no_candidate_clears_threshold():
    items = [evidence("a1", "d1", 0.3)]
    result = augment_baseline_answer(
        "The capital is Canberra.",
        [candidate("Sydney", "a1", qa=0.2, score=0.3)],
        evidence=items,
        graph=[],
        threshold=0.8,
    )
    assert result.text == "The capital is Canberra."
    assert result.additions == []


def test_candidate_already_in_baseline_is_not_duplicated():
    items = [evidence("a1", "d1")]
    result = augment_baseline_answer(
        "The answer is Canberra.",
        [candidate("Canberra", "a1")],
        evidence=items,
        graph=[],
        threshold=0.5,
    )
    assert result.text == "The answer is Canberra."
    assert result.additions == []


def test_strong_novel_candidate_is_appended_without_replacing_baseline():
    items = [evidence("a1", "d1", 0.9)]
    result = augment_baseline_answer(
        "The work opened in 2001.",
        [candidate("2012", "a1", qa=0.95, score=0.9)],
        evidence=items,
        graph=[],
        threshold=0.6,
    )
    assert result.text.startswith("The work opened in 2001.")
    assert "2012" in result.text
    assert len(result.additions) == 1
