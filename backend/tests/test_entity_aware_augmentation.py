from app.research.entity_aware_augmentation import augment_with_distinct_entities
from app.research.qa_aggregate import AggregatedCandidate
from app.schemas import EvidenceItem


class FakeClassifier:
    def __init__(self, probability: float, threshold: float = 0.5):
        self.probability = probability
        self.threshold = threshold

    def same_entity_probability(self, left: str, right: str) -> float:
        return self.probability


def ev(eid: str, did: str, score: float, text: str) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        document_id=did,
        document_title=did,
        text=text,
        claim=text,
        retrieval_score=score,
        source_reliability=0.8,
        agreement_score=0.8,
        evidence_score=score,
    )


def cand(answer: str, eid: str) -> AggregatedCandidate:
    return AggregatedCandidate(
        answer=answer,
        score=0.9,
        support_documents=1,
        max_qa_score=0.95,
        max_evidence_score=0.9,
        evidence_ids=[eid],
    )


def test_same_entity_candidate_is_blocked():
    evidence = [
        ev("b1", "base", 0.95, "Alex Morgan the athlete was born in 1989."),
        ev("x1", "other", 0.70, "Alex Morgan the athlete was born in 1992."),
    ]
    result = augment_with_distinct_entities(
        "Alex Morgan was born in 1989.",
        [cand("1992", "x1")],
        evidence=evidence,
        graph=[],
        classifier=FakeClassifier(0.9),
    )
    assert result.additions == []


def test_distinct_entity_candidate_can_be_appended():
    evidence = [
        ev("b1", "base", 0.95, "Alex Morgan the athlete was born in 1989."),
        ev("x1", "other", 0.70, "Alex Morgan the visual artist was born in 1974."),
    ]
    result = augment_with_distinct_entities(
        "Alex Morgan was born in 1989.",
        [cand("1974", "x1")],
        evidence=evidence,
        graph=[],
        classifier=FakeClassifier(0.1),
    )
    assert len(result.additions) == 1
    assert "1974" in result.text


def test_candidate_from_baseline_document_is_never_readded():
    evidence = [
        ev("b1", "base", 0.95, "The work opened in 2001 and was revised later."),
    ]
    result = augment_with_distinct_entities(
        "The work opened in 2001.",
        [cand("revised later", "b1")],
        evidence=evidence,
        graph=[],
        classifier=FakeClassifier(0.0),
    )
    assert result.additions == []
