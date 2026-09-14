from app.research.qa_aggregate import (
    QAPrediction,
    _normalize_answer,
    aggregate_predictions,
)


def prediction(
    answer: str,
    *,
    qa_score: float,
    evidence_score: float,
    document_id: str,
) -> QAPrediction:
    return QAPrediction(
        answer=answer,
        qa_score=qa_score,
        evidence_id=f"ev-{document_id}",
        document_id=document_id,
        evidence_score=evidence_score,
    )


def test_normalize_answer_collapses_case_and_punctuation():
    assert _normalize_answer("Guido van Rossum.") == "guido van rossum"
    assert _normalize_answer("  Guido---van Rossum ") == "guido van rossum"


def test_repeated_cross_document_candidate_gets_support_bonus():
    predictions = [
        prediction("Canberra", qa_score=0.72, evidence_score=0.80, document_id="a"),
        prediction("Canberra", qa_score=0.70, evidence_score=0.78, document_id="b"),
        prediction("Sydney", qa_score=0.78, evidence_score=0.76, document_id="c"),
    ]

    candidates = aggregate_predictions(predictions, max_candidates=3)

    assert candidates[0].answer == "Canberra"
    assert candidates[0].support_documents == 2


def test_duplicate_claims_from_same_document_do_not_fake_support():
    predictions = [
        prediction("Canberra", qa_score=0.80, evidence_score=0.80, document_id="a"),
        prediction("Canberra", qa_score=0.70, evidence_score=0.80, document_id="a"),
    ]

    candidates = aggregate_predictions(predictions)

    assert candidates[0].support_documents == 1


def test_multiple_plausible_answers_can_survive_relative_floor():
    predictions = [
        prediction("Alpha", qa_score=0.80, evidence_score=0.80, document_id="a"),
        prediction("Beta", qa_score=0.72, evidence_score=0.80, document_id="b"),
        prediction("Weak", qa_score=0.10, evidence_score=0.20, document_id="c"),
    ]

    candidates = aggregate_predictions(
        predictions,
        max_candidates=3,
        relative_score_floor=0.50,
    )
    answers = [candidate.answer for candidate in candidates]

    assert "Alpha" in answers
    assert "Beta" in answers
    assert "Weak" not in answers
