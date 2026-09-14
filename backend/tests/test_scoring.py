from app.schemas import ConflictEdge
from app.services.scoring import agreement_scores, answer_confidence, conflict_rate


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
