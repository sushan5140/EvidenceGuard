from app.services.nli import Relation
from app.services.pipeline import EvidenceGuardPipeline


class RecordingNLI:
    def __init__(self):
        self.pairs = []

    def compare_many(self, pairs, *, use_model=True):
        self.pairs.extend(pairs)
        return [Relation(label="neutral", confidence=0.9) for _ in pairs]


def test_graph_compares_cross_document_claims_without_lexical_overlap_gate():
    pipeline = EvidenceGuardPipeline.__new__(EvidenceGuardPipeline)
    pipeline.nli = RecordingNLI()

    claims = [
        {
            "id": "a",
            "document_id": "doc-a",
            "claim": "Mercury freezes at a very low temperature.",
        },
        {
            "id": "b",
            "document_id": "doc-b",
            "claim": "The innermost planet remains extremely hot.",
        },
    ]

    graph = pipeline._graph(claims, use_nli=True)

    assert len(pipeline.nli.pairs) == 1
    assert pipeline.nli.pairs[0] == (claims[0]["claim"], claims[1]["claim"])
    assert len(graph) == 1


def test_graph_still_skips_same_document_pairs():
    pipeline = EvidenceGuardPipeline.__new__(EvidenceGuardPipeline)
    pipeline.nli = RecordingNLI()

    claims = [
        {"id": "a", "document_id": "doc-a", "claim": "Claim one."},
        {"id": "b", "document_id": "doc-a", "claim": "Claim two."},
    ]

    graph = pipeline._graph(claims, use_nli=True)

    assert pipeline.nli.pairs == []
    assert graph == []
