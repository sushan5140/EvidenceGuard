from app.services.chunking import chunk_text, extract_claims


def test_chunking_preserves_document_metadata():
    chunks = chunk_text(
        "word " * 250,
        document_id="doc-1",
        document_title="Demo",
        source_reliability=0.8,
        injected=False,
        chunk_words=100,
        overlap_words=20,
    )
    assert len(chunks) >= 3
    assert all(chunk.document_id == "doc-1" for chunk in chunks)


def test_claim_extraction_splits_contrast():
    claims = extract_claims(
        "The system is accurate on clean data, but it performs worse under conflicting evidence."
    )
    assert len(claims) == 2
