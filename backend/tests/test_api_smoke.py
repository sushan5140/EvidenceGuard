"""End-to-end, CPU-only smoke tests for the actual FastAPI routes.

Neural backends are disabled explicitly so these checks do not download models.
They check the demo experience and persistence semantics, not model accuracy.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app.schemas import DocumentCreate
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    main = importlib.import_module("app.main")
    settings = main.settings.model_copy(
        update={
            "data_path": str(tmp_path / "documents.json"),
            "enable_local_models": False,
            "llm_api_base": None,
            "llm_api_key": None,
            "llm_model": None,
        }
    )
    store = DocumentStore(settings.data_file)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "pipeline", EvidenceGuardPipeline(settings, store))
    with TestClient(main.app) as test_client:
        yield test_client, store


def test_demo_query_and_document_lifecycle(client):
    http, _ = client
    assert http.get("/api/health").status_code == 200
    loaded = http.post("/api/demo/load")
    assert loaded.status_code == 200
    assert loaded.json()["added"] == 3

    # Re-loading demo must not silently duplicate the sources.
    assert http.post("/api/demo/load").json()["added"] == 0
    documents = http.get("/api/documents").json()
    assert len(documents) == 3

    for mode in ("basic_rag", "evidenceguard"):
        response = http.post(
            "/api/query",
            json={
                "question": "When did the Eiffel Tower open to the public?",
                "top_k": 8,
                "use_nli": False,
                "mode": mode,
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["mode"] == mode
        assert data["answer"]
        assert isinstance(data["evidence"], list)
        assert isinstance(data["graph"], list)

    document_id = documents[0]["id"]
    assert http.delete(f"/api/documents/{document_id}").status_code == 204
    assert http.delete(f"/api/documents/{document_id}").status_code == 404
    assert len(http.get("/api/documents").json()) == 2


def test_attack_keeps_preexisting_synthetic_evidence(client):
    http, store = client
    http.post("/api/demo/load")
    earlier = store.add(
        DocumentCreate(
            title="Earlier injection",
            text="A previous experimental claim said the tower opened in 1900.",
            source_reliability=0.2,
        ),
        injected=True,
    )
    response = http.post(
        "/api/experiments/attack",
        json={
            "question": "When did the Eiffel Tower open?",
            "injected_claims": [
                "The Eiffel Tower opened to the public in 1905, not 1889."
            ],
            "mode": "conflict_aware",
            "top_k": 8,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["baseline"]["answer"]
    assert payload["attacked"]["answer"]
    ids = {item["id"] for item in http.get("/api/documents").json()}
    assert earlier.id in ids
    assert all(item not in ids for item in payload["injected_document_ids"])


def test_text_file_upload_and_invalid_pdf(client):
    http, _ = client
    uploaded = http.post(
        "/api/documents/file",
        files={"file": ("source.txt", b"The Eiffel Tower opened in 1889.", "text/plain")},
    )
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["title"] == "source.txt"

    malformed = http.post(
        "/api/documents/file",
        files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
    )
    assert malformed.status_code == 400
