from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.demo import load_demo
from app.schemas import (
    DocumentCreate,
    DocumentRecord,
    ExperimentRequest,
    ExperimentResponse,
    HealthResponse,
    InjectionRequest,
    QueryRequest,
    QueryResponse,
)
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


settings = get_settings()
store = DocumentStore(settings.data_file)
pipeline = EvidenceGuardPipeline(settings, store)

app = FastAPI(
    title="EvidenceGuard API",
    version="0.1.0",
    description="Conflict-aware RAG research prototype.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents=store.count(),
        embedding_engine=pipeline.retriever.engine_status,
        nli_engine=pipeline.nli.status,
        llm_configured=pipeline.generator.configured,
    )


@app.get("/api/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return store.list()


@app.post("/api/documents", response_model=DocumentRecord, status_code=201)
def add_document(document: DocumentCreate) -> DocumentRecord:
    return store.add(document)


@app.post("/api/documents/file", response_model=DocumentRecord, status_code=201)
async def add_file(
    file: UploadFile = File(...),
    source_reliability: float = 0.70,
) -> DocumentRecord:
    if not 0.0 <= source_reliability <= 1.0:
        raise HTTPException(400, "source_reliability must be between 0 and 1")

    data = await file.read()
    filename = file.filename or "uploaded document"
    content_type = (file.content_type or "").lower()

    try:
        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = data.decode("utf-8")
    except Exception as exc:
        raise HTTPException(400, f"Could not read file: {type(exc).__name__}") from exc

    if len(text.strip()) < 10:
        raise HTTPException(400, "No usable text found in the uploaded file")

    return store.add(
        DocumentCreate(
            title=filename,
            text=text,
            source_reliability=source_reliability,
            tags=["upload"],
        )
    )


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    if not store.delete(document_id):
        raise HTTPException(404, "Document not found")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    return await pipeline.query(
        request.question,
        top_k=request.top_k,
        abstain_threshold=request.abstain_threshold,
        use_nli=request.use_nli,
    )


@app.post("/api/inject", response_model=DocumentRecord, status_code=201)
def inject_conflict(request: InjectionRequest) -> DocumentRecord:
    return store.add(
        DocumentCreate(
            title=request.title,
            text=request.claim,
            source_reliability=request.source_reliability,
            tags=["synthetic", "conflict-injection"],
        ),
        injected=True,
    )


@app.delete("/api/inject", status_code=200)
def clear_injected() -> dict[str, int]:
    return {"removed": store.clear_injected()}


@app.post("/api/experiments/attack", response_model=ExperimentResponse)
async def attack_experiment(request: ExperimentRequest) -> ExperimentResponse:
    baseline = await pipeline.query(request.question, top_k=request.top_k)
    ids: list[str] = []
    try:
        for index, claim in enumerate(request.injected_claims, start=1):
            record = store.add(
                DocumentCreate(
                    title=f"Synthetic conflict {index}",
                    text=claim,
                    source_reliability=0.65,
                    tags=["experiment", "synthetic"],
                ),
                injected=True,
            )
            ids.append(record.id)
        attacked = await pipeline.query(request.question, top_k=request.top_k)
    finally:
        store.clear_injected()

    return ExperimentResponse(
        baseline=baseline,
        attacked=attacked,
        injected_document_ids=ids,
    )


@app.post("/api/demo/load")
def demo_load() -> dict[str, int]:
    return {"added": load_demo(store), "documents": store.count()}
