from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


ResearchMode = Literal[
    "basic_rag",
    "hybrid_rag",
    "conflict_aware",
    "consensus_rag",
    "evidenceguard",
]


class DocumentCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    text: str = Field(min_length=10)
    source_url: HttpUrl | None = None
    source_reliability: float = Field(default=0.70, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class DocumentRecord(DocumentCreate):
    id: str
    created_at: datetime
    injected: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=8, ge=2, le=20)
    abstain_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    use_nli: bool = True
    mode: ResearchMode = "evidenceguard"


class EvidenceItem(BaseModel):
    id: str
    document_id: str
    document_title: str
    text: str
    claim: str
    retrieval_score: float
    source_reliability: float
    agreement_score: float
    evidence_score: float
    injected: bool = False


class ConflictEdge(BaseModel):
    source: str
    target: str
    relation: Literal["supports", "contradicts", "neutral"]
    confidence: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    abstained: bool
    conflict_rate: float
    evidence: list[EvidenceItem]
    graph: list[ConflictEdge]
    generator: Literal["extractive", "llm"]
    model_status: dict[str, str]
    mode: ResearchMode = "evidenceguard"


class InjectionRequest(BaseModel):
    title: str = Field(default="Injected conflicting evidence", min_length=2)
    claim: str = Field(min_length=5)
    source_reliability: float = Field(default=0.60, ge=0.0, le=1.0)


class ExperimentRequest(BaseModel):
    question: str = Field(min_length=3)
    injected_claims: list[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=8, ge=2, le=20)
    mode: ResearchMode = "evidenceguard"


class ExperimentResponse(BaseModel):
    baseline: QueryResponse
    attacked: QueryResponse
    injected_document_ids: list[str]


class BenchmarkRequest(BaseModel):
    modes: list[ResearchMode] = Field(
        default_factory=lambda: [
            "basic_rag",
            "hybrid_rag",
            "conflict_aware",
            "consensus_rag",
            "evidenceguard",
        ]
    )
    conflict_ratios: list[float] = Field(
        default_factory=lambda: [0.0, 0.10, 0.25, 0.50, 0.75]
    )
    use_nli: bool = False
    use_local_models: bool = False


class BenchmarkSummaryRow(BaseModel):
    mode: ResearchMode
    conflict_ratio: float
    samples: int
    accuracy: float
    selective_accuracy: float
    coverage: float
    abstention_rate: float
    mean_confidence: float
    ece: float
    conflict_precision: float
    conflict_recall: float
    conflict_f1: float


class BenchmarkResponse(BaseModel):
    benchmark: str
    cases: int
    rows: list[BenchmarkSummaryRow]
    notes: list[str]


class HealthResponse(BaseModel):
    status: str
    documents: int
    embedding_engine: str
    nli_engine: str
    llm_configured: bool
