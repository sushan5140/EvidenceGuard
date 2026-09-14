from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "EvidenceGuard API"
    data_path: str = "backend/data/documents.json"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    enable_local_models: bool = True
    cors_origins: str = "http://localhost:3000"

    # These aliases intentionally accept provider-standard environment names.
    llm_api_base: str | None = Field(default=None, validation_alias="LLM_API_BASE")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")

    retrieval_bm25_weight: float = 0.45
    retrieval_dense_weight: float = 0.55
    evidence_retrieval_weight: float = 0.45
    evidence_reliability_weight: float = 0.25
    evidence_agreement_weight: float = 0.30
    default_abstain_threshold: float = 0.48

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EVIDENCEGUARD_",
        extra="ignore",
    )

    @property
    def data_file(self) -> Path:
        path = Path(self.data_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cors_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
