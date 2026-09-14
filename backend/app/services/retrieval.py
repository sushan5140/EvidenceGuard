from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from app.schemas import DocumentRecord
from app.services.chunking import Chunk, chunk_text


_TOKEN = re.compile(r"[A-Za-z0-9']+")
RetrievalStrategy = Literal["bm25", "dense", "hybrid"]


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    chunk: Chunk
    score: float
    bm25_score: float
    dense_score: float


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(lo, hi):
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


class HybridRetriever:
    """BM25 + semantic retrieval with a TF-IDF fallback.

    Sentence-transformers is loaded lazily and cached process-wide. The cache is
    important for research evaluation, where hundreds of isolated query
    pipelines are created in one process.
    """

    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        embedding_model: str,
        *,
        bm25_weight: float = 0.45,
        dense_weight: float = 0.55,
        enable_local_models: bool = True,
    ):
        self.embedding_model_name = embedding_model
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.enable_local_models = enable_local_models
        self._embedding_model = None
        self.engine_status = "not-loaded"

    def _chunks(self, documents: list[DocumentRecord]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                chunk_text(
                    document.text,
                    document_id=document.id,
                    document_title=document.title,
                    source_reliability=document.source_reliability,
                    injected=document.injected,
                )
            )
        return chunks

    def _bm25(self, question: str, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=float)
        try:
            from rank_bm25 import BM25Okapi

            corpus = [_tokens(chunk.text) for chunk in chunks]
            model = BM25Okapi(corpus)
            return np.asarray(model.get_scores(_tokens(question)), dtype=float)
        except Exception:
            q = set(_tokens(question))
            scores = []
            for chunk in chunks:
                tokens = _tokens(chunk.text)
                if not tokens or not q:
                    scores.append(0.0)
                    continue
                overlap = sum(1 for token in tokens if token in q)
                scores.append(overlap / math.sqrt(len(tokens)))
            return np.asarray(scores, dtype=float)

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        cached = self._MODEL_CACHE.get(self.embedding_model_name)
        if cached is not None:
            self._embedding_model = cached
            self.engine_status = f"sentence-transformers:{self.embedding_model_name}:cached"
            return cached

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model_name)
        self._MODEL_CACHE[self.embedding_model_name] = model
        self._embedding_model = model
        self.engine_status = f"sentence-transformers:{self.embedding_model_name}"
        return model

    def _dense(self, question: str, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=float)

        if self.enable_local_models:
            try:
                model = self._load_embedding_model()
                texts = [question, *[chunk.text for chunk in chunks]]
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(64, max(8, len(texts))),
                )
                q = np.asarray(vectors[0], dtype=float)
                matrix = np.asarray(vectors[1:], dtype=float)
                return matrix @ q
            except Exception as exc:
                self.engine_status = f"tfidf-fallback:{type(exc).__name__}"

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform([question, *[chunk.text for chunk in chunks]])
            self.engine_status = "tfidf-fallback"
            return cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        except Exception:
            self.engine_status = "lexical-fallback"
            return self._bm25(question, chunks)

    def retrieve(
        self,
        question: str,
        documents: list[DocumentRecord],
        *,
        top_k: int = 8,
        strategy: RetrievalStrategy = "hybrid",
    ) -> list[RetrievedChunk]:
        chunks = self._chunks(documents)
        if not chunks:
            self.engine_status = "empty-index"
            return []

        bm25 = _minmax(self._bm25(question, chunks))

        # Basic RAG is intentionally BM25-only and should not load the dense model.
        if strategy == "bm25":
            combined = bm25
            self.engine_status = "bm25"
            dense = np.zeros_like(bm25)
        else:
            dense = _minmax(self._dense(question, chunks))
            if strategy == "dense":
                combined = dense
            else:
                combined = self.bm25_weight * bm25 + self.dense_weight * dense

        order = np.argsort(combined)[::-1][:top_k]
        return [
            RetrievedChunk(
                id=f"{chunks[i].document_id}:{chunks[i].index}",
                chunk=chunks[i],
                score=float(combined[i]),
                bm25_score=float(bm25[i]),
                dense_score=float(dense[i]),
            )
            for i in order
            if float(combined[i]) > 0
        ]
