from __future__ import annotations

import hashlib
import re

from app.config import Settings
from app.schemas import ConflictEdge, EvidenceItem, QueryResponse, ResearchMode
from app.services.chunking import extract_claims
from app.services.generator import AnswerGenerator
from app.services.nli import NLIEngine
from app.services.retrieval import HybridRetriever, RetrievedChunk
from app.services.scoring import (
    agreement_scores,
    answer_confidence,
    conflict_rate,
    evidence_score,
)
from app.services.store import DocumentStore


_WORD = re.compile(r"[A-Za-z0-9']+")


def _question_overlap(question: str, claim: str) -> float:
    q = {w.lower() for w in _WORD.findall(question) if len(w) > 2}
    c = {w.lower() for w in _WORD.findall(claim) if len(w) > 2}
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def _evidence_id(document_id: str, claim: str) -> str:
    digest = hashlib.sha1(f"{document_id}|{claim}".encode("utf-8")).hexdigest()[:10]
    return f"ev-{digest}"


class EvidenceGuardPipeline:
    def __init__(self, settings: Settings, store: DocumentStore):
        self.settings = settings
        self.store = store
        self.retriever = HybridRetriever(
            settings.embedding_model,
            bm25_weight=settings.retrieval_bm25_weight,
            dense_weight=settings.retrieval_dense_weight,
            enable_local_models=settings.enable_local_models,
        )
        self.nli = NLIEngine(
            settings.nli_model,
            enabled=settings.enable_local_models,
        )
        self.generator = AnswerGenerator(
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    def _claims(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
    ) -> list[dict]:
        candidates: list[dict] = []
        for item in retrieved:
            claims = extract_claims(item.chunk.text)
            ranked = sorted(
                claims,
                key=lambda claim: _question_overlap(question, claim),
                reverse=True,
            )
            for claim in ranked[:2]:
                candidates.append(
                    {
                        "id": _evidence_id(item.chunk.document_id, claim),
                        "document_id": item.chunk.document_id,
                        "document_title": item.chunk.document_title,
                        "claim": claim,
                        "text": item.chunk.text,
                        "retrieval_score": item.score,
                        "source_reliability": item.chunk.source_reliability,
                        "injected": item.chunk.injected,
                    }
                )

        unique: dict[str, dict] = {}
        for item in candidates:
            old = unique.get(item["id"])
            if old is None or item["retrieval_score"] > old["retrieval_score"]:
                unique[item["id"]] = item
        return list(unique.values())[:12]

    def _graph(self, claims: list[dict], *, use_nli: bool) -> list[ConflictEdge]:
        candidates: list[tuple[dict, dict]] = []
        pairs: list[tuple[str, str]] = []

        for i, left in enumerate(claims):
            for right in claims[i + 1 :]:
                if left["document_id"] == right["document_id"]:
                    continue
                overlap = _question_overlap(left["claim"], right["claim"])
                reverse_overlap = _question_overlap(right["claim"], left["claim"])
                if max(overlap, reverse_overlap) < 0.18:
                    continue
                candidates.append((left, right))
                pairs.append((left["claim"], right["claim"]))

        relations = self.nli.compare_many(pairs, use_model=use_nli)
        return [
            ConflictEdge(
                source=left["id"],
                target=right["id"],
                relation=relation.label,
                confidence=round(relation.confidence, 4),
            )
            for (left, right), relation in zip(candidates, relations)
        ]

    async def query(
        self,
        question: str,
        *,
        top_k: int = 8,
        abstain_threshold: float | None = None,
        use_nli: bool = True,
        mode: ResearchMode = "evidenceguard",
    ) -> QueryResponse:
        documents = self.store.list()
        strategy = "bm25" if mode == "basic_rag" else "hybrid"
        retrieved = self.retriever.retrieve(
            question,
            documents,
            top_k=top_k,
            strategy=strategy,
        )
        claims = self._claims(question, retrieved)

        conflict_enabled = mode in {"conflict_aware", "evidenceguard"}
        graph = self._graph(claims, use_nli=use_nli) if conflict_enabled else []
        agreements = agreement_scores([item["id"] for item in claims], graph)

        evidence: list[EvidenceItem] = []
        for item in claims:
            if conflict_enabled:
                agreement = agreements.get(item["id"], 0.5)
                score = evidence_score(
                    item["retrieval_score"],
                    item["source_reliability"],
                    agreement,
                    retrieval_weight=self.settings.evidence_retrieval_weight,
                    reliability_weight=self.settings.evidence_reliability_weight,
                    agreement_weight=self.settings.evidence_agreement_weight,
                )
            else:
                agreement = 0.5
                score = item["retrieval_score"]

            evidence.append(
                EvidenceItem(
                    **item,
                    agreement_score=round(agreement, 4),
                    evidence_score=round(score, 4),
                )
            )

        evidence.sort(key=lambda item: item.evidence_score, reverse=True)
        c_rate = conflict_rate(graph) if conflict_enabled else 0.0

        if conflict_enabled:
            confidence = answer_confidence(
                [item.evidence_score for item in evidence],
                c_rate,
            )
        else:
            confidence = (
                sum(item.evidence_score for item in evidence[:4])
                / max(1, min(4, len(evidence)))
                if evidence
                else 0.0
            )

        threshold = (
            abstain_threshold
            if abstain_threshold is not None
            else self.settings.default_abstain_threshold
        )
        abstention_enabled = mode == "evidenceguard"
        abstained = (confidence < threshold or not evidence) if abstention_enabled else not evidence

        if abstained:
            answer = (
                "EvidenceGuard abstained: the retrieved evidence is too weak or "
                "too conflicted to support a reliable answer."
            )
            generator = "extractive"
        else:
            generated = await self.generator.generate(
                question,
                evidence,
                conflict=c_rate,
            )
            answer = generated.text
            generator = generated.generator

        return QueryResponse(
            question=question,
            answer=answer,
            confidence=round(confidence, 4),
            abstained=abstained,
            conflict_rate=round(c_rate, 4),
            evidence=evidence,
            graph=graph,
            generator=generator,
            mode=mode,
            model_status={
                "retrieval": self.retriever.engine_status,
                "nli": self.nli.status if conflict_enabled else "disabled-by-baseline",
                "generation": (
                    f"llm:{self.settings.llm_model}"
                    if self.generator.configured
                    else "extractive-fallback"
                ),
                "research_mode": mode,
            },
        )
