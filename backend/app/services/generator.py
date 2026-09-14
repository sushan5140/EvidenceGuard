from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.schemas import EvidenceItem


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    generator: str


class AnswerGenerator:
    def __init__(
        self,
        *,
        api_base: str | None,
        api_key: str | None,
        model: str | None,
    ):
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)

    def _extractive(self, evidence: list[EvidenceItem]) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(
                "I do not have enough retrieved evidence to answer this question.",
                "extractive",
            )

        top = sorted(evidence, key=lambda item: item.evidence_score, reverse=True)[:3]
        statements = [
            f"{item.claim.rstrip('.')} [{index}]."
            for index, item in enumerate(top, start=1)
        ]
        return GeneratedAnswer(" ".join(statements), "extractive")

    async def generate(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        conflict: float,
    ) -> GeneratedAnswer:
        if not self.configured:
            return self._extractive(evidence)

        context = "\n".join(
            f"[{index}] {item.document_title}: {item.claim} "
            f"(evidence_score={item.evidence_score:.2f})"
            for index, item in enumerate(
                sorted(evidence, key=lambda item: item.evidence_score, reverse=True)[:8],
                start=1,
            )
        )
        prompt = f"""You are EvidenceGuard, a conservative evidence-grounded assistant.
Answer ONLY from the evidence below.
Cite claims using [1], [2], etc.
If evidence conflicts, state the disagreement explicitly.
Do not invent facts not present in evidence.

Question: {question}
Detected conflict rate: {conflict:.2f}

Evidence:
{context}

Return a concise answer with citations."""

        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Ground every claim in supplied evidence."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    return GeneratedAnswer(text, "llm")
        except Exception:
            pass

        return self._extractive(evidence)
