from __future__ import annotations

import re
from dataclasses import dataclass


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ATOMIC_SPLIT = re.compile(
    r"\s+(?:but|however|although|while|whereas|yet)\s+",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Chunk:
    document_id: str
    document_title: str
    text: str
    source_reliability: float
    injected: bool
    index: int


def chunk_text(
    text: str,
    *,
    document_id: str,
    document_title: str,
    source_reliability: float,
    injected: bool,
    chunk_words: int = 120,
    overlap_words: int = 25,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(words), step)):
        part = words[start : start + chunk_words]
        if not part:
            break
        chunks.append(
            Chunk(
                document_id=document_id,
                document_title=document_title,
                text=" ".join(part),
                source_reliability=source_reliability,
                injected=injected,
                index=idx,
            )
        )
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_claims(text: str, max_claims: int = 6) -> list[str]:
    """Extract compact atomic-ish claims without requiring an LLM.

    This deliberately stays deterministic for the baseline. A later research
    iteration can replace it with structured LLM claim decomposition.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    claims: list[str] = []
    for sentence in sentences:
        pieces = [p.strip(" ,;:") for p in _ATOMIC_SPLIT.split(sentence) if p.strip()]
        for piece in pieces:
            # Avoid fragments that are too short to compare meaningfully.
            if len(piece.split()) >= 4:
                claims.append(piece)
            if len(claims) >= max_claims:
                return claims

    if not claims and text.strip():
        claims.append(text.strip())
    return claims[:max_claims]
