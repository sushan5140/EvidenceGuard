# EvidenceGuard

**Conflict-aware retrieval-augmented generation for reliable question answering.**

EvidenceGuard retrieves evidence from multiple documents, extracts claims, detects supporting and contradictory evidence, scores evidence quality, and can abstain when the available evidence is too weak or conflicted.

## Research question

> Can explicit conflict detection, evidence reliability scoring, and uncertainty-aware abstention make RAG systems more robust when retrieved documents contain contradictory or misleading information?

## Core capabilities

- Hybrid retrieval: BM25 + dense semantic retrieval
- Claim extraction from retrieved passages
- Natural-language-inference conflict detection
- Evidence reliability and agreement scoring
- Confidence estimation and abstention
- Evidence graph API for visual inspection
- Controlled misinformation injection for experiments
- Baseline-vs-EvidenceGuard evaluation hooks
- Lightweight web dashboard

## Repository layout

```
backend/   FastAPI API and research pipeline
frontend/  Next.js research/demo dashboard
```

## Status

Initial implementation in progress.
