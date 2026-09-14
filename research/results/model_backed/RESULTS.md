# EvidenceGuard model-backed evaluation

This experiment keeps the frozen decision weights unchanged and replaces the fallback retrieval/NLI engines with local neural models.

## Model configuration

~~~json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
  "evidence_retrieval_weight": 0.5,
  "evidence_reliability_weight": 0.3,
  "evidence_agreement_weight": 0.2,
  "abstain_threshold": 0.6,
  "ramdocs_samples": 500,
  "configuration_policy": "Transferred unchanged from the frozen fallback validation run."
}
~~~

## RAMDocs results

| Mode | Strict acc. | Selective acc. | Coverage | Wrong-answer | Wrong among answered | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| basic_rag | 16.4% | 16.7% | 98.4% | 23.0% | 23.4% | 0.000 |
| hybrid_rag | 18.0% | 18.0% | 100.0% | 31.2% | 31.2% | 0.000 |
| conflict_aware | 14.8% | 14.8% | 100.0% | 19.6% | 19.6% | 0.666 |
| evidenceguard | 3.8% | 27.9% | 13.6% | 2.6% | 19.1% | 0.666 |

## Difference from frozen fallback run

Positive accuracy/F1 deltas are improvements; negative wrong-answer deltas are improvements.

| Mode | Δ strict acc. | Δ selective acc. | Δ wrong-answer | Δ coverage | Δ conflict F1 |
|---|---:|---:|---:|---:|---:|
| basic_rag | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| hybrid_rag | -0.046 | -0.046 | +0.068 | +0.000 | +0.000 |
| conflict_aware | -0.048 | -0.048 | +0.006 | +0.000 | +0.089 |
| evidenceguard | -0.110 | +0.042 | -0.106 | -0.486 | +0.089 |

## EvidenceGuard failure counts

~~~json
{
  "strict_correct": 19,
  "wrong_answer_hits": 13,
  "abstentions": 432,
  "conflict_misses": 11,
  "strict_improvements_over_hybrid": 2,
  "wrong_answer_harm_avoided": 143,
  "regressions_vs_hybrid": 73
}
~~~

## Engine verification

~~~json
{
  "retrieval_engines": {
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2": 1,
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2:cached": 529,
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2:score-cache": 1060
  },
  "nli_engines": {
    "transformers:cross-encoder/nli-MiniLM2-L6-H768": 1,
    "transformers:cross-encoder/nli-MiniLM2-L6-H768:cached": 493,
    "transformers:cross-encoder/nli-MiniLM2-L6-H768:relation-cache": 512,
    "not-loaded": 54
  }
}
~~~

## Interpretation constraints

- The model-backed run uses the same evidence weights and abstention threshold as the frozen fallback experiment.
- No RAMDocs label is passed into retrieval, NLI, scoring, generation, or abstention logic.
- Strict correctness requires every listed gold answer and no listed wrong answer after normalized phrase matching.
- The generator remains the extractive fallback so this run isolates retrieval/NLI changes rather than mixing in an LLM generator.
- The workflow fails if dense retrieval or NLI silently drops to a fallback engine.
