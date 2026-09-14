# EvidenceGuard model-backed evaluation

This experiment keeps the frozen evidence weights unchanged, uses local neural retrieval/NLI, and calibrates only the abstention threshold on the pre-declared controlled validation split.

## Model configuration

~~~json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
  "evidence_retrieval_weight": 0.5,
  "evidence_reliability_weight": 0.3,
  "evidence_agreement_weight": 0.2,
  "abstain_threshold": 0.2,
  "source_fallback_abstain_threshold": 0.6,
  "ramdocs_samples": 500,
  "configuration_policy": "Evidence weights transfer unchanged from the frozen fallback run; the abstention threshold is recalibrated only on the pre-declared controlled validation split with neural retrieval/NLI."
}
~~~

## Neural abstention calibration

~~~json
{
  "abstain_threshold": 0.2,
  "validation_score": 0.866667,
  "validation_utility": 0.866667,
  "validation_coverage": 1.0,
  "validation_selective_accuracy": 0.933333,
  "answered": 30,
  "samples": 30,
  "candidate_thresholds": [
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65
  ],
  "validation_case_ids": [
    "apollo_11",
    "c_creator",
    "gold_symbol",
    "japan_capital",
    "python_creator",
    "solar_system"
  ],
  "conflict_ratios": [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75
  ],
  "policy": "Threshold selected only on controlled validation cases with the model-backed retrieval/NLI stack; RAMDocs is not used for tuning."
}
~~~

The calibration uses forced-answer validation runs so threshold candidates are evaluated from one fixed set of neural predictions. RAMDocs labels are never consulted during calibration.

## RAMDocs results

| Mode | Strict acc. | Selective acc. | Coverage | Wrong-answer | Wrong among answered | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| basic_rag | 16.4% | 16.7% | 98.4% | 23.0% | 23.4% | 0.000 |
| hybrid_rag | 18.0% | 18.0% | 100.0% | 31.2% | 31.2% | 0.000 |
| conflict_aware | 14.8% | 14.8% | 100.0% | 19.6% | 19.6% | 0.666 |
| evidenceguard | 13.8% | 14.1% | 98.2% | 19.6% | 20.0% | 0.666 |

## Difference from frozen fallback run

Positive accuracy/F1 deltas are improvements; negative wrong-answer deltas are improvements. The model-backed EvidenceGuard row uses its validation-calibrated abstention threshold, while the fallback row retains its own frozen threshold.

| Mode | Δ strict acc. | Δ selective acc. | Δ wrong-answer | Δ coverage | Δ conflict F1 |
|---|---:|---:|---:|---:|---:|
| basic_rag | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| hybrid_rag | -0.046 | -0.046 | +0.068 | +0.000 | +0.000 |
| conflict_aware | -0.048 | -0.048 | +0.006 | +0.000 | +0.089 |
| evidenceguard | -0.010 | -0.097 | +0.064 | +0.360 | +0.089 |

## EvidenceGuard failure counts

~~~json
{
  "strict_correct": 69,
  "wrong_answer_hits": 98,
  "abstentions": 9,
  "conflict_misses": 11,
  "strict_improvements_over_hybrid": 16,
  "wrong_answer_harm_avoided": 61,
  "regressions_vs_hybrid": 37
}
~~~

## Engine verification

~~~json
{
  "retrieval_engines": {
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2": 1,
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2:cached": 559,
    "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2:score-cache": 1060
  },
  "nli_engines": {
    "transformers:cross-encoder/nli-MiniLM2-L6-H768": 1,
    "transformers:cross-encoder/nli-MiniLM2-L6-H768:cached": 515,
    "transformers:cross-encoder/nli-MiniLM2-L6-H768:relation-cache": 520,
    "not-loaded": 54
  }
}
~~~

## Interpretation constraints

- Evidence-score weights are transferred unchanged from the frozen fallback experiment.
- The neural abstention threshold is selected only on the pre-declared controlled validation cases.
- No RAMDocs label is passed into retrieval, NLI, scoring, generation, abstention calibration, or abstention logic.
- Strict correctness requires every listed gold answer and no listed wrong answer after normalized phrase matching.
- The generator remains the extractive fallback so this run isolates retrieval/NLI and selective-answering changes rather than mixing in an LLM generator.
- The risk-coverage curve is post-hoc evaluation built from the forced-answer conflict-aware mode; it is not used to choose the threshold.
- The workflow fails if dense retrieval or NLI silently drops to a fallback engine.
