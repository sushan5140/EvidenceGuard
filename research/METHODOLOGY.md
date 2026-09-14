# EvidenceGuard research methodology

## Research question

Can explicit conflict detection, source reliability scoring, and uncertainty-aware abstention improve RAG robustness when retrieved documents contain contradictory or misleading information?

## Proposed system

EvidenceGuard decomposes the RAG pipeline into auditable stages:

1. **Hybrid retrieval** — BM25 lexical relevance and dense semantic similarity.
2. **Claim extraction** — retrieved passages are decomposed into short claims.
3. **Cross-source NLI** — claims from different documents are classified as supporting, contradictory, or neutral.
4. **Evidence graph** — claims become nodes and NLI relations become edges.
5. **Evidence scoring** — retrieval relevance, source reliability, and cross-source agreement are combined.
6. **Uncertainty decision** — confidence is penalized by conflicts. The system may abstain.
7. **Grounded generation** — an optional LLM receives only the ranked evidence and conflict information.

## Primary baselines

- LLM-only answer generation.
- Dense/lexical RAG without explicit conflict handling.
- Hybrid RAG without NLI.
- EvidenceGuard without abstention.
- Full EvidenceGuard.

## Controlled conflict experiment

For each evaluation question:

1. Measure the clean-corpus answer.
2. Inject a synthetic conflicting claim.
3. Re-run retrieval and conflict detection.
4. Increase injected conflict ratio across experimental conditions.
5. Compare answer accuracy, conflict-detection F1, unsupported-answer rate, confidence calibration, and abstention quality.

Suggested conflict conditions:

- 0% synthetic conflicting evidence
- 10%
- 25%
- 50%
- 75%

## Ablation study

Remove one module at a time:

- no dense retrieval
- no BM25 retrieval
- no NLI conflict detection
- no source-reliability signal
- no agreement signal
- no abstention

This shows which component actually contributes to robustness.

## Metrics

### Retrieval
- Recall@K
- MRR / nDCG (when relevance labels are available)

### Conflict detection
- Precision
- Recall
- F1

### Answer quality
- Exact match / token F1 for factoid datasets
- citation support rate
- unsupported claim rate

### Reliability
- accuracy under injected conflict
- abstention precision / recall
- selective accuracy
- expected calibration error (ECE)

## Important evaluation rule

Source-reliability labels must not be tuned using the final test set. Thresholds and score weights should be selected on a validation split and frozen before final evaluation.
