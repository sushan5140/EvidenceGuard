# EvidenceGuard research methodology

## Research question

Can explicit conflict detection, source reliability scoring, and uncertainty-aware abstention improve RAG robustness when retrieved documents contain contradictory or misleading information?

## Proposed system

EvidenceGuard decomposes RAG into auditable stages:

1. **Retrieval** — BM25 lexical retrieval or BM25 + dense semantic retrieval.
2. **Claim extraction** — retrieved passages are decomposed into short claims.
3. **Cross-source NLI** — claims are classified as supporting, contradictory, or neutral.
4. **Evidence graph** — claims become nodes and NLI relations become edges.
5. **Evidence scoring** — retrieval relevance, source reliability, and agreement are combined.
6. **Consensus selection** — strongly contradictory claims are prevented from coexisting in the final answer evidence set.
7. **Uncertainty decision** — confidence is penalized by conflict density.
8. **Abstention** — the full system may decline to answer below a frozen threshold.
9. **Grounded generation** — optional LLM generation receives only the selected evidence set.

## Implemented ablations

The repository exposes five modes using the same corpus/question:

1. **basic_rag** — BM25 retrieval only; no explicit conflict handling.
2. **hybrid_rag** — BM25 + semantic retrieval; no explicit conflict handling.
3. **conflict_aware** — hybrid retrieval + NLI + evidence scoring, forced to answer from raw top-ranked evidence.
4. **consensus_rag** — conflict-aware scoring + contradiction-pruned consensus selection, forced to answer.
5. **evidenceguard** — consensus-selected evidence + validation-calibrated abstention.

This isolates the contribution of retrieval, conflict reasoning/scoring, consensus answer assembly, and abstention.

## Controlled conflict experiment

The built-in `controlled-conflicts-v1` suite evaluates six factual questions. For every case and research mode, the corpus is regenerated at:

- 0% conflicting evidence
- 10%
- 25%
- 50%
- 75%

Each condition contains ten documents so the requested ratio is exact.

The runner records per-example outcomes and aggregates:

- accuracy
- selective accuracy
- coverage
- abstention rate
- mean confidence
- Expected Calibration Error (ECE)
- query-level conflict precision / recall / F1

### Reproducibility

Dashboard runs disable local transformer models and use deterministic fallbacks for fast repeatability. Final experiments should additionally run with local sentence-transformer retrieval and transformer NLI enabled.

```bash
cd backend
PYTHONPATH=. python scripts/run_benchmark.py --local-models --nli
```

## External validation: RAMDocs

The primary external benchmark is **RAMDocs**, released with *Retrieval-Augmented Generation with Conflicting Evidence* (COLM 2025).

The official format contains:

- question
- retrieved documents
- document type: correct / misinfo / noise
- document-level answer
- gold answers
- wrong answers

EvidenceGuard intentionally does **not** expose RAMDocs document-type labels to its inference pipeline. All documents receive the same default source-reliability value during inference. Labels are used only after prediction for evaluation.

```bash
cd backend
PYTHONPATH=. python scripts/run_ramdocs.py /path/to/RAMDocs_test.jsonl
```

Report the controlled benchmark and RAMDocs separately.

## Metrics

### Retrieval

When relevance labels are available:

- Recall@K
- MRR
- nDCG

### Conflict detection

- Precision
- Recall
- F1

The built-in benchmark currently reports **query-level conflict-presence F1**. Do not describe it as edge-level NLI F1.

### Answer quality

- controlled benchmark keyword accuracy
- external gold-answer hit rate
- wrong-answer rate
- citation support rate (future extension)
- unsupported-claim rate (future extension)

### Reliability

- accuracy under increasing conflict
- selective accuracy
- coverage
- abstention rate
- expected calibration error
- confidence degradation under attack

## Validation protocol

Thresholds and score weights must be selected on a validation split and frozen before final testing.

Do not:

- tune on RAMDocs test outcomes
- use RAMDocs correct/misinfo/noise labels as inference features
- combine synthetic and external scores into one headline metric
- claim statistical significance without an appropriate test and enough samples

## Recommended final tables

### Table A — controlled robustness

Rows: five research modes.

Columns: accuracy at 0/10/25/50/75% conflict, selective accuracy, ECE, conflict F1.

### Table B — RAMDocs external validation

Rows: five research modes.

Columns: gold-hit rate, wrong-answer rate, abstention rate, ECE, conflict F1.

### Figure A

Conflict ratio (x-axis) vs answer accuracy (y-axis), one line per mode.

### Figure B

Conflict ratio (x-axis) vs mean confidence / coverage.

### Figure C

Accuracy–coverage or risk–coverage curve for EvidenceGuard.
