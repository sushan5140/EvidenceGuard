# Frozen model-backed evaluation snapshot

**Source of truth:** [model-backed results](../research/results/model_backed/RESULTS.md), [RAMDocs summary CSV](../research/results/model_backed/ramdocs_summary.csv) and [experiment log](../research/EXPERIMENT_LOG.md). These are **historical, frozen results**, not measurements of the CPU-friendly live demo.

Dataset: **500 RAMDocs test examples**. Retrieval: `sentence-transformers/all-MiniLM-L6-v2`. NLI: `cross-encoder/nli-MiniLM2-L6-H768`. Evidence weights: relevance **0.50**, reliability **0.30**, agreement **0.20**. Neural abstention threshold **0.20** selected on a separate controlled validation split. The generator was the extractive fallback. RAMDocs gold/wrong and document-type labels were evaluation-only.

| Mode | Strict accuracy | Any-gold hit | All-gold hit | Wrong-answer rate | Coverage | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| Basic RAG | 16.4% | 78.2% | 22.2% | 23.0% | 98.4% | N/A |
| Hybrid RAG | 18.0% | 79.2% | 26.4% | 31.2% | 100.0% | N/A |
| Conflict-aware | 14.8% | 68.8% | 17.8% | 19.6% | 100.0% | 0.666 |
| Consensus RAG (side ablation) | 11.4% | 63.2% | 13.8% | 19.2% | 100.0% | 0.666 |
| EvidenceGuard (abstention) | 13.8% | 67.0% | 16.8% | 19.6% | 98.2% | 0.666 |

**Evaluation definition:** Strict accuracy requires all listed gold answers and no listed known wrong answer. A hit rate can count a response containing *both* valid and wrong answers, so report strict accuracy and wrong-answer rate together.

**What the evidence supports:** The conflict graph exposes disagreement, and the system supports selective answering. Conflict-aware scoring reduced known wrong-answer rate compared with the hybrid mode (**31.2% → 19.6%**) but also lowered strict accuracy (**18.0% → 14.8%**). At the calibrated EvidenceGuard threshold, wrong-answer rate was unchanged against the forced conflict-aware parent (**19.6%**) while coverage decreased (**100% → 98.2%**). The system is a prototype, not a verified truth engine.

**Negative experiments:** All-pair NLI, support clustering, independent-document QA aggregation, hard corroboration, hypothesis verification, baseline augmentation and external entity-aware augmentation were evaluated and not merged. The final entity-aware 100-case gate showed no change from its paired baseline (strict **45%**, all-gold **57%**, wrong **33%** in both arms). See the full log for the exact protocol and caveats.

**No leakage claim:** RAMDocs answer labels must never be used to set answer-selection weights, confidence thresholds or tuned candidate filters. Small test-gate results are exploratory diagnostics and are not comparable with the 500-case frozen model-backed test as if they were the same sample.
