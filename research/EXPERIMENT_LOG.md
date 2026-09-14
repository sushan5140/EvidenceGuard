# EvidenceGuard experiment log

This file records research decisions after reproducible runs. Negative results are kept deliberately so later changes are not justified by selective reporting.

## Canonical neural baseline

Canonical model-backed result commit: `97bb69e2`.

RAMDocs test set: 500 examples. Neural retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; NLI uses `cross-encoder/nli-MiniLM2-L6-H768`. Evidence weights are frozen from the declared validation protocol.

### RAMDocs snapshot

| Mode | Strict accuracy | Any-gold hit | All-gold hit | Wrong-answer rate | Coverage | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| basic_rag | 16.4% | 78.2% | 22.2% | 23.0% | 98.4% | 0.000 |
| hybrid_rag | 18.0% | 79.2% | 26.4% | 31.2% | 100.0% | 0.000 |
| conflict_aware | 14.8% | 68.8% | 17.8% | 19.6% | 100.0% | 0.666 |
| consensus_rag | 11.4% | 63.2% | 13.8% | 19.2% | 100.0% | 0.666 |
| evidenceguard | 13.8% | 67.0% | 16.8% | 19.6% | 98.2% | 0.666 |

## Finding 1 — controlled threshold calibration does not transfer cleanly

The neural validation split selected an abstention threshold of 0.20 with 100% validation coverage and 93.3% selective accuracy.

On RAMDocs, that threshold yields 98.2% coverage and leaves the known wrong-answer rate unchanged at 19.6% relative to forced-answer conflict-aware mode. Strict accuracy falls from 14.8% to 13.8%.

Decision: do not claim abstention improves RAMDocs reliability at the currently validated operating point. Keep the risk-coverage curve as a post-hoc diagnostic and treat cross-domain calibration as an open limitation.

## Finding 2 — contradiction-pruned consensus is a negative result

`consensus_rag` lowers wrong-answer rate only from 19.6% to 19.2% while reducing strict accuracy from 14.8% to 11.4%.

RAMDocs contains legitimate ambiguity with multiple gold answers. Removing one side of a contradiction can therefore remove valid alternative answers as well as misinformation.

Decision: keep `consensus_rag` as a side ablation, but do not include consensus pruning inside the final EvidenceGuard mode.

## Finding 3 — incomplete multi-answer coverage is the dominant answer-stage failure

For canonical `conflict_aware` RAMDocs runs:

- 344 / 500 contain at least one gold answer.
- 89 / 500 contain every listed gold answer.
- 255 / 500 contain at least one gold answer but miss one or more additional gold answers.
- 98 / 500 contain a known wrong answer.
- 74 / 500 satisfy the strict all-gold-and-no-wrong criterion.

This indicates that incomplete multi-entity answer assembly is a larger bottleneck than simply detecting misinformation.

## Active experiments

### all-pair NLI conflict discovery

Branch: `experiment/all-pair-nli`.

Hypothesis: the lexical-overlap prefilter may hide semantically phrased contradictions. The experiment sends every cross-document claim pair to NLI (maximum 66 pairs for 12 retained claims).

Acceptance criteria:

- material conflict-recall/F1 improvement,
- no disproportionate precision collapse,
- answer metrics must improve enough to justify the added inference cost.

### independent-document QA aggregation

Branch: `experiment/qa-aggregation`.

Hypothesis: independently extracting answer spans from multiple retrieved documents and aggregating repeated candidates can recover additional legitimate disambiguated answers that are currently present in evidence but omitted from the final extractive response.

Acceptance criteria:

- improve strict and/or all-gold accuracy over canonical `conflict_aware`,
- avoid a material increase in known wrong-answer rate,
- keep inference cost reasonable enough for a final-year prototype.

## Reporting rules

- RAMDocs labels remain evaluation-only and must never enter retrieval, NLI, evidence scoring, answer aggregation, or abstention decisions.
- Synthetic controlled results and RAMDocs external results remain separate.
- Negative ablations remain documented.
- Test-set diagnostics can motivate new hypotheses, but any tunable parameters must be selected without optimizing directly on RAMDocs test labels.
