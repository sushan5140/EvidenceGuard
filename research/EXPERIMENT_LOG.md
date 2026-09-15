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

## Finding 4 — all-pair NLI improves recall but worsens the system

The `experiment/all-pair-nli` branch removed the lexical-overlap gate and sent every cross-document claim pair to NLI.

Compared with the canonical neural baseline:

| Metric | Canonical | All-pair NLI | Delta |
|---|---:|---:|---:|
| Conflict precision | 51.1% | 49.8% | -1.3 pp |
| Conflict recall | 95.4% | 97.9% | +2.5 pp |
| Conflict F1 | 0.666 | 0.660 | -0.006 |
| Conflict-aware strict accuracy | 14.8% | 14.0% | -0.8 pp |
| Conflict-aware wrong-answer rate | 19.6% | 21.8% | +2.2 pp |
| EvidenceGuard strict accuracy | 13.8% | 12.6% | -1.2 pp |
| EvidenceGuard wrong-answer rate | 19.6% | 21.4% | +1.8 pp |

The full all-pair evaluation also took about 26 minutes versus about 15 minutes for the canonical run.

Decision: reject the all-pair NLI change and close its draft PR. Higher recall alone is not useful when precision, conflict F1, answer accuracy, wrong-answer rate, and compute cost all move in the wrong direction.

## Finding 5 — independent-document QA aggregation increases answer coverage but also misinformation

The `experiment/qa-aggregation` branch applied `deepset/minilm-uncased-squad2` independently to retrieved documents and aggregated extracted answer candidates.

The 500-case exploratory run increased strict accuracy from the canonical 14.8% to 23.8% and all-gold hit rate from 17.8% to 39.8%, but known wrong-answer rate rose from 19.6% to 39.4%. Because that comparison did not share the exact same per-example inference path, a paired 100-case check was run next.

In the paired 100-case check, both systems shared the same retrieved/NLI evidence before answer assembly:

| Metric | Conflict-aware | QA aggregation | Delta |
|---|---:|---:|---:|
| Strict accuracy | 45.0% | 34.0% | -11.0 pp |
| All-gold hit rate | 57.0% | 67.0% | +10.0 pp |
| Wrong-answer rate | 33.0% | 46.0% | +13.0 pp |

Paired outcomes: 10 strict improvements, 21 strict regressions, 11 wrong answers avoided, and 24 new wrong answers introduced.

Decision: reject direct independent-document QA aggregation as a final answer strategy. It recovers more possible answers, but without a verifier it amplifies misinformation faster than it improves strict correctness.

## Active experiments

### all-pair NLI conflict discovery — rejected

Branch: `experiment/all-pair-nli`. Draft PR closed after the full RAMDocs experiment. See Finding 4.

### independent-document QA aggregation — rejected

Branches: `experiment/qa-aggregation` and `experiment/qa-aggregation-quick`. See Finding 5.

### ambiguity-preserving answer clusters — rejected

Branch: `experiment/ambiguity-clusters`. Draft PR #2 closed after the full 500-case model-backed run.

The cluster selector preserved multiple support groups, but compared with canonical `conflict_aware` it reduced strict accuracy from 14.8% to 14.0% and increased known wrong-answer rate from 19.6% to 21.6%.

Decision: reject. Support clustering alone does not distinguish legitimate ambiguity from misinformation reliably enough.

### corroboration-filtered QA verifier — rejected as final policy

Branches: `experiment/qa-verifier-quick` and `experiment/qa-verifier-full`.

The frozen rule retained a single candidate freely, but when multiple QA candidates were proposed it required each surviving candidate to be independently extracted from at least two source documents; otherwise it fell back to the strongest single candidate.

On the paired 500-case run:

| Metric | Conflict-aware | Corroboration verifier | Delta |
|---|---:|---:|---:|
| Strict accuracy | 14.8% | 10.4% | -4.4 pp |
| All-gold hit rate | 17.8% | 11.6% | -6.2 pp |
| Any-gold hit rate | 68.8% | 61.2% | -7.6 pp |
| Wrong-answer rate | 19.6% | 17.8% | -1.8 pp |

Paired outcomes: 18 strict improvements, 40 strict regressions, 48 wrong answers avoided, and 39 new wrong answers introduced.

Decision: reject the hard corroboration rule as the final answer policy. It provides a genuine safety signal but removes too many legitimate low-frequency answers.

### answer-hypothesis verification — rejected at 100-case gate

Planned branch: `experiment/hypothesis-verifier`.

Hypothesis: score each extracted answer candidate independently using candidate-specific evidence support, contradiction pressure, source reliability, retrieval quality, and evidence diversity. Secondary answers should survive when their own evidence is strong rather than being accepted or rejected solely by a global document-count rule.

Protocol:

- tune any candidate-verification thresholds only on a synthetic ambiguity validation suite,
- freeze the policy before RAMDocs evaluation,
- run a paired 100-case RAMDocs gate first,
- proceed to the paired 500-case run only if the 100-case gate does not materially worsen wrong-answer rate or strict correctness.

## Finding 6 — ambiguity support clusters do not solve answer validity

The full model-backed `ambiguity_rag` run reached 14.0% strict accuracy and 21.6% wrong-answer rate versus 14.8% and 19.6% for canonical `conflict_aware`.

Decision: reject the ambiguity-cluster selector. Preserving support groups is necessary for ambiguous questions, but support mass alone is not a sufficient verifier.

## Finding 7 — hard corroboration trades too much recall for safety

The paired 500-case corroboration verifier reduced wrong-answer rate by 1.8 percentage points, from 19.6% to 17.8%, but strict accuracy fell by 4.4 points, from 14.8% to 10.4%.

The rule removed 48 baseline wrong answers but introduced 39 new wrong-answer cases and produced 40 strict regressions versus 18 strict improvements.

Decision: retain corroboration as a feature for a softer candidate-level verifier, not as a hard acceptance rule.

## Finding 8 — candidate-level hypothesis scoring still fails when it replaces the baseline answer

The `experiment/hypothesis-verifier` branch froze its feature weights before RAMDocs and tuned only the secondary acceptance threshold on a 12-case synthetic ambiguity/misinformation suite. The synthetic suite selected a threshold of 0.60.

The paired first-100 RAMDocs gate then produced:

| Metric | Conflict-aware | Hypothesis verifier | Delta |
|---|---:|---:|---:|
| Strict accuracy | 45.0% | 34.0% | -11.0 pp |
| All-gold hit rate | 57.0% | 57.0% | 0.0 pp |
| Any-gold hit rate | 57.0% | 57.0% | 0.0 pp |
| Wrong-answer rate | 33.0% | 40.0% | +7.0 pp |

Paired outcomes: 10 strict improvements, 21 strict regressions, 14 wrong answers avoided, and 21 new wrong answers introduced.

Decision: reject before the 500-case stage. Candidate verification scores did not separate valid and invalid replacement answers strongly enough. The next experiment will preserve the conflict-aware baseline answer and use QA only for conservative secondary-answer augmentation.

## Reporting rules

- RAMDocs labels remain evaluation-only and must never enter retrieval, NLI, evidence scoring, answer aggregation, or abstention decisions.
- Synthetic controlled results and RAMDocs external results remain separate.
- Negative ablations remain documented.
- Test-set diagnostics can motivate new hypotheses, but any tunable parameters must be selected without optimizing directly on RAMDocs test labels.
