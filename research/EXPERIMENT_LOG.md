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

### ambiguity-preserving answer clusters — running

Branch: `experiment/ambiguity-clusters`. Draft PR #2.

Hypothesis: group strongly supporting claims into answer clusters, rank clusters by accumulated evidence mass, and emit one representative per strongest cluster. This preserves multiple legitimate ambiguous answers without automatically treating every contradiction as false or blindly aggregating every document-level QA span.

Acceptance criteria:

- improve strict and/or all-gold accuracy over canonical `conflict_aware`,
- avoid a material increase in known wrong-answer rate,
- keep the selector label-free at inference time.

## Reporting rules

- RAMDocs labels remain evaluation-only and must never enter retrieval, NLI, evidence scoring, answer aggregation, or abstention decisions.
- Synthetic controlled results and RAMDocs external results remain separate.
- Negative ablations remain documented.
- Test-set diagnostics can motivate new hypotheses, but any tunable parameters must be selected without optimizing directly on RAMDocs test labels.
