# External benchmark integration

## RAMDocs — primary external benchmark

EvidenceGuard includes an adapter for **RAMDocs**, the benchmark released with
*Retrieval-Augmented Generation with Conflicting Evidence* (COLM 2025).

Official sources:

- GitHub: https://github.com/HanNight/RAMDocs
- Hugging Face: https://huggingface.co/datasets/HanNight/RAMDocs

The official test file is named `RAMDocs_test.jsonl`. Each example contains a
question and retrieved documents categorized as `correct`, `misinfo`, or
`noise`, together with gold and wrong answers.

### Run

Place the official JSONL file somewhere outside the repository, then:

```bash
cd backend
PYTHONPATH=. python scripts/run_ramdocs.py /path/to/RAMDocs_test.jsonl
```

For a quick smoke run:

```bash
PYTHONPATH=. python scripts/run_ramdocs.py /path/to/RAMDocs_test.jsonl --limit 25 --fallback-only --heuristic-nli
```

Outputs are written to `research/results/ramdocs/` by default:

- `summary.json`
- `summary.csv`
- `runs.csv`

### Leakage protection

RAMDocs exposes whether each document is correct, misinformation, or noise.
EvidenceGuard **does not pass these labels into retrieval, scoring, or
generation**. Every RAMDocs document receives the same default reliability
score during inference. The labels are used only after inference for
evaluation.

### Metrics

The adapter reports:

- gold-answer hit rate
- wrong-answer rate
- abstention rate
- mean confidence
- calibration error against gold-answer hits
- query-level conflict precision / recall / F1

These are EvidenceGuard adapter metrics and should not be presented as the
official RAMDocs exact-match score without reproducing the paper's evaluation
protocol.

## Controlled benchmark

The repository also contains `controlled-conflicts-v1`, a six-case synthetic
suite for ablations and regression testing. It should be used to explain
mechanism behavior and conflict-ratio sweeps, while RAMDocs should provide the
main external-validity experiment.

## Recommended final evaluation

Use both:

1. **Controlled benchmark** — interpretable 0/10/25/50/75% conflict sweeps.
2. **RAMDocs** — external benchmark with correct/misinformation/noise evidence.

Report both results separately. Do not merge the synthetic and external scores
into one headline metric.
