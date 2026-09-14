# EvidenceGuard

**Conflict-aware Retrieval-Augmented Generation (RAG) for reliable question answering.**

EvidenceGuard is a final-year AI/Computer Science research prototype built around one question:

> What should a RAG system do when its retrieved sources disagree?

Instead of merging retrieved text into one prompt, EvidenceGuard retrieves evidence, extracts claims, detects support/contradiction relationships, scores evidence quality, estimates uncertainty, and can **abstain** when the evidence is too weak or conflicted.

## V2 status

V2 adds a reproducible research layer on top of the working application:

- five switchable system modes / ablations
- 0/10/25/50/75% controlled conflict sweeps
- accuracy, selective accuracy, coverage, abstention rate, ECE, and conflict F1
- CSV + JSON + Markdown experiment exports
- in-app benchmark matrix
- RAMDocs external-benchmark adapter
- tests + GitHub Actions CI

## Research modes

| Mode | Retrieval | Conflict reasoning | Reliability/agreement scoring | Abstention |
|---|---|---|---|---|
| `basic_rag` | BM25 | No | No | No |
| `hybrid_rag` | BM25 + semantic | No | No | No |
| `conflict_aware` | BM25 + semantic | Yes | Yes | No |
| `consensus_rag` | BM25 + semantic | Yes | Yes + contradiction-pruned answer selection | No |
| `evidenceguard` | BM25 + semantic | Yes | Yes + contradiction-pruned answer selection | Yes |

These modes let the final report measure what each added mechanism contributes.

## Core pipeline

```text
Question
   |
   v
Retrieval (BM25 / Hybrid)
   |
   v
Claim Extraction
   |
   v
Cross-source NLI
   |
   v
Evidence Conflict Graph
(support / contradict / neutral)
   |
   v
Evidence Scoring
(relevance + reliability + agreement)
   |
   v
Confidence Estimation
   |------------------|
   v                  v
Answer             Abstain
```

## Implemented features

- hybrid BM25 + semantic retrieval
- TF-IDF/lexical laptop fallback
- deterministic claim extraction
- transformer NLI + heuristic fallback
- support/contradiction evidence graph
- source-reliability and agreement scoring
- contradiction-pruned consensus answer selection
- confidence estimation + abstention
- optional OpenAI-compatible grounded generation
- PDF/text ingestion
- controlled misinformation injection
- adversarial before/after experiment
- research-mode selector in the dashboard
- controlled benchmark matrix in the dashboard
- benchmark CLI with report-ready outputs
- RAMDocs adapter with label-leakage protection
- Docker Compose
- GitHub Actions CI

## Quick start

```bash
git clone https://github.com/sushan5140/EvidenceGuard.git
cd EvidenceGuard
cp .env.example .env

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r backend/requirements.txt
```

Start the backend:

```bash
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --reload --port 8000
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. FastAPI docs are at **http://localhost:8000/docs**.

## Controlled benchmark

The built-in suite contains six factual QA cases and generates twenty evidence documents per case at each requested conflict level.

From the UI, click **Run 5 × 5 benchmark**.

For report artifacts:

```bash
cd backend
PYTHONPATH=. python scripts/run_benchmark.py
```

Outputs:

```text
research/results/
├── report.json
├── summary.csv
├── runs.csv
└── REPORT.md
```

Enable local embedding/NLI models for the final run:

```bash
PYTHONPATH=. python scripts/run_benchmark.py --local-models --nli
```

## External benchmark: RAMDocs

EvidenceGuard includes an adapter for the public **RAMDocs** benchmark from *Retrieval-Augmented Generation with Conflicting Evidence* (COLM 2025).

Official dataset sources:

- https://github.com/HanNight/RAMDocs
- https://huggingface.co/datasets/HanNight/RAMDocs

After obtaining `RAMDocs_test.jsonl`:

```bash
cd backend
PYTHONPATH=. python scripts/run_ramdocs.py /path/to/RAMDocs_test.jsonl
```

Quick smoke test:

```bash
PYTHONPATH=. python scripts/run_ramdocs.py /path/to/RAMDocs_test.jsonl --limit 25 --fallback-only --heuristic-nli
```

RAMDocs `correct` / `misinfo` / `noise` labels are used **only for evaluation**. They are not passed into retrieval or scoring, which avoids label leakage.

See [research/EXTERNAL_BENCHMARKS.md](research/EXTERNAL_BENCHMARKS.md).

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime/model status |
| GET | `/api/documents` | List indexed sources |
| POST | `/api/documents` | Add text evidence |
| POST | `/api/documents/file` | Ingest PDF/text |
| DELETE | `/api/documents/{id}` | Delete a source |
| POST | `/api/query` | Run any research mode |
| POST | `/api/inject` | Add synthetic conflicting evidence |
| DELETE | `/api/inject` | Remove injected evidence |
| POST | `/api/experiments/attack` | Clean-vs-attacked comparison |
| POST | `/api/benchmarks/controlled` | Run controlled benchmark matrix |
| POST | `/api/demo/load` | Load demonstration corpus |

## V2 metrics

The controlled benchmark reports:

- answer accuracy
- selective accuracy (accuracy among answered questions)
- coverage
- abstention rate
- mean confidence
- Expected Calibration Error (ECE)
- query-level conflict precision / recall / F1

RAMDocs additionally reports gold-answer hit rate and wrong-answer rate.

## Scoring model

The current engineering score is:

```text
EvidenceScore =
  0.50 * retrieval relevance
+ 0.30 * source reliability
+ 0.20 * cross-source agreement
```

These are the currently frozen engineering weights selected by the repository's validation protocol, **not universal research conclusions**. Any future retuning must use validation data only and remain frozen for held-out evaluation.

## Research documentation

- [Methodology](research/METHODOLOGY.md)
- [External benchmarks](research/EXTERNAL_BENCHMARKS.md)

## Important limitations

This is a research prototype, not a universal truth engine.

- source reliability is currently ingestion metadata
- deterministic claim extraction can miss complex propositions
- NLI may misclassify subtle temporal/numeric/contextual conflicts
- confidence is an internal system score, not a universal probability of truth
- controlled benchmark results are synthetic and must be reported separately from RAMDocs
- external claims should be based on frozen held-out evaluation, not demo examples

## License

Add the license required by your university/team before final submission.
