# EvidenceGuard

**A conflict-aware Retrieval-Augmented Generation (RAG) system for reliable question answering.**

EvidenceGuard is a final-year AI/Computer Science research prototype that asks a harder question than ordinary RAG:

> What should an AI do when its retrieved sources disagree?

Instead of merging retrieved text into one prompt and hoping the LLM resolves it correctly, EvidenceGuard extracts claims, compares evidence across sources, builds a support/contradiction graph, scores evidence quality, estimates uncertainty, and can **abstain** when the evidence is too weak or conflicted.

## Research question

**Can explicit conflict detection, evidence reliability scoring, and uncertainty-aware abstention make RAG systems more robust when retrieved documents contain contradictory or misleading information?**

## Implemented features

- **Hybrid retrieval** — BM25 + semantic retrieval.
- **Laptop-friendly fallback** — TF-IDF/lexical retrieval if local embedding models are unavailable.
- **Claim extraction** — deterministic atomic-ish claim decomposition.
- **Natural Language Inference** — transformer NLI with a deterministic fallback.
- **Evidence Conflict Graph** — support and contradiction relationships across independent sources.
- **Composite evidence scoring** — combines retrieval, source reliability, and agreement.
- **Confidence + abstention** — confidence decreases as meaningful contradictions increase.
- **Grounded answer generation** — optional OpenAI-compatible LLM; extractive fallback requires no API key.
- **PDF/text ingestion**.
- **Controlled misinformation injection**.
- **Baseline vs attacked experiment endpoint**.
- **Research dashboard** showing confidence, evidence scores, conflicts, model status, and adversarial experiments.
- **Docker Compose** and **GitHub Actions CI**.

## Architecture

```text
Question
   |
   v
Hybrid Retrieval
(BM25 + Semantic)
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
   |
   v
Citations + Evidence Inspector
```

## Repository structure

```text
EvidenceGuard/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── demo.py
│   │   └── services/
│   │       ├── chunking.py
│   │       ├── retrieval.py
│   │       ├── nli.py
│   │       ├── scoring.py
│   │       ├── generator.py
│   │       ├── pipeline.py
│   │       └── store.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   └── lib/
├── research/
│   └── METHODOLOGY.md
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/sushan5140/EvidenceGuard.git
cd EvidenceGuard
cp .env.example .env
```

No LLM key is required for the first run.

### 2. Start the backend

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r backend/requirements.txt
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --reload --port 8000
```

### 3. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

### 4. Load the built-in conflict demo

Click **Load demo evidence** in the UI, then ask:

> When did the Eiffel Tower open to the public?

The demo includes two mutually supporting sources and one low-reliability source containing a conflicting date.

## Optional LLM generation

EvidenceGuard accepts any provider exposing an OpenAI-compatible `/chat/completions` endpoint.

In `.env`:

```env
LLM_API_BASE=https://your-provider.example/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model
```

Without these variables, EvidenceGuard returns a deterministic extractive answer, so retrieval/conflict experiments remain reproducible.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime/model status |
| GET | `/api/documents` | List indexed sources |
| POST | `/api/documents` | Add a text source |
| POST | `/api/documents/file` | Ingest PDF/text |
| DELETE | `/api/documents/{id}` | Delete a source |
| POST | `/api/query` | Run EvidenceGuard |
| POST | `/api/inject` | Add synthetic conflicting evidence |
| DELETE | `/api/inject` | Remove injected evidence |
| POST | `/api/experiments/attack` | Compare clean vs attacked corpus |
| POST | `/api/demo/load` | Load demonstration corpus |

FastAPI docs are available at **http://localhost:8000/docs**.

## Current scoring model

The v1 evidence score is:

```text
EvidenceScore =
  0.45 * retrieval relevance
+ 0.25 * source reliability
+ 0.30 * cross-source agreement
```

These are **initial engineering weights, not research conclusions**. For a dissertation/final report, tune them on a validation split and freeze them before evaluating the test set.

Confidence is based on top evidence scores with an explicit penalty for contradiction density. The abstention threshold is configurable.

## Final-year evaluation plan

Compare:

1. basic RAG
2. hybrid RAG
3. hybrid RAG + conflict detection
4. EvidenceGuard without abstention
5. full EvidenceGuard

Then inject contradictory evidence at increasing ratios and evaluate:

- answer accuracy
- conflict-detection precision/recall/F1
- unsupported claim rate
- citation support
- calibration
- abstention precision/recall
- selective accuracy

See [research/METHODOLOGY.md](research/METHODOLOGY.md) for the complete experiment design.

## Important limitations

This is a research prototype, not a production truth engine.

- Source reliability currently comes from metadata supplied at ingestion.
- The fallback claim extractor is deterministic rather than a trained claim-decomposition model.
- NLI can misclassify subtle temporal, numerical, or contextual disagreements.
- A confidence score is an internal system score; it must not be interpreted as a universal probability that an answer is true.
- Real evaluation requires a labeled benchmark and frozen validation/test protocol.

## License

Add the license required by your university/team before public release or final submission.
