# Run EvidenceGuard and locate reproducible results

## Quick demo on Windows PowerShell

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --port 8000
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, or run `.venv\Scripts\python.exe` directly. The template uses `EVIDENCEGUARD_ENABLE_LOCAL_MODELS=false` for a predictable **CPU-friendly demo**. This mode uses fallback retrieval/NLI and **does not reproduce the neural-paper metrics**.

In a **second terminal** at the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`; FastAPI docs are at `http://localhost:8000/docs`. Click **Load demo evidence**, ask a question, inspect the graph, upload a source and delete it. The model-status panel exposes fallback or neural engine status.

For macOS/Linux, replace the venv activation with `source .venv/bin/activate`, `Copy-Item` with `cp`, and use `export PYTHONPATH=backend` before starting Uvicorn.

## Docker Compose (local browser)

```bash
cp .env.example .env
docker compose up --build
```

The backend's data path is explicitly fixed to `/app/data/documents.json` in Compose so the `evidenceguard_data` volume survives container recreation. The frontend targets `http://localhost:8000` **from your browser**, suitable for the local Compose setup. Remote hosting requires a publicly accessible backend URL and correct CORS origins, which are not supplied by this localhost configuration.

## Test the application

```bash
# From repo root in an environment with backend requirements and pytest installed
python -m pip install pytest
PYTHONPATH=backend python -m pytest backend/tests -q
cd frontend
npm install
npm run lint
npm run build
```

On Windows PowerShell, set `$env:PYTHONPATH = "backend"` instead of the inline Unix assignment. CI runs backend smoke tests and frontend TypeScript/build checks on pull requests and main.

## Frozen research results

The authoritative external 500-case results are **already committed** in `research/results/model_backed/`, alongside model config, validation calibration, engine statuses, run-level CSV and methodology. Do not confuse a fresh lightweight dashboard benchmark with this model-backed experiment.

Review:
- `research/results/model_backed/RESULTS.md`
- `research/results/model_backed/ramdocs_summary.csv`
- `research/results/model_backed/CALIBRATION.json`
- `research/results/model_backed/ENGINE_STATUS.json`
- `research/results/model_backed/RAMDOCS_SOURCE.sha256`
- `research/METHODOLOGY.md`
- `research/EXPERIMENT_LOG.md`

To **re-run** the model-backed experiment, use the workflow in `.github/workflows/model-backed-eval.yml`; it installs the neural dependencies, fetches the official RAMDocs data, and checks engine status. Full model-backed evaluation is significantly heavier than the offline demo. The archived CSV/plots are sufficient to inspect results without retraining or hitting paid APIs.

Optional LLM generation uses `LLM_API_BASE`, `LLM_API_KEY` and `LLM_MODEL` in `.env`. Keep these empty for reproducible extractive demos; never commit API keys.
