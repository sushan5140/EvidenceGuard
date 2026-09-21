# EvidenceGuard — 3–5 minute demonstration

Use this script with **the canonical V2 application** on the submission branch/main after CI passes. Do not demonstrate rejected experimental branches as final functionality.

## Before recording

1. Run the FastAPI server and the Next.js dashboard as documented in [RUN_AND_REPRODUCE.md](RUN_AND_REPRODUCE.md).
2. Open the dashboard and check that the API indicator says **ok**. Turn off optional hosted LLM generation so the demo uses deterministic extractive answers.
3. Click **Load demo evidence**. The corpus includes Eiffel Tower notes discussing **1889**, plus an explicitly *unverified* claim about **1905**. A loaded demo is idempotent.
4. If you have tested other documents, delete them using **Remove source** before recording, or use a fresh local data file.
5. Keep the research-mode selector visible when taking screenshots.

## Spoken walkthrough

**0:00–0:35 — Problem.** “Ordinary retrieval-augmented question answering retrieves passages and produces an answer. But retrieved sources can conflict or refer to different entities. EvidenceGuard makes disagreement visible rather than treating the highest-ranked claim as certain.”

**0:35–1:15 — Corpus.** Show the three loaded demo documents. Point out that each source carries an editable reliability *metadata* value; it is not an independently audited trust score. Optionally upload a PDF/TXT/Markdown source to demonstrate ingestion.

**1:15–2:15 — Comparison.** Ask: **When did the Eiffel Tower open to the public?** Run `basic_rag`, then `conflict_aware`, then `evidenceguard`. Explain BM25 versus hybrid retrieval, claim extraction, cross-document support/contradiction edges, score components and the configured abstention mechanism. The actual answer and abstention may depend on selected mode, local models, and corpus; narrate the output shown rather than promising a specific classification.

**2:15–3:00 — Inspectability.** Scroll through the ranked evidence and graph. Green lines indicate predicted support; dashed red lines indicate predicted contradiction. Show the model-status panel and say whether this run used neural models or heuristic fallbacks.

**3:00–3:40 — Adversarial lab.** Temporarily inject the sample 1905 statement, compare clean and attacked outputs, and explain that the injected evidence is removed after the experiment. Existing user-injected documents are preserved.

**3:40–4:30 — Evaluation.** Open [RESULTS_SNAPSHOT.md](RESULTS_SNAPSHOT.md) or the supplied slides. Clearly separate the live fallback demo from the archived **500-case model-backed RAMDocs evaluation**. Do not claim the current calibrated abstention point reduced known wrong-answer rate on RAMDocs: it did **not**.

**4:30–5:00 — Conclusion.** “EvidenceGuard demonstrates conflict detection, evidence inspection and selective answering. Its remaining challenge is distinguishing misinformation from legitimate ambiguity when multiple answers may all be valid.”

## Recording checklist

- API status visible and no uncaught browser errors.
- Show at least one indexed source, successful answer, ranked evidence and graph.
- Use before/after comparison; do not leave temporary experiments as permanent corpus data.
- Show accurate benchmark figures and the explicit limitations.
- Export an MP4 locally; no special video service is required.
