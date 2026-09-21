# Submission checklist — September 22, 2026

## Code and reproducibility
- [x] Canonical V2 experiment results remain untouched; experimental branches were **not** merged.
- [x] Dashboard supports loading sample evidence, querying all five modes, graph inspection, adversarial comparison, manual source entry, PDF/text upload and deletion.
- [x] Temporary attack experiments remove only their own injected documents.
- [x] Docker Compose data is stored on the mounted volume.
- [x] CI covers API lifecycle smoke tests, backend unit tests, TypeScript and production frontend build.
- [ ] Verify the merged main branch has green CI at the final commit.
- [ ] Run the sample workflow locally and capture actual screenshots of the running app.

## Submission assets (provided separately in the conversation package)
- [x] `EvidenceGuard_Final_Report.pdf` and editable `.docx` created.
- [x] `EvidenceGuard_Presentation.pptx` created.
- [x] Archived results and research log are in the GitHub repository.
- [ ] Fill institution, department, guide/supervisor, author details, required declarations and signatures in the report **as applicable**.
- [ ] Verify institutional requirements for number of pages, citation style and naming conventions.
- [ ] Record a 3–5 minute demo MP4 (see [DEMO_SCRIPT.md](DEMO_SCRIPT.md)).
- [ ] Zip final report/slides/demo video and include the main-branch repository URL; verify contents after extracting.

## Scientific honesty
- [x] Report current external-benchmark strict accuracy and known wrong-answer rate, not just best-looking ablation numbers.
- [x] State source reliability is input metadata, not an independently measured credibility score.
- [x] State live heuristic/fallback demo is **not** the frozen model-backed 500-case experiment.
- [x] Explain that at the selected RAMDocs operating point, EvidenceGuard abstention did not improve wrong-answer rate versus forced conflict-aware answers.

## Release decision
Release when backend and frontend CI are green, the local demo is verified and the institution-specific metadata are filled. No new research feature is required for the deadline.
