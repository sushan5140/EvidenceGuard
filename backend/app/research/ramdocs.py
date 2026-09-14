from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from app.schemas import ResearchMode
from app.research.metrics import binary_metrics, expected_calibration_error

if TYPE_CHECKING:
    from app.config import Settings


@dataclass(slots=True)
class RAMDocsCase:
    question: str
    documents: list[dict]
    gold_answers: list[str]
    wrong_answers: list[str]


@dataclass(slots=True)
class RAMDocsRun:
    index: int
    mode: ResearchMode
    gold_hit: bool
    wrong_answer_hit: bool
    abstained: bool
    confidence: float
    conflict_expected: bool
    conflict_detected: bool


def load_ramdocs(path: str | Path, *, limit: int | None = None) -> list[RAMDocsCase]:
    """Load the official RAMDocs JSONL format.

    Expected fields follow HanNight/RAMDocs:
    question, documents[{text,type,answer}], gold_answers, wrong_answers.
    Document type labels are retained for evaluation only and are never passed
    into the inference pipeline.
    """
    cases: list[RAMDocsCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cases.append(
                RAMDocsCase(
                    question=item["question"],
                    documents=item.get("documents", []),
                    gold_answers=[str(x) for x in item.get("gold_answers", [])],
                    wrong_answers=[str(x) for x in item.get("wrong_answers", [])],
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _contains_any(text: str, candidates: list[str]) -> bool:
    normalized = text.casefold()
    return any(candidate.casefold() in normalized for candidate in candidates if candidate)


async def run_ramdocs(
    settings: Settings,
    path: str | Path,
    *,
    modes: list[ResearchMode],
    limit: int | None = None,
    use_nli: bool = True,
    use_local_models: bool = True,
) -> tuple[list[dict], list[RAMDocsRun]]:
    # Heavy application imports stay inside the runner so the JSONL adapter can
    # be unit-tested without importing transformer/HTTP dependencies.
    from app.schemas import DocumentCreate
    from app.services.pipeline import EvidenceGuardPipeline
    from app.services.store import DocumentStore

    cases = load_ramdocs(path, limit=limit)
    runs: list[RAMDocsRun] = []

    for mode in modes:
        for index, case in enumerate(cases):
            with TemporaryDirectory(prefix="evidenceguard-ramdocs-") as tmp:
                run_settings = settings.model_copy(
                    update={
                        "data_path": str(Path(tmp) / "documents.json"),
                        "enable_local_models": use_local_models,
                        "llm_api_base": None,
                        "llm_api_key": None,
                        "llm_model": None,
                    }
                )
                store = DocumentStore(run_settings.data_file)

                # IMPORTANT: every RAMDocs document receives the same source reliability.
                # Correct/misinfo/noise labels are evaluation metadata only, preventing
                # label leakage into EvidenceGuard inference.
                for doc_index, document in enumerate(case.documents):
                    text = str(document.get("text", "")).strip()
                    if len(text) < 10:
                        continue
                    store.add(
                        DocumentCreate(
                            title=f"RAMDocs document {doc_index + 1}",
                            text=text,
                            source_reliability=0.70,
                            tags=["ramdocs"],
                        )
                    )

                pipeline = EvidenceGuardPipeline(run_settings, store)
                response = await pipeline.query(
                    case.question,
                    top_k=min(12, max(2, len(case.documents))),
                    use_nli=use_nli,
                    mode=mode,
                )

                doc_types = {str(doc.get("type", "")) for doc in case.documents}
                conflict_expected = "correct" in doc_types and "misinfo" in doc_types
                runs.append(
                    RAMDocsRun(
                        index=index,
                        mode=mode,
                        gold_hit=(
                            not response.abstained
                            and _contains_any(response.answer, case.gold_answers)
                        ),
                        wrong_answer_hit=(
                            not response.abstained
                            and _contains_any(response.answer, case.wrong_answers)
                        ),
                        abstained=response.abstained,
                        confidence=response.confidence,
                        conflict_expected=conflict_expected,
                        conflict_detected=any(
                            edge.relation == "contradicts"
                            for edge in response.graph
                        ),
                    )
                )

    summary: list[dict] = []
    for mode in modes:
        subset = [run for run in runs if run.mode == mode]
        if not subset:
            continue
        conflict = binary_metrics(
            [run.conflict_expected for run in subset],
            [run.conflict_detected for run in subset],
        )
        summary.append(
            {
                "mode": mode,
                "samples": len(subset),
                "gold_hit_rate": round(
                    sum(run.gold_hit for run in subset) / len(subset), 4
                ),
                "wrong_answer_rate": round(
                    sum(run.wrong_answer_hit for run in subset) / len(subset), 4
                ),
                "abstention_rate": round(
                    sum(run.abstained for run in subset) / len(subset), 4
                ),
                "mean_confidence": round(
                    sum(run.confidence for run in subset) / len(subset), 4
                ),
                "ece_gold_hit": round(
                    expected_calibration_error(
                        [run.gold_hit for run in subset],
                        [run.confidence for run in subset],
                    ),
                    4,
                ),
                "conflict_precision": round(conflict.precision, 4),
                "conflict_recall": round(conflict.recall, 4),
                "conflict_f1": round(conflict.f1, 4),
            }
        )

    return summary, runs
