from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.schemas import (
    BenchmarkResponse,
    BenchmarkSummaryRow,
    DocumentCreate,
    ResearchMode,
)
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore
from app.research.metrics import (
    binary_metrics,
    coverage,
    expected_calibration_error,
    selective_accuracy,
    strict_answer_correct,
)


DATASET_PATH = Path(__file__).parent / "datasets" / "controlled_conflicts.json"


@dataclass(slots=True)
class RunRecord:
    case_id: str
    mode: ResearchMode
    requested_conflict_ratio: float
    actual_conflict_ratio: float
    correct: bool
    abstained: bool
    confidence: float
    conflict_expected: bool
    conflict_detected: bool
    retrieval_engine: str = ""
    nli_engine: str = ""


def load_controlled_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _documents_for_case(case: dict, ratio: float) -> list[DocumentCreate]:
    ratio = max(0.0, min(0.9, ratio))
    total = 20
    conflict_count = int(round(total * ratio))
    clean_count = total - conflict_count

    documents: list[DocumentCreate] = []
    for index in range(clean_count):
        context = case["clean_contexts"][index % len(case["clean_contexts"])]
        documents.append(
            DocumentCreate(
                title=f'{case["id"]} clean source {index + 1}',
                text=f'{context} {case["clean_fact"]}',
                source_reliability=max(0.72, 0.94 - index * 0.01),
                tags=["benchmark", "clean"],
            )
        )

    for index in range(conflict_count):
        context = case["conflict_contexts"][index % len(case["conflict_contexts"])]
        documents.append(
            DocumentCreate(
                title=f'{case["id"]} conflict source {index + 1}',
                text=f'{context} {case["conflict_fact"]}',
                source_reliability=max(0.52, 0.70 - index * 0.01),
                tags=["benchmark", "conflict"],
            )
        )
    return documents


async def run_controlled_benchmark(
    settings: Settings,
    *,
    modes: list[ResearchMode],
    conflict_ratios: list[float],
    use_nli: bool,
    use_local_models: bool,
    case_ids: set[str] | None = None,
) -> tuple[BenchmarkResponse, list[RunRecord]]:
    all_cases = load_controlled_cases()
    cases = (
        [case for case in all_cases if case["id"] in case_ids]
        if case_ids is not None
        else all_cases
    )
    records: list[RunRecord] = []

    for mode in modes:
        for ratio in conflict_ratios:
            for case in cases:
                with TemporaryDirectory(prefix="evidenceguard-benchmark-") as tmp:
                    benchmark_settings = settings.model_copy(
                        update={
                            "data_path": str(Path(tmp) / "documents.json"),
                            "enable_local_models": use_local_models,
                            "llm_api_base": None,
                            "llm_api_key": None,
                            "llm_model": None,
                        }
                    )
                    store = DocumentStore(benchmark_settings.data_file)
                    docs = _documents_for_case(case, ratio)
                    for document in docs:
                        store.add(document)

                    pipeline = EvidenceGuardPipeline(benchmark_settings, store)
                    response = await pipeline.query(
                        case["question"],
                        top_k=10,
                        use_nli=use_nli,
                        mode=mode,
                    )
                    correct = strict_answer_correct(
                        response.answer,
                        case["answer_keywords"],
                        case.get("wrong_answer_keywords", []),
                        abstained=response.abstained,
                    )
                    conflict_detected = any(
                        edge.relation == "contradicts"
                        for edge in response.graph
                    )
                    actual_ratio = sum(
                        "conflict" in document.tags for document in docs
                    ) / len(docs)

                    records.append(
                        RunRecord(
                            case_id=case["id"],
                            mode=mode,
                            requested_conflict_ratio=ratio,
                            actual_conflict_ratio=actual_ratio,
                            correct=correct,
                            abstained=response.abstained,
                            confidence=response.confidence,
                            conflict_expected=actual_ratio > 0,
                            conflict_detected=conflict_detected,
                            retrieval_engine=response.model_status.get("retrieval", ""),
                            nli_engine=response.model_status.get("nli", ""),
                        )
                    )

    rows: list[BenchmarkSummaryRow] = []
    for mode in modes:
        for ratio in conflict_ratios:
            subset = [
                record
                for record in records
                if record.mode == mode
                and math.isclose(record.requested_conflict_ratio, ratio)
            ]
            correctness = [record.correct for record in subset]
            abstentions = [record.abstained for record in subset]
            confidences = [record.confidence for record in subset]
            conflict_stats = binary_metrics(
                [record.conflict_expected for record in subset],
                [record.conflict_detected for record in subset],
            )

            rows.append(
                BenchmarkSummaryRow(
                    mode=mode,
                    conflict_ratio=ratio,
                    samples=len(subset),
                    accuracy=round(sum(correctness) / len(subset), 4) if subset else 0.0,
                    selective_accuracy=round(
                        selective_accuracy(correctness, abstentions), 4
                    ),
                    coverage=round(coverage(abstentions), 4),
                    abstention_rate=round(
                        sum(abstentions) / len(subset), 4
                    ) if subset else 0.0,
                    mean_confidence=round(
                        sum(confidences) / len(subset), 4
                    ) if subset else 0.0,
                    ece=round(
                        expected_calibration_error(correctness, confidences), 4
                    ),
                    conflict_precision=round(conflict_stats.precision, 4),
                    conflict_recall=round(conflict_stats.recall, 4),
                    conflict_f1=round(conflict_stats.f1, 4),
                )
            )

    return (
        BenchmarkResponse(
            benchmark="controlled-conflicts-v3-strict",
            cases=len(cases),
            rows=rows,
            notes=[
                "Synthetic suite with exact conflict ratios and held-out case support.",
                "Strict correctness requires all required gold keywords and zero known wrong-answer keywords.",
                "Conflict F1 is query-level conflict-presence detection.",
                "Use RAMDocs results separately for external validity.",
            ],
        ),
        records,
    )
