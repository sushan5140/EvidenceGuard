from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from app.schemas import ResearchMode
from app.research.metrics import (
    binary_metrics,
    contains_answer,
    expected_calibration_error,
)

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
    any_gold_hit: bool
    all_gold_hit: bool
    wrong_answer_hit: bool
    strict_correct: bool
    abstained: bool
    confidence: float
    conflict_expected: bool
    conflict_detected: bool
    retrieval_engine: str = ""
    nli_engine: str = ""


def load_ramdocs(path: str | Path, *, limit: int | None = None) -> list[RAMDocsCase]:
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


def _answer_flags(
    answer: str,
    case: RAMDocsCase,
    *,
    abstained: bool,
) -> tuple[bool, bool, bool, bool]:
    if abstained:
        return False, False, False, False

    gold_matches = [contains_answer(answer, item) for item in case.gold_answers if item]
    wrong_matches = [contains_answer(answer, item) for item in case.wrong_answers if item]

    any_gold = any(gold_matches) if gold_matches else False
    all_gold = all(gold_matches) if gold_matches else False
    wrong = any(wrong_matches)
    strict = all_gold and not wrong
    return any_gold, all_gold, wrong, strict


async def run_ramdocs(
    settings: Settings,
    path: str | Path,
    *,
    modes: list[ResearchMode],
    limit: int | None = None,
    use_nli: bool = True,
    use_local_models: bool = True,
) -> tuple[list[dict], list[RAMDocsRun]]:
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

                any_gold, all_gold, wrong, strict = _answer_flags(
                    response.answer,
                    case,
                    abstained=response.abstained,
                )
                doc_types = {str(doc.get("type", "")) for doc in case.documents}
                conflict_expected = "correct" in doc_types and "misinfo" in doc_types

                runs.append(
                    RAMDocsRun(
                        index=index,
                        mode=mode,
                        any_gold_hit=any_gold,
                        all_gold_hit=all_gold,
                        wrong_answer_hit=wrong,
                        strict_correct=strict,
                        abstained=response.abstained,
                        confidence=response.confidence,
                        conflict_expected=conflict_expected,
                        conflict_detected=any(
                            edge.relation == "contradicts"
                            for edge in response.graph
                        ),
                        retrieval_engine=response.model_status.get("retrieval", ""),
                        nli_engine=response.model_status.get("nli", ""),
                    )
                )

    summary: list[dict] = []
    for mode in modes:
        subset = [run for run in runs if run.mode == mode]
        if not subset:
            continue

        answered = [run for run in subset if not run.abstained]
        conflict = binary_metrics(
            [run.conflict_expected for run in subset],
            [run.conflict_detected for run in subset],
        )
        summary.append(
            {
                "mode": mode,
                "samples": len(subset),
                "strict_accuracy": round(
                    sum(run.strict_correct for run in subset) / len(subset), 4
                ),
                "selective_strict_accuracy": round(
                    sum(run.strict_correct for run in answered) / len(answered), 4
                ) if answered else 0.0,
                "coverage": round(len(answered) / len(subset), 4),
                "all_gold_hit_rate": round(
                    sum(run.all_gold_hit for run in subset) / len(subset), 4
                ),
                "any_gold_hit_rate": round(
                    sum(run.any_gold_hit for run in subset) / len(subset), 4
                ),
                "wrong_answer_rate": round(
                    sum(run.wrong_answer_hit for run in subset) / len(subset), 4
                ),
                "wrong_answer_among_answered": round(
                    sum(run.wrong_answer_hit for run in answered) / len(answered), 4
                ) if answered else 0.0,
                "abstention_rate": round(
                    sum(run.abstained for run in subset) / len(subset), 4
                ),
                "mean_confidence": round(
                    sum(run.confidence for run in subset) / len(subset), 4
                ),
                "ece_strict": round(
                    expected_calibration_error(
                        [run.strict_correct for run in subset],
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
