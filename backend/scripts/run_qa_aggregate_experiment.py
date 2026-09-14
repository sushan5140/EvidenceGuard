from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import get_settings
from app.research.qa_aggregate import ExtractiveQAAggregator
from app.research.ramdocs import _answer_flags, load_ramdocs
from app.schemas import DocumentCreate
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


@dataclass(slots=True)
class QAExperimentRun:
    index: int
    baseline_answer: str
    qa_answer: str
    baseline_any_gold: bool
    baseline_all_gold: bool
    baseline_wrong: bool
    baseline_strict: bool
    qa_any_gold: bool
    qa_all_gold: bool
    qa_wrong: bool
    qa_strict: bool
    candidate_count: int
    qa_engine: str
    retrieval_engine: str
    nli_engine: str


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metrics(
    runs: list[QAExperimentRun],
    *,
    prefix: str,
) -> dict:
    n = len(runs)
    return {
        "strict_accuracy": round(
            sum(getattr(run, f"{prefix}_strict") for run in runs) / n,
            4,
        ),
        "all_gold_hit_rate": round(
            sum(getattr(run, f"{prefix}_all_gold") for run in runs) / n,
            4,
        ),
        "any_gold_hit_rate": round(
            sum(getattr(run, f"{prefix}_any_gold") for run in runs) / n,
            4,
        ),
        "wrong_answer_rate": round(
            sum(getattr(run, f"{prefix}_wrong") for run in runs) / n,
            4,
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired evaluation of QA aggregation against conflict-aware answers"
    )
    parser.add_argument("--ramdocs", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output", default="../research/results/qa_aggregate_quick")
    parser.add_argument("--qa-model", default="deepset/minilm-uncased-squad2")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
    settings = get_settings().model_copy(
        update={
            "evidence_retrieval_weight": frozen["evidence_retrieval_weight"],
            "evidence_reliability_weight": frozen["evidence_reliability_weight"],
            "evidence_agreement_weight": frozen["evidence_agreement_weight"],
            "default_abstain_threshold": 0.0,
            "enable_local_models": True,
            "llm_api_base": None,
            "llm_api_key": None,
            "llm_model": None,
        }
    )

    cases = load_ramdocs(args.ramdocs, limit=args.limit)
    aggregator = ExtractiveQAAggregator(args.qa_model)
    runs: list[QAExperimentRun] = []

    for index, case in enumerate(cases):
        with TemporaryDirectory(prefix="evidenceguard-qa-quick-") as tmp:
            run_settings = settings.model_copy(
                update={"data_path": str(Path(tmp) / "documents.json")}
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
                use_nli=True,
                mode="conflict_aware",
            )
            baseline = _answer_flags(
                response.answer,
                case,
                abstained=response.abstained,
            )

            aggregated = aggregator.answer(case.question, response.evidence)
            qa_flags = _answer_flags(
                aggregated.text,
                case,
                abstained=False,
            )

            runs.append(
                QAExperimentRun(
                    index=index,
                    baseline_answer=response.answer,
                    qa_answer=aggregated.text,
                    baseline_any_gold=baseline[0],
                    baseline_all_gold=baseline[1],
                    baseline_wrong=baseline[2],
                    baseline_strict=baseline[3],
                    qa_any_gold=qa_flags[0],
                    qa_all_gold=qa_flags[1],
                    qa_wrong=qa_flags[2],
                    qa_strict=qa_flags[3],
                    candidate_count=len(aggregated.candidates),
                    qa_engine=aggregated.engine,
                    retrieval_engine=response.model_status.get("retrieval", ""),
                    nli_engine=response.model_status.get("nli", ""),
                )
            )

    baseline_metrics = metrics(runs, prefix="baseline")
    qa_metrics = metrics(runs, prefix="qa")
    summary = {
        "experiment": "qa_aggregate_paired_quick",
        "qa_model": args.qa_model,
        "samples": len(runs),
        "baseline_conflict_aware": baseline_metrics,
        "qa_aggregate": qa_metrics,
        "delta": {
            key: round(qa_metrics[key] - baseline_metrics[key], 4)
            for key in baseline_metrics
        },
        "mean_candidates": round(
            sum(run.candidate_count for run in runs) / len(runs),
            4,
        ),
        "paired_outcomes": {
            "strict_improvements": sum(
                not run.baseline_strict and run.qa_strict for run in runs
            ),
            "strict_regressions": sum(
                run.baseline_strict and not run.qa_strict for run in runs
            ),
            "wrong_answer_avoided": sum(
                run.baseline_wrong and not run.qa_wrong for run in runs
            ),
            "wrong_answer_added": sum(
                not run.baseline_wrong and run.qa_wrong for run in runs
            ),
        },
    }

    write_csv(output / "runs.csv", [asdict(run) for run in runs])
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(
        "# QA aggregation paired quick check\n\n"
        "First 100 RAMDocs cases; conflict-aware and QA-aggregated answers share "
        "the exact same retrieval/NLI/evidence response before answer assembly.\n\n"
        "~~~json\n"
        + json.dumps(summary, indent=2)
        + "\n~~~\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
