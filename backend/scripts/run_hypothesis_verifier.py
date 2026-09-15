from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import get_settings
from app.research.hypothesis_verifier import verify_answer_hypotheses
from app.research.metrics import contains_answer
from app.research.qa_aggregate import ExtractiveQAAggregator
from app.research.ramdocs import _answer_flags, load_ramdocs
from app.schemas import DocumentCreate
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


@dataclass(slots=True)
class SyntheticCase:
    case_id: str
    question: str
    documents: list[dict]
    gold_answers: list[str]
    wrong_answers: list[str]


@dataclass(slots=True)
class GateRun:
    index: int
    baseline_answer: str
    verifier_answer: str
    baseline_any_gold: bool
    baseline_all_gold: bool
    baseline_wrong: bool
    baseline_strict: bool
    verifier_any_gold: bool
    verifier_all_gold: bool
    verifier_wrong: bool
    verifier_strict: bool
    raw_candidate_count: int
    verified_candidate_count: int
    selected_scores: str
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


def flags(answer: str, gold_answers: list[str], wrong_answers: list[str]):
    gold = [contains_answer(answer, item) for item in gold_answers if item]
    wrong = [contains_answer(answer, item) for item in wrong_answers if item]
    any_gold = any(gold) if gold else False
    all_gold = all(gold) if gold else False
    wrong_hit = any(wrong)
    return any_gold, all_gold, wrong_hit, all_gold and not wrong_hit


def load_synthetic(path: Path) -> list[SyntheticCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        SyntheticCase(
            case_id=item["id"],
            question=item["question"],
            documents=item["documents"],
            gold_answers=[str(x) for x in item.get("gold_answers", [])],
            wrong_answers=[str(x) for x in item.get("wrong_answers", [])],
        )
        for item in payload
    ]


async def build_response(settings, case, *, prefix: str):
    with TemporaryDirectory(prefix=prefix) as tmp:
        run_settings = settings.model_copy(update={"data_path": str(Path(tmp) / "documents.json")})
        store = DocumentStore(run_settings.data_file)
        for index, document in enumerate(case.documents):
            text = str(document.get("text", "")).strip()
            if len(text) < 10:
                continue
            store.add(
                DocumentCreate(
                    title=f"Evidence document {index + 1}",
                    text=text,
                    source_reliability=float(document.get("source_reliability", 0.70)),
                    tags=["hypothesis-verifier"],
                )
            )
        pipeline = EvidenceGuardPipeline(run_settings, store)
        return await pipeline.query(
            case.question,
            top_k=min(12, max(2, len(case.documents))),
            use_nli=True,
            mode="conflict_aware",
        )


def metric_summary(rows):
    n = len(rows)
    return {
        "any_gold_hit_rate": round(sum(row[0] for row in rows) / n, 4),
        "all_gold_hit_rate": round(sum(row[1] for row in rows) / n, 4),
        "wrong_answer_rate": round(sum(row[2] for row in rows) / n, 4),
        "strict_accuracy": round(sum(row[3] for row in rows) / n, 4),
    }


def gate_metrics(runs: list[GateRun], *, prefix: str) -> dict:
    n = len(runs)
    return {
        "strict_accuracy": round(sum(getattr(run, f"{prefix}_strict") for run in runs) / n, 4),
        "all_gold_hit_rate": round(sum(getattr(run, f"{prefix}_all_gold") for run in runs) / n, 4),
        "any_gold_hit_rate": round(sum(getattr(run, f"{prefix}_any_gold") for run in runs) / n, 4),
        "wrong_answer_rate": round(sum(getattr(run, f"{prefix}_wrong") for run in runs) / n, 4),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-calibrated answer-hypothesis verifier experiment")
    parser.add_argument("--ramdocs", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--output", default="../research/results/hypothesis_verifier_gate")
    parser.add_argument("--qa-model", default="deepset/minilm-uncased-squad2")
    parser.add_argument("--ramdocs-limit", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
    settings = get_settings().model_copy(update={
        "evidence_retrieval_weight": frozen["evidence_retrieval_weight"],
        "evidence_reliability_weight": frozen["evidence_reliability_weight"],
        "evidence_agreement_weight": frozen["evidence_agreement_weight"],
        "default_abstain_threshold": 0.0,
        "enable_local_models": True,
        "llm_api_base": None,
        "llm_api_key": None,
        "llm_model": None,
    })

    aggregator = ExtractiveQAAggregator(args.qa_model)
    synthetic_records = []
    synthetic_cases = load_synthetic(Path(args.synthetic))
    for case in synthetic_cases:
        response = await build_response(settings, case, prefix="evidenceguard-hv-synth-")
        candidates = aggregator.extract_candidates(case.question, response.evidence)
        synthetic_records.append((case, response, candidates))

    trials = []
    for threshold in THRESHOLDS:
        rows, counts = [], []
        for case, response, candidates in synthetic_records:
            verified = verify_answer_hypotheses(
                candidates,
                evidence=response.evidence,
                graph=response.graph,
                secondary_threshold=threshold,
            )
            rows.append(flags(verified.text, case.gold_answers, case.wrong_answers))
            counts.append(len(verified.candidates))
        metrics = metric_summary(rows)
        utility = metrics["strict_accuracy"] + 0.25 * metrics["all_gold_hit_rate"] - 0.75 * metrics["wrong_answer_rate"]
        trials.append({
            "threshold": threshold,
            **metrics,
            "mean_candidates": round(sum(counts) / len(counts), 4),
            "utility": round(utility, 6),
        })

    best = max(
        trials,
        key=lambda row: (
            row["utility"],
            row["strict_accuracy"],
            -row["wrong_answer_rate"],
            row["all_gold_hit_rate"],
            row["threshold"],
        ),
    )
    selected_threshold = float(best["threshold"])
    calibration = {
        "selected_secondary_threshold": selected_threshold,
        "candidate_thresholds": THRESHOLDS,
        "synthetic_cases": [case.case_id for case in synthetic_cases],
        "selected_trial": best,
        "policy": "Only the secondary acceptance threshold is selected on the hand-authored synthetic ambiguity suite. RAMDocs labels are not used for tuning or inference.",
    }

    ramdocs_cases = load_ramdocs(args.ramdocs, limit=args.ramdocs_limit)
    gate_runs: list[GateRun] = []
    for index, case in enumerate(ramdocs_cases):
        response = await build_response(settings, case, prefix="evidenceguard-hv-gate-")
        baseline = _answer_flags(response.answer, case, abstained=response.abstained)

        candidates = aggregator.extract_candidates(case.question, response.evidence)
        verified = verify_answer_hypotheses(
            candidates,
            evidence=response.evidence,
            graph=response.graph,
            secondary_threshold=selected_threshold,
        )
        verifier = _answer_flags(verified.text, case, abstained=False)

        gate_runs.append(GateRun(
            index=index,
            baseline_answer=response.answer,
            verifier_answer=verified.text,
            baseline_any_gold=baseline[0],
            baseline_all_gold=baseline[1],
            baseline_wrong=baseline[2],
            baseline_strict=baseline[3],
            verifier_any_gold=verifier[0],
            verifier_all_gold=verifier[1],
            verifier_wrong=verifier[2],
            verifier_strict=verifier[3],
            raw_candidate_count=len(candidates),
            verified_candidate_count=len(verified.candidates),
            selected_scores=";".join(f"{item.answer}:{item.verification_score:.4f}" for item in verified.candidates),
            qa_engine=aggregator.status,
            retrieval_engine=response.model_status.get("retrieval", ""),
            nli_engine=response.model_status.get("nli", ""),
        ))

    baseline_metrics = gate_metrics(gate_runs, prefix="baseline")
    verifier_metrics = gate_metrics(gate_runs, prefix="verifier")
    delta = {key: round(verifier_metrics[key] - baseline_metrics[key], 4) for key in baseline_metrics}
    paired = {
        "strict_improvements": sum(not run.baseline_strict and run.verifier_strict for run in gate_runs),
        "strict_regressions": sum(run.baseline_strict and not run.verifier_strict for run in gate_runs),
        "wrong_answer_avoided": sum(run.baseline_wrong and not run.verifier_wrong for run in gate_runs),
        "wrong_answer_added": sum(not run.baseline_wrong and run.verifier_wrong for run in gate_runs),
    }
    gate_pass = (
        verifier_metrics["wrong_answer_rate"] <= baseline_metrics["wrong_answer_rate"] + 0.01
        and verifier_metrics["strict_accuracy"] >= baseline_metrics["strict_accuracy"] - 0.02
        and paired["wrong_answer_added"] <= paired["wrong_answer_avoided"]
    )

    summary = {
        "experiment": "answer_hypothesis_verifier_gate",
        "qa_model": args.qa_model,
        "ramdocs_samples": len(gate_runs),
        "calibration": calibration,
        "baseline_conflict_aware": baseline_metrics,
        "hypothesis_verifier": verifier_metrics,
        "delta": delta,
        "paired_outcomes": paired,
        "mean_raw_candidates": round(sum(run.raw_candidate_count for run in gate_runs) / len(gate_runs), 4),
        "mean_verified_candidates": round(sum(run.verified_candidate_count for run in gate_runs) / len(gate_runs), 4),
        "gate_pass": gate_pass,
        "gate_rule": "wrong-answer delta <= +1pp; strict delta >= -2pp; wrong additions <= wrong avoidances",
    }

    write_csv(output / "synthetic_threshold_trials.csv", trials)
    write_csv(output / "gate_runs.csv", [asdict(run) for run in gate_runs])
    (output / "CALIBRATION.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(
        "# Answer Hypothesis Verifier - paired gate\n\n"
        "The verifier threshold is selected only on a synthetic ambiguity suite, then frozen before the RAMDocs gate.\n\n"
        "~~~json\n" + json.dumps(summary, indent=2) + "\n~~~\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
