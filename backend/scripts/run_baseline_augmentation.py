from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import get_settings
from app.research.baseline_augmentation import augment_baseline_answer
from app.research.metrics import contains_answer
from app.research.qa_aggregate import ExtractiveQAAggregator
from app.research.ramdocs import _answer_flags, load_ramdocs
from app.schemas import DocumentCreate
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


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
    augmented_answer: str
    baseline_any_gold: bool
    baseline_all_gold: bool
    baseline_wrong: bool
    baseline_strict: bool
    augmented_any_gold: bool
    augmented_all_gold: bool
    augmented_wrong: bool
    augmented_strict: bool
    raw_candidate_count: int
    addition_count: int
    additions: str
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
                    tags=["baseline-augmentation"],
                )
            )
        pipeline = EvidenceGuardPipeline(run_settings, store)
        return await pipeline.query(
            case.question,
            top_k=min(12, max(2, len(case.documents))),
            use_nli=True,
            mode="conflict_aware",
        )


def metrics(rows):
    n = len(rows)
    return {
        "any_gold_hit_rate": round(sum(row[0] for row in rows) / n, 4),
        "all_gold_hit_rate": round(sum(row[1] for row in rows) / n, 4),
        "wrong_answer_rate": round(sum(row[2] for row in rows) / n, 4),
        "strict_accuracy": round(sum(row[3] for row in rows) / n, 4),
    }


def gate_metrics(runs: list[GateRun], *, prefix: str):
    n = len(runs)
    return {
        "strict_accuracy": round(sum(getattr(run, f"{prefix}_strict") for run in runs) / n, 4),
        "all_gold_hit_rate": round(sum(getattr(run, f"{prefix}_all_gold") for run in runs) / n, 4),
        "any_gold_hit_rate": round(sum(getattr(run, f"{prefix}_any_gold") for run in runs) / n, 4),
        "wrong_answer_rate": round(sum(getattr(run, f"{prefix}_wrong") for run in runs) / n, 4),
    }


async def main():
    parser = argparse.ArgumentParser(description="Baseline-preserving QA augmentation gate")
    parser.add_argument("--ramdocs", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--output", default="../research/results/baseline_augmentation_gate")
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

    synthetic_cases = load_synthetic(Path(args.synthetic))
    records = []
    for case in synthetic_cases:
        response = await build_response(settings, case, prefix="evidenceguard-ba-synth-")
        candidates = aggregator.extract_candidates(case.question, response.evidence)
        records.append((case, response, candidates))

    baseline_rows = [
        flags(response.answer, case.gold_answers, case.wrong_answers)
        for case, response, _ in records
    ]
    synthetic_baseline = metrics(baseline_rows)

    trials = []
    for threshold in THRESHOLDS:
        rows, additions = [], []
        for case, response, candidates in records:
            augmented = augment_baseline_answer(
                response.answer,
                candidates,
                evidence=response.evidence,
                graph=response.graph,
                threshold=threshold,
            )
            rows.append(flags(augmented.text, case.gold_answers, case.wrong_answers))
            additions.append(len(augmented.additions))
        m = metrics(rows)
        utility = m["strict_accuracy"] + 0.35 * m["all_gold_hit_rate"] - 1.0 * m["wrong_answer_rate"]
        trials.append({
            "threshold": threshold,
            **m,
            "mean_additions": round(sum(additions) / len(additions), 4),
            "utility": round(utility, 6),
        })

    best = max(
        trials,
        key=lambda row: (
            row["utility"],
            row["strict_accuracy"],
            -row["wrong_answer_rate"],
            row["all_gold_hit_rate"],
            -row["mean_additions"],
            row["threshold"],
        ),
    )
    threshold = float(best["threshold"])
    calibration = {
        "selected_augmentation_threshold": threshold,
        "candidate_thresholds": THRESHOLDS,
        "synthetic_baseline": synthetic_baseline,
        "selected_trial": best,
        "policy": "Baseline answer is never replaced. Only the augmentation threshold is selected on the synthetic suite; RAMDocs labels are evaluation-only.",
    }

    gate_runs = []
    for index, case in enumerate(load_ramdocs(args.ramdocs, limit=args.ramdocs_limit)):
        response = await build_response(settings, case, prefix="evidenceguard-ba-gate-")
        baseline = _answer_flags(response.answer, case, abstained=response.abstained)
        candidates = aggregator.extract_candidates(case.question, response.evidence)
        augmented = augment_baseline_answer(
            response.answer,
            candidates,
            evidence=response.evidence,
            graph=response.graph,
            threshold=threshold,
        )
        af = _answer_flags(augmented.text, case, abstained=False)
        gate_runs.append(GateRun(
            index=index,
            baseline_answer=response.answer,
            augmented_answer=augmented.text,
            baseline_any_gold=baseline[0],
            baseline_all_gold=baseline[1],
            baseline_wrong=baseline[2],
            baseline_strict=baseline[3],
            augmented_any_gold=af[0],
            augmented_all_gold=af[1],
            augmented_wrong=af[2],
            augmented_strict=af[3],
            raw_candidate_count=len(candidates),
            addition_count=len(augmented.additions),
            additions=";".join(f"{item.answer}:{item.verification_score:.4f}" for item in augmented.additions),
            qa_engine=aggregator.status,
            retrieval_engine=response.model_status.get("retrieval", ""),
            nli_engine=response.model_status.get("nli", ""),
        ))

    base = gate_metrics(gate_runs, prefix="baseline")
    aug = gate_metrics(gate_runs, prefix="augmented")
    delta = {key: round(aug[key] - base[key], 4) for key in base}
    paired = {
        "strict_improvements": sum(not r.baseline_strict and r.augmented_strict for r in gate_runs),
        "strict_regressions": sum(r.baseline_strict and not r.augmented_strict for r in gate_runs),
        "wrong_answer_avoided": sum(r.baseline_wrong and not r.augmented_wrong for r in gate_runs),
        "wrong_answer_added": sum(not r.baseline_wrong and r.augmented_wrong for r in gate_runs),
        "all_gold_improvements": sum(not r.baseline_all_gold and r.augmented_all_gold for r in gate_runs),
    }

    gate_pass = (
        aug["wrong_answer_rate"] <= base["wrong_answer_rate"] + 0.02
        and aug["strict_accuracy"] >= base["strict_accuracy"]
        and aug["all_gold_hit_rate"] >= base["all_gold_hit_rate"]
        and paired["strict_improvements"] >= paired["strict_regressions"]
        and (delta["strict_accuracy"] >= 0.01 or delta["all_gold_hit_rate"] >= 0.03)
    )

    summary = {
        "experiment": "baseline_preserving_augmentation_gate",
        "qa_model": args.qa_model,
        "ramdocs_samples": len(gate_runs),
        "calibration": calibration,
        "baseline_conflict_aware": base,
        "augmented": aug,
        "delta": delta,
        "paired_outcomes": paired,
        "mean_raw_candidates": round(sum(r.raw_candidate_count for r in gate_runs) / len(gate_runs), 4),
        "mean_additions": round(sum(r.addition_count for r in gate_runs) / len(gate_runs), 4),
        "gate_pass": gate_pass,
        "gate_rule": "baseline preserved; wrong delta <= +2pp; strict non-decreasing; all-gold non-decreasing; improvements >= regressions; at least +1pp strict or +3pp all-gold",
    }

    write_csv(output / "synthetic_threshold_trials.csv", trials)
    write_csv(output / "gate_runs.csv", [asdict(r) for r in gate_runs])
    (output / "CALIBRATION.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(
        "# Baseline-preserving augmentation - paired gate\n\n~~~json\n"
        + json.dumps(summary, indent=2)
        + "\n~~~\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
