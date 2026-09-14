from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.research.benchmark import run_controlled_benchmark
from app.research.ramdocs import load_ramdocs, run_ramdocs
from app.research.tuning import (
    MODEL_THRESHOLD_CANDIDATES,
    calibrate_abstention_threshold_from_records,
)


MODES = ["basic_rag", "hybrid_rag", "conflict_aware", "consensus_rag", "ambiguity_rag", "evidenceguard"]
CONFLICT_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_model_backed(controlled_runs: list, ramdocs_runs: list) -> dict:
    all_runs = [*controlled_runs, *ramdocs_runs]

    retrieval_statuses = Counter(
        run.retrieval_engine
        for run in all_runs
        if run.mode != "basic_rag"
    )
    nli_statuses = Counter(
        run.nli_engine
        for run in all_runs
        if run.mode in {"conflict_aware", "consensus_rag", "ambiguity_rag", "evidenceguard"}
    )

    bad_retrieval = [
        status
        for status in retrieval_statuses
        if "fallback" in status or not status.startswith("sentence-transformers:")
    ]
    bad_nli = [
        status
        for status in nli_statuses
        if "fallback" in status
    ]
    observed_transformer_nli = any(
        status.startswith("transformers:")
        for status in nli_statuses
    )

    if bad_retrieval:
        raise RuntimeError(
            "Model-backed run used non-model retrieval engines: "
            + ", ".join(sorted(bad_retrieval))
        )
    if bad_nli:
        raise RuntimeError(
            "Model-backed run used heuristic NLI fallback: "
            + ", ".join(sorted(bad_nli))
        )
    if not observed_transformer_nli:
        raise RuntimeError("Transformer NLI was never observed in the model-backed run.")

    return {
        "retrieval_engines": dict(retrieval_statuses),
        "nli_engines": dict(nli_statuses),
    }


def compare_with_fallback(
    model_rows: list[dict],
    fallback_rows: list[dict],
) -> list[dict]:
    fallback = {row["mode"]: row for row in fallback_rows}
    comparison: list[dict] = []

    for row in model_rows:
        base = fallback.get(row["mode"])
        if base is None:
            continue
        comparison.append(
            {
                "mode": row["mode"],
                "model_strict_accuracy": row["strict_accuracy"],
                "fallback_strict_accuracy": float(base["strict_accuracy"]),
                "delta_strict_accuracy": round(
                    row["strict_accuracy"] - float(base["strict_accuracy"]), 4
                ),
                "model_selective_accuracy": row["selective_strict_accuracy"],
                "fallback_selective_accuracy": float(base["selective_strict_accuracy"]),
                "delta_selective_accuracy": round(
                    row["selective_strict_accuracy"]
                    - float(base["selective_strict_accuracy"]),
                    4,
                ),
                "model_wrong_answer_rate": row["wrong_answer_rate"],
                "fallback_wrong_answer_rate": float(base["wrong_answer_rate"]),
                "delta_wrong_answer_rate": round(
                    row["wrong_answer_rate"] - float(base["wrong_answer_rate"]), 4
                ),
                "model_coverage": row["coverage"],
                "fallback_coverage": float(base["coverage"]),
                "delta_coverage": round(
                    row["coverage"] - float(base["coverage"]), 4
                ),
                "model_conflict_f1": row["conflict_f1"],
                "fallback_conflict_f1": float(base["conflict_f1"]),
                "delta_conflict_f1": round(
                    row["conflict_f1"] - float(base["conflict_f1"]), 4
                ),
            }
        )
    return comparison


def ramdocs_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "| Mode | Strict acc. | Selective acc. | Coverage | Wrong-answer | Wrong among answered | Conflict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {pct(row['strict_accuracy'])} | "
            f"{pct(row['selective_strict_accuracy'])} | {pct(row['coverage'])} | "
            f"{pct(row['wrong_answer_rate'])} | "
            f"{pct(row['wrong_answer_among_answered'])} | "
            f"{row['conflict_f1']:.3f} |"
        )
    return lines


def comparison_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "| Mode | Δ strict acc. | Δ selective acc. | Δ wrong-answer | Δ coverage | Δ conflict F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['delta_strict_accuracy']:+.3f} | "
            f"{row['delta_selective_accuracy']:+.3f} | "
            f"{row['delta_wrong_answer_rate']:+.3f} | "
            f"{row['delta_coverage']:+.3f} | "
            f"{row['delta_conflict_f1']:+.3f} |"
        )
    return lines



def neural_ablation_markdown(rows: list[dict]) -> list[str]:
    by_mode = {row["mode"]: row for row in rows}
    conflict = by_mode.get("conflict_aware")
    consensus = by_mode.get("consensus_rag")
    ambiguity = by_mode.get("ambiguity_rag")
    full = by_mode.get("evidenceguard")
    lines = [
        "| Transition | Strict acc. | Wrong-answer | Coverage |",
        "|---|---:|---:|---:|",
    ]
    if conflict and consensus:
        lines.append(
            f"| conflict_aware → consensus_rag | "
            f"{consensus['strict_accuracy'] - conflict['strict_accuracy']:+.3f} | "
            f"{consensus['wrong_answer_rate'] - conflict['wrong_answer_rate']:+.3f} | "
            f"{consensus['coverage'] - conflict['coverage']:+.3f} |"
        )
    if conflict and ambiguity:
        lines.append(
            f"| conflict_aware → ambiguity_rag | "
            f"{ambiguity['strict_accuracy'] - conflict['strict_accuracy']:+.3f} | "
            f"{ambiguity['wrong_answer_rate'] - conflict['wrong_answer_rate']:+.3f} | "
            f"{ambiguity['coverage'] - conflict['coverage']:+.3f} |"
        )
    if conflict and full:
        lines.append(
            f"| conflict_aware → evidenceguard | "
            f"{full['strict_accuracy'] - conflict['strict_accuracy']:+.3f} | "
            f"{full['wrong_answer_rate'] - conflict['wrong_answer_rate']:+.3f} | "
            f"{full['coverage'] - conflict['coverage']:+.3f} |"
        )
    return lines


def create_comparison_figure(
    output: Path,
    model_rows: list[dict],
    fallback_rows: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    base = {row["mode"]: row for row in fallback_rows}
    comparable_rows = [row for row in model_rows if row["mode"] in base]
    labels = [row["mode"] for row in comparable_rows]
    x = list(range(len(labels)))
    width = 0.35

    model_wrong = [100 * row["wrong_answer_rate"] for row in comparable_rows]
    fallback_wrong = [
        100 * float(base[row["mode"]]["wrong_answer_rate"])
        for row in comparable_rows
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], fallback_wrong, width, label="Fallback")
    ax.bar([i + width / 2 for i in x], model_wrong, width, label="Model-backed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Known wrong-answer rate (%)")
    ax.set_title("RAMDocs: model-backed vs fallback")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "wrong_answer_comparison.png", dpi=180)
    plt.close(fig)


def risk_coverage_rows(runs: list) -> list[dict]:
    forced = [run for run in runs if run.mode == "conflict_aware"]
    rows: list[dict] = []
    for threshold in [i / 20 for i in range(0, 20)]:
        answered = [run for run in forced if run.confidence >= threshold]
        correct = sum(run.strict_correct for run in answered)
        coverage = len(answered) / len(forced) if forced else 0.0
        selective_accuracy = correct / len(answered) if answered else 0.0
        rows.append(
            {
                "threshold": round(threshold, 2),
                "coverage": round(coverage, 4),
                "selective_strict_accuracy": round(selective_accuracy, 4),
                "selective_risk": round(1.0 - selective_accuracy, 4)
                if answered
                else 0.0,
                "answered": len(answered),
                "samples": len(forced),
            }
        )
    return rows


def create_risk_coverage_figure(
    output: Path,
    rows: list[dict],
    *,
    selected_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [100 * row["coverage"] for row in rows],
        [100 * row["selective_risk"] for row in rows],
        marker="o",
    )

    selected = min(
        rows,
        key=lambda row: abs(row["threshold"] - selected_threshold),
    )
    ax.scatter(
        [100 * selected["coverage"]],
        [100 * selected["selective_risk"]],
        s=70,
        label=f"calibrated threshold={selected_threshold:.2f}",
    )
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Selective risk (%)")
    ax.set_title("RAMDocs risk-coverage curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "risk_coverage.png", dpi=180)
    plt.close(fig)


def failure_counts(runs: list) -> dict:
    hybrid = {run.index: run for run in runs if run.mode == "hybrid_rag"}
    full = [run for run in runs if run.mode == "evidenceguard"]

    strict_improvements = 0
    wrong_harm_avoided = 0
    regressions = 0
    for run in full:
        base = hybrid.get(run.index)
        if base is None:
            continue
        if not base.strict_correct and run.strict_correct:
            strict_improvements += 1
        if base.wrong_answer_hit and not run.wrong_answer_hit:
            wrong_harm_avoided += 1
        if base.strict_correct and not run.strict_correct:
            regressions += 1

    return {
        "strict_correct": sum(run.strict_correct for run in full),
        "wrong_answer_hits": sum(run.wrong_answer_hit for run in full),
        "abstentions": sum(run.abstained for run in full),
        "conflict_misses": sum(
            run.conflict_expected and not run.conflict_detected
            for run in full
        ),
        "strict_improvements_over_hybrid": strict_improvements,
        "wrong_answer_harm_avoided": wrong_harm_avoided,
        "regressions_vs_hybrid": regressions,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvidenceGuard model-backed evaluation")
    parser.add_argument("--ramdocs", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--fallback-summary", required=True)
    parser.add_argument("--output", default="../research/results/model_backed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
    fallback_rows = read_csv(Path(args.fallback_summary))
    base_settings = get_settings().model_copy(
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

    validation_ids = set(frozen["protocol"]["validation_case_ids"])
    heldout_ids = set(frozen["protocol"]["heldout_test_case_ids"])

    _, validation_runs = await run_controlled_benchmark(
        base_settings,
        modes=["evidenceguard"],
        conflict_ratios=CONFLICT_RATIOS,
        use_nli=True,
        use_local_models=True,
        case_ids=validation_ids,
    )
    calibration, calibration_trials = calibrate_abstention_threshold_from_records(
        validation_runs,
        candidates=MODEL_THRESHOLD_CANDIDATES,
    )

    settings = base_settings.model_copy(
        update={"default_abstain_threshold": calibration.abstain_threshold}
    )

    controlled, controlled_runs = await run_controlled_benchmark(
        settings,
        modes=MODES,
        conflict_ratios=CONFLICT_RATIOS,
        use_nli=True,
        use_local_models=True,
        case_ids=heldout_ids,
    )
    controlled_rows = [row.model_dump() for row in controlled.rows]

    ramdocs_rows, ramdocs_runs = await run_ramdocs(
        settings,
        args.ramdocs,
        modes=MODES,
        limit=args.limit,
        use_nli=True,
        use_local_models=True,
    )

    engine_status = assert_model_backed(
        [*validation_runs, *controlled_runs],
        ramdocs_runs,
    )
    comparison = compare_with_fallback(ramdocs_rows, fallback_rows)
    failures = failure_counts(ramdocs_runs)
    risk_rows = risk_coverage_rows(ramdocs_runs)

    write_csv(output / "validation_calibration_runs.csv", [asdict(row) for row in validation_runs])
    write_csv(output / "validation_threshold_trials.csv", calibration_trials)
    write_csv(output / "controlled_test_summary.csv", controlled_rows)
    write_csv(output / "controlled_test_runs.csv", [asdict(row) for row in controlled_runs])
    write_csv(output / "ramdocs_summary.csv", ramdocs_rows)
    write_csv(output / "ramdocs_runs.csv", [asdict(row) for row in ramdocs_runs])
    write_csv(output / "comparison_vs_fallback.csv", comparison)
    write_csv(output / "risk_coverage.csv", risk_rows)

    calibration_payload = {
        **asdict(calibration),
        "candidate_thresholds": MODEL_THRESHOLD_CANDIDATES,
        "validation_case_ids": sorted(validation_ids),
        "conflict_ratios": CONFLICT_RATIOS,
        "policy": (
            "Threshold selected only on controlled validation cases with the "
            "model-backed retrieval/NLI stack; RAMDocs is not used for tuning."
        ),
    }
    (output / "CALIBRATION.json").write_text(
        json.dumps(calibration_payload, indent=2),
        encoding="utf-8",
    )

    model_config = {
        "embedding_model": settings.embedding_model,
        "nli_model": settings.nli_model,
        "evidence_retrieval_weight": settings.evidence_retrieval_weight,
        "evidence_reliability_weight": settings.evidence_reliability_weight,
        "evidence_agreement_weight": settings.evidence_agreement_weight,
        "abstain_threshold": settings.default_abstain_threshold,
        "source_fallback_abstain_threshold": frozen["abstain_threshold"],
        "ramdocs_samples": len(load_ramdocs(args.ramdocs, limit=args.limit)),
        "configuration_policy": (
            "Evidence weights transfer unchanged from the frozen fallback run; "
            "the abstention threshold is recalibrated only on the pre-declared "
            "controlled validation split with neural retrieval/NLI."
        ),
    }
    (output / "MODEL_CONFIG.json").write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )
    (output / "ENGINE_STATUS.json").write_text(
        json.dumps(engine_status, indent=2),
        encoding="utf-8",
    )

    results = [
        "# EvidenceGuard model-backed evaluation",
        "",
        "This experiment keeps the frozen evidence weights unchanged, uses local neural retrieval/NLI, and calibrates only the abstention threshold on the pre-declared controlled validation split.",
        "",
        "## Model configuration",
        "",
        "~~~json",
        json.dumps(model_config, indent=2),
        "~~~",
        "",
        "## Neural abstention calibration",
        "",
        "~~~json",
        json.dumps(calibration_payload, indent=2),
        "~~~",
        "",
        "The calibration uses forced-answer validation runs so threshold candidates are evaluated from one fixed set of neural predictions. RAMDocs labels are never consulted during calibration.",
        "",
        "## RAMDocs results",
        "",
        *ramdocs_markdown(ramdocs_rows),
        "",
        "## Neural stage ablation",
        "",
        "Negative wrong-answer deltas are improvements. consensus_rag tests contradiction pruning; ambiguity_rag tests support-cluster answer aggregation; evidenceguard remains the canonical conflict-aware + abstention system unless an ablation clearly wins.",
        "",
        *neural_ablation_markdown(ramdocs_rows),
        "",
        "## Difference from frozen fallback run",
        "",
        "Positive accuracy/F1 deltas are improvements; negative wrong-answer deltas are improvements. The model-backed EvidenceGuard row uses its validation-calibrated abstention threshold, while the fallback row retains its own frozen threshold.",
        "",
        *comparison_markdown(comparison),
        "",
        "## EvidenceGuard failure counts",
        "",
        "~~~json",
        json.dumps(failures, indent=2),
        "~~~",
        "",
        "## Engine verification",
        "",
        "~~~json",
        json.dumps(engine_status, indent=2),
        "~~~",
        "",
        "## Interpretation constraints",
        "",
        "- Evidence-score weights are transferred unchanged from the frozen fallback experiment.",
        "- The neural abstention threshold is selected only on the pre-declared controlled validation cases.",
        "- No RAMDocs label is passed into retrieval, NLI, scoring, generation, abstention calibration, or abstention logic.",
        "- Strict correctness requires every listed gold answer and no listed wrong answer after normalized phrase matching.",
        "- The generator remains the extractive fallback so this run isolates retrieval/NLI and selective-answering changes rather than mixing in an LLM generator.",
        "- The risk-coverage curve is post-hoc evaluation built from conflict_aware, the forced-answer parent of EvidenceGuard; consensus_rag is reported separately as a side ablation.",
        "- The workflow fails if dense retrieval or NLI silently drops to a fallback engine.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    create_comparison_figure(output, ramdocs_rows, fallback_rows)
    create_risk_coverage_figure(
        output,
        risk_rows,
        selected_threshold=calibration.abstain_threshold,
    )

    print(json.dumps({
        "output": str(output),
        "model_config": model_config,
        "calibration": calibration_payload,
        "engine_status": engine_status,
        "failures": failures,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
