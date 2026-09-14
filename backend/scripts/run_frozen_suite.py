from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.research.benchmark import load_controlled_cases, run_controlled_benchmark
from app.research.ramdocs import load_ramdocs, run_ramdocs
from app.research.tuning import tune_on_controlled_validation


MODES = ["basic_rag", "hybrid_rag", "conflict_aware", "evidenceguard"]
CONFLICT_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def stable_split() -> tuple[set[str], set[str]]:
    ids = sorted(case["id"] for case in load_controlled_cases())
    validation = set(ids[::2])
    test = set(ids[1::2])
    return validation, test


def controlled_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "| Mode | Conflict | Accuracy | Selective accuracy | Coverage | Mean conf. | ECE | Conflict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {pct(row['conflict_ratio'])} | {pct(row['accuracy'])} | "
            f"{pct(row['selective_accuracy'])} | {pct(row['coverage'])} | "
            f"{pct(row['mean_confidence'])} | {row['ece']:.3f} | {row['conflict_f1']:.3f} |"
        )
    return lines


def ramdocs_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "| Mode | Samples | Strict accuracy | All-gold hit | Any-gold hit | Wrong-answer hit | Abstention | Mean conf. | ECE | Conflict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['samples']} | {pct(row['strict_accuracy'])} | "
            f"{pct(row['all_gold_hit_rate'])} | {pct(row['any_gold_hit_rate'])} | "
            f"{pct(row['wrong_answer_rate'])} | {pct(row['abstention_rate'])} | "
            f"{pct(row['mean_confidence'])} | {row['ece_strict']:.3f} | "
            f"{row['conflict_f1']:.3f} |"
        )
    return lines


def create_figures(output: Path, controlled_rows: list[dict], ramdocs_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        subset = [row for row in controlled_rows if row["mode"] == mode]
        ax.plot(
            [100 * row["conflict_ratio"] for row in subset],
            [100 * row["accuracy"] for row in subset],
            marker="o",
            label=mode,
        )
    ax.set_xlabel("Conflicting evidence (%)")
    ax.set_ylabel("Held-out answer accuracy (%)")
    ax.set_title("EvidenceGuard controlled robustness")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "controlled_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        subset = [row for row in controlled_rows if row["mode"] == mode]
        ax.plot(
            [100 * row["conflict_ratio"] for row in subset],
            [100 * row["mean_confidence"] for row in subset],
            marker="o",
            label=mode,
        )
    ax.set_xlabel("Conflicting evidence (%)")
    ax.set_ylabel("Mean confidence (%)")
    ax.set_title("Confidence under increasing conflict")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "controlled_confidence.png", dpi=180)
    plt.close(fig)

    labels = [row["mode"] for row in ramdocs_rows]
    gold = [100 * row["strict_accuracy"] for row in ramdocs_rows]
    wrong = [100 * row["wrong_answer_rate"] for row in ramdocs_rows]
    abstain = [100 * row["abstention_rate"] for row in ramdocs_rows]
    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], gold, width, label="Strict correct")
    ax.bar(x, wrong, width, label="Wrong-answer hit")
    ax.bar([i + width for i in x], abstain, width, label="Abstention")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("RAMDocs outcomes with frozen configuration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ramdocs_outcomes.png", dpi=180)
    plt.close(fig)


def failure_analysis(ramdocs_path: Path, runs: list, output: Path) -> dict:
    cases = load_ramdocs(ramdocs_path)
    by_mode_index = {(run.mode, run.index): run for run in runs}

    eg = [run for run in runs if run.mode == "evidenceguard"]
    hybrid = [run for run in runs if run.mode == "hybrid_rag"]

    categories = {
        "strict_correct": [run for run in eg if run.strict_correct],
        "wrong_answer_hits": [run for run in eg if run.wrong_answer_hit],
        "abstentions": [run for run in eg if run.abstained],
        "conflict_misses": [
            run for run in eg if run.conflict_expected and not run.conflict_detected
        ],
        "improvements_over_hybrid": [],
        "regressions_vs_hybrid": [],
    }

    for base in hybrid:
        full = by_mode_index.get(("evidenceguard", base.index))
        if full is None:
            continue
        if (not base.strict_correct) and (full.strict_correct or full.abstained):
            categories["improvements_over_hybrid"].append(full)
        if base.strict_correct and not full.strict_correct:
            categories["regressions_vs_hybrid"].append(full)

    lines = [
        "# EvidenceGuard failure analysis — frozen RAMDocs run",
        "",
        "This report is generated automatically from the same frozen run as the result tables.",
        "Document-type labels are used only after inference for evaluation.",
        "",
        "## Aggregate categories",
        "",
        "| Category | Count | Rate |",
        "|---|---:|---:|",
    ]
    total = max(1, len(eg))
    for name, items in categories.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {len(items)} | {pct(len(items) / total)} |"
        )

    def examples(title: str, items: list) -> None:
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("No examples in this run.")
            return
        for run in items[:5]:
            case = cases[run.index]
            lines.extend(
                [
                    f"### Case {run.index + 1}",
                    "",
                    f"**Question:** {case.question}",
                    "",
                    f"**Gold answers:** {', '.join(case.gold_answers) or 'n/a'}",
                    "",
                    f"**Wrong answers:** {', '.join(case.wrong_answers) or 'n/a'}",
                    "",
                    f"**Outcome:** strict_correct={run.strict_correct}, all_gold={run.all_gold_hit}, "
                    f"any_gold={run.any_gold_hit}, wrong_hit={run.wrong_answer_hit}, "
                    f"abstained={run.abstained}, confidence={run.confidence:.3f}, "
                    f"conflict_detected={run.conflict_detected}",
                    "",
                ]
            )

    examples("Wrong-answer adoption", categories["wrong_answer_hits"])
    examples("Conflict misses", categories["conflict_misses"])
    examples("Cases improved relative to Hybrid RAG", categories["improvements_over_hybrid"])
    examples("Regressions relative to Hybrid RAG", categories["regressions_vs_hybrid"])

    (output / "FAILURE_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {name: len(items) for name, items in categories.items()}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Tune, freeze, and evaluate EvidenceGuard")
    parser.add_argument("--ramdocs", required=True, help="Path to official RAMDocs_test.jsonl")
    parser.add_argument("--output", default="../research/results/frozen")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ramdocs_path = Path(args.ramdocs).resolve()
    settings = get_settings()

    validation_ids, test_ids = stable_split()

    frozen, trials = await tune_on_controlled_validation(
        settings,
        validation_case_ids=validation_ids,
        conflict_ratios=CONFLICT_RATIOS,
    )
    frozen_settings = settings.model_copy(
        update={
            "evidence_retrieval_weight": frozen.evidence_retrieval_weight,
            "evidence_reliability_weight": frozen.evidence_reliability_weight,
            "evidence_agreement_weight": frozen.evidence_agreement_weight,
            "default_abstain_threshold": frozen.abstain_threshold,
        }
    )

    frozen_payload = {
        **asdict(frozen),
        "protocol": {
            "suite": "controlled-conflicts-v2",
            "validation_case_ids": sorted(validation_ids),
            "heldout_test_case_ids": sorted(test_ids),
            "conflict_ratios": CONFLICT_RATIOS,
            "tuning_models": "deterministic fallback retrieval + heuristic NLI",
            "external_evaluation": "RAMDocs_test.jsonl, full file, no tuning on RAMDocs labels",
        },
    }
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_payload, indent=2),
        encoding="utf-8",
    )
    write_csv(output / "validation_trials.csv", trials)

    controlled_response, controlled_runs = await run_controlled_benchmark(
        frozen_settings,
        modes=MODES,
        conflict_ratios=CONFLICT_RATIOS,
        use_nli=False,
        use_local_models=False,
        case_ids=test_ids,
    )
    controlled_rows = [row.model_dump() for row in controlled_response.rows]
    write_csv(output / "controlled_test_summary.csv", controlled_rows)
    write_csv(output / "controlled_test_runs.csv", [asdict(record) for record in controlled_runs])

    ramdocs_summary, ramdocs_runs = await run_ramdocs(
        frozen_settings,
        ramdocs_path,
        modes=MODES,
        limit=None,
        use_nli=False,
        use_local_models=False,
    )
    write_csv(output / "ramdocs_summary.csv", ramdocs_summary)
    write_csv(output / "ramdocs_runs.csv", [asdict(record) for record in ramdocs_runs])
    (output / "ramdocs_summary.json").write_text(
        json.dumps(ramdocs_summary, indent=2),
        encoding="utf-8",
    )

    failure_counts = failure_analysis(ramdocs_path, ramdocs_runs, output)
    create_figures(output, controlled_rows, ramdocs_summary)

    results_lines = [
        "# EvidenceGuard frozen evaluation results",
        "",
        "These numbers are generated by the repository workflow; they are not hand-entered.",
        "",
        "## Frozen configuration",
        "",
        "~~~json",
        json.dumps(frozen_payload, indent=2),
        "~~~",
        "",
        "## Held-out controlled benchmark",
        "",
        *controlled_markdown(controlled_rows),
        "",
        "## RAMDocs external evaluation",
        "",
        *ramdocs_markdown(ramdocs_summary),
        "",
        "## Failure counts",
        "",
        "~~~json",
        json.dumps(failure_counts, indent=2),
        "~~~",
        "",
        "## Interpretation constraints",
        "",
        "- Controlled tuning uses only validation cases; the listed controlled results use held-out case IDs.",
        "- RAMDocs labels are not used as inference features and are never used for tuning.",
        "- RAMDocs strict correctness requires all listed gold answers and zero listed wrong answers after normalized phrase matching.",
        "- This is a transparent adapter metric, not a claim of byte-for-byte equivalence with the paper's official evaluator.",
        "- This frozen run intentionally uses deterministic fallback retrieval and heuristic NLI for reproducibility on CI.",
        "- Model-backed experiments should be reported as a separate run rather than silently replacing these results.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(results_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output),
                "validation_cases": len(validation_ids),
                "heldout_cases": len(test_ids),
                "ramdocs_cases": len(load_ramdocs(ramdocs_path)),
                "frozen_config": frozen_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
