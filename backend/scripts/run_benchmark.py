from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.research.benchmark import run_controlled_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvidenceGuard controlled benchmark")
    parser.add_argument(
        "--output",
        default="research/results",
        help="Output directory for JSON/CSV/Markdown artifacts",
    )
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Enable sentence-transformers and transformer NLI models",
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Use the transformer NLI path when local models are enabled",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    response, records = await run_controlled_benchmark(
        get_settings(),
        modes=[
            "basic_rag",
            "hybrid_rag",
            "conflict_aware",
            "consensus_rag",
            "evidenceguard",
        ],
        conflict_ratios=[0.0, 0.10, 0.25, 0.50, 0.75],
        use_nli=args.nli,
        use_local_models=args.local_models,
    )

    (output / "report.json").write_text(
        response.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [row.model_dump() for row in response.rows]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [asdict(record) for record in records]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# EvidenceGuard controlled benchmark",
        "",
        f"Cases: **{response.cases}**",
        "",
        "| Mode | Conflict | Accuracy | Selective accuracy | Coverage | ECE | Conflict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in response.rows:
        markdown.append(
            f"| {row.mode} | {row.conflict_ratio:.0%} | {row.accuracy:.1%} | "
            f"{row.selective_accuracy:.1%} | {row.coverage:.1%} | "
            f"{row.ece:.3f} | {row.conflict_f1:.3f} |"
        )
    markdown.extend(["", "## Notes", *[f"- {note}" for note in response.notes]])
    (output / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(output), "rows": len(response.rows)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
