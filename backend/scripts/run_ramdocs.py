from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.research.ramdocs import run_ramdocs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EvidenceGuard on RAMDocs")
    parser.add_argument("dataset", help="Path to RAMDocs_test.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="research/results/ramdocs")
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Disable local transformer embedding/NLI models",
    )
    parser.add_argument(
        "--heuristic-nli",
        action="store_true",
        help="Use heuristic conflict comparison instead of transformer NLI",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary, runs = await run_ramdocs(
        get_settings(),
        args.dataset,
        modes=[
            "basic_rag",
            "hybrid_rag",
            "conflict_aware",
            "evidenceguard",
        ],
        limit=args.limit,
        use_nli=not args.heuristic_nli,
        use_local_models=not args.fallback_only,
    )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    run_rows = [asdict(run) for run in runs]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        if run_rows:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)

    print(json.dumps({"output": str(output), "samples": len(runs)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
