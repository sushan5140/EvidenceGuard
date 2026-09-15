from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import get_settings
from app.research.entity_aware_augmentation import augment_with_distinct_entities
from app.research.entity_pair import EntityPairClassifier, load_ambigdocs
from app.research.qa_aggregate import ExtractiveQAAggregator
from app.research.ramdocs import _answer_flags, load_ramdocs
from app.schemas import DocumentCreate
from app.services.pipeline import EvidenceGuardPipeline
from app.services.store import DocumentStore


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


def metrics(runs: list[GateRun], prefix: str) -> dict:
    n = len(runs)
    return {
        "strict_accuracy": round(sum(getattr(run, f"{prefix}_strict") for run in runs) / n, 4),
        "all_gold_hit_rate": round(sum(getattr(run, f"{prefix}_all_gold") for run in runs) / n, 4),
        "any_gold_hit_rate": round(sum(getattr(run, f"{prefix}_any_gold") for run in runs) / n, 4),
        "wrong_answer_rate": round(sum(getattr(run, f"{prefix}_wrong") for run in runs) / n, 4),
    }


async def build_response(settings, case):
    with TemporaryDirectory(prefix="evidenceguard-entity-aware-") as tmp:
        run_settings = settings.model_copy(update={"data_path": str(Path(tmp) / "documents.json")})
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
        return await pipeline.query(
            case.question,
            top_k=min(12, max(2, len(case.documents))),
            use_nli=True,
            mode="conflict_aware",
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Externally trained entity-aware augmentation gate")
    parser.add_argument("--ambigdocs-dev", required=True)
    parser.add_argument("--ramdocs", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output", default="../research/results/entity_aware_augmentation_gate")
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

    # External development only: no RAMDocs field is used here.
    ambigdocs = load_ambigdocs(args.ambigdocs_dev)
    entity_classifier = EntityPairClassifier(settings.embedding_model)
    entity_calibration = entity_classifier.fit(ambigdocs)

    qa = ExtractiveQAAggregator(args.qa_model)
    runs: list[GateRun] = []
    cases = load_ramdocs(args.ramdocs, limit=args.ramdocs_limit)

    for index, case in enumerate(cases):
        response = await build_response(settings, case)
        baseline = _answer_flags(response.answer, case, abstained=response.abstained)

        candidates = qa.extract_candidates(case.question, response.evidence)
        augmented = augment_with_distinct_entities(
            response.answer,
            candidates,
            evidence=response.evidence,
            graph=response.graph,
            classifier=entity_classifier,
            candidate_score_floor=0.60,
        )
        flags = _answer_flags(augmented.text, case, abstained=False)

        runs.append(GateRun(
            index=index,
            baseline_answer=response.answer,
            augmented_answer=augmented.text,
            baseline_any_gold=baseline[0],
            baseline_all_gold=baseline[1],
            baseline_wrong=baseline[2],
            baseline_strict=baseline[3],
            augmented_any_gold=flags[0],
            augmented_all_gold=flags[1],
            augmented_wrong=flags[2],
            augmented_strict=flags[3],
            raw_candidate_count=len(candidates),
            addition_count=len(augmented.additions),
            additions=";".join(
                f"{item.answer}:{item.verification_score:.4f}:same={item.max_same_entity_probability:.4f}"
                for item in augmented.additions
            ),
            retrieval_engine=response.model_status.get("retrieval", ""),
            nli_engine=response.model_status.get("nli", ""),
        ))

    baseline_metrics = metrics(runs, "baseline")
    augmented_metrics = metrics(runs, "augmented")
    delta = {
        key: round(augmented_metrics[key] - baseline_metrics[key], 4)
        for key in baseline_metrics
    }
    paired = {
        "strict_improvements": sum(not r.baseline_strict and r.augmented_strict for r in runs),
        "strict_regressions": sum(r.baseline_strict and not r.augmented_strict for r in runs),
        "all_gold_improvements": sum(not r.baseline_all_gold and r.augmented_all_gold for r in runs),
        "wrong_answer_added": sum(not r.baseline_wrong and r.augmented_wrong for r in runs),
    }

    gate_pass = (
        augmented_metrics["strict_accuracy"] >= baseline_metrics["strict_accuracy"]
        and augmented_metrics["all_gold_hit_rate"] >= baseline_metrics["all_gold_hit_rate"] + 0.03
        and augmented_metrics["wrong_answer_rate"] <= baseline_metrics["wrong_answer_rate"] + 0.03
        and paired["strict_improvements"] >= paired["strict_regressions"]
        and paired["all_gold_improvements"] > paired["wrong_answer_added"]
    )

    calibration_payload = {
        **asdict(entity_calibration),
        "embedding_model": settings.embedding_model,
        "train_cases": 900,
        "calibration_cases": 350,
        "candidate_score_floor": 0.60,
        "policy": (
            "Entity identity classifier and operating threshold are learned only from "
            "AmbigDocs development data. RAMDocs is used only after training is frozen."
        ),
    }
    summary = {
        "experiment": "entity_aware_baseline_augmentation_gate",
        "ramdocs_samples": len(runs),
        "external_entity_calibration": calibration_payload,
        "baseline_conflict_aware": baseline_metrics,
        "entity_aware_augmented": augmented_metrics,
        "delta": delta,
        "paired_outcomes": paired,
        "mean_raw_candidates": round(sum(r.raw_candidate_count for r in runs) / len(runs), 4),
        "mean_additions": round(sum(r.addition_count for r in runs) / len(runs), 4),
        "gate_pass": gate_pass,
        "gate_rule": (
            "strict non-decreasing; all-gold +3pp or better; wrong-answer increase <=3pp; "
            "strict improvements >= regressions; all-gold improvements > wrong additions"
        ),
    }

    write_csv(output / "gate_runs.csv", [asdict(run) for run in runs])
    (output / "ENTITY_CALIBRATION.json").write_text(
        json.dumps(calibration_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(
        "# Entity-aware baseline augmentation - paired gate\n\n"
        "Entity identity is learned from AmbigDocs development data before RAMDocs is opened.\n\n"
        "~~~json\n" + json.dumps(summary, indent=2) + "\n~~~\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
