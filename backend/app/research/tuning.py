from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.research.benchmark import RunRecord, run_controlled_benchmark


@dataclass(slots=True)
class FrozenConfig:
    evidence_retrieval_weight: float
    evidence_reliability_weight: float
    evidence_agreement_weight: float
    abstain_threshold: float
    validation_score: float
    validation_utility: float
    validation_ece: float
    validation_coverage: float


WEIGHT_CANDIDATES = [
    (0.50, 0.20, 0.30),
    (0.45, 0.25, 0.30),
    (0.40, 0.20, 0.40),
    (0.40, 0.30, 0.30),
    (0.50, 0.30, 0.20),
]
THRESHOLD_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _utility(records: list[RunRecord]) -> float:
    if not records:
        return -1.0
    total = 0.0
    for record in records:
        if record.correct:
            total += 1.0
        elif record.abstained:
            total += 0.20
        else:
            total -= 1.0
    return total / len(records)


def _ece(records: list[RunRecord], bins: int = 10) -> float:
    if not records:
        return 1.0
    total = len(records)
    result = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        group = [
            record
            for record in records
            if lo <= record.confidence < hi
            or (index == bins - 1 and record.confidence == 1.0)
        ]
        if not group:
            continue
        acc = sum(record.correct for record in group) / len(group)
        conf = sum(record.confidence for record in group) / len(group)
        result += len(group) / total * abs(acc - conf)
    return result


def _coverage(records: list[RunRecord]) -> float:
    if not records:
        return 0.0
    return sum(not record.abstained for record in records) / len(records)


async def tune_on_controlled_validation(
    settings: Settings,
    *,
    validation_case_ids: set[str],
    conflict_ratios: list[float],
) -> tuple[FrozenConfig, list[dict]]:
    trials: list[dict] = []

    for retrieval, reliability, agreement in WEIGHT_CANDIDATES:
        for threshold in THRESHOLD_CANDIDATES:
            candidate = settings.model_copy(
                update={
                    "evidence_retrieval_weight": retrieval,
                    "evidence_reliability_weight": reliability,
                    "evidence_agreement_weight": agreement,
                    "default_abstain_threshold": threshold,
                }
            )
            _, records = await run_controlled_benchmark(
                candidate,
                modes=["evidenceguard"],
                conflict_ratios=conflict_ratios,
                use_nli=False,
                use_local_models=False,
                case_ids=validation_case_ids,
            )
            utility = _utility(records)
            ece = _ece(records)
            coverage = _coverage(records)

            # Pre-registered engineering objective:
            # prefer correct > abstain > wrong, penalize poor calibration,
            # and discourage degenerate near-total abstention.
            low_coverage_penalty = max(0.0, 0.40 - coverage) * 0.50
            score = utility - 0.15 * ece - low_coverage_penalty

            trials.append(
                {
                    "evidence_retrieval_weight": retrieval,
                    "evidence_reliability_weight": reliability,
                    "evidence_agreement_weight": agreement,
                    "abstain_threshold": threshold,
                    "validation_utility": round(utility, 6),
                    "validation_ece": round(ece, 6),
                    "validation_coverage": round(coverage, 6),
                    "validation_score": round(score, 6),
                }
            )

    trials.sort(
        key=lambda row: (
            row["validation_score"],
            row["validation_utility"],
            -row["validation_ece"],
            row["validation_coverage"],
        ),
        reverse=True,
    )
    best = trials[0]
    return (
        FrozenConfig(
            evidence_retrieval_weight=best["evidence_retrieval_weight"],
            evidence_reliability_weight=best["evidence_reliability_weight"],
            evidence_agreement_weight=best["evidence_agreement_weight"],
            abstain_threshold=best["abstain_threshold"],
            validation_score=best["validation_score"],
            validation_utility=best["validation_utility"],
            validation_ece=best["validation_ece"],
            validation_coverage=best["validation_coverage"],
        ),
        trials,
    )
