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


@dataclass(slots=True)
class ThresholdCalibration:
    abstain_threshold: float
    validation_score: float
    validation_utility: float
    validation_coverage: float
    validation_selective_accuracy: float
    answered: int
    samples: int


WEIGHT_CANDIDATES = [
    (0.50, 0.20, 0.30),
    (0.45, 0.25, 0.30),
    (0.40, 0.20, 0.40),
    (0.40, 0.30, 0.30),
    (0.50, 0.30, 0.20),
]
THRESHOLD_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MODEL_THRESHOLD_CANDIDATES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


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


def calibrate_abstention_threshold_from_records(
    records: list[RunRecord],
    *,
    candidates: list[float] | None = None,
    minimum_coverage: float = 0.40,
) -> tuple[ThresholdCalibration, list[dict]]:
    """Tune only the abstention threshold from forced-answer validation runs.

    records should come from EvidenceGuard with the abstention threshold set to
    zero so correctness and confidence are observed before selective answering.
    The function performs no model inference and therefore can sweep many
    thresholds without changing retrieval or NLI outputs.
    """

    if not records:
        raise ValueError("threshold calibration requires at least one validation record")

    thresholds = candidates or MODEL_THRESHOLD_CANDIDATES
    trials: list[dict] = []

    for threshold in thresholds:
        answered = [record for record in records if record.confidence >= threshold]
        abstained_count = len(records) - len(answered)
        correct_answered = sum(record.correct for record in answered)
        wrong_answered = len(answered) - correct_answered

        utility = (
            correct_answered * 1.0
            + abstained_count * 0.20
            - wrong_answered * 1.0
        ) / len(records)
        coverage = len(answered) / len(records)
        selective_accuracy = (
            correct_answered / len(answered)
            if answered
            else 0.0
        )

        low_coverage_penalty = max(0.0, minimum_coverage - coverage) * 0.50
        score = utility - low_coverage_penalty

        trials.append(
            {
                "abstain_threshold": threshold,
                "validation_utility": round(utility, 6),
                "validation_coverage": round(coverage, 6),
                "validation_selective_accuracy": round(selective_accuracy, 6),
                "validation_score": round(score, 6),
                "answered": len(answered),
                "samples": len(records),
            }
        )

    trials.sort(
        key=lambda row: (
            row["validation_score"],
            row["validation_utility"],
            row["validation_selective_accuracy"],
            row["validation_coverage"],
            -row["abstain_threshold"],
        ),
        reverse=True,
    )
    best = trials[0]
    return (
        ThresholdCalibration(
            abstain_threshold=best["abstain_threshold"],
            validation_score=best["validation_score"],
            validation_utility=best["validation_utility"],
            validation_coverage=best["validation_coverage"],
            validation_selective_accuracy=best["validation_selective_accuracy"],
            answered=best["answered"],
            samples=best["samples"],
        ),
        trials,
    )


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
