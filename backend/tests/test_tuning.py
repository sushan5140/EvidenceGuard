from app.research.benchmark import RunRecord
from app.research.tuning import (
    _coverage,
    _utility,
    calibrate_abstention_threshold_from_records,
)


def record(*, correct=False, abstained=False, confidence=0.5):
    return RunRecord(
        case_id="x",
        mode="evidenceguard",
        requested_conflict_ratio=0.5,
        actual_conflict_ratio=0.5,
        correct=correct,
        abstained=abstained,
        confidence=confidence,
        conflict_expected=True,
        conflict_detected=True,
    )


def test_utility_prefers_correct_then_abstain_then_wrong():
    assert _utility([record(correct=True)]) > _utility([record(abstained=True)])
    assert _utility([record(abstained=True)]) > _utility([record()])


def test_coverage_counts_only_answered_records():
    assert _coverage([record(correct=True), record(abstained=True)]) == 0.5


def test_model_threshold_calibration_prefers_selective_answering():
    records = [
        record(correct=True, confidence=0.90),
        record(correct=False, confidence=0.30),
        record(correct=False, confidence=0.10),
    ]
    calibrated, trials = calibrate_abstention_threshold_from_records(
        records,
        candidates=[0.20, 0.40],
    )

    assert calibrated.abstain_threshold == 0.40
    assert calibrated.validation_selective_accuracy == 1.0
    assert calibrated.validation_coverage == 0.333333
    assert len(trials) == 2
