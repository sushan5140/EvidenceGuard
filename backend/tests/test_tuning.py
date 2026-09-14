from app.research.benchmark import RunRecord
from app.research.tuning import _coverage, _utility


def record(*, correct=False, abstained=False):
    return RunRecord(
        case_id="x",
        mode="evidenceguard",
        requested_conflict_ratio=0.5,
        actual_conflict_ratio=0.5,
        correct=correct,
        abstained=abstained,
        confidence=0.5,
        conflict_expected=True,
        conflict_detected=True,
    )


def test_utility_prefers_correct_then_abstain_then_wrong():
    assert _utility([record(correct=True)]) > _utility([record(abstained=True)])
    assert _utility([record(abstained=True)]) > _utility([record()])


def test_coverage_counts_only_answered_records():
    assert _coverage([record(correct=True), record(abstained=True)]) == 0.5
