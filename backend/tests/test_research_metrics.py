from app.research.metrics import (
    binary_metrics,
    contains_answer,
    coverage,
    expected_calibration_error,
    keyword_answer_correct,
    selective_accuracy,
    strict_answer_correct,
)


def test_binary_metrics_perfect():
    metrics = binary_metrics([True, False, True], [True, False, True])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_calibration_error_zero_for_perfect_extremes():
    assert expected_calibration_error([True, False], [1.0, 0.0], bins=2) == 0.0


def test_keyword_accuracy_and_selective_metrics():
    assert keyword_answer_correct(
        "Python was created by Guido van Rossum [1].",
        ["Guido", "van", "Rossum"],
        abstained=False,
    )
    correctness = [True, False, False]
    abstained = [False, True, False]
    assert selective_accuracy(correctness, abstained) == 0.5
    assert round(coverage(abstained), 4) == 0.6667


def test_answer_matching_uses_boundaries():
    assert contains_answer("The symbol is Au.", "Au")
    assert not contains_answer("Australia is a country.", "Au")


def test_strict_correct_rejects_gold_plus_wrong_answer():
    assert strict_answer_correct(
        "The evidence mentions Canberra.",
        ["Canberra"],
        ["Sydney"],
        abstained=False,
    )
    assert not strict_answer_correct(
        "Sources mention Canberra and Sydney.",
        ["Canberra"],
        ["Sydney"],
        abstained=False,
    )
