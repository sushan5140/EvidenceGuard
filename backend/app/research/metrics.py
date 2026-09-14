from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> BinaryMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(precision, recall, f1)


def expected_calibration_error(
    correctness: list[bool],
    confidence: list[float],
    *,
    bins: int = 10,
) -> float:
    if len(correctness) != len(confidence):
        raise ValueError("correctness and confidence must have equal length")
    if not correctness:
        return 0.0

    total = len(correctness)
    ece = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidence)
            if (lo <= value < hi) or (index == bins - 1 and value == 1.0)
        ]
        if not members:
            continue
        bin_acc = sum(correctness[i] for i in members) / len(members)
        bin_conf = sum(confidence[i] for i in members) / len(members)
        ece += (len(members) / total) * abs(bin_acc - bin_conf)
    return ece


def normalize_answer_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(normalized.split())


def contains_answer(text: str, answer: str) -> bool:
    candidate = normalize_answer_text(answer)
    if not candidate:
        return False
    haystack = f" {normalize_answer_text(text)} "
    return f" {candidate} " in haystack


def keyword_answer_correct(
    answer: str,
    required_keywords: list[str],
    *,
    abstained: bool,
) -> bool:
    if abstained or not required_keywords:
        return False
    return all(contains_answer(answer, keyword) for keyword in required_keywords)


def strict_answer_correct(
    answer: str,
    required_keywords: list[str],
    forbidden_keywords: list[str],
    *,
    abstained: bool,
) -> bool:
    if not keyword_answer_correct(
        answer,
        required_keywords,
        abstained=abstained,
    ):
        return False
    return not any(contains_answer(answer, keyword) for keyword in forbidden_keywords)


def selective_accuracy(correctness: list[bool], abstained: list[bool]) -> float:
    answered = [ok for ok, abstain in zip(correctness, abstained) if not abstain]
    return sum(answered) / len(answered) if answered else 0.0


def coverage(abstained: list[bool]) -> float:
    return (
        sum(not value for value in abstained) / len(abstained)
        if abstained
        else 0.0
    )
