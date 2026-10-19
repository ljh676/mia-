"""Validated binary metrics. Unavailable measurements are None, never zero."""

import math


METRIC_NAMES = ("accuracy", "auc", "precision", "recall", "f1")


def _to_list(values):
    return [] if values is None else list(values)


def _binary(values, name):
    result = [float(v) for v in _to_list(values)]
    if any(v not in (0.0, 1.0) for v in result):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return [int(v) for v in result]


def _counts(labels, predictions):
    labels = _binary(labels, "labels")
    predictions = _binary(predictions, "predictions")
    if labels and predictions and len(labels) != len(predictions):
        raise ValueError("labels and predictions have different lengths")
    if not labels or not predictions:
        return None
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    return tp, fp, fn, tn


def accuracy(labels, predictions):
    counts = _counts(labels, predictions)
    if counts is None:
        return None
    tp, fp, fn, tn = counts
    return (tp + tn) / sum(counts)


def precision(labels, predictions):
    counts = _counts(labels, predictions)
    if counts is None:
        return None
    tp, fp, _, _ = counts
    return tp / (tp + fp) if tp + fp else 0.0


def recall(labels, predictions):
    counts = _counts(labels, predictions)
    if counts is None:
        return None
    tp, _, fn, _ = counts
    return tp / (tp + fn) if tp + fn else 0.0


def f1(labels, predictions):
    counts = _counts(labels, predictions)
    if counts is None:
        return None
    tp, fp, fn, _ = counts
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def auc(labels, scores):
    labels = _binary(labels, "labels")
    scores = [float(v) for v in _to_list(scores)]
    if any(not math.isfinite(v) for v in scores):
        raise ValueError("scores must be finite")
    if labels and scores and len(labels) != len(scores):
        raise ValueError("labels and scores have different lengths")
    if not labels or not scores or len(set(labels)) < 2:
        return None
    ordered = sorted(zip(scores, labels))
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = (index + 1 + end) / 2
        positive_rank_sum += rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def compute_metrics(labels, predictions, scores=None):
    labels, predictions, scores = map(_to_list, (labels, predictions, scores))
    return {"accuracy": accuracy(labels, predictions), "auc": auc(labels, scores),
            "precision": precision(labels, predictions), "recall": recall(labels, predictions),
            "f1": f1(labels, predictions)}


def metric_delta(before, after):
    return None if before is None or after is None else float(before) - float(after)


def compare_metrics(before_result, after_result):
    before = before_result.get("metrics") or {}
    after = after_result.get("metrics") or {}
    return {"auc_drop": metric_delta(before.get("auc"), after.get("auc")),
            "accuracy_drop": metric_delta(before.get("accuracy"), after.get("accuracy")),
            "interpretation": "score change only; a lower AUC alone does not establish privacy protection"}


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"
