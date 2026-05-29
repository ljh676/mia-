"""统一评价指标工具。"""

from typing import Dict, Iterable, List, Optional


def _to_list(values: Optional[Iterable[float]]) -> List[float]:
    """把输入统一转换为列表，便于后续计算。"""
    if values is None:
        return []
    return list(values)


def accuracy(labels: Iterable[int], predictions: Iterable[int]) -> float:
    """计算准确率。"""
    y_true = _to_list(labels)
    y_pred = _to_list(predictions)
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for label, pred in zip(y_true, y_pred) if int(label) == int(pred))
    return correct / len(y_true)


def precision(labels: Iterable[int], predictions: Iterable[int]) -> float:
    """计算二分类 precision，默认正类为 1。"""
    y_true = _to_list(labels)
    y_pred = _to_list(predictions)
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    tp = sum(1 for label, pred in zip(y_true, y_pred) if int(label) == 1 and int(pred) == 1)
    fp = sum(1 for label, pred in zip(y_true, y_pred) if int(label) == 0 and int(pred) == 1)
    denominator = tp + fp
    return tp / denominator if denominator else 0.0


def recall(labels: Iterable[int], predictions: Iterable[int]) -> float:
    """计算二分类 recall，默认正类为 1。"""
    y_true = _to_list(labels)
    y_pred = _to_list(predictions)
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    tp = sum(1 for label, pred in zip(y_true, y_pred) if int(label) == 1 and int(pred) == 1)
    fn = sum(1 for label, pred in zip(y_true, y_pred) if int(label) == 1 and int(pred) == 0)
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def f1(labels: Iterable[int], predictions: Iterable[int]) -> float:
    """计算二分类 F1。"""
    p_value = precision(labels, predictions)
    r_value = recall(labels, predictions)
    denominator = p_value + r_value
    return 2 * p_value * r_value / denominator if denominator else 0.0


def auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """计算 ROC-AUC。

    这里使用纯 Python 排名法，避免框架强依赖 sklearn。
    当标签为空、长度不一致、或只有一个类别时返回 0。
    """
    y_true = [int(v) for v in _to_list(labels)]
    y_score = [float(v) for v in _to_list(scores)]
    if not y_true or len(y_true) != len(y_score):
        return 0.0

    positives = sum(1 for v in y_true if v == 1)
    negatives = sum(1 for v in y_true if v == 0)
    if positives == 0 or negatives == 0:
        return 0.0

    # 处理并列分数：同分样本使用平均排名。
    sorted_pairs = sorted(enumerate(y_score), key=lambda item: item[1])
    ranks = [0.0] * len(y_score)
    index = 0
    while index < len(sorted_pairs):
        end = index
        while end + 1 < len(sorted_pairs) and sorted_pairs[end + 1][1] == sorted_pairs[index][1]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        for pair_index in range(index, end + 1):
            original_index = sorted_pairs[pair_index][0]
            ranks[original_index] = average_rank
        index = end + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    scores: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """统一计算所有指标。"""
    y_true = _to_list(labels)
    y_pred = _to_list(predictions)
    y_score = _to_list(scores)
    return {
        "accuracy": float(accuracy(y_true, y_pred)),
        "auc": float(auc(y_true, y_score)),
        "precision": float(precision(y_true, y_pred)),
        "recall": float(recall(y_true, y_pred)),
        "f1": float(f1(y_true, y_pred)),
    }
