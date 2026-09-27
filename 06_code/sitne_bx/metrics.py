"""不依赖 scikit-learn 的确定性二分类指标与 MCC 阈值选择。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validated(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or y.size == 0:
        raise ValueError("labels/scores 必须为非空、同 shape 一维数组")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("labels 必须为 0/1 且 scores 必须有限")
    return y, s


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _validated(labels, scores)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC 要求正负类均存在")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _validated(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("AUPRC 要求至少一个正例")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s_sorted[end] == s_sorted[start]:
            end += 1
        group = y_sorted[start:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denominator)


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """最大化 MCC；并列依次选择更接近 0.5、再选择更高阈值。"""

    y, scores = _validated(labels, probabilities)
    if bool((scores < 0).any() or (scores > 1).any()):
        raise ValueError("threshold selection 需要 [0,1] 概率")
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    ss = scores[order]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    candidates: list[tuple[float, float]] = []
    all_negative_threshold = float(np.nextafter(float(ss[0]), math.inf))
    candidates.append((_mcc(0, negatives, 0, positives), all_negative_threshold))
    tp = fp = 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ss[end] == ss[start]:
            end += 1
        group = ys[start:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        fn = positives - tp
        tn = negatives - fp
        candidates.append((_mcc(tp, tn, fp, fn), float(ss[start])))
        start = end
    best_mcc, best_threshold = max(
        candidates,
        key=lambda value: (value[0], -abs(value[1] - 0.5), value[1]),
    )
    return best_threshold, best_mcc


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    y, scores = _validated(labels, probabilities)
    if not math.isfinite(threshold):
        raise ValueError("threshold 必须有限")
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "auroc": auroc(y, scores),
        "auprc": auprc(y, scores),
        "mcc": _mcc(tp, tn, fp, fn),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
