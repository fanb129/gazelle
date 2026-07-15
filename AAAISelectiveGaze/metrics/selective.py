"""Metrics for selective prediction where larger scores mean higher risk."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


ArrayLike = Any


def _as_vector(value: ArrayLike, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _paired_vectors(
    risk_scores: ArrayLike, errors: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    scores = _as_vector(risk_scores, name="risk_scores")
    losses = _as_vector(errors, name="errors")
    if scores.shape != losses.shape:
        raise ValueError("risk_scores and errors must have the same length")
    if np.any(losses < 0):
        raise ValueError("errors must be non-negative")
    return scores, losses


def _binary_labels(labels: ArrayLike) -> np.ndarray:
    array = _as_vector(labels, name="failure_labels")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("failure_labels must contain only 0 and 1")
    return array.astype(np.int8)


def _labels_and_scores(
    failure_labels: ArrayLike, risk_scores: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    labels = _binary_labels(failure_labels)
    scores = _as_vector(risk_scores, name="risk_scores")
    if labels.shape != scores.shape:
        raise ValueError("failure_labels and risk_scores must have the same length")
    return labels, scores


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based ranks with average ranks for ties."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def failure_auroc(failure_labels: ArrayLike, risk_scores: ArrayLike) -> float:
    """Area under the ROC curve for failure detection.

    A larger risk score predicts failure.  If either class is absent, AUROC is
    mathematically undefined and this function returns ``nan``.
    """

    labels, scores = _labels_and_scores(failure_labels, risk_scores)
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = labels.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = ranks[positives].sum()
    statistic = rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


def failure_aupr(failure_labels: ArrayLike, risk_scores: ArrayLike) -> float:
    """Average precision (area under stepwise precision-recall curve).

    Tied scores are processed as a group, making the result independent of
    sample order.  With no positive failures the metric is undefined and
    returns ``nan``; with all samples positive it is exactly one.
    """

    labels, scores = _labels_and_scores(failure_labels, risk_scores)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_total = np.arange(1, labels.size + 1)
    group_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_at_threshold = cumulative_true[group_ends]
    total_at_threshold = cumulative_total[group_ends]
    recalls = true_at_threshold / positive_count
    precisions = true_at_threshold / total_at_threshold
    recall_increments = np.diff(np.r_[0.0, recalls])
    return float(np.sum(recall_increments * precisions))


def risk_coverage_curve(
    risk_scores: ArrayLike, errors: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical coverage and retained mean risk curves.

    Samples are retained from lowest predicted risk to highest.  The returned
    coverages are ``1/N, 2/N, ..., 1``; no value is emitted for zero coverage
    because mean retained error would be undefined.  Equal risk scores preserve
    input order via stable sorting.
    """

    scores, losses = _paired_vectors(risk_scores, errors)
    order = np.argsort(scores, kind="stable")
    sorted_losses = losses[order]
    retained = np.arange(1, losses.size + 1, dtype=np.float64)
    coverages = retained / losses.size
    selective_risks = np.cumsum(sorted_losses) / retained
    return coverages, selective_risks


def risk_at_coverage(
    risk_scores: ArrayLike, errors: ArrayLike, coverage: float
) -> float:
    """Mean retained error at requested coverage.

    ``ceil(coverage * N)`` low-risk samples are retained, so actual empirical
    coverage is never lower than requested.  Coverage must lie in ``(0, 1]``.
    """

    if not np.isscalar(coverage) or not np.isfinite(coverage):
        raise ValueError("coverage must be a finite scalar in (0, 1]")
    coverage = float(coverage)
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    scores, losses = _paired_vectors(risk_scores, errors)
    retained_count = int(np.ceil(coverage * losses.size))
    order = np.argsort(scores, kind="stable")
    return float(np.mean(losses[order[:retained_count]]))


def aurc(risk_scores: ArrayLike, errors: ArrayLike) -> float:
    """Empirical area under the risk-coverage curve (lower is better).

    This uses the standard discrete estimator: the mean of selective risks at
    coverages ``1/N`` through ``1``.
    """

    _, selective_risks = risk_coverage_curve(risk_scores, errors)
    return float(np.mean(selective_risks))


def spearman_correlation(risk_scores: ArrayLike, errors: ArrayLike) -> float:
    """Spearman rank correlation with average tie ranks.

    Returns ``nan`` when either input is constant because correlation is then
    undefined.
    """

    scores, losses = _paired_vectors(risk_scores, errors)
    score_ranks = _average_ranks(scores)
    loss_ranks = _average_ranks(losses)
    score_centered = score_ranks - score_ranks.mean()
    loss_centered = loss_ranks - loss_ranks.mean()
    denominator = np.sqrt(
        np.sum(score_centered**2) * np.sum(loss_centered**2)
    )
    if denominator == 0:
        return float("nan")
    return float(np.sum(score_centered * loss_centered) / denominator)


def selective_metrics(
    risk_scores: ArrayLike,
    errors: ArrayLike,
    *,
    failure_labels: ArrayLike | None = None,
    failure_threshold: float | None = None,
    coverages: Iterable[float] = (0.5, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, Any]:
    """Compute the metric bundle used by the one-day pilot.

    Supply exactly one of ``failure_labels`` or ``failure_threshold``.  The
    latter derives failures as ``errors > failure_threshold``.
    """

    scores, losses = _paired_vectors(risk_scores, errors)
    if (failure_labels is None) == (failure_threshold is None):
        raise ValueError(
            "supply exactly one of failure_labels or failure_threshold"
        )
    if failure_threshold is not None:
        if not np.isscalar(failure_threshold) or not np.isfinite(failure_threshold):
            raise ValueError("failure_threshold must be a finite non-negative scalar")
        if float(failure_threshold) < 0:
            raise ValueError("failure_threshold must be non-negative")
        labels = (losses > float(failure_threshold)).astype(np.int8)
    else:
        labels = _binary_labels(failure_labels)
        if labels.shape != losses.shape:
            raise ValueError("failure_labels and errors must have the same length")

    requested_coverages = [float(value) for value in coverages]
    if not requested_coverages:
        raise ValueError("coverages must not be empty")
    risks_by_coverage = {
        str(coverage): risk_at_coverage(scores, losses, coverage)
        for coverage in requested_coverages
    }
    curve_coverages, curve_risks = risk_coverage_curve(scores, losses)
    return {
        "failure_auroc": failure_auroc(labels, scores),
        "failure_aupr": failure_aupr(labels, scores),
        "spearman": spearman_correlation(scores, losses),
        "aurc": float(np.mean(curve_risks)),
        "risk_at_coverage": risks_by_coverage,
        "coverage_curve": curve_coverages,
        "risk_curve": curve_risks,
    }
