import numpy as np
import pytest

from AAAISelectiveGaze.metrics.selective import (
    aurc,
    failure_aupr,
    failure_auroc,
    risk_at_coverage,
    risk_coverage_curve,
    selective_metrics,
    spearman_correlation,
)


def test_failure_detection_is_perfect_when_failures_have_high_risk() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert failure_auroc(labels, scores) == pytest.approx(1.0)
    assert failure_aupr(labels, scores) == pytest.approx(1.0)


def test_failure_detection_handles_ties_without_order_dependence() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.ones(4)
    assert failure_auroc(labels, scores) == pytest.approx(0.5)
    assert failure_aupr(labels, scores) == pytest.approx(0.5)
    permutation = [3, 0, 2, 1]
    assert failure_aupr(labels[permutation], scores[permutation]) == pytest.approx(0.5)


def test_single_class_detection_boundaries_are_explicit() -> None:
    scores = [0.1, 0.2, 0.3]
    assert np.isnan(failure_auroc([0, 0, 0], scores))
    assert np.isnan(failure_auroc([1, 1, 1], scores))
    assert np.isnan(failure_aupr([0, 0, 0], scores))
    assert failure_aupr([1, 1, 1], scores) == pytest.approx(1.0)


def test_risk_coverage_retains_low_risk_first() -> None:
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    errors = np.array([1.0, 0.0, 1.0, 0.0])
    coverage, selective_risk = risk_coverage_curve(scores, errors)
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(selective_risk, [0.0, 0.0, 1.0 / 3.0, 0.5])
    assert risk_at_coverage(scores, errors, 0.5) == pytest.approx(0.0)
    assert risk_at_coverage(scores, errors, 1.0) == pytest.approx(0.5)
    assert aurc(scores, errors) == pytest.approx((0.0 + 0.0 + 1.0 / 3.0 + 0.5) / 4.0)


def test_risk_at_coverage_rounds_up_to_reachable_coverage() -> None:
    scores = [0.1, 0.2, 0.3]
    errors = [0.0, 1.0, 1.0]
    assert risk_at_coverage(scores, errors, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        risk_at_coverage(scores, errors, 0.0)


def test_spearman_handles_ties_and_constant_inputs() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(spearman_correlation([1, 1, 1], [1, 2, 3]))


def test_selective_metric_bundle_derives_failure_labels() -> None:
    result = selective_metrics(
        [0.1, 0.2, 0.8, 0.9],
        [0.01, 0.02, 0.3, 0.4],
        failure_threshold=0.15,
        coverages=[0.5, 1.0],
    )
    assert result["failure_auroc"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["risk_at_coverage"]["0.5"] == pytest.approx(0.015)


@pytest.mark.parametrize(
    ("function", "args", "match"),
    [
        (failure_auroc, ([0, 2], [0.1, 0.2]), "only 0 and 1"),
        (failure_aupr, ([0, 1], [0.1]), "same length"),
        (risk_coverage_curve, ([0.1], [-1.0]), "non-negative"),
        (spearman_correlation, ([0.1, np.nan], [0.0, 1.0]), "finite"),
    ],
)
def test_strict_input_validation(function, args, match) -> None:
    with pytest.raises(ValueError, match=match):
        function(*args)
