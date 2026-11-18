"""Train-only, one-parameter state calibration with a fixed output ordering.

The bounded log-odds offset is applied after the frozen causal rollout. Its
float64 outputs are not fed back into the model's prediction history. Exact
probability endpoints stay fixed; no input clipping changes scores or ties.
"""

import numpy as np
from sklearn.metrics import average_precision_score


CALIBRATION_VERSION = "r4_state_calibration_v1"
BIAS_BOUNDS = (-16.0, 16.0)
BISECTION_ITERATIONS = 80


def _scores(values):
    scores = np.asarray(values, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("Calibration requires a nonempty vector of finite probabilities")
    return scores


def _labels(values, shape):
    labels = np.asarray(values, dtype=np.float64)
    if labels.shape != shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("Calibration labels must be aligned binary state labels")
    return labels


def apply_state_bias(scores, bias):
    """Map p to p*exp(b)/(1-p+p*exp(b)), retaining 0/1 and all float32 ranks.

    The predeclared [-16, 16] bound leaves enough float64 resolution between
    distinct float32 scores close to one. The two branches avoid cancellation
    for probabilities close to zero and one, respectively.
    """
    scores = _scores(scores)
    if not np.isfinite(bias) or not BIAS_BOUNDS[0] <= bias <= BIAS_BOUNDS[1]:
        raise ValueError("State calibration bias must be finite and within [-16, 16]")
    multiplier = np.exp(bias)
    denominator = (1 - scores) + scores * multiplier
    calibrated = np.empty_like(scores)
    lower = scores <= 0.5
    calibrated[lower] = scores[lower] * multiplier / denominator[lower]
    calibrated[~lower] = 1 - (1 - scores[~lower]) / denominator[~lower]
    return calibrated


def _loss_summary(scores, labels, logits, interior, bias):
    shifted = logits + bias
    # This logistic loss avoids subtracting nearly equal positive logits.
    loss = np.logaddexp(0, np.where(labels[interior] == 1, -shifted, shifted))
    variable_nll = float(loss.sum() / len(scores))
    endpoint_errors = int(np.count_nonzero((~interior) & (scores != labels)))
    return {
        "person_frames": len(scores),
        "interior_frames": int(interior.sum()),
        "fixed_endpoint_frames": int((~interior).sum()),
        "uncalibratable_endpoint_errors": endpoint_errors,
        "full_nll": None if endpoint_errors else variable_nll,
        "full_nll_finite": endpoint_errors == 0,
        "variable_nll": variable_nll,
        "variable_nll_normalization": "all_train_person_frames",
    }


def fit_state_bias(scores, labels):
    """Fit the bounded offset using train state labels and unweighted BCE.

    A wrong fixed endpoint contributes +infinity independent of the offset.
    Its count is explicit, full NLL is then null, and the optimizer minimizes
    the remaining finite, bias-dependent NLL divided by all train frames. No
    clipping creates an artificial finite loss for such endpoints.
    """
    scores = _scores(scores)
    labels = _labels(labels, scores.shape)
    interior = (scores > 0) & (scores < 1)
    logits = np.log(scores[interior]) - np.log1p(-scores[interior])

    def derivative(bias):
        calibrated = apply_state_bias(scores[interior], bias)
        return float(np.sum(calibrated - labels[interior]) / len(scores))

    lower, upper = BIAS_BOUNDS
    iterations = 0
    if not interior.any():
        bias, status, final_derivative = 0.0, "no_adjustable_scores", 0.0
    elif derivative(lower) >= 0:
        bias, status = lower, "lower_bound"
        final_derivative = derivative(bias)
    elif derivative(upper) <= 0:
        bias, status = upper, "upper_bound"
        final_derivative = derivative(bias)
    else:
        for _ in range(BISECTION_ITERATIONS):
            midpoint = (lower + upper) / 2
            if derivative(midpoint) > 0:
                upper = midpoint
            else:
                lower = midpoint
        bias, status = (lower + upper) / 2, "interior_optimum"
        iterations = BISECTION_ITERATIONS
        final_derivative = derivative(bias)
    return {
        "bias": float(bias),
        "fit_split": "train",
        "objective": "unweighted_BCE_finite_bias_dependent_component",
        "parameterization": "p*exp(bias)/(1-p+p*exp(bias)); endpoints_fixed",
        "bias_bounds": list(BIAS_BOUNDS),
        "optimizer": {"name": "bounded_monotone_derivative_bisection",
                      "iterations": iterations, "status": status,
                      "derivative_per_all_frames": final_derivative},
        "train_before": _loss_summary(scores, labels, logits, interior, 0.0),
        "train_after": _loss_summary(scores, labels, logits, interior, bias),
    }


def ranking_checks(original, calibrated, labels):
    """Check exact ordering, ties and AP independently of threshold metrics."""
    original = _scores(original)
    calibrated = _scores(calibrated)
    labels = _labels(labels, original.shape)
    if calibrated.shape != original.shape:
        raise ValueError("Calibrated scores must have the original frame alignment")
    order = np.argsort(original, kind="stable")
    original_steps = np.diff(original[order])
    calibrated_steps = np.diff(calibrated[order])
    ap_before = float(average_precision_score(labels, original))
    ap_after = float(average_precision_score(labels, calibrated))
    return {
        "checks": {
            "strict_order_preserved": bool((calibrated_steps[original_steps > 0] > 0).all()),
            "original_ties_preserved": bool((calibrated_steps[original_steps == 0] == 0).all()),
            "average_precision_unchanged": ap_before == ap_after,
        },
        "inout_ap_before": ap_before,
        "inout_ap_after": ap_after,
        "unique_scores_before": len(np.unique(original)),
        "unique_scores_after": len(np.unique(calibrated)),
    }
