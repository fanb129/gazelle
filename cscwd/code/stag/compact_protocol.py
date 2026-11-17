"""Predeclared compact R4 budget, checkpoint ranking and Pareto reporting."""


PLAN_VERSION = "r4_compact_v2"
COMPACT_BUDGET = {
    "epochs": 5, "learning_rate": 0.0001, "lambda_state": 0.1,
    "hidden_dim": 128, "clip_frames": 16, "batch_size": 8, "seed": 0,
    "weight_decay": 0.01, "gradient_clip_norm": 1.0, "temperature": 0.1,
}

PARETO_AXES = {
    "auc": 1, "inout_ap": 1, "accurate_stable_ratio": 1,
    "joint_gaze_shift_response_rate": 1, "l2": -1, "state_error_rate": -1,
}


def validate_budget(config):
    """Require the fixed first-round budget for every trainable variant."""
    differences = [f"{name}={config.get(name)!r} (expected {expected!r})"
                   for name, expected in COMPACT_BUDGET.items()
                   if config.get(name) != expected]
    if differences:
        raise ValueError("Compact training budget mismatch: " + "; ".join(differences))


def _metrics(validation):
    standard, temporal, state = (validation[name] for name in ("standard", "temporal", "state"))
    return {
        "auc": standard["auc"], "l2": standard["l2"], "inout_ap": standard["inout_ap"],
        "jitter_mean": temporal["jitter"]["predicted"]["mean"],
        "accurate_stable_ratio": temporal["stability"]["accurate_stable_ratio"],
        "state_error_rate": state["state_error"]["error_rate"],
        "state_flip_rate": state["stable_state_flips"]["flip_rate"],
        **{f"joint_{kind}_response_rate": temporal["response"][kind]["response_rate"]
           for kind in ("enter_frame", "leave_frame", "gaze_shift")},
        **{f"state_{kind}_response_rate": state["response"][kind]["response_rate"]
           for kind in ("all", "enter_frame", "leave_frame")},
    }


def make_candidate(model_id, epoch, validation, baseline):
    """Flatten validation metrics and apply the three precision tolerances.

    Response rates are reported and ranked, without the R1 per-class hard veto.
    ``baseline`` uses the same nested metric schema as ``validation``.
    """
    metrics, reference = _metrics(validation), baseline["standard"]
    checks = {}
    for name, direction in (("auc", 1), ("l2", -1), ("inout_ap", 1)):
        value, original = metrics[name], reference[name]
        checks[name] = (value is not None and original is not None
                        and direction * value >= direction * original - 0.002)
    return {"candidate_id": f"{model_id}:epoch_{epoch:03d}",
            "model_id": model_id, "epoch": epoch, "metrics": metrics,
            "precision_eligible": all(checks.values()), "precision_checks": checks}


def _descending(value):
    return float("inf") if value is None else -value


def select_checkpoint(candidates):
    """Select an eligible checkpoint; undefined response ranks after defined."""
    eligible = [item for item in candidates if item["precision_eligible"]]
    return min(eligible, key=lambda item: (
        _descending(item["metrics"]["accurate_stable_ratio"]),
        _descending(item["metrics"]["joint_gaze_shift_response_rate"]),
        _descending(item["metrics"]["inout_ap"]), item["metrics"]["l2"],
        _descending(item["metrics"]["auc"]), item["epoch"], item["model_id"],
    ), default=None)


def _dominates(first, second):
    strictly_better = False
    for name, direction in PARETO_AXES.items():
        a, b = first["metrics"][name], second["metrics"][name]
        if a is None or b is None:
            if a is not b:
                return False
            continue
        if direction * a < direction * b:
            return False
        strictly_better |= direction * a > direction * b
    return strictly_better


def pareto_front(candidates):
    """Return nondominated IDs, including ineligible and single-frame rows.

    If both values are undefined, that axis supplies no evidence. If only one
    is undefined, neither row can establish dominance over the other.
    """
    return sorted(item["candidate_id"] for item in candidates
                  if not any(_dominates(other, item) for other in candidates))
