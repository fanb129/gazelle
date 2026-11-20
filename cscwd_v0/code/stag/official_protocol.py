"""Declared full-train/per-epoch-test protocol and whole-checkpoint ranking."""

from .compact_protocol import COMPACT_BUDGET, _metrics


PROTOCOL_VERSION = "r4_official_train_test_v1"
TRAINING_BUDGET = dict(COMPACT_BUDGET)
SELECTION_RULE = "test_auc_desc; test_l2_asc; test_inout_ap_desc; earlier_epoch; model_id"


def validate_budget(config):
    differences = [f"{name}={config.get(name)!r} (expected {expected!r})"
                   for name, expected in TRAINING_BUDGET.items()
                   if config.get(name) != expected]
    if differences:
        raise ValueError("Official training budget mismatch: " + "; ".join(differences))


def make_candidate(model_id, epoch, evaluation):
    """Record one epoch's complete test metrics without precision eligibility."""
    return {"candidate_id": f"{model_id}:epoch_{epoch:03d}",
            "model_id": model_id, "epoch": epoch, "metrics": _metrics(evaluation)}


def select_checkpoint(candidates):
    """Choose one complete row by AUC, L2, AP, earlier epoch, then model ID."""
    return min(candidates, key=lambda item: (
        -item["metrics"]["auc"], item["metrics"]["l2"],
        -item["metrics"]["inout_ap"], item["epoch"], item["model_id"],
    ), default=None)
