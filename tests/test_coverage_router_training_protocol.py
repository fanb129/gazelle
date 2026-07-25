from argparse import Namespace

import pytest

pytest.importorskip("torch")

from scripts.train_coverage_router import (
    selection_improved,
    selection_spec,
    validate_checkpoint_split_provenance,
)


def _selection_args(**overrides):
    values = {
        "dataset": "gazefollow",
        "router_stage": "backbone_sparse",
        "heatmap_loss_weight": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def _formal_split(**overrides):
    values = {
        "selection_is_formal": True,
        "strategy": "stable_sha256_group_holdout_v1",
        "evaluation_split": "gazefollow_train_holdout",
        "source_annotation_sha256": "source",
        "source_group_fingerprint": "groups",
        "assignment_fingerprint": "assignment",
        "validation_fraction": 0.1,
        "seed": 3106,
    }
    values.update(overrides)
    return values


def test_gaze_training_selects_one_checkpoint_by_validation_avg_l2():
    assert selection_spec(_selection_args()) == (
        "avg_l2",
        "min",
        (("auc", "max"),),
    )


def test_router_only_support_selects_by_validation_coverage():
    assert selection_spec(
        _selection_args(router_stage="support_pilot", heatmap_loss_weight=0.0)
    ) == (
        "routing_hard_coverage",
        "max",
        (("routing_gt_point_coverage", "max"), ("auc", "max")),
    )


def test_selection_uses_tie_breaker_and_keeps_earliest_exact_tie():
    selection = {
        "metric": "avg_l2",
        "mode": "min",
        "value": 0.1,
        "tie_breakers": [("auc", "max")],
        "metrics": {"avg_l2": 0.1, "auc": 0.95},
    }

    assert selection_improved({"avg_l2": 0.09, "auc": 0.90}, selection)
    assert selection_improved({"avg_l2": 0.1, "auc": 0.96}, selection)
    assert not selection_improved({"avg_l2": 0.1, "auc": 0.95}, selection)
    assert not selection_improved({"avg_l2": 0.11, "auc": 0.99}, selection)


def test_formal_initialization_requires_exact_split_provenance():
    current = _formal_split()

    validate_checkpoint_split_provenance(
        {"data_split": dict(current)},
        current,
        checkpoint_role="initialization checkpoint",
    )

    with pytest.raises(ValueError, match="no data_split provenance"):
        validate_checkpoint_split_provenance(
            {},
            current,
            checkpoint_role="initialization checkpoint",
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_split_provenance(
            {"data_split": _formal_split(assignment_fingerprint="other")},
            current,
            checkpoint_role="initialization checkpoint",
        )


def test_legacy_exploratory_runs_do_not_require_split_provenance():
    validate_checkpoint_split_provenance(
        {},
        {"selection_is_formal": False},
        checkpoint_role="legacy checkpoint",
    )
