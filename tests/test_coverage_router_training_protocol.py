from argparse import Namespace
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from scripts.train_coverage_router import (
    load_model_state,
    make_dataloaders,
    parse_args,
    selection_improved,
    selection_spec,
    validate_args,
    validate_checkpoint_split_provenance,
)


class _TinyRoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 1)


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


def test_vat_router_only_support_also_selects_by_validation_coverage():
    assert selection_spec(
        _selection_args(
            dataset="vat",
            router_stage="support_pilot",
            heatmap_loss_weight=0.0,
        )
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


def test_formal_holdout_can_explicitly_allow_legacy_pretrained_initialization():
    validate_checkpoint_split_provenance(
        {},
        _formal_split(),
        checkpoint_role="legacy pretrained initialization",
        allow_missing_provenance=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_split_provenance(
            {"data_split": _formal_split(assignment_fingerprint="other")},
            _formal_split(),
            checkpoint_role="mismatched formal initialization",
            allow_missing_provenance=True,
        )


def test_router_can_be_reinitialized_while_other_checkpoint_weights_load():
    source = _TinyRoutingModel()
    target = _TinyRoutingModel()
    with torch.no_grad():
        source.router.weight.fill_(7.0)
        source.router.bias.fill_(8.0)
        source.decoder.weight.fill_(9.0)
        source.decoder.bias.fill_(10.0)

    payload = {"model_state": source.state_dict()}
    original_keys = tuple(payload["model_state"])
    initial_router = {
        key: value.clone()
        for key, value in target.state_dict().items()
        if key.startswith("router.")
    }

    load_model_state(
        target,
        payload,
        initialization=True,
        excluded_prefixes=("router.",),
    )

    target_state = target.state_dict()
    assert tuple(payload["model_state"]) == original_keys
    assert torch.equal(target_state["router.weight"], initial_router["router.weight"])
    assert torch.equal(target_state["router.bias"], initial_router["router.bias"])
    assert torch.equal(
        target_state["decoder.weight"], payload["model_state"]["decoder.weight"]
    )
    assert torch.equal(
        target_state["decoder.bias"], payload["model_state"]["decoder.bias"]
    )


def test_default_initialization_and_resume_still_load_router_weights():
    source = _TinyRoutingModel()
    initialized = _TinyRoutingModel()
    resumed = _TinyRoutingModel()

    load_model_state(initialized, source.state_dict(), initialization=True)
    load_model_state(resumed, source.state_dict(), initialization=False)

    for key, expected in source.state_dict().items():
        assert torch.equal(initialized.state_dict()[key], expected)
        assert torch.equal(resumed.state_dict()[key], expected)

    with pytest.raises(ValueError, match="only allowed for initialization"):
        load_model_state(
            resumed,
            source.state_dict(),
            initialization=False,
            excluded_prefixes=("router.",),
        )


def test_router_reinitialization_uses_training_seed_but_keeps_common_decoder():
    source = _TinyRoutingModel()
    payload = {"model_state": source.state_dict()}

    torch.manual_seed(3107)
    seed_3107 = _TinyRoutingModel()
    torch.manual_seed(3108)
    seed_3108 = _TinyRoutingModel()

    for model in (seed_3107, seed_3108):
        load_model_state(
            model,
            payload,
            initialization=True,
            excluded_prefixes=("router.",),
        )

    assert not torch.equal(
        seed_3107.router.weight,
        seed_3108.router.weight,
    )
    assert torch.equal(seed_3107.decoder.weight, source.decoder.weight)
    assert torch.equal(seed_3108.decoder.weight, source.decoder.weight)


def test_router_reinitialization_flag_requires_initialization_checkpoint():
    valid = parse_args(
        ["--init_ckpt", "dense.pt", "--reinitialize_router_on_init"]
    )
    validate_args(valid)

    without_init = parse_args(["--reinitialize_router_on_init"])
    with pytest.raises(ValueError, match="requires --init_ckpt"):
        validate_args(without_init)

    with_resume = parse_args(
        ["--resume", "partial.pt", "--reinitialize_router_on_init"]
    )
    with pytest.raises(ValueError, match="requires --init_ckpt"):
        validate_args(with_resume)


def test_vat_holdout_is_formal_and_has_no_source_video_leakage(tmp_path):
    sequences = []
    for source_video_index in range(4):
        for clip_index in range(2):
            clip_path = f"images/video_{source_video_index}/clip_{clip_index}"
            frames = [
                {
                    "path": f"{clip_path}/{frame_index:08d}.jpg",
                    "heads": [{"inout": 1}],
                }
                for frame_index in range(2)
            ]
            sequences.append({"path": clip_path, "frames": frames})
    (tmp_path / "train_preprocessed.json").write_text(
        json.dumps(sequences), encoding="utf-8"
    )

    args = Namespace(
        dataset="vat",
        data_path=str(tmp_path),
        n_workers=0,
        batch_size=2,
        eval_batch_size=2,
        frame_sample_every=2,
        eval_frame_sample_every=2,
        vat_val_fraction=0.5,
        vat_split_seed=3106,
        gazefollow_val_fraction=0.0,
        gazefollow_split_seed=3106,
    )
    train_dataset, eval_dataset, _, _, split = make_dataloaders(
        args, lambda image: image
    )

    train_source_videos = {
        record["path"].split("/", 2)[1] for record in train_dataset.data
    }
    validation_source_videos = {
        record["path"].split("/", 2)[1] for record in eval_dataset.data
    }
    assert train_source_videos.isdisjoint(validation_source_videos)
    assert len(train_source_videos) == 2
    assert len(validation_source_videos) == 2
    assert split["selection_is_formal"] is True
    assert split["evaluation_split"] == "vat_train_source_video_holdout"
    assert split["group_key"] == "normalized_source_video_path"
    assert split["train_sequence_count"] == 4
    assert split["validation_sequence_count"] == 4
