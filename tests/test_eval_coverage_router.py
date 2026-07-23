from argparse import Namespace

import pytest

pytest.importorskip("torch")

from scripts.eval_coverage_router import apply_eval_overrides, validate_args
from scripts.train_coverage_router import current_keep_ratio


def _args(**overrides):
    values = {
        "batch_size": 16,
        "n_workers": 8,
        "frame_sample_every": 1,
        "max_eval_batches": None,
        "router_stage_override": None,
        "keep_ratio_override": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_apply_eval_overrides_does_not_mutate_checkpoint_config():
    checkpoint_config = {
        "router_stage": "support_pilot",
        "keep_ratio": 0.25,
        "model": "gazelle_dinov3_vitb16",
    }

    effective = apply_eval_overrides(
        checkpoint_config,
        _args(router_stage_override="backbone_sparse", keep_ratio_override=0.5),
    )

    assert effective["router_stage"] == "backbone_sparse"
    assert effective["keep_ratio"] == 0.5
    assert checkpoint_config["router_stage"] == "support_pilot"
    assert checkpoint_config["keep_ratio"] == 0.25


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.01])
def test_validate_args_rejects_invalid_keep_ratio_override(keep_ratio):
    with pytest.raises(ValueError, match="keep_ratio_override"):
        validate_args(_args(keep_ratio_override=keep_ratio))


def test_keep_ratio_curriculum_includes_dense_and_sparse_endpoints():
    ratios = [current_keep_ratio(0.25, epoch, 4) for epoch in range(5)]

    assert ratios == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.25])
