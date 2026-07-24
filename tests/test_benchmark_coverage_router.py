from argparse import Namespace

import pytest

pytest.importorskip("torch")
import torch

from scripts.benchmark_coverage_router import (
    equalize_autocast_weight_cache,
    make_bboxes,
    percentile,
    validate_args,
)


def _args(**overrides):
    values = {
        "variants": ["dense", "k25"],
        "batch_size": 1,
        "num_people": 1,
        "image_size": 512,
        "warmup_iters": 0,
        "latency_iters": 2,
        "throughput_iters": 2,
        "repeats": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_percentile_interpolates_sorted_values():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == pytest.approx(3.85)


def test_make_bboxes_matches_batch_and_people_counts():
    bboxes = make_bboxes(batch_size=2, num_people=3)

    assert len(bboxes) == 2
    assert all(len(image_boxes) == 3 for image_boxes in bboxes)
    assert all(
        0.0 <= coordinate <= 1.0
        for image_boxes in bboxes
        for box in image_boxes
        for coordinate in box
    )


def test_validate_args_rejects_duplicate_variants():
    with pytest.raises(ValueError, match="duplicates"):
        validate_args(_args(variants=["k25", "k25"]))


def test_equalize_autocast_weight_cache_makes_variants_symmetric():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)

    metadata = equalize_autocast_weight_cache(model)

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert metadata["autocast_cache_eligible_parameters"] == metadata["total_parameters"]
