import pytest

torch = pytest.importorskip("torch")

from AAAICOTB.binding import BindingConfig, compute_frame_binding
from AAAICOTB.geometry import cluster_targets
from gazelle.utils import get_heatmap


def _inputs(swapped=False):
    left = get_heatmap(0.20, 0.25, 64, 64)
    right = get_heatmap(0.80, 0.75, 64, 64)
    heatmaps = torch.stack([right, left] if swapped else [left, right]).requires_grad_(True)
    targets = torch.tensor([[0.20, 0.25], [0.80, 0.75]])
    inout = torch.tensor([True, True])
    bboxes = [[0.05, 0.05, 0.15, 0.20], [0.65, 0.05, 0.75, 0.20]]
    return heatmaps, targets, inout, bboxes


def test_shared_target_clustering_is_single_link():
    labels, centers = cluster_targets([(0.1, 0.1), (0.14, 0.1), (0.18, 0.1), (0.8, 0.8)], 0.05)
    assert labels == [0, 0, 0, 1]
    assert len(centers) == 2


def test_correct_diagonal_has_positive_margin_and_zero_hinge():
    result = compute_frame_binding(*_inputs(swapped=False), BindingConfig(margin=0.2))
    assert result["pair_count"] == 1
    assert result["pairs"][0]["diagonal_correct"] == 1.0
    assert result["pairs"][0]["diag_margin"] > 0.2
    assert float(result["loss"].detach()) == 0.0


def test_swapped_predictions_are_falsifiable_and_differentiable():
    heatmaps, targets, inout, bboxes = _inputs(swapped=True)
    result = compute_frame_binding(heatmaps, targets, inout, bboxes, BindingConfig(margin=0.2))
    assert result["pairs"][0]["swap_error"] == 1.0
    assert float(result["loss"].detach()) > 0.2
    result["loss"].backward()
    assert heatmaps.grad is not None
    assert float(heatmaps.grad.abs().sum()) > 0


def test_shared_target_pair_is_not_a_negative():
    heatmaps = torch.stack([get_heatmap(0.5, 0.5, 64, 64), get_heatmap(0.51, 0.5, 64, 64)])
    result = compute_frame_binding(
        heatmaps,
        torch.tensor([[0.5, 0.5], [0.51, 0.5]]),
        torch.tensor([True, True]),
        [[0.1, 0.1, 0.2, 0.2], [0.7, 0.1, 0.8, 0.2]],
        BindingConfig(shared_target_radius=0.06),
    )
    assert result["cluster_count"] == 1
    assert result["pair_count"] == 0


def test_likely_duplicate_head_tracks_are_excluded():
    heatmaps, targets, inout, _ = _inputs(swapped=False)
    result = compute_frame_binding(
        heatmaps,
        targets,
        inout,
        [[0.1, 0.1, 0.3, 0.3], [0.11, 0.11, 0.31, 0.31]],
        BindingConfig(duplicate_bbox_iou=0.8),
    )
    assert result["pair_count"] == 0
