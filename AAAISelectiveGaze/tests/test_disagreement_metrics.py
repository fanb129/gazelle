import numpy as np
import pytest

from AAAISelectiveGaze.metrics.disagreement import (
    disagreement_features,
    geometry_semantics_conflict,
    heatmap_statistics,
    js_divergence,
    normalize_heatmaps,
    pairwise_mean_js,
    pairwise_peak_distance,
    peak_coordinates,
)


def _one_hot(y: int, x: int, size: int = 3) -> np.ndarray:
    heatmap = np.zeros((size, size), dtype=np.float64)
    heatmap[y, x] = 1.0
    return heatmap


def test_normalization_handles_scale_zero_maps_and_roundoff() -> None:
    heatmaps = np.array(
        [
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[-1e-14, 1.0], [0.0, 0.0]],
        ]
    )
    normalized = normalize_heatmaps(heatmaps)
    np.testing.assert_allclose(normalized.sum(axis=(-2, -1)), 1.0)
    np.testing.assert_allclose(normalized[1], 0.25)
    assert np.all(normalized >= 0)


def test_normalization_rejects_invalid_heatmaps() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalize_heatmaps([[[-0.1, 1.1]]])
    with pytest.raises(ValueError, match="finite"):
        normalize_heatmaps([[[np.nan]]])
    with pytest.raises(ValueError, match="at least two dimensions"):
        normalize_heatmaps([1.0, 2.0])


def test_peak_coordinates_are_xy_and_normalized() -> None:
    heatmaps = np.zeros((2, 3, 5), dtype=np.float64)
    heatmaps[0, 2, 4] = 1.0
    heatmaps[1, 1, 2] = 1.0
    np.testing.assert_allclose(peak_coordinates(heatmaps), [[1.0, 1.0], [0.5, 0.5]])


def test_pairwise_peak_distance_uses_all_pairs() -> None:
    layers = np.stack([_one_hot(0, 0), _one_hot(0, 2), _one_hot(2, 2)])
    distances = pairwise_peak_distance(layers, reduction="none")
    np.testing.assert_allclose(distances, [1.0, np.sqrt(2.0), 1.0])
    assert pairwise_peak_distance(layers) == pytest.approx((2.0 + np.sqrt(2.0)) / 3.0)


def test_js_divergence_identities_and_pairwise_mean() -> None:
    left = _one_hot(1, 0)
    right = _one_hot(1, 2)
    assert js_divergence(left, left) == pytest.approx(0.0)
    assert js_divergence(left, right) == pytest.approx(np.log(2.0))
    layers = np.stack([left, left * 7.0, right])
    assert pairwise_mean_js(layers) == pytest.approx(2.0 * np.log(2.0) / 3.0)


def test_heatmap_entropy_peak_and_margin() -> None:
    maps = np.stack([np.ones((2, 2)), _one_hot(0, 0, size=2)])
    stats = heatmap_statistics(maps)
    np.testing.assert_allclose(stats["entropy"], [1.0, 0.0])
    np.testing.assert_allclose(stats["peak"], [0.25, 1.0])
    np.testing.assert_allclose(stats["margin"], [0.0, 1.0])


def test_geometry_semantics_conflict_detects_supported_and_conflicting_peaks() -> None:
    geometry = np.stack([_one_hot(1, 0), _one_hot(1, 0)])[None, ...]
    aligned_semantics = np.stack([_one_hot(1, 0)])[None, ...]
    aligned = geometry_semantics_conflict(geometry, aligned_semantics)
    assert aligned["peak_displacement"][0] == pytest.approx(0.0)
    assert not bool(aligned["semantic_peak_outside_geometry_support"][0])
    assert aligned["conflict_score"][0] == pytest.approx(0.0)

    conflicting_semantics = np.stack([_one_hot(1, 2)])[None, ...]
    conflict = geometry_semantics_conflict(geometry, conflicting_semantics)
    assert conflict["peak_displacement"][0] == pytest.approx(1.0)
    assert bool(conflict["semantic_peak_outside_geometry_support"][0])
    assert conflict["semantic_peak_geometry_support_ratio"][0] == pytest.approx(0.0)
    assert conflict["conflict_score"][0] > 0.8


def test_disagreement_bundle_preserves_batch_and_layer_shapes() -> None:
    sample = np.stack([_one_hot(0, 0), _one_hot(2, 2)])
    batched = np.stack([sample, sample])
    features = disagreement_features(batched)
    assert features["pairwise_peak_distance_mean"].shape == (2,)
    assert features["pairwise_js_mean"].shape == (2,)
    assert features["layer_entropy"].shape == (2, 2)


def test_optional_torch_tensor_input_is_detached_and_converted() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    result = peak_coordinates(tensor)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [[1.0, 0.0]])


def test_pairwise_metrics_reject_missing_or_spatial_layer_axis() -> None:
    with pytest.raises(ValueError, match="at least two layer"):
        pairwise_mean_js(np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="spatial"):
        pairwise_peak_distance(np.ones((2, 3, 4)), layer_axis=-1)
    with pytest.raises(ValueError, match="out of bounds"):
        pairwise_peak_distance(np.ones((2, 3, 4)), layer_axis=4)
