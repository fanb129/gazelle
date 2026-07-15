"""Disagreement features for fixed layer gaze probes.

All public functions accept NumPy arrays and array-like objects.  CPU or CUDA
``torch.Tensor`` inputs are also accepted: tensors are detached and copied to
CPU before the metric is computed.  The last two dimensions always represent
``(height, width)``.
"""

from __future__ import annotations

from itertools import combinations
import operator
from typing import Any, Literal

import numpy as np


ArrayLike = Any
Reduction = Literal["mean", "max", "none"]


def _as_numpy(value: ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like value to a finite float64 NumPy array."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_heatmaps(value: ArrayLike, *, name: str = "heatmaps") -> np.ndarray:
    array = _as_numpy(value, name=name)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions (..., H, W)")
    if array.shape[-2] < 1 or array.shape[-1] < 1:
        raise ValueError(f"{name} spatial dimensions must be non-empty")
    return array


def _layer_first_last_three(
    heatmaps: ArrayLike, *, layer_axis: int, name: str
) -> np.ndarray:
    """Return shape ``(..., layers, H, W)`` with a validated layer axis."""

    array = _validate_heatmaps(heatmaps, name=name)
    if array.ndim < 3:
        raise ValueError(f"{name} must include a layer dimension")
    try:
        axis = operator.index(layer_axis)
    except TypeError as exc:
        raise ValueError("layer_axis must be an integer") from exc
    if axis < 0:
        axis += array.ndim
    if not 0 <= axis < array.ndim:
        raise ValueError(
            f"layer_axis {layer_axis} is out of bounds for an array with "
            f"{array.ndim} dimensions"
        )
    if axis >= array.ndim - 2:
        raise ValueError("layer_axis cannot refer to a spatial dimension")
    array = np.moveaxis(array, axis, -3)
    if array.shape[-3] < 1:
        raise ValueError(f"{name} must contain at least one layer")
    return array


def normalize_heatmaps(heatmaps: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """Normalize each heatmap over its spatial dimensions.

    Inputs are interpreted as non-negative heatmap mass, not logits.  Tiny
    negative round-off (down to ``-eps``) is clipped.  An all-zero map is
    converted to a uniform distribution, which keeps downstream entropy and JS
    metrics finite and deterministic.
    """

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    array = _validate_heatmaps(heatmaps)
    if np.any(array < -eps):
        raise ValueError("heatmaps must be non-negative probability mass")
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=(-2, -1), keepdims=True)
    zero = totals <= eps
    safe_totals = np.where(zero, 1.0, totals)
    normalized = array / safe_totals
    if np.any(zero):
        uniform = 1.0 / float(array.shape[-2] * array.shape[-1])
        normalized = np.where(zero, uniform, normalized)
    return normalized


def peak_coordinates(heatmaps: ArrayLike) -> np.ndarray:
    """Return normalized peak coordinates in ``(x, y)`` order.

    Coordinates lie in ``[0, 1]`` and use ``x / (W - 1)`` and
    ``y / (H - 1)``.  A singleton spatial dimension has coordinate zero.
    Ties follow NumPy's deterministic row-major ``argmax`` rule.
    """

    probabilities = normalize_heatmaps(heatmaps)
    height, width = probabilities.shape[-2:]
    flat = probabilities.reshape(*probabilities.shape[:-2], height * width)
    indices = np.argmax(flat, axis=-1)
    y = indices // width
    x = indices % width
    x_norm = x.astype(np.float64) / max(width - 1, 1)
    y_norm = y.astype(np.float64) / max(height - 1, 1)
    return np.stack((x_norm, y_norm), axis=-1)


def _reduce_pairs(values: np.ndarray, reduction: Reduction) -> np.ndarray:
    if reduction == "none":
        return values
    if reduction == "mean":
        return np.mean(values, axis=-1)
    if reduction == "max":
        return np.max(values, axis=-1)
    raise ValueError("reduction must be one of: 'mean', 'max', 'none'")


def pairwise_peak_distance(
    heatmaps: ArrayLike,
    *,
    layer_axis: int = -3,
    reduction: Reduction = "mean",
) -> np.ndarray:
    """Compute Euclidean distances between every pair of layer peaks.

    With ``reduction='none'``, the last output dimension follows lexicographic
    layer pairs ``(0, 1), (0, 2), ...``.  Distances use normalized coordinates
    and therefore range from zero to ``sqrt(2)``.
    """

    layered = _layer_first_last_three(
        heatmaps, layer_axis=layer_axis, name="heatmaps"
    )
    layer_count = layered.shape[-3]
    if layer_count < 2:
        raise ValueError("at least two layer heatmaps are required")
    coordinates = peak_coordinates(layered)
    distances = [
        np.linalg.norm(coordinates[..., i, :] - coordinates[..., j, :], axis=-1)
        for i, j in combinations(range(layer_count), 2)
    ]
    return _reduce_pairs(np.stack(distances, axis=-1), reduction)


def js_divergence(
    first: ArrayLike, second: ArrayLike, *, eps: float = 1e-12
) -> np.ndarray:
    """Jensen-Shannon divergence for corresponding heatmaps.

    Natural logarithms are used, so the result lies in ``[0, log(2)]`` up to
    floating-point error.  Leading dimensions must match exactly; implicit
    broadcasting is intentionally rejected.
    """

    first_array = _validate_heatmaps(first, name="first")
    second_array = _validate_heatmaps(second, name="second")
    if first_array.shape != second_array.shape:
        raise ValueError("first and second heatmaps must have identical shapes")
    p = normalize_heatmaps(first_array, eps=eps)
    q = normalize_heatmaps(second_array, eps=eps)
    mixture = 0.5 * (p + q)

    def kl_term(distribution: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(distribution)
        positive = distribution > 0
        terms[positive] = distribution[positive] * np.log(
            distribution[positive] / mixture[positive]
        )
        return terms.sum(axis=(-2, -1))

    result = 0.5 * (kl_term(p) + kl_term(q))
    return np.maximum(result, 0.0)


def pairwise_mean_js(
    heatmaps: ArrayLike,
    *,
    layer_axis: int = -3,
    reduction: Reduction = "mean",
) -> np.ndarray:
    """Compute JS divergence between every pair of layer heatmaps."""

    layered = _layer_first_last_three(
        heatmaps, layer_axis=layer_axis, name="heatmaps"
    )
    layer_count = layered.shape[-3]
    if layer_count < 2:
        raise ValueError("at least two layer heatmaps are required")
    divergences = [
        js_divergence(layered[..., i, :, :], layered[..., j, :, :])
        for i, j in combinations(range(layer_count), 2)
    ]
    return _reduce_pairs(np.stack(divergences, axis=-1), reduction)


def heatmap_statistics(
    heatmaps: ArrayLike, *, normalized_entropy: bool = True
) -> dict[str, np.ndarray]:
    """Return entropy, peak probability, and top-1/top-2 margin per map."""

    probabilities = normalize_heatmaps(heatmaps)
    pixel_count = probabilities.shape[-2] * probabilities.shape[-1]
    flattened = probabilities.reshape(*probabilities.shape[:-2], pixel_count)
    log_probabilities = np.zeros_like(flattened)
    positive = flattened > 0
    log_probabilities[positive] = np.log(flattened[positive])
    entropy = -np.sum(flattened * log_probabilities, axis=-1)
    if normalized_entropy and pixel_count > 1:
        entropy = entropy / np.log(pixel_count)
    peak = np.max(flattened, axis=-1)
    if pixel_count == 1:
        second = np.zeros_like(peak)
    else:
        second = np.partition(flattened, -2, axis=-1)[..., -2]
    return {
        "entropy": entropy,
        "peak": peak,
        "margin": peak - second,
    }


def _mean_layer_distribution(
    heatmaps: ArrayLike, *, layer_axis: int, name: str
) -> np.ndarray:
    layered = _layer_first_last_three(heatmaps, layer_axis=layer_axis, name=name)
    return normalize_heatmaps(layered).mean(axis=-3)


def geometry_semantics_conflict(
    geometry_heatmaps: ArrayLike,
    semantic_heatmaps: ArrayLike,
    *,
    layer_axis: int = -3,
    support_mass: float = 0.8,
) -> dict[str, np.ndarray]:
    """Measure conflict between geometry-layer and semantic-layer evidence.

    Each input has shape ``(..., layers, H, W)``.  Layers within each group are
    averaged after per-layer normalization.  The geometry support region is the
    smallest highest-density set whose cumulative probability reaches
    ``support_mass``.  ``conflict_score`` equally combines normalized peak
    displacement and whether the semantic peak falls outside that support.
    """

    if not np.isfinite(support_mass) or not 0 < support_mass <= 1:
        raise ValueError("support_mass must be in (0, 1]")
    geometry = _mean_layer_distribution(
        geometry_heatmaps, layer_axis=layer_axis, name="geometry_heatmaps"
    )
    semantics = _mean_layer_distribution(
        semantic_heatmaps, layer_axis=layer_axis, name="semantic_heatmaps"
    )
    if geometry.shape != semantics.shape:
        raise ValueError(
            "geometry_heatmaps and semantic_heatmaps must have matching batch "
            "and spatial shapes after layer reduction"
        )

    geometry_peaks = peak_coordinates(geometry)
    semantic_peaks = peak_coordinates(semantics)
    displacement = np.linalg.norm(geometry_peaks - semantic_peaks, axis=-1)

    pixel_count = geometry.shape[-2] * geometry.shape[-1]
    geometry_flat = geometry.reshape(-1, pixel_count)
    semantic_indices = np.argmax(semantics.reshape(-1, pixel_count), axis=-1)
    outside = np.empty(geometry_flat.shape[0], dtype=bool)
    support_ratio = np.empty(geometry_flat.shape[0], dtype=np.float64)
    for row_index, distribution in enumerate(geometry_flat):
        order = np.argsort(-distribution, kind="stable")
        sorted_values = distribution[order]
        cumulative_before = np.cumsum(sorted_values) - sorted_values
        support_indices = order[cumulative_before < support_mass]
        semantic_index = semantic_indices[row_index]
        outside[row_index] = not np.any(support_indices == semantic_index)
        geometry_peak = distribution[order[0]]
        support_ratio[row_index] = (
            distribution[semantic_index] / geometry_peak if geometry_peak > 0 else 0.0
        )

    prefix_shape = geometry.shape[:-2]
    outside = outside.reshape(prefix_shape)
    support_ratio = support_ratio.reshape(prefix_shape)
    conflict = 0.5 * (displacement / np.sqrt(2.0) + outside.astype(np.float64))
    return {
        "peak_displacement": displacement,
        "semantic_peak_outside_geometry_support": outside,
        "semantic_peak_geometry_support_ratio": support_ratio,
        "conflict_score": conflict,
    }


def disagreement_features(
    heatmaps: ArrayLike, *, layer_axis: int = -3
) -> dict[str, np.ndarray]:
    """Convenience bundle of layer disagreement and per-layer confidence."""

    layered = _layer_first_last_three(
        heatmaps, layer_axis=layer_axis, name="heatmaps"
    )
    statistics = heatmap_statistics(layered)
    return {
        "pairwise_peak_distance_mean": pairwise_peak_distance(layered),
        "pairwise_peak_distance_max": pairwise_peak_distance(
            layered, reduction="max"
        ),
        "pairwise_js_mean": pairwise_mean_js(layered),
        "pairwise_js_max": pairwise_mean_js(layered, reduction="max"),
        "layer_entropy": statistics["entropy"],
        "layer_peak": statistics["peak"],
        "layer_margin": statistics["margin"],
    }
