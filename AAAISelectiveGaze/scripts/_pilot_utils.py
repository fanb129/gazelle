"""Shared, non-neural utilities for the one-day disagreement pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from AAAISelectiveGaze.data.prediction_cache import (
    attach_risk_targets,
    heatmap_peak,
    make_sample_id,
)
from AAAISelectiveGaze.metrics.disagreement import (
    disagreement_features,
    geometry_semantics_conflict,
    heatmap_statistics,
    normalize_heatmaps,
)


FINAL_FEATURES = ("final_entropy", "final_negative_peak", "final_negative_margin", "final_spread")
DISAGREEMENT_FEATURES = (
    "pairwise_peak_distance_mean",
    "pairwise_peak_distance_max",
    "pairwise_js_mean",
    "pairwise_js_max",
    "geometry_semantics_conflict",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _spread(heatmap: Any) -> float:
    probability = normalize_heatmaps(heatmap)
    height, width = probability.shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij"
    )
    mean_x = float(np.sum(probability * xx))
    mean_y = float(np.sum(probability * yy))
    return float(np.sqrt(np.sum(probability * ((xx - mean_x) ** 2 + (yy - mean_y) ** 2))))


def records_to_rows(
    records: Sequence[Mapping[str, Any]], failure_l2_threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in attach_risk_targets(records, failure_l2_threshold):
        if record["localization_l2"] is None:
            continue
        layer_ids = sorted(record["probe_heatmaps"], key=lambda value: int(value))
        probes = np.stack([record["probe_heatmaps"][layer] for layer in layer_ids])
        disagreement = disagreement_features(probes)
        if len(layer_ids) >= 3:
            # ViT-B pilot contract: middle layers (5, 8) provide the putative
            # geometry evidence and the deepest layer (11) the semantic target.
            # Layer 2 remains in generic disagreement but not in this specific
            # conflict feature, so a weak shallow probe cannot define geometry.
            geometry_maps, semantic_maps = probes[1:-1], probes[-1:]
        else:
            geometry_maps, semantic_maps = probes[:1], probes[1:]
        conflict = geometry_semantics_conflict(geometry_maps, semantic_maps)
        final_stats = heatmap_statistics(record["final_heatmap"])
        gt = np.asarray(record["gt_gaze"], dtype=float)
        gt = gt[(gt[:, 0] >= 0) & (gt[:, 1] >= 0)]
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "image_path": record["image_path"],
            "person_index": record["person_index"],
            "localization_l2": record["localization_l2"],
            "failure": record["localization_failure"],
            "final_entropy": float(final_stats["entropy"]),
            "final_negative_peak": -float(final_stats["peak"]),
            "final_negative_margin": -float(final_stats["margin"]),
            "final_spread": _spread(record["final_heatmap"]),
            "pairwise_peak_distance_mean": float(disagreement["pairwise_peak_distance_mean"]),
            "pairwise_peak_distance_max": float(disagreement["pairwise_peak_distance_max"]),
            "pairwise_js_mean": float(disagreement["pairwise_js_mean"]),
            "pairwise_js_max": float(disagreement["pairwise_js_max"]),
            "geometry_semantics_conflict": float(conflict["conflict_score"]),
            "semantic_peak_outside_geometry_support": int(
                conflict["semantic_peak_outside_geometry_support"]
            ),
        }
        for layer, heatmap in zip(layer_ids, probes):
            peak = np.asarray(heatmap_peak(heatmap))
            row[f"probe_l2_{layer}"] = float(
                np.linalg.norm(gt - peak[None, :], axis=1).min()
            )
        rows.append(row)
    if not rows:
        raise ValueError("no in-frame records with localization targets were found")
    return rows


def gaussian_heatmap(
    x: float, y: float, *, size: int = 32, sigma: float = 1.8
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center_x = np.clip(x, 0.0, 1.0) * (size - 1)
    center_y = np.clip(y, 0.0, 1.0) * (size - 1)
    return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))


def synthetic_records(
    *, samples: int = 160, seed: int = 3106, dataset: str = "synthetic"
) -> list[dict[str, Any]]:
    """Create non-trivial heatmaps where disagreement adds incremental signal."""

    rng = np.random.default_rng(seed)
    records = []
    layers = (2, 5, 8, 11)
    for index in range(samples):
        gt = rng.uniform(0.12, 0.88, size=2)
        hidden_difficulty = rng.beta(1.5, 2.0)
        confidence_noise = rng.normal(0.0, 0.18)
        semantic_mislead = rng.random() < (0.05 + 0.7 * hidden_difficulty)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-12
        final_offset = direction * (0.02 + 0.28 * hidden_difficulty * semantic_mislead)
        final_center = np.clip(gt + final_offset + rng.normal(0, 0.012, 2), 0.02, 0.98)
        final_sigma = float(np.clip(1.4 + 1.5 * (hidden_difficulty + confidence_noise), 0.8, 4.5))
        final = gaussian_heatmap(*final_center, sigma=final_sigma)

        probe_maps = {}
        for layer_position, layer in enumerate(layers):
            depth = layer_position / (len(layers) - 1)
            # Difficult cases retain geometry in middle layers while deep layers
            # follow the misleading final semantic target.
            center = (1.0 - depth) * gt + depth * final_center
            jitter = rng.normal(0.0, 0.012 + 0.018 * hidden_difficulty, size=2)
            probe_maps[str(layer)] = gaussian_heatmap(
                *np.clip(center + jitter, 0.02, 0.98), sigma=1.7
            )
        image_path = f"synthetic/{dataset}_{index:05d}.jpg"
        records.append(
            {
                "sample_id": make_sample_id(dataset, image_path, 0),
                "dataset": dataset,
                "split": "smoke",
                "image_path": image_path,
                "person_index": 0,
                "final_heatmap": final,
                "probe_heatmaps": probe_maps,
                "inout_probability": 0.95,
                "gt_gaze": [gt.tolist()],
                "gt_inout": 1,
                "bbox": [0.05, 0.1, 0.2, 0.32],
                "checkpoint_hash": "synthetic",
            }
        )
    return records


def feature_matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


__all__ = [
    "DISAGREEMENT_FEATURES",
    "FINAL_FEATURES",
    "feature_matrix",
    "json_safe",
    "records_to_rows",
    "synthetic_records",
    "write_json",
]
