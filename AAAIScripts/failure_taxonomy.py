#!/usr/bin/env python3
"""Stratified failure and query-fidelity evaluation for Gazelle checkpoints.

This script intentionally lives outside the model package and does not modify the
Gazelle implementation.  It supports VAT and GazeFollow preprocessed JSON files.

Outputs (``<output_prefix>.*``):
  - records.csv: one row per gaze query/person
  - frames.csv: one row per frame, including heatmap-collapse diagnostics
  - summary.csv: stratified person- and frame-level aggregates
  - report.json: metadata, bin definitions, and the same aggregate results

No result is fabricated: normal execution requires a checkpoint and real data.
``--self-test`` only checks deterministic taxonomy/query-fidelity pure functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest, strict_load_task_checkpoint


PERSON_FIELDS = [
    "dataset", "model_label", "path", "person_index", "people_count", "crowd_bin",
    "head_size", "head_size_bin", "inout", "target_x", "target_y",
    "target_distance", "target_distance_bin", "target_cluster", "num_target_clusters",
    "target_separation", "target_separation_bin", "pred_x", "pred_y", "auc", "l2",
    "avg_l2", "min_l2", "inout_score", "target_confusion", "association_margin",
]

FRAME_FIELDS = [
    "dataset", "model_label", "path", "people_count", "inframe_count", "crowd_bin",
    "mean_head_size", "head_size_bin", "num_target_clusters", "shared_target_fraction",
    "target_separation", "target_separation_bin", "query_cosine_mean", "query_cosine_max",
    "distinct_target_cosine_mean", "distinct_target_cosine_max", "distinct_target_collapse_rate",
    "query_peak_distance_mean",
]


@dataclass(frozen=True)
class BinConfig:
    head_small_max: float = 0.08
    head_large_min: float = 0.16
    target_near_max: float = 0.25
    target_far_min: float = 0.50
    separation_close_max: float = 0.15
    separation_far_min: float = 0.35
    shared_target_radius: float = 0.06
    collapse_cosine: float = 0.95

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def scalar(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def crowd_bin(count: int) -> str:
    return str(count) if count <= 4 else "5+"


def head_size(bbox: Sequence[float]) -> Optional[float]:
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    # Square root of normalized area has an intuitive linear-scale interpretation.
    return math.sqrt((x2 - x1) * (y2 - y1))


def bin_head(value: Optional[float], cfg: BinConfig) -> str:
    if value is None:
        return "missing"
    if value < cfg.head_small_max:
        return "small"
    if value < cfg.head_large_min:
        return "medium"
    return "large"


def target_distance(bbox: Sequence[float], x: Optional[float], y: Optional[float]) -> Optional[float]:
    if x is None or y is None or x < 0 or y < 0 or bbox is None:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    return math.hypot((x1 + x2) / 2.0 - x, (y1 + y2) / 2.0 - y)


def bin_distance(value: Optional[float], cfg: BinConfig) -> str:
    if value is None:
        return "out_or_missing"
    if value < cfg.target_near_max:
        return "near"
    if value < cfg.target_far_min:
        return "medium"
    return "far"


def bin_separation(value: Optional[float], num_clusters: int, cfg: BinConfig) -> str:
    if num_clusters == 1:
        return "shared_only"
    if value is None:
        return "undefined"
    if value < cfg.separation_close_max:
        return "close"
    if value < cfg.separation_far_min:
        return "medium"
    return "far"


def cluster_targets(points: Sequence[Tuple[float, float]], radius: float) -> Tuple[List[int], List[Tuple[float, float]]]:
    """Single-link cluster gaze targets, so shared/near-identical targets count once."""
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if math.dist(points[i], points[j]) <= radius:
                union(i, j)

    roots: Dict[int, int] = {}
    labels: List[int] = []
    for i in range(n):
        root = find(i)
        roots.setdefault(root, len(roots))
        labels.append(roots[root])
    centers = []
    for label in range(len(roots)):
        members = [points[i] for i, assigned in enumerate(labels) if assigned == label]
        centers.append((float(np.mean([p[0] for p in members])), float(np.mean([p[1] for p in members]))))
    return labels, centers


def minimum_separation(centers: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(centers) < 2:
        return None
    return min(math.dist(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers)))


def heatmap_peak(heatmap: np.ndarray) -> Tuple[float, float]:
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return x / float(heatmap.shape[1]), y / float(heatmap.shape[0])


def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else None


def query_fidelity(
    heatmaps: Sequence[np.ndarray], targets: Sequence[Tuple[float, float]], cfg: BinConfig
) -> Tuple[List[Dict[str, Optional[float]]], Dict[str, Optional[float]]]:
    """Compute post-hoc cross-target and cross-query metrics for in-frame queries.

    Cross-Target Confusion Rate (X-TCR) uses shared-target clusters. A query is
    marked confused when its predicted peak is closest to a cluster other than
    its own. This does not imply the model observes or binds to another person's
    bbox; it is a post-hoc classification of where the localization error lands.
    Association margin = distance-to-nearest-wrong - distance-to-own (larger is
    better).
    """
    if len(heatmaps) != len(targets):
        raise ValueError("heatmaps and targets must have equal length")
    labels, centers = cluster_targets(targets, cfg.shared_target_radius)
    separation = minimum_separation(centers)
    peaks = [heatmap_peak(hm) for hm in heatmaps]
    per_query: List[Dict[str, Optional[float]]] = []
    for peak, own_label in zip(peaks, labels):
        distances = [math.dist(peak, center) for center in centers]
        if len(centers) < 2:
            confused, margin = None, None
        else:
            wrong = min(distance for label, distance in enumerate(distances) if label != own_label)
            confused = float(int(np.argmin(distances)) != own_label)
            margin = float(wrong - distances[own_label])
        per_query.append({"target_cluster": own_label, "target_confusion": confused, "association_margin": margin})

    all_cos, distinct_cos, peak_dists = [], [], []
    for i in range(len(heatmaps)):
        for j in range(i + 1, len(heatmaps)):
            value = cosine(heatmaps[i], heatmaps[j])
            if value is not None:
                all_cos.append(value)
                if labels[i] != labels[j]:
                    distinct_cos.append(value)
            peak_dists.append(math.dist(peaks[i], peaks[j]))

    def mean_or_none(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    def max_or_none(values: Sequence[float]) -> Optional[float]:
        return float(np.max(values)) if values else None

    frame = {
        "num_target_clusters": len(centers),
        "target_separation": separation,
        "shared_target_fraction": 1.0 - len(centers) / len(targets) if targets else None,
        "query_cosine_mean": mean_or_none(all_cos),
        "query_cosine_max": max_or_none(all_cos),
        "distinct_target_cosine_mean": mean_or_none(distinct_cos),
        "distinct_target_cosine_max": max_or_none(distinct_cos),
        "distinct_target_collapse_rate": mean_or_none([float(x >= cfg.collapse_cosine) for x in distinct_cos]),
        "query_peak_distance_mean": mean_or_none(peak_dists),
    }
    return per_query, frame


def flatten_annotations(dataset: str, payload: Any) -> List[Dict[str, Any]]:
    if dataset == "gazefollow":
        if not isinstance(payload, list):
            raise ValueError("GazeFollow JSON must be a list of frames")
        return payload
    frames = []
    if not isinstance(payload, list):
        raise ValueError("VAT JSON must be a list of sequences")
    for sequence in payload:
        frames.extend(sequence.get("frames", []))
    return frames


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else None


def safe_ap(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    pairs = [(int(row["inout"]), float(row["inout_score"])) for row in rows if row.get("inout_score") not in (None, "")]
    if not pairs or len({label for label, _ in pairs}) < 2:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score([x[0] for x in pairs], [x[1] for x in pairs]))


def summarize(records: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    person_metrics = ["auc", "l2", "avg_l2", "min_l2", "target_confusion", "association_margin"]
    frame_metrics = [
        "query_cosine_mean", "query_cosine_max", "distinct_target_cosine_mean",
        "distinct_target_cosine_max", "distinct_target_collapse_rate", "query_peak_distance_mean",
    ]

    def aggregate(unit: str, dimension: str, bin_name: str, rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> None:
        item: Dict[str, Any] = {"unit": unit, "dimension": dimension, "bin": bin_name, "count": len(rows)}
        for metric in metrics:
            item[metric] = safe_mean(rows, metric)
            item[f"{metric}_count"] = sum(row.get(metric) not in (None, "") for row in rows)
        if unit == "person":
            item["inout_ap"] = safe_ap(rows)
            item["inout_ap_count"] = sum(row.get("inout_score") not in (None, "") for row in rows)
        output.append(item)

    aggregate("person", "overall", "all", records, person_metrics)
    for dimension, key in [
        ("crowd", "crowd_bin"), ("head_size", "head_size_bin"),
        ("target_distance", "target_distance_bin"), ("inout", "inout"),
        ("target_separation", "target_separation_bin"),
    ]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in records:
            groups[str(row.get(key, "missing"))].append(row)
        for name in sorted(groups):
            aggregate("person", dimension, name, groups[name], person_metrics)

    aggregate("frame", "overall", "all", frames, frame_metrics)
    for dimension, key in [("crowd", "crowd_bin"), ("head_size", "head_size_bin"), ("target_separation", "target_separation_bin")]:
        groups = defaultdict(list)
        for row in frames:
            groups[str(row.get(key, "missing"))].append(row)
        for name in sorted(groups):
            aggregate("frame", dimension, name, groups[name], frame_metrics)
    return output


def load_model(args: argparse.Namespace):
    if args.model_source == "v0":
        from gazelle.model_v0 import get_gazelle_model
        model, transform = get_gazelle_model(args.model)
    elif args.model_source == "aaai_router":
        from AAAIModules.factory import build_person_hierarchical_gazelle
        model, transform = build_person_hierarchical_gazelle(args.model)
    else:
        from gazelle.model import get_gazelle_model
        model, transform = get_gazelle_model(
            args.model, spatial_prior=args.spatial_prior, fusion=args.fusion,
            selected_layers=args.selected_layers,
        )
    load_manifest = strict_load_task_checkpoint(model, args.checkpoint)
    return model, transform, load_manifest


def evaluate(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch
    from PIL import Image
    from gazelle.utils import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2

    cfg = BinConfig(
        shared_target_radius=args.shared_target_radius,
        collapse_cosine=args.collapse_cosine,
    )
    with open(args.json_path, "r", encoding="utf-8") as handle:
        frames = flatten_annotations(args.dataset, json.load(handle))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, load_manifest = load_model(args)
    args.checkpoint_load_manifest = load_manifest
    model.to(device).eval()
    person_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []

    from tqdm import tqdm
    for frame in tqdm(frames, desc=f"Taxonomy ({args.dataset})"):
        heads = frame.get("heads", [])
        if not heads:
            continue
        image = transform(Image.open(os.path.join(args.data_path, frame["path"])).convert("RGB"))
        bboxes = [head["bbox_norm"] for head in heads]
        with torch.no_grad():
            output = model({"images": image.unsqueeze(0).to(device), "bboxes": [bboxes]})
        heatmaps = [output["heatmap"][0][j].detach().cpu().numpy() for j in range(len(heads))]
        inout_scores = output.get("inout")
        inout_scores = [float(inout_scores[0][j].item()) for j in range(len(heads))] if inout_scores is not None else [None] * len(heads)

        targets: List[Optional[Tuple[float, float]]] = []
        for head in heads:
            xs = [float(x) for x in head.get("gazex_norm", []) if float(x) >= 0]
            ys = [float(y) for y in head.get("gazey_norm", []) if float(y) >= 0]
            inout = int(head.get("inout", bool(xs and ys)))
            targets.append((float(np.mean(xs)), float(np.mean(ys))) if inout and xs and ys else None)

        valid_indices = [i for i, target in enumerate(targets) if target is not None]
        valid_heatmaps = [heatmaps[i] for i in valid_indices]
        valid_targets = [targets[i] for i in valid_indices]
        fidelity_by_person: Dict[int, Dict[str, Optional[float]]] = {}
        if valid_indices:
            fidelity, frame_fidelity = query_fidelity(valid_heatmaps, valid_targets, cfg)  # type: ignore[arg-type]
            fidelity_by_person = dict(zip(valid_indices, fidelity))
        else:
            frame_fidelity = {
                "num_target_clusters": 0, "target_separation": None, "shared_target_fraction": None,
                "query_cosine_mean": None, "query_cosine_max": None,
                "distinct_target_cosine_mean": None, "distinct_target_cosine_max": None,
                "distinct_target_collapse_rate": None, "query_peak_distance_mean": None,
            }
        separation = frame_fidelity["target_separation"]
        separation_bin = bin_separation(separation, int(frame_fidelity["num_target_clusters"]), cfg)
        sizes = [head_size(bbox) for bbox in bboxes]
        valid_sizes = [size for size in sizes if size is not None]
        frame_rows.append({
            "dataset": args.dataset, "model_label": args.model_label, "path": frame["path"],
            "people_count": len(heads), "inframe_count": len(valid_indices), "crowd_bin": crowd_bin(len(heads)),
            "mean_head_size": float(np.mean(valid_sizes)) if valid_sizes else None,
            "head_size_bin": bin_head(float(np.mean(valid_sizes)) if valid_sizes else None, cfg),
            "target_separation_bin": separation_bin, **frame_fidelity,
        })

        for j, (head, hm, target, size) in enumerate(zip(heads, heatmaps, targets, sizes)):
            px, py = heatmap_peak(hm)
            inout = int(head.get("inout", target is not None))
            tx, ty = target if target is not None else (None, None)
            dist = target_distance(head["bbox_norm"], tx, ty)
            row: Dict[str, Any] = {
                "dataset": args.dataset, "model_label": args.model_label, "path": frame["path"],
                "person_index": j, "people_count": len(heads), "crowd_bin": crowd_bin(len(heads)),
                "head_size": size, "head_size_bin": bin_head(size, cfg), "inout": inout,
                "target_x": tx, "target_y": ty, "target_distance": dist,
                "target_distance_bin": bin_distance(dist, cfg), "target_cluster": None,
                "num_target_clusters": frame_fidelity["num_target_clusters"],
                "target_separation": separation, "target_separation_bin": separation_bin,
                "pred_x": px, "pred_y": py, "auc": None, "l2": None, "avg_l2": None,
                "min_l2": None, "inout_score": inout_scores[j], "target_confusion": None,
                "association_margin": None,
            }
            row.update(fidelity_by_person.get(j, {}))
            if target is not None:
                if args.dataset == "vat":
                    row["auc"] = float(vat_auc(torch.from_numpy(hm), tx, ty))
                    row["l2"] = float(vat_l2(torch.from_numpy(hm), tx, ty))
                else:
                    xs, ys = head["gazex_norm"], head["gazey_norm"]
                    height, width = int(frame["height"]), int(frame["width"])
                    row["auc"] = float(gazefollow_auc(torch.from_numpy(hm), xs, ys, height, width))
                    avg_l2, min_l2 = gazefollow_l2(torch.from_numpy(hm), xs, ys)
                    row["avg_l2"], row["min_l2"] = float(avg_l2), float(min_l2)
            person_rows.append(row)
    return person_rows, frame_rows


def run_self_test() -> None:
    cfg = BinConfig(shared_target_radius=0.05, collapse_cosine=0.95)
    assert crowd_bin(5) == "5+"
    assert bin_head(0.04, cfg) == "small" and bin_distance(0.7, cfg) == "far"
    labels, centers = cluster_targets([(0.10, 0.10), (0.12, 0.11), (0.80, 0.80)], 0.05)
    assert labels == [0, 0, 1] and len(centers) == 2
    left = np.zeros((4, 4)); left[0, 0] = 1
    right = np.zeros((4, 4)); right[3, 3] = 1
    # First query predicts the other distinct target and must be marked confused.
    per_query, frame = query_fidelity([right, left], [(0.0, 0.0), (0.75, 0.75)], cfg)
    assert per_query[0]["target_confusion"] == 1.0
    assert per_query[0]["association_margin"] < 0
    assert frame["num_target_clusters"] == 2
    assert frame["distinct_target_cosine_mean"] == 0.0
    print("Self-test passed: bins, shared-target clustering, cross-target confusion, margin, and heatmap similarity.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["vat", "gazefollow"])
    parser.add_argument("--data-path", help="Dataset image root")
    parser.add_argument("--json-path", help="Preprocessed test JSON")
    parser.add_argument("--checkpoint", help="Gazelle checkpoint")
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--model-source", choices=["current", "v0", "aaai_router"], default="current")
    parser.add_argument("--model", default=None, help="Defaults to the dataset-appropriate ViT-B model")
    parser.add_argument("--spatial-prior", default="ggsf")
    parser.add_argument("--fusion", default="sasa")
    parser.add_argument("--selected-layers", default=None)
    parser.add_argument("--output-prefix", default="AAAIScripts/results/failure_taxonomy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--collapse-cosine", type=float, default=0.95)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    missing = [name for name in ("dataset", "data_path", "json_path", "checkpoint") if getattr(args, name) is None]
    if missing:
        parser.error("normal evaluation requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.model is None:
        if args.model_source == "aaai_router":
            args.model = "aaai_person_router_dinov3_vitb16_inout" if args.dataset == "vat" else "aaai_person_router_dinov3_vitb16"
        else:
            args.model = "gazelle_dinov3_vitb16_inout" if args.dataset == "vat" else "gazelle_dinov3_vitb16"
    if args.model_source == "v0":
        args.spatial_prior, args.fusion = None, None
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    records, frames = evaluate(args)
    summary = summarize(records, frames)
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".records.csv"), records, PERSON_FIELDS)
    write_csv(prefix.with_suffix(".frames.csv"), frames, FRAME_FIELDS)
    summary_fields = sorted({key for row in summary for key in row})
    write_csv(prefix.with_suffix(".summary.csv"), summary, summary_fields)
    report = {
        "status": "measured",
        "dataset": args.dataset,
        "model_label": args.model_label,
        "model_source": args.model_source,
        "model": args.model,
        "checkpoint": file_manifest(args.checkpoint),
        "checkpoint_load": args.checkpoint_load_manifest,
        "annotation": file_manifest(args.json_path),
        "num_person_records": len(records),
        "num_frame_records": len(frames),
        "bin_config": BinConfig(shared_target_radius=args.shared_target_radius, collapse_cosine=args.collapse_cosine).as_dict(),
        "metric_notes": {
            "target_confusion": "Post-hoc X-TCR indicator: 1 iff the predicted peak is closest to another shared-target cluster; it does not imply access to or binding with another person's bbox; undefined with <2 clusters",
            "association_margin": "distance(pred, nearest wrong cluster) - distance(pred, own cluster); larger is better",
            "distinct_target_collapse_rate": "fraction of different-target query pairs with cosine >= collapse_cosine",
            "counterfactual_query_swap": "not implemented; requires an additional controlled forward runner",
        },
        "summary": summary,
    }
    report_path = prefix.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(records)} person records and {len(frames)} frame records to {prefix.parent}")


if __name__ == "__main__":
    main()
