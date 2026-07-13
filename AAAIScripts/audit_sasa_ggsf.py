#!/usr/bin/env python3
"""Audit GGSF gates and SASA weights from unmodified, raw model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest, strict_load_task_checkpoint


def entropy(values: np.ndarray, eps=1e-12) -> float:
    values = np.clip(values.astype(np.float64), eps, None)
    probs = values / values.sum()
    return float(-(probs * np.log(probs)).sum() / math.log(len(probs))) if len(probs) > 1 else 0.0


def mask_rows(mask: np.ndarray, sample_id: str, attrs: dict) -> list[dict]:
    rows = []
    for person, array in enumerate(mask):
        flat = array.reshape(-1)
        metrics = {
            "mean": flat.mean(), "variance": flat.var(), "std": flat.std(), "min": flat.min(), "max": flat.max(),
            "spatial_entropy": entropy(flat), "identity_l1": np.abs(1.0 - flat).mean(),
            "dynamic_range": flat.max() - flat.min(),
        }
        rows.extend({**attrs, "sample_id": sample_id, "person": person, "component": "ggsf", "metric": key, "value": float(value)} for key, value in metrics.items())
    if len(mask) > 1:
        distances = [np.abs(mask[i] - mask[j]).mean() for i in range(len(mask)) for j in range(i + 1, len(mask))]
        rows.append({**attrs, "sample_id": sample_id, "person": "all", "component": "ggsf", "metric": "inter_person_l1", "value": float(np.mean(distances))})
    return rows


def weight_rows(weights: np.ndarray, sample_id: str, attrs: dict) -> list[dict]:
    rows = []
    uniform = np.full(weights.shape[1], 1.0 / weights.shape[1])
    for person, vector in enumerate(weights):
        metrics = {"entropy": entropy(vector), "max_weight": vector.max(), "uniform_l1": np.abs(vector - uniform).mean()}
        for layer, value in enumerate(vector):
            metrics[f"weight_layer_{layer}"] = value
        rows.extend({**attrs, "sample_id": sample_id, "person": person, "component": "sasa", "metric": key, "value": float(value)} for key, value in metrics.items())
    if len(weights) > 1:
        rows.append({**attrs, "sample_id": sample_id, "person": "all", "component": "sasa", "metric": "inter_person_l1", "value": float(np.abs(weights - weights.mean(axis=0)).mean())})
    return rows


def synthetic_rows():
    attrs = {"people_count": 2, "crowd_bin": "2-3", "head_area_bin": "medium", "inout_bin": "all_in"}
    masks = np.stack([np.linspace(.2, 1, 64).reshape(1, 8, 8), np.full((1, 8, 8), .88)])
    weights = np.array([[.1, .2, .3, .4], [.25, .25, .25, .25]])
    return mask_rows(masks, "synthetic_0", attrs) + weight_rows(weights, "synthetic_0", attrs)


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["component"], row["metric"], row["crowd_bin"])].append(row["value"])
    output = []
    for (component, metric, crowd_bin), values in sorted(buckets.items()):
        output.append({"component": component, "metric": metric, "crowd_bin": crowd_bin, "n": len(values),
                       "mean": float(np.mean(values)), "variance_across_observations": float(np.var(values, ddof=1)) if len(values) > 1 else None,
                       "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return output


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--dataset", choices=("vat", "gazefollow"), default="vat")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.synthetic_smoke:
        rows = synthetic_rows()
    else:
        if not args.data_path or not args.json_path or not args.checkpoint:
            raise SystemExit("--data-path, --json-path and --checkpoint are required outside --synthetic-smoke")
        import torch
        from PIL import Image
        from gazelle.model import get_gazelle_model
        from AAAIScripts.hierarchical_feature_probe import iter_frames, sample_attributes

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, transform = get_gazelle_model(args.model, use_sasa=True, use_ggsf=True)
        checkpoint_load = strict_load_task_checkpoint(model, args.checkpoint)
        model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for index, frame, image_path in iter_frames(args.dataset, args.data_path, args.json_path, args.max_samples):
                boxes = [head.get("bbox_norm") for head in frame.get("heads", []) if head.get("bbox_norm")]
                if not boxes or not image_path.exists():
                    continue
                image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                # This is the model's unmodified forward output. No eval utility or GT postprocessing is used.
                output = model({"images": image, "bboxes": [boxes]})
                attrs = sample_attributes(frame)
                if output.get("geo_mask") is not None:
                    rows.extend(mask_rows(output["geo_mask"].detach().float().cpu().numpy(), str(index), attrs))
                if output.get("layer_weights") is not None:
                    rows.extend(weight_rows(output["layer_weights"].detach().float().cpu().numpy(), str(index), attrs))
    aggregates = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    payload = {"schema_version": 1, "analysis": "raw_sasa_ggsf_audit", "synthetic_smoke": args.synthetic_smoke,
               "no_gt_postprocessing": True, "row_count": len(rows), "aggregates": aggregates,
               "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if not args.synthetic_smoke:
        payload["checkpoint"] = file_manifest(args.checkpoint)
        payload["checkpoint_load"] = checkpoint_load
        payload["annotation"] = file_manifest(args.json_path)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} raw gate/weight audit rows to {args.output_dir}")


if __name__ == "__main__":
    main()
