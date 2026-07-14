#!/usr/bin/env python3
"""Training-free probes for hierarchical DINO features used by GazeSpot.

The script intentionally analyzes raw backbone tensors.  It never modifies model
predictions with ground truth and it does not call the evaluation helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest, strict_load_task_checkpoint

LAYER_NAMES = ("layer_2", "layer_5", "layer_8", "layer_11")
PROBE_MAX_TOKENS = 64


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _center_tokens(feature: np.ndarray) -> np.ndarray:
    tokens = feature.reshape(feature.shape[0], -1).T.astype(np.float64)
    if len(tokens) > PROBE_MAX_TOKENS:
        # Deterministic coverage of the spatial grid keeps the probe inexpensive
        # enough to run before committing to full training.
        indices = np.linspace(0, len(tokens) - 1, PROBE_MAX_TOKENS, dtype=int)
        tokens = tokens[indices]
    return tokens - tokens.mean(axis=0, keepdims=True)


def token_cosine(feature: np.ndarray, eps: float = 1e-12) -> float | None:
    tokens = feature.reshape(feature.shape[0], -1).T.astype(np.float64)
    if len(tokens) < 2:
        return None
    tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), eps)
    # Equivalent to the off-diagonal mean without materializing an NxN matrix.
    total = np.square(tokens.sum(axis=0)).sum() - len(tokens)
    return _safe_float(total / (len(tokens) * (len(tokens) - 1)))


def effective_rank(feature: np.ndarray, eps: float = 1e-12) -> tuple[float | None, float | None]:
    tokens = _center_tokens(feature)
    singular = np.linalg.svd(tokens, compute_uv=False)
    energy = np.square(singular)
    energy = energy[energy > eps]
    if not len(energy):
        return None, None
    probs = energy / energy.sum()
    rank = float(np.exp(-(probs * np.log(probs)).sum()))
    return _safe_float(rank), _safe_float(rank / min(tokens.shape))


def linear_cka(first: np.ndarray, second: np.ndarray, eps: float = 1e-12) -> float | None:
    x, y = _center_tokens(first), _center_tokens(second)
    if x.shape[0] != y.shape[0]:
        return None
    # Sample-space identity avoids constructing large channel-by-channel matrices.
    gram_x, gram_y = x @ x.T, y @ y.T
    cross = float((gram_x * gram_y).sum())
    denom = np.linalg.norm(gram_x, ord="fro") * np.linalg.norm(gram_y, ord="fro")
    return _safe_float(cross / max(denom, eps))


def directional_novelty(source: np.ndarray, target: np.ndarray, rank: int = 32, eps: float = 1e-12) -> float | None:
    """Fraction of target energy outside the source's leading channel subspace.

    This is a representation diagnostic, not evidence that the novel component is
    task-useful.  Directionality is deliberate: novelty(A->B) != novelty(B->A).
    """
    x, y = _center_tokens(source), _center_tokens(target)
    if x.shape[1] != y.shape[1] or min(x.shape) < 2:
        return None
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[: min(rank, len(vh))].T
    residual = y - (y @ basis) @ basis.T
    return _safe_float(np.square(residual).sum() / max(np.square(y).sum(), eps))


def _head_area_bin(heads: list[dict]) -> str:
    areas = []
    for head in heads:
        box = head.get("bbox_norm")
        if box and len(box) == 4:
            areas.append(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    if not areas:
        return "unknown"
    area = float(np.mean(areas))
    return "small" if area < 0.01 else ("medium" if area < 0.04 else "large")


def sample_attributes(frame: dict) -> dict[str, str | int | float]:
    heads = frame.get("heads", [])
    count = len(heads)
    inout = [int(h.get("inout", 1)) for h in heads]
    return {
        "people_count": count,
        "crowd_bin": "1" if count <= 1 else ("2-3" if count <= 3 else "4+"),
        "head_area_bin": _head_area_bin(heads),
        "inout_bin": "unknown" if not inout else ("all_in" if all(inout) else ("all_out" if not any(inout) else "mixed")),
    }


def iter_frames(
    dataset: str,
    data_path: Path,
    json_path: Path,
    max_samples: int,
    *,
    sampling: str = "prefix",
    sampling_seed: int = 3106,
):
    payload = json.loads(json_path.read_text())
    frames = payload if dataset == "gazefollow" else [f for seq in payload for f in seq.get("frames", [])]
    if sampling == "uniform" and max_samples < len(frames):
        rng = np.random.default_rng(sampling_seed)
        indices = sorted(rng.choice(len(frames), size=max_samples, replace=False).tolist())
    else:
        indices = list(range(min(max_samples, len(frames))))
    for index in indices:
        frame = frames[index]
        image_path = Path(frame.get("path", ""))
        if not image_path.is_absolute():
            image_path = data_path / image_path
        yield index, frame, image_path


def analyze_features(features: list[np.ndarray], sample_id: str, attrs: dict) -> list[dict]:
    rows = []
    for name, feature in zip(LAYER_NAMES, features):
        rank, normalized_rank = effective_rank(feature)
        rows.extend([
            {**attrs, "sample_id": sample_id, "scope": "layer", "layer_a": name, "layer_b": "", "metric": "token_cosine", "value": token_cosine(feature)},
            {**attrs, "sample_id": sample_id, "scope": "layer", "layer_a": name, "layer_b": "", "metric": "effective_rank", "value": rank},
            {**attrs, "sample_id": sample_id, "scope": "layer", "layer_a": name, "layer_b": "", "metric": "normalized_effective_rank", "value": normalized_rank},
            {**attrs, "sample_id": sample_id, "scope": "layer", "layer_a": name, "layer_b": "", "metric": "spatial_variance", "value": _safe_float(_center_tokens(feature).var(axis=0).mean())},
        ])
    for first in range(len(features)):
        for second in range(first + 1, len(features)):
            common = {**attrs, "sample_id": sample_id, "scope": "pair", "layer_a": LAYER_NAMES[first], "layer_b": LAYER_NAMES[second]}
            rows.append({**common, "metric": "linear_cka", "value": linear_cka(features[first], features[second])})
            rows.append({**common, "metric": "directional_novelty", "value": directional_novelty(features[first], features[second])})
    return rows


def synthetic_rows() -> list[dict]:
    rng = np.random.default_rng(3106)
    base = rng.normal(size=(16, 8, 8)).astype(np.float32)
    features = [base, base * 0.9 + rng.normal(scale=0.1, size=base.shape), rng.normal(size=base.shape), np.ones_like(base)]
    return analyze_features(features, "synthetic_0", {"people_count": 4, "crowd_bin": "4+", "head_area_bin": "small", "inout_bin": "all_in"})


def aggregate(rows: list[dict], group_fields: Iterable[str]) -> list[dict]:
    buckets = defaultdict(list)
    key_fields = ["scope", "layer_a", "layer_b", "metric", *group_fields]
    for row in rows:
        if row["value"] is not None:
            buckets[tuple(row.get(field, "all") for field in key_fields)].append(float(row["value"]))
    output = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(key_fields, key))
        record.update({"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--dataset", choices=("vat", "gazefollow"), default="vat")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--checkpoint", type=Path, help="Optional GazeSpot checkpoint; loaded for architecture compatibility. Raw backbone features are unchanged.")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--group-by", nargs="*", default=["crowd_bin", "head_area_bin", "inout_bin"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.synthetic_smoke:
        rows = synthetic_rows()
    else:
        if not args.data_path or not args.json_path:
            raise SystemExit("--data-path and --json-path are required outside --synthetic-smoke")
        import torch
        from PIL import Image
        from gazelle.model import get_gazelle_model

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, transform = get_gazelle_model(args.model, use_sasa=True, use_ggsf=True)
        checkpoint_load = None
        if args.checkpoint:
            checkpoint_load = strict_load_task_checkpoint(model, args.checkpoint)
        model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for index, frame, image_path in iter_frames(args.dataset, args.data_path, args.json_path, args.max_samples):
                if not image_path.exists():
                    continue
                image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                raw = model.backbone(image)
                features = [tensor[0].detach().float().cpu().numpy() for tensor in raw]
                rows.extend(analyze_features(features, str(index), sample_attributes(frame)))

    aggregates = aggregate(rows, []) + aggregate(rows, args.group_by)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    payload = {
        "schema_version": 1,
        "analysis": "raw_hierarchical_feature_probe",
        "synthetic_smoke": args.synthetic_smoke,
        "caveat": "Representation complementarity proxies do not establish task utility; confirm with matched controls. Matrix probes use deterministic spatial subsampling.",
        "probe_max_spatial_tokens": PROBE_MAX_TOKENS,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "row_count": len(rows),
        "aggregates": aggregates,
    }
    if not args.synthetic_smoke:
        payload["annotation"] = file_manifest(args.json_path)
        if args.checkpoint:
            payload["checkpoint"] = file_manifest(args.checkpoint)
            payload["checkpoint_load"] = checkpoint_load
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} raw probe rows to {args.output_dir}")


if __name__ == "__main__":
    main()
