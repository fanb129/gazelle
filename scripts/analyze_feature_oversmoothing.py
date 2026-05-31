import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


LAYER_INDEX = {
    "shallow": 0,
    "mid": 1,
    "deep": 2,
    "last": 3,
}

REQUIRED_METRICS = {
    "crowd_token_cosine",
    "inter_person_boundary_separability",
    "foreground_background_contrast",
}

OPTIONAL_METRICS = {"effective_rank", "layerwise_probe"}

BACKBONE_ALIASES = {
    "dinov2_vitb16": "dinov2_vitb14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov3_vitb16": "dinov3_vitb16",
}


def _empty_metric(reason, optional=False):
    return {
        "computed": False,
        "value": None,
        "sample_count": 0,
        "skipped_count": 1,
        "skip_reasons": {reason: 1},
        "optional": optional,
    }


def _computed_metric(value, optional=False, extra=None):
    metric = {
        "computed": True,
        "value": float(value),
        "sample_count": 1,
        "skipped_count": 0,
        "skip_reasons": {},
        "optional": optional,
    }
    if extra:
        metric.update(extra)
    return metric


def _normalize_vectors(vectors, eps=1e-8):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _cosine_distance(a, b, eps=1e-8):
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return 1.0 - float(np.dot(a, b) / denom)


def _tokens_from_mask(features, mask):
    if features.ndim != 3:
        raise ValueError("features must have shape [channels, height, width]")
    if mask.shape != features.shape[1:]:
        raise ValueError("mask shape must match feature height/width")
    return features[:, mask].T


def _union_mask(masks, height, width):
    if not masks:
        return np.zeros((height, width), dtype=bool)
    result = np.zeros((height, width), dtype=bool)
    for mask in masks:
        result |= mask
    return result


def masks_from_boxes(boxes, height, width):
    masks = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        left = min(width - 1, max(0, int(math.floor(x1 * width))))
        right = min(width, max(left + 1, int(math.ceil(x2 * width))))
        top = min(height - 1, max(0, int(math.floor(y1 * height))))
        bottom = min(height, max(top + 1, int(math.ceil(y2 * height))))
        mask = np.zeros((height, width), dtype=bool)
        mask[top:bottom, left:right] = True
        masks.append(mask)
    return masks


def compute_crowd_token_cosine(features, region_masks):
    foreground = _union_mask(region_masks, features.shape[1], features.shape[2])
    tokens = _tokens_from_mask(features, foreground)
    if tokens.shape[0] < 2:
        return _empty_metric("crowd_region_lt_two_tokens")

    normalized = _normalize_vectors(tokens.astype(np.float64))
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(tokens.shape[0], k=1)]
    value = float(np.clip(upper.mean(), -1.0, 1.0))
    if np.isclose(value, 1.0):
        value = 1.0
    return _computed_metric(value, extra={"token_count": int(tokens.shape[0])})


def _dilate(mask, radius=1):
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def compute_inter_person_boundary_separability(features, region_masks):
    valid_masks = [mask for mask in region_masks if np.any(mask)]
    if len(valid_masks) < 2:
        return _empty_metric("requires_at_least_two_regions")

    distances = []
    for mask in valid_masks:
        ring = _dilate(mask, radius=1) & ~mask
        ring &= ~_union_mask([other for other in valid_masks if other is not mask], *mask.shape)
        head_tokens = _tokens_from_mask(features, mask)
        boundary_tokens = _tokens_from_mask(features, ring)
        if head_tokens.shape[0] == 0 or boundary_tokens.shape[0] == 0:
            continue
        distances.append(_cosine_distance(head_tokens.mean(axis=0), boundary_tokens.mean(axis=0)))
    head_means = [_tokens_from_mask(features, mask).mean(axis=0) for mask in valid_masks]
    for first_idx in range(len(head_means)):
        for second_idx in range(first_idx + 1, len(head_means)):
            distances.append(_cosine_distance(head_means[first_idx], head_means[second_idx]))

    if not distances:
        return _empty_metric("no_valid_boundary_band_tokens")
    return _computed_metric(float(np.mean(distances)), extra={"region_count": len(valid_masks)})


def compute_foreground_background_contrast(features, region_masks):
    foreground = _union_mask(region_masks, features.shape[1], features.shape[2])
    background = ~foreground
    fg_tokens = _tokens_from_mask(features, foreground)
    bg_tokens = _tokens_from_mask(features, background)
    if fg_tokens.shape[0] == 0:
        return _empty_metric("no_foreground_tokens")
    if bg_tokens.shape[0] == 0:
        return _empty_metric("no_background_tokens")
    value = _cosine_distance(fg_tokens.mean(axis=0), bg_tokens.mean(axis=0))
    return _computed_metric(value, extra={"foreground_tokens": int(fg_tokens.shape[0]), "background_tokens": int(bg_tokens.shape[0])})


def compute_effective_rank(features):
    tokens = features.reshape(features.shape[0], -1).T
    if tokens.shape[0] < 2:
        return _empty_metric("effective_rank_requires_two_tokens", optional=True)
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-8]
    if singular_values.size == 0:
        return _empty_metric("effective_rank_zero_variance", optional=True)
    probabilities = singular_values / singular_values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return _computed_metric(float(np.exp(entropy)), optional=True, extra={"token_count": int(tokens.shape[0])})


def skipped_layerwise_probe(args):
    return {
        "computed": False,
        "value": None,
        "sample_count": 0,
        "skipped_count": 1,
        "skip_reasons": {"layerwise_probe_not_implemented_for_section5_smoke": 1},
        "optional": True,
        "probe_epochs": args.probe_epochs,
    }


def combine_metric_values(metric_values, optional=False):
    computed = [metric for metric in metric_values if metric["computed"]]
    skip_reasons = Counter()
    skipped_count = 0
    for metric in metric_values:
        skipped_count += int(metric.get("skipped_count", 0))
        skip_reasons.update(metric.get("skip_reasons", {}))
    if not computed:
        return {
            "computed": False,
            "value": None,
            "sample_count": 0,
            "skipped_count": skipped_count,
            "skip_reasons": dict(skip_reasons),
            "optional": optional,
        }
    return {
        "computed": True,
        "value": float(np.mean([metric["value"] for metric in computed])),
        "sample_count": len(computed),
        "skipped_count": skipped_count,
        "skip_reasons": dict(skip_reasons),
        "optional": optional,
    }


def compute_metrics_for_feature(features, boxes, metrics, args):
    masks = masks_from_boxes(boxes, features.shape[1], features.shape[2])
    results = {}
    for metric in metrics:
        if metric == "crowd_token_cosine":
            results[metric] = compute_crowd_token_cosine(features, masks)
        elif metric == "inter_person_boundary_separability":
            results[metric] = compute_inter_person_boundary_separability(features, masks)
        elif metric == "foreground_background_contrast":
            results[metric] = compute_foreground_background_contrast(features, masks)
        elif metric == "effective_rank":
            results[metric] = compute_effective_rank(features)
        elif metric == "layerwise_probe":
            results[metric] = skipped_layerwise_probe(args)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return results


def validate_args(args):
    invalid_metrics = sorted(set(args.metrics) - REQUIRED_METRICS - OPTIONAL_METRICS)
    if invalid_metrics:
        raise ValueError(f"Unsupported metrics: {', '.join(invalid_metrics)}")
    invalid_layers = sorted(set(args.layers) - set(LAYER_INDEX))
    if invalid_layers:
        raise ValueError(f"Unsupported layers: {', '.join(invalid_layers)}")
    invalid_backbones = sorted(set(args.backbones) - set(BACKBONE_ALIASES))
    if invalid_backbones:
        raise ValueError(f"Unsupported backbones: {', '.join(invalid_backbones)}")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")


def synthetic_feature_for_layer(layer):
    scale = LAYER_INDEX[layer] + 1
    features = np.zeros((4, 6, 6), dtype=np.float32)
    features[0, :3, :3] = 1.0 * scale
    features[1, 3:, 3:] = 1.0 * scale
    features[2] = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    features[3] = 1.0
    return features


def build_synthetic_payload(args):
    boxes = [[0.05, 0.05, 0.45, 0.45], [0.55, 0.55, 0.95, 0.95]]
    rows = []
    for backbone in args.backbones:
        for layer in args.layers:
            features = synthetic_feature_for_layer(layer)
            rows.append(
                {
                    "backbone": backbone,
                    "resolved_backbone": BACKBONE_ALIASES[backbone],
                    "checkpoint_path": f"./checkpoints/{BACKBONE_ALIASES[backbone]}_pretrain.pth",
                    "transform": "synthetic_identity",
                    "input_size": None,
                    "layer": layer,
                    "layer_index": LAYER_INDEX[layer],
                    "metrics": compute_metrics_for_feature(features, boxes, args.metrics, args),
                    "sample_count": 1,
                }
            )
    return {
        "section": "5.feature_oversmoothing_analysis",
        "synthetic_smoke": True,
        "data_path": args.data_path,
        "crowd_json": args.crowd_json,
        "backbones": args.backbones,
        "layers": args.layers,
        "metrics": args.metrics,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": 1,
        "rows": rows,
    }


def iter_vat_crowd_samples(data_path, crowd_json, max_samples):
    with open(crowd_json, "r") as handle:
        sequences = json.load(handle)
    count = 0
    for sequence in sequences:
        for frame in sequence.get("frames", []):
            heads = [head for head in frame.get("heads", []) if head.get("bbox_norm")]
            if not heads:
                continue
            frame_path = frame.get("path", "")
            image_path = frame_path
            if not os.path.isabs(image_path):
                image_path = os.path.join(data_path, frame_path)
            boxes = [head["bbox_norm"] for head in heads]
            yield {"image_path": image_path, "boxes": boxes}
            count += 1
            if count >= max_samples:
                return


def load_backbone(backbone_name, device):
    resolved = BACKBONE_ALIASES[backbone_name]
    if resolved.startswith("dinov2"):
        from gazelle.backbone_dinov2 import DinoV2Backbone

        backbone = DinoV2Backbone(resolved)
    elif resolved.startswith("dinov3"):
        from gazelle.backbone import DinoV3Backbone

        backbone = DinoV3Backbone(resolved)
    else:
        raise ValueError(f"Unsupported resolved backbone: {resolved}")
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_layer_features(backbone, image_path, input_size, device):
    from PIL import Image

    import torch

    image = Image.open(image_path).convert("RGB")
    transformed = backbone.get_transform((input_size, input_size))(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(transformed)
    return [feature.squeeze(0).detach().cpu().numpy().astype(np.float32) for feature in features]


def build_real_payload(args):
    import torch

    samples = list(iter_vat_crowd_samples(args.data_path, args.crowd_json, args.max_samples))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for backbone_name in args.backbones:
        backbone = load_backbone(backbone_name, device)
        layer_metric_values = {
            layer: {metric: [] for metric in args.metrics}
            for layer in args.layers
        }
        extraction_skips = Counter()
        for sample in samples:
            try:
                features_by_depth = extract_layer_features(backbone, sample["image_path"], args.input_size, device)
            except Exception as exc:
                extraction_skips[type(exc).__name__] += 1
                continue
            for layer in args.layers:
                layer_features = features_by_depth[LAYER_INDEX[layer]]
                sample_metrics = compute_metrics_for_feature(layer_features, sample["boxes"], args.metrics, args)
                for metric_name, metric_result in sample_metrics.items():
                    layer_metric_values[layer][metric_name].append(metric_result)

        for layer in args.layers:
            metrics = {
                metric: combine_metric_values(
                    layer_metric_values[layer][metric],
                    optional=metric in OPTIONAL_METRICS,
                )
                for metric in args.metrics
            }
            for metric in metrics.values():
                metric["extraction_skips"] = dict(extraction_skips)
            resolved = BACKBONE_ALIASES[backbone_name]
            rows.append(
                {
                    "backbone": backbone_name,
                    "resolved_backbone": resolved,
                    "checkpoint_path": f"./checkpoints/{resolved}_pretrain.pth",
                    "transform": f"backbone.get_transform(({args.input_size}, {args.input_size}))",
                    "input_size": [args.input_size, args.input_size],
                    "layer": layer,
                    "layer_index": LAYER_INDEX[layer],
                    "metrics": metrics,
                    "sample_count": len(samples),
                }
            )
    return {
        "section": "5.feature_oversmoothing_analysis",
        "synthetic_smoke": False,
        "data_path": args.data_path,
        "crowd_json": args.crowd_json,
        "backbones": args.backbones,
        "layers": args.layers,
        "metrics": args.metrics,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": len(samples),
        "rows": rows,
    }


def write_payload(payload, output):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute P1 DINO feature over-smoothing metrics.")
    parser.add_argument("--data_path", default="", help="VAT root directory.")
    parser.add_argument("--crowd_json", default="", help="VAT Crowd dense preprocessed JSON.")
    parser.add_argument("--backbones", nargs="+", default=["dinov2_vitb16", "dinov3_vitb16"])
    parser.add_argument("--layers", nargs="+", default=["shallow", "mid", "deep", "last"])
    parser.add_argument("--metrics", nargs="+", default=sorted(REQUIRED_METRICS))
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--probe_epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic_smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        validate_args(args)
        if not args.synthetic_smoke and (not args.data_path or not args.crowd_json):
            raise ValueError("--data_path and --crowd_json are required outside --synthetic_smoke")
        payload = build_synthetic_payload(args) if args.synthetic_smoke else build_real_payload(args)
        write_payload(payload, args.output)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote feature over-smoothing analysis: {args.output}")


if __name__ == "__main__":
    main()
