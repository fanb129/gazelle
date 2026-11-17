"""Cache frozen native-grid VAT test features for per-epoch official test scoring."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from cache_frozen_features import UniqueImages, backbone_batches, collate_images
from cache_single_frame import load_frozen_checkpoint
from prepare_development import artifact
from stag.cache import argmax_points, file_sha256, write_json
from stag.feature_cache import feature_contract, write_feature_cache
from stag.official_data import load_official_predictions


def test_image_records(tracks):
    """Collect all current true head boxes for each unique official test image."""
    grouped = {}
    for track in tracks:
        for position, frame in enumerate(track["frames"], start=track["start"]):
            grouped.setdefault(frame["path"], []).append({
                "track_id": track["track_id"], "global_index": position,
                "bbox_norm": frame["bbox_norm"], "frame_index": frame["frame_index"],
            })
    return [{"path": path, "people": sorted(people, key=lambda item: item["global_index"])}
            for path, people in sorted(grouped.items())]


def test_feature_sources(prediction_report, train_feature_report):
    prediction, index, arrays = load_official_predictions(prediction_report, "test")
    train_features = json.loads(Path(train_feature_report).read_text())
    if not train_features["passed"] or train_features["mode"] != "cache":
        raise ValueError("Test features require the completed train feature contract report")
    for name in ("checkpoint", "backbone_checkpoint"):
        if train_features["sources"][name + "_sha256"] != prediction[name]["sha256"]:
            raise ValueError("Test prediction model differs from the retained train features")
    sources = {"prediction_report_sha256": file_sha256(prediction_report),
               "prediction_index_sha256": prediction["cache"]["index_sha256"],
               "checkpoint_sha256": prediction["checkpoint"]["sha256"],
               "backbone_checkpoint_sha256": prediction["backbone_checkpoint"]["sha256"],
               "train_feature_report_sha256": file_sha256(train_feature_report)}
    return prediction, index, arrays, train_features, sources


def probe_test_batch(model, images, records, prediction_arrays, expected_contract, device):
    """Check a first RGB batch against retained FP32 predictions before caching."""
    with torch.inference_mode():
        images = images.to(device)
        pyramid = model.backbone(images)
        current = pyramid[-1]
        contract = feature_contract(current, model.backbone.out_indices[-1])
        if contract != expected_contract:
            raise ValueError("Test native feature contract differs from the train cache")
        boxes = [[person["bbox_norm"] for person in record["people"]] for record in records]
        output = model.forward_from_features({"images": images, "bboxes": boxes}, pyramid)
        maps = torch.cat(output["heatmap"]).cpu().numpy()
        scores = torch.cat(output["inout"]).cpu().numpy()
        positions = [person["global_index"] for record in records for person in record["people"]]
        reference_maps = prediction_arrays["heatmap"][positions]
        reference_scores = prediction_arrays["inout"][positions]
        map_error = float(np.abs(maps - reference_maps).max())
        score_error = float(np.abs(scores - reference_scores).max())
        argmax_changes = int(np.any(argmax_points(maps) != argmax_points(reference_maps), axis=1).sum())
    checks = {"all_model_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
              "model_in_eval": not model.training,
              "fp32_heatmaps_match_retained_test": map_error <= 1e-5,
              "fp32_inout_matches_retained_test": score_error <= 1e-5,
              "argmax_matches_retained_test": argmax_changes == 0}
    if not all(checks.values()):
        raise ValueError(f"Official test feature probe failed: {checks}")
    return {"images": len(records), "person_frames": len(positions),
            "image_paths": [record["path"] for record in records], "absolute_tolerance": 1e-5,
            "heatmap_max_abs_error": map_error, "inout_max_abs_error": score_error,
            "argmax_changed_people": argmax_changes, "checks": checks, "passed": True}


def cache_test_features(model, loader, prediction_arrays, image_paths,
                        contract, sources, output_dir, device):
    """Probe then stream the same frozen RGB encoder over the complete test split."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = len(image_paths) * contract["channels"] * int(np.prod(contract["grid"])) * 2
    available_bytes = shutil.disk_usage(output_dir.parent).free
    if available_bytes < estimated_bytes + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk space for the official test native-grid feature cache")
    images, records = next(iter(loader))
    probe = probe_test_batch(model, images, records, prediction_arrays, contract, device)
    cache = write_feature_cache(image_paths, backbone_batches(model, loader, device),
                                output_dir, contract, sources)
    index_path = output_dir / "index.json"
    index = json.loads(index_path.read_text())
    index.update({"task": "R4_official_test_features", "official_split": "test"})
    write_json(index_path, index)
    cache["index_sha256"] = file_sha256(index_path)
    return {"cache": cache, "probe": probe,
            "storage": {"unique_images": len(image_paths), "estimated_feature_bytes": estimated_bytes,
                        "available_bytes_before_cache": available_bytes},
            "checks": {**probe["checks"], **cache["checks"]}, "passed": cache["passed"]}


def code_hashes():
    paths = [Path(__file__), *(REPO_ROOT / f"cscwd/code/scripts/{name}.py" for name in
                ("cache_frozen_features", "cache_single_frame", "prepare_development")),
             *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
                ("official_data", "feature_cache", "cache")),
             REPO_ROOT / "gazelle/model.py", REPO_ROOT / "gazelle/backbone.py"]
    return {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths}


def summary_markdown(report):
    return "\n".join([
        "# Official VAT test frozen features", "",
        f"- passed: `{report['passed']}`; official split: `test`.",
        f"- Contract: `{json.dumps(report['contract'])}`.",
        f"- Coverage: `{json.dumps(report['coverage'])}`.",
        f"- First-batch FP32 comparison: `{json.dumps(report['probe'])}`.",
        f"- Cache: `{json.dumps(report['cache'])}`.",
        "- Original FP32 single-frame predictions are retained unchanged.",
        "- This job caches current RGB features; it does not train the temporal module or select an epoch.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-path", "prediction-report", "train-feature-report", "checkpoint",
                 "output-dir", "report", "summary", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    prediction, index, arrays, train_features, sources = test_feature_sources(
        args.prediction_report, args.train_feature_report)
    if file_sha256(args.checkpoint) != prediction["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from the retained single-frame predictions")
    backbone_path = REPO_ROOT / "checkpoints/dinov3_vitb16_pretrain.pth"
    if file_sha256(backbone_path) != prediction["backbone_checkpoint"]["sha256"]:
        raise ValueError("Backbone differs from the retained single-frame predictions")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from gazelle.model import get_gazelle_model
    model, transform = get_gazelle_model(prediction["model"], use_sasa=True, use_ggsf=True, use_aux=False)
    load_frozen_checkpoint(model, args.checkpoint)
    model.to(args.device)
    if (train_features["runtime"]["transform"] != repr(transform)
            or train_features["runtime"]["torch"] != torch.__version__):
        raise ValueError("Test transform or Torch runtime differs from the train feature cache")
    records = test_image_records(index["tracks"])
    loader = DataLoader(UniqueImages(records, args.data_path, transform), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate_images, shuffle=False)
    result = cache_test_features(model, loader, arrays, [r["path"] for r in records],
        train_features["contract"], sources, args.output_dir, args.device)
    result.update({"task": "R4_official_test_features", "mode": "cache", "official_split": "test",
        "contract": train_features["contract"], "sources": sources, "model": prediction["model"],
        "checkpoint": prediction["checkpoint"], "backbone_checkpoint": prediction["backbone_checkpoint"],
        "coverage": {key: prediction["cache"][key] for key in ("tracks", "person_frames", "unique_image_frames")},
        "source_reports": {"prediction_report": artifact(args.prediction_report),
                           "train_feature_report": artifact(args.train_feature_report)},
        "code_sha256": code_hashes(),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": args.device,
            "device_name": torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda" else "CPU",
            "batch_size": args.batch_size, "num_workers": args.num_workers, "transform": repr(transform),
            "precision": "float32_forward; FP16_feature_storage; no_autocast; tf32_disabled"},
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]), "elapsed_seconds": time.perf_counter() - started})
    write_json(args.report, result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(result))
    with args.provenance.open("a") as stream:
        stream.write(f"\n## Official VAT test frozen features\n\n"
            f"- UTC: `{result['created_at_utc']}`; commit: `{result['git_commit']}`.\n"
            f"- Command: `{result['command']}`.\n"
            f"- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Cache index: `{args.output_dir / 'index.json'}`; SHA256: `{result['cache']['index_sha256']}`.\n")
        for item in result["cache"]["files"].values():
            stream.write(f"- Server artifact: `{args.output_dir / item['path']}`; SHA256: `{item['sha256']}`.\n")
    print(f"Official test feature cache completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
