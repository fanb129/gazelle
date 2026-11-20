"""R2: probe, then cache native-grid frozen last-layer RGB features."""

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
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from cache_single_frame import load_frozen_checkpoint
from prepare_development import artifact, verified_train_cache
from stag.cache import argmax_points, file_sha256, write_json
from stag.feature_cache import (feature_contract, select_probe_records, unique_image_records,
                                write_feature_cache)


class UniqueImages(Dataset):
    def __init__(self, records, data_path, transform):
        self.records, self.data_path, self.transform = records, Path(data_path), transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.data_path / record["path"]) as image:
            image = self.transform(image.convert("RGB"))
        return image, record


def collate_images(batch):
    images, records = zip(*batch)
    return torch.stack(images), list(records)


def feature_sources(r1_report, r1_review, cache_report, recovery_report, reference_report):
    r1 = json.loads(Path(r1_report).read_text())
    review = json.loads(Path(r1_review).read_text())
    if (not r1["passed"] or review["status"] != "accepted_for_R2_preparation"
            or review["source_report"]["sha256"] != file_sha256(r1_report)):
        raise ValueError("R2 requires the accepted R1 report and its matching manual review")
    if file_sha256(cache_report) != r1["sources"]["cache_report"]["sha256"]:
        raise ValueError("Prediction report does not match the accepted R1 preparation")
    run, index, arrays, coverage = verified_train_cache(
        r1["cache"]["path"], cache_report, recovery_report, reference_report)
    manifest_path = Path(r1["artifacts"]["split_manifest"]["path"])
    if file_sha256(manifest_path) != r1["artifacts"]["split_manifest"]["sha256"]:
        raise ValueError("R1 split manifest SHA256 mismatch")
    split = json.loads(manifest_path.read_text())
    sources = {"r1_report_sha256": file_sha256(r1_report),
               "r1_review_sha256": file_sha256(r1_review),
               "split_manifest_sha256": file_sha256(manifest_path),
               "prediction_index_sha256": run["cache"]["index_sha256"],
               "prediction_cache_dir": run["cache"]["path"],
               "checkpoint_sha256": run["checkpoint"]["sha256"],
               "backbone_checkpoint_sha256": run["backbone_checkpoint"]["sha256"]}
    return r1, run, index, arrays, coverage, split, sources


def probe_features(model, loader, prediction_arrays, total_images, cache_dir, device):
    """Measure actual tensors and agreement with retained FP32 predictions."""
    contract = None
    map_error = score_error = quant_max = quant_squares = feature_squares = 0.0
    elements = heads = zero_center_heads = argmax_changes = sampled_images = 0
    min_patch_area = float("inf")
    sampled_paths = []
    with torch.inference_mode():
        for images, records in loader:
            images = images.to(device)
            pyramid = model.backbone(images)
            current = pyramid[-1]
            observed = feature_contract(current, model.backbone.out_indices[-1])
            if contract is not None and contract != observed:
                raise ValueError("Probe batches have inconsistent native feature contracts")
            contract = observed
            boxes = [[person["bbox_norm"] for person in record["people"]] for record in records]
            output = model.forward_from_features({"images": images, "bboxes": boxes}, pyramid)
            maps = torch.cat(output["heatmap"]).cpu().numpy()
            scores = torch.cat(output["inout"]).cpu().numpy()
            positions = [person["global_index"] for record in records for person in record["people"]]
            reference_maps = prediction_arrays["heatmap"][positions]
            reference_scores = prediction_arrays["inout"][positions]
            map_error = max(map_error, float(np.abs(maps - reference_maps).max()))
            score_error = max(score_error, float(np.abs(scores - reference_scores).max()))
            argmax_changes += int(np.any(argmax_points(maps) != argmax_points(reference_maps), axis=1).sum())
            raw = current.float().cpu().numpy()
            stored = raw.astype(np.float16)
            if not np.isfinite(raw).all() or not np.isfinite(stored).all():
                raise ValueError("Probe features are non-finite or cannot be stored in FP16")
            difference = stored.astype(np.float32) - raw
            quant_max = max(quant_max, float(np.abs(difference).max()))
            quant_squares += float(np.square(difference, dtype=np.float64).sum())
            feature_squares += float(np.square(raw, dtype=np.float64).sum())
            elements += raw.size
            height, width = contract["grid"]
            for image_boxes in boxes:
                for left, top, right, bottom in image_boxes:
                    min_patch_area = min(min_patch_area, (right - left) * width * (bottom - top) * height)
                    xhits = ((np.arange(width) + 0.5) / width >= left) & ((np.arange(width) + 0.5) / width < right)
                    yhits = ((np.arange(height) + 0.5) / height >= top) & ((np.arange(height) + 0.5) / height < bottom)
                    zero_center_heads += int(not (xhits.any() and yhits.any()))
                    heads += 1
            sampled_images += len(records)
            sampled_paths.extend(record["path"] for record in records)
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = int(total_images * contract["channels"] * np.prod(contract["grid"]) * 2)
    required_bytes = estimated_bytes + 64 * 1024 * 1024
    disk_free = shutil.disk_usage(cache_dir.parent).free
    checks = {"model_all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
              "model_in_eval": not model.training,
              "fp32_heatmaps_agree_with_R1": map_error <= 1e-5,
              "fp32_inout_agrees_with_R1": score_error <= 1e-5,
              "argmax_agrees_with_R1": argmax_changes == 0,
              "native_grid_kept": contract["spatial_pooling"] == "none",
              "fp16_features_finite": True,
              "disk_space_for_native_cache": disk_free >= required_bytes}
    return {"contract": contract, "sampled_images": sampled_images, "sampled_people": heads,
            "sampled_image_paths": sampled_paths, "base_prediction_comparison": {
                "absolute_tolerance": 1e-5, "heatmap_max_abs_error": map_error,
                "inout_max_abs_error": score_error, "argmax_changed_people": argmax_changes,
                "retained_prediction_policy": "original_R1_FP32_arrays_unchanged"},
            "fp16_quantization": {"max_abs_error": quant_max,
                "rmse": float(np.sqrt(quant_squares / elements)),
                "relative_l2": float(np.sqrt(quant_squares / feature_squares)) if feature_squares else 0.0,
                "interpretation": "lossy_feature_storage; not_lossless_FP32_features"},
            "head_grid_coverage": {"minimum_bbox_area_in_patch_units": min_patch_area,
                "heads_with_no_patch_center_inside": zero_center_heads,
                "pooling_policy": "fractional_bbox_patch_overlap; no_coarse_grid_pooling"},
            "storage": {"total_unique_images": total_images, "estimated_feature_bytes": estimated_bytes,
                        "required_bytes_with_metadata_reserve": required_bytes,
                        "available_bytes": disk_free, "target_parent": str(cache_dir.parent.resolve())},
            "checks": checks, "passed": all(checks.values())}


def backbone_batches(model, loader, device):
    with torch.inference_mode():
        for number, (images, _) in enumerate(loader, start=1):
            yield model.backbone(images.to(device))[-1].float().cpu().numpy()
            if number % 100 == 0:
                print(f"Cached {number}/{len(loader)} unique-image batches", flush=True)


def code_hashes():
    paths = [Path(__file__), REPO_ROOT / "cscwd/code/stag/feature_cache.py",
             REPO_ROOT / "cscwd/code/scripts/cache_single_frame.py",
             REPO_ROOT / "gazelle/backbone.py", REPO_ROOT / "gazelle/model.py"]
    return {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths}


def verify_probe(probe, sources, hashes):
    if not probe["passed"] or probe["sources"] != sources or probe["code_sha256"] != hashes:
        raise ValueError("Accepted probe, source identities or feature code differ")
    return probe["contract"]


def write_summary(report, path):
    lines = [f"# R2 frozen feature {report['mode']}", "",
             f"- passed: `{report['passed']}`", f"- contract: `{json.dumps(report['contract'])}`",
             "- 原始 R1 heatmap/In/Out float32 缓存保留；特征为显式有损 FP16，未训练时序模块。"]
    if report["mode"] == "probe":
        for key in ("base_prediction_comparison", "fp16_quantization", "head_grid_coverage", "storage", "checks"):
            lines.append(f"- {key}: `{json.dumps(report[key])}`")
        lines.append("- 下一步：回传本报告，确认原生网格、量化与存储契约后再运行 cache。")
    else:
        lines.append(f"- cache: `{json.dumps(report['cache'])}`")
        lines.append("- 下一步：普通因果联合头 smoke train；不是最终模型或 test 结果。")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "cache"], required=True)
    for name in ("data-path", "r1-report", "r1-review", "cache-report", "recovery-report",
                 "reference-report", "checkpoint", "output-dir", "report", "summary", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--probe-images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    r1, source, index, arrays, coverage, split, sources = feature_sources(
        args.r1_report, args.r1_review, args.cache_report, args.recovery_report, args.reference_report)
    if file_sha256(args.checkpoint) != source["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from the accepted R1 base model")
    backbone_path = REPO_ROOT / "checkpoints/dinov3_vitb16_pretrain.pth"
    if file_sha256(backbone_path) != source["backbone_checkpoint"]["sha256"]:
        raise ValueError("Backbone weights differ from the accepted R1 checkpoint")
    records = unique_image_records(index["tracks"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from gazelle.model import get_gazelle_model
    model, transform = get_gazelle_model(source["model"], use_sasa=True, use_ggsf=True, use_aux=False)
    load_frozen_checkpoint(model, args.checkpoint)
    model.to(args.device)
    hashes = code_hashes()
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                     "collate_fn": collate_images, "shuffle": False}
    if args.mode == "probe":
        selected = select_probe_records(records, split["partitions"]["train"]["track_ids"],
                                        r1["diagnosis"]["selected"], args.probe_images, args.seed)
        loader = DataLoader(UniqueImages(selected, args.data_path, transform), **loader_kwargs)
        result = probe_features(model, loader, arrays, len(records), args.output_dir, args.device)
        result["probe_selection"] = selected
    else:
        if args.probe_report is None:
            raise ValueError("Cache mode requires --probe-report from an accepted probe run")
        probe = json.loads(args.probe_report.read_text())
        contract = verify_probe(probe, sources, hashes)
        expected_bytes = probe["storage"]["required_bytes_with_metadata_reserve"]
        if shutil.disk_usage(args.output_dir.parent).free < expected_bytes:
            raise ValueError("Insufficient disk space for the accepted native-grid cache")
        if probe["runtime"]["transform"] != repr(transform) or probe["runtime"]["torch"] != torch.__version__:
            raise ValueError("Transform or Torch runtime differs from the accepted feature probe")
        loader = DataLoader(UniqueImages(records, args.data_path, transform), **loader_kwargs)
        cache = write_feature_cache([record["path"] for record in records],
            backbone_batches(model, loader, args.device), args.output_dir, contract, sources)
        result = {"contract": contract, "cache": cache, "passed": cache["passed"],
                  "probe_source": artifact(args.probe_report)}
    result.update({"task": "R2_features", "mode": args.mode, "sources": sources,
                   "coverage": coverage, "code_sha256": hashes,
                   "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                               "device": args.device, "batch_size": args.batch_size,
                               "device_name": torch.cuda.get_device_name(torch.device(args.device))
                                   if torch.device(args.device).type == "cuda" else "CPU",
                               "num_workers": args.num_workers, "transform": repr(transform),
                               "precision": "float32_forward; FP16_feature_storage; no_autocast; tf32_disabled"},
                   "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "elapsed_seconds": time.perf_counter() - started})
    write_json(args.report, result)
    write_summary(result, args.summary)
    if not result["passed"]:
        raise SystemExit(f"R2 feature probe verification failed; see {args.report}")
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R2 frozen feature {args.mode}\n\n- UTC: `{result['created_at_utc']}`; commit: `{result['git_commit']}`.\n"
                     f"- Command: `{result['command']}`.\n- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
                     f"- Contract: `{json.dumps(result['contract'])}`; original FP32 predictions retained.\n")
        if args.mode == "cache":
            for item in result["cache"]["files"].values():
                stream.write(f"- Server artifact: `{args.output_dir / item['path']}`; SHA256: `{item['sha256']}`.\n")
            stream.write(f"- Cache index: `{args.output_dir / 'index.json'}`; SHA256: `{result['cache']['index_sha256']}`.\n")
    print(f"R2 feature {args.mode} completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
