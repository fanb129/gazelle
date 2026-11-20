"""Portable offline image review packages from cached single-frame predictions."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from cscwd.code.stag.failure_cases import PredictionCache, select_cases


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = {
    "purpose": "human_review_of_frozen_GazeSpot_single_frame_failures_and_controls",
    "splits": ["train", "test"], "validation_split": None,
    "test_used_for_manual_analysis": True,
    "test_used_for_method_development": True,
    "test_is_independent_validation": False,
    "sampling": "stratified_examples_not_population_failure_prevalence",
    "history": "same_person_contiguous_original_frames_current_and_past_only",
    "gt_usage": "offline_case_selection_and_reveal_for_human_review_only",
    "inout_threshold_usage": "case_selection_only_not_AP_optimization",
    "heatmap_preview": "uint8_round_255_times_original_probability_no_per_frame_normalization",
    "image_preview": "aspect_preserving_resize_no_crop_no_GT_baked_into_images",
    "head_preview": "raw_head_box_expanded_30_percent_each_side_original_RGB_max_side_192",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cache_source(cache):
    files = {}
    for name in ("heatmap", "inout", "argmax_xy"):
        artifact = cache.index["files"][name]
        path = cache.root / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"Prediction cache SHA256 mismatch: {path}")
        files[name] = {**artifact, "path": str(path.resolve()), "sha256_verified": True}
    return {
        "path": str(cache.root.resolve()), "index_sha256": sha256(cache.root / "index.json"),
        "metadata": cache.index["metadata"], "person_frames": len(cache.frames),
        "files": files,
    }


def export_assets(cache, cases, data_root, output_dir, image_max_side, jpeg_quality):
    rgb_dir, heatmap_dir = output_dir / "assets/rgb", output_dir / "assets/heatmap"
    head_dir = output_dir / "assets/head"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)
    for case_number, case in enumerate(cases, start=1):
        for frame in case["frames"]:
            key = hashlib.sha256(frame["path"].encode()).hexdigest()[:20]
            image_path = rgb_dir / (key + ".jpg")
            head_path = head_dir / f"{cache.split}_{frame['row']:07d}.jpg"
            if not image_path.exists() or not head_path.exists():
                with Image.open(data_root / frame["path"]) as image:
                    rgb = image.convert("RGB")
                    if not image_path.exists():
                        preview = rgb.copy()
                        preview.thumbnail((image_max_side, image_max_side), Image.Resampling.LANCZOS)
                        preview.save(image_path, quality=jpeg_quality)
                    box = np.asarray(frame["bbox_norm"], dtype=float)
                    padding = (box[2:] - box[:2]) * 0.3
                    first = np.floor((box[:2] - padding) * rgb.size).astype(int)
                    last = np.ceil((box[2:] + padding) * rgb.size).astype(int)
                    crop = rgb.crop((max(0, first[0]), max(0, first[1]),
                                     min(rgb.width, last[0]), min(rgb.height, last[1])))
                    crop.thumbnail((192, 192), Image.Resampling.LANCZOS)
                    crop.save(head_path, quality=jpeg_quality)
            heatmap_path = heatmap_dir / f"{cache.split}_{frame['row']:07d}.png"
            if not heatmap_path.exists():
                values = np.asarray(cache.heatmap[frame["row"]])
                pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(pixels).save(heatmap_path)
            frame["relative_to_anchor"] = frame["frame_index"] - case["anchor_frame_index"]
            frame["image"] = image_path.relative_to(output_dir).as_posix()
            frame["heatmap"] = heatmap_path.relative_to(output_dir).as_posix()
            frame["head_crop"] = head_path.relative_to(output_dir).as_posix()
        print(f"EXPORT {cache.split}: case {case_number}/{len(cases)}", flush=True)


def annotation_template(cases, package_id):
    return {
        "schema_version": 1, "artifact_type": "cscwd_failure_review_labels",
        "package_id": package_id, "exported_at": None,
        "labels": [{
            "case_id": case["case_id"], "current_difficulty": "", "current_guess": "",
            "history_evidence": "", "history_notes": "", "failure_mechanism": "",
            "notes": "", "gt_revealed_at": "", "updated_at": "",
        } for case in cases],
    }


def build_review_package(data_root, train_cache, test_cache, output_dir, report_dir,
                         provenance, seed=0, history_frames=16, max_history_frames=64,
                         counts=None, score_threshold=0.5, large_l2_quantile=0.9,
                         max_per_track=2, max_per_video=6, image_max_side=768,
                         jpeg_quality=85, archive=False, command=None):
    """CPU-only export, with no model loading, new inference or training."""
    data_root = Path(data_root).resolve()
    output_dir, report_dir = Path(output_dir).resolve(), Path(report_dir).resolve()
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Use fresh case-package and report directories")
    if history_frames < 0 or max_history_frames < history_frames:
        raise ValueError("Maximum history must be at least the nonnegative default history")
    caches = {"train": PredictionCache(train_cache, "train"), "test": PredictionCache(test_cache, "test")}
    sources = {split: cache_source(cache) for split, cache in caches.items()}
    for key in ("checkpoint", "backbone_checkpoint"):
        left = sources["train"]["metadata"][key]["sha256"]
        right = sources["test"]["metadata"][key]["sha256"]
        if left != right:
            raise ValueError(f"Train and test must use the same original GazeSpot {key}")
    selections = {
        split: select_cases(cache, seed=seed, counts=counts, history_frames=history_frames,
                            max_history_frames=max_history_frames, max_per_track=max_per_track,
                            max_per_video=max_per_video, score_threshold=score_threshold,
                            large_l2_quantile=large_l2_quantile)
        for split, cache in caches.items()
    }
    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for split, cache in caches.items():
        export_assets(cache, selections[split]["cases"], data_root, output_dir, image_max_side, jpeg_quality)
    cases = [case for selection in selections.values() for case in selection["cases"]]
    package_identity = {
        "source_indexes": {split: source["index_sha256"] for split, source in sources.items()},
        "selection_configs": {split: value["config"] for split, value in selections.items()},
        "case_ids": [case["case_id"] for case in cases],
    }
    package_id = "gazespot-review-" + hashlib.sha256(
        json.dumps(package_identity, sort_keys=True).encode()
    ).hexdigest()[:20]
    payload = {
        "schema_version": 1, "package_id": package_id,
        "title": "GazeSpot 失败案例人工分析 · Train / Test",
        "protocol": PROTOCOL, "history_frames": history_frames,
        "max_history_frames": max_history_frames, "cases": cases,
    }
    write_json(output_dir / "case_manifest.json", payload)
    write_json(output_dir / "annotations_blank.json", annotation_template(cases, package_id))
    template = (Path(__file__).with_name("review_viewer.html")).read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("__REVIEW_DATA__", encoded)
    (output_dir / "index.html").write_text(html)
    manifest_sha = sha256(output_dir / "case_manifest.json")
    selected_rows = [{key: value for key, value in case.items() if key != "frames"} for case in cases]
    write_json(report_dir / "selected_cases.json", {
        "package_id": package_id, "protocol": PROTOCOL, "cases": selected_rows,
    })
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": command or shlex.join([sys.executable, *sys.argv]), "package_id": package_id,
        "protocol": PROTOCOL, "sources": sources,
        "selection": {split: {"summary": value["summary"], "config": value["config"]}
                      for split, value in selections.items()},
        "output_dir": str(output_dir), "image_max_side": image_max_side, "jpeg_quality": jpeg_quality,
        "case_count": len(cases), "case_manifest_sha256": manifest_sha,
        "index_html_sha256": sha256(output_dir / "index.html"),
    }
    if archive:
        archive_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
        report["archive"] = {"path": str(archive_path), "sha256": sha256(archive_path),
                             "bytes": archive_path.stat().st_size}
    write_json(report_dir / "review_report.json", report)
    lines = ["# GazeSpot 人工案例包\n\n",
             "官方 train 和 test 均用于人工分析/方法开发；无 validation。后续 test 不是独立验证。\n\n",
             f"Package: `{package_id}`。打开 `{output_dir / 'index.html'}`，无需网络或服务。\n\n",
             "默认只显示当前帧，不揭示 GT/类别/L2。先填写当前判断，再展开历史记录证据，最后揭示 GT。\n\n",
             "浏览器本地保存标注；请主动导出 JSON/CSV 备份。JSON 可导回同一 package。\n\n",
             "| Split | Category | Candidates | Requested | Selected | Shortfall |\n|---|---|---|---|---|---|\n"]
    for split, selection in selections.items():
        for category, values in selection["summary"]["categories"].items():
            lines.append(f"| {split} | {category} | {values['candidate_count']} | {values['requested']} | "
                         f"{values['selected']} | {values['shortfall']} |\n")
    lines.append("\n按类别抽样的比例不代表总体失败占比。所有帧来自同一人物的连续原帧，包含当前和过去，无未来帧。\n")
    (report_dir / "review_report.md").write_text("".join(lines))
    Path(provenance).parent.mkdir(parents=True, exist_ok=True)
    with Path(provenance).open("a") as stream:
        stream.write(f"\n## {report_dir.name} / 人工失败案例包\n\n"
                     f"- Code: `{report['git_commit']}`; command: `{report['command']}`.\n"
                     "- Train and test both used for manual analysis and method development; no validation.\n"
                     f"- Package: `{package_id}`; case count: {len(cases)}.\n"
                     f"- Manifest: `{output_dir / 'case_manifest.json'}`; SHA256: `{manifest_sha}`.\n"
                     f"- Viewer: `{output_dir / 'index.html'}`; SHA256: `{report['index_html_sha256']}`.\n")
        for split, source in sources.items():
            stream.write(f"- {split} prediction index: `{Path(source['path']) / 'index.json'}`; "
                         f"SHA256: `{source['index_sha256']}`.\n")
            for name, artifact in source["files"].items():
                stream.write(f"  - {name}: `{artifact['path']}`; SHA256: `{artifact['sha256']}` (verified).\n")
        if archive:
            stream.write(f"- Archive: `{report['archive']['path']}`; SHA256: `{report['archive']['sha256']}`.\n")
    print(f"Review package complete: {len(cases)} cases, {output_dir / 'index.html'}", flush=True)
    return report
