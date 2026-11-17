"""R1: group VAT train, freeze development protocols, and export diagnosis cases."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from data_prep.development_split import build_development_split
from eval_temporal import verify_completed_cache
from stag.cache import file_sha256, load_cache, write_json
from stag.development_diagnostics import render_development_case, select_development_cases
from stag.development_protocol import development_selection_plan
from stag.metrics import (STANDARD_PROTOCOL, TemporalConfig, evaluate_predictions,
                          evaluate_temporal, temporal_protocol)
from stag.state_metrics import evaluate_state_temporal, state_temporal_protocol


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def verified_train_cache(cache_dir, cache_report, recovery_report, reference_report):
    run = json.loads(Path(cache_report).read_text())
    recovery = json.loads(Path(recovery_report).read_text())
    reference = json.loads(Path(reference_report).read_text())
    verify_completed_cache(cache_dir, run)
    index, arrays = load_cache(cache_dir)
    train = recovery["splits"]["train"]
    if not recovery["passed"] or not train["passed"] or not reference["passed"]:
        raise ValueError("T1 train recovery and checkpoint reference reports must be complete")
    if (run["model"] != "gazelle_dinov3_vitb16_inout"
            or run["model"] != reference["model"]
            or any(run[key]["sha256"] != reference[key]["sha256"]
                   for key in ("checkpoint", "backbone_checkpoint"))
            or run["protocol"] != reference["protocol"]):
        raise ValueError("Train cache does not use the confirmed frozen ViT-B configuration")
    track_path = Path(run["track_json"]["path"])
    if not (file_sha256(track_path) == run["track_json"]["sha256"] == train["output"]["sha256"]):
        raise ValueError("Train track JSON SHA256 differs from T1 recovery")
    sequences = json.loads(track_path.read_text())
    from stag.cache import index_tracks
    expected_tracks, _ = index_tracks(sequences)
    if index["tracks"] != expected_tracks:
        raise ValueError("Cached track/frame records differ from the T1 train JSON")
    for key in ("model", "checkpoint", "backbone_checkpoint", "track_json", "protocol"):
        if index["metadata"][key] != run[key]:
            raise ValueError(f"Cache metadata differs from its report: {key}")
    coverage = {key: run["cache"][key] for key in ("tracks", "person_frames", "unique_image_frames")}
    actual = {"tracks": len(index["tracks"]),
              "person_frames": sum(len(track["frames"]) for track in index["tracks"]),
              "unique_image_frames": len({frame["path"] for track in index["tracks"] for frame in track["frames"]})}
    if coverage != actual or any(actual[key] != train[key] for key in actual):
        raise ValueError("Train prediction coverage differs from the verified T1 train counts")
    return run, index, arrays, coverage


def standard_metrics(tracks, arrays):
    """Keep the existing metric implementation; define an out-only partition."""
    labels = [frame["inout"] for track in tracks for frame in track["frames"]]
    if any(labels):
        return evaluate_predictions(tracks, arrays["heatmap"], arrays["inout"])
    return {"auc": None, "l2": None, "inout_ap": None,
            "person_frames": len(labels), "localization_frames": 0}


def run_preparation(data_path, cache_dir, cache_report, recovery_report, reference_report,
                    output_dir, preview_dir, validation_fraction=0.2, seed=20261117,
                    cases_per_category=3, preview_count=12, context_before=10,
                    context_after=20, source_video_map=None, snapshots=5):
    source, index, arrays, coverage = verified_train_cache(
        cache_dir, cache_report, recovery_report, reference_report)
    split = build_development_split(index["tracks"], validation_fraction, seed, source_video_map)
    by_id = {track["track_id"]: track for track in index["tracks"]}
    partitions = {name: [by_id[identity] for identity in part["track_ids"]]
                  for name, part in split["partitions"].items()}
    config = TemporalConfig()
    selection = select_development_cases(
        partitions["train"], arrays["argmax_xy"], arrays["inout"], config,
        cases_per_category=cases_per_category, seed=seed,
        context_before=context_before, context_after=context_after)
    output_dir, preview_dir = Path(output_dir), Path(preview_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    preview_dir.mkdir(parents=True, exist_ok=False)
    split_path = output_dir / "split_manifest.json"
    plan_path = output_dir / "selection_config.json"
    diagnosis_path = output_dir / "diagnosis_manifest.json"
    write_json(split_path, split)
    plan = development_selection_plan()
    plan.update({"split_manifest_sha256": file_sha256(split_path),
                 "standard_protocol": {**STANDARD_PROTOCOL, "no_GT_inframe_partition": "AUC_L2_AP_null"},
                 "joint_temporal_protocol": temporal_protocol(config),
                 "state_temporal_protocol": state_temporal_protocol(config)})
    write_json(plan_path, plan)
    baseline, details = {}, {}
    for name, tracks in partitions.items():
        print(f"Evaluating development {name}: {len(tracks)} tracks", flush=True)
        temporal = evaluate_temporal(tracks, arrays["argmax_xy"], arrays["inout"], config)
        state = evaluate_state_temporal(tracks, arrays["inout"], config)
        baseline[name] = {"standard": standard_metrics(tracks, arrays),
                          "temporal": temporal["summary"], "state": state["summary"]}
        details[name] = {"joint": {key: temporal[key] for key in ("tracks", "events")},
                         "state": {key: state[key] for key in ("tracks", "events")}}
    details_path = output_dir / "baseline_details.json"
    write_json(details_path, details)
    preview_files = []
    # Round-robin categories so a partial preview limit still covers each stratum.
    categories = list(selection["counts"])
    grouped = {category: [case for case in selection["selected"] if case["category"] == category]
               for category in categories}
    ordered = [cases[position] for position in range(cases_per_category)
               for cases in grouped.values() if position < len(cases)]
    preview_ids = {case["case_id"] for case in ordered[:preview_count]}
    source_images = {}
    for case in selection["selected"]:
        track = by_id[case["track_id"]]
        case_dir = output_dir / "cases" / case["case_id"]
        case_dir.mkdir(parents=True)
        frames = track["frames"][case["context_start"] - track["start"]:
                                  case["context_stop"] - track["start"]]
        start, stop = case["context_start"], case["context_stop"]
        write_json(case_dir / "case_data.json", {"case": case, "frames": frames,
                   "argmax_xy": arrays["argmax_xy"][start:stop].tolist(),
                   "inout": arrays["inout"][start:stop].tolist()})
        path = case_dir / "frames.png"
        rendered = render_development_case(case, track, arrays["argmax_xy"], arrays["inout"],
                                           Path(data_path), path, snapshots=snapshots)
        case["render"] = {**rendered, **artifact(path)}
        for position in rendered["snapshot_global_indices"]:
            image_path = Path(data_path) / track["frames"][position - track["start"]]["path"]
            source_images[str(image_path)] = file_sha256(image_path)
        if case["case_id"] in preview_ids:
            preview = preview_dir / f"{case['case_id']}.png"
            with Image.open(path) as image:
                image.thumbnail((1280, 900))
                image.save(preview)
            preview_files.append({"case_id": case["case_id"], **artifact(preview)})
    write_json(diagnosis_path, {**selection, "source_snapshot_sha256": source_images})
    checks = {"train_sources_and_coverage_verified": True, **split["checks"],
              "diagnosis_uses_development_train_only": all(
                  case["track_id"] in set(split["partitions"]["train"]["track_ids"])
                  for case in selection["selected"]),
              "cached_prediction_arrays_read_only": all(not array.flags.writeable for array in arrays.values())}
    return {
        "task": "R1", "schema_version": 1, "passed": all(checks.values()),
        "stage_status": "awaiting_local_review", "coverage": coverage,
        "sources": {name: artifact(path) for name, path in (
            ("cache_report", cache_report), ("recovery_report", recovery_report),
            ("checkpoint_reference_report", reference_report))},
        "cache": source["cache"], "model": source["model"], "checkpoint": source["checkpoint"],
        "split": {key: value for key, value in split.items() if key != "groups"} | {
            "partitions": {name: {key: value for key, value in part.items() if key != "track_ids"}
                           for name, part in split["partitions"].items()}},
        "baseline": baseline, "selection_plan": plan,
        "diagnosis": {"counts": selection["counts"], "selected": selection["selected"],
                      "preview_files": preview_files, "protocol": selection["protocol"]},
        "artifacts": {name: artifact(path) for name, path in (
            ("split_manifest", split_path), ("selection_config", plan_path),
            ("diagnosis_manifest", diagnosis_path), ("details", details_path))},
        "checks": checks,
        "interpretation": "passed=machine_preparation_only; human_review_and_source_group_acceptance_pending",
    }


def summary_markdown(report, summary_path):
    lines = ["# R1 开发集分组与单帧诊断", "",
             "机器准备完成；人物/目标诊断尚未人工填写，R1 阶段尚未验收。", "",
             f"- 分组：`{report['split']['grouping']['default_semantics']}`。",
             f"- 配置：seed `{report['split']['seed']}`，validation 组比例 "
             f"`{report['split']['requested_validation_group_fraction']}`（请求值）。", "",
             "| 集合 | groups | tracks | 人物帧 | 图像帧 | AUC | L2 | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, part in report["split"]["partitions"].items():
        counts, metrics = part["counts"], report["baseline"][name]["standard"]
        lines.append(f"| {name} | {len(part['group_ids'])} | {counts['tracks']} | {counts['person_frames']} | "
                     f"{counts['unique_image_frames']} | {metrics['auc']} | {metrics['l2']} | {metrics['inout_ap']} |")
    lines.append("")
    for name, part in report["split"]["partitions"].items():
        lines.append(f"- {name} groups：`{json.dumps(part['group_ids'], ensure_ascii=False)}`。")
    lines.extend(["", "| 集合 | 准确稳定占比 | 状态错误率 | 稳定 GT 状态翻转率 | 状态响应率 | joint gaze_shift 响应率 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, result in report["baseline"].items():
        state, temporal = result["state"], result["temporal"]
        lines.append(f"| {name} | {temporal['stability']['accurate_stable_ratio']} | "
                     f"{state['state_error']['error_rate']} | {state['stable_state_flips']['flip_rate']} | "
                     f"{state['response']['all']['response_rate']} | {temporal['response']['gaze_shift']['response_rate']} |")
    lines.extend(["", "## 训练组诊断抽样", "",
                  "代理事件不自动标注为真实换目标。每例 review_status 保持 unreviewed，回传后查看 RGB 与预测再填写。", "",
                  "| 类别 | 候选 | 独立 clip | 选中 |", "|---|---:|---:|---:|"])
    for category, counts in report["diagnosis"]["counts"].items():
        lines.append(f"| {category} | {counts['candidates']} | {counts['candidate_clips']} | {counts['selected']} |")
    lines.extend(["", "## 小预览", ""])
    for preview in report["diagnosis"]["preview_files"]:
        relative_path = Path(os.path.relpath(preview["path"], Path(summary_path).resolve().parent)).as_posix()
        lines.extend([f"### {preview['case_id']}", "",
                      f"![frames](<{relative_path}>)", ""])
    lines.extend(["## 协议与后续", "",
                  "旧 joint 时序协议与独立 state_temporal_v1 分列。AP 使用连续分数；状态窗口不被画内坐标跳变截断。",
                  "完整分组/选择配置/候选诊断/逐事件数据留服务器，路径及 SHA256 见 JSON。",
                  "下一步：本地核对分组和来源，查看所选训练片段并记录诊断，再验收 R1；尚未开始 R2 或模型训练。", ""])
    return "\n".join(lines)


def record_provenance(path, report, report_path):
    lines = ["", "## R1 开发集分组与单帧诊断（服务器机器准备）", "",
             f"- UTC：`{report['created_at_utc']}`；commit：`{report['git_commit']}`。",
             f"- 命令：`{report['command']}`。", f"- 小报告：`{report_path}`。",
             f"- 机器检查：`{report['passed']}`；阶段状态：`{report['stage_status']}`；人工诊断未完成。"]
    for name, item in report["artifacts"].items():
        lines.append(f"- {name}（服务器保留）：`{item['path']}`；SHA256：`{item['sha256']}`。")
    for name, item in report["sources"].items():
        lines.append(f"- {name}：`{item['path']}`；SHA256：`{item['sha256']}`。")
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-path", "cache-dir", "cache-report", "recovery-report", "reference-report",
                 "output-dir", "preview-dir", "report", "summary", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-video-map", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--cases-per-category", type=int, default=3)
    parser.add_argument("--preview-cases", type=int, default=12)
    parser.add_argument("--context-before", type=int, default=10)
    parser.add_argument("--context-after", type=int, default=20)
    parser.add_argument("--snapshots", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    mapping = json.loads(args.source_video_map.read_text()) if args.source_video_map else None
    report = run_preparation(args.data_path, args.cache_dir, args.cache_report, args.recovery_report,
        args.reference_report, args.output_dir, args.preview_dir, args.validation_fraction,
        args.seed, args.cases_per_category, args.preview_cases, args.context_before,
        args.context_after, mapping, args.snapshots)
    code_paths = [Path(__file__), REPO_ROOT / "gazelle/data_splits.py",
                  *(REPO_ROOT / f"cscwd/code/{name}" for name in (
                      "data_prep/development_split.py", "stag/development_protocol.py",
                      "stag/development_diagnostics.py", "stag/state_metrics.py", "stag/metrics.py",
                      "stag/cache.py", "scripts/eval_temporal.py"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in code_paths},
                   "elapsed_seconds": time.perf_counter() - started})
    if args.source_video_map:
        report["sources"]["source_video_map"] = artifact(args.source_video_map)
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report, args.summary))
    record_provenance(args.provenance, report, args.report.resolve())
    print(f"R1 preparation completed; stage=awaiting_local_review; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
