"""T6: deterministic success/failure diagnostics from immutable T5 predictions."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from eval_temporal import event_signature, verify_completed_cache
from stag.cache import file_sha256, load_cache, write_json
from stag.case_selection import CATEGORY_ORDER, select_cases
from stag.metrics import TemporalConfig, temporal_protocol
from stag.plotting import render_case


DISPLAY_METHODS = ("single_frame", "stag", "ema_alpha_0p25", "ema_alpha_0p5")


def load_verified_inputs(ablation_report):
    path = Path(ablation_report)
    report = json.loads(path.read_text())
    if not report["passed"] or not all(report["checks"].values()):
        raise ValueError("T5 ablation must be complete before visualization")
    for source in report["sources"].values():
        if file_sha256(source["report_path"]) != source["report_sha256"]:
            raise ValueError(f"Source report SHA256 mismatch: {source['report_path']}")
    for method in report["methods"].values():
        verify_completed_cache(method["cache"]["path"], {"passed": method["cache"]["passed"], "cache": method["cache"]})
    details_path = Path(report["details"]["path"])
    if file_sha256(details_path) != report["details"]["sha256"]:
        raise ValueError("T5 per-track/event details SHA256 mismatch")
    details = json.loads(details_path.read_text())
    config = TemporalConfig(**report["protocol"]["temporal"]["thresholds"])
    if details["temporal_protocol"] != temporal_protocol(config):
        raise ValueError("T5 details and temporal protocol differ")
    index, _ = load_cache(report["methods"]["single_frame"]["cache"]["path"])
    arrays = {}
    reference_events = [event_signature(event) for event in details["methods"]["single_frame"]["events"]]
    for name in DISPLAY_METHODS:
        method_index, arrays[name] = load_cache(report["methods"][name]["cache"]["path"])
        if method_index["tracks"] != index["tracks"]:
            raise ValueError(f"Prediction track records differ: {name}")
        if [event_signature(event) for event in details["methods"][name]["events"]] != reference_events:
            raise ValueError(f"Proxy events or observation windows differ: {name}")
    stag_cache = report["methods"]["stag"]["cache"]
    diagnostics = {name: np.load(Path(stag_cache["path"]) / artifact["path"], mmap_mode="r", allow_pickle=False)
                   for name, artifact in stag_cache["diagnostics"].items()}
    return report, index, arrays, diagnostics, details["methods"], config


def heads_by_image(index):
    heads = defaultdict(dict)
    for track in index["tracks"]:
        for frame in track["frames"]:
            heads[frame["path"]][track["track_id"]] = frame["bbox_norm"]
    return {path: [{"track_id": identity, "bbox_norm": bbox} for identity, bbox in sorted(people.items())]
            for path, people in heads.items()}


def preview_cases(selected, limit):
    """Take one case per available category before additional examples."""
    groups = {category: [case for case in selected if case["category"] == category] for category in CATEGORY_ORDER}
    ordered = []
    for position in range(max((len(cases) for cases in groups.values()), default=0)):
        for cases in groups.values():
            if position < len(cases):
                ordered.append(cases[position]["case_id"])
    return set(ordered[:limit])


def run_visualization(data_path, ablation_report, output_dir, preview_dir, cases_per_group=2,
                      preview_count=5, context_before=10, context_after=20, min_heads=4, snapshots=5, dpi=140):
    source, index, arrays, diagnostics, details, config = load_verified_inputs(ablation_report)
    selection = select_cases(index, arrays, details, config, cases_per_group=cases_per_group,
                             context_before=context_before, context_after=context_after, min_heads=min_heads)
    failure_count = source["paired_response"]["stag_minus_single_frame"]["gaze_shift"]["reference_only_responded_events"]
    benefit_count = source["paired_response"]["ema_alpha_0p25_minus_stag"]["gaze_shift"]["reference_only_responded_events"]
    checks = {
        "t5_sources_and_caches_verified": True,
        "failure_candidates_match_t5": len(selection["candidates"]["single_frame_only"]) == failure_count,
        "all_single_frame_only_failures_selected": selection["counts"]["single_frame_only"]["selected"] == failure_count,
        "gate_benefit_candidates_match_t5": len(selection["candidates"]["stag_over_ema"]) == benefit_count,
        "cached_prediction_arrays_read_only": all(not values.flags.writeable
                                                  for method in arrays.values() for values in method.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"T6 selection does not match T5 event accounting: {checks}")
    output_dir, preview_dir = Path(output_dir), Path(preview_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    preview_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "selection_manifest.json"
    write_json(manifest_path, selection)
    preview_ids = preview_cases(selection["selected"], preview_count)
    tracks = {track["track_id"]: track for track in index["tracks"]}
    frames = [frame for track in index["tracks"] for frame in track["frames"]]
    heads = heads_by_image(index)
    artifacts, source_images = [], {}
    for number, case in enumerate(selection["selected"], start=1):
        print(f"Rendering {number}/{len(selection['selected'])}: {case['category']} / {case['case_id']}", flush=True)
        result = render_case(case, tracks[case["track_id"]], arrays, diagnostics, Path(data_path), heads,
                             output_dir / "cases" / case["case_id"], config, snapshots=snapshots,
                             dpi=dpi, preview_dir=preview_dir if case["case_id"] in preview_ids else None)
        for artifact in result["files"]:
            artifact["size_bytes"] = Path(artifact["path"]).stat().st_size
        artifacts.append({"case": case, **result})
        for position in result["frame_global_indices"]:
            path = Path(data_path) / frames[position]["path"]
            if str(path) not in source_images:
                source_images[str(path)] = file_sha256(path)
    return {
        "source": {"report_path": str(Path(ablation_report).resolve()), "report_sha256": file_sha256(ablation_report),
                   "details": source["details"], "cache_index_sha256": {
                       name: method["cache"]["index_sha256"] for name, method in source["methods"].items()}},
        "model": source["model"], "checkpoint": source["checkpoint"], "stag_weights": source["stag_weights"],
        "data_path": str(Path(data_path).resolve()), "coverage": source["coverage"],
        "selection_protocol": selection["protocol"], "temporal_protocol": temporal_protocol(config),
        "render_protocol": {"methods": list(DISPLAY_METHODS), "heatmap_range": [0, 1],
                            "heatmap_overlay_alpha": 0.38,
                            "prediction_usage": "cached_only; no_forward; no_resmoothing; no_GT_blending",
                            "context": "offline_diagnostics; may_show_frames_after_anchor; no_prediction_state_changes",
                            "heads": "native_T1_ground_truth; queried_person_highlighted",
                            "snapshot_policy": "anchor_then_adjacent_frames_then_context_endpoints_then_uniform_fill",
                            "snapshots": snapshots, "dpi": dpi, "preview_dpi": 70, "preview_case_limit": preview_count},
        "candidate_counts": selection["counts"],
        "single_frame_only_failures": selection["candidates"]["single_frame_only"],
        "manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        "cases": artifacts, "source_images_sha256": source_images,
        "output_dir": str(output_dir.resolve()), "preview_dir": str(preview_dir.resolve()),
        "checks": checks, "passed": True,
    }


def markdown_summary(report, summary_path):
    lines = ["# T6 时序案例诊断", "", f"- T5 来源：`{report['source']['report_path']}`。",
             "- 仅显示已缓存预测；标签用于离线选样和计分，不修改预测。",
             "- 热图统一使用原值色标 [0,1]；包括收益、失响应、滞后、稳定收益与拥挤场景。", "",
             "| 类别 | 候选数 | 完整输出案例数 |", "|---|---:|---:|"]
    for category, counts in report["candidate_counts"].items():
        lines.append(f"| {category} | {counts['candidates']} | {counts['selected']} |")
    lines.extend(["", "## 已输出案例", "",
                  "| case_id | 类别 | track_id | 原帧号 | 上下文全局范围 |", "|---|---|---|---:|---|"])
    for item in report["cases"]:
        case = item["case"]
        lines.append(f"| {case['case_id']} | {case['category']} | {case['track_id']} | "
                     f"{case['anchor_frame_index']} | [{case['context_start']},{case['context_stop']}) |")
    lines.extend(["", "## 小预览", ""])
    for item in report["cases"]:
        previews = [artifact for artifact in item["files"] if artifact["kind"].startswith("preview_")]
        if previews:
            lines.extend([f"### {item['case']['category']} · {item['case']['case_id']}", ""])
            for artifact in previews:
                relative = Path(os.path.relpath(artifact["path"], Path(summary_path).parent)).as_posix()
                lines.extend([f"![{artifact['kind']}]({relative})", ""])
    lines.extend(["完整候选清单、逐帧数据和全分辨率图留在服务器，路径与 SHA256 见 JSON/provenance。",
                  "结合预览、原图与逐帧数据，判断旧峰残留、错误峰写入和门控触发机制。"])
    return "\n".join(lines) + "\n"


def record_provenance(path, report, report_path, summary_path):
    lines = ["", "## T6 时序案例诊断（服务器运行）", "",
             f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
             f"- T5 报告：`{report['source']['report_path']}`；SHA256：`{report['source']['report_sha256']}`。",
             f"- checkpoint：`{report['checkpoint']['path']}`；SHA256：`{report['checkpoint']['sha256']}`。",
             f"- 命令：`{report['command']}`。",
             f"- 小结果：`{report_path}`；摘要：`{summary_path}`；预览：`{report['preview_dir']}`。",
             f"- 完整候选清单：`{report['manifest']['path']}`；SHA256：`{report['manifest']['sha256']}`。"]
    for category, counts in report["candidate_counts"].items():
        lines.append(f"- {category}：候选 {counts['candidates']}，完整输出 {counts['selected']}。")
    for item in report["cases"]:
        lines.append(f"- 案例 `{item['case']['case_id']}`：{item['case']['track_id']} / 原帧号 {item['case']['anchor_frame_index']}。")
        for artifact in item["files"]:
            lines.append(f"  - {artifact['kind']}：`{artifact['path']}`；SHA256：`{artifact['sha256']}`。")
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New server-only directory for complete figures/data")
    parser.add_argument("--preview-dir", type=Path, required=True, help="New repository directory for small PNG previews")
    parser.add_argument("--cases-per-group", type=int, default=2)
    parser.add_argument("--preview-cases", type=int, default=5)
    parser.add_argument("--context-before", type=int, default=10)
    parser.add_argument("--context-after", type=int, default=20)
    parser.add_argument("--min-heads", type=int, default=4)
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t6_visualization.json")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "cscwd/results/t6_visualization.md")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "task": "T6", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "numpy_version": np.__version__,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path)
                        for path in (Path(__file__).resolve(), REPO_ROOT / "cscwd/code/stag/case_selection.py",
                                     REPO_ROOT / "cscwd/code/stag/plotting.py", REPO_ROOT / "cscwd/code/stag/cache.py",
                                     REPO_ROOT / "cscwd/code/stag/metrics.py", REPO_ROOT / "cscwd/code/scripts/eval_temporal.py")},
    }
    report.update(run_visualization(args.data_path, args.ablation_report, args.output_dir, args.preview_dir,
                                    cases_per_group=args.cases_per_group, preview_count=args.preview_cases,
                                    context_before=args.context_before, context_after=args.context_after,
                                    min_heads=args.min_heads, snapshots=args.snapshots, dpi=args.dpi))
    report["elapsed_seconds"] = time.perf_counter() - started
    summary = markdown_summary(report, args.summary.resolve())
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary)
    summary_link = Path(os.path.relpath(args.summary.resolve(), args.preview_dir.resolve())).as_posix()
    (args.preview_dir / "index.md").write_text(f"# T6 previews\n\n[Selection criteria and source records]({summary_link})\n")
    record_provenance(args.provenance, report, args.report.resolve(), args.summary.resolve())
    print(f"T6 visualization completed: {len(report['cases'])} cases; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
