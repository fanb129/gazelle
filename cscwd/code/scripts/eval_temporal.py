"""T4: paired standard and temporal VAT evaluation of completed T2/T3 caches."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from stag.cache import file_sha256, load_cache, write_json
from stag.metrics import (STANDARD_PROTOCOL, TemporalConfig, evaluate_predictions,
                          evaluate_temporal, temporal_protocol)


def verify_completed_cache(cache_dir, report):
    cache_dir = Path(cache_dir)
    if not report["passed"] or file_sha256(cache_dir / "index.json") != report["cache"]["index_sha256"]:
        raise ValueError("Completed cache report or index verification failed")
    artifacts = {**report["cache"]["files"], **report["cache"].get("diagnostics", {})}
    for artifact in artifacts.values():
        if file_sha256(cache_dir / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Cache SHA256 mismatch: {cache_dir / artifact['path']}")


def event_signature(event):
    return (event["track_id"], event["type"], event["event_global_index"],
            event["event_frame_index"], event["window_frames"])


def paired_response_summary(baseline_events, stag_events):
    """Compare delay on exactly the events where both methods respond."""
    results = {}
    for kind in ("all", "enter_frame", "leave_frame", "gaze_shift"):
        pairs = [(base, stag) for base, stag in zip(baseline_events, stag_events)
                 if kind == "all" or base["type"] == kind]
        common_delays = [stag["response_delay_frames"] - base["response_delay_frames"]
                         for base, stag in pairs
                         if base["status"] == stag["status"] == "responded"]
        results[kind] = {
            "events": len(pairs), "both_responded_events": len(common_delays),
            "single_frame_only_responded_events": sum(base["status"] == "responded" and stag["status"] != "responded"
                                                       for base, stag in pairs),
            "stag_only_responded_events": sum(stag["status"] == "responded" and base["status"] != "responded"
                                               for base, stag in pairs),
            "both_censored_events": sum(base["status"] == stag["status"] == "censored" for base, stag in pairs),
            "insufficient_followup_events": sum(base["status"] == "insufficient_followup" for base, _ in pairs),
            "common_responded_delay_delta_frames": {
                "count": len(common_delays),
                **{key: float(function(common_delays)) if common_delays else None
                   for key, function in (("mean", np.mean), ("median", np.median),
                                         ("p90", lambda values: np.percentile(values, 90)),
                                         ("p95", lambda values: np.percentile(values, 95)))},
            },
        }
    return results


def run_evaluation(single_frame_cache, single_frame_report, stag_cache, stag_report, details_dir, config):
    with Path(single_frame_report).open() as stream:
        baseline_run = json.load(stream)
    with Path(stag_report).open() as stream:
        stag_run = json.load(stream)
    verify_completed_cache(single_frame_cache, baseline_run)
    verify_completed_cache(stag_cache, stag_run)
    if (stag_run["source"]["index_sha256"] != baseline_run["cache"]["index_sha256"]
            or stag_run["source"]["report_sha256"] != file_sha256(single_frame_report)):
        raise ValueError("STAG parent cache does not match the supplied single-frame run")
    baseline_index, baseline_arrays = load_cache(single_frame_cache)
    stag_index, stag_arrays = load_cache(stag_cache)
    if baseline_index["tracks"] != stag_index["tracks"]:
        raise ValueError("Single-frame and STAG track records must be identical")
    details_dir = Path(details_dir)
    details_dir.mkdir(parents=True, exist_ok=False)
    standard, temporal, details = {}, {}, {}
    for name, arrays in (("single_frame", baseline_arrays), ("stag", stag_arrays)):
        print(f"Evaluating standard and temporal metrics: {name}", flush=True)
        standard[name] = evaluate_predictions(baseline_index["tracks"], arrays["heatmap"], arrays["inout"])
        result = evaluate_temporal(baseline_index["tracks"], arrays["argmax_xy"], arrays["inout"], config)
        temporal[name] = result["summary"]
        details[name] = {"tracks": result["tracks"], "events": result["events"]}
    checks = {
        "standard_metrics_match_t3": all(
            math.isclose(standard[name][metric], stag_run["metrics"][name][metric], rel_tol=1e-12, abs_tol=1e-12)
            for name in standard for metric in ("auc", "l2", "inout_ap", "person_frames", "localization_frames")),
        "same_eligible_jitter_pairs": temporal["single_frame"]["jitter"]["pairs"] == temporal["stag"]["jitter"]["pairs"],
        "same_eligible_stability_windows": (temporal["single_frame"]["stability"]["eligible_windows"]
                                            == temporal["stag"]["stability"]["eligible_windows"]),
        "same_ground_truth_reference": (temporal["single_frame"]["jitter"]["ground_truth"]
                                         == temporal["stag"]["jitter"]["ground_truth"]
                                         and temporal["single_frame"]["stability"]["stable_gt_frames"]
                                         == temporal["stag"]["stability"]["stable_gt_frames"]),
        "same_proxy_events_and_windows": ([event_signature(event) for event in details["single_frame"]["events"]]
                                          == [event_signature(event) for event in details["stag"]["events"]]),
    }
    if not all(checks.values()):
        raise ValueError(f"Paired temporal evaluation alignment failed: {checks}")
    protocol = temporal_protocol(config)
    details_path = details_dir / "temporal_details.json"
    write_json(details_path, {"task": "T4", "protocol": protocol, "methods": details})
    return {
        "sources": {
            name: {"report_path": str(Path(path).resolve()), "report_sha256": file_sha256(path),
                   "cache": run["cache"]}
            for name, path, run in (("single_frame", single_frame_report, baseline_run),
                                    ("stag", stag_report, stag_run))
        },
        "model": baseline_run["model"], "checkpoint": baseline_run["checkpoint"],
        "track_json": baseline_run["track_json"], "stag_weights": stag_run["config"]["weights"],
        "coverage": {key: baseline_run["cache"][key] for key in ("tracks", "person_frames", "unique_image_frames")},
        "standard_protocol": STANDARD_PROTOCOL, "temporal_protocol": protocol,
        "standard_metrics": standard, "temporal_metrics": temporal,
        "paired_response": paired_response_summary(details["single_frame"]["events"], details["stag"]["events"]),
        "details": {"path": str(details_path.resolve()), "sha256": file_sha256(details_path)},
        "checks": checks, "passed": True,
    }


def record_provenance(path, report, report_path):
    lines = [
        "", "## T4 标准与时序指标（服务器运行）", "",
        f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
        f"- checkpoint：`{report['checkpoint']['path']}`；SHA256：`{report['checkpoint']['sha256']}`。",
        f"- STAG 权重：`{report['stag_weights']}`；评测协议和阈值完整保存在小报告。",
        f"- 命令：`{report['command']}`。",
        f"- 小报告：`{report_path}`。",
        f"- 逐轨迹/逐事件明细（服务器保留）：`{report['details']['path']}`；SHA256：`{report['details']['sha256']}`。",
    ]
    for name, source in report["sources"].items():
        lines.append(f"- {name} 来源报告：`{source['report_path']}`；SHA256：`{source['report_sha256']}`；"
                     f"缓存 index SHA256：`{source['cache']['index_sha256']}`。")
    for name in ("single_frame", "stag"):
        standard = report["standard_metrics"][name]
        temporal = report["temporal_metrics"][name]
        jitter, stability = temporal["jitter"], temporal["stability"]
        response = temporal["response"]["all"]
        lines.extend([
            f"- {name}：AUC `{standard['auc']}` / L2 `{standard['l2']}` / AP `{standard['inout_ap']}`。",
            f"- {name}：Jitter mean `{jitter['predicted']['mean']}`，同帧对 GT 位移 mean "
            f"`{jitter['ground_truth']['mean']}`；稳定占比 `{stability['stable_ratio']}`，"
            f"准确且稳定占比 `{stability['accurate_stable_ratio']}`。",
            f"- {name}：切换代理事件 `{response['events']}`，响应率 `{response['response_rate']}`，"
            f"已响应起始延迟 mean `{response['responded_delay_frames']['mean']}` 帧，"
            f"未响应 `{response['censored_events']}`，短窗口 `{response['insufficient_followup_events']}`。",
        ])
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-frame-cache", type=Path, required=True)
    parser.add_argument("--single-frame-report", type=Path, required=True)
    parser.add_argument("--stag-cache", type=Path, required=True)
    parser.add_argument("--stag-report", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True, help="New server-only directory for per-track/event details")
    parser.add_argument("--shift-threshold", type=float, default=0.15)
    parser.add_argument("--response-radius", type=float, default=0.10)
    parser.add_argument("--stable-threshold", type=float, default=0.02)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--max-response-frames", type=int, default=30)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t4_temporal_comparison.json")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    config = TemporalConfig(shift_threshold=args.shift_threshold, response_radius=args.response_radius,
                            stable_threshold=args.stable_threshold, hold_frames=args.hold_frames,
                            max_response_frames=args.max_response_frames)
    started = time.perf_counter()
    report = {
        "task": "T4", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "numpy_version": np.__version__,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path)
                        for path in (Path(__file__).resolve(), REPO_ROOT / "cscwd/code/stag/metrics.py",
                                     REPO_ROOT / "cscwd/code/stag/cache.py")},
    }
    report.update(run_evaluation(args.single_frame_cache, args.single_frame_report,
                                 args.stag_cache, args.stag_report, args.details_dir, config))
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.report, report)
    record_provenance(args.provenance, report, args.report.resolve())
    print(json.dumps({"standard": report["standard_metrics"], "temporal": report["temporal_metrics"]}, indent=2))
    print(f"T4 evaluation completed: {report['coverage']['tracks']} tracks, "
          f"{report['coverage']['person_frames']} person frames; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
