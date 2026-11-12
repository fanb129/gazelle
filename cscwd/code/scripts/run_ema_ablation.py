"""T5: predeclared fixed-EMA controls against single-frame and rule STAG caches."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from eval_temporal import event_signature, paired_response_summary, verify_completed_cache
from run_stag import sequences_from_index
from stag.cache import file_sha256, iter_cached_tracks, load_cache, write_json, write_prediction_cache
from stag.ema import smooth_ema_track
from stag.metrics import (STANDARD_PROTOCOL, TemporalConfig, evaluate_predictions,
                          evaluate_temporal, temporal_protocol)


def fixed_variants(bias, alphas):
    matched = float(np.exp(-np.logaddexp(0.0, -bias)))
    return [{"name": "ema_bias_matched", "alpha": matched, "alpha_source": "sigmoid_of_stag_bias"},
            *[{"name": f"ema_alpha_{str(alpha).replace('.', 'p')}", "alpha": alpha,
               "alpha_source": "predeclared_constant"} for alpha in alphas]]


def ema_batches(index, arrays, cache_dir, alpha):
    # The prediction writer creates cache_dir before consuming this generator.
    coefficients = np.lib.format.open_memmap(cache_dir / "alpha.npy", mode="w+", dtype=np.float32,
                                            shape=(index["person_frames"],))
    resets = np.lib.format.open_memmap(cache_dir / "reset.npy", mode="w+", dtype=np.bool_,
                                      shape=(index["person_frames"],))
    for track, values in iter_cached_tracks(index, arrays):
        result = smooth_ema_track(values["heatmap"], [frame["frame_index"] for frame in track["frames"]], alpha)
        selection = slice(track["start"], track["stop"])
        coefficients[selection], resets[selection] = result["alpha"], result["reset"]
        yield result["heatmap"], values["inout"]
    coefficients.flush()
    resets.flush()


def write_ema_cache(index, arrays, cache_dir, variant, source):
    metadata = {
        **index["metadata"], "method": "fixed_ema", **variant,
        "protocol": {**index["metadata"]["protocol"],
                     "temporal_context": "causal_current_and_past; independent_per_track"},
        "inout_policy": "unchanged_single_frame_probability",
        "state_reset": "track_start_or_nonconsecutive_frame_index; no_label_based_reset",
        "parameter_selection": "predeclared_controls; all_variants_reported; no_test_set_selection",
        "parent_cache": source,
    }
    result = write_prediction_cache(sequences_from_index(index),
                                    ema_batches(index, arrays, cache_dir, variant["alpha"]),
                                    cache_dir, metadata, task="T5")
    output_index, output_arrays = load_cache(cache_dir)
    diagnostics = {}
    for name in ("alpha", "reset"):
        path = cache_dir / f"{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        diagnostics[name] = {"path": path.name, "shape": list(values.shape),
                             "dtype": str(values.dtype), "sha256": file_sha256(path)}
    output_index["diagnostics"] = diagnostics
    write_json(cache_dir / "index.json", output_index)
    result.update(index_sha256=file_sha256(cache_dir / "index.json"), diagnostics=diagnostics)
    if output_index["tracks"] != index["tracks"] or not np.array_equal(output_arrays["inout"], arrays["inout"]):
        raise ValueError("EMA output records or in/out scores differ from the single-frame source")
    return result, output_arrays


def evaluate_method(index, arrays, config):
    standard = evaluate_predictions(index["tracks"], arrays["heatmap"], arrays["inout"])
    temporal = evaluate_temporal(index["tracks"], arrays["argmax_xy"], arrays["inout"], config)
    return {"standard": standard, "temporal": temporal["summary"]}, {
        "tracks": temporal["tracks"], "events": temporal["events"],
    }


def response_comparison(reference_events, candidate_events):
    groups = paired_response_summary(reference_events, candidate_events)
    names = {"single_frame_only_responded_events": "reference_only_responded_events",
             "stag_only_responded_events": "candidate_only_responded_events"}
    return {kind: {names.get(key, key): value for key, value in group.items()}
            for kind, group in groups.items()}


def run_ablation(single_frame_cache, single_frame_report, stag_cache, stag_report,
                 temporal_report, output_dir, alphas=(0.25, 0.5, 0.75)):
    paths = {"single_frame": Path(single_frame_report), "stag": Path(stag_report),
             "temporal": Path(temporal_report)}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline_run, stag_run, t4 = (reports[name] for name in ("single_frame", "stag", "temporal"))
    verify_completed_cache(single_frame_cache, baseline_run)
    verify_completed_cache(stag_cache, stag_run)
    source_checks = {
        "t4_completed": t4["passed"] and all(t4["checks"].values()),
        "t4_sources_match": all(t4["sources"][name]["report_sha256"] == file_sha256(paths[name])
                                and t4["sources"][name]["cache"] == reports[name]["cache"]
                                for name in ("single_frame", "stag")),
        "stag_parent_matches": (stag_run["source"]["index_sha256"] == baseline_run["cache"]["index_sha256"]
                                 and stag_run["source"]["report_sha256"] == file_sha256(single_frame_report)),
        "weights_match_t4": t4["stag_weights"] == stag_run["config"]["weights"],
    }
    config = TemporalConfig(**t4["temporal_protocol"]["thresholds"])
    source_checks["same_t4_metric_protocol"] = (temporal_protocol(config) == t4["temporal_protocol"]
                                                and STANDARD_PROTOCOL == t4["standard_protocol"])
    if not all(source_checks.values()):
        raise ValueError(f"T5 source verification failed: {source_checks}")
    index, baseline_arrays = load_cache(single_frame_cache)
    stag_index, stag_arrays = load_cache(stag_cache)
    if index["tracks"] != stag_index["tracks"]:
        raise ValueError("Single-frame and STAG track records must match")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    methods, details = {}, {}
    for name, arrays, run in (("single_frame", baseline_arrays, baseline_run), ("stag", stag_arrays, stag_run)):
        print(f"Reproducing T4 reference metrics: {name}", flush=True)
        methods[name], details[name] = evaluate_method(index, arrays, config)
        methods[name].update(kind="single_frame" if name == "single_frame" else "rule_gate", cache=run["cache"])
        if methods[name]["standard"] != t4["standard_metrics"][name] or methods[name]["temporal"] != t4["temporal_metrics"][name]:
            raise ValueError(f"Reference metrics do not reproduce T4: {name}")
    variants = fixed_variants(stag_run["config"]["weights"][0], alphas)
    parent = {"path": str(Path(single_frame_cache).resolve()),
              "index_sha256": baseline_run["cache"]["index_sha256"],
              "report_path": str(paths["single_frame"].resolve()),
              "report_sha256": file_sha256(paths["single_frame"])}
    for variant in variants:
        name = variant["name"]
        print(f"Running {name}: alpha={variant['alpha']}", flush=True)
        cache, arrays = write_ema_cache(index, baseline_arrays, output_dir / name, variant, parent)
        methods[name], details[name] = evaluate_method(index, arrays, config)
        methods[name].update(kind="fixed_ema", **variant, cache=cache)

    event_keys = [event_signature(event) for event in details["single_frame"]["events"]]
    reference = methods["single_frame"]["temporal"]
    checks = {
        **source_checks, "reference_metrics_reproduce_t4": True,
        "all_output_caches_verified": all(method["cache"]["passed"] for method in methods.values()),
        "same_frame_coverage": all(method["standard"]["person_frames"] == index["person_frames"]
                                   for method in methods.values()),
        "same_ap": all(method["standard"]["inout_ap"] == methods["single_frame"]["standard"]["inout_ap"]
                        for method in methods.values()),
        "same_gt_pairs_and_windows": all(
            method["temporal"]["jitter"]["pairs"] == reference["jitter"]["pairs"]
            and method["temporal"]["jitter"]["ground_truth"] == reference["jitter"]["ground_truth"]
            and method["temporal"]["stability"]["eligible_windows"] == reference["stability"]["eligible_windows"]
            for method in methods.values()),
        "same_events_and_windows": all([event_signature(event) for event in method["events"]] == event_keys
                                        for method in details.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"T5 method alignment failed: {checks}")
    raw_stag_response = paired_response_summary(details["single_frame"]["events"], details["stag"]["events"])
    pairs = {"stag_minus_single_frame": response_comparison(details["single_frame"]["events"], details["stag"]["events"])}
    for variant in variants:
        name = variant["name"]
        for reference_name in ("single_frame", "stag"):
            pairs[f"{name}_minus_{reference_name}"] = response_comparison(details[reference_name]["events"],
                                                                          details[name]["events"])
    checks["stag_paired_response_reproduces_t4"] = raw_stag_response == t4["paired_response"]
    if not checks["stag_paired_response_reproduces_t4"]:
        raise ValueError("STAG paired event response does not reproduce T4")
    details_path = output_dir / "ablation_details.json"
    write_json(details_path, {"task": "T5", "temporal_protocol": temporal_protocol(config), "methods": details})
    return {
        "sources": {name: {"report_path": str(path.resolve()), "report_sha256": file_sha256(path)}
                    for name, path in paths.items()},
        "model": baseline_run["model"], "checkpoint": baseline_run["checkpoint"], "track_json": baseline_run["track_json"],
        "coverage": t4["coverage"], "stag_weights": stag_run["config"]["weights"],
        "protocol": {"standard": STANDARD_PROTOCOL, "temporal": temporal_protocol(config),
                     "parameter_policy": "predeclared_fixed_controls; all_methods_reported; no_test_selection_or_training",
                     "ema_recurrence": "H_t=alpha*h_t+(1-alpha)*H_prev; reset_to_current_on_track_start_or_frame_gap",
                     "inout_policy": "unchanged_single_frame_probability_for_all_methods",
                     "paired_response": "candidate_minus_reference; reference_is_method_after_minus_in_pair_key; common_responded_events_only_for_delay_delta"},
        "variants": variants, "methods": methods, "paired_response": pairs,
        "details": {"path": str(details_path.resolve()), "sha256": file_sha256(details_path)},
        "checks": checks, "passed": True,
    }


def markdown_summary(report):
    lines = ["# T5 固定 EMA 对照", "", f"- JSON 报告：`{report['report_path']}`。",
             f"- 模型：`{report['model']}`；STAG 权重：`{report['stag_weights']}`。",
             "- 全部预先声明的配置均列出；阈值、事件分母、删失计数和配对延迟详见 JSON。", "",
             "| 配置 | α | AUC↑ | L2↓ | AP↑ | Jitter↓ | GT 位移 | 准确稳定占比↑ | gaze_shift 响应率↑ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "n/a" if value is None else f"{value:.6f}"
    for name, method in report["methods"].items():
        standard, temporal = method["standard"], method["temporal"]
        values = [method.get("alpha"), standard["auc"], standard["l2"], standard["inout_ap"],
                  temporal["jitter"]["predicted"]["mean"], temporal["jitter"]["ground_truth"]["mean"],
                  temporal["stability"]["accurate_stable_ratio"], temporal["response"]["gaze_shift"]["response_rate"]]
        lines.append(f"| {name} | " + " | ".join(number(value) for value in values) + " |")
    lines.extend(["", "延迟差为“前者−后者”，只在该对方法共同响应的同一批事件上计算；结合响应率和独有响应数解释。", "",
                  "| 配对（gaze_shift） | 共同响应数 | 延迟差 mean（帧） | 后者独有响应数 | 前者独有响应数 |",
                  "|---|---:|---:|---:|---:|"])
    for name, response in report["paired_response"].items():
        gaze = response["gaze_shift"]
        lines.append(f"| {name} | {gaze['both_responded_events']} | "
                     f"{number(gaze['common_responded_delay_delta_frames']['mean'])} | "
                     f"{gaze['reference_only_responded_events']} | {gaze['candidate_only_responded_events']} |")
    return "\n".join(lines) + "\n"


def record_provenance(path, report):
    lines = ["", "## T5 固定 EMA 对照（服务器运行）", "",
             f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
             f"- checkpoint：`{report['checkpoint']['path']}`；SHA256：`{report['checkpoint']['sha256']}`。",
             f"- 命令：`{report['command']}`。",
             f"- 小结果：`{report['report_path']}`；Markdown：`{report['summary_path']}`。",
             f"- 逐轨迹/事件明细：`{report['details']['path']}`；SHA256：`{report['details']['sha256']}`。"]
    for name, source in report["sources"].items():
        lines.append(f"- 来源 {name}：`{source['report_path']}`；SHA256：`{source['report_sha256']}`。")
    for name, method in report["methods"].items():
        standard, temporal = method["standard"], method["temporal"]
        lines.append(f"- {name}：AUC `{standard['auc']}` / L2 `{standard['l2']}` / AP `{standard['inout_ap']}` / "
                     f"Jitter `{temporal['jitter']['predicted']['mean']}` / 准确稳定占比 "
                     f"`{temporal['stability']['accurate_stable_ratio']}` / gaze_shift 响应率 "
                     f"`{temporal['response']['gaze_shift']['response_rate']}`。")
        if method["kind"] == "fixed_ema":
            cache = method["cache"]
            lines.append(f"- {name} 缓存：`{cache['path']}`；index SHA256：`{cache['index_sha256']}`；α=`{method['alpha']}`。")
            for artifact in {**cache["files"], **cache["diagnostics"]}.values():
                lines.append(f"  - 大文件：`{Path(cache['path']) / artifact['path']}`；SHA256：`{artifact['sha256']}`。")
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-frame-cache", type=Path, required=True)
    parser.add_argument("--single-frame-report", type=Path, required=True)
    parser.add_argument("--stag-cache", type=Path, required=True)
    parser.add_argument("--stag-report", type=Path, required=True)
    parser.add_argument("--temporal-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New server-only directory for all EMA caches/details")
    parser.add_argument("--ema-alphas", nargs="+", type=float, choices=[0.25, 0.5, 0.75], default=[0.25, 0.5, 0.75])
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t5_ema_ablation.json")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "cscwd/results/t5_ema_ablation.md")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "task": "T5", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "numpy_version": np.__version__,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path)
                        for path in (Path(__file__).resolve(), REPO_ROOT / "cscwd/code/stag/ema.py",
                                     REPO_ROOT / "cscwd/code/stag/metrics.py", REPO_ROOT / "cscwd/code/stag/cache.py",
                                     REPO_ROOT / "cscwd/code/scripts/run_stag.py", REPO_ROOT / "cscwd/code/scripts/eval_temporal.py")},
        "report_path": str(args.report.resolve()), "summary_path": str(args.summary.resolve()),
    }
    report.update(run_ablation(args.single_frame_cache, args.single_frame_report, args.stag_cache,
                                args.stag_report, args.temporal_report, args.output_dir, args.ema_alphas))
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(markdown_summary(report))
    record_provenance(args.provenance, report)
    print(f"T5 ablation completed: {len(report['methods'])} methods, "
          f"{report['coverage']['person_frames']} person frames; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
