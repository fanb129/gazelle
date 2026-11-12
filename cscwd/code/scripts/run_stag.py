"""T3: causal rule-gated smoothing and paired VAT standard metrics from T2 NPYs."""

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

from stag.cache import file_sha256, iter_cached_tracks, load_cache, write_json, write_prediction_cache
from stag.gating import smooth_track
from stag.metrics import STANDARD_PROTOCOL, evaluate_predictions


def verify_source(cache_dir, cache_report):
    """Bind this comparison to the exact completed T2 run, including its NPYs."""
    with Path(cache_report).open() as stream:
        source = json.load(stream)
    cache_dir = Path(cache_dir)
    if not source["passed"] or file_sha256(cache_dir / "index.json") != source["cache"]["index_sha256"]:
        raise ValueError("T2 cache report or index verification failed")
    for artifact in source["cache"]["files"].values():
        if file_sha256(cache_dir / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"T2 cache SHA256 mismatch: {artifact['path']}")
    return source


def sequences_from_index(index):
    sequences = {}
    for track in index["tracks"]:
        path = track["sequence_path"]
        if path not in sequences:
            sequences[path] = {"path": path, "width": track["width"],
                               "height": track["height"], "tracks": []}
        sequences[path]["tracks"].append({key: track[key]
                                         for key in ("person_id", "track_id", "source_csv", "frames")})
    return list(sequences.values())


def smoothed_batches(index, arrays, output_dir, weights):
    # The writer creates output_dir before consuming this generator.
    shapes = {"alpha": (index["person_frames"],), "signals": (index["person_frames"], 3),
              "reset": (index["person_frames"],)}
    diagnostics = {name: np.lib.format.open_memmap(output_dir / f"{name}.npy", mode="w+",
                                                  dtype=np.bool_ if name == "reset" else np.float32,
                                                  shape=shape)
                   for name, shape in shapes.items()}
    for track_number, (track, values) in enumerate(iter_cached_tracks(index, arrays), start=1):
        # True gaze coordinates and true in/out labels stay in evaluation only.
        result = smooth_track(values["heatmap"], values["inout"],
                              [frame["bbox_norm"] for frame in track["frames"]],
                              [frame["frame_index"] for frame in track["frames"]], weights)
        for name, target in diagnostics.items():
            target[track["start"]:track["stop"]] = result[name]
        yield result["heatmap"], values["inout"]
        if track_number % 50 == 0:
            print(f"Smoothed {track_number}/{index['track_count']} tracks", flush=True)
    for values in diagnostics.values():
        values.flush()


def run_comparison(cache_dir, cache_report, output_dir, weights):
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    source = verify_source(cache_dir, cache_report)
    index, arrays = load_cache(cache_dir)
    metadata = {
        **index["metadata"], "method": "STAG_rule_v1", "weights": list(weights),
        "protocol": {**index["metadata"]["protocol"],
                     "temporal_context": "causal_current_and_past; independent_per_track"},
        "weight_order": ["bias", "d_head", "trans", "d_heat"],
        "parameter_selection": "fixed_before_test_evaluation; no_test_set_search",
        "gate_inputs": ["current_and_previous_head_boxes", "raw_predicted_inout",
                        "current_raw_heatmap_and_previous_smoothed_heatmap"],
        "inout_policy": "unchanged_single_frame_probability",
        "state_reset": "track_start_or_nonconsecutive_frame_index; no_label_based_reset",
        "parent_cache": {"path": str(cache_dir.resolve()), "index_sha256": source["cache"]["index_sha256"],
                         "report_path": str(Path(cache_report).resolve()), "report_sha256": file_sha256(cache_report)},
    }
    print("Running causal STAG from the verified T2 cache", flush=True)
    output = write_prediction_cache(sequences_from_index(index),
                                    smoothed_batches(index, arrays, output_dir, weights),
                                    output_dir, metadata, task="T3")
    smoothed_index, smoothed = load_cache(output_dir)
    diagnostics = {}
    for name in ("alpha", "signals", "reset"):
        path = output_dir / f"{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        diagnostics[name] = {"path": path.name, "shape": list(values.shape),
                             "dtype": str(values.dtype), "sha256": file_sha256(path)}
    smoothed_index["diagnostics"] = diagnostics
    write_json(output_dir / "index.json", smoothed_index)
    output["index_sha256"] = file_sha256(output_dir / "index.json")
    output["diagnostics"] = diagnostics
    checks = {
        "coverage_matches_t2": all(output[key] == source["cache"][key]
                                   for key in ("tracks", "person_frames", "unique_image_frames")),
        "track_order_and_frames_match_t2": smoothed_index["tracks"] == index["tracks"],
        "inout_unchanged": np.array_equal(smoothed["inout"], arrays["inout"]),
        "output_exact_readback": output["passed"],
    }
    if not all(checks.values()):
        raise ValueError(f"STAG comparison alignment verification failed: {checks}")
    print("Evaluating single-frame baseline on the same track records", flush=True)
    baseline = evaluate_predictions(index["tracks"], arrays["heatmap"], arrays["inout"])
    print("Evaluating STAG on the same track records", flush=True)
    stag = evaluate_predictions(index["tracks"], smoothed["heatmap"], smoothed["inout"])
    return {
        "source": {"model": source["model"], "checkpoint": source["checkpoint"],
                   "track_json": source["track_json"], **metadata["parent_cache"]},
        "config": metadata, "cache": output, "checks": checks, "passed": True,
        "metrics": {"single_frame": baseline, "stag": stag,
                    "delta_stag_minus_single_frame": {key: stag[key] - baseline[key]
                                                      for key in ("auc", "l2", "inout_ap")}},
        "metric_protocol": {
            **STANDARD_PROTOCOL,
            "paired_samples": "same_track_records; all_person_frames; no_subsampling",
            "auc": "mean_per_inframe_person_frame; gazelle.utils.vat_auc_64x64_sigma3_rectangle",
            "l2": "mean_per_inframe_person_frame; argmax_xy=(column/64,row/64)",
            "inout_ap": "global_AP; positive=in_frame; all_person_frames; unchanged_scores",
            "aggregation": "person_frame_weighted; not_equal_track_weighted",
        },
    }


def record_provenance(path, report, report_path):
    lines = [
        "", "## T3 STAG 初始规则门控对比（服务器运行）", "",
        f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
        f"- 来源模型：`{report['source']['model']}`；checkpoint：`{report['source']['checkpoint']['path']}`；"
        f"SHA256：`{report['source']['checkpoint']['sha256']}`。",
        f"- 输入缓存：`{report['source']['path']}`；index SHA256：`{report['source']['index_sha256']}`。",
        f"- 门控权重 [bias, d_head, trans, d_heat]：`{report['config']['weights']}`，测试前固定；仅平滑热图，画内概率保持单帧输出。",
        f"- 命令：`{report['command']}`。",
        f"- 小报告：`{report_path}`；输出缓存（服务器保留）：`{report['cache']['path']}`；"
        f"index SHA256：`{report['cache']['index_sha256']}`。",
    ]
    for name in ("single_frame", "stag", "delta_stag_minus_single_frame"):
        values = report["metrics"][name]
        lines.append(f"- {name}：AUC `{values['auc']}` / L2 `{values['l2']}` / In-Out AP `{values['inout_ap']}`。")
    for artifact in {**report["cache"]["files"], **report["cache"]["diagnostics"]}.values():
        lines.append(f"- 大文件：`{Path(report['cache']['path']) / artifact['path']}`；SHA256：`{artifact['sha256']}`。")
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Completed T2 cache")
    parser.add_argument("--cache-report", type=Path, required=True, help="T2 small result with source SHA256s")
    parser.add_argument("--output-dir", type=Path, required=True, help="New server-only STAG cache directory")
    parser.add_argument("--weights", nargs=4, type=float, default=[-2.0, 20.0, 2.0, 10.0],
                        metavar=("BIAS", "HEAD", "TRANS", "HEAT"))
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t3_stag_comparison.json")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "task": "T3", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "numpy_version": np.__version__,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path)
                        for path in (Path(__file__).resolve(), REPO_ROOT / "cscwd/code/stag/gating.py",
                                     REPO_ROOT / "cscwd/code/stag/metrics.py", REPO_ROOT / "cscwd/code/stag/cache.py")},
    }
    report.update(run_comparison(args.cache_dir, args.cache_report, args.output_dir, args.weights))
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.report, report)
    record_provenance(args.provenance, report, args.report.resolve())
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"T3 run completed: {report['cache']['tracks']} tracks, "
          f"{report['cache']['person_frames']} person frames; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
