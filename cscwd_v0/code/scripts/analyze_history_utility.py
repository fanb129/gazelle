"""Measure fixed causal history candidates on official train before training a gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from stag.cache import file_sha256, write_json
from stag.history_candidates import CANDIDATE_NAMES, VERSION, maps_to_coordinates, step_candidates
from stag.history_utility import analyze_history_utility
from stag.metrics import TemporalConfig, temporal_protocol
from stag.official_data import _load_split


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def generate_coordinates(part, output_dir, device):
    """Write observations-only candidates, finishing before any utility scoring."""
    count = part["prediction_index"]["person_frames"]
    xy_path, available_path = output_dir / "candidate_xy.npy", output_dir / "history_available.npy"
    xy = np.lib.format.open_memmap(xy_path, mode="w+", dtype=np.float32,
                                  shape=(count, len(CANDIDATE_NAMES), 2))
    available = np.lib.format.open_memmap(available_path, mode="w+", dtype=np.bool_, shape=(count,))
    feature_rows = {path: row for row, path in enumerate(part["feature_index"]["image_paths"])}
    live_xy, live_available = hashlib.sha256(), hashlib.sha256()
    written = 0
    with torch.no_grad():
        for number, track in enumerate(part["tracks"], start=1):
            state = None
            for position, frame in enumerate(track["frames"], start=track["start"]):
                heatmap = torch.tensor(np.array(part["arrays"]["heatmap"][position]),
                                       dtype=torch.float32, device=device)
                feature = torch.tensor(np.array(part["features"][feature_rows[frame["path"]]]),
                                       dtype=torch.float32, device=device)
                maps, state, has_history = step_candidates(heatmap, feature, frame["frame_index"], state)
                coordinates = maps_to_coordinates(maps)
                xy[position] = coordinates
                available[position] = has_history
                live_xy.update(coordinates.tobytes(order="C"))
                live_available.update(np.asarray(has_history, dtype=np.bool_).tobytes())
                written += 1
            print(f"History candidates track {number}/{len(part['tracks'])}; frames {written}/{count}", flush=True)
    xy.flush()
    available.flush()
    del xy, available
    xy = np.load(xy_path, mmap_mode="r", allow_pickle=False)
    available = np.load(available_path, mmap_mode="r", allow_pickle=False)
    checks = {
        "all_train_frames_written_once": written == count,
        "coordinate_readback_matches_generated_values": hashlib.sha256(xy.tobytes(order="C")).digest() == live_xy.digest(),
        "history_mask_readback_matches_generated_values": hashlib.sha256(available.tobytes()).digest() == live_available.digest(),
        "coordinates_finite": bool(np.isfinite(xy).all()),
        "current_coordinates_reproduce_raw_cache": np.array_equal(xy[:, 0], part["arrays"]["argmax_xy"]),
    }
    if not all(checks.values()):
        raise ValueError(f"History candidate generation verification failed: {checks}")
    return xy, available, checks


def run_history_utility(feature_cache, feature_report, prediction_report, plan, output_dir, device="cuda:0"):
    if VERSION not in Path(plan).read_text():
        raise ValueError("Plan does not contain the history utility version")
    # Only this official train loader is called; no test/validation or checkpoint inputs.
    part, prediction, feature_run = _load_split("train", feature_cache, feature_report, prediction_report)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_reports = {"prediction": artifact(prediction_report), "features": artifact(feature_report),
                      "plan": artifact(plan)}
    sources = {"prediction_index_sha256": prediction["cache"]["index_sha256"],
               "feature_index_sha256": feature_run["cache"]["index_sha256"],
               "checkpoint_sha256": prediction["checkpoint"]["sha256"],
               "backbone_checkpoint_sha256": prediction["backbone_checkpoint"]["sha256"]}
    xy, available, checks = generate_coordinates(part, output_dir, device)
    print("All candidates stored; starting train-only GT utility scoring", flush=True)
    analysis = analyze_history_utility(part["tracks"], xy, part["arrays"]["inout"], available)
    artifacts = {"candidate_xy": artifact(output_dir / "candidate_xy.npy"),
                 "history_available": artifact(output_dir / "history_available.npy")}
    for name, values in analysis["arrays"].items():
        path = output_dir / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        artifacts[name] = artifact(path)
    details_path = output_dir / "utility_details.json"
    write_json(details_path, analysis["details"])
    artifacts["details"] = artifact(details_path)
    checks.update(analysis["summary"]["checks"])
    checks.update({
        "frozen_features_read_only": not part["features"].flags.writeable,
        "raw_predictions_read_only": all(not a.flags.writeable for a in part["arrays"].values()),
        "generated_candidates_read_only_during_scoring": not xy.flags.writeable and not available.flags.writeable,
        "only_official_train_tracks": all(t["track_id"].startswith("train/") for t in part["tracks"]),
    })
    if not all(checks.values()):
        raise ValueError(f"History utility verification failed: {checks}")
    index_path = output_dir / "index.json"
    write_json(index_path, {"task": "R4_history_utility", "utility_version": VERSION,
        "evaluation_split": "train", "person_frames": len(xy), "candidate_names": list(CANDIDATE_NAMES),
        "tracks": part["tracks"], "sources": sources, "source_reports": source_reports,
        "feature_contract": part["feature_index"]["contract"], "artifacts": artifacts,
        "coordinate_convention": "column/64,row/64; first_flat_argmax",
        "history_policy": "same_person_previous_contiguous_frame; reset_at_track_start_or_gap",
        "checks": checks, "passed": True})
    artifacts["index"] = artifact(index_path)
    return {"task": "R4_history_utility", "utility_version": VERSION, "passed": True,
            "stage_status": "awaiting_result_review", "evaluation_split": "train",
            "training": False, "test_accessed": False, "source_reports": source_reports,
            "sources": sources, "feature_contract": part["feature_index"]["contract"],
            "config": {"ema_current_weight": .75, "transport_temperature": .1,
                       "transport": "source_to_target_cosine_softmax_mass; previous_raw_only",
                       "precision": "float32; no_autocast; tf32_disabled", "device": str(device),
                       "inout": "raw_unchanged", "parameter_search": False,
                       "oracle": "offline_min_L2; no_feedback; excludes_current_grid"},
            "temporal_protocol": temporal_protocol(TemporalConfig()), "artifacts": artifacts,
            "summary": analysis["summary"], "checks": checks,
            "interpretation": "train_only_fixed_candidate_feasibility; oracle_is_not_a_deployable_model_or_video_information_upper_bound"}


def summary_markdown(report):
    summary = report["summary"]
    lines = ["# R4-D5 训练集历史候选效用", "",
             f"- Version: `{report['utility_version']}`; official train only; no training or test access.",
             "- Original InOut is fixed. Candidate AUC is not computed; only coordinates are stored.",
             f"- Coverage: `{json.dumps(summary['coverage'])}`.",
             f"- Raw train InOut AP (unchanged): `{summary['inout_ap']}`.",
             "- Utility denominator: current GT in-frame and contiguous past available; correctness radius 0.1.", "",
             "| Candidate | Train L2 | Both correct | Current only | Candidate only (rescue) | Both wrong | Rescue blocked by raw out | Accurate stable | Gaze responses |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in summary["candidates"].items():
        utility = item["utility"]
        quadrant = utility["localization"]
        temporal = item["temporal"]
        lines.append(f"| {name} | {item['l2']} | {quadrant['both_correct']} | "
                     f"{quadrant['current_only_correct']} | {quadrant['history_only_correct']} | "
                     f"{quadrant['both_wrong']} | {utility['rescue_blocked_by_raw_out']} | "
                     f"{temporal['stability']['accurate_stable_ratio']} | "
                     f"{temporal['response']['gaze_shift']['responded_events']} |")
    lines.extend(("", "## Utility by offline GT group", "",
                  "| Candidate | Group | Frames | Candidate only (rescue) | Current only (harm) |",
                  "|---|---|---:|---:|---:|"))
    for name, item in summary["candidates"].items():
        for group, utility in item["groups"].items():
            quadrant = utility["localization"]
            lines.append(f"| {name} | {group} | {utility['frames']} | "
                         f"{quadrant['history_only_correct']} | {quadrant['current_only_correct']} |")
    lines.extend(("", "## GT-informed fixed-set reference", "",
                  "```json", json.dumps(summary["oracle"], indent=2), "```", "",
                  "Oracle choices are scoring only; no GT/oracle result enters candidate memory.",
                  "This reference is not a deployable model or a bound on all video information.",
                  "Detailed coordinates, groups, errors and event records remain on the server."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature_cache", "feature_report", "prediction_report", "plan", "output_dir", "report", "summary", "provenance"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_history_utility(args.feature_cache, args.feature_report, args.prediction_report,
                                 args.plan, args.output_dir, args.device)
    source_paths = [Path(__file__), *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
                    ("history_candidates", "history_utility", "official_data", "feature_cache", "cache", "metrics", "state_metrics"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in source_paths},
                   "elapsed_seconds": time.perf_counter() - started,
                   "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                       "device_name": torch.cuda.get_device_name(torch.device(args.device))
                           if torch.device(args.device).type == "cuda" else "CPU"}})
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report))
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R4-D5 train-only history utility\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Plan: `{args.plan}`; SHA256: `{report['source_reports']['plan']['sha256']}`.\n"
            f"- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Summary: `{args.summary}`; SHA256: `{file_sha256(args.summary)}`.\n"
            "- Complete official train only; no test access, training or checkpoint; original InOut fixed.\n"
            "- GT/oracle scoring follows candidate generation and never feeds memory.\n")
        for name, item in report["artifacts"].items():
            stream.write(f"- {name}: `{item['path']}`; SHA256: `{item['sha256']}`.\n")
    print(f"R4 history utility completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
