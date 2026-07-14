#!/usr/bin/env python3
"""Audit VAT reliability around in/out transitions without model inference.

The script aligns adjacent-frame people with greedy bbox-IoU matching, joins
raw ``failure_taxonomy.py`` records, and measures calibration and localization
errors for stable versus switching in/out states.  It never reads checkpoints
or changes predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest
from AAAIScripts.failure_taxonomy import flatten_annotations


STATES = {
    (1, 1): "stable_in",
    (1, 0): "in_to_out",
    (0, 1): "out_to_in",
    (0, 0): "stable_out",
}


def optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_iou_matches(previous: list[dict], current: list[dict], threshold: float):
    candidates = sorted(
        (
            (bbox_iou(prev["bbox_norm"], curr["bbox_norm"]), prev_index, curr_index)
            for prev_index, prev in enumerate(previous)
            for curr_index, curr in enumerate(current)
            if prev.get("bbox_norm") is not None and curr.get("bbox_norm") is not None
        ),
        reverse=True,
    )
    used_previous, used_current, matches = set(), set(), []
    for iou, prev_index, curr_index in candidates:
        if iou < threshold:
            break
        if prev_index in used_previous or curr_index in used_current:
            continue
        used_previous.add(prev_index)
        used_current.add(curr_index)
        matches.append((prev_index, curr_index, iou))
    return sorted(matches)


def frame_number(path: str) -> int | None:
    try:
        return int(Path(path).stem)
    except ValueError:
        return None


def read_records(specification: str):
    if "=" not in specification:
        raise ValueError("--records must use label=/absolute/path.csv")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {(row["path"], int(row["person_index"])): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate path/person_index key in {path}")
    return label, path, mapping


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def build_transition_rows(
    annotation_frames: list[dict],
    record_sets: list[tuple[str, Path, dict]],
    iou_threshold: float,
    max_frame_gap: int,
    max_sequences: int | None,
) -> tuple[list[dict], dict]:
    sequences: dict[str, list[dict]] = defaultdict(list)
    for frame in annotation_frames:
        sequences[str(Path(frame["path"]).parent)].append(frame)
    selected_sequences = sorted(sequences)
    if max_sequences is not None:
        selected_sequences = selected_sequences[:max_sequences]

    output, adjacent_pairs, matched_people = [], 0, 0
    possible_people = 0
    for sequence_id in selected_sequences:
        frames = sorted(
            sequences[sequence_id],
            key=lambda frame: (frame_number(frame["path"]) is None, frame_number(frame["path"]) or 0),
        )
        for previous_frame, current_frame in zip(frames, frames[1:]):
            previous_number = frame_number(previous_frame["path"])
            current_number = frame_number(current_frame["path"])
            if (
                previous_number is not None
                and current_number is not None
                and current_number - previous_number > max_frame_gap
            ):
                continue
            adjacent_pairs += 1
            previous_heads = previous_frame.get("heads", [])
            current_heads = current_frame.get("heads", [])
            possible_people += min(len(previous_heads), len(current_heads))
            matches = greedy_iou_matches(previous_heads, current_heads, iou_threshold)
            matched_people += len(matches)
            for previous_index, current_index, iou in matches:
                previous_head = previous_heads[previous_index]
                current_head = current_heads[current_index]
                previous_inout = int(previous_head.get("inout", 1))
                current_inout = int(current_head.get("inout", 1))
                state = STATES[(previous_inout, current_inout)]
                for model_label, _, records in record_sets:
                    previous_record = records.get((previous_frame["path"], previous_index))
                    current_record = records.get((current_frame["path"], current_index))
                    if previous_record is None or current_record is None:
                        continue
                    record_previous_inout = int(previous_record["inout"])
                    record_current_inout = int(current_record["inout"])
                    if (record_previous_inout, record_current_inout) != (
                        previous_inout,
                        current_inout,
                    ):
                        raise ValueError(
                            "annotation/records inout mismatch for "
                            f"{model_label}: {previous_frame['path']}[{previous_index}] -> "
                            f"{current_frame['path']}[{current_index}]"
                        )
                    previous_score = optional_float(previous_record.get("inout_score"))
                    current_score = optional_float(current_record.get("inout_score"))
                    if previous_score is None or current_score is None:
                        continue
                    if not 0.0 <= previous_score <= 1.0 or not 0.0 <= current_score <= 1.0:
                        raise ValueError(f"inout_score outside [0, 1] for {model_label}")
                    row = {
                        "model": model_label,
                        "sequence_id": sequence_id,
                        "previous_path": previous_frame["path"],
                        "current_path": current_frame["path"],
                        "previous_person_index": previous_index,
                        "current_person_index": current_index,
                        "bbox_iou": iou,
                        "state": state,
                        "regime": "stable" if previous_inout == current_inout else "switch",
                        "previous_inout": previous_inout,
                        "current_inout": current_inout,
                        "previous_score": previous_score,
                        "current_score": current_score,
                        "score_step": abs(current_score - previous_score),
                        "absolute_calibration_error": abs(current_score - current_inout),
                        "brier": (current_score - current_inout) ** 2,
                        "people_count": int(current_record["people_count"]),
                        "crowd_bin": current_record["crowd_bin"],
                        "head_size_bin": current_record["head_size_bin"],
                        "target_distance_bin": current_record["target_distance_bin"],
                        "current_l2": optional_float(current_record.get("l2")),
                        "prediction_step": None,
                        "gt_step": None,
                        "motion_residual": None,
                    }
                    previous_pred_x = optional_float(previous_record.get("pred_x"))
                    previous_pred_y = optional_float(previous_record.get("pred_y"))
                    current_pred_x = optional_float(current_record.get("pred_x"))
                    current_pred_y = optional_float(current_record.get("pred_y"))
                    if None not in (previous_pred_x, previous_pred_y, current_pred_x, current_pred_y):
                        row["prediction_step"] = euclidean(
                            previous_pred_x, previous_pred_y, current_pred_x, current_pred_y
                        )
                    previous_target_x = optional_float(previous_record.get("target_x"))
                    previous_target_y = optional_float(previous_record.get("target_y"))
                    current_target_x = optional_float(current_record.get("target_x"))
                    current_target_y = optional_float(current_record.get("target_y"))
                    if None not in (
                        previous_target_x, previous_target_y, current_target_x, current_target_y,
                        previous_pred_x, previous_pred_y, current_pred_x, current_pred_y,
                    ):
                        row["gt_step"] = euclidean(
                            previous_target_x, previous_target_y, current_target_x, current_target_y
                        )
                        row["motion_residual"] = euclidean(
                            current_pred_x - previous_pred_x,
                            current_pred_y - previous_pred_y,
                            current_target_x - previous_target_x,
                            current_target_y - previous_target_y,
                        )
                    output.append(row)
    tracking = {
        "sequence_count": len(selected_sequences),
        "adjacent_frame_pairs": adjacent_pairs,
        "possible_people_upper_bound": possible_people,
        "matched_people": matched_people,
        "match_rate_upper_bound": matched_people / possible_people if possible_people else None,
    }
    return output, tracking


def expected_calibration_error(rows: list[dict], bins: int) -> float | None:
    if not rows:
        return None
    scores = np.asarray([row["current_score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["current_inout"] for row in rows], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return ece


def safe_ap(rows: list[dict]) -> float | None:
    labels = [row["current_inout"] for row in rows]
    if len(set(labels)) < 2:
        return None
    return float(average_precision_score(labels, [row["current_score"] for row in rows]))


def safe_mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict], ece_bins: int) -> list[dict]:
    groups = []
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, model_rows in sorted(by_model.items()):
        groups.append((model, "overall", "all", model_rows))
        for dimension in ("state", "regime", "crowd_bin", "head_size_bin", "target_distance_bin"):
            buckets: dict[str, list[dict]] = defaultdict(list)
            for row in model_rows:
                buckets[str(row[dimension])].append(row)
            groups.extend((model, dimension, name, bucket) for name, bucket in sorted(buckets.items()))
    output = []
    for model, dimension, bin_name, group_rows in groups:
        output.append({
            "model": model,
            "dimension": dimension,
            "bin": bin_name,
            "n": len(group_rows),
            "sequence_n": len({row["sequence_id"] for row in group_rows}),
            "positive_rate": safe_mean(group_rows, "current_inout"),
            "inout_ap": safe_ap(group_rows),
            "ece": expected_calibration_error(group_rows, ece_bins),
            "brier": safe_mean(group_rows, "brier"),
            "absolute_calibration_error": safe_mean(group_rows, "absolute_calibration_error"),
            "score_step": safe_mean(group_rows, "score_step"),
            "l2": safe_mean(group_rows, "current_l2"),
            "prediction_step": safe_mean(group_rows, "prediction_step"),
            "gt_step": safe_mean(group_rows, "gt_step"),
            "motion_residual": safe_mean(group_rows, "motion_residual"),
        })
    return output


def clustered_delta(
    rows: list[dict],
    first_filter,
    second_filter,
    field: str,
    iterations: int,
    seed: int,
) -> dict:
    first = [row for row in rows if first_filter(row) and row.get(field) is not None]
    second = [row for row in rows if second_filter(row) and row.get(field) is not None]
    clusters = sorted({row["sequence_id"] for row in first + second})
    if not first or not second or not clusters:
        return {"delta": None, "ci95_low": None, "ci95_high": None, "first_n": len(first), "second_n": len(second)}
    point = safe_mean(first, field) - safe_mean(second, field)
    cluster_index = {name: index for index, name in enumerate(clusters)}
    first_sums = np.zeros(len(clusters), dtype=np.float64)
    first_counts = np.zeros(len(clusters), dtype=np.int64)
    second_sums = np.zeros(len(clusters), dtype=np.float64)
    second_counts = np.zeros(len(clusters), dtype=np.int64)
    for row in first:
        index = cluster_index[row["sequence_id"]]
        first_sums[index] += float(row[field])
        first_counts[index] += 1
    for row in second:
        index = cluster_index[row["sequence_id"]]
        second_sums[index] += float(row[field])
        second_counts[index] += 1
    rng = np.random.default_rng(seed)
    bootstraps = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        first_count = int(first_counts[sampled].sum())
        second_count = int(second_counts[sampled].sum())
        if first_count and second_count:
            bootstraps.append(
                float(first_sums[sampled].sum()) / first_count
                - float(second_sums[sampled].sum()) / second_count
            )
    return {
        "delta": point,
        "ci95_low": float(np.percentile(bootstraps, 2.5)) if bootstraps else None,
        "ci95_high": float(np.percentile(bootstraps, 97.5)) if bootstraps else None,
        "first_n": len(first),
        "second_n": len(second),
        "cluster_n": len(clusters),
    }


def comparisons(rows: list[dict], iterations: int, seed: int) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        specifications = (
            ("switch_minus_stable_brier", lambda row: row["regime"] == "switch", lambda row: row["regime"] == "stable", "brier"),
            ("switch_minus_stable_absolute_error", lambda row: row["regime"] == "switch", lambda row: row["regime"] == "stable", "absolute_calibration_error"),
            ("out_to_in_minus_stable_in_l2", lambda row: row["state"] == "out_to_in", lambda row: row["state"] == "stable_in", "current_l2"),
            ("stable_in_motion_residual", lambda row: row["state"] == "stable_in", lambda row: False, "motion_residual"),
        )
        for name, first_filter, second_filter, field in specifications:
            if name == "stable_in_motion_residual":
                values = [row for row in model_rows if first_filter(row) and row.get(field) is not None]
                result = {"delta": safe_mean(values, field), "ci95_low": None, "ci95_high": None, "first_n": len(values), "second_n": 0, "cluster_n": len({row['sequence_id'] for row in values})}
            else:
                result = clustered_delta(model_rows, first_filter, second_filter, field, iterations, seed)
            output.append({"model": model, "comparison": name, "positive_means": "first_group_worse", **result})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    previous = [{"bbox_norm": [0.1, 0.1, 0.3, 0.3]}, {"bbox_norm": [0.6, 0.1, 0.8, 0.3]}]
    current = [{"bbox_norm": [0.62, 0.1, 0.82, 0.3]}, {"bbox_norm": [0.11, 0.1, 0.31, 0.3]}]
    matches = greedy_iou_matches(previous, current, 0.3)
    assert [(first, second) for first, second, _ in matches] == [(0, 1), (1, 0)]
    rows = [
        {"current_score": 0.9, "current_inout": 1},
        {"current_score": 0.1, "current_inout": 0},
    ]
    assert expected_calibration_error(rows, 5) < 0.11
    assert safe_ap(rows) == 1.0
    frames = [
        {"path": "images/show/clip/0001.jpg", "heads": [{"bbox_norm": [0.1, 0.1, 0.3, 0.3], "inout": 1}]},
        {"path": "images/show/clip/0002.jpg", "heads": [{"bbox_norm": [0.11, 0.1, 0.31, 0.3], "inout": 1}]},
        {"path": "images/show/clip/0003.jpg", "heads": [{"bbox_norm": [0.12, 0.1, 0.32, 0.3], "inout": 0}]},
    ]
    records = {}
    for index, frame in enumerate(frames):
        records[(frame["path"], 0)] = {
            "inout_score": ("0.9", "0.8", "0.4")[index],
            "inout": ("1", "1", "0")[index],
            "people_count": "1", "crowd_bin": "1", "head_size_bin": "medium",
            "target_distance_bin": "near" if index < 2 else "out_or_missing",
            "l2": "0.1" if index < 2 else "", "pred_x": "0.5", "pred_y": "0.5",
            "target_x": "0.5" if index < 2 else "", "target_y": "0.5" if index < 2 else "",
        }
    transition_rows, tracking = build_transition_rows(
        frames, [("synthetic", Path("synthetic.csv"), records)], 0.3, 2, None
    )
    assert [row["state"] for row in transition_rows] == ["stable_in", "in_to_out"]
    assert tracking["matched_people"] == 2
    assert len(summarize(transition_rows, 5)) > 1
    comparison_rows = comparisons(transition_rows, 20, 3106)
    brier_row = next(
        row for row in comparison_rows if row["comparison"] == "switch_minus_stable_brier"
    )
    assert brier_row["delta"] is not None
    print("Self-test passed: IoU tracking, transition states, ECE, and AP.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--annotation-json", type=Path)
    parser.add_argument("--records", action="append", default=[], help="label=/absolute/path.records.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-frame-gap", type=int, default=2)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and (args.annotation_json is None or not args.records or args.output_dir is None):
        parser.error("--annotation-json, at least one --records, and --output-dir are required")
    if not 0 < args.iou_threshold <= 1 or args.max_frame_gap < 1 or args.ece_bins < 2:
        parser.error("invalid IoU/frame-gap/ECE configuration")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    payload = json.loads(args.annotation_json.read_text())
    frames = flatten_annotations("vat", payload)
    record_sets = [read_records(specification) for specification in args.records]
    transition_rows, tracking = build_transition_rows(
        frames, record_sets, args.iou_threshold, args.max_frame_gap, args.max_sequences
    )
    summary = summarize(transition_rows, args.ece_bins)
    comparison_rows = comparisons(transition_rows, args.bootstrap_iterations, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_transition.csv", transition_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "comparisons.csv", comparison_rows)
    report = {
        "status": "measured",
        "analysis": "vat_transition_conditioned_reliability",
        "annotation": file_manifest(args.annotation_json),
        "record_files": {label: file_manifest(path) for label, path, _ in record_sets},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "tracking": tracking,
        "transition_row_count": len(transition_rows),
        "summary": summary,
        "comparisons": comparison_rows,
        "claim_boundary": (
            "Greedy bbox-IoU produces approximate tracks; this audit diagnoses association-level "
            "reliability and does not establish a novel temporal method."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(transition_rows)} transition rows to {args.output_dir}")


if __name__ == "__main__":
    main()
