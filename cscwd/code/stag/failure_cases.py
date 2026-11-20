"""Read raw GazeSpot prediction caches and sample reproducible review cases.

This module uses NumPy only. Ground truth defines offline sampling strata and
review labels; it is never used to alter predictions or construct model inputs.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


DEFAULT_COUNTS = {
    "large_l2": 16,
    "inout_false_negative": 8,
    "inout_false_positive": 8,
    "medium_l2": 8,
    "correct_control": 8,
}
QUANTILES = {"p00": 0.0, "p25": 0.25, "p40": 0.4, "p50": 0.5,
             "p60": 0.6, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p100": 1.0}


class PredictionCache:
    """Read complete original per-person predictions in their stored row order."""

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        with (self.root / "index.json").open() as stream:
            self.index = json.load(stream)
        if (split not in ("train", "test") or self.index.get("task") != "T2"
                or not self.index.get("passed")):
            raise ValueError("Expected a complete raw T2 cache for official train or test")
        protocol = self.index["metadata"]["protocol"]
        if protocol["temporal_context"] != "none" or protocol["labels_used_in_forward"]:
            raise ValueError("Review requires raw single-frame predictions without GT in model forward")
        for name in ("heatmap", "inout", "argmax_xy"):
            array = np.load(self.root / self.index["files"][name]["path"],
                            mmap_mode="r", allow_pickle=False)
            setattr(self, name, array)
        n = self.index["person_frames"]
        if (self.heatmap.shape != (n, 64, 64) or self.inout.shape != (n,)
                or self.argmax_xy.shape != (n, 2)
                or any(value.dtype != np.float32
                       for value in (self.heatmap, self.inout, self.argmax_xy))):
            raise ValueError("Raw cache requires aligned FP32 heatmap[N,64,64], inout[N], argmax_xy[N,2]")
        # Read the large heatmap mmap once, in bounded chunks. Check that the
        # supplied coordinates are the raw maps' argmax, not adjusted targets.
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            maps = self.heatmap[start:stop]
            scores, points = self.inout[start:stop], self.argmax_xy[start:stop]
            if not all(np.isfinite(value).all() for value in (maps, scores, points)):
                raise ValueError("Raw prediction cache contains non-finite values")
            if ((maps < 0).any() or (maps > 1).any()
                    or (scores < 0).any() or (scores > 1).any()):
                raise ValueError("Raw sigmoid heatmaps and InOut scores must lie in [0,1]")
            flat = maps.reshape(len(maps), -1).argmax(axis=1)
            expected = np.column_stack((flat % 64 / 64, flat // 64 / 64)).astype(np.float32)
            if not np.array_equal(points, expected):
                raise ValueError("Cached argmax coordinates disagree with the raw heatmaps")
        self.frames = [None] * n
        self.segment_start = np.empty(n, dtype=np.int64)
        cursor = 0
        seen_ids = set()
        for track in sorted(self.index["tracks"], key=lambda item: item["start"]):
            track_id = track["track_id"]
            if (not track_id.startswith(split + "/") or track_id in seen_ids
                    or track["start"] != cursor
                    or track["stop"] - cursor != len(track["frames"])):
                raise ValueError("Tracks must uniquely cover this split's contiguous prediction rows")
            seen_ids.add(track_id)
            previous = None
            segment = cursor
            video_id = "/".join(PurePosixPath(track_id).parts[:2])
            for local_index, frame in enumerate(track["frames"]):
                row = track["start"] + local_index
                number = int(frame["frame_index"])
                if (previous is not None and number <= previous
                        or str(frame["person_id"]) != str(track["person_id"])):
                    raise ValueError("Track frame numbers must increase and person identity must stay fixed")
                if previous is None or number != previous + 1:
                    segment = row
                bbox = np.asarray(frame["bbox_norm"], dtype=float)
                gx, gy = float(frame["gazex_norm"][0]), float(frame["gazey_norm"][0])
                gt = int(frame["inout"])
                if (bbox.shape != (4,) or not np.isfinite(bbox).all()
                        or not np.isfinite([gx, gy]).all() or gt not in (0, 1)
                        or gt == 1 and (gx < 0 or gy < 0)):
                    raise ValueError("Each person frame requires a finite box, gaze annotation and binary InOut GT")
                l2 = float(np.linalg.norm(self.argmax_xy[row].astype(float) - [gx, gy])) if gt else None
                self.frames[row] = {
                    **frame, "track_id": track_id, "person_id": str(track["person_id"]),
                    "sequence_path": track["sequence_path"], "video_id": video_id,
                    "row": row, "local_index": local_index, "frame_index": number,
                    "score": float(self.inout[row]), "pred_xy": self.argmax_xy[row].astype(float).tolist(),
                    "l2": l2,
                }
                self.segment_start[row] = segment
                previous = number
            cursor = track["stop"]
        if cursor != n or not n:
            raise ValueError("The track index must cover every nonempty prediction row")
        self.gt_inside = np.asarray([frame["inout"] == 1 for frame in self.frames])
        self.l2 = np.asarray([frame["l2"] if frame["l2"] is not None else np.nan
                              for frame in self.frames])

    def __len__(self):
        return len(self.frames)


def _interleave(first, second):
    """Alternate inside/outside controls, retaining every candidate exactly once."""
    result = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            result.append(int(first[index]))
        if index < len(second):
            result.append(int(second[index]))
    return result


def select_cases(cache, seed=0, counts=None, history_frames=16, max_history_frames=64,
                 max_per_track=2, max_per_video=6, large_l2_quantile=0.9,
                 score_threshold=0.5):
    """Sample strata without relaxing diversity or non-overlapping-window limits.

    Category candidate pools may overlap; selected maximum history windows may
    not overlap within a track. Each split is sampled independently. Shortfalls
    describe unavailable candidates or the stated constraints, never replacement
    thresholds. Sampling proportions are not population failure frequencies.
    """
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if (set(counts) - set(DEFAULT_COUNTS) or any(value < 0 for value in counts.values())
            or not 0 <= history_frames <= max_history_frames or max_per_track < 1
            or max_per_video < 1 or not 0 <= large_l2_quantile <= 1
            or not 0 <= score_threshold <= 1):
        raise ValueError("Invalid review sampling configuration")
    rng = np.random.default_rng(seed)
    inside = cache.gt_inside
    predicted_inside = cache.inout >= score_threshold
    inside_l2 = cache.l2[inside]
    quantiles = ({name: float(np.quantile(inside_l2, q)) for name, q in QUANTILES.items()}
                 if len(inside_l2) else {name: None for name in QUANTILES})
    large_threshold = float(np.quantile(inside_l2, large_l2_quantile)) if len(inside_l2) else None
    empty = np.zeros(len(cache), dtype=bool)
    pools = {
        "large_l2": np.flatnonzero(inside & (cache.l2 >= large_threshold)) if len(inside_l2) else np.flatnonzero(empty),
        "inout_false_negative": np.flatnonzero(inside & ~predicted_inside),
        "inout_false_positive": np.flatnonzero(~inside & predicted_inside),
        "medium_l2": np.flatnonzero(inside & (cache.l2 >= quantiles["p40"])
                                     & (cache.l2 <= quantiles["p60"])) if len(inside_l2) else np.flatnonzero(empty),
        "correct_control": np.flatnonzero((inside & predicted_inside & (cache.l2 <= quantiles["p25"]))
                                            | (~inside & ~predicted_inside)) if len(inside_l2)
                           else np.flatnonzero(~inside & ~predicted_inside),
    }
    order = [name for name in DEFAULT_COUNTS if name in counts]
    candidates = {name: rng.permutation(pools[name]).tolist() for name in order}
    if "correct_control" in candidates:
        rows = candidates["correct_control"]
        candidates["correct_control"] = _interleave([row for row in rows if inside[row]],
                                                      [row for row in rows if not inside[row]])
    positions = Counter()
    selected = Counter()
    track_counts, video_counts = Counter(), Counter()
    selected_windows = defaultdict(list)
    rejected = {name: Counter() for name in order}
    cases = []
    for _ in range(max(counts.values(), default=0)):
        for category in order:
            if selected[category] >= counts[category]:
                continue
            rows = candidates[category]
            while positions[category] < len(rows):
                row = rows[positions[category]]
                positions[category] += 1
                anchor = cache.frames[row]
                track_id, video_id = anchor["track_id"], anchor["video_id"]
                start = max(int(cache.segment_start[row]), row - max_history_frames)
                reason = None
                if track_counts[track_id] >= max_per_track:
                    reason = "max_per_track"
                elif video_counts[video_id] >= max_per_video:
                    reason = "max_per_video"
                elif any(start <= last and row >= first for first, last in selected_windows[track_id]):
                    reason = "overlapping_history_window"
                if reason:
                    rejected[category][reason] += 1
                    continue
                identity = f"{cache.split}\0{category}\0{track_id}\0{anchor['frame_index']}\0{anchor['path']}"
                digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
                available = row - int(cache.segment_start[row])
                cases.append({
                    "case_id": f"{cache.split}_{category}_{digest}", "category": category,
                    "split": cache.split, "track_id": track_id, "person_id": anchor["person_id"],
                    "sequence_path": anchor["sequence_path"], "video_id": video_id,
                    "anchor_row": row, "anchor_frame_index": anchor["frame_index"],
                    "anchor_l2": anchor["l2"], "anchor_score": anchor["score"],
                    "anchor_gt_inout": int(anchor["inout"]), "anchor_pred_inout": int(predicted_inside[row]),
                    "available_history": available, "included_history": row - start,
                    "default_history": min(history_frames, available),
                    "window_start_row": start, "window_stop_row": row + 1,
                    "frames": cache.frames[start:row + 1],
                })
                selected[category] += 1
                track_counts[track_id] += 1
                video_counts[video_id] += 1
                selected_windows[track_id].append((start, row))
                break
    # Hide the round-robin category pattern from the human review order.
    rng.shuffle(cases)
    categories = {
        name: {"candidate_count": len(pools[name]), "requested": counts[name],
               "selected": selected[name], "shortfall": counts[name] - selected[name],
               "skipped_by_constraint": dict(rejected[name])}
        for name in order
    }
    confusion = {
        "TP": int(np.sum(inside & predicted_inside)), "TN": int(np.sum(~inside & ~predicted_inside)),
        "FP": int(np.sum(~inside & predicted_inside)), "FN": int(np.sum(inside & ~predicted_inside)),
    }
    summary = {
        "split": cache.split, "person_frames": len(cache), "track_count": len(cache.index["tracks"]),
        "video_count": len({frame["video_id"] for frame in cache.frames}),
        "in_frame_person_frames": int(inside.sum()), "out_of_frame_person_frames": int((~inside).sum()),
        "confusion": confusion, "l2_mean": float(inside_l2.mean()) if len(inside_l2) else None,
        "l2_quantiles": quantiles, "large_l2_threshold": large_threshold,
        "categories": categories, "selected_total": len(cases), "requested_total": sum(counts.values()),
        "shortfall_total": sum(counts.values()) - len(cases),
        "selected_tracks": len(track_counts), "selected_videos": len(video_counts),
        "correct_control_in_frame": sum(case["category"] == "correct_control" and case["anchor_gt_inout"] == 1 for case in cases),
        "correct_control_out_of_frame": sum(case["category"] == "correct_control" and case["anchor_gt_inout"] == 0 for case in cases),
    }
    config = {
        "split": cache.split, "seed": seed, "counts": counts, "history_frames": history_frames,
        "max_history_frames": max_history_frames, "max_per_track": max_per_track,
        "max_per_video": max_per_video, "large_l2_quantile": large_l2_quantile,
        "score_threshold": score_threshold, "medium_l2_quantiles": [0.4, 0.6],
        "correct_in_frame_l2_max_quantile": 0.25,
        "selection_order": order, "selection_schedule": "round_robin_categories",
        "display_order": "seeded_shuffle_after_selection",
        "history_policy": "current_and_past_same_person_contiguous_original_frame_numbers",
        "overlap_policy": "maximum_included_history_windows_disjoint_within_each_track",
        "video_grouping": "track_id_first_two_components_split_and_annotation_video",
        "sampling_note": "stratified_review_cases_are_not_population_failure_frequencies",
    }
    return {"cases": cases, "summary": summary, "config": config}
