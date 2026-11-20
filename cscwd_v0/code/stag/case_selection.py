"""Deterministic, outcome-labelled T6 case selection from frozen T5 outputs."""

from collections import defaultdict
import hashlib

import numpy as np


CATEGORY_ORDER = ("stag_over_ema", "single_frame_only", "stag_lag", "stable_gain", "crowded")
METHODS = ("single_frame", "stag", "ema_alpha_0p25", "ema_alpha_0p5")


def select_cases(index, arrays_by_method, details, config, cases_per_group=2,
                 context_before=10, context_after=20, min_heads=4):
    """Select diagnostic cases without changing predictions or ranking by gain.

    Context indices are global half-open cache slices, confined to the anchor's
    track and contiguous original-frame segment. All single-frame-only failures
    are retained; other groups use the first cases in track/frame order.
    """
    tracks = {track["track_id"]: track for track in index["tracks"]}
    heads = defaultdict(set)
    for track in index["tracks"]:
        for frame in track["frames"]:
            heads[frame["path"]].add(track["track_id"])
    candidates = {category: [] for category in CATEGORY_ORDER}

    def make_case(category, track_id, anchor, reason, event=None):
        track = tracks[track_id]
        position = anchor - track["start"]
        frames = track["frames"]
        left, right = position, position + 1
        while left > 0 and frames[left]["frame_index"] - frames[left - 1]["frame_index"] == 1:
            left -= 1
        while right < len(frames) and frames[right]["frame_index"] - frames[right - 1]["frame_index"] == 1:
            right += 1
        frame = frames[position]
        identity = f"{track_id}\0{frame['frame_index']}".encode()
        return {
            "category": category,
            "case_id": f"{category}_{hashlib.sha256(identity).hexdigest()[:16]}",
            "track_id": track_id, "anchor_global_index": int(anchor),
            "anchor_frame_index": int(frame["frame_index"]),
            "context_start": int(track["start"] + max(left, position - context_before)),
            "context_stop": int(track["start"] + min(right, position + context_after + 1)),
            "reason": reason, "event": event, "head_count": len(heads[frame["path"]]),
        }

    events = {method: {(event["track_id"], event["event_global_index"]): event
                       for event in details[method]["events"] if event["type"] == "gaze_shift"}
              for method in METHODS}
    for key in sorted(events["single_frame"]):
        aligned = {method: events[method][key] for method in METHODS}
        baseline, stag, ema = (aligned[method] for method in METHODS[:3])
        track_id, anchor = key
        if stag["status"] == "responded" and ema["status"] != "responded":
            candidates["stag_over_ema"].append(make_case(
                "stag_over_ema", track_id, anchor, "STAG responds; fixed EMA alpha=0.25 does not", aligned))
        if baseline["status"] == "responded" and stag["status"] != "responded":
            candidates["single_frame_only"].append(make_case(
                "single_frame_only", track_id, anchor, "Single-frame responds; STAG does not", aligned))
        if (baseline["status"] == stag["status"] == "responded"
                and stag["response_delay_frames"] > baseline["response_delay_frames"]):
            candidates["stag_lag"].append(make_case(
                "stag_lag", track_id, anchor, "STAG responds later than single-frame", aligned))

    for track_id, track in sorted(tracks.items()):
        frames = track["frames"]
        selection = slice(track["start"], track["stop"])
        indices = np.asarray([frame["frame_index"] for frame in frames])
        inside = np.asarray([frame["inout"] == 1 for frame in frames])
        gt = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames])
        baseline = np.asarray(arrays_by_method["single_frame"]["argmax_xy"][selection], dtype=np.float64)
        stag = np.asarray(arrays_by_method["stag"]["argmax_xy"][selection], dtype=np.float64)
        scores = arrays_by_method["stag"]["inout"][selection]
        adjacent = np.diff(indices) == 1
        gt_steps = np.linalg.norm(np.diff(gt, axis=0), axis=1)
        baseline_steps = np.linalg.norm(np.diff(baseline, axis=0), axis=1)
        stag_steps = np.linalg.norm(np.diff(stag, axis=0), axis=1)
        accurate = (scores > 0.5) & (np.linalg.norm(stag - gt, axis=1) <= config.response_radius)
        previous_hit = None
        for end in range(config.hold_frames - 1, len(frames)):
            start = end - config.hold_frames + 1
            hit = (inside[start:end + 1].all() and adjacent[start:end].all()
                   and (gt_steps[start:end] <= config.stable_threshold).all()
                   and (stag_steps[start:end] <= config.stable_threshold).all()
                   and accurate[start:end + 1].all()
                   and (baseline_steps[start:end] > config.stable_threshold).any())
            if hit:
                if previous_hit != end - 1 or indices[end] - indices[end - 1] != 1:
                    candidates["stable_gain"].append(make_case(
                        "stable_gain", track_id, track["start"] + end,
                        "First window of episode: GT/STAG stable, STAG accurate, single-frame unstable"))
                previous_hit = end

    crowded_clips = {}
    for track_id, track in sorted(tracks.items()):
        for position, frame in enumerate(track["frames"]):
            if len(heads[frame["path"]]) >= min_heads:
                choice = (frame["frame_index"], track_id, track["start"] + position)
                clip = track["sequence_path"]
                if clip not in crowded_clips or choice < crowded_clips[clip]:
                    crowded_clips[clip] = choice
    for _, track_id, anchor in crowded_clips.values():
        candidates["crowded"].append(make_case(
            "crowded", track_id, anchor, f"Earliest clip frame with at least {min_heads} annotated people"))

    selected = []
    counts = {}
    for category, group in candidates.items():
        group.sort(key=lambda case: (case["track_id"], case["anchor_frame_index"]))
        chosen = group if category == "single_frame_only" else group[:cases_per_group]
        selected.extend(chosen)
        counts[category] = {"candidates": len(group), "selected": len(chosen)}
    return {
        "protocol": {
            "category_order": list(CATEGORY_ORDER), "event_type": "gaze_shift",
            "event_pairing": "track_id_and_event_global_index; frozen_T5_details",
            "ranking": "track_id_then_anchor_frame_index; not_effect_size",
            "failure_retention": "all_single_frame_only_candidates_selected",
            "cases_per_other_group": cases_per_group,
            "stable_window_frames": config.hold_frames, "stable_threshold": config.stable_threshold,
            "response_radius": config.response_radius,
            "stable_episode": "first_hit_window_of_consecutive_hit_end_frames",
            "crowd": "distinct_annotated_track_ids_per_image; earliest_frame_per_clip; smallest_track_id_on_ties",
            "min_annotated_heads": min_heads, "context_before": context_before,
            "context_after": context_after,
            "context": "offline_display_only; clipped_to_same_track_and_contiguous_original_frames",
            "empty_groups": "reported_as_zero; no_relaxed_thresholds_or_substitution",
        },
        "candidates": candidates, "selected": selected, "counts": counts,
    }
