"""Shared frame-weighted VAT metrics for single-frame and STAG predictions.

This module evaluates predictions only. Ground-truth gaze and in/out labels are
never used by the temporal gate. The AUC rectangle matches gazelle.utils.vat_auc,
including its exclusive right/bottom slice capped at 63.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


STANDARD_PROTOCOL = {
    "localization_filter": "ground_truth_inout == 1",
    "auc_reference": "gazelle/utils.py::vat_auc",
    "auc_grid": [64, 64],
    "auc_tolerance": "binary_rectangle; sigma=3; radius=3*sigma",
    "auc_slice": "exclusive_right_bottom; upper_bound_capped_at_63",
    "l2_coordinates": "x=column/64; y=row/64; first_flat_argmax_on_ties",
    "inout_ap_positive_class": 1,
    "inout_ap_samples": "all_person_frames",
    "localization_aggregation": "mean_per_in_frame_person_frame",
}


def vat_auc_numpy(heatmap, gazex, gazey):
    """Reproduce the existing 64x64 VAT tolerance-region AUC without PyTorch."""
    res = 64
    sigma = 3
    x, y = gazex * res, gazey * res
    left, top = max(0, int(x - 3 * sigma)), max(0, int(y - 3 * sigma))
    right = min(int(x + 3 * sigma + 1), res - 1)
    bottom = min(int(y + 3 * sigma + 1), res - 1)
    target = np.zeros((res, res))
    target[top:bottom, left:right] = 1
    return float(roc_auc_score(target.ravel(), heatmap.ravel()))


def evaluate_predictions(tracks, heatmaps, inout):
    """Evaluate all person frames; localization uses only in-frame annotations.

    Tracks use T2 global [start, stop) slices and retain native frame records.
    AUC and L2 are averaged over individual in-frame annotations, so longer
    tracks contribute proportionally. AP treats in-frame=1 as the positive class.
    """
    aucs, distances, labels, scores = [], [], [], []
    for track in tracks:
        for position, frame in enumerate(track["frames"], start=track["start"]):
            label = frame["inout"]
            labels.append(label)
            scores.append(float(inout[position]))
            if label == 1:
                heatmap = heatmaps[position]
                gazex, gazey = frame["gazex_norm"][0], frame["gazey_norm"][0]
                aucs.append(vat_auc_numpy(heatmap, gazex, gazey))
                row, column = np.unravel_index(heatmap.argmax(), (64, 64))
                distances.append(np.hypot(column / 64 - gazex, row / 64 - gazey))
    return {
        "auc": float(np.mean(aucs)),
        "l2": float(np.mean(distances)),
        "inout_ap": float(average_precision_score(labels, scores)),
        "person_frames": len(labels),
        "localization_frames": len(distances),
    }


@dataclass(frozen=True)
class TemporalConfig:
    shift_threshold: float = 0.15
    response_radius: float = 0.10
    stable_threshold: float = 0.02
    hold_frames: int = 3
    max_response_frames: int = 30


def temporal_protocol(config):
    """Return the complete, fixed temporal evaluation definition for provenance."""
    return {
        "thresholds": asdict(config),
        "coordinates": "normalized_xy; cached_argmax=(column/64,row/64)",
        "adjacency": "same_track; original_frame_index_difference==1; no_gap_bridging",
        "jitter": "prediction_displacement_on_adjacent_GT_inframe_pairs",
        "ground_truth_reference": "GT_displacement_on_the_identical_jitter_pair_mask",
        "stability": "hold_frames_contiguous_GT_inframe_window; all_prediction_steps<=stable_threshold",
        "stability_denominator": "eligible_window_end_frames; each_window_has_hold_frames_frames",
        "accurate_stability": "stable_prediction_window; every_predicted_inout>0.5; every_GT_error<=response_radius",
        "ground_truth_stability": "same_windows_and_displacement_threshold_as_prediction_stability",
        "events": "GT_proxy; contiguous_inout_flip_or_both_GT_inframe_and_GT_displacement>shift_threshold",
        "event_types": ["enter_frame", "leave_frame", "gaze_shift"],
        "event_window": "from_event_until_next_GT_event_gap_track_end_or_max_response_frames; exclusive_end",
        "response_success": "predicted_inout>0.5_matches_GT; inframe_also_requires_current_GT_error<=response_radius",
        "response_delay": "first_start_of_hold_frames_consecutive_successes_minus_event_original_frame_index; zero_allowed",
        "censoring": "window_shorter_than_hold_frames=insufficient_followup; otherwise_no_success_run=censored",
        "response_rate_denominator": "events_with_at_least_hold_frames_followup; includes_censored_events",
        "delay_statistics": "responded_events_only; always_report_response_rate_and_censored_counts",
        "aggregation": "pool_person_frame_pairs_windows_and_events; not_equal_track_weighted",
        "distribution": "count_mean_median_p90_p95; empty_statistics=null",
        "label_usage": "offline_evaluation_only; never_gate_inputs_or_prediction_state_resets",
    }


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95))}


def _response_summary(events):
    summary = {}
    for kind in ("all", "enter_frame", "leave_frame", "gaze_shift"):
        selected = events if kind == "all" else [event for event in events if event["type"] == kind]
        responded = [event["response_delay_frames"] for event in selected if event["status"] == "responded"]
        short = sum(event["status"] == "insufficient_followup" for event in selected)
        censored = sum(event["status"] == "censored" for event in selected)
        assessable = len(selected) - short
        summary[kind] = {
            "events": len(selected), "assessable_events": assessable,
            "insufficient_followup_events": short, "responded_events": len(responded),
            "censored_events": censored,
            "response_rate": len(responded) / assessable if assessable else None,
            "responded_delay_frames": _distribution(responded),
        }
    return summary


def _temporal_summary(predicted_steps, gt_steps, stability, events):
    eligible, predicted, gt, accurate = stability
    return {
        "jitter": {"pairs": len(predicted_steps), "predicted": _distribution(predicted_steps),
                   "ground_truth": _distribution(gt_steps)},
        "stability": {
            "eligible_windows": eligible, "stable_predicted_frames": predicted,
            "stable_gt_frames": gt, "accurate_stable_frames": accurate,
            "stable_ratio": predicted / eligible if eligible else None,
            "ground_truth_stable_ratio": gt / eligible if eligible else None,
            "accurate_stable_ratio": accurate / eligible if eligible else None,
        },
        "response": _response_summary(events),
    }


def _track_events(track, indices, labels, gt_steps, contiguous, success, config):
    transitions = labels[1:] != labels[:-1]
    gaze_shifts = labels[1:] & labels[:-1] & (gt_steps > config.shift_threshold)
    positions = np.flatnonzero(contiguous & (transitions | gaze_shifts)) + 1
    gaps = np.flatnonzero(~contiguous) + 1
    events = []
    for number, position in enumerate(positions):
        next_event = positions[number + 1] if number + 1 < len(positions) else len(indices)
        next_gap_number = np.searchsorted(gaps, position, side="right")
        next_gap = gaps[next_gap_number] if next_gap_number < len(gaps) else len(indices)
        stop = min(next_event, next_gap, len(indices), position + config.max_response_frames)
        window_frames = int(stop - position)
        delay = None
        status = "insufficient_followup" if window_frames < config.hold_frames else "censored"
        for start in range(position, stop - config.hold_frames + 1):
            if success[start:start + config.hold_frames].all():
                delay = int(indices[start] - indices[position])
                status = "responded"
                break
        if transitions[position - 1]:
            kind = "enter_frame" if labels[position] else "leave_frame"
        else:
            kind = "gaze_shift"
        events.append({
            "track_id": track["track_id"], "type": kind,
            "event_frame_index": int(indices[position]),
            "event_global_index": int(track["start"] + position),
            "window_frames": window_frames, "status": status, "response_delay_frames": delay,
        })
    return events


def evaluate_temporal(tracks, argmax_xy, inout, config):
    """Evaluate fixed predictions with GT-defined pairs, windows and proxy events.

    All future labels used here belong only to offline scoring. The temporal gate
    is evaluated as saved; this function never alters predictions or model state.
    """
    all_predicted_steps, all_gt_steps, all_events, track_summaries = [], [], [], []
    total_stability = np.zeros(4, dtype=np.int64)
    for track in tracks:
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        labels = np.asarray([frame["inout"] == 1 for frame in frames])
        gt = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames])
        predicted = np.asarray(argmax_xy[track["start"]:track["stop"]], dtype=np.float64)
        scores = np.asarray(inout[track["start"]:track["stop"]])
        contiguous = np.diff(indices) == 1
        predicted_steps = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        gt_steps = np.linalg.norm(np.diff(gt, axis=0), axis=1)
        pair_mask = contiguous & labels[:-1] & labels[1:]
        selected_predicted, selected_gt = predicted_steps[pair_mask], gt_steps[pair_mask]
        errors = np.linalg.norm(predicted - gt, axis=1)
        predicted_inframe = scores > 0.5
        success = (predicted_inframe == labels) & (~labels | (errors <= config.response_radius))
        stability = np.zeros(4, dtype=np.int64)
        for start in range(len(frames) - config.hold_frames + 1):
            stop = start + config.hold_frames
            if labels[start:stop].all() and contiguous[start:stop - 1].all():
                stable_predicted = bool((predicted_steps[start:stop - 1] <= config.stable_threshold).all())
                stable_gt = bool((gt_steps[start:stop - 1] <= config.stable_threshold).all())
                accurate_stable = stable_predicted and bool(success[start:stop].all())
                stability += [1, stable_predicted, stable_gt, accurate_stable]
        events = _track_events(track, indices, labels, gt_steps, contiguous, success, config)
        counts = [int(value) for value in stability]
        track_summaries.append({"track_id": track["track_id"],
                                **_temporal_summary(selected_predicted, selected_gt, counts, events)})
        all_predicted_steps.extend(selected_predicted)
        all_gt_steps.extend(selected_gt)
        all_events.extend(events)
        total_stability += stability
    return {
        "summary": _temporal_summary(all_predicted_steps, all_gt_steps,
                                     [int(value) for value in total_stability], all_events),
        "tracks": track_summaries, "events": all_events,
    }
