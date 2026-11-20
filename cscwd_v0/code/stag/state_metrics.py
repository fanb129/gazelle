"""State-only temporal evaluation, separate from joint gaze localization.

Ground-truth labels define offline evaluation masks and event windows only.
No heatmap or gaze-coordinate inputs are used and predictions stay unchanged.
"""

import numpy as np

from .metrics import TemporalConfig


STATE_THRESHOLD = 0.5


def state_temporal_protocol(config=TemporalConfig()):
    """Describe the state-only protocol without changing the existing T4 one."""
    return {
        "version": "state_temporal_v1",
        "prediction": "in_frame_if_continuous_score>0.5; equality_is_out_of_frame",
        "state_threshold": STATE_THRESHOLD,
        "state_error": "all_valid_person_frames; predicted_state!=GT_state",
        "stable_pair_mask": "same_track; original_frame_index_difference==1; GT_state_unchanged",
        "state_flip": "prediction_state_change_on_stable_pair_mask",
        "state_flip_denominator": "all_eligible_stable_GT_pairs; also_split_by_GT_state",
        "events": "contiguous_GT_inout_flip_only; no_gaze_shift_events",
        "event_types": ["enter_frame", "leave_frame"],
        "event_window": "from_event_until_next_GT_state_event_gap_track_end_or_max_response_frames; exclusive_end; gaze_displacement_does_not_truncate",
        "response_success": "predicted_state_matches_GT; no_localization_condition",
        "response_delay": "first_start_of_hold_frames_consecutive_successes_minus_event_original_frame_index; zero_allowed",
        "hold_frames": config.hold_frames,
        "max_response_frames": config.max_response_frames,
        "censoring": "window_shorter_than_hold_frames=insufficient_followup; otherwise_no_success_run=censored",
        "response_rate_denominator": "events_with_at_least_hold_frames_followup; includes_censored_events",
        "delay_statistics": "responded_events_only; also_report_response_rate_and_censored_counts",
        "aggregation": "pool_person_frames_pairs_and_events; not_equal_track_weighted",
        "label_usage": "offline_evaluation_only; never_prediction_inputs_or_memory_resets",
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
    for kind in ("all", "enter_frame", "leave_frame"):
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


def _count_summary(counts, events):
    def errors(row):
        frames, wrong, _, _ = map(int, row)
        return {"person_frames": frames, "errors": wrong,
                "error_rate": wrong / frames if frames else None}

    def flips(row):
        _, _, pairs, changed = map(int, row)
        return {"eligible_pairs": pairs, "flips": changed,
                "flip_rate": changed / pairs if pairs else None}

    classes = {"out_of_frame": counts[0], "in_frame": counts[1]}
    return {
        "state_error": {**errors(counts.sum(axis=0)),
                        "by_gt_state": {name: errors(row) for name, row in classes.items()}},
        "stable_state_flips": {**flips(counts.sum(axis=0)),
                               "by_gt_state": {name: flips(row) for name, row in classes.items()}},
        "response": _response_summary(events),
    }


def _state_events(track, indices, labels, contiguous, correct, config):
    positions = np.flatnonzero(contiguous & (labels[1:] != labels[:-1])) + 1
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
            if correct[start:start + config.hold_frames].all():
                delay = int(indices[start] - indices[position])
                status = "responded"
                break
        events.append({
            "track_id": track["track_id"],
            "type": "enter_frame" if labels[position] else "leave_frame",
            "event_frame_index": int(indices[position]),
            "event_global_index": int(track["start"] + position),
            "window_frames": window_frames, "status": status, "response_delay_frames": delay,
        })
    return events


def evaluate_state_temporal(tracks, inout, config=TemporalConfig()):
    """Evaluate score>0.5 state errors, false flips and state-only responses.

    Tracks use the existing global [start, stop) cache slices. Response windows
    depend on GT state transitions, gaps and the configured horizon, never on
    target-coordinate movement or predicted changes.
    """
    total_counts = np.zeros((2, 4), dtype=np.int64)
    all_events, track_summaries = [], []
    for track in tracks:
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        labels = np.asarray([frame["inout"] == 1 for frame in frames])
        predicted = np.asarray(inout[track["start"]:track["stop"]]) > STATE_THRESHOLD
        correct = predicted == labels
        contiguous = np.diff(indices) == 1
        stable_pairs = contiguous & (labels[1:] == labels[:-1])
        predicted_flips = predicted[1:] != predicted[:-1]
        counts = np.zeros((2, 4), dtype=np.int64)
        for label in (0, 1):
            selected_frames = labels == label
            selected_pairs = stable_pairs & (labels[:-1] == label)
            counts[label] = (selected_frames.sum(), (selected_frames & ~correct).sum(),
                             selected_pairs.sum(), (selected_pairs & predicted_flips).sum())
        events = _state_events(track, indices, labels, contiguous, correct, config)
        track_summaries.append({"track_id": track["track_id"], **_count_summary(counts, events)})
        total_counts += counts
        all_events.extend(events)
    return {"summary": _count_summary(total_counts, all_events),
            "tracks": track_summaries, "events": all_events}
