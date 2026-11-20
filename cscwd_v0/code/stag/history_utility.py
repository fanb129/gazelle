"""Offline train-only utility scoring of already generated causal candidates.

GT is read only here, after candidate generation. No choice or evaluation
result is returned to candidate memory. The oracle is a reference for this
fixed candidate set, not a predictor or an upper bound on all video evidence.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from .metrics import TemporalConfig, _distribution, evaluate_temporal
from .state_metrics import evaluate_state_temporal


VERSION = "r4_history_utility_v1"
CANDIDATE_NAMES = ("current", "current_grid", "previous_raw", "ema_075", "transport_previous")
ORACLE_NAMES = ("current", "previous_raw", "ema_075", "transport_previous")
GROUP_NAMES = ("start_or_gap", "gt_out", "enter_frame", "stable_pair",
               "intermediate_motion", "gaze_shift")


def _frame_labels(tracks, person_frames, history_available, config):
    labels = np.empty(person_frames, dtype=bool)
    gt = np.empty((person_frames, 2), dtype=np.float64)
    expected_history = np.zeros(person_frames, dtype=bool)
    group_codes = np.zeros(person_frames, dtype=np.int8)
    for track in tracks:
        start, stop = track["start"], track["stop"]
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        inside = np.asarray([frame["inout"] == 1 for frame in frames])
        points = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]]
                             for frame in frames], dtype=np.float64)
        contiguous = np.r_[False, np.diff(indices) == 1]
        labels[start:stop], gt[start:stop] = inside, points
        expected_history[start:stop] = contiguous
        groups = np.zeros(len(frames), dtype=np.int8)
        groups[contiguous & ~inside] = GROUP_NAMES.index("gt_out")
        previous_inside = np.r_[False, inside[:-1]]
        groups[contiguous & inside & ~previous_inside] = GROUP_NAMES.index("enter_frame")
        displacement = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1)]
        pair = contiguous & inside & previous_inside
        groups[pair & (displacement <= config.stable_threshold)] = GROUP_NAMES.index("stable_pair")
        groups[pair & (displacement > config.stable_threshold)
               & (displacement <= config.shift_threshold)] = GROUP_NAMES.index("intermediate_motion")
        groups[pair & (displacement > config.shift_threshold)] = GROUP_NAMES.index("gaze_shift")
        group_codes[start:stop] = groups
    if not np.array_equal(expected_history, history_available):
        raise ValueError("History availability differs from same-track contiguous frame metadata")
    return labels, gt, group_codes


def _quadrants(current, candidate, selected):
    return {
        "both_correct": int(np.sum(selected & current & candidate)),
        "current_only_correct": int(np.sum(selected & current & ~candidate)),
        "history_only_correct": int(np.sum(selected & ~current & candidate)),
        "both_wrong": int(np.sum(selected & ~current & ~candidate)),
    }


def _utility(current_error, candidate_error, predicted_inside, selected, radius):
    current, candidate = current_error <= radius, candidate_error <= radius
    localization = _quadrants(current, candidate, selected)
    joint = _quadrants(current & predicted_inside, candidate & predicted_inside, selected)
    gain = current_error[selected] - candidate_error[selected]
    return {
        "frames": int(selected.sum()),
        "localization": localization,
        "fixed_inout_joint": joint,
        "rescue_blocked_by_raw_out": int(np.sum(selected & ~current & candidate & ~predicted_inside)),
        "net_rescue_minus_harm": localization["history_only_correct"] - localization["current_only_correct"],
        "l2_gain": _distribution(gain),
        "candidate_better": int(np.sum(gain > 0)),
        "candidate_equal": int(np.sum(gain == 0)),
        "candidate_worse": int(np.sum(gain < 0)),
    }


def analyze_history_utility(tracks, candidate_xy, raw_inout, history_available, config=None):
    """Score fixed [N,5,2] coordinates without generating or changing predictions.

    The main utility denominator is current GT-in-frame with an immediately
    preceding same-person frame. Array rows use the source cache's global row
    order. ``errors`` is NaN outside GT-in-frame; oracle choices are original
    candidate column numbers, or -1 outside GT-in-frame.
    """
    config = config or TemporalConfig()
    candidate_xy = np.asarray(candidate_xy)
    raw_inout = np.asarray(raw_inout)
    history_available = np.asarray(history_available, dtype=bool)
    if (candidate_xy.shape != (len(raw_inout), len(CANDIDATE_NAMES), 2)
            or history_available.shape != raw_inout.shape):
        raise ValueError("Candidate coordinates, scores and history mask have different coverage")
    labels, gt, group_codes = _frame_labels(tracks, len(raw_inout), history_available, config)
    errors = np.linalg.norm(candidate_xy.astype(np.float64) - gt[:, None, :], axis=-1)
    errors[~labels] = np.nan
    eligible = labels & history_available
    predicted_inside = raw_inout > 0.5
    state = evaluate_state_temporal(tracks, raw_inout, config)
    candidates, details = {}, {"candidates": {}, "state": {key: state[key] for key in ("tracks", "events")}}
    for column, name in enumerate(CANDIDATE_NAMES):
        temporal = evaluate_temporal(tracks, candidate_xy[:, column], raw_inout, config)
        candidates[name] = {
            "l2": float(errors[labels, column].mean()) if labels.any() else None,
            "temporal": temporal["summary"],
            "state": state["summary"],
            "utility": _utility(errors[:, 0], errors[:, column], predicted_inside,
                                eligible, config.response_radius),
            "groups": {group: _utility(errors[:, 0], errors[:, column], predicted_inside,
                                       eligible & (group_codes == number), config.response_radius)
                       for number, group in enumerate(GROUP_NAMES)},
        }
        details["candidates"][name] = {key: temporal[key] for key in ("tracks", "events")}

    oracle_columns = np.asarray([CANDIDATE_NAMES.index(name) for name in ORACLE_NAMES])
    inside_positions = np.flatnonzero(labels)
    oracle_choice = np.full(len(raw_inout), -1, dtype=np.int8)
    oracle_choice[labels] = oracle_columns[np.argmin(errors[np.ix_(labels, oracle_columns)], axis=1)]
    oracle_error = np.full(len(raw_inout), np.nan)
    oracle_error[labels] = errors[inside_positions, oracle_choice[labels]]
    current_wrong = labels & (errors[:, 0] > config.response_radius)
    rescued = current_wrong & (oracle_error <= config.response_radius)
    oracle = {
        "interpretation": "GT_informed_fixed_candidate_set_reference; not_deployable_or_all_video_information_upper_bound",
        "candidate_names": list(ORACLE_NAMES),
        "tie_break": "current_then_fixed_ORACLE_NAMES_order",
        "localization_frames": int(labels.sum()),
        "l2": float(oracle_error[labels].mean()) if labels.any() else None,
        "current_l2": candidates["current"]["l2"],
        "current_wrong_frames": int(current_wrong.sum()),
        "rescued_current_errors": int(rescued.sum()),
        "choice_counts": {name: int(np.sum(oracle_choice == CANDIDATE_NAMES.index(name)))
                          for name in ORACLE_NAMES},
        "utility_history_available": {
            "frames": int(eligible.sum()),
            "current_wrong_frames": int(np.sum(current_wrong & eligible)),
            "rescued_current_errors": int(np.sum(rescued & eligible)),
            "fixed_inout_joint_rescues": int(np.sum(rescued & eligible & predicted_inside)),
            "rescue_blocked_by_raw_out": int(np.sum(rescued & eligible & ~predicted_inside)),
        },
    }
    utility_rows = [row for item in candidates.values()
                    for row in (item["utility"], *item["groups"].values())]
    leave_reference = candidates["current"]["temporal"]["response"]["leave_frame"]
    checks = {
        "history_availability_matches_metadata": True,
        "history_candidates_fallback_to_current_without_history": all(
            np.array_equal(candidate_xy[~history_available, column], candidate_xy[~history_available, 0])
            for column in (2, 3, 4)),
        "all_quadrant_counts_match_their_denominators": all(
            sum(row[kind].values()) == row["frames"]
            for row in utility_rows for kind in ("localization", "fixed_inout_joint")),
        "group_counts_cover_all_frames": int(sum(np.sum(group_codes == number)
            for number in range(len(GROUP_NAMES)))) == len(raw_inout),
        "utility_groups_partition_main_denominator": all(
            sum(row["frames"] for row in item["groups"].values()) == item["utility"]["frames"]
            for item in candidates.values()),
        "candidate_state_summaries_equal_current": all(
            item["state"] == candidates["current"]["state"] for item in candidates.values()),
        "candidate_leave_responses_equal_current": all(
            item["temporal"]["response"]["leave_frame"] == leave_reference for item in candidates.values()),
        "oracle_never_has_larger_point_error_than_current": bool(
            np.all(oracle_error[labels] <= errors[labels, 0])),
        "oracle_excludes_resolution_control": 1 not in oracle_columns,
        "oracle_choices_cover_all_in_frame_frames": sum(oracle["choice_counts"].values()) == int(labels.sum()),
    }
    if not all(checks.values()):
        raise ValueError(f"History utility verification failed: {checks}")
    return {
        "summary": {
            "version": VERSION,
            "candidate_names": list(CANDIDATE_NAMES),
            "coverage": {"tracks": len(tracks), "person_frames": len(raw_inout),
                         "localization_frames": int(labels.sum()),
                         "history_available_frames": int(history_available.sum()),
                         "utility_frames": int(eligible.sum()),
                         "group_counts": {group: int(np.sum(group_codes == number))
                                          for number, group in enumerate(GROUP_NAMES)}},
            "inout_ap": float(average_precision_score(labels, raw_inout)) if labels.any() else 0.,
            "inout_policy": "same_raw_current_score_for_all_candidates; AP_computed_once",
            "localization_correctness_radius": config.response_radius,
            "group_code_names": list(GROUP_NAMES),
            "candidates": candidates,
            "oracle": oracle,
            "checks": checks,
        },
        "details": details,
        "arrays": {"errors": errors, "group_codes": group_codes, "oracle_choice": oracle_choice},
    }
