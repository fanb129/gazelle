"""Offline coordinate/score cross-combinations and exact stability decomposition.

These combinations reuse saved outputs; they are not retrained ablations or
interventions on the recurrent model. Labels only define offline scoring.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from .metrics import TemporalConfig, _distribution, evaluate_temporal
from .state_metrics import evaluate_state_temporal


VARIANTS = {
    "raw_xy_raw_score": ("raw", "raw"),
    "refined_xy_raw_score": ("refined", "raw"),
    "raw_xy_refined_score": ("raw", "refined"),
    "refined_xy_refined_score": ("refined", "refined"),
}
FAILURE_MASK_BITS = {
    "1": "coordinate_unstable",
    "2": "localization_inaccurate",
    "4": "predicted_out_in_window",
}


def _window_conditions(tracks, coordinates, scores, config):
    """Return three pass conditions on the exact T4 eligible-window population."""
    conditions = []
    for track in tracks:
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        labels = np.asarray([frame["inout"] == 1 for frame in frames])
        gt = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames])
        xy = np.asarray(coordinates[track["start"]:track["stop"]], dtype=np.float64)
        score = np.asarray(scores[track["start"]:track["stop"]])
        contiguous = np.diff(indices) == 1
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        errors = np.linalg.norm(xy - gt, axis=1)
        for start in range(len(frames) - config.hold_frames + 1):
            stop = start + config.hold_frames
            if labels[start:stop].all() and contiguous[start:stop - 1].all():
                conditions.append(((steps[start:stop - 1] <= config.stable_threshold).all(),
                                   (errors[start:stop] <= config.response_radius).all(),
                                   (score[start:stop] > 0.5).all()))
    return np.asarray(conditions, dtype=bool).reshape(-1, 3)


def _effect(count, denominator):
    return {"window_count": count, "ratio": count / denominator if denominator else None}


def _stability_attribution(tracks, raw_xy, raw_scores, refined_xy, refined_scores, config):
    raw = _window_conditions(tracks, raw_xy, raw_scores, config)
    refined = _window_conditions(tracks, refined_xy, refined_scores, config)
    raw_coordinate = raw[:, :2].all(axis=1)
    refined_coordinate = refined[:, :2].all(axis=1)
    passes = {
        "M00": raw_coordinate & raw[:, 2],
        "M10": refined_coordinate & raw[:, 2],
        "M01": raw_coordinate & refined[:, 2],
        "M11": refined_coordinate & refined[:, 2],
    }
    counts = {key: int(value.sum()) for key, value in passes.items()}
    n = len(raw)
    m00, m10, m01, m11 = (counts[key] for key in ("M00", "M10", "M01", "M11"))
    lost = passes["M00"] & ~passes["M11"]
    gained = ~passes["M00"] & passes["M11"]

    def failure_masks(conditions, selected):
        masks = (~conditions).astype(np.int64) @ np.asarray([1, 2, 4])
        return {str(mask): int(np.sum(selected & (masks == mask))) for mask in range(1, 8)}

    return {
        "interpretation": "saved_output_numerical_decomposition; not_retrained_ablation_or_mechanism_causality",
        "eligible_windows": n,
        "accurate_stable_counts": counts,
        "accurate_stable_ratios": {key: value / n if n else None for key, value in counts.items()},
        "matrix_definition": {"M00": "raw_xy_raw_score", "M10": "refined_xy_raw_score",
                              "M01": "raw_xy_refined_score", "M11": "refined_xy_refined_score"},
        "total_delta": _effect(m11 - m00, n),
        "coordinate_effect_at_raw_score": _effect(m10 - m00, n),
        "score_effect_at_raw_xy": _effect(m01 - m00, n),
        "interaction": _effect(m11 - m10 - m01 + m00, n),
        "shapley": {
            "coordinate": _effect(((m10 - m00) + (m11 - m01)) / 2, n),
            "score": _effect(((m01 - m00) + (m11 - m10)) / 2, n),
            "interpretation": "symmetric_average_of_the_two_output_replacement_orders; fractional_window_count_equivalents",
        },
        "window_transitions": {
            "lost": int(lost.sum()), "gained": int(gained.sum()),
            "both_accurate": int((passes["M00"] & passes["M11"]).sum()),
            "neither_accurate": int((~passes["M00"] & ~passes["M11"]).sum()),
            "failure_mask_bits": FAILURE_MASK_BITS,
            "lost_refined_failure_masks": failure_masks(refined, lost),
            "gained_raw_failure_masks": failure_masks(raw, gained),
            "failure_mask_interpretation": "mutually_exclusive_bit_patterns; multiple_failed_conditions_can_overlap_in_one_window",
        },
    }


def _prediction_changes(positions, labels, raw_xy, raw_scores, refined_xy, refined_scores):
    raw_xy, refined_xy = np.asarray(raw_xy)[positions], np.asarray(refined_xy)[positions]
    raw_scores = np.asarray(raw_scores)[positions].astype(np.float64)
    refined_scores = np.asarray(refined_scores)[positions].astype(np.float64)
    displacement = np.linalg.norm(refined_xy.astype(np.float64) - raw_xy, axis=1)
    score_delta = refined_scores - raw_scores
    raw_state, refined_state = raw_scores > 0.5, refined_scores > 0.5
    changed = np.any(raw_xy != refined_xy, axis=1)
    results = {}
    for name, selected in (("all", np.ones(len(labels), dtype=bool)),
                           ("gt_in_frame", labels == 1), ("gt_out_of_frame", labels == 0)):
        count = int(selected.sum())
        results[name] = {
            "person_frames": count,
            "coordinates_changed": int((selected & changed).sum()),
            "coordinates_changed_ratio": float(changed[selected].mean()) if count else None,
            "coordinate_displacement": _distribution(displacement[selected]),
            "score_delta": _distribution(score_delta[selected]),
            "absolute_score_delta": _distribution(np.abs(score_delta[selected])),
            "crossed_to_in_frame": int((selected & ~raw_state & refined_state).sum()),
            "crossed_to_out_of_frame": int((selected & raw_state & ~refined_state).sum()),
            "state_errors_corrected": int((selected & (raw_state != labels) & (refined_state == labels)).sum()),
            "state_errors_introduced": int((selected & (raw_state == labels) & (refined_state != labels)).sum()),
        }
    return results


def diagnose_prediction_pair(tracks, raw_xy, raw_scores, refined_xy, refined_scores, config=None):
    """Score four frozen output combinations and decompose accurate stability.

    ``tracks`` uses the supplied arrays' global slices. Neither arrays nor
    metadata are mutated; no visual encoder, checkpoint or recurrent forward
    is needed. AUC is omitted because coordinates do not specify a heatmap.
    """
    config = config or TemporalConfig()
    positions = np.asarray([position for track in tracks
                            for position in range(track["start"], track["stop"])], dtype=np.int64)
    frames = [frame for track in tracks for frame in track["frames"]]
    labels = np.asarray([frame["inout"] for frame in frames])
    gt = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames]).reshape(-1, 2)
    xy_sources = {"raw": raw_xy, "refined": refined_xy}
    score_sources = {"raw": raw_scores, "refined": refined_scores}
    localization = {
        name: float(np.linalg.norm(np.asarray(xy)[positions][labels == 1].astype(np.float64)
                                   - gt[labels == 1], axis=1).mean()) if np.any(labels == 1) else None
        for name, xy in xy_sources.items()
    }
    ap = {name: float(average_precision_score(labels, np.asarray(scores)[positions]))
          if len(labels) else None for name, scores in score_sources.items()}
    state_results = {name: evaluate_state_temporal(tracks, scores, config)
                     for name, scores in score_sources.items()}
    variants, details = {}, {}
    for name, (coordinate_source, state_source) in VARIANTS.items():
        temporal = evaluate_temporal(tracks, xy_sources[coordinate_source], score_sources[state_source], config)
        state = state_results[state_source]
        variants[name] = {
            "coordinate_source": coordinate_source, "state_source": state_source,
            "standard": {"l2": localization[coordinate_source], "inout_ap": ap[state_source],
                         "person_frames": len(labels), "localization_frames": int((labels == 1).sum())},
            "temporal": temporal["summary"], "state": state["summary"],
        }
        details[name] = {"joint": {key: temporal[key] for key in ("tracks", "events")},
                         "state": {key: state[key] for key in ("tracks", "events")}}
    return {
        "variants": variants,
        "attribution": _stability_attribution(tracks, raw_xy, raw_scores, refined_xy, refined_scores, config),
        "prediction_changes": _prediction_changes(positions, labels, raw_xy, raw_scores,
                                                   refined_xy, refined_scores),
        "details": details,
    }
