"""Offline summaries of cached history controls, correspondence and corrections."""

import numpy as np

from .history_diagnostics import CONTROL_COLUMNS
from .metrics import TemporalConfig, evaluate_temporal


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, **{key: None for key in
                ("mean", "std", "min", "max", "p10", "p50", "p90")}}
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "max": float(values.max()),
            **{f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90)}}


def _gate_fractions(values):
    return {"count": len(values), "low_threshold": 0.1, "high_threshold": 0.9,
            "low_fraction": float(np.mean(values <= .1)) if len(values) else None,
            "high_fraction": float(np.mean(values >= .9)) if len(values) else None}


def _pearson(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"count": len(first), "pearson": None}
    first, second = first - np.mean(first), second - np.mean(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return {"count": len(first),
            "pearson": float(np.clip(np.dot(first, second) / denominator, -1, 1))
                       if denominator else None}


def _aligned_arrays(tracks, diagnostic_index, diagnostic_arrays, source_frames):
    columns = diagnostic_index["control_columns"]
    if len(columns) != len(CONTROL_COLUMNS) or set(columns) != set(CONTROL_COLUMNS):
        raise ValueError("History diagnostic control columns differ from the declared controls")
    expected = np.concatenate([np.arange(track["start"], track["stop"]) for track in tracks])
    source = np.asarray(diagnostic_arrays["source_global_indices"])
    controls = np.asarray(diagnostic_arrays["controls"])
    correspondence = np.asarray(diagnostic_arrays["correspondence_xy"])
    if (diagnostic_index["source_person_frames"] != source_frames
            or diagnostic_index["person_frames"] != len(expected)
            or source.shape != (len(expected),)
            or controls.shape != (len(expected), len(columns))
            or correspondence.shape != (len(expected), 2)
            or not np.array_equal(np.sort(source), np.sort(expected))
            or len(np.unique(source)) != len(source)):
        raise ValueError("History diagnostics do not cover each supplied source frame exactly once")
    recorded = {track["track_id"]: track for track in diagnostic_index["tracks"]}
    if len(recorded) != len(tracks) or len(recorded) != len(diagnostic_index["tracks"]):
        raise ValueError("History diagnostic track identities differ from source tracks")
    for track in tracks:
        item = recorded.get(track["track_id"])
        if (item is None or item["source_global_start"] != track["start"]
                or item["source_global_stop"] != track["stop"]
                or item["frame_indices"] != [frame["frame_index"] for frame in track["frames"]]):
            raise ValueError("History diagnostic track/frame mapping differs from source tracks")
    if not np.isfinite(controls).all() or not np.isfinite(correspondence).all():
        raise ValueError("History diagnostic arrays contain non-finite values")
    order = np.argsort(source)
    rows = order[np.searchsorted(source[order], expected)]
    return {name: controls[rows, columns.index(name)].astype(np.float64)
            for name in CONTROL_COLUMNS}, correspondence[rows], expected


def _offline_groups(tracks, raw_xy, raw_scores, config, event_radius):
    labels, gt, stable = [], [], []
    offsets, segments, offset = {}, {}, 0
    for track in tracks:
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        inside = np.asarray([frame["inout"] == 1 for frame in frames])
        points = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames])
        contiguous = np.diff(indices) == 1
        endpoint = np.zeros(len(frames), dtype=bool)
        endpoint[1:] = contiguous & inside[:-1] & inside[1:] & (
            np.linalg.norm(np.diff(points, axis=0), axis=1) <= config.stable_threshold)
        boundaries = np.r_[0, np.flatnonzero(~contiguous) + 1, len(frames)]
        segments[track["track_id"]] = (indices, boundaries)
        offsets[track["track_id"]] = offset
        offset += len(frames)
        labels.extend(inside)
        gt.extend(points)
        stable.extend(endpoint)
    labels, gt = np.asarray(labels, dtype=bool), np.asarray(gt)
    masks = {"all": np.ones(len(labels), dtype=bool), "GT_inframe": labels,
             "GT_out_of_frame": ~labels,
             "GT_stable_inframe_pair_endpoints": np.asarray(stable, dtype=bool)}
    events = evaluate_temporal(tracks, raw_xy, raw_scores, config)["events"]
    counts = {}
    for kind in ("enter_frame", "leave_frame", "gaze_shift"):
        mask = np.zeros(len(labels), dtype=bool)
        selected = [event for event in events if event["type"] == kind]
        counts[kind] = len(selected)
        for event in selected:
            indices, boundaries = segments[event["track_id"]]
            position = int(np.searchsorted(indices, event["event_frame_index"]))
            segment = np.searchsorted(boundaries, position, side="right") - 1
            first, stop = boundaries[segment:segment + 2]
            local = np.arange(first, stop)
            local = local[np.abs(indices[local] - event["event_frame_index"]) <= event_radius]
            mask[offsets[event["track_id"]] + local] = True
        masks["near_" + kind] = mask
    return labels, gt, masks, counts


def summarize_history_controls(tracks, raw_xy, raw_scores, refined_xy, refined_scores,
                               diagnostic_index, diagnostic_arrays, config=None, event_radius=3):
    """Summarize observable controls using offline, possibly overlapping GT groups.

    Correspondence distances are reported only when a past state exists. Gaze
    point proximity and gate correlations do not establish target identity or
    the causal contribution of the learned historical mechanism.
    """
    config = TemporalConfig() if config is None else config
    if event_radius < 0:
        raise ValueError("Event neighborhood radius must be nonnegative")
    controls, correspondence, source = _aligned_arrays(
        tracks, diagnostic_index, diagnostic_arrays, len(raw_scores))
    labels, gt, groups, events = _offline_groups(tracks, raw_xy, raw_scores, config, event_radius)
    available = controls["history_available"] > .5
    groups["history_available"] = available
    groups["no_history"] = ~available
    raw_xy, refined_xy = np.asarray(raw_xy)[source], np.asarray(refined_xy)[source]
    raw_scores, refined_scores = np.asarray(raw_scores)[source], np.asarray(refined_scores)[source]
    correction = {"coordinate_delta_l2": np.linalg.norm(refined_xy - raw_xy, axis=1),
                  "score_signed_delta": refined_scores - raw_scores,
                  "score_absolute_delta": np.abs(refined_scores - raw_scores)}
    summary = {}
    for name, mask in groups.items():
        historical, historical_inside = mask & available, mask & available & labels
        summary[name] = {
            "frames": int(mask.sum()), "history_available_frames": int(historical.sum()),
            "controls": {key: _distribution(values[mask]) for key, values in controls.items()},
            "gate_fractions": {key: _gate_fractions(controls[key][mask])
                               for key in ("history_validity", "refresh")},
            "validity_on_available_history": {
                "distribution": _distribution(controls["history_validity"][historical]),
                "gate_fractions": _gate_fractions(controls["history_validity"][historical])},
            "correspondence_on_available_history": {
                "to_raw_prediction": _distribution(np.linalg.norm(correspondence[historical] - raw_xy[historical], axis=1)),
                "to_refined_prediction": _distribution(np.linalg.norm(correspondence[historical] - refined_xy[historical], axis=1)),
                "to_GT_inframe": _distribution(np.linalg.norm(correspondence[historical_inside] - gt[historical_inside], axis=1))},
            "prediction_corrections": {key: _distribution(values[mask]) for key, values in correction.items()},
            "correction_associations_on_available_history": {
                gate: {key: _pearson(controls[gate][historical], values[historical])
                       for key, values in correction.items()}
                for gate in ("history_validity", "refresh")},
        }
    return {
        "person_frames": len(source), "event_radius_original_frames": event_radius,
        "proxy_event_counts": events, "groups": summary,
        "grouping": {
            "overlap": "Groups may overlap; near-event groups union their frame neighborhoods.",
            "near_events": "GT_proxy events; original-frame +/- radius within the same contiguous track segment.",
            "GT_stable_inframe_pair_endpoints": "Current endpoint of an adjacent GT-inframe pair with GT displacement <= stable_threshold.",
            "history_available": "Past-state existence only; not a GT validity label.",
            "correspondence": "Only available-history rows; GT distance only on GT-inframe rows; not target-identity matching accuracy.",
            "correlations": "Pearson associations of cached outputs; constant/insufficient values yield null, not causal mechanism evidence.",
            "label_usage": "GT and future-side event neighborhoods are offline scoring groups only; no prediction is recomputed."},
    }
