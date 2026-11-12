"""Shared frame-weighted VAT metrics for single-frame and STAG predictions.

This module evaluates predictions only. Ground-truth gaze and in/out labels are
never used by the temporal gate. The AUC rectangle matches gazelle.utils.vat_auc,
including its exclusive right/bottom slice capped at 63.
"""

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
