"""Strictly causal, per-track shift-aware heatmap smoothing."""

import numpy as np


def _argmax_xy(heatmap):
    height, width = heatmap.shape
    row, column = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return np.array([column / width, row / height], dtype=np.float32)


def smooth_track(heatmaps, inout, bboxes, frame_indices, weights):
    """Fuse one track using only current observations and previous state.

    ``weights`` is (bias, head displacement, in/out transition, heatmap shift).
    In/out values are the original single-frame probabilities. A first frame or
    a gap in original frame numbers starts fresh state; no labels are inputs.
    ``signals`` columns are (d_head, trans, d_heat), with zeroes on resets.
    """
    probabilities = np.asarray(inout)
    boxes = np.asarray(bboxes, dtype=np.float32)
    indices = np.asarray(frame_indices)
    weights = np.asarray(weights, dtype=np.float64)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    reset = np.concatenate(([True], np.diff(indices) != 1))
    smoothed = np.empty_like(heatmaps, dtype=np.float32)
    alpha = np.ones(len(heatmaps), dtype=np.float32)
    signals = np.zeros((len(heatmaps), 3), dtype=np.float32)

    for index, current in enumerate(heatmaps):
        if reset[index]:
            smoothed[index] = current
            continue
        signals[index] = (
            np.linalg.norm(centers[index] - centers[index - 1]),
            (probabilities[index] > 0.5) != (probabilities[index - 1] > 0.5),
            np.linalg.norm(_argmax_xy(current) - _argmax_xy(smoothed[index - 1])),
        )
        score = weights[0] + weights[1:] @ signals[index]
        alpha[index] = np.exp(-np.logaddexp(0.0, -score))
        smoothed[index] = alpha[index] * current + (1 - alpha[index]) * smoothed[index - 1]

    return {"heatmap": smoothed, "alpha": alpha, "signals": signals, "reset": reset}
