"""Fixed-alpha, strictly causal heatmap EMA for the T5 smoothing control."""

import numpy as np


def smooth_ema_track(heatmaps, frame_indices, alpha):
    """Smooth one track, resetting at its start and original frame-number gaps.

    Only heatmaps and frame numbers are inputs. The output uses the same float32
    arithmetic and reset policy as STAG, with alpha fixed on non-reset frames.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("EMA alpha must be in [0, 1]")
    reset = np.concatenate(([True], np.diff(frame_indices) != 1))
    coefficients = np.full(len(heatmaps), alpha, dtype=np.float32)
    coefficients[reset] = 1
    smoothed = np.empty_like(heatmaps, dtype=np.float32)
    for position, current in enumerate(heatmaps):
        if reset[position]:
            smoothed[position] = current
        else:
            smoothed[position] = (coefficients[position] * current
                                  + (1 - coefficients[position]) * smoothed[position - 1])
    return {"heatmap": smoothed, "alpha": coefficients, "reset": reset}
