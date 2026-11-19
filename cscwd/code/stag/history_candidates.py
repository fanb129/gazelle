"""Fixed, label-free causal spatial candidates for the D5 train-only analysis."""

import numpy as np
import torch
from torch.nn import functional as F


VERSION = "r4_history_utility_v1"
CANDIDATE_NAMES = ("current", "current_grid", "previous_raw", "ema_075",
                   "transport_previous")


def _resize_grid(grid):
    return F.interpolate(grid[None, None], size=(64, 64), mode="bilinear",
                         align_corners=False)[0, 0]


def _transport(previous_heatmap, previous_features, current_features):
    """Distribute each previous patch's attention mass over current patches."""
    mass = F.adaptive_avg_pool2d(previous_heatmap[None, None].clamp_min(0),
                                previous_features.shape[-2:]).flatten()
    mass = mass / mass.sum().clamp_min(1e-12)
    previous = F.normalize(previous_features.flatten(1), dim=0)
    current = F.normalize(current_features.flatten(1), dim=0)
    correspondence = (previous.T @ current / 0.1).softmax(dim=1)
    transported = (mass @ correspondence).reshape(current_features.shape[-2:])
    transported = _resize_grid(transported)
    return transported / transported.sum().clamp_min(1e-12)


@torch.no_grad()
def step_candidates(current_heatmap, current_features, frame_index, state=None):
    """Return five FP32 maps and raw-only state for one person at one frame.

    The caller starts each track with ``state=None``. A frame-number gap also
    resets all history. Neither labels nor state scores enter this function;
    transported maps never become future attention memory.
    """
    current = current_heatmap.detach().float()
    features = current_features.detach().float()
    available = state is not None and frame_index - state["frame_index"] == 1
    current_grid = _resize_grid(F.adaptive_avg_pool2d(
        current[None, None], features.shape[-2:])[0, 0])
    if available:
        previous = state["previous_heatmap"]
        ema = 0.75 * current + 0.25 * state["ema"]
        transport = _transport(previous, state["previous_features"], features)
    else:
        previous, ema, transport = current, current, current
    maps = dict(zip(CANDIDATE_NAMES,
                    (current, current_grid, previous, ema, transport)))
    next_state = {"previous_heatmap": current.clone(),
                  "previous_features": features.clone(),
                  "ema": ema.clone(), "frame_index": int(frame_index)}
    return maps, next_state, available


def maps_to_coordinates(maps):
    """Use the original 64-grid first-flat-argmax coordinate convention."""
    positions = torch.stack([maps[name].flatten() for name in CANDIDATE_NAMES]).argmax(dim=1)
    coordinates = torch.stack((positions % 64, positions // 64), dim=1).float() / 64
    return coordinates.detach().cpu().numpy().astype(np.float32, copy=False)
