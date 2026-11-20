"""Ordinary causal GRU refinement of frozen gaze heatmaps and in/out scores.

The model owns only new projections and refinement heads. It receives cached
current-frame features and predictions; gaze labels are used by ``joint_loss``
only. The recurrent state contains its own previous outputs, never GT history.
"""

import torch
from torch import nn
from torch.nn import functional as F


def bbox_pool(features, bboxes):
    """Pool by fractional bbox/patch overlap, retaining sub-patch head boxes."""
    height, width = features.shape[-2:]
    xs = torch.linspace(0, 1, width + 1, device=features.device, dtype=features.dtype)
    ys = torch.linspace(0, 1, height + 1, device=features.device, dtype=features.dtype)
    x_overlap = (torch.minimum(bboxes[:, 2:3], xs[1:])
                 - torch.maximum(bboxes[:, 0:1], xs[:-1])).clamp_min(0)
    y_overlap = (torch.minimum(bboxes[:, 3:4], ys[1:])
                 - torch.maximum(bboxes[:, 1:2], ys[:-1])).clamp_min(0)
    weights = y_overlap[:, :, None] * x_overlap[:, None, :]
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (features * weights[:, None]).sum(dim=(-2, -1))


def detach_state(state):
    """Keep predicted history while truncating gradients between chunks."""
    return None if state is None else {name: value.detach() for name, value in state.items()}


class CausalJointRefiner(nn.Module):
    """A per-person GRU baseline without explicit target correspondence gates."""

    def __init__(self, feature_channels, hidden_dim=128):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.feature_projection = nn.Conv2d(feature_channels, hidden_dim, kernel_size=1)
        self.gru = nn.GRUCell(3 * hidden_dim + 2 * 8 * 8 + 4 + 2, hidden_dim)
        self.heatmap_residual = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.inout_residual = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.heatmap_residual.weight)
        nn.init.zeros_(self.heatmap_residual.bias)
        nn.init.zeros_(self.inout_residual.weight)
        nn.init.zeros_(self.inout_residual.bias)

    def step(self, features, raw_heatmap, raw_inout, bboxes, state=None, reset=None):
        """Process only this frame and the supplied same-person past state.

        ``reset[B]`` is determined by track starts/gaps, never by labels. Features
        have shape [B,C,H,W], maps [B,64,64], scores [B], and boxes [B,4].
        """
        features, raw_heatmap = features.float(), raw_heatmap.float()
        raw_inout, bboxes = raw_inout.float(), bboxes.float()
        batch = features.shape[0]
        if state is None:
            state = {"hidden": features.new_zeros(batch, self.hidden_dim),
                     "heatmap": torch.zeros_like(raw_heatmap),
                     "inout": torch.zeros_like(raw_inout)}
        if reset is not None:
            reset = reset.bool()
            state = {name: value.masked_fill(
                reset.reshape(batch, *([1] * (value.ndim - 1))), 0)
                     for name, value in state.items()}

        spatial = F.gelu(self.feature_projection(features))
        global_token = spatial.mean(dim=(-2, -1))
        person_token = bbox_pool(spatial, bboxes)
        weights = F.adaptive_avg_pool2d(raw_heatmap[:, None], spatial.shape[-2:]).clamp_min(0)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        target_token = (spatial * weights).sum(dim=(-2, -1))
        current_map = F.adaptive_avg_pool2d(raw_heatmap[:, None], (8, 8)).flatten(1)
        previous_map = F.adaptive_avg_pool2d(state["heatmap"][:, None], (8, 8)).flatten(1)
        observation = torch.cat((global_token, person_token, target_token,
                                 current_map, previous_map, bboxes,
                                 raw_inout[:, None], state["inout"][:, None]), dim=1)
        hidden = self.gru(observation, state["hidden"])

        delta = self.heatmap_residual(spatial * hidden[:, :, None, None])
        delta = F.interpolate(delta, size=raw_heatmap.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(1)
        heatmap = torch.sigmoid(torch.logit(raw_heatmap.clamp(1e-6, 1 - 1e-6)) + delta)
        score_delta = self.inout_residual(hidden).squeeze(1)
        inout = torch.sigmoid(torch.logit(raw_inout.clamp(1e-6, 1 - 1e-6)) + score_delta)
        return heatmap, inout, {"hidden": hidden, "heatmap": heatmap, "inout": inout}


def joint_loss(pred_heatmap, pred_inout, target_heatmap, target_inout, valid, lambda_state):
    """Use valid GT-inframe map targets and all valid-frame state targets.

    Leading dimensions may represent [batch] or [batch,time]. Empty selections
    contribute differentiable zero losses, including fully padded steps.
    """
    valid = valid.bool()
    inside = valid & (target_inout > 0.5)
    valid_count, inside_count = valid.sum(), inside.sum()
    map_pixels = pred_heatmap.shape[-2] * pred_heatmap.shape[-1]
    heatmap = F.mse_loss(pred_heatmap[inside], target_heatmap[inside], reduction="sum")
    heatmap = heatmap / (inside_count.clamp_min(1) * map_pixels)
    state = F.binary_cross_entropy(pred_inout[valid], target_inout[valid].float(), reduction="sum")
    state = state / valid_count.clamp_min(1)
    return {"total": heatmap + lambda_state * state, "heatmap": heatmap, "state": state,
            "valid_frames": valid_count, "inframe_frames": inside_count}
