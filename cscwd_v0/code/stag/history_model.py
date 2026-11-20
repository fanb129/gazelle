"""Causal visual correspondence with learned validity of predicted history."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from stag.temporal_model import CausalJointRefiner, bbox_pool


def normalized_map_weights(heatmap, spatial_size):
    """Read a predicted region using nonnegative, unit-sum spatial weights.

    An all-zero map reads a zero descriptor. These weights are a region readout,
    not calibrated gaze probabilities.
    """
    weights = F.adaptive_avg_pool2d(heatmap[:, None].clamp_min(0), spatial_size).squeeze(1)
    return weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def target_correspondence(features, target, temperature=0.1):
    """Locate a past predicted descriptor in this frame's frozen feature grid."""
    cosine = (F.normalize(features, dim=1)
              * F.normalize(target, dim=1)[:, :, None, None]).sum(dim=1)
    distribution = (cosine.flatten(1) / temperature).softmax(dim=1).reshape_as(cosine)
    return distribution, cosine


class HistoryValidityRefiner(CausalJointRefiner):
    """R3: keep current evidence and selectively transport predicted history.

    ``step`` has the R2 interface. ``step_with_diagnostics`` also returns tensors
    for per-frame inspection. No labels, object IDs or future features enter
    either forward path. Templates stay in the frozen input feature space.
    """

    def __init__(self, feature_channels, hidden_dim=128, history_mode="learned", temperature=0.1):
        super().__init__(feature_channels, hidden_dim)
        if history_mode not in ("learned", "always_keep"):
            raise ValueError("history_mode must be learned or always_keep")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.history_mode = history_mode
        self.temperature = temperature
        self.history_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim + 6, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2))
        # Start at the same raw prediction as R2; learn spatial support jointly.
        self.correspondence_gain = nn.Parameter(torch.zeros(()))

    def step(self, features, raw_heatmap, raw_inout, bboxes, state=None, reset=None):
        return self.step_with_diagnostics(
            features, raw_heatmap, raw_inout, bboxes, state, reset)[:3]

    def step_with_diagnostics(self, features, raw_heatmap, raw_inout, bboxes,
                              state=None, reset=None):
        features, raw_heatmap = features.float(), raw_heatmap.float()
        raw_inout, bboxes = raw_inout.float(), bboxes.float()
        batch, channels, height, width = features.shape
        if state is None:
            state = {"hidden": features.new_zeros(batch, self.hidden_dim),
                     "heatmap": torch.zeros_like(raw_heatmap),
                     "inout": torch.zeros_like(raw_inout),
                     "target": features.new_zeros(batch, channels),
                     "person": features.new_zeros(batch, channels),
                     "available": torch.zeros(batch, dtype=torch.bool, device=features.device)}
        if reset is not None:
            state = {name: value.masked_fill(
                reset.bool().reshape(batch, *([1] * (value.ndim - 1))), 0)
                     for name, value in state.items()}

        available = state["available"].to(features.dtype)
        spatial = F.gelu(self.feature_projection(features))
        global_token = spatial.mean(dim=(-2, -1))
        person_token = bbox_pool(spatial, bboxes)
        person = bbox_pool(features, bboxes)
        current_weights = normalized_map_weights(raw_heatmap, (height, width))
        current_token = (spatial * current_weights[:, None]).sum(dim=(-2, -1))

        correspondence, cosine = target_correspondence(features, state["target"], self.temperature)
        matched_token = (spatial * correspondence[:, None]).sum(dim=(-2, -1))
        peak = cosine.flatten(1).amax(dim=1)
        entropy = -(correspondence * correspondence.clamp_min(1e-12).log()).sum(dim=(-2, -1))
        entropy = entropy / math.log(max(2, height * width))
        agreement = F.cosine_similarity(correspondence.flatten(1), current_weights.flatten(1), dim=1)
        person_change = 1 - F.cosine_similarity(person, state["person"], dim=1)
        summaries = torch.stack((peak, entropy, agreement, person_change, raw_inout, state["inout"]), dim=1)
        controls = self.history_gate(torch.cat((person_token, current_token, matched_token, summaries), dim=1)).sigmoid()
        validity = controls[:, 0] if self.history_mode == "learned" else torch.ones_like(available)
        validity = validity * available
        refresh = controls[:, 1]

        # Every historical predictor path is gated; current tokens remain intact.
        evidence = cosine - cosine.mean(dim=(-2, -1), keepdim=True)
        supported_map = evidence * validity[:, None, None]
        current_map = F.adaptive_avg_pool2d(raw_heatmap[:, None], (8, 8)).flatten(1)
        aligned_map = F.adaptive_avg_pool2d(supported_map[:, None], (8, 8)).flatten(1)
        observation = torch.cat((global_token, person_token, current_token, current_map, aligned_map,
                                 bboxes, raw_inout[:, None], (validity * state["inout"])[:, None]), dim=1)
        hidden_seed = validity[:, None] * (state["hidden"] + matched_token)
        hidden = self.gru(observation, hidden_seed)

        delta = self.heatmap_residual(spatial * hidden[:, :, None, None]).squeeze(1)
        delta = delta + self.correspondence_gain * supported_map
        delta = F.interpolate(delta[:, None], size=raw_heatmap.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(1)
        heatmap = torch.sigmoid(torch.logit(raw_heatmap.clamp(1e-6, 1 - 1e-6)) + delta)
        inout = torch.sigmoid(torch.logit(raw_inout.clamp(1e-6, 1 - 1e-6))
                              + self.inout_residual(hidden).squeeze(1))

        predicted_weights = normalized_map_weights(heatmap, (height, width))
        predicted_target = (features * predicted_weights[:, None]).sum(dim=(-2, -1))
        retention = validity * (1 - refresh)
        target = retention[:, None] * state["target"] + (1 - retention[:, None]) * predicted_target
        next_state = {"hidden": hidden, "heatmap": heatmap, "inout": inout,
                      "target": target, "person": person,
                      "available": torch.ones_like(state["available"])}

        position = correspondence.flatten(1).argmax(dim=1)
        xy = torch.stack(((position % width + 0.5) / width,
                          (position // width + 0.5) / height), dim=1)
        diagnostics = {"history_validity": validity, "refresh": refresh,
                       "write_strength": 1 - retention,
                       "correspondence_xy": xy * available[:, None],
                       "correspondence_map": correspondence * available[:, None, None],
                       "match_peak": peak, "match_entropy": entropy,
                       "agreement": agreement, "person_change": person_change,
                       "history_available": state["available"]}
        return heatmap, inout, next_state, diagnostics
