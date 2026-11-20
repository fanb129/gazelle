"""D6: cost-supervised hard selection between fixed current and past evidence."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .history_candidates import step_candidates
from .temporal_model import bbox_pool


VERSION = "r4_utility_selector_v1"
MODEL_ID = "utility_selector"
BANK_CANDIDATES = ("current", "current_grid", "ema_075", "grid_ema_075",
                   "transport_previous")
ANCHOR_INDEX, HISTORY_INDEX = 1, 4
SCALAR_NAMES = (
    "raw_inout", "previous_raw_inout", "inout_absolute_change",
    "bbox_center_displacement", "person_cosine", "candidate_region_cosine",
    "history_region_previous_region_cosine", "anchor_entropy", "history_entropy",
    "anchor_peak_fraction", "history_peak_fraction", "candidate_argmax_displacement",
    "anchor_previous_raw_displacement", "history_previous_raw_displacement",
    "raw_grid_displacement", "scene_cosine", "history_available",
)


class UtilitySelector(nn.Module):
    """Use cached label-free observations; the model owns no temporal state."""

    def __init__(self, feature_channels, hidden_dim=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(nn.Linear(6 * feature_channels + len(SCALAR_NAMES), hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observations):
        return self.network(observations).squeeze(-1)


def hard_choices(logits, history_available):
    """Return bank column indices, preserving the anchor on zero-logit ties."""
    use_history = (logits > 0) & history_available.bool()
    return ANCHOR_INDEX + use_history.long() * (HISTORY_INDEX - ANCHOR_INDEX)


def _region(features, heatmap):
    weights = F.adaptive_avg_pool2d(heatmap[None, None].clamp_min(0),
                                   features.shape[-2:])[0, 0]
    weights = weights / weights.sum().clamp_min(1e-12)
    return F.normalize((features * weights[None]).sum(dim=(-2, -1)), dim=0)


def _xy(heatmap):
    position = heatmap.flatten().argmax()
    return torch.stack((position % 64, position // 64)).float() / 64


def _map_stats(heatmap):
    mass = heatmap.clamp_min(0).flatten()
    mass = mass / mass.sum().clamp_min(1e-12)
    entropy = -(mass * mass.clamp_min(1e-12).log()).sum() / math.log(64 * 64)
    return entropy, mass.max()


@torch.no_grad()
def step_selector_inputs(raw_heatmap, features, raw_inout, bbox, frame_index, state=None):
    """Generate fixed maps and six frozen tokens plus 17 scalars for one frame.

    Only raw observations update state. The caller passes None at a new person;
    D5 metadata handling also clears history after an original-frame gap.
    Neither labels nor selector outputs are inputs to this operation.
    """
    raw_heatmap, features = raw_heatmap.detach().float(), features.detach().float()
    raw_inout, bbox = raw_inout.detach().float(), bbox.detach().float()
    candidate_state = None if state is None else state["candidates"]
    candidates, next_candidates, available = step_candidates(
        raw_heatmap, features, frame_index, candidate_state)
    anchor, history = candidates["current_grid"], candidates["transport_previous"]
    grid_ema = .75 * anchor + .25 * state["grid_ema"] if available else anchor
    maps = dict(zip(BANK_CANDIDATES,
                    (candidates["current"], anchor, candidates["ema_075"], grid_ema, history)))

    scene = F.normalize(features.mean(dim=(-2, -1)), dim=0)
    person = F.normalize(bbox_pool(features[None], bbox[None])[0], dim=0)
    anchor_region, history_region = _region(features, anchor), _region(features, history)
    raw_region = _region(features, raw_heatmap)
    zero = raw_inout.new_zeros(())
    previous_person = state["previous_person"] if available else torch.zeros_like(person)
    previous_region = state["previous_region"] if available else torch.zeros_like(raw_region)
    anchor_xy, history_xy = _xy(anchor), _xy(history)
    anchor_entropy, anchor_peak = _map_stats(anchor)
    history_entropy, history_peak = _map_stats(history)
    if available:
        previous_inout = state["previous_inout"]
        score_change = (raw_inout - previous_inout).abs()
        center = (bbox[:2] + bbox[2:]) / 2
        old_box = state["previous_bbox"]
        previous_center = (old_box[:2] + old_box[2:]) / 2
        bbox_displacement = torch.linalg.vector_norm(center - previous_center)
        person_cosine = (person * previous_person).sum()
        history_previous_cosine = (history_region * previous_region).sum()
        previous_xy = _xy(candidate_state["previous_heatmap"])
        anchor_previous_displacement = torch.linalg.vector_norm(anchor_xy - previous_xy)
        history_previous_displacement = torch.linalg.vector_norm(history_xy - previous_xy)
        scene_cosine = (scene * state["previous_scene"]).sum()
    else:
        previous_inout = score_change = bbox_displacement = person_cosine = zero
        history_previous_cosine = anchor_previous_displacement = history_previous_displacement = scene_cosine = zero
    scalars = torch.stack((raw_inout, previous_inout, score_change, bbox_displacement,
        person_cosine, (anchor_region * history_region).sum(), history_previous_cosine,
        anchor_entropy, history_entropy, anchor_peak, history_peak,
        torch.linalg.vector_norm(anchor_xy - history_xy), anchor_previous_displacement,
        history_previous_displacement, torch.linalg.vector_norm(_xy(raw_heatmap) - anchor_xy),
        scene_cosine, raw_inout.new_tensor(float(available))))
    observation = torch.cat((scene, person, previous_person, anchor_region,
                             history_region, previous_region, scalars))
    next_state = {"candidates": next_candidates, "grid_ema": grid_ema.clone(),
                  "previous_person": person.clone(), "previous_scene": scene.clone(),
                  "previous_region": raw_region.clone(), "previous_inout": raw_inout.clone(),
                  "previous_bbox": bbox.clone()}
    return maps, observation, available, next_state


def cost_weighted_bce(logits, anchor_xy, history_xy, gt_xy, in_frame, history_available):
    """Supervise expected point-error advantage without feeding GT into memory."""
    anchor_error = torch.linalg.vector_norm(anchor_xy.double() - gt_xy.double(), dim=-1)
    history_error = torch.linalg.vector_norm(history_xy.double() - gt_xy.double(), dim=-1)
    advantage = anchor_error - history_error
    supervised = in_frame.bool() & history_available.bool()
    targets = (advantage > 0).to(logits.dtype)
    weights = advantage.abs() * supervised
    weighted_sum = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum()
    weight_sum = weights.sum()
    return {"loss": weighted_sum / weight_sum.clamp_min(1e-12),
            "weighted_bce_sum": weighted_sum, "weight_sum": weight_sum,
            "supervised_frames": supervised.sum(),
            "prefer_history_frames": (supervised & (advantage > 0)).sum(),
            "tie_frames": (supervised & (advantage == 0)).sum()}
