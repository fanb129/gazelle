import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gazelle.routing.router import RoutingOutput


def coverage_router_loss(
    routing: RoutingOutput,
    target_heatmaps: torch.Tensor,
    *,
    inout: Optional[torch.Tensor] = None,
    coverage_weight: float = 1.0,
    budget_weight: float = 0.05,
    entropy_weight: float = 0.0,
    budget_target: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Train a coarse, high-recall support without duplicating the heatmap head.

    The gaze heatmap is normalized into a target distribution.  The main term
    maximizes how much target mass falls inside the router's soft support.  It
    does not ask the router to reproduce the final gaze heatmap pixel by pixel.
    """
    if target_heatmaps.ndim == 4 and target_heatmaps.shape[1] == 1:
        target_heatmaps = target_heatmaps[:, 0]
    if target_heatmaps.ndim != 3:
        raise ValueError("target_heatmaps must have shape [P, H, W] or [P, 1, H, W]")
    if target_heatmaps.shape[0] != routing.support_probs.shape[0]:
        raise ValueError("target_heatmaps and router supports must describe the same number of people")

    target = F.interpolate(
        target_heatmaps.unsqueeze(1).float(),
        size=routing.support_probs.shape[-2:],
        mode="area",
    ).squeeze(1).clamp_min(0.0)
    target_mass = target.sum(dim=(1, 2))
    target_distribution = target / target_mass.clamp_min(1e-8).view(-1, 1, 1)
    target_peak = target.amax(dim=(1, 2), keepdim=True)
    positive_region = target >= (0.05 * target_peak)

    valid = target_mass > 0
    if inout is not None:
        if inout.numel() != target.shape[0]:
            raise ValueError("inout and target_heatmaps must have the same length")
        valid = valid & inout.to(device=target.device).bool().flatten()

    # support_probs is a spatial softmax distribution. Maximizing its mass in
    # a dilated GT region trains the ordering used by Top-K and cannot be
    # solved by assigning a high independent sigmoid to every patch.
    soft_coverage_per_person = (
        routing.support_probs * positive_region.to(routing.support_probs.dtype)
    ).sum(dim=(1, 2))
    hard_support = routing.image_hard_masks.index_select(0, routing.person_to_image).to(target.dtype)
    hard_coverage_per_person = (hard_support * target_distribution).sum(dim=(1, 2))

    valid_weights = valid.to(routing.support_probs.dtype)
    valid_count = valid_weights.sum().clamp_min(1.0)
    coverage = (
        -torch.log(soft_coverage_per_person.clamp_min(1e-6)) * valid_weights
    ).sum() / valid_count
    soft_coverage = (soft_coverage_per_person * valid_weights).sum() / valid_count
    hard_coverage = (hard_coverage_per_person * valid_weights).sum() / valid_count

    target_ratio = routing.actual_keep_ratio if budget_target is None else float(budget_target)
    union_distribution = routing.image_union_probs.float().flatten(1)
    union_distribution = union_distribution / union_distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)
    union_entropy = -(union_distribution.clamp_min(1e-8).log() * union_distribution).sum(dim=1)
    num_tokens = union_distribution.shape[1]
    effective_support_ratio = union_entropy.exp() / num_tokens
    budget = (effective_support_ratio - target_ratio).square().mean()
    entropy = (union_entropy / max(1.0, math.log(num_tokens))).mean()
    mean_support = effective_support_ratio.mean()

    total = coverage_weight * coverage + budget_weight * budget + entropy_weight * entropy
    return {
        "total": total,
        "coverage": coverage,
        "budget": budget,
        "entropy": entropy,
        "soft_coverage": soft_coverage,
        "hard_coverage": hard_coverage,
        "mean_support": mean_support,
    }
