"""Pure COTB target clustering, score construction, loss, and diagnostics."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from AAAICOTB.geometry import bbox_iou, cluster_targets


@dataclass(frozen=True)
class BindingConfig:
    """Pre-registered definitions shared by training and evaluation."""

    shared_target_radius: float = 0.06
    min_pair_separation: float = 0.10
    target_sigma: float = 0.04
    margin: float = 0.20
    duplicate_bbox_iou: float = 0.80
    eps: float = 1e-6

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def local_log_scores(
    heatmaps: torch.Tensor,
    centers: torch.Tensor,
    sigma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return differentiable query-by-target log activation scores.

    The model emits independent sigmoid heatmaps rather than a spatial
    probability distribution.  We therefore use the Gaussian-weighted local
    mean around each target.  In the diagonal-vs-swapped contrast each query
    appears once on each side, preventing a global heatmap-scale shortcut.
    """

    if heatmaps.ndim != 3:
        raise ValueError(f"heatmaps must have shape [queries,H,W], got {tuple(heatmaps.shape)}")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must have shape [targets,2], got {tuple(centers.shape)}")
    _, height, width = heatmaps.shape
    dtype, device = heatmaps.dtype, heatmaps.device
    xs = (torch.arange(width, dtype=dtype, device=device) + 0.5) / float(width)
    ys = (torch.arange(height, dtype=dtype, device=device) + 0.5) / float(height)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dx = grid_x.unsqueeze(0) - centers[:, 0].view(-1, 1, 1)
    dy = grid_y.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
    kernels = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma**2))
    kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    activation = torch.einsum("qhw,chw->qc", heatmaps, kernels)
    return activation.clamp_min(eps).log()


def compute_frame_binding(
    heatmaps: torch.Tensor,
    targets: torch.Tensor,
    inout: torch.Tensor,
    bboxes: Sequence[Sequence[float]],
    config: BindingConfig,
) -> dict:
    """Compute cluster-aware COTB loss and interpretable per-frame records.

    Shared-target query pairs, too-close target pairs, out-of-frame targets, and
    likely duplicate head tracks are excluded from the contrastive loss.  They
    remain available to the ordinary heatmap/in-out losses.
    """

    if len(heatmaps) != len(targets) or len(targets) != len(inout) or len(inout) != len(bboxes):
        raise ValueError("heatmaps, targets, inout, and bboxes must have equal query counts")
    valid_indices = [
        index
        for index in range(len(inout))
        if bool(inout[index].item()) and bool((targets[index] >= 0).all().item())
    ]
    differentiable_zero = heatmaps.sum() * 0.0
    if len(valid_indices) < 2:
        return {
            "loss": differentiable_zero,
            "pair_count": 0,
            "pairs": [],
            "queries": [],
            "valid_query_count": len(valid_indices),
            "cluster_count": len(valid_indices),
        }

    points = [tuple(map(float, targets[index].detach().cpu().tolist())) for index in valid_indices]
    labels, centers_list = cluster_targets(points, config.shared_target_radius)
    if len(centers_list) < 2:
        return {
            "loss": differentiable_zero,
            "pair_count": 0,
            "pairs": [],
            "queries": [],
            "valid_query_count": len(valid_indices),
            "cluster_count": len(centers_list),
        }

    valid_heatmaps = heatmaps[torch.as_tensor(valid_indices, device=heatmaps.device)]
    centers = torch.tensor(centers_list, dtype=heatmaps.dtype, device=heatmaps.device)
    scores = local_log_scores(valid_heatmaps, centers, config.target_sigma, config.eps)

    query_rows = []
    for local_index, (global_index, own_label) in enumerate(zip(valid_indices, labels)):
        own_score = scores[local_index, own_label]
        wrong_scores = torch.cat((scores[local_index, :own_label], scores[local_index, own_label + 1 :]))
        highest_wrong = wrong_scores.max()
        rank = int((scores[local_index] > own_score).sum().detach().cpu().item()) + 1
        query_rows.append(
            {
                "query_index": global_index,
                "target_cluster": own_label,
                "own_target_rank": rank,
                "ownership_correct": float(rank == 1),
                "ownership_margin": float((own_score - highest_wrong).detach().cpu().item()),
            }
        )

    losses: list[torch.Tensor] = []
    pair_rows: list[dict] = []
    for first, second in itertools.combinations(range(len(valid_indices)), 2):
        first_label, second_label = labels[first], labels[second]
        if first_label == second_label:
            continue
        target_separation = math.dist(centers_list[first_label], centers_list[second_label])
        if target_separation < config.min_pair_separation:
            continue
        first_global, second_global = valid_indices[first], valid_indices[second]
        head_iou = bbox_iou(bboxes[first_global], bboxes[second_global])
        if head_iou >= config.duplicate_bbox_iou:
            continue
        diagonal = scores[first, first_label] + scores[second, second_label]
        swapped = scores[first, second_label] + scores[second, first_label]
        delta = diagonal - swapped
        losses.append(F.relu(config.margin - delta))
        pair_rows.append(
            {
                "query_i": first_global,
                "query_j": second_global,
                "cluster_i": first_label,
                "cluster_j": second_label,
                "target_separation": target_separation,
                "head_bbox_iou": head_iou,
                "diag_margin": float(delta.detach().cpu().item()),
                "diagonal_correct": float(delta.detach().cpu().item() > 0.0),
                "swap_error": float(delta.detach().cpu().item() <= 0.0),
            }
        )

    loss = torch.stack(losses).mean() if losses else differentiable_zero
    return {
        "loss": loss,
        "pair_count": len(losses),
        "pairs": pair_rows,
        "queries": query_rows,
        "valid_query_count": len(valid_indices),
        "cluster_count": len(centers_list),
    }
