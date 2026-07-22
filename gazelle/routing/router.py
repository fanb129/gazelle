from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class RoutingOutput:
    """Differentiable support predictions and discrete per-image routes."""

    support_logits: torch.Tensor
    support_probs: torch.Tensor
    person_to_image: torch.Tensor
    person_head_masks: torch.Tensor
    image_union_probs: torch.Tensor
    image_hard_masks: torch.Tensor
    image_keep_indices: torch.Tensor
    actual_keep_ratio: float


def _flatten_bboxes(bboxes: Sequence[Sequence]) -> tuple[List, List[int]]:
    flat_bboxes = []
    person_to_image = []
    for image_index, image_bboxes in enumerate(bboxes):
        for bbox in image_bboxes:
            flat_bboxes.append(bbox)
            person_to_image.append(image_index)
    return flat_bboxes, person_to_image


def _bbox_tensor(flat_bboxes: Sequence, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = []
    for bbox in flat_bboxes:
        if bbox is None:
            values.append((0.5, 0.5, 0.5, 0.5))
        else:
            xmin, ymin, xmax, ymax = [float(value) for value in bbox]
            values.append((xmin, ymin, xmax, ymax))
    if not values:
        return torch.empty(0, 4, device=device, dtype=dtype)
    boxes = torch.tensor(values, device=device, dtype=dtype)
    xmin = torch.minimum(boxes[:, 0], boxes[:, 2]).clamp(0.0, 1.0)
    ymin = torch.minimum(boxes[:, 1], boxes[:, 3]).clamp(0.0, 1.0)
    xmax = torch.maximum(boxes[:, 0], boxes[:, 2]).clamp(0.0, 1.0)
    ymax = torch.maximum(boxes[:, 1], boxes[:, 3]).clamp(0.0, 1.0)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def _head_masks(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x0 = torch.floor(boxes[:, 0] * width).long().clamp(0, width - 1)
    y0 = torch.floor(boxes[:, 1] * height).long().clamp(0, height - 1)
    x1 = torch.ceil(boxes[:, 2] * width).long().clamp(1, width)
    y1 = torch.ceil(boxes[:, 3] * height).long().clamp(1, height)
    x1 = torch.maximum(x1, x0 + 1)
    y1 = torch.maximum(y1, y0 + 1)

    x_grid = torch.arange(width, device=boxes.device).view(1, 1, width)
    y_grid = torch.arange(height, device=boxes.device).view(1, height, 1)
    masks = (
        (x_grid >= x0.view(-1, 1, 1))
        & (x_grid < x1.view(-1, 1, 1))
        & (y_grid >= y0.view(-1, 1, 1))
        & (y_grid < y1.view(-1, 1, 1))
    )
    return masks.unsqueeze(1).to(boxes.dtype)


class CoverageAwareSpatialRouter(nn.Module):
    """Predict a high-recall person support and a shared sparse route per image.

    The support is non-parametric over the patch grid: it is not constrained to a
    cone or a single direction. Multiple people in the same image are aggregated
    with a stable max-union before Top-K, so the DINO suffix is still shared.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        keep_ratio: float = 0.25,
        temperature: float = 1.0,
        escape_tokens: int = 8,
    ) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if escape_tokens < 0:
            raise ValueError("escape_tokens must be non-negative")

        self.keep_ratio = float(keep_ratio)
        self.temperature = float(temperature)
        self.escape_tokens = int(escape_tokens)

        self.scene_projection = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.head_projection = nn.Sequential(
            nn.Linear(in_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Start close to a uniform support. Coverage supervision, rather than an
        # arbitrary cone, determines the initial search shape.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _geometry(self, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=boxes.device, dtype=boxes.dtype),
            torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=boxes.device, dtype=boxes.dtype),
            indexing="ij",
        )
        cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).view(-1, 1, 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).view(-1, 1, 1)
        box_w = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0).view(-1, 1, 1)
        box_h = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0).view(-1, 1, 1)
        dx = x_grid.unsqueeze(0) - cx
        dy = y_grid.unsqueeze(0) - cy
        return torch.stack(
            [
                dx,
                dy,
                dx.abs(),
                dy.abs(),
                box_w.expand(-1, height, width),
                box_h.expand(-1, height, width),
            ],
            dim=1,
        )

    def _escape_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        num_tokens = height * width
        mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
        if self.escape_tokens:
            count = min(self.escape_tokens, num_tokens)
            indices = torch.linspace(0, num_tokens - 1, steps=count, device=device).round().long().unique()
            mask[indices] = True
        return mask.view(height, width)

    def forward(self, scene_features: torch.Tensor, bboxes: Sequence[Sequence]) -> RoutingOutput:
        if scene_features.ndim != 4:
            raise ValueError("scene_features must have shape [B, C, H, W]")
        batch_size, _, height, width = scene_features.shape
        if len(bboxes) != batch_size:
            raise ValueError("bboxes must contain one list per input image")
        if any(len(image_bboxes) == 0 for image_bboxes in bboxes):
            raise ValueError("coverage routing requires at least one queried person per image")

        flat_bboxes, mapping = _flatten_bboxes(bboxes)
        if not flat_bboxes:
            raise ValueError("coverage routing requires at least one queried person")

        person_to_image = torch.tensor(mapping, device=scene_features.device, dtype=torch.long)
        boxes = _bbox_tensor(flat_bboxes, scene_features.device, scene_features.dtype)
        person_scenes = scene_features.index_select(0, person_to_image)
        head_masks = _head_masks(boxes, height, width)

        head_denominator = head_masks.sum(dim=(2, 3)).clamp_min(1.0)
        head_vectors = (person_scenes * head_masks).sum(dim=(2, 3)) / head_denominator
        head_condition = torch.cat([head_vectors, boxes], dim=1)

        routed_features = self.scene_projection(person_scenes)
        routed_features = routed_features + self.head_projection(head_condition).unsqueeze(-1).unsqueeze(-1)
        routed_features = routed_features + self.geometry_projection(self._geometry(boxes, height, width))
        support_logits = self.score_head(torch.tanh(routed_features)).squeeze(1)
        # A spatial distribution trains relative ranking directly.  Unlike
        # independent sigmoids, it cannot minimize coverage by turning the
        # entire image on. Keep the probability normalization in FP32 under
        # AMP; the map is tiny relative to DINO activations.
        support_probs = torch.softmax(
            support_logits.float().flatten(1) / self.temperature,
            dim=1,
        ).view_as(support_logits)

        image_union_probs = []
        image_head_masks = []
        for image_index in range(batch_size):
            person_mask = person_to_image == image_index
            # A max-union preserves disjoint modes without making scores grow
            # solely because an image contains more queried people.
            probabilities = support_probs[person_mask]
            union = probabilities.amax(dim=0)
            head_union = head_masks[person_mask, 0].amax(dim=0).bool()
            image_union_probs.append(union)
            image_head_masks.append(head_union)

        image_union_probs = torch.stack(image_union_probs)
        image_head_masks = torch.stack(image_head_masks)
        escape_mask = self._escape_mask(height, width, scene_features.device).unsqueeze(0)

        num_tokens = height * width
        keep_count = min(num_tokens, max(1, int(round(self.keep_ratio * num_tokens))))
        # The fixed-budget method always keeps exactly K patches.  Head cells
        # and escape cells are prioritized inside that budget; if their union
        # is larger than K, the learned score resolves which ones survive.
        ranking_scores = image_union_probs.flatten(1)
        ranking_scores = ranking_scores + image_head_masks.flatten(1).to(image_union_probs.dtype) * 3.0
        ranking_scores = ranking_scores + escape_mask.expand(batch_size, -1, -1).flatten(1).to(
            image_union_probs.dtype
        ) * 2.0
        keep_indices = torch.topk(ranking_scores, k=keep_count, dim=1, largest=True, sorted=False).indices
        keep_indices = torch.sort(keep_indices, dim=1).values

        hard_masks = torch.zeros_like(ranking_scores, dtype=torch.bool)
        hard_masks.scatter_(1, keep_indices, True)
        hard_masks = hard_masks.view(batch_size, height, width)

        return RoutingOutput(
            support_logits=support_logits,
            support_probs=support_probs,
            person_to_image=person_to_image,
            person_head_masks=head_masks.squeeze(1),
            image_union_probs=image_union_probs,
            image_hard_masks=hard_masks,
            image_keep_indices=keep_indices,
            actual_keep_ratio=keep_count / num_tokens,
        )
