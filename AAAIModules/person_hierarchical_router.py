"""Person-conditioned routing over a shared hierarchy of scene features.

This module is deliberately narrow: it tests whether different gaze queries need
different backbone layers.  It does not model interactions between people.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_normalized_bboxes(
    bboxes: Sequence[Sequence[Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten nested normalized xyxy boxes and retain their image indices."""
    flat_boxes: list[torch.Tensor] = []
    image_indices: list[int] = []
    for image_index, image_boxes in enumerate(bboxes):
        for bbox in image_boxes:
            if bbox is None:
                box = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype)
            else:
                box = torch.as_tensor(bbox, device=device, dtype=dtype).flatten()
                if box.numel() != 4:
                    raise ValueError(f"Each bbox must contain four xyxy values, got shape {tuple(box.shape)}")
            x1, y1, x2, y2 = box.unbind()
            # Sorting also makes jittered or malformed boxes safe for sampling.
            box = torch.stack(
                [torch.minimum(x1, x2), torch.minimum(y1, y2),
                 torch.maximum(x1, x2), torch.maximum(y1, y2)]
            ).clamp(0.0, 1.0)
            flat_boxes.append(box)
            image_indices.append(image_index)

    if not flat_boxes:
        return (
            torch.empty(0, 4, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
        )
    return torch.stack(flat_boxes), torch.tensor(image_indices, device=device, dtype=torch.long)


def normalized_box_pool(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    image_indices: torch.Tensor,
    output_size: int = 3,
) -> torch.Tensor:
    """Differentiably pool normalized boxes without depending on custom ROI ops.

    Args:
        feature: Shared per-image feature map ``[B, C, H, W]``.
        boxes: Flattened normalized xyxy boxes ``[P, 4]``.
        image_indices: Source-image index for every person ``[P]``.
        output_size: Number of samples per spatial dimension inside each box.

    Returns:
        Mean-pooled descriptors with shape ``[P, C]``.
    """
    if feature.ndim != 4:
        raise ValueError(f"feature must be [B,C,H,W], got {tuple(feature.shape)}")
    if boxes.shape[0] == 0:
        return feature.new_empty((0, feature.shape[1]))
    if output_size < 1:
        raise ValueError("output_size must be positive")

    sampled_features = feature.index_select(0, image_indices)
    steps = (torch.arange(output_size, device=feature.device, dtype=feature.dtype) + 0.5) / output_size
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    xs = x1[:, None] + (x2 - x1)[:, None] * steps[None, :]
    ys = y1[:, None] + (y2 - y1)[:, None] * steps[None, :]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(output_size, device=feature.device),
        torch.arange(output_size, device=feature.device),
        indexing="ij",
    )
    grid = torch.stack(
        [xs[:, grid_x] * 2.0 - 1.0, ys[:, grid_y] * 2.0 - 1.0], dim=-1
    )
    pooled = F.grid_sample(
        sampled_features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return pooled.mean(dim=(-2, -1))


class PersonConditionedHierarchicalRouter(nn.Module):
    """Predict one hierarchy distribution for each person/query bbox."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        roi_size: int = 3,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("Hierarchical routing requires at least two feature layers")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.roi_size = roi_size
        self.temperature = temperature

        self.roi_norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(feature_dim * 2 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start close to equal weighting; learning must justify specialization.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        shared_features: Sequence[torch.Tensor],
        bboxes: Sequence[Sequence[Any]],
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        if len(shared_features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature layers, got {len(shared_features)}")
        reference = shared_features[0]
        if reference.ndim != 4:
            raise ValueError("Each shared feature must have shape [B,C,H,W]")
        for feature in shared_features:
            if feature.shape != reference.shape:
                raise ValueError("All hierarchy features must have the same shape")
            if feature.shape[1] != self.feature_dim:
                raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature.shape[1]}")
        if len(bboxes) != reference.shape[0]:
            raise ValueError("bboxes must contain one nested list per input image")

        boxes, image_indices = flatten_normalized_bboxes(
            bboxes, device=reference.device, dtype=reference.dtype
        )
        num_people = boxes.shape[0]
        if num_people == 0:
            weights = reference.new_empty((0, self.num_layers))
            empty = [feature.new_empty((0, *feature.shape[1:])) for feature in shared_features]
            return empty, weights, self._metadata(num_people)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        geometry = torch.stack(
            [centers_x, centers_y, widths, heights, widths * heights, widths / heights.clamp_min(1e-6)],
            dim=-1,
        )
        geometry_code = self.geometry_encoder(geometry)

        roi_descriptors = []
        context_descriptors = []
        person_features = []
        for feature in shared_features:
            roi_descriptors.append(normalized_box_pool(feature, boxes, image_indices, self.roi_size))
            global_context = feature.mean(dim=(-2, -1)).index_select(0, image_indices)
            context_descriptors.append(global_context)
            person_features.append(feature.index_select(0, image_indices))

        roi_tensor = self.roi_norm(torch.stack(roi_descriptors, dim=1))
        context_tensor = self.context_norm(torch.stack(context_descriptors, dim=1))
        layer_ids = torch.arange(self.num_layers, device=reference.device)
        layer_code = self.layer_embedding(layer_ids)[None].expand(num_people, -1, -1)
        geometry_code = geometry_code[:, None].expand(-1, self.num_layers, -1)
        logits = self.score(torch.cat([roi_tensor, context_tensor, geometry_code, layer_code], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits / self.temperature, dim=1)

        # Keep Gazelle's historical 4C -> dim projection compatible by weighting
        # each layer and concatenating, rather than replacing it with a C -> dim sum.
        weighted_features = [
            feature * weights[:, layer_index, None, None, None]
            for layer_index, feature in enumerate(person_features)
        ]
        return weighted_features, weights, self._metadata(num_people)

    def _metadata(self, num_people: int) -> dict[str, Any]:
        return {
            "router": "person_conditioned_hierarchical",
            "routing_scope": "per_query_bbox",
            "models_query_interaction": False,
            "pooling": "normalized_box_grid_mean",
            "roi_size": self.roi_size,
            "num_layers": self.num_layers,
            "num_people": num_people,
            "temperature": self.temperature,
            "fusion_after_routing": "weighted_layer_concat",
        }
