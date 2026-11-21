"""Connect the existing gaze-conditioned Router to DINOv2 block-5 features."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from gazelle.routing.router import CoverageAwareSpatialRouter, RoutingOutput
from cscwd_router.code.stag.backbone_sparse_dinov2 import PrefixState, RoutedDinoV2Backbone


@dataclass
class RoutedEncoderOutput:
    features: torch.Tensor
    routing: RoutingOutput


class GazeConditionedDinoV2Encoder(nn.Module):
    """One frozen image encoder, per-person support, one shared sparse suffix.

    The Router and its scene/head/geometry branches are reused without changes.
    Support supervision uses ``gazelle.routing.losses.coverage_router_loss``.
    No target heatmap is an input to routing.
    """

    def __init__(
        self,
        backbone: RoutedDinoV2Backbone,
        *,
        route_after_block: int = 5,
        keep_ratio: float = 0.5,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        escape_tokens: int = 8,
    ):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.route_after_block = route_after_block
        self.router = CoverageAwareSpatialRouter(
            in_dim=backbone.get_dimension(), hidden_dim=hidden_dim,
            keep_ratio=keep_ratio, temperature=temperature, escape_tokens=escape_tokens,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward_prefix(self, images: torch.Tensor) -> tuple[PrefixState, torch.Tensor]:
        state = self.backbone.forward_prefix(images, self.route_after_block)
        features = self.backbone.tokens_to_map(state.tokens, state.height, state.width)
        return state, features

    def route_features(self, features: torch.Tensor, bboxes: Sequence[Sequence]) -> RoutingOutput:
        """Allow support training to reuse frozen prefix features in memory."""
        if any(bbox is None for image_boxes in bboxes for bbox in image_boxes):
            raise ValueError("gaze-conditioned routing requires actual head bboxes for every person")
        return self.router(features, bboxes)

    def predict_support(self, images: torch.Tensor, bboxes: Sequence[Sequence]) -> RoutingOutput:
        _, features = self.forward_prefix(images)
        return self.route_features(features, bboxes)

    def forward(self, images: torch.Tensor, bboxes: Sequence[Sequence]) -> RoutedEncoderOutput:
        state, features = self.forward_prefix(images)
        routing = self.route_features(features, bboxes)
        with torch.no_grad():
            output = self.backbone.forward_suffix(state, routing.image_keep_indices)
        return RoutedEncoderOutput(output, routing)
