"""Zero-initialized residual fusion branches used by P3.

The branches consume the same four SASA-weighted, GGSF-gated feature maps as
the historical projection.  Their final projection is initialized to exactly
zero, so inserting either branch leaves the inherited decoder input unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _check_features(features: Sequence[torch.Tensor], num_layers: int) -> tuple[int, int, int, int]:
    if len(features) != num_layers:
        raise ValueError(f"expected {num_layers} feature layers, got {len(features)}")
    if not features:
        raise ValueError("features cannot be empty")
    shape = tuple(features[0].shape)
    if len(shape) != 4:
        raise ValueError(f"features must be NCHW tensors, got {shape}")
    if any(tuple(feature.shape) != shape for feature in features[1:]):
        raise ValueError("all hierarchy features must have the same shape")
    return shape  # type: ignore[return-value]


class ResidualRefinement(nn.Module):
    """R1: compact spatial/channel refinement over concatenated hierarchy maps."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 256, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.input_projection = nn.Conv2d(in_channels * num_layers, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.spatial = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.channel = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )
        self.output_projection = nn.Conv2d(width, out_channels, kernel_size=1)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        _check_features(features, self.num_layers)
        hidden = self.input_projection(torch.cat(list(features), dim=1))
        hidden = self.norm(hidden)
        hidden = hidden + self.spatial(hidden)
        hidden = hidden + self.channel(hidden)
        return self.output_projection(hidden)

    def metadata(self) -> dict:
        return {
            "branch": "residual_refinement",
            "num_layers": self.num_layers,
            "zero_initialized_output": True,
        }


class CrossLayerAttentionResidual(nn.Module):
    """R2: per-location self-attention across the four hierarchy levels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.input_projection = nn.Conv2d(in_channels, attention_dim, kernel_size=1)
        self.layer_embedding = nn.Parameter(torch.zeros(num_layers, attention_dim))
        nn.init.normal_(self.layer_embedding, std=0.02)
        self.pre_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads, batch_first=True)
        self.post_norm = nn.LayerNorm(attention_dim)
        self.output_projection = nn.Conv2d(attention_dim, out_channels, kernel_size=1)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        n, _, h, w = _check_features(features, self.num_layers)
        projected = torch.stack([self.input_projection(feature) for feature in features], dim=1)
        # [N, L, D, H, W] -> one four-layer token sequence per spatial location.
        tokens = projected.permute(0, 3, 4, 1, 2).reshape(n * h * w, self.num_layers, self.attention_dim)
        tokens = tokens + self.layer_embedding.unsqueeze(0).to(dtype=tokens.dtype)
        normalized = self.pre_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = self.post_norm(tokens + attended).mean(dim=1)
        fused = tokens.reshape(n, h, w, self.attention_dim).permute(0, 3, 1, 2)
        return self.output_projection(fused)

    def metadata(self) -> dict:
        return {
            "branch": "cross_layer_attention",
            "aggregation": "per_location_four_layer_self_attention",
            "num_layers": self.num_layers,
            "attention_dim": self.attention_dim,
            "zero_initialized_output": True,
        }
