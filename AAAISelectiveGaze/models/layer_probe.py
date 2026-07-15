"""Lightweight, independent gaze probes for fixed backbone layers.

The probes deliberately consume *pre-fusion* feature maps.  They do not import
or instantiate Gazelle/DINO, which keeps synthetic tests and cached-feature
training independent from heavyweight backbone checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_LAYERS: Final[tuple[int, ...]] = (2, 5, 8, 11)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return the number of scalar parameters in ``module``."""

    parameters = (
        parameter
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
    return sum(parameter.numel() for parameter in parameters)


def freeze_module(module: nn.Module, *, eval_mode: bool = True) -> nn.Module:
    """Freeze a module in place and return it for convenient composition."""

    module.requires_grad_(False)
    if eval_mode:
        module.eval()
    return module


class LayerGazeProbe(nn.Module):
    """Decode one backbone layer into a 64x64 gaze heatmap.

    ``forward`` returns logits so callers can use ``binary_cross_entropy_with_logits``.
    Use :meth:`predict_heatmap` when normalized ``[0, 1]`` heatmaps are needed.

    Args:
        in_channels: Number of feature channels in ``[N, C, H, W]``.
        hidden_channels: Width shared by the projection and conditioned block.
        output_size: Spatial output size.  The pilot protocol fixes this to 64x64.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        output_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}"
            )
        if tuple(output_size) != (64, 64):
            raise ValueError(
                "The fixed-layer pilot requires output_size=(64, 64), "
                f"got {output_size}"
            )

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.output_size = (64, 64)

        # Exactly one layer-specific 1x1 projection.
        self.projection = nn.Conv2d(self.in_channels, self.hidden_channels, 1)

        # A lightweight head-conditioned residual block. Concatenating the binary
        # head map makes person conditioning explicit rather than relying on the
        # feature extractor to have encoded a particular subject.
        self.head_conditioned_block = nn.Sequential(
            nn.Conv2d(self.hidden_channels + 1, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(self.hidden_channels, 1, 1)

    def _validate_inputs(self, feature: Tensor, head_map: Tensor) -> None:
        if feature.ndim != 4:
            raise ValueError(
                "feature must have shape [N, C, H, W], "
                f"got {tuple(feature.shape)}"
            )
        if head_map.ndim != 3:
            raise ValueError(
                "head_map must have shape [N, H, W], "
                f"got {tuple(head_map.shape)}"
            )
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} feature channels, "
                f"got {feature.shape[1]}"
            )
        if feature.shape[0] != head_map.shape[0]:
            raise ValueError(
                "feature and head_map batch sizes differ: "
                f"{feature.shape[0]} != {head_map.shape[0]}"
            )
        if tuple(feature.shape[-2:]) != tuple(head_map.shape[-2:]):
            raise ValueError(
                "feature and head_map spatial shapes differ: "
                f"{tuple(feature.shape[-2:])} != {tuple(head_map.shape[-2:])}"
            )
        if not feature.is_floating_point():
            raise TypeError(f"feature must be floating point, got {feature.dtype}")

    def forward(self, feature: Tensor, head_map: Tensor) -> Tensor:
        """Return gaze logits with shape ``[N, 64, 64]``."""

        self._validate_inputs(feature, head_map)
        projected = self.projection(feature)
        condition = head_map.to(device=feature.device, dtype=feature.dtype).unsqueeze(1)
        conditioned = self.head_conditioned_block(
            torch.cat((projected, condition), dim=1)
        )
        logits = self.heatmap_head(projected + conditioned)
        logits = F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits.squeeze(1)

    def predict_heatmap(self, feature: Tensor, head_map: Tensor) -> Tensor:
        """Return sigmoid gaze heatmaps with shape ``[N, 64, 64]``."""

        return self(feature, head_map).sigmoid()

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)


class FixedLayerProbes(nn.Module):
    """One independent :class:`LayerGazeProbe` for every fixed layer.

    Feature input may be either a mapping keyed by integer layer id or a sequence
    in exactly the configured layer order.  The result is always a ``dict[int,
    Tensor]``, preserving that order.
    """

    def __init__(
        self,
        in_channels: int | Mapping[int, int],
        layers: Sequence[int] = DEFAULT_LAYERS,
        hidden_channels: int = 64,
        output_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        normalized_layers = tuple(int(layer) for layer in layers)
        if not normalized_layers:
            raise ValueError("layers must not be empty")
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError(f"layers must be unique, got {normalized_layers}")

        if isinstance(in_channels, Mapping):
            normalized_channels = {int(key): int(value) for key, value in in_channels.items()}
            missing = set(normalized_layers) - set(normalized_channels)
            extra = set(normalized_channels) - set(normalized_layers)
            if missing or extra:
                raise ValueError(
                    "in_channels keys must exactly match layers; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
        else:
            normalized_channels = {layer: int(in_channels) for layer in normalized_layers}

        self.layers = normalized_layers
        self.in_channels = normalized_channels
        self.probes = nn.ModuleDict(
            {
                str(layer): LayerGazeProbe(
                    in_channels=normalized_channels[layer],
                    hidden_channels=hidden_channels,
                    output_size=output_size,
                )
                for layer in self.layers
            }
        )

    def _as_feature_mapping(
        self, features: Mapping[int, Tensor] | Sequence[Tensor]
    ) -> dict[int, Tensor]:
        if isinstance(features, Mapping):
            normalized = {int(key): value for key, value in features.items()}
            missing = set(self.layers) - set(normalized)
            extra = set(normalized) - set(self.layers)
            if missing or extra:
                raise ValueError(
                    "feature layers must exactly match configured layers; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            return normalized

        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError("features must be a layer mapping or a tensor sequence")
        if len(features) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} feature tensors, got {len(features)}"
            )
        return dict(zip(self.layers, features))

    def forward(
        self,
        features: Mapping[int, Tensor] | Sequence[Tensor],
        head_map: Tensor,
    ) -> dict[int, Tensor]:
        """Return layer-keyed 64x64 logits."""

        feature_map = self._as_feature_mapping(features)
        outputs: dict[int, Tensor] = {}
        reference_shape: tuple[int, int, int] | None = None
        for layer in self.layers:
            feature = feature_map[layer]
            if not isinstance(feature, Tensor):
                raise TypeError(f"feature for layer {layer} is not a Tensor")
            shape = (feature.shape[0], feature.shape[-2], feature.shape[-1]) if feature.ndim == 4 else None
            if reference_shape is None and shape is not None:
                reference_shape = shape
            elif shape is not None and shape != reference_shape:
                raise ValueError(
                    "all fixed-layer features must share [N, H, W]; "
                    f"layer {layer} has {shape}, expected {reference_shape}"
                )
            outputs[layer] = self.probes[str(layer)](feature, head_map)
        return outputs

    def predict_heatmaps(
        self,
        features: Mapping[int, Tensor] | Sequence[Tensor],
        head_map: Tensor,
    ) -> dict[int, Tensor]:
        return {
            layer: logits.sigmoid()
            for layer, logits in self(features, head_map).items()
        }

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)


__all__ = [
    "DEFAULT_LAYERS",
    "FixedLayerProbes",
    "LayerGazeProbe",
    "count_parameters",
    "freeze_module",
]
