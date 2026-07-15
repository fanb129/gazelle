"""Non-invasive wrapper that exposes pre-fusion layer-probe predictions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from AAAISelectiveGaze import FIXED_DINOV3_VITB_LAYERS
from AAAISelectiveGaze.models.layer_probe import FixedLayerProbes, freeze_module


def _repeat_per_person(feature: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    index = torch.arange(feature.shape[0], device=feature.device).repeat_interleave(
        torch.as_tensor(counts, device=feature.device)
    )
    return feature.index_select(0, index)


def _split_per_image(tensor: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    return list(torch.split(tensor, [int(count) for count in counts], dim=0))


@contextmanager
def _reuse_backbone_features(backbone: nn.Module, features: Sequence[torch.Tensor]):
    """Make legacy ``backbone.forward`` calls reuse one already-computed result.

    The current Gazelle calls ``self.backbone.forward`` directly, so module hooks
    cannot intercept it.  This narrow, exception-safe substitution avoids a second
    expensive DINO pass without modifying the Gazelle source tree.
    """

    original_forward = backbone.forward
    backbone.forward = lambda _images: features  # type: ignore[method-assign]
    try:
        yield
    finally:
        backbone.forward = original_forward  # type: ignore[method-assign]


class SelectiveGazelle(nn.Module):
    """Compose a frozen Gazelle predictor with four independent fixed probes."""

    def __init__(
        self,
        predictor: nn.Module,
        probes: FixedLayerProbes,
        layers: Sequence[int] = FIXED_DINOV3_VITB_LAYERS,
        freeze_predictor: bool = True,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.probes = probes
        self.layers = tuple(int(layer) for layer in layers)
        if tuple(probes.layers) != self.layers:
            raise ValueError(
                f"probe layers {tuple(probes.layers)} do not match wrapper layers {self.layers}"
            )
        backbone_layers = getattr(getattr(predictor, "backbone", None), "out_indices", None)
        if backbone_layers is not None and tuple(backbone_layers) != self.layers:
            raise ValueError(
                f"predictor backbone layers {tuple(backbone_layers)} do not match {self.layers}"
            )
        if freeze_predictor:
            freeze_module(self.predictor)

    def train(self, mode: bool = True):
        super().train(mode)
        # The base predictor remains a deterministic, frozen teacher while probes train.
        self.predictor.eval()
        self.probes.train(mode)
        return self

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        images = inputs.get("images")
        bboxes = inputs.get("bboxes")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("inputs['images'] must be a [B,C,H,W] tensor")
        if not isinstance(bboxes, Sequence) or len(bboxes) != images.shape[0]:
            raise ValueError("inputs['bboxes'] must contain one bbox list per image")

        counts = [len(items) for items in bboxes]
        if sum(counts) == 0:
            raise ValueError("a SelectiveGazelle batch must contain at least one person")

        with torch.no_grad():
            raw_features = self.predictor.backbone(images)
            if len(raw_features) != len(self.layers):
                raise ValueError(
                    f"backbone returned {len(raw_features)} features for {len(self.layers)} layers"
                )
            with _reuse_backbone_features(self.predictor.backbone, raw_features):
                predictor_output = self.predictor(dict(inputs))

        person_features = {
            layer: _repeat_per_person(feature, counts)
            for layer, feature in zip(self.layers, raw_features)
        }
        head_maps = torch.cat(self.predictor.get_input_head_maps(bboxes), dim=0).to(
            device=images.device, dtype=person_features[self.layers[0]].dtype
        )
        probe_logits_flat = self.probes(person_features, head_maps)
        probe_heatmaps_flat = {
            layer: torch.sigmoid(logits) for layer, logits in probe_logits_flat.items()
        }
        output = dict(predictor_output)
        output.update(
            {
                "probe_logits_flat": probe_logits_flat,
                "probe_heatmaps_flat": probe_heatmaps_flat,
                "probe_heatmap": {
                    layer: _split_per_image(heatmaps, counts)
                    for layer, heatmaps in probe_heatmaps_flat.items()
                },
                "probe_layers": self.layers,
            }
        )
        return output


__all__ = ["SelectiveGazelle"]
