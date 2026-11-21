"""Frozen routed DINOv2 and the unmodified official single-layer decoder."""

import torch
from torch import nn

from gazelle.backbone_v0_dinov2 import DinoV2Backbone
from gazelle.model_v0_dinov2 import GazeLLE
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone
from cscwd_router.code.stag.router_dinov2 import GazeConditionedDinoV2Encoder


STAGES = ("dense_official", "support", "sparse_decoder", "matched_dense")


class FeatureGridBackbone(nn.Module):
    """Pass a precomputed grid into GazeLLE.forward without changing its code."""

    def __init__(self, dimension, grid_size):
        super().__init__()
        self.dimension = dimension
        self.grid_size = grid_size

    def get_dimension(self):
        return self.dimension

    def get_out_size(self, in_size):
        return self.grid_size

    def forward(self, features):
        return features


class RoutedGazeLLE(nn.Module):
    def __init__(
        self, backbone: RoutedDinoV2Backbone, *, image_size=448, decoder_dim=256,
        decoder_layers=3, route_after_block=5, router_hidden_dim=256,
        router_temperature=1.0, escape_tokens=8,
    ):
        super().__init__()
        self.encoder = GazeConditionedDinoV2Encoder(
            backbone, route_after_block=route_after_block, keep_ratio=1.0,
            hidden_dim=router_hidden_dim, temperature=router_temperature, escape_tokens=escape_tokens,
        )
        grid = backbone.get_out_size((image_size, image_size))
        self.decoder = GazeLLE(
            FeatureGridBackbone(backbone.get_dimension(), grid), dim=decoder_dim,
            num_layers=decoder_layers, in_size=(image_size, image_size), out_size=(64, 64),
        )
        self.config = {
            "backbone": "dinov2_vitb14", "image_size": image_size, "decoder_dim": decoder_dim,
            "decoder_layers": decoder_layers, "route_after_block": route_after_block,
            "router_hidden_dim": router_hidden_dim, "router_temperature": router_temperature,
            "escape_tokens": escape_tokens,
        }
        self.set_stage("dense_official")

    def load_official_decoder(self, state):
        # The official non-backbone format is preserved, including pos_embed.
        self.decoder.load_state_dict({key: value for key, value in state.items() if not key.startswith("backbone.")}, strict=True)

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError(f"unknown training stage: {stage}")
        self.stage = stage
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        if stage == "support":
            self.encoder.router.requires_grad_(True)
        elif stage in ("sparse_decoder", "matched_dense"):
            self.decoder.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone.eval()
        self.encoder.router.train(mode and self.stage == "support")
        self.decoder.train(mode and self.stage in ("sparse_decoder", "matched_dense"))
        return self

    def predict_support(self, inputs):
        return self.encoder.predict_support(inputs["images"], inputs["bboxes"])

    def forward(self, inputs, *, mode="sparse"):
        if mode == "dense":
            with torch.no_grad():
                features = DinoV2Backbone.forward(self.encoder.backbone, inputs["images"])
            routing = None
        elif mode == "sparse":
            encoded = self.encoder(inputs["images"], inputs["bboxes"])
            features, routing = encoded.features, encoded.routing
        else:
            raise ValueError(f"unknown inference mode: {mode}")
        predictions = self.decoder({"images": features, "bboxes": inputs["bboxes"]})
        predictions["routing"] = routing
        return predictions


def build_model(config=None, *, backbone=None):
    config = dict(config or {})
    name = config.pop("backbone", "dinov2_vitb14")
    if backbone is None:
        backbone = RoutedDinoV2Backbone(name)
    return RoutedGazeLLE(backbone, **config)
