"""Scoring-input ablations; retain native routing, priorities and state keys."""

import torch
from torch import nn

from gazelle.routing.model import CoverageAwareGazeLLE
from gazelle.routing.backbone import RoutedDinoV3Backbone

CONDITIONS = {"scene_only": (False, False), "scene_geometry": (False, True),
              "scene_head": (True, False), "full": (True, True), "unconditional": (False, False)}


class MaskedHeadProjection(nn.Sequential):
    def __init__(self, projection, appearance, geometry):
        super().__init__(*projection.children())
        self.appearance, self.geometry = appearance, geometry

    def forward(self, value):
        if not self.appearance and not self.geometry:
            return super().forward(torch.zeros_like(value)) * 0
        head, box = value[:, :-4], value[:, -4:]
        value = torch.cat((head if self.appearance else torch.zeros_like(head),
                           box if self.geometry else torch.zeros_like(box)), dim=1)
        return super().forward(value)


class ZeroProjection(nn.Sequential):
    def forward(self, value):
        return super().forward(torch.zeros_like(value)) * 0


def configure_inputs(router, condition):
    if condition == "unconditional":
        from cscwd_router.code.stag.unconditional import without_person_conditions
        return without_person_conditions(router)
    appearance, geometry = CONDITIONS[condition]
    if condition != "full":
        router.head_projection = MaskedHeadProjection(router.head_projection, appearance, geometry)
        if not geometry:
            router.geometry_projection = ZeroProjection(*router.geometry_projection.children())
    return router


class ConditionalGazeLLE(CoverageAwareGazeLLE):
    def __init__(self, backbone, *, condition, **kwargs):
        super().__init__(backbone, **kwargs)
        self.condition = condition
        self.router = configure_inputs(self.router, condition)

    def get_model_config(self):
        config = super().get_model_config()
        return config if self.condition == "full" else {**config, "routing_condition": self.condition}


def get_conditional_model(model_name, *, condition="full", router_stage="backbone_sparse",
                          route_after_block=None, keep_ratio=.5, router_hidden_dim=256,
                          router_temperature=1., escape_tokens=8, spatial_prior="none", fusion="raw_concat"):
    size = "b" if "vitb16" in model_name else "l"
    expected = f"gazelle_dinov3_vit{size}16"
    if model_name not in (expected, expected + "_inout"):
        raise ValueError(f"invalid model name: {model_name}")
    backbone = RoutedDinoV3Backbone(f"dinov3_vit{size}16")
    model = ConditionalGazeLLE(backbone, condition=condition, router_stage=router_stage,
        route_after_block=(5 if size == "b" else 11) if route_after_block is None else route_after_block,
        keep_ratio=keep_ratio, router_hidden_dim=router_hidden_dim, router_temperature=router_temperature,
        escape_tokens=escape_tokens, spatial_prior=spatial_prior, fusion=fusion,
        inout=model_name.endswith("_inout"))
    model.model_name = model_name
    model.set_trainable_backbone_suffix(False)
    return model, backbone.get_transform((512, 512))
