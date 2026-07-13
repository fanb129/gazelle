"""Factories for the isolated AAAI candidate model."""

from __future__ import annotations

from typing import Any

from gazelle.backbone import DinoV3Backbone

from .person_hierarchical_gazelle import PersonHierarchicalGazeLLE


MODEL_NAMES = (
    "aaai_person_router_dinov3_vitb16",
    "aaai_person_router_dinov3_vitb16_inout",
    "aaai_person_router_dinov3_vitl16",
    "aaai_person_router_dinov3_vitl16_inout",
)


def build_person_hierarchical_gazelle(model_name: str, **model_kwargs: Any):
    """Build the P1 candidate and the matching historical Gazelle transform."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name={model_name!r}; choose one of {MODEL_NAMES}")
    use_large = "vitl16" in model_name
    backbone_name = "dinov3_vitl16" if use_large else "dinov3_vitb16"
    inout = model_name.endswith("_inout")
    backbone = DinoV3Backbone(backbone_name)
    transform = backbone.get_transform((512, 512))
    model = PersonHierarchicalGazeLLE(backbone, inout=inout, **model_kwargs)
    return model, transform
