"""Factories and frozen candidate definitions for the P3 alchemy study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gazelle.backbone import DinoV3Backbone

from .model import AlchemyGazeLLE


@dataclass(frozen=True)
class CandidateConfig:
    candidate: str
    label: str
    refinement: str
    coordinate_loss_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CANDIDATES = {
    "r0": CandidateConfig("r0", "continued_baseline", "none", 0.0),
    "r1": CandidateConfig("r1", "residual_refinement", "residual_refinement", 0.0),
    "r2": CandidateConfig("r2", "cross_layer_attention", "cross_layer_attention", 0.0),
    "r3": CandidateConfig("r3", "residual_coord005", "residual_refinement", 0.05),
}
CANDIDATE_CHOICES = tuple(_CANDIDATES)


def resolve_candidate(candidate: str) -> CandidateConfig:
    normalized = candidate.strip().lower()
    if normalized not in _CANDIDATES:
        raise ValueError(f"unknown candidate={candidate!r}; choose one of {CANDIDATE_CHOICES}")
    return _CANDIDATES[normalized]


def build_p3_alchemy_model(
    candidate: str,
    *,
    dataset: str,
    backbone_name: str = "dinov3_vitb16",
    **model_kwargs: Any,
):
    if dataset not in ("vat", "gazefollow"):
        raise ValueError("dataset must be 'vat' or 'gazefollow'")
    if backbone_name != "dinov3_vitb16":
        raise ValueError("P3 is frozen to the ViT-B comparison line")
    config = resolve_candidate(candidate)
    backbone = DinoV3Backbone(backbone_name)
    transform = backbone.get_transform((512, 512))
    model = AlchemyGazeLLE(
        backbone,
        refinement=config.refinement,
        inout=(dataset == "vat"),
        **model_kwargs,
    )
    return model, transform, config
