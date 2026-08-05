from typing import Optional

from gazelle.model import GazeLLE
from gazelle.routing.backbone import RoutedDinoV3Backbone
from gazelle.routing.router import CoverageAwareSpatialRouter
from gazelle.routing.timing import record_stage


ROUTER_STAGES = ("support_pilot", "backbone_sparse")


class CoverageAwareGazeLLE(GazeLLE):
    """GazeLLE with person-conditioned routing inside the DINOv3 encoder."""

    def __init__(
        self,
        backbone: RoutedDinoV3Backbone,
        *,
        router_stage: str = "backbone_sparse",
        route_after_block: int,
        keep_ratio: float = 0.25,
        router_hidden_dim: int = 256,
        router_temperature: float = 1.0,
        escape_tokens: int = 8,
        inout: bool = False,
        spatial_prior: str = "none",
        fusion: str = "raw_concat",
    ) -> None:
        if router_stage not in ROUTER_STAGES:
            raise ValueError(f"router_stage must be one of {ROUTER_STAGES}")
        backbone._validate_route(route_after_block)
        if router_stage == "support_pilot" and route_after_block not in backbone.out_indices:
            raise ValueError("support_pilot requires route_after_block to be one of backbone.out_indices")

        super().__init__(
            backbone,
            inout=inout,
            spatial_prior=spatial_prior,
            fusion=fusion,
            use_sasa=False,
            use_ggsf=False,
            use_aux=False,
        )
        self.router_stage = router_stage
        self.route_after_block = int(route_after_block)
        self.router = CoverageAwareSpatialRouter(
            in_dim=backbone.get_dimension(),
            hidden_dim=router_hidden_dim,
            keep_ratio=keep_ratio,
            temperature=router_temperature,
            escape_tokens=escape_tokens,
        )
        self.router_hidden_dim = int(router_hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        # DINOv3 RoPE may jitter coordinates while training.  Keeping the
        # pretrained backbone in eval mode gives stable routing coordinates;
        # trainable suffix blocks still receive gradients.
        self.backbone.eval()
        return self

    def set_trainable_backbone_suffix(self, trainable: bool) -> None:
        self.backbone.set_trainable_suffix(self.route_after_block, trainable=trainable)

    def get_gazelle_state_dict(self, include_backbone=False):
        if not include_backbone:
            raise RuntimeError(
                "Routed models must use the structured coverage-router checkpoint writer; "
                "the legacy non-backbone format can silently lose a fine-tuned DINO suffix. "
                "Pass include_backbone=True only when a full raw state_dict is intentional."
            )
        return super().get_gazelle_state_dict(include_backbone=include_backbone)

    def forward(self, input):
        images = input["images"]
        bboxes = input["bboxes"]

        if self.router_stage == "support_pilot":
            raw_features = self.backbone(images)
            feature_index = self.backbone.out_indices.index(self.route_after_block)
            route_features = raw_features[feature_index]
            routing = self.router(route_features, bboxes)
        else:
            prefix_state = self.backbone.forward_prefix(images, self.route_after_block)
            route_features = prefix_state.dense_features.get(self.route_after_block)
            if route_features is None:
                route_features = self.backbone.tokens_to_map(
                    prefix_state.tokens,
                    prefix_state.height,
                    prefix_state.width,
                )
            routing = self.router(route_features, bboxes)
            raw_features = self.backbone.forward_suffix(prefix_state, routing.image_keep_indices)

        predictions = self.forward_from_features(input, raw_features)
        predictions["routing"] = routing
        return predictions

    def forward_profiled(self, input, stage_recorder):
        """Run the same forward graph with diagnostic stage boundaries."""
        images = input["images"]
        bboxes = input["bboxes"]

        if self.router_stage == "support_pilot":
            with record_stage(stage_recorder, "dense_backbone"):
                raw_features = self.backbone(images)
            feature_index = self.backbone.out_indices.index(self.route_after_block)
            route_features = raw_features[feature_index]
            with record_stage(stage_recorder, "router"):
                routing = self.router(route_features, bboxes)
        else:
            prefix_state = self.backbone.forward_prefix_profiled(
                images,
                self.route_after_block,
                stage_recorder,
            )
            with record_stage(stage_recorder, "route_map"):
                route_features = prefix_state.dense_features.get(self.route_after_block)
                if route_features is None:
                    route_features = self.backbone.tokens_to_map(
                        prefix_state.tokens,
                        prefix_state.height,
                        prefix_state.width,
                    )
            with record_stage(stage_recorder, "router"):
                routing = self.router(route_features, bboxes)
            raw_features = self.backbone.forward_suffix_profiled(
                prefix_state,
                routing.image_keep_indices,
                stage_recorder,
            )

        with record_stage(stage_recorder, "decoder"):
            predictions = self.forward_from_features(input, raw_features)
        predictions["routing"] = routing
        return predictions

    def get_model_config(self) -> dict:
        return {
            "model": self.model_name,
            "router_stage": self.router_stage,
            "route_after_block": self.route_after_block,
            "keep_ratio": self.router.keep_ratio,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router.temperature,
            "escape_tokens": self.router.escape_tokens,
            "spatial_prior": self.spatial_prior,
            "fusion": self.fusion,
        }


def get_coverage_router_model(
    model_name: str,
    *,
    router_stage: str = "backbone_sparse",
    route_after_block: Optional[int] = None,
    keep_ratio: float = 0.25,
    router_hidden_dim: int = 256,
    router_temperature: float = 1.0,
    escape_tokens: int = 8,
    spatial_prior: str = "none",
    fusion: str = "raw_concat",
):
    valid_models = {
        "gazelle_dinov3_vitb16": ("dinov3_vitb16", False, 5),
        "gazelle_dinov3_vitb16_inout": ("dinov3_vitb16", True, 5),
        "gazelle_dinov3_vitl16": ("dinov3_vitl16", False, 11),
        "gazelle_dinov3_vitl16_inout": ("dinov3_vitl16", True, 11),
    }
    if model_name not in valid_models:
        raise ValueError(f"invalid model name: {model_name}")

    backbone_name, inout, default_route = valid_models[model_name]
    route_after_block = default_route if route_after_block is None else route_after_block
    backbone = RoutedDinoV3Backbone(backbone_name)
    transform = backbone.get_transform((512, 512))
    model = CoverageAwareGazeLLE(
        backbone,
        router_stage=router_stage,
        route_after_block=route_after_block,
        keep_ratio=keep_ratio,
        router_hidden_dim=router_hidden_dim,
        router_temperature=router_temperature,
        escape_tokens=escape_tokens,
        inout=inout,
        spatial_prior=spatial_prior,
        fusion=fusion,
    )
    model.model_name = model_name
    model.set_trainable_backbone_suffix(False)
    return model, transform
