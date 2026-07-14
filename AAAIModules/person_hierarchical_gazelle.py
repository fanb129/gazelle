"""Gazelle decoder with person-conditioned hierarchical feature routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision

import gazelle.utils as gazelle_utils
from gazelle.model import GazeLLE

from .person_hierarchical_router import PersonConditionedHierarchicalRouter


class PersonHierarchicalGazeLLE(GazeLLE):
    """Minimal P1 candidate that changes only Gazelle's layer fusion policy.

    The backbone runs once per image batch. Its shared hierarchy is routed for
    each bbox, then passed through the inherited Gazelle projection, transformer,
    heatmap head, and optional in/out head.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        inout: bool = False,
        dim: int = 256,
        num_layers: int = 3,
        in_size: tuple[int, int] = (512, 512),
        out_size: tuple[int, int] = (64, 64),
        router_hidden_dim: int = 128,
        router_roi_size: int = 3,
        router_dropout: float = 0.1,
        router_temperature: float = 1.0,
        router_prior_weights: tuple[float, float, float, float] = (
            0.0340173, 0.0974448, 0.2007198, 0.6678180
        ),
        router_residual_scale: float = 1.0,
        hierarchy_layers: int = 4,
    ) -> None:
        super().__init__(
            backbone,
            inout=inout,
            dim=dim,
            num_layers=num_layers,
            in_size=in_size,
            out_size=out_size,
            spatial_prior="none",
            fusion="raw_concat",
            use_aux=False,
        )
        if hierarchy_layers != 4:
            raise ValueError(
                "The inherited Gazelle projection expects four concatenated layers; "
                "hierarchy_layers must remain 4 for checkpoint compatibility."
            )
        self.hierarchy_layers = hierarchy_layers
        self.layer_router = PersonConditionedHierarchicalRouter(
            feature_dim=self.raw_dim,
            num_layers=hierarchy_layers,
            hidden_dim=router_hidden_dim,
            roi_size=router_roi_size,
            dropout=router_dropout,
            temperature=router_temperature,
            prior_weights=router_prior_weights,
            residual_scale=router_residual_scale,
        )

    def forward(self, input: Mapping[str, Any]) -> dict[str, Any]:
        if "images" not in input or "bboxes" not in input:
            raise KeyError("input must provide 'images' and nested 'bboxes'")
        bboxes = input["bboxes"]
        num_ppl_per_img = [len(image_boxes) for image_boxes in bboxes]

        # Exactly one shared backbone call. No person-level image duplication occurs here.
        shared_features = self.backbone(input["images"])
        weighted_features, layer_weights, router_metadata = self.layer_router(shared_features, bboxes)
        if layer_weights.shape[0] == 0:
            raise ValueError("A Gazelle forward pass requires at least one gaze query bbox")

        x = self.linear(torch.cat(weighted_features, dim=1))
        x = x + self.pos_embed
        head_maps = torch.cat(self.get_input_head_maps(bboxes), dim=0).to(device=x.device, dtype=x.dtype)
        x = x + head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            inout_token = self.inout_token.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([inout_token, x], dim=1)
        x = self.transformer(x)

        inout_preds = None
        if self.inout:
            inout_preds = gazelle_utils.split_tensors(
                self.inout_head(x[:, 0]).squeeze(-1), num_ppl_per_img
            )
            x = x[:, 1:]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = gazelle_utils.split_tensors(x, num_ppl_per_img)

        backbone_indices = getattr(self.backbone, "out_indices", None)
        metadata = {
            **router_metadata,
            "candidate": "P1a_global_prior_plus_person_residual_router",
            "backbone_passes_per_forward": 1,
            "shared_scene_encoding": True,
            "backbone_layer_indices": list(backbone_indices) if backbone_indices is not None else None,
            "decoder": "inherited_gazelle",
            "supports_inout": self.inout,
            "claim_boundary": "No relational loss or query-interaction mechanism is implemented.",
        }
        return {
            "heatmap": heatmap_preds,
            "aux_heatmap": None,
            "inout": inout_preds,
            "layer_weights": layer_weights,
            "layer_weights_split": gazelle_utils.split_tensors(layer_weights, num_ppl_per_img),
            "geo_mask": None,
            "spatial_prior": "none",
            "fusion": "person_conditioned_hierarchical",
            "fusion_metadata": metadata,
            "metadata": metadata,
        }

    def load_base_checkpoint(
        self,
        checkpoint: str | Path | Mapping[str, torch.Tensor],
        *,
        map_location: str | torch.device = "cpu",
        allow_legacy_sasa_ggsf: bool = False,
    ) -> dict[str, list[str]]:
        """Load a historical Gazelle checkpoint and report compatibility.

        Router parameters are intentionally left at their initialization when the
        checkpoint predates this candidate. A nested ``state_dict`` and DDP's
        ``module.`` prefix are accepted.
        """
        if isinstance(checkpoint, (str, Path)):
            state = torch.load(checkpoint, map_location=map_location, weights_only=True)
        else:
            state = dict(checkpoint)
        if "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = dict(state["state_dict"])
        state = {key.removeprefix("module."): value for key, value in state.items()}
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        incompatible_shapes = sorted(
            key for key, value in state.items() if key in current and current[key].shape != value.shape
        )
        unexpected = sorted(key for key in state if key not in current)
        ignored_source_keys: list[str] = []
        if allow_legacy_sasa_ggsf:
            ignored_source_keys = [
                key for key in unexpected if key.startswith(("sasa.", "ggsf."))
            ]
            unexpected = [key for key in unexpected if key not in ignored_source_keys]
        missing = sorted(key for key in current if key not in compatible)
        self.load_state_dict(compatible, strict=False)
        return {
            "loaded": sorted(compatible),
            "missing": missing,
            "unexpected": unexpected,
            "ignored_source_keys": sorted(ignored_source_keys),
            "incompatible_shapes": incompatible_shapes,
        }
