"""P3 model wrapper that preserves historical SASA+GGSF and adds a residual."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision

import gazelle.utils as gazelle_utils
from gazelle.model import GazeLLE

from .fusion import CrossLayerAttentionResidual, ResidualRefinement


REFINEMENT_CHOICES = ("none", "residual_refinement", "cross_layer_attention")


class AlchemyGazeLLE(GazeLLE):
    """Historical SASA+GGSF model plus an optional zero-init residual branch."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        refinement: str = "none",
        inout: bool = False,
        dim: int = 256,
        num_layers: int = 3,
        in_size: tuple[int, int] = (512, 512),
        out_size: tuple[int, int] = (64, 64),
        refinement_width: int = 256,
        attention_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        if refinement not in REFINEMENT_CHOICES:
            raise ValueError(f"unknown refinement={refinement!r}; choose one of {REFINEMENT_CHOICES}")
        super().__init__(
            backbone,
            inout=inout,
            dim=dim,
            num_layers=num_layers,
            in_size=in_size,
            out_size=out_size,
            use_sasa=True,
            use_ggsf=True,
            use_aux=False,
            dropout=dropout,
        )
        self.refinement = refinement
        if refinement == "residual_refinement":
            self.fusion_refiner = ResidualRefinement(
                self.raw_dim, dim, width=refinement_width, num_layers=4
            )
        elif refinement == "cross_layer_attention":
            self.fusion_refiner = CrossLayerAttentionResidual(
                self.raw_dim,
                dim,
                attention_dim=attention_dim,
                num_heads=attention_heads,
                num_layers=4,
            )
        else:
            self.fusion_refiner = None

    def forward(self, input: Mapping[str, Any]) -> dict[str, Any]:
        if "images" not in input or "bboxes" not in input:
            raise KeyError("input must provide 'images' and nested 'bboxes'")
        bboxes = input["bboxes"]
        num_ppl_per_img = [len(image_boxes) for image_boxes in bboxes]
        if sum(num_ppl_per_img) == 0:
            raise ValueError("a forward pass requires at least one gaze query bbox")

        shared_features = self.backbone(input["images"])
        person_features = [gazelle_utils.repeat_tensors(feature, num_ppl_per_img) for feature in shared_features]
        geo_mask = self.ggsf(bboxes, person_features[0].device)
        gated_features = [feature * geo_mask for feature in person_features]
        weighted_features, layer_weights = self.sasa(gated_features)

        base_fused = self.linear(torch.cat(weighted_features, dim=1))
        residual = (
            self.fusion_refiner(weighted_features)
            if self.fusion_refiner is not None
            else torch.zeros_like(base_fused)
        )
        x = base_fused + residual
        x = x + self.pos_embed
        head_maps = torch.cat(self.get_input_head_maps(bboxes), dim=0).to(device=x.device, dtype=x.dtype)
        x = x + head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            token = self.inout_token.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([token, x], dim=1)
        x = self.transformer(x)

        inout_preds = None
        if self.inout:
            inout_preds = gazelle_utils.split_tensors(
                self.inout_head(x[:, 0]).squeeze(-1), num_ppl_per_img
            )
            x = x[:, 1:]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        heatmaps = self.heatmap_head(x).squeeze(1)
        heatmaps = torchvision.transforms.functional.resize(heatmaps, self.out_size)
        heatmap_preds = gazelle_utils.split_tensors(heatmaps, num_ppl_per_img)

        refinement_metadata = (
            self.fusion_refiner.metadata() if self.fusion_refiner is not None else {
                "branch": "none", "zero_initialized_output": True
            }
        )
        metadata = {
            "base_fusion": "historical_sasa_ggsf",
            "refinement": self.refinement,
            "residual_addition_point": "after_historical_linear_projection",
            "backbone_passes_per_forward": 1,
            **refinement_metadata,
        }
        return {
            "heatmap": heatmap_preds,
            "aux_heatmap": None,
            "inout": inout_preds,
            "layer_weights": layer_weights,
            "layer_weights_split": gazelle_utils.split_tensors(layer_weights, num_ppl_per_img),
            "geo_mask": geo_mask,
            "spatial_prior": "ggsf",
            "fusion": "sasa",
            "fusion_metadata": metadata,
            "metadata": metadata,
        }

    def load_alchemy_checkpoint(
        self,
        checkpoint: str | Path | Mapping[str, torch.Tensor],
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load legacy or P3 task weights and return an auditable coverage report."""
        checkpoint_path = None
        if isinstance(checkpoint, (str, Path)):
            checkpoint_path = str(Path(checkpoint).expanduser().resolve())
            state: Mapping[str, Any] = torch.load(checkpoint, map_location=map_location, weights_only=True)
        else:
            state = checkpoint
        if "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]
        tensors = {
            key.removeprefix("module."): value
            for key, value in state.items()
            if torch.is_tensor(value)
        }
        current = self.state_dict()
        compatible = {
            key: value for key, value in tensors.items()
            if not key.startswith("backbone.")
            and key in current
            and tuple(current[key].shape) == tuple(value.shape)
        }
        incompatible_shapes = {
            key: {"expected": list(current[key].shape), "provided": list(value.shape)}
            for key, value in tensors.items()
            if not key.startswith("backbone.")
            and key in current
            and tuple(current[key].shape) != tuple(value.shape)
        }
        ignored_source_keys = sorted(key for key in tensors if key.startswith("backbone."))
        unexpected = sorted(
            key for key in tensors if key not in current and not key.startswith("backbone.")
        )
        full_state = dict(current)
        full_state.update(compatible)
        self.load_state_dict(full_state, strict=True)

        expected_task = [key for key in current if not key.startswith("backbone.")]
        loaded_task = sorted(key for key in compatible if not key.startswith("backbone."))
        missing = sorted(key for key in expected_task if key not in compatible)
        new_branch_expected = [key for key in expected_task if key.startswith("fusion_refiner.")]
        new_branch_loaded = [key for key in loaded_task if key.startswith("fusion_refiner.")]
        new_branch_missing = [key for key in missing if key.startswith("fusion_refiner.")]
        shared_expected = [key for key in expected_task if not key.startswith("fusion_refiner.")]
        shared_loaded = [key for key in loaded_task if not key.startswith("fusion_refiner.")]
        cross_dataset_expected = [
            key for key in shared_expected
            if not key.startswith(("inout_token.", "inout_head."))
        ]
        cross_dataset_loaded = [
            key for key in shared_loaded
            if not key.startswith(("inout_token.", "inout_head."))
        ]
        return {
            "checkpoint": checkpoint_path,
            "loaded": loaded_task,
            "loaded_task_tensors": len(loaded_task),
            "expected_task_tensors": len(expected_task),
            "coverage": len(loaded_task) / len(expected_task) if expected_task else 1.0,
            "shared_base_coverage": len(shared_loaded) / len(shared_expected) if shared_expected else 1.0,
            "cross_dataset_shared_coverage": (
                len(cross_dataset_loaded) / len(cross_dataset_expected)
                if cross_dataset_expected else 1.0
            ),
            "missing": missing,
            "new_branch_expected_tensors": len(new_branch_expected),
            "new_branch_loaded_tensors": len(new_branch_loaded),
            "new_branch_missing": new_branch_missing,
            "unexpected": unexpected,
            "ignored_source_keys": ignored_source_keys,
            "incompatible_shapes": incompatible_shapes,
        }
