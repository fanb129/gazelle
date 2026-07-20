"""Model construction and checkpoint safeguards for COTB experiments."""

from __future__ import annotations

from pathlib import Path

import torch

from AAAIScripts.common import file_manifest, strict_load_task_checkpoint


def build_model(
    model_source: str,
    model_name: str,
    spatial_prior: str = "none",
    fusion: str = "raw_concat",
    selected_layers: str = "all",
):
    if model_source == "v0":
        from gazelle.model_v0 import get_gazelle_model

        return get_gazelle_model(model_name)
    if model_source != "current":
        raise ValueError(f"unsupported model_source={model_source!r}")
    from gazelle.model import get_gazelle_model

    return get_gazelle_model(
        model_name,
        spatial_prior=spatial_prior,
        fusion=fusion,
        selected_layers=selected_layers,
    )


def load_evaluation_checkpoint(model, checkpoint: str | Path) -> dict:
    return strict_load_task_checkpoint(model, checkpoint)


def load_gazefollow_initialization(model, checkpoint: str | Path) -> dict:
    """Strictly load a matching GF task checkpoint into a VAT architecture.

    Only the VAT-specific in/out token and head may be absent.  This prevents a
    mislabeled fusion/spatial-prior checkpoint from silently initializing a
    different model.
    """

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a state dict, got {type(state)!r}")
    current = model.state_dict()
    expected = {key: value for key, value in current.items() if not key.startswith("backbone.")}
    provided = {key: value for key, value in state.items() if not key.startswith("backbone.")}
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(provided)
        if tuple(expected[key].shape) != tuple(provided[key].shape)
    )
    allowed_missing_prefixes = ("inout_token.", "inout_head.")
    unsafe_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if unexpected or mismatched or unsafe_missing:
        raise RuntimeError(
            "GazeFollow initialization does not match the requested VAT architecture: "
            f"unexpected={unexpected}, shape_mismatch={mismatched}, unsafe_missing={unsafe_missing}"
        )
    current.update(provided)
    model.load_state_dict(current, strict=True)
    return {
        "checkpoint": file_manifest(checkpoint),
        "loaded_task_tensors": len(provided),
        "allowed_missing": missing,
        "coverage_excluding_vat_head": len(provided) / max(1, len(expected) - len(missing)),
    }


def task_state_dict(model) -> dict[str, torch.Tensor]:
    if hasattr(model, "get_gazelle_state_dict"):
        return model.get_gazelle_state_dict(include_backbone=False)
    return {key: value for key, value in model.state_dict().items() if not key.startswith("backbone.")}
