"""Shared safeguards for AAAI diagnostic and experiment scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def strict_load_task_checkpoint(model, checkpoint: str | Path) -> dict[str, Any]:
    """Load every non-backbone tensor, failing on missing/unexpected/shape mismatch.

    Gazelle task checkpoints intentionally omit the frozen backbone.  The legacy
    loader uses ``strict=False`` and can silently leave a new variant partially
    random, which is unacceptable for diagnostic comparisons.
    """
    import torch

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a state dict, got {type(state)!r}")

    expected = {
        key: tensor
        for key, tensor in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    provided = {
        key: tensor
        for key, tensor in state.items()
        if not key.startswith("backbone.")
    }
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = {
        key: {
            "expected": list(expected[key].shape),
            "provided": list(provided[key].shape),
        }
        for key in sorted(set(expected) & set(provided))
        if tuple(expected[key].shape) != tuple(provided[key].shape)
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )

    full_state = model.state_dict()
    full_state.update(provided)
    model.load_state_dict(full_state, strict=True)
    return {
        "checkpoint": file_manifest(checkpoint),
        "expected_task_tensors": len(expected),
        "loaded_task_tensors": len(provided),
        "coverage": 1.0,
        "backbone_in_checkpoint": any(key.startswith("backbone.") for key in state),
    }
