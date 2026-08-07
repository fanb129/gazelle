"""Unified trainer for coverage-aware routing on GazeFollow and VAT.

The script intentionally keeps the two research stages explicit:

* ``support_pilot`` predicts and supervises a support map while DINOv3 still
  runs densely.  It validates the coverage/budget hypothesis but is not an
  efficiency result.
* ``backbone_sparse`` applies the selected image-level token union inside the
  DINOv3 backbone and therefore measures the actual routed model.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import posixpath
import random
import subprocess
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn

from gazelle.data_splits import grouped_holdout_split
from gazelle.dataloader import (
    GazeDataset,
    GazeFollowImageDataset,
    collate_fn,
    collate_gazefollow_images,
    load_data_gazefollow,
)
from gazelle.routing.losses import coverage_router_loss
from gazelle.routing.model import ROUTER_STAGES, get_coverage_router_model
from gazelle.utils import gazefollow_auc, gazefollow_l2, get_heatmap, vat_auc, vat_l2


CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_DATA_PATHS = {
    "gazefollow": "/newhome/fb/dataset/gazefollow_extended",
    "vat": "/newhome/fb/dataset/videoattentiontarget",
}


class WeightedMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def update(self, value, weight: float = 1.0) -> None:
        if value is None or weight <= 0:
            return
        numeric = float(value)
        if math.isfinite(numeric):
            self.total += numeric * float(weight)
            self.count += float(weight)

    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the coverage-aware gaze router."
    )
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--gazefollow_val_fraction",
        "--gf_val_fraction",
        dest="gazefollow_val_fraction",
        type=float,
        default=0.0,
        help=(
            "Hold out this fraction of GazeFollow training-image groups for "
            "validation. The default 0 keeps the legacy official-test eval."
        ),
    )
    parser.add_argument(
        "--gazefollow_split_seed",
        "--gf_split_seed",
        dest="gazefollow_split_seed",
        type=int,
        default=3106,
        help="Seed for the stable image-level GazeFollow validation split.",
    )
    parser.add_argument(
        "--vat_val_fraction",
        type=float,
        default=0.0,
        help=(
            "Hold out this fraction of VAT training sequences for validation. "
            "The default 0 keeps the legacy official-test-per-epoch mode."
        ),
    )
    parser.add_argument(
        "--vat_split_seed",
        type=int,
        default=3106,
        help="Seed for the stable sequence-level VAT validation split.",
    )
    parser.add_argument(
        "--formal_full_train_no_eval",
        action="store_true",
        help=(
            "Use every GazeFollow training sample for a fixed-epoch formal refit "
            "without constructing or reading any validation/test dataset. This "
            "mode saves final.pt and never creates a best-selection checkpoint."
        ),
    )
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--ckpt_save_dir", default="./experiments/coverage_router")
    parser.add_argument("--exp_name", default="coverage_router")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--init_ckpt", default=None)
    checkpoint_group.add_argument("--resume", default=None)
    parser.add_argument(
        "--reinitialize_router_on_init",
        action="store_true",
        help=(
            "When --init_ckpt is used, keep the current seed-specific router "
            "initialization instead of loading router.* tensors from the checkpoint. "
            "All other checkpoint tensors are still loaded."
        ),
    )
    parser.add_argument(
        "--allow_init_without_split_provenance",
        action="store_true",
        help=(
            "Allow a legacy pretrained --init_ckpt with no data_split metadata "
            "to initialize a formal holdout run. A checkpoint carrying a "
            "different formal split is still rejected."
        ),
    )

    parser.add_argument(
        "--router_stage", choices=ROUTER_STAGES, default="support_pilot"
    )
    parser.add_argument(
        "--route_after_block",
        type=int,
        default=None,
        help=(
            "Zero-based DINOv3 block index after which routing is applied. "
            "Defaults to 5 for ViT-B and 11 for ViT-L."
        ),
    )
    parser.add_argument(
        "--keep_ratio",
        "--router_keep_ratio",
        dest="keep_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument(
        "--escape_tokens",
        "--router_escape_tokens",
        dest="escape_tokens",
        type=int,
        default=8,
    )
    parser.add_argument("--router_coverage_weight", type=float, default=1.0)
    parser.add_argument("--router_budget_weight", type=float, default=0.05)
    parser.add_argument("--router_entropy_weight", type=float, default=0.0)
    parser.add_argument(
        "--router_warmup_epochs",
        type=int,
        default=0,
        help="Linearly move the sparse keep ratio from 1.0 to --keep_ratio over these epochs.",
    )
    parser.add_argument("--train_backbone_after_router", action="store_true")

    parser.add_argument("--spatial_prior", choices=("none",), default="none")
    parser.add_argument("--fusion", choices=("raw_concat",), default="raw_concat")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help=(
            "Evaluation images per batch. Defaults to --batch_size; use a "
            "smaller value for image-grouped multi-person validation."
        ),
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr_router", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_inout", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--inout_loss_lambda", type=float, default=1.0)
    parser.add_argument(
        "--heatmap_loss_weight",
        type=float,
        default=1.0,
        help="Set to 0 for a router-only support pilot.",
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--frame_sample_every", type=int, default=6)
    parser.add_argument("--eval_frame_sample_every", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--log_iter", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb_project", default="GazeRoute")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep_ratio must be in (0, 1]")
    if args.router_hidden_dim <= 0:
        raise ValueError("--router_hidden_dim must be positive")
    if args.router_temperature <= 0:
        raise ValueError("--router_temperature must be positive")
    if args.escape_tokens < 0:
        raise ValueError("--escape_tokens must be non-negative")
    if args.router_warmup_epochs < 0:
        raise ValueError("--router_warmup_epochs must be non-negative")
    if (
        args.batch_size <= 0
        or args.grad_accum_steps <= 0
        or (args.eval_batch_size is not None and args.eval_batch_size <= 0)
    ):
        raise ValueError(
            "--batch_size, --grad_accum_steps, and --eval_batch_size must be positive"
        )
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise ValueError("--max_epochs must be positive")
    if args.frame_sample_every <= 0:
        raise ValueError("--frame_sample_every must be positive")
    if args.eval_frame_sample_every is not None and args.eval_frame_sample_every <= 0:
        raise ValueError("--eval_frame_sample_every must be positive")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative")
    if args.log_iter <= 0:
        raise ValueError("--log_iter must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max_train_batches must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max_eval_batches must be positive")
    for name in ("lr_router", "lr_decoder", "lr_backbone", "lr_inout"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.heatmap_loss_weight < 0:
        raise ValueError("--heatmap_loss_weight must be non-negative")
    if not 0.0 <= args.gazefollow_val_fraction < 1.0:
        raise ValueError("--gazefollow_val_fraction must be in [0, 1)")
    if not 0.0 <= args.vat_val_fraction < 1.0:
        raise ValueError("--vat_val_fraction must be in [0, 1)")
    if args.dataset == "vat" and args.gazefollow_val_fraction:
        raise ValueError("--gazefollow_val_fraction is only valid for GazeFollow")
    if args.dataset == "gazefollow" and args.vat_val_fraction:
        raise ValueError("--vat_val_fraction is only valid for VAT")
    if args.formal_full_train_no_eval:
        if args.dataset not in (None, "gazefollow"):
            raise ValueError(
                "--formal_full_train_no_eval currently supports only GazeFollow"
            )
        if args.gazefollow_val_fraction or args.vat_val_fraction:
            raise ValueError(
                "--formal_full_train_no_eval cannot be combined with a validation split"
            )
        if args.max_eval_batches is not None or args.eval_batch_size is not None:
            raise ValueError(
                "--formal_full_train_no_eval cannot accept evaluation batch options"
            )
        if args.max_train_batches is not None:
            raise ValueError(
                "--formal_full_train_no_eval must use the complete training set"
            )
        if args.save_every:
            raise ValueError(
                "--formal_full_train_no_eval only saves last.resume.pt and final.pt"
            )
        if args.max_epochs is None and not args.resume:
            raise ValueError(
                "--formal_full_train_no_eval requires an explicit --max_epochs; "
                "the final epoch is fixed before training"
            )
    if args.reinitialize_router_on_init and not args.init_ckpt:
        raise ValueError("--reinitialize_router_on_init requires --init_ckpt")
    if args.allow_init_without_split_provenance and not args.init_ckpt:
        raise ValueError("--allow_init_without_split_provenance requires --init_ckpt")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument
        return torch.load(path, map_location="cpu")


def _is_tensor_state_dict(payload) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in payload.items()
        )
    )


def checkpoint_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Store every task head plus every trainable DINO parameter.

    Frozen DINO weights are reconstructed from the repository's official local
    pretrain checkpoint.  This keeps ViT-L checkpoints manageable without ever
    dropping a fine-tuned suffix block.
    """
    trainable_backbone = {
        name
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.") or name in trainable_backbone
    }


def load_model_state(
    model: nn.Module,
    payload,
    *,
    initialization: bool,
    excluded_prefixes: Iterable[str] = (),
) -> None:
    excluded_prefixes = tuple(excluded_prefixes)
    if excluded_prefixes and not initialization:
        raise ValueError(
            "checkpoint key exclusions are only allowed for initialization"
        )

    if _is_tensor_state_dict(payload):
        state = payload
    elif isinstance(payload, dict) and isinstance(payload.get("model_state"), dict):
        state = payload["model_state"]
    else:
        raise ValueError(
            "checkpoint is neither a legacy state_dict nor a coverage-router checkpoint"
        )

    excluded_keys = {
        key for key in state if excluded_prefixes and key.startswith(excluded_prefixes)
    }
    if excluded_keys:
        state = {key: value for key, value in state.items() if key not in excluded_keys}
        print(
            "Initialization keeps current parameters for "
            f"{len(excluded_keys)} checkpoint tensors matching {excluded_prefixes}"
        )

    incompatible = model.load_state_dict(state, strict=False)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_required = [
        key
        for key in incompatible.missing_keys
        if key not in excluded_keys
        and (not key.startswith("backbone.") or key in trainable_names)
    ]
    if not initialization and missing_required:
        raise RuntimeError(
            f"resume checkpoint is missing model state: {missing_required}"
        )
    if incompatible.unexpected_keys:
        print(f"WARNING: unused checkpoint keys: {incompatible.unexpected_keys}")
    if initialization and missing_required:
        print(
            f"Initialization leaves new task parameters at their defaults: {missing_required}"
        )


def _resume_model_overrides(args: argparse.Namespace, checkpoint: dict) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or missing checkpoint format_version")
    model_config = checkpoint.get("model_config")
    train_config = checkpoint.get("train_config", {})
    if not isinstance(model_config, dict):
        raise ValueError("resume checkpoint has no model_config")
    checkpoint_dataset = train_config.get("dataset")
    if (
        checkpoint_dataset is not None
        and args.dataset is not None
        and args.dataset != checkpoint_dataset
    ):
        raise ValueError(
            f"--dataset={args.dataset} conflicts with resume dataset={checkpoint_dataset}"
        )
    if checkpoint_dataset is not None:
        args.dataset = checkpoint_dataset
    for key in (
        "model",
        "router_stage",
        "route_after_block",
        "keep_ratio",
        "router_hidden_dim",
        "router_temperature",
        "escape_tokens",
        "spatial_prior",
        "fusion",
    ):
        if key in model_config:
            setattr(args, key, model_config[key])
    if "train_backbone_after_router" in train_config:
        args.train_backbone_after_router = bool(
            train_config["train_backbone_after_router"]
        )
    # Resume preserves every optimization/data-sampling decision that changes
    # the mathematical run. Runtime-only settings (device, workers, W&B and
    # output location) remain overridable.
    for key in (
        "seed",
        "batch_size",
        "grad_accum_steps",
        "lr_router",
        "lr_decoder",
        "lr_inout",
        "lr_backbone",
        "weight_decay",
        "clip_grad_norm",
        "heatmap_loss_weight",
        "inout_loss_lambda",
        "router_coverage_weight",
        "router_budget_weight",
        "router_entropy_weight",
        "router_warmup_epochs",
        "gazefollow_val_fraction",
        "gazefollow_split_seed",
        "vat_val_fraction",
        "vat_split_seed",
        "eval_batch_size",
        "frame_sample_every",
        "eval_frame_sample_every",
        "amp",
        "formal_full_train_no_eval",
    ):
        if key in train_config:
            setattr(args, key, train_config[key])
    if args.data_path is None and train_config.get("data_path"):
        args.data_path = train_config["data_path"]
    if args.max_epochs is None and train_config.get("max_epochs"):
        args.max_epochs = int(train_config["max_epochs"])


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model is None:
        return (
            "gazelle_dinov3_vitb16_inout"
            if args.dataset == "vat"
            else "gazelle_dinov3_vitb16"
        )
    has_inout = args.model.endswith("_inout")
    if args.dataset == "vat" and not has_inout:
        raise ValueError("VAT requires an *_inout model")
    if args.dataset == "gazefollow" and has_inout:
        raise ValueError("GazeFollow requires a model without the _inout head")
    return args.model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gazefollow_group_key(record: dict, index: int) -> str:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"GazeFollow record {index} has no non-empty string 'path' group key"
        )
    normalized = posixpath.normpath(raw_path.strip().replace("\\", "/"))
    if normalized in ("", "."):
        raise ValueError(f"GazeFollow record {index} has an invalid image path")
    return normalized


def _full_train_group_metadata(group_keys: Iterable[str]) -> dict:
    """Fingerprint an immutable all-train assignment without making a split."""

    normalized = tuple(str(group) for group in group_keys)
    if not normalized:
        raise ValueError("formal full-train data must contain at least one group")
    unique_groups = sorted(set(normalized))

    def fingerprint(lines: Iterable[str]) -> str:
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    return {
        "source_group_fingerprint": fingerprint(
            f"{index}\0{group}" for index, group in enumerate(normalized)
        ),
        "train_group_fingerprint": fingerprint(unique_groups),
        "assignment_fingerprint": fingerprint(
            f"{index}\0{group}\0train" for index, group in enumerate(normalized)
        ),
        "train_group_count": len(unique_groups),
    }


def _vat_source_video_key(sequence: dict, index: int) -> str:
    raw_path = sequence.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"VAT sequence {index} has no non-empty string 'path' group key"
        )
    normalized_clip = posixpath.normpath(raw_path.strip().replace("\\", "/"))
    normalized_source_video = posixpath.dirname(normalized_clip)
    if normalized_clip in ("", ".") or normalized_source_video in ("", "."):
        raise ValueError(f"VAT sequence {index} has an invalid path")
    return normalized_source_video


def _flatten_vat_sequences(sequences, indices, *, sample_rate: int) -> list[dict]:
    frames = []
    for sequence_index in indices:
        sequence = sequences[int(sequence_index)]
        sequence_frames = sequence.get("frames")
        if not isinstance(sequence_frames, list):
            raise ValueError(
                f"VAT sequence {sequence_index} has no list-valued 'frames'"
            )
        frames.extend(sequence_frames[::sample_rate])
    return frames


def make_dataloaders(args: argparse.Namespace, transform):
    common = {
        "num_workers": args.n_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.n_workers > 0,
    }
    if args.dataset == "gazefollow":
        if getattr(args, "formal_full_train_no_eval", False):
            source_path = Path(args.data_path) / "train_preprocessed.json"
            records = load_data_gazefollow(source_path)
            group_keys = [
                _gazefollow_group_key(record, index)
                for index, record in enumerate(records)
            ]
            train_dataset = GazeDataset(
                "gazefollow",
                args.data_path,
                "train",
                transform,
                augment=True,
                return_heatmap=True,
                records=records,
            )
            eval_dataset = None
            eval_collate_fn = None
            data_split = {
                "strategy": "gazefollow_full_train_no_eval_v1",
                "evaluation_split": None,
                "evaluation_mode": "none",
                "selection_policy": "fixed_final_epoch_no_validation",
                "selection_is_formal": False,
                "provenance_requires_match": True,
                "official_test_accessed": False,
                "validation_fraction": 0.0,
                "group_key": "normalized_image_path",
                "source_annotation_file": str(source_path.resolve()),
                "source_annotation_sha256": _sha256_file(source_path),
                "train_record_count": len(records),
                "train_head_sample_count": len(train_dataset),
                "validation_record_count": 0,
                "validation_head_sample_count": 0,
                **_full_train_group_metadata(group_keys),
            }
        elif args.gazefollow_val_fraction:
            source_path = Path(args.data_path) / "train_preprocessed.json"
            records = load_data_gazefollow(source_path)
            group_keys = [
                _gazefollow_group_key(record, index)
                for index, record in enumerate(records)
            ]
            split = grouped_holdout_split(
                group_keys,
                validation_fraction=args.gazefollow_val_fraction,
                seed=args.gazefollow_split_seed,
            )
            train_dataset = GazeDataset(
                "gazefollow",
                args.data_path,
                "train",
                transform,
                image_indices=split.train_indices,
                augment=True,
                return_heatmap=True,
                records=records,
            )
            eval_dataset = GazeFollowImageDataset(
                args.data_path,
                "train",
                transform,
                image_indices=split.validation_indices,
                records=records,
            )
            eval_collate_fn = collate_gazefollow_images
            data_split = {
                **split.metadata(),
                "evaluation_split": "gazefollow_train_holdout",
                "selection_is_formal": True,
                "group_key": "normalized_image_path",
                "source_annotation_file": str(source_path.resolve()),
                "source_annotation_sha256": _sha256_file(source_path),
                "train_head_sample_count": len(train_dataset),
                "validation_image_count": len(eval_dataset),
                "validation_head_sample_count": eval_dataset.person_count,
                "validation_query_unit": "image_with_all_heads",
            }
        else:
            train_dataset = GazeDataset(
                "gazefollow", args.data_path, "train", transform
            )
            eval_dataset = GazeDataset("gazefollow", args.data_path, "test", transform)
            eval_collate_fn = collate_fn
            train_source = Path(args.data_path) / "train_preprocessed.json"
            eval_source = Path(args.data_path) / "test_preprocessed.json"
            data_split = {
                "strategy": "legacy_official_test_per_epoch",
                "evaluation_split": "gazefollow_official_test",
                "selection_is_formal": False,
                "warning": (
                    "Official test is evaluated every epoch; checkpoints from "
                    "this mode are exploratory and must not be reported as "
                    "validation-selected paper results."
                ),
                "train_annotation_file": str(train_source.resolve()),
                "train_annotation_sha256": _sha256_file(train_source),
                "eval_annotation_file": str(eval_source.resolve()),
                "eval_annotation_sha256": _sha256_file(eval_source),
                "train_head_sample_count": len(train_dataset),
                "eval_head_sample_count": len(eval_dataset),
            }
    else:
        eval_rate = args.eval_frame_sample_every or args.frame_sample_every
        if args.vat_val_fraction:
            source_path = Path(args.data_path) / "train_preprocessed.json"
            sequences = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(sequences, list):
                raise ValueError("VAT train_preprocessed.json must contain a list")
            source_video_keys = [
                _vat_source_video_key(sequence, index)
                for index, sequence in enumerate(sequences)
            ]
            split = grouped_holdout_split(
                source_video_keys,
                validation_fraction=args.vat_val_fraction,
                seed=args.vat_split_seed,
            )
            train_frames = _flatten_vat_sequences(
                sequences,
                split.train_indices,
                sample_rate=args.frame_sample_every,
            )
            validation_frames = _flatten_vat_sequences(
                sequences,
                split.validation_indices,
                sample_rate=eval_rate,
            )
            train_dataset = GazeDataset(
                "videoattentiontarget",
                args.data_path,
                "train",
                transform,
                in_frame_only=False,
                sample_rate=args.frame_sample_every,
                augment=True,
                return_heatmap=True,
                records=train_frames,
            )
            eval_dataset = GazeDataset(
                "videoattentiontarget",
                args.data_path,
                "train",
                transform,
                in_frame_only=False,
                sample_rate=eval_rate,
                augment=False,
                return_heatmap=False,
                records=validation_frames,
            )
            data_split = {
                **split.metadata(),
                "evaluation_split": "vat_train_source_video_holdout",
                "selection_is_formal": True,
                "group_key": "normalized_source_video_path",
                "source_annotation_file": str(source_path.resolve()),
                "source_annotation_sha256": _sha256_file(source_path),
                "train_sequence_count": len(split.train_indices),
                "validation_sequence_count": len(split.validation_indices),
                "train_frame_count": len(train_frames),
                "validation_frame_count": len(validation_frames),
                "train_head_sample_count": len(train_dataset),
                "validation_head_sample_count": len(eval_dataset),
                "train_frame_sample_every": args.frame_sample_every,
                "eval_frame_sample_every": eval_rate,
            }
        else:
            train_dataset = GazeDataset(
                "videoattentiontarget",
                args.data_path,
                "train",
                transform,
                in_frame_only=False,
                sample_rate=args.frame_sample_every,
            )
            eval_dataset = GazeDataset(
                "videoattentiontarget",
                args.data_path,
                "test",
                transform,
                in_frame_only=False,
                sample_rate=eval_rate,
            )
            train_source = Path(args.data_path) / "train_preprocessed.json"
            eval_source = Path(args.data_path) / "test_preprocessed.json"
            data_split = {
                "strategy": "vat_official_train_test",
                "evaluation_split": "vat_official_test",
                "selection_is_formal": False,
                "warning": (
                    "VAT test is evaluated every epoch; use a grouped validation "
                    "protocol before treating checkpoint selection as formal."
                ),
                "train_annotation_file": str(train_source.resolve()),
                "train_annotation_sha256": _sha256_file(train_source),
                "eval_annotation_file": str(eval_source.resolve()),
                "eval_annotation_sha256": _sha256_file(eval_source),
                "train_frame_sample_every": args.frame_sample_every,
                "eval_frame_sample_every": eval_rate,
            }
        eval_collate_fn = collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **common,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=eval_collate_fn,
            **common,
        )
    return train_dataset, eval_dataset, train_loader, eval_loader, data_split


def parameter_group_name(name: str) -> str:
    if name.startswith("router."):
        return "router"
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("inout_head.") or name.startswith("inout_token."):
        return "inout"
    return "decoder"


def apply_zero_lr_freezing(model: nn.Module, args: argparse.Namespace) -> None:
    learning_rates = {
        "router": args.lr_router,
        "decoder": args.lr_decoder,
        "inout": args.lr_inout,
        "backbone": args.lr_backbone,
    }
    for name, parameter in model.named_parameters():
        if learning_rates[parameter_group_name(name)] == 0.0:
            parameter.requires_grad = False


def make_optimizer(model: nn.Module, args: argparse.Namespace):
    groups = {"router": [], "decoder": [], "inout": [], "backbone": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[parameter_group_name(name)].append(parameter)

    learning_rates = {
        "router": args.lr_router,
        "decoder": args.lr_decoder,
        "inout": args.lr_inout,
        "backbone": args.lr_backbone,
    }
    param_groups = [
        {"params": parameters, "lr": learning_rates[name], "name": name}
        for name, parameters in groups.items()
        if parameters
    ]
    if not param_groups:
        raise RuntimeError("the selected configuration has no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    return optimizer, counts


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def current_keep_ratio(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return target
    if warmup_epochs == 1:
        return target
    # Include both endpoints: the first curriculum epoch is fully dense and
    # the last warmup epoch reaches the requested sparse budget.
    progress = min(1.0, float(epoch) / float(warmup_epochs - 1))
    return 1.0 + progress * (target - 1.0)


def unpack_batch(batch, device: torch.device, *, training: bool):
    if training:
        images, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch
        heatmaps = heatmaps.to(device, non_blocking=True)
    else:
        images, bboxes, gazex, gazey, inout, heights, widths = batch
        heatmaps = None
    image_grouped = bool(
        bboxes
        and isinstance(bboxes[0], (list, tuple))
        and bboxes[0]
        and isinstance(bboxes[0][0], (list, tuple, torch.Tensor))
    )
    model_input = {
        "images": images.to(device, non_blocking=True),
        "bboxes": bboxes if image_grouped else [[bbox] for bbox in bboxes],
        "_query_unit": "image" if image_grouped else "person",
    }
    return (
        model_input,
        gazex,
        gazey,
        inout.to(device, non_blocking=True),
        heights,
        widths,
        heatmaps,
    )


def stack_predictions(predictions: dict):
    heatmaps = torch.cat(predictions["heatmap"], dim=0)
    inout = None
    if predictions.get("inout") is not None:
        inout = torch.cat(predictions["inout"], dim=0)
    return heatmaps, inout


def coordinate_heatmaps(gazex, gazey, inout: torch.Tensor) -> torch.Tensor:
    """Build evaluation targets, averaging all valid GF annotations."""
    maps = []
    flags = inout.detach().cpu().bool().tolist()
    for xs, ys, is_in_frame in zip(gazex, gazey, flags):
        target = torch.zeros(64, 64)
        valid_maps = []
        if is_in_frame:
            for x, y in zip(xs, ys):
                if float(x) >= 0.0 and float(y) >= 0.0:
                    valid_maps.append(get_heatmap(float(x), float(y), 64, 64))
        if valid_maps:
            target = torch.stack(valid_maps).mean(dim=0)
        maps.append(target)
    return torch.stack(maps)


def exact_point_coverage(routing, gazex, gazey, inout: torch.Tensor):
    """Return exact GT patch recall in the discrete routed token set."""
    hard_masks = routing.image_hard_masks.detach().cpu()
    person_to_image = routing.person_to_image.detach().cpu().tolist()
    flags = inout.detach().cpu().bool().tolist()
    covered = 0.0
    point_count = 0
    for person_index, (xs, ys, is_in_frame) in enumerate(zip(gazex, gazey, flags)):
        if not is_in_frame:
            continue
        image_index = person_to_image[person_index]
        height, width = hard_masks.shape[-2:]
        for x, y in zip(xs, ys):
            x = float(x)
            y = float(y)
            if x < 0.0 or y < 0.0:
                continue
            patch_x = max(0, min(width - 1, int(x * width)))
            patch_y = max(0, min(height - 1, int(y * height)))
            covered += float(hard_masks[image_index, patch_y, patch_x].item())
            point_count += 1
    return (covered / point_count if point_count else None), point_count


def routing_metrics(routing, target_heatmaps, inout, args) -> dict:
    return coverage_router_loss(
        fp32_routing(routing),
        target_heatmaps,
        inout=inout,
        coverage_weight=args.router_coverage_weight,
        budget_weight=args.router_budget_weight,
        entropy_weight=args.router_entropy_weight,
        budget_target=model_keep_ratio(routing, args),
    )


def fp32_routing(routing):
    """Keep probability/entropy losses numerically stable under AMP."""
    return replace(
        routing,
        support_logits=routing.support_logits.float(),
        support_probs=routing.support_probs.float(),
        image_union_probs=routing.image_union_probs.float(),
    )


def model_keep_ratio(routing, args) -> float:
    # During evaluation this is the configured target. During training the
    # caller updates model.router.keep_ratio and actual_keep_ratio reflects the
    # warmup schedule; head/escape priorities never change the fixed K.
    return float(getattr(args, "_active_keep_ratio", args.keep_ratio))


def evaluate_model(
    model: nn.Module,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    auc_meter = WeightedMean()
    l2_meter = WeightedMean()
    min_l2_meter = WeightedMean()
    route_soft = WeightedMean()
    route_hard = WeightedMean()
    route_point = WeightedMean()
    route_support = WeightedMean()
    route_keep = WeightedMean()
    inout_predictions = []
    inout_targets = []
    sample_count = 0
    image_count = 0
    inframe_count = 0
    query_units = set()
    gazefollow_strata = {
        name: {
            "auc": WeightedMean(),
            "avg_l2": WeightedMean(),
            "min_l2": WeightedMean(),
        }
        for name in ("single_head", "multi_head", "two_head", "three_plus_head")
    }

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            model_input, gazex, gazey, inout, heights, widths, _ = unpack_batch(
                batch, device, training=False
            )
            with autocast_context(device, args.amp):
                predictions = model(model_input)
            heatmap_predictions, inout_scores = stack_predictions(predictions)
            batch_size = heatmap_predictions.shape[0]
            batch_image_count = int(model_input["images"].shape[0])
            sample_count += batch_size
            image_count += batch_image_count
            query_units.add(model_input["_query_unit"])

            if args.dataset == "gazefollow":
                routing = predictions["routing"]
                people_per_image = torch.bincount(
                    routing.person_to_image.detach().cpu(),
                    minlength=batch_image_count,
                ).tolist()
                head_counts = [
                    people_per_image[image_index]
                    for image_index in routing.person_to_image.detach().cpu().tolist()
                ]
                for index in range(batch_size):
                    heatmap = heatmap_predictions[index].float().cpu()
                    auc_value = gazefollow_auc(
                        heatmap,
                        gazex[index],
                        gazey[index],
                        heights[index],
                        widths[index],
                    )
                    auc_meter.update(auc_value)
                    avg_l2, min_l2 = gazefollow_l2(heatmap, gazex[index], gazey[index])
                    l2_meter.update(avg_l2)
                    min_l2_meter.update(min_l2)
                    if model_input["_query_unit"] == "image":
                        count = head_counts[index]
                        stratum_names = (
                            ("single_head",)
                            if count == 1
                            else (
                                ("multi_head", "two_head")
                                if count == 2
                                else ("multi_head", "three_plus_head")
                            )
                        )
                        for name in stratum_names:
                            gazefollow_strata[name]["auc"].update(auc_value)
                            gazefollow_strata[name]["avg_l2"].update(avg_l2)
                            gazefollow_strata[name]["min_l2"].update(min_l2)
                inframe_count += batch_size
            else:
                flags = inout.detach().cpu().bool()
                for index in range(batch_size):
                    if flags[index]:
                        heatmap = heatmap_predictions[index].float().cpu()
                        auc_meter.update(
                            vat_auc(heatmap, gazex[index][0], gazey[index][0])
                        )
                        l2_meter.update(
                            vat_l2(heatmap, gazex[index][0], gazey[index][0])
                        )
                        inframe_count += 1
                if inout_scores is not None:
                    inout_predictions.extend(inout_scores.float().cpu().tolist())
                    inout_targets.extend(inout.float().cpu().tolist())

            targets = coordinate_heatmaps(gazex, gazey, inout).to(device)
            route_values = coverage_router_loss(
                fp32_routing(predictions["routing"]),
                targets,
                inout=inout,
                coverage_weight=0.0,
                budget_weight=0.0,
                entropy_weight=0.0,
                budget_target=args.keep_ratio,
            )
            valid_people = int(inout.bool().sum().item())
            route_soft.update(route_values["soft_coverage"].item(), valid_people)
            route_hard.update(route_values["hard_coverage"].item(), valid_people)
            point_coverage, point_count = exact_point_coverage(
                predictions["routing"], gazex, gazey, inout
            )
            route_point.update(point_coverage, point_count)
            route_support.update(route_values["mean_support"].item(), batch_image_count)
            route_keep.update(
                predictions["routing"].actual_keep_ratio, batch_image_count
            )

    metrics = {
        "dataset": args.dataset,
        "query_unit": (next(iter(query_units)) if len(query_units) == 1 else "mixed"),
        "image_count": image_count,
        "sample_count": sample_count,
        "inframe_count": inframe_count,
        "auc": auc_meter.mean(),
        "l2": l2_meter.mean(),
        "routing_soft_coverage": route_soft.mean(),
        "routing_hard_coverage": route_hard.mean(),
        "routing_gt_point_coverage": route_point.mean(),
        "routing_mean_support": route_support.mean(),
        "routing_actual_keep_ratio": route_keep.mean(),
    }
    if args.dataset == "gazefollow":
        metrics["avg_l2"] = metrics.pop("l2")
        metrics["min_l2"] = min_l2_meter.mean()
        if "image" in query_units:
            metrics["head_count_strata"] = {
                name: {
                    "sample_count": int(values["auc"].count),
                    "auc": values["auc"].mean(),
                    "avg_l2": values["avg_l2"].mean(),
                    "min_l2": values["min_l2"].mean(),
                }
                for name, values in gazefollow_strata.items()
            }
    elif inout_targets and len(set(inout_targets)) > 1:
        metrics["inout_ap"] = float(
            average_precision_score(inout_targets, inout_predictions)
        )
    else:
        metrics["inout_ap"] = None
    return metrics


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _split_identity(data_split: dict) -> dict:
    keys = (
        "strategy",
        "evaluation_split",
        "evaluation_mode",
        "selection_policy",
        "provenance_requires_match",
        "official_test_accessed",
        "source_annotation_sha256",
        "source_group_fingerprint",
        "assignment_fingerprint",
        "validation_fraction",
        "seed",
        "group_key",
        "train_frame_sample_every",
        "eval_frame_sample_every",
    )
    return {key: data_split.get(key) for key in keys}


def split_requires_matching_provenance(data_split: dict) -> bool:
    return bool(
        data_split.get("selection_is_formal")
        or data_split.get("provenance_requires_match")
    )


def validate_checkpoint_split_provenance(
    checkpoint: dict,
    data_split: dict,
    *,
    checkpoint_role: str,
    allow_missing_provenance: bool = False,
) -> None:
    """Fail fast when a formal data protocol would cross provenance."""

    if not split_requires_matching_provenance(data_split):
        return
    checkpoint_split = checkpoint.get("data_split")
    if not isinstance(checkpoint_split, dict):
        if allow_missing_provenance:
            print(
                f"WARNING: {checkpoint_role} has no data_split provenance; "
                "using it only as an explicitly allowed pretrained initialization"
            )
            return
        raise ValueError(
            f"{checkpoint_role} has no data_split provenance. A formal run "
            "must start from scratch, from an explicitly allowed legacy "
            "pretrained initialization, or from a checkpoint produced with "
            "the exact same data assignment."
        )
    expected = _split_identity(data_split)
    observed = _split_identity(checkpoint_split)
    if observed != expected:
        raise ValueError(
            f"{checkpoint_role} data split does not match the current holdout: "
            f"checkpoint={observed}, current={expected}"
        )


def selection_spec(
    args: argparse.Namespace,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    if args.router_stage == "support_pilot" and args.heatmap_loss_weight == 0.0:
        return (
            "routing_hard_coverage",
            "max",
            (("routing_gt_point_coverage", "max"), ("auc", "max")),
        )
    if args.dataset == "gazefollow":
        return "avg_l2", "min", (("auc", "max"),)
    return "l2", "min", (("auc", "max"), ("inout_ap", "max"))


def _metric_is_better(candidate, incumbent, mode: str) -> Optional[bool]:
    if candidate is None or not math.isfinite(float(candidate)):
        return False
    if incumbent is None or not math.isfinite(float(incumbent)):
        return True
    candidate = float(candidate)
    incumbent = float(incumbent)
    if math.isclose(candidate, incumbent, rel_tol=0.0, abs_tol=1e-12):
        return None
    return candidate < incumbent if mode == "min" else candidate > incumbent


def selection_improved(eval_metrics: dict, selection: dict) -> bool:
    comparison = _metric_is_better(
        eval_metrics.get(selection["metric"]),
        selection.get("value"),
        selection["mode"],
    )
    if comparison is not None:
        return comparison
    incumbent_metrics = selection.get("metrics") or {}
    for metric, mode in selection["tie_breakers"]:
        comparison = _metric_is_better(
            eval_metrics.get(metric), incumbent_metrics.get(metric), mode
        )
        if comparison is not None:
            return comparison
    # Exact ties keep the earlier checkpoint.
    return False


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    *,
    epoch: int,
    global_step: int,
    metrics: dict,
    best_metrics: Optional[dict],
    data_split: dict,
    selection: Optional[dict],
    checkpoint_role: str = "training_checkpoint",
    initialization_checkpoint: Optional[dict] = None,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": checkpoint_role,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model.get_model_config(),
        "train_config": vars(args).copy(),
        "state_scope": "all_non_backbone_and_all_trainable_backbone_parameters",
        "model_state": checkpoint_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": metrics,
        "best_metrics": best_metrics.copy() if best_metrics is not None else None,
        "data_split": data_split.copy(),
        "selection": selection.copy() if selection is not None else None,
        "initialization_checkpoint": (
            initialization_checkpoint.copy()
            if initialization_checkpoint is not None
            else None
        ),
        "rng_state": rng_state(),
        "git_commit": git_commit(),
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"Saved checkpoint to {path}")


def reconcile_formal_resume_history(run_dir: Path, checkpoint: dict) -> None:
    """Drop only epoch rows that are newer than the atomic resume checkpoint.

    Formal training appends an epoch's JSON row before atomically replacing
    ``last.resume.pt``.  A process interruption in that narrow interval leaves
    one uncommitted row behind.  The checkpoint is the commit boundary: its
    epoch prefix must be intact and identical, while any later rows are safely
    replayed after truncation.
    """

    checkpoint_epoch = checkpoint.get("epoch")
    if not isinstance(checkpoint_epoch, int) or isinstance(checkpoint_epoch, bool):
        raise ValueError("formal resume checkpoint epoch must be an integer")
    expected_rows = checkpoint_epoch + 1
    if expected_rows <= 0:
        raise ValueError("formal resume checkpoint epoch must be non-negative")

    history_path = run_dir / "history.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"formal resume checkpoint has no history.jsonl beside it: {history_path}"
        )
    lines = [
        line
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < expected_rows:
        raise ValueError(
            "formal resume history ends before the checkpoint epoch: "
            f"rows={len(lines)}, required={expected_rows}"
        )

    committed_rows = []
    for index, line in enumerate(lines[:expected_rows]):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"formal resume history row {index} is invalid before the "
                "checkpoint boundary"
            ) from error
        actual_epoch = row.get("epoch") if isinstance(row, dict) else None
        if actual_epoch != index:
            raise ValueError(
                "formal resume history has a non-contiguous committed prefix: "
                f"row={index}, epoch={actual_epoch}"
            )
        committed_rows.append(row)

    if checkpoint.get("metrics") != committed_rows[-1]:
        raise ValueError(
            "formal resume checkpoint metrics do not match its committed history row"
        )

    has_uncommitted_suffix = len(lines) > expected_rows
    if has_uncommitted_suffix:
        for index, line in enumerate(lines[expected_rows:], start=expected_rows):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partially written trailing row is precisely the crash case
                # this reconciliation is designed to recover.
                break
            if (
                not isinstance(row, dict)
                or row.get("epoch", expected_rows) < expected_rows
            ):
                raise ValueError(
                    "formal resume history suffix overlaps the committed prefix: "
                    f"row={index}"
                )

        temporary = history_path.with_suffix(history_path.suffix + ".resume.tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
                for row in committed_rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, history_path)
        print(
            "Recovered formal history by dropping rows newer than "
            f"last.resume.pt epoch {checkpoint_epoch}"
        )


def init_wandb(args: argparse.Namespace, model_config: dict):
    try:
        import wandb
    except ModuleNotFoundError:
        if args.wandb_mode == "disabled":
            return None
        raise
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        mode=args.wandb_mode,
        config={**vars(args), "model_config": model_config},
    )
    return wandb


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    requested_max_epochs = args.max_epochs
    resume_payload = _torch_load(args.resume) if args.resume else None
    init_payload = _torch_load(args.init_ckpt) if args.init_ckpt else None
    if resume_payload is not None:
        if not isinstance(resume_payload, dict):
            raise ValueError(
                "--resume requires a structured coverage-router checkpoint"
            )
        _resume_model_overrides(args, resume_payload)
        resume_train_config = resume_payload.get("train_config", {})
        if resume_train_config.get("formal_full_train_no_eval"):
            checkpoint_max_epochs = int(resume_train_config["max_epochs"])
            if (
                requested_max_epochs is not None
                and requested_max_epochs != checkpoint_max_epochs
            ):
                raise ValueError(
                    "a formal full-train run cannot change --max_epochs on resume: "
                    f"checkpoint={checkpoint_max_epochs}, requested={requested_max_epochs}"
                )
            args.max_epochs = checkpoint_max_epochs

    args.dataset = args.dataset or "gazefollow"
    args.model = resolve_model_name(args)
    args.data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    args.max_epochs = args.max_epochs or (8 if args.dataset == "vat" else 15)
    if not args.formal_full_train_no_eval:
        args.eval_frame_sample_every = (
            args.eval_frame_sample_every or args.frame_sample_every
        )
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model, transform = get_coverage_router_model(
        args.model,
        router_stage=args.router_stage,
        route_after_block=args.route_after_block,
        keep_ratio=args.keep_ratio,
        router_hidden_dim=args.router_hidden_dim,
        router_temperature=args.router_temperature,
        escape_tokens=args.escape_tokens,
        spatial_prior=args.spatial_prior,
        fusion=args.fusion,
    )
    args.route_after_block = model.route_after_block
    model.set_trainable_backbone_suffix(args.train_backbone_after_router)
    apply_zero_lr_freezing(model, args)

    if init_payload is not None:
        print(f"Initializing model weights from {args.init_ckpt}")
        excluded_prefixes = ("router.",) if args.reinitialize_router_on_init else ()
        load_model_state(
            model,
            init_payload,
            initialization=True,
            excluded_prefixes=excluded_prefixes,
        )
    elif resume_payload is not None:
        load_model_state(model, resume_payload, initialization=False)

    model.to(device)
    (
        train_dataset,
        eval_dataset,
        train_loader,
        eval_loader,
        data_split,
    ) = make_dataloaders(args, transform)
    if resume_payload is not None:
        validate_checkpoint_split_provenance(
            resume_payload,
            data_split,
            checkpoint_role="resume checkpoint",
        )
    if init_payload is not None and split_requires_matching_provenance(data_split):
        if not isinstance(init_payload, dict):
            raise ValueError(
                "initialization checkpoint is not structured and has no "
                "verifiable data_split provenance"
            )
        validate_checkpoint_split_provenance(
            init_payload,
            data_split,
            checkpoint_role="initialization checkpoint",
            allow_missing_provenance=args.allow_init_without_split_provenance,
        )
    optimizer, parameter_counts = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=0.0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    bce = nn.BCELoss()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        run_dir = Path(args.resume).resolve().parent
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(args.ckpt_save_dir) / args.exp_name / timestamp
    if (
        args.formal_full_train_no_eval
        and not args.resume
        and run_dir.exists()
        and any(run_dir.iterdir())
    ):
        raise FileExistsError(
            "formal full-train run_dir must be absent or empty so history and "
            f"fixed-final artifacts cannot be mixed: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.init_ckpt:
        initialization_checkpoint = {
            "path": str(Path(args.init_ckpt).resolve()),
            "sha256": _sha256_file(Path(args.init_ckpt)),
        }
    elif resume_payload is not None:
        resumed_initialization = resume_payload.get("initialization_checkpoint")
        initialization_checkpoint = (
            resumed_initialization.copy()
            if isinstance(resumed_initialization, dict)
            else None
        )
    else:
        initialization_checkpoint = None
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "train_sample_count": len(train_dataset),
        "eval_sample_count": (
            None
            if eval_dataset is None
            else (
                eval_dataset.person_count
                if isinstance(eval_dataset, GazeFollowImageDataset)
                else len(eval_dataset)
            )
        ),
        "eval_image_count": (
            len(eval_dataset)
            if isinstance(eval_dataset, GazeFollowImageDataset)
            else None
        ),
        "data_split": data_split,
        "model_config": model.get_model_config(),
        "train_config": vars(args),
        "initialization_checkpoint": initialization_checkpoint,
        "trainable_parameters": parameter_counts,
        "git_commit": git_commit(),
        "note": (
            "formal full-train refit; no evaluation dataset is constructed"
            if args.formal_full_train_no_eval
            else (
                "support_pilot runs the full DINOv3 backbone and is not an efficiency result"
                if args.router_stage == "support_pilot"
                else "backbone_sparse routes patch tokens before the DINOv3 suffix"
            )
        ),
    }
    manifest_path = run_dir / "run_manifest.json"
    split_path = run_dir / "data_split.json"
    if args.formal_full_train_no_eval and resume_payload is not None:
        if not manifest_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                "formal resume requires the original run_manifest.json and "
                "data_split.json beside last.resume.pt"
            )
        original_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_split = json.loads(split_path.read_text(encoding="utf-8"))
        if (
            original_split != data_split
            or original_metadata.get("data_split") != data_split
        ):
            raise ValueError(
                "formal resume metadata does not match the reconstructed full-train "
                "data provenance"
            )
        metadata = original_metadata
    else:
        manifest_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        split_path.write_text(
            json.dumps(data_split, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.formal_full_train_no_eval and resume_payload is not None:
        reconcile_formal_resume_history(run_dir, resume_payload)
    wandb = init_wandb(args, model.get_model_config())
    print(json.dumps(metadata, indent=2, sort_keys=True))

    start_epoch = 0
    global_step = 0
    if args.formal_full_train_no_eval:
        best_metrics = None
        selection = None
    else:
        best_metrics = {
            "auc": float("-inf"),
            "routing_hard_coverage": float("-inf"),
        }
        if args.dataset == "gazefollow":
            best_metrics.update(
                {
                    "avg_l2": float("inf"),
                    "min_l2": float("inf"),
                }
            )
        else:
            best_metrics["l2"] = float("inf")
        primary_metric, primary_mode, tie_breakers = selection_spec(args)
        selection = {
            "metric": primary_metric,
            "mode": primary_mode,
            "tie_breakers": list(tie_breakers),
            "value": float("inf") if primary_mode == "min" else float("-inf"),
            "epoch": None,
            "metrics": None,
            "checkpoint": (
                "best_val_selection.pt"
                if data_split.get("selection_is_formal")
                else "best_selection.pt"
            ),
        }
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        scaler.load_state_dict(resume_payload.get("scaler_state", {}))
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload.get("global_step", 0))
        resumed_selection = resume_payload.get("selection")
        if args.formal_full_train_no_eval:
            if resume_payload.get("best_metrics") is not None:
                raise ValueError(
                    "formal full-train resume checkpoint unexpectedly contains "
                    "best-metric selection state"
                )
            if resumed_selection is not None:
                raise ValueError(
                    "formal full-train resume checkpoint unexpectedly contains "
                    "validation selection state"
                )
        else:
            resumed_best_metrics = resume_payload.get("best_metrics", {})
            for name in best_metrics:
                if name in resumed_best_metrics:
                    best_metrics[name] = resumed_best_metrics[name]
            if isinstance(resumed_selection, dict):
                if (
                    resumed_selection.get("metric") != selection["metric"]
                    or resumed_selection.get("mode") != selection["mode"]
                ):
                    raise ValueError(
                        "resume checkpoint selection rule conflicts with the "
                        "current training configuration"
                    )
                selection.update(resumed_selection)
        restore_rng_state(resume_payload.get("rng_state"))
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")

    if start_epoch > args.max_epochs:
        raise ValueError(
            f"resume checkpoint starts at epoch {start_epoch}, beyond the fixed "
            f"training length {args.max_epochs}"
        )
    if (
        args.formal_full_train_no_eval
        and start_epoch == args.max_epochs
        and not (run_dir / "final.pt").exists()
    ):
        # Recover the narrow interruption window after last.resume.pt was saved
        # for the final epoch but before final.pt/summary.json were written.
        recovered_final = {
            **resume_payload,
            "checkpoint_role": "fixed_epoch_final",
            "initialization_checkpoint": initialization_checkpoint,
        }
        save_checkpoint(run_dir / "final.pt", recovered_final)

    for epoch in range(start_epoch, args.max_epochs):
        active_keep_ratio = current_keep_ratio(
            args.keep_ratio, epoch, args.router_warmup_epochs
        )
        args._active_keep_ratio = active_keep_ratio
        model.router.keep_ratio = active_keep_ratio
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation = 0
        train_meters: Dict[str, WeightedMean] = {
            name: WeightedMean()
            for name in (
                "total",
                "heatmap",
                "inout",
                "router_total",
                "router_coverage",
                "router_budget",
                "router_entropy",
                "router_soft_coverage",
                "router_hard_coverage",
                "router_mean_support",
                "router_actual_keep_ratio",
            )
        }

        total_train_batches = len(train_loader)
        if args.max_train_batches is not None:
            total_train_batches = min(total_train_batches, args.max_train_batches)
        for batch_index, batch in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            model_input, _, _, inout, _, _, target_heatmaps = unpack_batch(
                batch, device, training=True
            )
            with autocast_context(device, args.amp):
                predictions = model(model_input)
            # CUDA autocast intentionally rejects BCELoss. Compute all losses
            # in FP32 while retaining the mixed-precision forward graph.
            heatmap_predictions, inout_predictions = stack_predictions(predictions)
            heatmap_predictions = heatmap_predictions.float()
            if inout_predictions is not None:
                inout_predictions = inout_predictions.float()
            valid = (
                inout.bool()
                if args.dataset == "vat"
                else torch.ones_like(inout, dtype=torch.bool)
            )
            if args.heatmap_loss_weight == 0.0:
                # A router-only support pilot should not traverse the decoder
                # backward graph merely to multiply its loss by 0.
                heatmap_loss = heatmap_predictions.detach().sum() * 0.0
            elif valid.any():
                heatmap_loss = bce(
                    heatmap_predictions[valid], target_heatmaps[valid].float()
                )
            else:
                # VAT can yield an all-out-of-frame batch. Keep a connected
                # zero instead of calling BCE on an empty tensor.
                heatmap_loss = heatmap_predictions.sum() * 0.0
            if args.dataset == "vat":
                if inout_predictions is None:
                    raise RuntimeError("VAT model did not return in/out predictions")
                inout_loss = bce(inout_predictions, inout.float())
            else:
                inout_loss = heatmap_loss.detach() * 0.0
            router_losses = routing_metrics(
                predictions["routing"], target_heatmaps.float(), inout, args
            )
            total_loss = (
                args.heatmap_loss_weight * heatmap_loss
                + args.inout_loss_lambda * inout_loss
                + router_losses["total"]
            )
            accumulation_group_start = (
                batch_index // args.grad_accum_steps
            ) * args.grad_accum_steps
            accumulation_group_size = min(
                args.grad_accum_steps,
                total_train_batches - accumulation_group_start,
            )
            scaled_loss = total_loss / accumulation_group_size

            scaler.scale(scaled_loss).backward()
            pending_accumulation += 1
            should_step = (
                pending_accumulation >= args.grad_accum_steps
                or batch_index + 1 >= total_train_batches
            )
            if should_step:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0
                global_step += 1

            batch_weight = heatmap_predictions.shape[0]
            valid_weight = int(valid.sum().item())
            train_meters["total"].update(total_loss.item(), batch_weight)
            train_meters["heatmap"].update(heatmap_loss.item(), valid_weight)
            train_meters["inout"].update(inout_loss.item(), batch_weight)
            train_meters["router_total"].update(
                router_losses["total"].item(), valid_weight
            )
            train_meters["router_coverage"].update(
                router_losses["coverage"].item(), valid_weight
            )
            train_meters["router_budget"].update(
                router_losses["budget"].item(), batch_weight
            )
            train_meters["router_entropy"].update(
                router_losses["entropy"].item(), batch_weight
            )
            train_meters["router_soft_coverage"].update(
                router_losses["soft_coverage"].item(), valid_weight
            )
            train_meters["router_hard_coverage"].update(
                router_losses["hard_coverage"].item(), valid_weight
            )
            train_meters["router_mean_support"].update(
                router_losses["mean_support"].item(), batch_weight
            )
            train_meters["router_actual_keep_ratio"].update(
                predictions["routing"].actual_keep_ratio, batch_weight
            )

            if batch_index % args.log_iter == 0:
                print(
                    f"TRAIN epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={total_loss.item():.5f} heatmap={heatmap_loss.item():.5f} "
                    f"route_coverage={router_losses['hard_coverage'].item():.5f} "
                    f"keep={predictions['routing'].actual_keep_ratio:.4f}"
                )

        if pending_accumulation:
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        scheduler.step()
        model.router.keep_ratio = args.keep_ratio
        args._active_keep_ratio = args.keep_ratio
        train_metrics = {name: meter.mean() for name, meter in train_meters.items()}
        epoch_metrics = {
            "epoch": epoch,
            "active_train_keep_ratio": active_keep_ratio,
            "train": train_metrics,
        }
        if args.formal_full_train_no_eval:
            eval_metrics = None
        else:
            if eval_loader is None:
                raise RuntimeError("evaluation loader is unexpectedly missing")
            eval_metrics = evaluate_model(
                model, eval_loader, args, device, max_batches=args.max_eval_batches
            )
            eval_metrics["split"] = data_split["evaluation_split"]
            epoch_metrics["eval"] = eval_metrics
        print(json.dumps(epoch_metrics, indent=2, sort_keys=True))
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")

        if wandb is not None:
            log_values = {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                "epoch": epoch,
                "train/active_keep_ratio": active_keep_ratio,
            }
            if eval_metrics is not None:
                log_values.update(
                    {
                        f"eval/{key}": value
                        for key, value in eval_metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
            wandb.log(
                {key: value for key, value in log_values.items() if value is not None}
            )

        improved_paths = []
        if eval_metrics is not None:
            if selection is None or best_metrics is None:
                raise RuntimeError("evaluation selection state is unexpectedly missing")
            if selection_improved(eval_metrics, selection):
                selection.update(
                    {
                        "value": eval_metrics[selection["metric"]],
                        "epoch": epoch,
                        "metrics": eval_metrics.copy(),
                    }
                )
                improved_paths.append(run_dir / selection["checkpoint"])
            coverage = eval_metrics.get("routing_hard_coverage")
            if (
                coverage is not None
                and coverage > best_metrics["routing_hard_coverage"]
            ):
                best_metrics["routing_hard_coverage"] = coverage
                improved_paths.append(run_dir / "best_coverage.pt")
            auc = eval_metrics.get("auc")
            if auc is not None and auc > best_metrics["auc"]:
                best_metrics["auc"] = auc
                improved_paths.append(run_dir / "best_auc.pt")
            if args.dataset == "gazefollow":
                avg_l2 = eval_metrics.get("avg_l2")
                if avg_l2 is not None and avg_l2 < best_metrics["avg_l2"]:
                    best_metrics["avg_l2"] = avg_l2
                    improved_paths.append(run_dir / "best_avg_l2.pt")
                min_l2 = eval_metrics.get("min_l2")
                if min_l2 is not None and min_l2 < best_metrics["min_l2"]:
                    best_metrics["min_l2"] = min_l2
                    improved_paths.append(run_dir / "best_min_l2.pt")
            else:
                l2 = eval_metrics.get("l2")
                if l2 is not None and l2 < best_metrics["l2"]:
                    best_metrics["l2"] = l2
                    improved_paths.append(run_dir / "best_l2.pt")

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            epoch=epoch,
            global_step=global_step,
            metrics=epoch_metrics,
            best_metrics=best_metrics,
            data_split=data_split,
            selection=selection,
            checkpoint_role=(
                "resume_checkpoint"
                if args.formal_full_train_no_eval
                else "training_checkpoint"
            ),
            initialization_checkpoint=initialization_checkpoint,
        )
        save_checkpoint(run_dir / "last.resume.pt", payload)
        if args.formal_full_train_no_eval and epoch + 1 == args.max_epochs:
            final_payload = {**payload, "checkpoint_role": "fixed_epoch_final"}
            save_checkpoint(run_dir / "final.pt", final_payload)
        for path in improved_paths:
            save_checkpoint(path, payload)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch}.pt", payload)

    if args.formal_full_train_no_eval and not (run_dir / "final.pt").is_file():
        raise RuntimeError("formal full-train run did not produce final.pt")

    final_metrics = {
        "run_dir": str(run_dir),
        "evaluation_split": data_split["evaluation_split"],
        "evaluation_performed": not args.formal_full_train_no_eval,
        "checkpoint_policy": data_split.get("selection_policy"),
        "selection": selection,
        "best_metrics": best_metrics,
        "last_epoch": args.max_epochs - 1,
        "final_checkpoint": ("final.pt" if args.formal_full_train_no_eval else None),
        "completed": True,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(final_metrics, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if wandb is not None:
        wandb.finish()
    return final_metrics


if __name__ == "__main__":
    main()
