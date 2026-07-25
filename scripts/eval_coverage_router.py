"""Evaluate a structured coverage-router checkpoint on GazeFollow or VAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import torch

from gazelle.dataloader import (
    GazeDataset,
    GazeFollowImageDataset,
    collate_fn,
    collate_gazefollow_images,
)
from gazelle.routing.model import ROUTER_STAGES, get_coverage_router_model

try:  # Works both as ``python scripts/eval_...py`` and as a module import.
    from train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        DEFAULT_DATA_PATHS,
        _torch_load,
        evaluate_model,
        load_model_state,
    )
except ModuleNotFoundError:
    from scripts.train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        DEFAULT_DATA_PATHS,
        _torch_load,
        evaluate_model,
        load_model_state,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a coverage-aware gaze router."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--router_stage_override",
        choices=ROUTER_STAGES,
        default=None,
        help=(
            "Evaluation-only override for diagnostic controls. For example, "
            "evaluate a support_pilot checkpoint through backbone_sparse "
            "without taking an optimizer step."
        ),
    )
    parser.add_argument(
        "--keep_ratio_override",
        type=float,
        default=None,
        help=(
            "Evaluation-only sparse keep-ratio override. This does not alter "
            "the checkpoint on disk."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--gazefollow_eval_unit",
        choices=("person", "image"),
        default="person",
        help=(
            "person preserves the legacy one-head-per-image evaluation; image "
            "queries every annotated head together and exercises the shared "
            "per-image router union."
        ),
    )
    parser.add_argument(
        "--frame_sample_every",
        type=int,
        default=1,
        help="VAT evaluation sampling rate. The default 1 evaluates every test frame.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.n_workers < 0:
        raise ValueError("--n_workers must be non-negative")
    if args.frame_sample_every <= 0:
        raise ValueError("--frame_sample_every must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max_eval_batches must be positive")
    if (
        args.keep_ratio_override is not None
        and not 0.0 < args.keep_ratio_override <= 1.0
    ):
        raise ValueError("--keep_ratio_override must be in (0, 1]")


def resolve_dataset(args: argparse.Namespace, checkpoint: dict) -> str:
    checkpoint_dataset = checkpoint.get("train_config", {}).get("dataset")
    if checkpoint_dataset not in ("gazefollow", "vat"):
        raise ValueError("checkpoint train_config does not contain a valid dataset")
    if args.dataset is not None and args.dataset != checkpoint_dataset:
        raise ValueError(
            f"--dataset={args.dataset} conflicts with checkpoint dataset={checkpoint_dataset}"
        )
    return checkpoint_dataset


def build_model(model_config: dict):
    required = {
        "model",
        "router_stage",
        "route_after_block",
        "keep_ratio",
        "router_hidden_dim",
        "router_temperature",
        "escape_tokens",
        "spatial_prior",
        "fusion",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise ValueError(f"checkpoint model_config is missing: {missing}")
    return get_coverage_router_model(
        model_config["model"],
        router_stage=model_config["router_stage"],
        route_after_block=int(model_config["route_after_block"]),
        keep_ratio=float(model_config["keep_ratio"]),
        router_hidden_dim=int(model_config["router_hidden_dim"]),
        router_temperature=float(model_config["router_temperature"]),
        escape_tokens=int(model_config["escape_tokens"]),
        spatial_prior=model_config["spatial_prior"],
        fusion=model_config["fusion"],
    )


def apply_eval_overrides(model_config: dict, args: argparse.Namespace) -> dict:
    """Return the effective model config without mutating checkpoint metadata."""
    effective = dict(model_config)
    if args.router_stage_override is not None:
        effective["router_stage"] = args.router_stage_override
    if args.keep_ratio_override is not None:
        effective["keep_ratio"] = float(args.keep_ratio_override)
    return effective


def make_eval_loader(args: argparse.Namespace, transform):
    if args.dataset == "gazefollow":
        if args.gazefollow_eval_unit == "image":
            dataset = GazeFollowImageDataset(
                args.data_path, "test", transform
            )
            batch_collate = collate_gazefollow_images
        else:
            dataset = GazeDataset(
                "gazefollow", args.data_path, "test", transform
            )
            batch_collate = collate_fn
    else:
        dataset = GazeDataset(
            "videoattentiontarget",
            args.data_path,
            "test",
            transform,
            in_frame_only=False,
            sample_rate=args.frame_sample_every,
        )
        batch_collate = collate_fn
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_collate,
        num_workers=args.n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.n_workers > 0,
    )
    return dataset, loader


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("--checkpoint must be a structured coverage-router checkpoint")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or missing checkpoint format_version")

    args.dataset = resolve_dataset(args, checkpoint)
    if args.dataset != "gazefollow" and args.gazefollow_eval_unit != "person":
        raise ValueError("--gazefollow_eval_unit=image is only valid for GazeFollow")
    args.data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("checkpoint has no model_config")
    model_config = apply_eval_overrides(checkpoint_model_config, args)
    args.keep_ratio = float(model_config["keep_ratio"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model, transform = build_model(model_config)
    # Evaluation reconstructs frozen DINO weights from the official local
    # checkpoint and then overlays every stored task/suffix parameter.
    model.set_trainable_backbone_suffix(False)
    load_model_state(model, checkpoint, initialization=False)
    model.to(device).eval()
    dataset, loader = make_eval_loader(args, transform)
    metrics = evaluate_model(
        model,
        loader,
        args,
        device,
        max_batches=args.max_eval_batches,
    )
    result = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "checkpoint_data_split": checkpoint.get("data_split"),
        "checkpoint_selection": checkpoint.get("selection"),
        "dataset": args.dataset,
        "data_path": args.data_path,
        "gazefollow_eval_unit": (
            args.gazefollow_eval_unit if args.dataset == "gazefollow" else None
        ),
        "dataset_sample_count": metrics["sample_count"],
        "dataset_image_count": metrics["image_count"],
        "dataset_loader_item_count": len(dataset),
        "vat_frame_sample_every": (
            args.frame_sample_every if args.dataset == "vat" else None
        ),
        "checkpoint_model_config": checkpoint_model_config,
        "model_config": model_config,
        "evaluation_overrides": {
            "router_stage": args.router_stage_override,
            "keep_ratio": args.keep_ratio_override,
        },
        "metrics": metrics,
        "note": (
            "support_pilot runs dense DINOv3 and is not an efficiency result"
            if model_config["router_stage"] == "support_pilot"
            else "backbone_sparse applies routing inside the DINOv3 encoder"
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")
    return result


if __name__ == "__main__":
    main()
