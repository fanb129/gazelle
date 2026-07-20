#!/usr/bin/env python3
"""Matched VAT training with optional COTB loss on same-frame observer groups."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from AAAICOTB.binding import BindingConfig
from AAAICOTB.annotations import load_sequences, split_sequence_indices
from AAAICOTB.data import VATFrameDataset, collate_frames
from AAAICOTB.engine import evaluate_loader, train_one_epoch
from AAAICOTB.models import build_model, load_gazefollow_initialization, task_state_dict
from AAAIScripts.common import file_manifest


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(args, transform):
    annotation = args.data_path / "train_preprocessed.json"
    sequences = load_sequences(annotation)
    train_indices, validation_indices = split_sequence_indices(
        sequences, args.validation_fraction, args.validation_seed
    )
    train_dataset = VATFrameDataset(
        args.data_path,
        annotation,
        transform,
        sequence_indices=train_indices,
        frame_sample_every=args.frame_sample_every,
        augment=not args.no_augment,
        horizontal_flip_probability=args.horizontal_flip_probability,
        bbox_jitter_probability=args.bbox_jitter_probability,
        bbox_jitter_scale=args.bbox_jitter_scale,
    )
    validation_dataset = VATFrameDataset(
        args.data_path,
        annotation,
        transform,
        sequence_indices=validation_indices,
        frame_sample_every=args.validation_frame_sample_every,
        augment=False,
    )
    common = {
        "batch_size": args.batch_size_frames,
        "num_workers": args.workers,
        "collate_fn": collate_frames,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **common,
    )
    split_manifest = {
        "source": file_manifest(annotation),
        "unit": "VAT sequence",
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "train_sequence_indices": train_indices,
        "validation_sequence_indices": validation_indices,
        "train_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "test_used_for_selection": False,
    }
    return train_loader, validation_loader, split_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="Matching GazeFollow task checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--spatial-prior", default="none")
    parser.add_argument("--fusion", default="raw_concat")
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size-frames", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-sample-every", type=int, default=6)
    parser.add_argument("--validation-frame-sample-every", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=9102)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-inout", type=float, default=1e-3)
    parser.add_argument("--inout-loss-weight", type=float, default=1.0)
    parser.add_argument("--bind-weight", type=float, default=0.10)
    parser.add_argument("--bind-margin", type=float, default=0.20)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--min-pair-separation", type=float, default=0.10)
    parser.add_argument("--target-sigma", type=float, default=0.04)
    parser.add_argument("--duplicate-bbox-iou", type=float, default=0.80)
    parser.add_argument("--joint-selection-weight", type=float, default=0.10, help="joint=l2+weight*swap_error")
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--bbox-jitter-probability", type=float, default=0.5)
    parser.add_argument("--bbox-jitter-scale", type=float, default=0.2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("Full training expects CUDA; use --allow-cpu only for a tiny smoke test")
    model, transform = build_model(
        "current", args.model, args.spatial_prior, args.fusion, args.selected_layers
    )
    initialization = load_gazefollow_initialization(model, args.init_checkpoint)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.to(device)
    train_loader, validation_loader, split_manifest = build_loaders(args, transform)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" in name],
                "lr": args.lr_inout,
            },
            {
                "params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" not in name],
                "lr": args.lr,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    config = BindingConfig(
        shared_target_radius=args.shared_target_radius,
        min_pair_separation=args.min_pair_separation,
        target_sigma=args.target_sigma,
        margin=args.bind_margin,
        duplicate_bbox_iou=args.duplicate_bbox_iou,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "method": "COTB" if args.bind_weight > 0 else "matched_grouped_control",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "binding_config": config.as_dict(),
        "initialization": initialization,
        "split": split_manifest,
        "learnable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selection_rule": f"validation_l2 + {args.joint_selection_weight} * validation_swap_error",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    history: list[dict] = []
    best = {"joint": (float("inf"), None), "l2": (float("inf"), None), "binding": (float("inf"), None)}
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            args.bind_weight,
            args.inout_loss_weight,
            args.log_every,
            args.max_train_batches,
        )
        validation, _, _, _ = evaluate_loader(
            model,
            validation_loader,
            device,
            config,
            bootstrap_iterations=0,
            seed=args.seed,
            max_batches=args.max_eval_batches,
        )
        scheduler.step()
        l2 = validation["standard"]["l2"]
        swap_error = validation["binding"]["overall"]["swap_error"]["value"]
        joint_value = (
            float(l2) + args.joint_selection_weight * float(swap_error)
            if l2 is not None and swap_error is not None
            else None
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "selection": {"joint": joint_value, "l2": l2, "swap_error": swap_error},
        }
        history.append(row)
        torch.save(task_state_dict(model), args.output_dir / f"epoch_{epoch}.pt")
        candidates = {
            "joint": float(joint_value) if joint_value is not None else float("inf"),
            "l2": float(l2) if l2 is not None else float("inf"),
            "binding": float(swap_error) if swap_error is not None else float("inf"),
        }
        for name, value in candidates.items():
            if value < best[name][0]:
                best[name] = (value, epoch)
                torch.save(task_state_dict(model), args.output_dir / f"best_{name}.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"epoch": epoch, "train": train_metrics, "selection": row["selection"]}, ensure_ascii=False), flush=True)

    manifest["status"] = "complete"
    manifest["best"] = {
        name: {"value": None if not np.isfinite(value) else value, "epoch": epoch}
        for name, (value, epoch) in best.items()
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
