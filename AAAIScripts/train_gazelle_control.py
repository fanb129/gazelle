#!/usr/bin/env python3
"""Train unchanged Gazelle controls with an explicit best-checkpoint protocol.

This runner imports the historical model read-only.  New candidate model code
remains isolated in ``AAAIModules/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest
from AAAIScripts.train_person_router import (
    build_loaders,
    checkpoint_state,
    evaluate,
    seed_everything,
    train_epoch,
)


def load_gf_initialization_for_vat(model, checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    expected = {key: value for key, value in current.items() if not key.startswith("backbone.")}
    provided = {key: value for key, value in state.items() if not key.startswith("backbone.")}
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        key for key in set(provided) & set(expected)
        if tuple(provided[key].shape) != tuple(expected[key].shape)
    )
    missing = sorted(set(expected) - set(provided))
    allowed_missing = ("inout_token.", "inout_head.")
    unsafe_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if unexpected or mismatched or unsafe_missing:
        raise RuntimeError(
            "GF initialization does not exactly match the VAT control architecture: "
            f"unexpected={unexpected}, shape_mismatch={mismatched}, unsafe_missing={unsafe_missing}"
        )
    current.update(provided)
    model.load_state_dict(current, strict=True)
    return {"loaded": len(provided), "missing_allowed": missing, "coverage": len(provided) / len(expected)}


def main(argv=None):
    args = parse_args(argv)
    from gazelle.model import get_gazelle_model

    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("Training expects CUDA; pass --allow-cpu only for a tiny smoke run")
    model_name = args.model or (
        "gazelle_dinov3_vitb16_inout" if args.dataset == "vat" else "gazelle_dinov3_vitb16"
    )
    model, transform = get_gazelle_model(
        model_name,
        spatial_prior=args.spatial_prior,
        fusion=args.fusion,
        selected_layers=args.selected_layers,
    )
    initialization = None
    if args.dataset == "vat":
        if not args.init_checkpoint:
            raise SystemExit("VAT controls require the matching GazeFollow control --init-checkpoint")
        initialization = load_gf_initialization_for_vat(model, args.init_checkpoint)
    elif args.init_checkpoint:
        raise SystemExit("GazeFollow controls train from scratch; omit --init-checkpoint")

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.to(device)
    train_loader, validation_loader, train_dataset, validation_dataset = build_loaders(args, transform)
    if args.dataset == "vat":
        optimizer = torch.optim.Adam([
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" in name], "lr": args.lr_inout},
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" not in name], "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_metric = "l2" if args.dataset == "vat" else "min_l2"
    manifest = {
        "status": "running",
        "architecture": "unchanged_gazelle_control",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": model_name,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "learnable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "annotation_files": {
            split: file_manifest(Path(args.data_path) / f"{split}_preprocessed.json")
            for split in ("train", "test")
        },
        "initialization_checkpoint": file_manifest(args.init_checkpoint) if args.init_checkpoint else None,
        "initialization_report": initialization,
        "selection_metric": selection_metric,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    history, best_value, best_epoch = [], float("inf"), None
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, args, device)
        metrics = evaluate(model, validation_loader, args, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        torch.save(checkpoint_state(model), args.output_dir / f"epoch_{epoch}.pt")
        selection = metrics[selection_metric]
        if selection is not None and selection < best_value:
            best_value, best_epoch = selection, epoch
            torch.save(checkpoint_state(model), args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} metrics={json.dumps(row, sort_keys=True)}", flush=True)

    manifest.update({"status": "complete", "best_epoch": best_epoch, "best_value": best_value if best_epoch is not None else None})
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--spatial-prior", choices=("none", "fixed_gaussian", "coordconv", "ggsf", "fixed_sector"), default="none")
    parser.add_argument("--fusion", choices=("raw_concat", "equal_weight", "sasa", "fpn", "selected_layers"), required=True)
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-sample-every", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-inout", type=float, default=1e-3)
    parser.add_argument("--inout-loss-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = 8 if args.dataset == "vat" else 15
    if args.lr is None:
        args.lr = 1e-5 if args.dataset == "vat" else 1e-3
    return args


if __name__ == "__main__":
    main()
