#!/usr/bin/env python3
"""Train the isolated AAAI person-conditioned hierarchy candidate.

The historical ``gazelle/`` package is imported but never modified.  This
runner saves ``best.pt`` according to the validation metric and records enough
metadata to distinguish exploratory runs from final matched controls.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest


DEFAULT_PRIOR_WEIGHTS = (0.0340173, 0.0974448, 0.2007198, 0.6678180)


def seed_everything(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_vat_train_dataset(dataset, validation_fraction: float, seed: int):
    """Split VAT train annotations by sequence directory without test leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    sequence_by_image = {
        image_index: str(Path(frame["path"]).parent)
        for image_index, frame in enumerate(dataset.data)
    }
    sequences = sorted(set(sequence_by_image.values()))
    if len(sequences) < 2:
        raise ValueError("VAT train validation split requires at least two sequences")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sequences))
    validation_count = min(len(sequences) - 1, max(1, round(len(sequences) * validation_fraction)))
    validation_sequences = set(shuffled[:validation_count])
    train_sequences = set(shuffled[validation_count:])

    train_dataset = copy.copy(dataset)
    validation_dataset = copy.copy(dataset)
    train_dataset.data_idxs = [
        item for item in dataset.data_idxs if sequence_by_image[item[0]] in train_sequences
    ]
    validation_dataset.data_idxs = [
        item for item in dataset.data_idxs if sequence_by_image[item[0]] in validation_sequences
    ]
    train_dataset.split, train_dataset.aug = "train", True
    # Reuse the train annotations but disable augmentation and return eval tuples.
    validation_dataset.split, validation_dataset.aug = "validation", False
    split_manifest = {
        "source": "train_preprocessed.json",
        "unit": "VAT sequence directory",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "train_queries": len(train_dataset.data_idxs),
        "validation_queries": len(validation_dataset.data_idxs),
        "test_used_for_selection": False,
    }
    train_dataset.split_manifest = split_manifest
    validation_dataset.split_manifest = split_manifest
    return train_dataset, validation_dataset


def build_loaders(args, transform):
    import torch
    from gazelle.dataloader import GazeDataset, collate_fn

    dataset_name = "videoattentiontarget" if args.dataset == "vat" else "gazefollow"
    train = GazeDataset(
        dataset_name, args.data_path, "train", transform,
        in_frame_only=(args.dataset == "gazefollow"),
        sample_rate=args.frame_sample_every if args.dataset == "vat" else 1,
    )
    if args.dataset == "vat" and getattr(args, "validation_from_train", False):
        train, validation = split_vat_train_dataset(
            train,
            getattr(args, "validation_fraction", 0.1),
            getattr(args, "validation_seed", args.seed),
        )
    else:
        validation = GazeDataset(
            dataset_name, args.data_path, "test", transform,
            in_frame_only=(args.dataset == "gazefollow"),
            sample_rate=args.frame_sample_every if args.dataset == "vat" else 1,
        )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": True,
    }
    return (
        torch.utils.data.DataLoader(train, shuffle=True, **common),
        torch.utils.data.DataLoader(validation, shuffle=False, **common),
        train,
        validation,
    )


def train_epoch(model, loader, optimizer, args, device):
    import torch
    import torch.nn.functional as F

    model.train()
    losses = []
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        images, bboxes, _, _, inout, _, _, heatmaps = batch
        images, heatmaps, inout = images.to(device), heatmaps.to(device), inout.to(device)
        output = model({"images": images, "bboxes": [[bbox] for bbox in bboxes]})
        heatmap_predictions = torch.stack(output["heatmap"]).squeeze(1)

        if args.dataset == "vat":
            inout_predictions = torch.stack(output["inout"]).squeeze(1)
            in_mask = inout.bool()
            # An all-out batch has no heatmap target. Keep a differentiable zero
            # instead of calling BCE on an empty tensor (which yields NaN).
            heatmap_loss = (
                F.binary_cross_entropy(heatmap_predictions[in_mask], heatmaps[in_mask])
                if in_mask.any() else heatmap_predictions.sum() * 0.0
            )
            inout_loss = F.binary_cross_entropy(inout_predictions, inout.float())
            loss = heatmap_loss + args.inout_loss_lambda * inout_loss
        else:
            loss = F.binary_cross_entropy(heatmap_predictions, heatmaps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if batch_index % args.log_every == 0:
            print(f"train batch={batch_index}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else None


def evaluate(model, loader, args, device):
    import torch
    from gazelle.utils import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2

    model.eval()
    aucs, l2s, avg_l2s, min_l2s = [], [], [], []
    inout_predictions, inout_labels = [], []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            images, bboxes, gazex, gazey, inout, heights, widths = batch
            output = model({"images": images.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmaps = torch.stack(output["heatmap"]).squeeze(1)
            if args.dataset == "vat":
                scores = torch.stack(output["inout"]).squeeze(1)
                for index in range(len(heatmaps)):
                    if int(inout[index]) == 1:
                        aucs.append(vat_auc(heatmaps[index], gazex[index][0], gazey[index][0]))
                        l2s.append(vat_l2(heatmaps[index], gazex[index][0], gazey[index][0]))
                    inout_predictions.append(float(scores[index].cpu()))
                    inout_labels.append(int(inout[index]))
            else:
                for index in range(len(heatmaps)):
                    aucs.append(gazefollow_auc(heatmaps[index], gazex[index], gazey[index], heights[index], widths[index]))
                    avg_l2, min_l2 = gazefollow_l2(heatmaps[index], gazex[index], gazey[index])
                    avg_l2s.append(avg_l2)
                    min_l2s.append(min_l2)

    metrics = {"sample_count": len(aucs), "auc": float(np.mean(aucs)) if aucs else None}
    if args.dataset == "vat":
        from sklearn.metrics import average_precision_score

        metrics.update({
            "l2": float(np.mean(l2s)) if l2s else None,
            "inout_ap": (
                float(average_precision_score(inout_labels, inout_predictions))
                if len(set(inout_labels)) > 1 else None
            ),
            "inout_count": len(inout_labels),
        })
    else:
        metrics.update({
            "avg_l2": float(np.mean(avg_l2s)) if avg_l2s else None,
            "min_l2": float(np.mean(min_l2s)) if min_l2s else None,
        })
    return metrics


def checkpoint_state(model):
    return model.get_gazelle_state_dict(include_backbone=False)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from AAAIModules.factory import build_person_hierarchical_gazelle

    prior_source = None
    if args.router_prior_json:
        prior_payload = json.loads(args.router_prior_json.read_text())
        annotation_path = Path(prior_payload.get("annotation", {}).get("path", ""))
        if annotation_path.name != "train_preprocessed.json":
            raise RuntimeError(
                "--router-prior-json must be computed from train_preprocessed.json"
            )
        measured_weights = prior_payload.get("mean_layer_weights")
        if not isinstance(measured_weights, list):
            raise RuntimeError("router prior report does not contain mean_layer_weights")
        args.router_prior_weights = parse_prior_weights(
            ",".join(str(value) for value in measured_weights)
        )
        prior_source = file_manifest(args.router_prior_json)

    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("Training expects CUDA; pass --allow-cpu only for a tiny smoke run")

    model_name = args.model or (
        "aaai_person_router_dinov3_vitb16_inout" if args.dataset == "vat"
        else "aaai_person_router_dinov3_vitb16"
    )
    model, transform = build_person_hierarchical_gazelle(
        model_name,
        router_hidden_dim=args.router_hidden_dim,
        router_roi_size=args.router_roi_size,
        router_dropout=args.router_dropout,
        router_temperature=args.router_temperature,
        router_prior_weights=args.router_prior_weights,
        router_residual_scale=args.router_residual_scale,
    )
    initialization = None
    if args.init_checkpoint:
        initialization = model.load_base_checkpoint(
            args.init_checkpoint,
            allow_legacy_sasa_ggsf=args.allow_legacy_sasa_ggsf,
        )
        if initialization["incompatible_shapes"] or initialization["unexpected"]:
            raise RuntimeError(f"unsafe initialization checkpoint: {initialization}")
        allowed_missing = ("backbone.", "layer_router.", "inout_token.", "inout_head.")
        unsafe_missing = [key for key in initialization["missing"] if not key.startswith(allowed_missing)]
        if unsafe_missing:
            raise RuntimeError(f"initialization leaves non-router/non-inout tensors missing: {unsafe_missing}")

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if args.train_scope == "router_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer_router.parameters():
            parameter.requires_grad = True
    model.to(device)
    train_loader, validation_loader, train_dataset, validation_dataset = build_loaders(args, transform)

    if args.train_scope == "router_only":
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.layer_router.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
    elif args.dataset == "vat":
        optimizer = torch.optim.Adam([
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" in name], "lr": args.lr_inout},
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and "inout" not in name], "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_files = {
        split: file_manifest(Path(args.data_path) / f"{split}_preprocessed.json")
        for split in ("train", "test")
    }
    manifest = {
        "status": "running",
        "candidate": "P1a_global_prior_plus_person_residual_router",
        "claim_boundary": (
            "Tests whether bbox-conditioned residual routing improves over the exact P0.5 "
            "task-global hierarchy prior; no relational loss or inter-person query interaction."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": model_name,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "learnable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "annotation_files": annotation_files,
        "initialization_checkpoint": file_manifest(args.init_checkpoint) if args.init_checkpoint else None,
        "initialization_report": initialization,
        "router_prior_source": prior_source,
        "selection_metric": "l2" if args.dataset == "vat" else "min_l2",
        "validation_protocol": getattr(validation_dataset, "split_manifest", {
            "source": "test_preprocessed.json",
            "test_used_for_selection": True,
        }),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save(checkpoint_state(model), args.output_dir / "initial.pt")

    best_value, best_epoch = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, args, device)
        metrics = evaluate(model, validation_loader, args, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        torch.save(checkpoint_state(model), args.output_dir / f"epoch_{epoch}.pt")
        selection = metrics[manifest["selection_metric"]]
        if selection is not None and selection < best_value:
            best_value, best_epoch = selection, epoch
            torch.save(checkpoint_state(model), args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} metrics={json.dumps(row, sort_keys=True)}", flush=True)

    manifest.update({"status": "complete", "best_epoch": best_epoch, "best_value": best_value if best_epoch is not None else None})
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Training complete; best_epoch={best_epoch}, best_value={manifest['best_value']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-sample-every", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-inout", type=float, default=1e-3)
    parser.add_argument("--inout-loss-lambda", type=float, default=1.0)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-roi-size", type=int, default=3)
    parser.add_argument("--router-dropout", type=float, default=0.1)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-prior-weights", type=parse_prior_weights, default=DEFAULT_PRIOR_WEIGHTS)
    parser.add_argument("--router-prior-json", type=Path)
    parser.add_argument("--router-residual-scale", type=float, default=1.0)
    parser.add_argument("--train-scope", choices=("router_only", "full_decoder"), default="full_decoder")
    parser.add_argument("--allow-legacy-sasa-ggsf", action="store_true")
    parser.add_argument("--validation-from-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=3106)
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
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if args.router_residual_scale <= 0:
        parser.error("--router-residual-scale must be positive")
    return args


def parse_prior_weights(value):
    if isinstance(value, tuple):
        return value
    try:
        weights = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prior weights must be comma-separated floats") from exc
    if len(weights) != 4 or any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("prior weights must contain four positive values")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


if __name__ == "__main__":
    main()
