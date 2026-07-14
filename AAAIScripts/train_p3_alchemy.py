#!/usr/bin/env python3
"""Train P3 R0/R1/R2/R3 with train-derived validation and audited warm-starts."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIAlchemyModels.factory import CANDIDATE_CHOICES, resolve_candidate
from AAAIScripts.common import file_manifest


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_train_dataset(dataset, dataset_name: str, validation_fraction: float, seed: int):
    """Split VAT by sequence directory and GazeFollow by image, never by query."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if dataset_name == "vat":
        group_by_image = {
            image_index: str(Path(frame["path"]).parent)
            for image_index, frame in enumerate(dataset.data)
        }
        unit = "VAT sequence directory"
    elif dataset_name == "gazefollow":
        group_by_image = {
            image_index: str(frame["path"])
            for image_index, frame in enumerate(dataset.data)
        }
        unit = "GazeFollow image path"
    else:
        raise ValueError(f"unknown dataset_name={dataset_name!r}")

    groups = sorted(set(group_by_image.values()))
    if len(groups) < 2:
        raise ValueError("train-derived validation requires at least two groups")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    validation_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(shuffled[:validation_count])
    train_groups = set(shuffled[validation_count:])

    train = copy.copy(dataset)
    validation = copy.copy(dataset)
    train.data_idxs = [item for item in dataset.data_idxs if group_by_image[item[0]] in train_groups]
    validation.data_idxs = [
        item for item in dataset.data_idxs if group_by_image[item[0]] in validation_groups
    ]
    train.split, train.aug = "train", True
    validation.split, validation.aug = "validation", False
    manifest = {
        "source": "train_preprocessed.json",
        "unit": unit,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "train_queries": len(train.data_idxs),
        "validation_queries": len(validation.data_idxs),
        "test_used_for_selection": False,
    }
    train.split_manifest = manifest
    validation.split_manifest = manifest
    return train, validation


def build_loaders(args, transform):
    from gazelle.dataloader import GazeDataset, collate_fn

    loader_name = "videoattentiontarget" if args.dataset == "vat" else "gazefollow"
    dataset = GazeDataset(
        loader_name,
        args.data_path,
        "train",
        transform,
        in_frame_only=(args.dataset == "gazefollow"),
        sample_rate=args.frame_sample_every if args.dataset == "vat" else 1,
    )
    train, validation = split_train_dataset(
        dataset, args.dataset, args.validation_fraction, args.validation_seed
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


def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Return differentiable normalized x/y coordinates from sigmoid heatmaps."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape [N,H,W]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n, height, width = heatmaps.shape
    logits = torch.logit(heatmaps.clamp(1e-6, 1.0 - 1e-6)) / temperature
    probabilities = torch.softmax(logits.reshape(n, -1), dim=-1).reshape(n, height, width)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    pred_x = (probabilities * xs.view(1, 1, width)).sum(dim=(1, 2))
    pred_y = (probabilities * ys.view(1, height, 1)).sum(dim=(1, 2))
    return torch.stack([pred_x, pred_y], dim=-1)


def coordinate_loss(
    heatmaps: torch.Tensor,
    gazex,
    gazey,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if not bool(valid_mask.any()):
        return heatmaps.sum() * 0.0
    predicted = soft_argmax_2d(heatmaps[valid_mask], temperature)
    target = torch.tensor(
        [[float(gazex[index][0]), float(gazey[index][0])] for index in range(len(gazex))],
        device=heatmaps.device,
        dtype=heatmaps.dtype,
    )[valid_mask]
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean()


def train_epoch(model, loader, optimizer, args, device, coordinate_weight: float):
    model.train()
    rows = []
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        images, bboxes, gazex, gazey, inout, _, _, heatmaps = batch
        images, heatmaps, inout = images.to(device), heatmaps.to(device), inout.to(device)
        output = model({"images": images, "bboxes": [[bbox] for bbox in bboxes]})
        heatmap_predictions = torch.stack(output["heatmap"]).squeeze(1)
        valid_mask = inout.bool() if args.dataset == "vat" else torch.ones_like(inout, dtype=torch.bool)
        heatmap_loss = (
            F.binary_cross_entropy(heatmap_predictions[valid_mask], heatmaps[valid_mask])
            if bool(valid_mask.any()) else heatmap_predictions.sum() * 0.0
        )
        inout_loss = heatmap_predictions.sum() * 0.0
        if args.dataset == "vat":
            inout_predictions = torch.stack(output["inout"]).squeeze(1)
            inout_loss = F.binary_cross_entropy(inout_predictions, inout.float())
        coord_loss = coordinate_loss(
            heatmap_predictions, gazex, gazey, valid_mask, args.coordinate_temperature
        ) if coordinate_weight > 0 else heatmap_predictions.sum() * 0.0
        total = heatmap_loss + args.inout_loss_lambda * inout_loss + coordinate_weight * coord_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        row = {
            "loss": float(total.detach().cpu()),
            "heatmap_loss": float(heatmap_loss.detach().cpu()),
            "inout_loss": float(inout_loss.detach().cpu()),
            "coordinate_loss": float(coord_loss.detach().cpu()),
        }
        rows.append(row)
        if batch_index % args.log_every == 0:
            print(f"train batch={batch_index}/{len(loader)} metrics={json.dumps(row, sort_keys=True)}", flush=True)
    return {
        key: float(np.mean([row[key] for row in rows])) if rows else None
        for key in ("loss", "heatmap_loss", "inout_loss", "coordinate_loss")
    }


def evaluate(model, loader, args, device):
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
                    aucs.append(gazefollow_auc(
                        heatmaps[index], gazex[index], gazey[index], heights[index], widths[index]
                    ))
                    avg_l2, min_l2 = gazefollow_l2(
                        heatmaps[index], gazex[index], gazey[index]
                    )
                    avg_l2s.append(avg_l2)
                    min_l2s.append(min_l2)
    metrics = {"sample_count": len(aucs), "auc": float(np.mean(aucs)) if aucs else None}
    if args.dataset == "vat":
        from sklearn.metrics import average_precision_score

        metrics.update({
            "l2": float(np.mean(l2s)) if l2s else None,
            "inout_ap": float(average_precision_score(inout_labels, inout_predictions))
            if len(set(inout_labels)) > 1 else None,
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


def build_optimizer(model, args):
    groups = {"fusion": [], "inout": [], "base": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fusion_refiner."):
            groups["fusion"].append(parameter)
        elif "inout" in name:
            groups["inout"].append(parameter)
        else:
            groups["base"].append(parameter)
    specifications = []
    for name, lr in (("base", args.lr), ("fusion", args.fusion_lr), ("inout", args.lr_inout)):
        if groups[name]:
            specifications.append({"params": groups[name], "lr": lr, "group_name": name})
    return torch.optim.Adam(specifications)


def audit_zero_initialization(model) -> dict:
    refiner = getattr(model, "fusion_refiner", None)
    if refiner is None:
        return {"branch": "none", "output_projection_l1": 0.0, "exact_zero": True}
    projection = refiner.output_projection
    total = float(projection.weight.detach().abs().sum())
    if projection.bias is not None:
        total += float(projection.bias.detach().abs().sum())
    return {"branch": model.refinement, "output_projection_l1": total, "exact_zero": total == 0.0}


def validate_initialization(report: dict, allow_missing_inout: bool) -> None:
    if report["unexpected"] or report["incompatible_shapes"]:
        raise RuntimeError(f"unsafe initialization checkpoint: {report}")
    allowed_prefixes = ["fusion_refiner."]
    if allow_missing_inout:
        allowed_prefixes.extend(["inout_token.", "inout_head."])
    unsafe = [
        key for key in report["missing"]
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]
    required_coverage = (
        report["cross_dataset_shared_coverage"]
        if allow_missing_inout else report["shared_base_coverage"]
    )
    branch_loaded = report["new_branch_loaded_tensors"]
    branch_expected = report["new_branch_expected_tensors"]
    partial_branch = branch_loaded not in (0, branch_expected)
    if unsafe or required_coverage < 1.0 or partial_branch:
        raise RuntimeError(
            "initialization failed coverage/atomic-branch checks: "
            f"unsafe_missing={unsafe}, partial_branch={partial_branch}, report={report}"
        )


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    from AAAIAlchemyModels.factory import build_p3_alchemy_model

    seed_everything(args.seed, args.deterministic)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("P3 training expects CUDA; pass --allow-cpu only for a bounded smoke run")
    model, transform, candidate = build_p3_alchemy_model(
        args.candidate,
        dataset=args.dataset,
        refinement_width=args.refinement_width,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
    )
    initialization = model.load_alchemy_checkpoint(args.init_checkpoint)
    validate_initialization(initialization, args.allow_missing_inout)
    zero_init = audit_zero_initialization(model)
    if not zero_init["exact_zero"] and initialization["new_branch_missing"]:
        raise RuntimeError(f"new residual branch is not zero-initialized: {zero_init}")

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.to(device)
    train_loader, validation_loader, train_dataset, validation_dataset = build_loaders(args, transform)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_files = {
        split: file_manifest(args.data_path / f"{split}_preprocessed.json")
        for split in ("train", "test")
    }
    coordinate_weight = candidate.coordinate_loss_weight
    manifest = {
        "status": "running",
        "phase": args.phase,
        "candidate": candidate.to_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "learnable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "annotation_files": annotation_files,
        "initialization_checkpoint": file_manifest(args.init_checkpoint),
        "initialization_report": initialization,
        "zero_initialization_audit": zero_init,
        "validation_protocol": validation_dataset.split_manifest,
        "test_used_for_selection": False,
        "selection_metric": "l2" if args.dataset == "vat" else "min_l2",
        "all_metrics_recorded": True,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    torch.save(checkpoint_state(model), args.output_dir / "initial.pt")

    history, best_value, best_epoch, best_metrics = [], float("inf"), None, None
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, args, device, coordinate_weight
        )
        validation_metrics = evaluate(model, validation_loader, args, device)
        learning_rates = {
            group.get("group_name", str(index)): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rates": learning_rates,
        }
        scheduler.step()
        history.append(row)
        torch.save(checkpoint_state(model), args.output_dir / f"epoch_{epoch}.pt")
        selection = validation_metrics[manifest["selection_metric"]]
        if selection is not None and selection < best_value:
            best_value, best_epoch, best_metrics = float(selection), epoch, validation_metrics
            torch.save(checkpoint_state(model), args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        print(f"epoch={epoch} metrics={json.dumps(row, sort_keys=True)}", flush=True)

    manifest.update({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_value": best_value if best_epoch is not None else None,
        "best_metrics": best_metrics,
    })
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"P3 training complete: candidate={candidate.candidate} best_epoch={best_epoch} best_metrics={best_metrics}")


def run_self_test() -> None:
    heatmaps = torch.full((2, 5, 5), 0.01)
    heatmaps[0, 1, 3] = 0.99
    heatmaps[1, 4, 0] = 0.99
    coordinates = soft_argmax_2d(heatmaps, temperature=0.05)
    torch.testing.assert_close(coordinates[0], torch.tensor([0.75, 0.25]), atol=1e-4, rtol=0)
    torch.testing.assert_close(coordinates[1], torch.tensor([0.0, 1.0]), atol=1e-4, rtol=0)
    for candidate in CANDIDATE_CHOICES:
        config = resolve_candidate(candidate)
        assert config.coordinate_loss_weight == (0.05 if candidate == "r3" else 0.0)
    differentiable = heatmaps.clone().requires_grad_(True)
    soft_argmax_2d(differentiable, temperature=0.1).sum().backward()
    assert differentiable.grad is not None and float(differentiable.grad.abs().sum()) > 0

    class SyntheticDataset:
        data = [
            {"path": f"show_{sequence}/clip/frame_{frame}.jpg"}
            for sequence in range(10) for frame in range(2)
        ]
        data_idxs = [(index, 0) for index in range(20)]
        split = "train"
        aug = True

    train, validation = split_train_dataset(SyntheticDataset(), "vat", 0.1, 3106)
    train_sequences = {
        str(Path(SyntheticDataset.data[index]["path"]).parent)
        for index, _ in train.data_idxs
    }
    validation_sequences = {
        str(Path(SyntheticDataset.data[index]["path"]).parent)
        for index, _ in validation.data_idxs
    }
    assert train_sequences.isdisjoint(validation_sequences)
    assert len(validation_sequences) == 1
    assert validation.split_manifest["test_used_for_selection"] is False
    print("Self-test passed: candidate mapping, differentiable coordinate loss, and sequence-disjoint 90/10 split.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("p3a", "p3b", "p3c", "p3d", "smoke"), default="p3a")
    parser.add_argument("--candidate", choices=CANDIDATE_CHOICES)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-missing-inout", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-sample-every", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=3106)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fusion-lr", type=float, default=1e-5)
    parser.add_argument("--lr-inout", type=float, default=1e-3)
    parser.add_argument("--inout-loss-lambda", type=float, default=1.0)
    parser.add_argument("--coordinate-temperature", type=float, default=0.1)
    parser.add_argument("--refinement-width", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    required = ("candidate", "dataset", "data_path", "init_checkpoint", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("normal training requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("--epochs/--batch-size must be positive and --workers non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if args.coordinate_temperature <= 0:
        parser.error("--coordinate-temperature must be positive")
    if args.dataset != "vat" and args.allow_missing_inout:
        parser.error("--allow-missing-inout is only valid when initializing VAT from GazeFollow")
    return args


if __name__ == "__main__":
    main()
