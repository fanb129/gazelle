"""Train fixed layer probes from synthetic, cached, or live frozen features.

The synthetic smoke path intentionally never imports or loads DINO/Gazelle.
For the real one-day pilot, ``--train-json`` and ``--val-json`` are expected to
be outputs of :mod:`AAAISelectiveGaze.scripts.make_splits`. The backbone and
base predictor are always frozen; only the four probes receive gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from AAAISelectiveGaze.models.layer_probe import DEFAULT_LAYERS, FixedLayerProbes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureCache:
    features: Tensor  # [N, L, C, H, W]
    head_maps: Tensor  # [N, H, W]
    targets: Tensor  # [N, 64, 64]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synthetic_cache(
    *, samples: int, layers: tuple[int, ...], channels: int, seed: int
) -> FeatureCache:
    generator = torch.Generator().manual_seed(seed)
    height = width = 8
    features = torch.randn(
        samples, len(layers), channels, height, width, generator=generator
    )
    head_maps = torch.zeros(samples, height, width)
    targets = torch.zeros(samples, 64, 64)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    for sample in range(samples):
        head_y = sample % height
        head_x = (sample * 3) % width
        head_maps[sample, head_y, head_x] = 1.0
        # A deterministic non-degenerate target; no backbone or checkpoint is used.
        target_y = (sample * 11 + 9) % 64
        target_x = (sample * 17 + 7) % 64
        targets[sample] = torch.exp(
            -((yy - target_y).float().square() + (xx - target_x).float().square())
            / (2.0 * 3.0**2)
        )
    return FeatureCache(features=features, head_maps=head_maps, targets=targets)


def _first_present(archive: Any, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        if key in archive:
            return archive[key]
    raise ValueError(
        f"cache is missing one of required arrays: {', '.join(candidates)}"
    )


def _load_npz_cache(path: Path, layers: tuple[int, ...]) -> FeatureCache:
    with np.load(path, allow_pickle=False) as archive:
        arrays = []
        for layer in layers:
            arrays.append(
                _first_present(
                    archive,
                    (f"features_{layer}", f"feature_layer_{layer}", f"layer_{layer}"),
                )
            )
        head_maps = _first_present(archive, ("head_maps", "head_map"))
        targets = _first_present(
            archive, ("target_heatmaps", "targets", "heatmaps")
        )

    try:
        features = torch.from_numpy(np.stack(arrays, axis=1)).float()
    except ValueError as error:
        raise ValueError("all cached layer feature arrays must have identical shapes") from error
    heads = torch.from_numpy(np.asarray(head_maps)).float()
    target_tensor = torch.from_numpy(np.asarray(targets)).float()
    if target_tensor.ndim == 4 and target_tensor.shape[1] == 1:
        target_tensor = target_tensor[:, 0]

    if features.ndim != 5:
        raise ValueError(
            f"cached features must form [N,L,C,H,W], got {tuple(features.shape)}"
        )
    if heads.ndim != 3 or tuple(heads.shape[-2:]) != tuple(features.shape[-2:]):
        raise ValueError(
            "cached head_maps must be [N,H,W] and match feature spatial size"
        )
    if target_tensor.ndim != 3:
        raise ValueError(
            f"cached targets must be [N,H,W], got {tuple(target_tensor.shape)}"
        )
    if not (features.shape[0] == heads.shape[0] == target_tensor.shape[0]):
        raise ValueError("cached features, head maps, and targets have different counts")
    if tuple(target_tensor.shape[-2:]) != (64, 64):
        target_tensor = F.interpolate(
            target_tensor.unsqueeze(1),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return FeatureCache(features=features, head_maps=heads, targets=target_tensor)


def _loader(cache: FeatureCache, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(cache.features, cache.head_maps, cache.targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _loss_for_batch(
    model: FixedLayerProbes,
    batch: tuple[Tensor, Tensor, Tensor],
    layers: tuple[int, ...],
    device: torch.device,
) -> tuple[Tensor, dict[int, float]]:
    feature_stack, head_maps, targets = (tensor.to(device) for tensor in batch)
    feature_map = {layer: feature_stack[:, index] for index, layer in enumerate(layers)}
    logits = model(feature_map, head_maps)
    layer_losses = {
        layer: F.binary_cross_entropy_with_logits(output, targets)
        for layer, output in logits.items()
    }
    loss = torch.stack(tuple(layer_losses.values())).mean()
    return loss, {layer: value.detach().item() for layer, value in layer_losses.items()}


def _run_epoch(
    model: FixedLayerProbes,
    loader: DataLoader,
    layers: tuple[int, ...],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[int, float]]:
    model.train(optimizer is not None)
    total = 0.0
    per_layer = {layer: 0.0 for layer in layers}
    batches = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, layer_losses = _loss_for_batch(model, batch, layers, device)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.detach().item()
            for layer, value in layer_losses.items():
                per_layer[layer] += value
            batches += 1
    if batches == 0:
        raise ValueError("feature cache contains no samples")
    return total / batches, {layer: value / batches for layer, value in per_layer.items()}


def _load_live_frames(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    frames: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("frames"), list):
            frames.extend(dict(frame) for frame in item["frames"])
        else:
            frames.append(dict(item))
    return [frame for frame in frames if frame.get("heads")]


def _live_loader(
    *, json_path: Path, data_path: Path, transform, batch_size: int, shuffle: bool,
    num_workers: int,
) -> DataLoader:
    from PIL import Image
    from gazelle.utils import get_heatmap

    frames = _load_live_frames(json_path)

    class LiveFrameDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(frames)

        def __getitem__(self, index: int):
            frame = frames[index]
            image = Image.open(data_path / frame["path"]).convert("RGB")
            bboxes = []
            targets = []
            for head in frame["heads"]:
                if int(head.get("inout", 1)) != 1:
                    continue
                xs, ys = head.get("gazex_norm", []), head.get("gazey_norm", [])
                valid = [(float(x), float(y)) for x, y in zip(xs, ys) if x >= 0 and y >= 0]
                if not valid:
                    continue
                # Training annotations normally contain one point. For a
                # multi-annotation validation item, average Gaussian targets.
                heatmaps = [get_heatmap(x, y, 64, 64) for x, y in valid]
                bboxes.append(head["bbox_norm"])
                targets.append(torch.stack(heatmaps).mean(dim=0))
            return transform(image), bboxes, targets

    def collate(items):
        valid = [item for item in items if item[1]]
        if not valid:
            return None
        images, bboxes, targets = zip(*valid)
        return torch.stack(images), list(bboxes), torch.cat(
            [torch.stack(item) for item in targets], dim=0
        )

    return DataLoader(
        LiveFrameDataset(), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate,
    )


def _run_live_epoch(
    model: FixedLayerProbes,
    predictor,
    loader: DataLoader,
    layers: tuple[int, ...],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[int, float]]:
    model.train(optimizer is not None)
    predictor.eval()
    total = 0.0
    per_layer = {layer: 0.0 for layer in layers}
    batches = 0
    for batch in loader:
        if batch is None:
            continue
        images, bboxes, targets = batch
        images, targets = images.to(device), targets.to(device)
        counts = [len(items) for items in bboxes]
        with torch.no_grad():
            raw_features = predictor.backbone(images)
            person_features = {}
            repeat_index = torch.arange(images.shape[0], device=device).repeat_interleave(
                torch.as_tensor(counts, device=device)
            )
            for layer, feature in zip(layers, raw_features):
                person_features[layer] = feature.index_select(0, repeat_index)
            head_maps = torch.cat(predictor.get_input_head_maps(bboxes), dim=0).to(
                device=device, dtype=raw_features[0].dtype
            )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(person_features, head_maps)
        layer_losses = {
            layer: F.binary_cross_entropy_with_logits(output, targets)
            for layer, output in logits.items()
        }
        loss = torch.stack(tuple(layer_losses.values())).mean()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += loss.detach().item()
        for layer, value in layer_losses.items():
            per_layer[layer] += value.detach().item()
        batches += 1
    if batches == 0:
        raise ValueError("live split contains no usable in-frame person annotations")
    return total / batches, {layer: value / batches for layer, value in per_layer.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--val-cache", type=Path)

    # Formal-plan-compatible live feature extraction arguments.
    parser.add_argument("--model", default="gazelle_dinov3_vitb16")
    parser.add_argument(
        "--fusion",
        default="sasa",
        help="Base-predictor fusion architecture; must match --base-checkpoint.",
    )
    parser.add_argument(
        "--spatial-prior",
        default="ggsf",
        help="Base-predictor spatial prior; must match --base-checkpoint.",
    )
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic-samples", type=int, default=8)
    parser.add_argument("--synthetic-channels", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layers = tuple(args.layers)
    if layers != DEFAULT_LAYERS:
        raise ValueError(
            f"phase-one fixed probes require --layers {' '.join(map(str, DEFAULT_LAYERS))}"
        )
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch-size, and lr must all be positive")
    _seed_everything(args.seed)

    live_mode = False
    predictor = None
    if args.synthetic_smoke:
        train_cache = _synthetic_cache(
            samples=args.synthetic_samples,
            layers=layers,
            channels=args.synthetic_channels,
            seed=args.seed,
        )
        val_cache = _synthetic_cache(
            samples=max(4, args.synthetic_samples // 2),
            layers=layers,
            channels=args.synthetic_channels,
            seed=args.seed + 1,
        )
    elif args.train_cache is not None:
        train_cache = _load_npz_cache(args.train_cache, layers)
        val_cache = (
            _load_npz_cache(args.val_cache, layers)
            if args.val_cache is not None
            else train_cache
        )
    elif all(value is not None for value in (args.base_checkpoint, args.data_path, args.train_json, args.val_json)):
        from gazelle.model import get_gazelle_model

        predictor, transform = get_gazelle_model(
            args.model,
            fusion=args.fusion,
            spatial_prior=args.spatial_prior,
        )
        try:
            checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        predictor.load_gazelle_state_dict(checkpoint)
        predictor.requires_grad_(False).eval()
        channels = int(predictor.backbone.get_dimension())
        live_mode = True
        train_cache = val_cache = None
    else:
        raise RuntimeError(
            "Choose --synthetic-smoke, provide --train-cache, or provide the complete "
            "live pilot set: --base-checkpoint --data-path --train-json --val-json."
        )

    if not live_mode:
        channels = int(train_cache.features.shape[2])
        if int(val_cache.features.shape[2]) != channels:
            raise ValueError("train and validation feature channel counts differ")
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = FixedLayerProbes(
        in_channels=channels,
        layers=layers,
        hidden_channels=args.hidden_channels,
    ).to(device)
    if predictor is not None:
        predictor.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if live_mode:
        train_loader = _live_loader(
            json_path=args.train_json, data_path=args.data_path, transform=transform,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        )
        val_loader = _live_loader(
            json_path=args.val_json, data_path=args.data_path, transform=transform,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
    else:
        train_loader = _loader(train_cache, args.batch_size, shuffle=True)
        val_loader = _loader(val_cache, args.batch_size, shuffle=False)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        runner = _run_live_epoch if live_mode else _run_epoch
        runner_args = (model, predictor, train_loader, layers, device, optimizer) if live_mode else (model, train_loader, layers, device, optimizer)
        train_loss, train_by_layer = runner(*runner_args)
        runner_args = (model, predictor, val_loader, layers, device, None) if live_mode else (model, val_loader, layers, device, None)
        val_loss, val_by_layer = runner(*runner_args)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_by_layer": {str(k): v for k, v in train_by_layer.items()},
                "val_loss_by_layer": {str(k): v for k, v in val_by_layer.items()},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "layer_probes.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "layers": list(layers),
            "in_channels": channels,
            "hidden_channels": args.hidden_channels,
            "output_size": [64, 64],
            "seed": args.seed,
            "synthetic_smoke": args.synthetic_smoke,
            "live_backbone": live_mode,
            "base_predictor_fusion": args.fusion if live_mode else None,
            "base_predictor_spatial_prior": args.spatial_prior if live_mode else None,
        },
        checkpoint_path,
    )
    summary = {
        "status": "ok",
        "synthetic_smoke": args.synthetic_smoke,
        "backbone_loaded": live_mode,
        "layers": list(layers),
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.parameter_count(trainable_only=True),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "hidden_channels": args.hidden_channels,
            "device": str(device),
            "base_predictor_fusion": args.fusion if live_mode else None,
            "base_predictor_spatial_prior": args.spatial_prior if live_mode else None,
        },
        "inputs": (
            {
                "base_checkpoint": str(args.base_checkpoint),
                "base_checkpoint_sha256": _sha256(args.base_checkpoint),
                "train_json": str(args.train_json),
                "train_json_sha256": _sha256(args.train_json),
                "val_json": str(args.val_json),
                "val_json_sha256": _sha256(args.val_json),
            }
            if live_mode
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
