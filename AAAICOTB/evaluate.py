#!/usr/bin/env python3
"""Evaluate standard VAT metrics and COTB observer--target binding metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from AAAICOTB.binding import BindingConfig
from AAAICOTB.data import VATFrameDataset, collate_frames
from AAAICOTB.engine import evaluate_loader, write_evaluation
from AAAICOTB.models import build_model, load_evaluation_checkpoint
from AAAIScripts.common import file_manifest


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-source", choices=("current", "v0"), default="current")
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--spatial-prior", default="none")
    parser.add_argument("--fusion", default="raw_concat")
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--batch-size-frames", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--min-pair-separation", type=float, default=0.10)
    parser.add_argument("--target-sigma", type=float, default=0.04)
    parser.add_argument("--bind-margin", type=float, default=0.20)
    parser.add_argument("--duplicate-bbox-iou", type=float, default=0.80)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("Full evaluation expects CUDA; use --allow-cpu only for a tiny smoke test")

    model, transform = build_model(
        args.model_source, args.model, args.spatial_prior, args.fusion, args.selected_layers
    )
    checkpoint_load = load_evaluation_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    dataset = VATFrameDataset(
        args.data_path,
        args.annotation,
        transform,
        frame_sample_every=args.frame_sample_every,
        augment=False,
    )
    if args.max_frames is not None:
        dataset.items = dataset.items[: args.max_frames]
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_frames,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_frames,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    config = BindingConfig(
        shared_target_radius=args.shared_target_radius,
        min_pair_separation=args.min_pair_separation,
        target_sigma=args.target_sigma,
        margin=args.bind_margin,
        duplicate_bbox_iou=args.duplicate_bbox_iou,
    )
    summary, persons, pairs, queries = evaluate_loader(
        model,
        loader,
        device,
        config,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        max_batches=args.max_batches,
    )
    report = {
        "status": "measured_raw_predictions",
        "model_label": args.model_label,
        "model_source": args.model_source,
        "model": args.model,
        "architecture": {
            "spatial_prior": None if args.model_source == "v0" else args.spatial_prior,
            "fusion": None if args.model_source == "v0" else args.fusion,
            "selected_layers": None if args.model_source == "v0" else args.selected_layers,
        },
        "annotation": file_manifest(args.annotation),
        "checkpoint_load": checkpoint_load,
        "binding_config": config.as_dict(),
        "frame_sample_every": args.frame_sample_every,
        "evaluated_frames": len({row["path"] for row in persons}),
        **summary,
    }
    write_evaluation(args.output_dir, report, persons, pairs, queries)
    print(json.dumps({"output": str(args.output_dir), "standard": summary["standard"], "binding_overall": summary["binding"]["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
