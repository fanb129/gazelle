"""Benchmark dense and coverage-routed GazeLLE inference on one CUDA device.

The benchmark intentionally separates synchronized single-request latency from
continuous-forward throughput. Checkpoint loading, data loading, and host to
device image transfer are excluded; the router's normal bbox handling remains
inside the routed model forward and is therefore included.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from gazelle.model import get_gazelle_model

try:  # Works as both ``python scripts/benchmark_...py`` and a module import.
    from eval_coverage_router import build_model
    from train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        _torch_load,
        load_model_state,
    )
except ModuleNotFoundError:
    from scripts.eval_coverage_router import build_model
    from scripts.train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        _torch_load,
        load_model_state,
    )


VARIANTS = {
    "dense": {"router_stage": None, "keep_ratio": 1.0},
    "support": {"router_stage": "support_pilot", "keep_ratio": 0.25},
    "k100": {"router_stage": "backbone_sparse", "keep_ratio": 1.0},
    "k50": {"router_stage": "backbone_sparse", "keep_ratio": 0.5},
    "k25": {"router_stage": "backbone_sparse", "keep_ratio": 0.25},
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and coverage-routed GazeLLE inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_people", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--latency_iters", type=int, default=200)
    parser.add_argument("--throughput_iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "num_people",
        "image_size",
        "latency_iters",
        "throughput_iters",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be non-negative")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must not contain duplicates")


def make_bboxes(batch_size: int, num_people: int):
    bboxes = []
    for _ in range(batch_size):
        image_boxes = []
        for person_index in range(num_people):
            offset = 0.04 * (person_index % 5)
            image_boxes.append(
                [
                    min(0.70, 0.15 + offset),
                    min(0.70, 0.10 + offset),
                    min(0.92, 0.32 + offset),
                    min(0.92, 0.35 + offset),
                ]
            )
        bboxes.append(image_boxes)
    return bboxes


def build_dense_model(checkpoint: dict, model_config: dict):
    model, _ = get_gazelle_model(
        model_config["model"],
        spatial_prior=model_config["spatial_prior"],
        fusion=model_config["fusion"],
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state")
    dense_state = {
        name: value for name, value in state.items() if not name.startswith("router.")
    }
    incompatible = model.load_state_dict(dense_state, strict=False)
    missing_required = [
        name for name in incompatible.missing_keys if not name.startswith("backbone.")
    ]
    if missing_required:
        raise RuntimeError(
            f"dense model is missing non-backbone checkpoint state: {missing_required}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"dense model received unexpected checkpoint state: "
            f"{incompatible.unexpected_keys}"
        )
    return model


def build_variant(checkpoint: dict, variant_name: str):
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint has no model_config")
    variant = VARIANTS[variant_name]
    if variant_name == "dense":
        return build_dense_model(checkpoint, checkpoint_config)

    model_config = dict(checkpoint_config)
    model_config["router_stage"] = variant["router_stage"]
    model_config["keep_ratio"] = variant["keep_ratio"]
    model, _ = build_model(model_config)
    model.set_trainable_backbone_suffix(False)
    load_model_state(model, checkpoint, initialization=False)
    return model


def percentile(values, fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def amp_context(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def benchmark_variant(
    model,
    model_input: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    latency_ms = []
    latency_repeat_medians = []
    throughput_fps = []
    output = None

    with torch.inference_mode(), amp_context(args.amp):
        for _ in range(args.warmup_iters):
            output = model(model_input)
        del output
        output = None
        synchronize(device)

        torch.cuda.empty_cache()
        synchronize(device)
        steady_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(args.repeats):
            repeat_latency_ms = []
            for _ in range(args.latency_iters):
                synchronize(device)
                start = time.perf_counter()
                output = model(model_input)
                synchronize(device)
                duration_ms = (time.perf_counter() - start) * 1000.0
                latency_ms.append(duration_ms)
                repeat_latency_ms.append(duration_ms)
                del output
                output = None
            latency_repeat_medians.append(statistics.median(repeat_latency_ms))

            synchronize(device)
            start = time.perf_counter()
            for _ in range(args.throughput_iters):
                output = model(model_input)
            synchronize(device)
            elapsed = time.perf_counter() - start
            throughput_fps.append(
                args.throughput_iters * args.batch_size / elapsed
            )
            del output
            output = None

        synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

    return {
        "latency_ms_per_batch_mean": statistics.fmean(latency_ms),
        "latency_ms_per_batch_median": statistics.median(latency_ms),
        "latency_ms_per_batch_p95": percentile(latency_ms, 0.95),
        "latency_ms_per_batch_std": (
            statistics.stdev(latency_ms) if len(latency_ms) > 1 else 0.0
        ),
        "latency_ms_per_image_median": (
            statistics.median(latency_ms) / args.batch_size
        ),
        "throughput_fps_mean": statistics.fmean(throughput_fps),
        "throughput_fps_median": statistics.median(throughput_fps),
        "throughput_fps_std": (
            statistics.stdev(throughput_fps)
            if len(throughput_fps) > 1
            else 0.0
        ),
        "steady_allocated_mib": steady_allocated / (1024.0**2),
        "peak_allocated_mib": peak_allocated / (1024.0**2),
        "activation_peak_mib": (
            max(0, peak_allocated - steady_allocated) / (1024.0**2)
        ),
        "latency_sample_count": len(latency_ms),
        "latency_repeat_medians_ms": latency_repeat_medians,
        "throughput_repeat_count": len(throughput_fps),
        "throughput_repeat_fps": throughput_fps,
    }


def token_metadata(model, variant_name: str, image_size: int) -> dict:
    patch_height, patch_width = model.backbone.get_out_size(
        (image_size, image_size)
    )
    patch_tokens = int(patch_height * patch_width)
    ratio = float(VARIANTS[variant_name]["keep_ratio"])
    kept_patch_tokens = (
        patch_tokens
        if variant_name in ("dense", "support")
        else min(patch_tokens, max(1, int(round(ratio * patch_tokens))))
    )
    special_tokens = int(getattr(model.backbone, "prefix_token_count", 0))
    return {
        "patch_grid": [int(patch_height), int(patch_width)],
        "dense_patch_tokens": patch_tokens,
        "kept_patch_tokens": kept_patch_tokens,
        "special_tokens": special_tokens,
        "suffix_sequence_tokens": kept_patch_tokens + special_tokens,
        "route_after_block": (
            None
            if variant_name == "dense"
            else int(getattr(model, "route_after_block"))
        ),
    }


def add_relative_metrics(results: list) -> None:
    dense = next(
        (item for item in results if item["variant"] == "dense"),
        None,
    )
    if dense is None:
        return
    dense_latency = dense["timing"]["latency_ms_per_batch_median"]
    dense_fps = dense["timing"]["throughput_fps_median"]
    dense_peak = dense["timing"]["peak_allocated_mib"]
    for item in results:
        timing = item["timing"]
        timing["latency_speedup_vs_dense"] = (
            dense_latency / timing["latency_ms_per_batch_median"]
        )
        timing["latency_reduction_vs_dense"] = (
            1.0 - timing["latency_ms_per_batch_median"] / dense_latency
        )
        timing["throughput_gain_vs_dense"] = (
            timing["throughput_fps_median"] / dense_fps - 1.0
        )
        timing["peak_memory_reduction_vs_dense"] = (
            1.0 - timing["peak_allocated_mib"] / dense_peak
        )


def render_table(results: list) -> str:
    header = (
        "| Variant | Kept patches | Median latency (ms/batch) | "
        "p95 (ms) | Median FPS | Peak allocated (MiB) | Speedup vs dense |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for item in results:
        timing = item["timing"]
        speedup = timing.get("latency_speedup_vs_dense")
        rows.append(
            "| {variant} | {kept} | {latency:.3f} | {p95:.3f} | "
            "{fps:.3f} | {memory:.1f} | {speedup} |".format(
                variant=item["variant"],
                kept=item["tokens"]["kept_patch_tokens"],
                latency=timing["latency_ms_per_batch_median"],
                p95=timing["latency_ms_per_batch_p95"],
                fps=timing["throughput_fps_median"],
                memory=timing["peak_allocated_mib"],
                speedup=(
                    f"{speedup:.3f}×" if speedup is not None else "N/A"
                ),
            )
        )
    return "\n".join(rows)


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "coverage-router performance claims require a CUDA device; "
            "CPU benchmarking is intentionally unsupported"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("--checkpoint must be a structured checkpoint")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or missing checkpoint format_version")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)
    model_input = {
        "images": images,
        "bboxes": make_bboxes(args.batch_size, args.num_people),
    }

    results = []
    for variant_name in args.variants:
        print(f"BENCHMARK variant={variant_name}", flush=True)
        model = build_variant(checkpoint, variant_name).to(device).eval()
        tokens = token_metadata(model, variant_name, args.image_size)
        timing = benchmark_variant(model, model_input, args, device)
        results.append(
            {
                "variant": variant_name,
                "router_stage": VARIANTS[variant_name]["router_stage"],
                "keep_ratio": VARIANTS[variant_name]["keep_ratio"],
                "tokens": tokens,
                "timing": timing,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device)

    add_relative_metrics(results)
    table = render_table(results)
    payload = {
        "metadata": {
            "script": "scripts/benchmark_coverage_router.py",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_git_commit": checkpoint.get("git_commit"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "amp": bool(args.amp),
            "amp_dtype": "float16" if args.amp else "float32",
            "batch_size": args.batch_size,
            "num_people_per_image": args.num_people,
            "image_size": [args.image_size, args.image_size],
            "input_source": "fixed_seed_dummy_preloaded_on_device",
            "warmup_iters": args.warmup_iters,
            "latency_iters": args.latency_iters,
            "throughput_iters": args.throughput_iters,
            "repeats": args.repeats,
            "single_request_latency_synchronizes_every_iteration": True,
            "throughput_synchronizes_only_at_repeat_boundaries": True,
            "checkpoint_loading_and_image_h2d_excluded": True,
        },
        "results": results,
        "markdown_table": table,
    }
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(table)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(f"Saved benchmark to {output}")
    return payload


if __name__ == "__main__":
    main()
