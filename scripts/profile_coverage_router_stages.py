"""Diagnose where coverage-routed inference spends CUDA time.

This script deliberately keeps two timing lines separate:

* uninstrumented synchronized wall-clock latency is the end-to-end reference;
* CUDA-event stage timing is diagnostic only and must not replace the clean
  benchmark reported by ``benchmark_coverage_router.py``.

Images are generated once and preloaded on the selected CUDA device.  Model
loading, data loading, preprocessing, and image H2D transfer are excluded.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Iterable, Optional

import torch

try:  # Works as a script and as an imported module in tests.
    from benchmark_coverage_router import (
        VARIANTS,
        amp_context,
        build_variant,
        equalize_autocast_weight_cache,
        make_bboxes,
        synchronize,
        token_metadata,
    )
    from train_coverage_router import CHECKPOINT_FORMAT_VERSION, _torch_load
except ModuleNotFoundError:
    from scripts.benchmark_coverage_router import (
        VARIANTS,
        amp_context,
        build_variant,
        equalize_autocast_weight_cache,
        make_bboxes,
        synchronize,
        token_metadata,
    )
    from scripts.train_coverage_router import CHECKPOINT_FORMAT_VERSION, _torch_load


SPARSE_PHASES = (
    "token_prepare_rope",
    "prefix_blocks",
    "prefix_maps",
    "route_map",
    "router",
    "sparse_prepare",
    "suffix_blocks",
    "scatter_norm",
    "decoder",
)
SUPPORT_PHASES = ("dense_backbone", "router", "decoder")
DENSE_PHASES = ("dense_backbone", "decoder")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile mutually exclusive stages of coverage-routed inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["k100", "k50"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_people", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--e2e_iters", type=int, default=100)
    parser.add_argument("--profile_iters", type=int, default=100)
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
        "e2e_iters",
        "profile_iters",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be non-negative")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must not contain duplicates")
    if Path(args.output).suffix.lower() != ".json":
        raise ValueError("--output must end in .json so the Markdown sidecar is distinct")
    missing_matched_variants = {"k100", "k50"} - set(args.variants)
    if missing_matched_variants:
        missing = ", ".join(sorted(missing_matched_variants))
        raise ValueError(
            "--variants must include both k100 and k50 for the matched "
            f"stage comparison; missing: {missing}"
        )


class CudaStageRecorder:
    """Collect non-overlapping stage intervals using CUDA events.

    Events are synchronized only after the complete forward pass.  Synchronizing
    inside each stage would distort the execution pattern this tool is trying to
    diagnose.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.pending = {}

    def reset(self) -> None:
        self.pending = {}

    @contextmanager
    def stage(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self.pending.setdefault(name, []).append((start, end))

    def elapsed_ms(self) -> dict:
        return {
            name: sum(start.elapsed_time(end) for start, end in intervals)
            for name, intervals in self.pending.items()
        }


def aggregate_stage_values(raw: dict) -> dict:
    """Aggregate detailed per-block events into mutually exclusive phases."""
    aggregated = {
        name: float(value)
        for name, value in raw.items()
        if not (
            name.startswith("prefix.block_")
            or name.startswith("prefix.map_")
            or name.startswith("suffix.block_")
            or name.startswith("scatter_norm_")
        )
    }
    aggregated["prefix_blocks"] = sum(
        value for name, value in raw.items() if name.startswith("prefix.block_")
    )
    aggregated["prefix_maps"] = sum(
        value for name, value in raw.items() if name.startswith("prefix.map_")
    )
    aggregated["suffix_blocks"] = sum(
        value for name, value in raw.items() if name.startswith("suffix.block_")
    )
    aggregated["scatter_norm"] = sum(
        value for name, value in raw.items() if name.startswith("scatter_norm_")
    )
    return aggregated


def expected_raw_stage_names(model, variant_name: str) -> set:
    if variant_name == "dense":
        return set(DENSE_PHASES)
    if variant_name == "support":
        return set(SUPPORT_PHASES)

    route_after_block = int(model.route_after_block)
    out_indices = tuple(int(index) for index in model.backbone.out_indices)
    num_blocks = len(model.backbone.model.blocks)
    names = {
        "token_prepare_rope",
        "route_map",
        "router",
        "sparse_prepare",
        "decoder",
    }
    names.update(
        f"prefix.block_{index}" for index in range(route_after_block + 1)
    )
    names.update(
        f"prefix.map_{index}"
        for index in out_indices
        if index <= route_after_block
    )
    names.update(
        f"suffix.block_{index}"
        for index in range(route_after_block + 1, num_blocks)
    )
    names.update(
        f"scatter_norm_{index}"
        for index in out_indices
        if index > route_after_block
    )
    return names


def validate_raw_stage_names(raw: dict, model, variant_name: str) -> None:
    expected = expected_raw_stage_names(model, variant_name)
    actual = set(raw)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise RuntimeError(
        f"incomplete stage instrumentation for {variant_name}: "
        f"missing={missing}, unexpected={unexpected}"
    )


def phases_for_variant(variant_name: str) -> tuple:
    if variant_name == "dense":
        return DENSE_PHASES
    if variant_name == "support":
        return SUPPORT_PHASES
    return SPARSE_PHASES


def profiled_forward(model, model_input: dict, variant_name: str, recorder):
    if variant_name == "dense":
        with recorder.stage("dense_backbone"):
            raw_features = model.backbone(model_input["images"])
        with recorder.stage("decoder"):
            return model.forward_from_features(model_input, raw_features)
    return model.forward_profiled(model_input, recorder)


def assert_outputs_close(reference, candidate, path="output") -> None:
    """Guard against diagnostic-path drift from the production forward graph."""
    if torch.is_tensor(reference) or torch.is_tensor(candidate):
        if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
            raise AssertionError(f"{path} tensor structure differs")
        torch.testing.assert_close(reference, candidate, rtol=1e-4, atol=1e-5)
        return
    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not isinstance(reference, dict) or not isinstance(candidate, dict):
            raise AssertionError(f"{path} mapping structure differs")
        if reference.keys() != candidate.keys():
            raise AssertionError(f"{path} keys differ")
        for key in reference:
            assert_outputs_close(reference[key], candidate[key], f"{path}.{key}")
        return
    if isinstance(reference, (list, tuple)) or isinstance(candidate, (list, tuple)):
        if not isinstance(reference, (list, tuple)) or not isinstance(candidate, (list, tuple)):
            raise AssertionError(f"{path} sequence structure differs")
        if len(reference) != len(candidate):
            raise AssertionError(f"{path} lengths differ")
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            assert_outputs_close(reference_item, candidate_item, f"{path}[{index}]")
        return
    if is_dataclass(reference) or is_dataclass(candidate):
        if not is_dataclass(reference) or not is_dataclass(candidate):
            raise AssertionError(f"{path} dataclass structure differs")
        if type(reference) is not type(candidate):
            raise AssertionError(f"{path} dataclass types differ")
        for field in fields(reference):
            assert_outputs_close(
                getattr(reference, field.name),
                getattr(candidate, field.name),
                f"{path}.{field.name}",
            )
        return
    if reference != candidate:
        raise AssertionError(f"{path} differs: {reference!r} != {candidate!r}")


def validate_profiled_forward(model, model_input, variant_name: str, device) -> None:
    reference = model(model_input)
    recorder = CudaStageRecorder(device)
    candidate = profiled_forward(model, model_input, variant_name, recorder)
    synchronize(device)
    assert_outputs_close(reference, candidate)


def coefficient_of_variation(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return 0.0 if mean == 0.0 else statistics.stdev(values) / mean


def summarize_scalar_repeats(repeat_values: list) -> dict:
    medians = [statistics.median(values) for values in repeat_values]
    return {
        "repeat_medians_ms": medians,
        "median_ms": statistics.median(medians),
        "mean_ms": statistics.fmean(medians),
        "min_ms": min(medians),
        "max_ms": max(medians),
        "repeat_median_cv": coefficient_of_variation(medians),
    }


def summarize_stage_repeats(repeat_samples: list, phase_names: tuple) -> dict:
    summaries = {}
    for phase_name in phase_names:
        values_by_repeat = [
            [sample[phase_name] for sample in repeat]
            for repeat in repeat_samples
        ]
        summaries[phase_name] = summarize_scalar_repeats(values_by_repeat)

    full_values = [
        [sample["instrumented_full_cuda"] for sample in repeat]
        for repeat in repeat_samples
    ]
    summaries["instrumented_full_cuda"] = summarize_scalar_repeats(full_values)

    phase_sum_values = []
    unattributed_values = []
    for repeat in repeat_samples:
        phase_sum_repeat = []
        unattributed_repeat = []
        for sample in repeat:
            phase_sum = sum(sample[name] for name in phase_names)
            phase_sum_repeat.append(phase_sum)
            unattributed_repeat.append(sample["instrumented_full_cuda"] - phase_sum)
        phase_sum_values.append(phase_sum_repeat)
        unattributed_values.append(unattributed_repeat)
    summaries["phase_sum"] = summarize_scalar_repeats(phase_sum_values)
    summaries["unattributed"] = summarize_scalar_repeats(unattributed_values)

    full_median = summaries["instrumented_full_cuda"]["median_ms"]
    for phase_name in phase_names:
        summaries[phase_name]["share_of_instrumented_full"] = (
            summaries[phase_name]["median_ms"] / full_median
            if full_median > 0.0
            else 0.0
        )
    return summaries


def benchmark_uninstrumented(model, model_input, args, device) -> dict:
    repeats = []
    for _ in range(args.repeats):
        values = []
        for _ in range(args.e2e_iters):
            synchronize(device)
            start = time.perf_counter()
            output = model(model_input)
            synchronize(device)
            values.append((time.perf_counter() - start) * 1000.0)
            del output
        repeats.append(values)
    return summarize_scalar_repeats(repeats)


def benchmark_profiled(model, model_input, variant_name, args, device) -> dict:
    recorder = CudaStageRecorder(device)
    repeat_samples = []
    for _ in range(args.repeats):
        samples = []
        for _ in range(args.profile_iters):
            recorder.reset()
            stream = torch.cuda.current_stream(device)
            full_start = torch.cuda.Event(enable_timing=True)
            full_end = torch.cuda.Event(enable_timing=True)
            full_start.record(stream)
            output = profiled_forward(model, model_input, variant_name, recorder)
            full_end.record(stream)
            full_end.synchronize()

            raw = recorder.elapsed_ms()
            validate_raw_stage_names(raw, model, variant_name)
            sample = aggregate_stage_values(raw)
            sample["instrumented_full_cuda"] = full_start.elapsed_time(full_end)
            samples.append(sample)
            del output
        repeat_samples.append(samples)
    return summarize_stage_repeats(
        repeat_samples,
        phases_for_variant(variant_name),
    )


def compare_k50_k100(results: list) -> Optional[dict]:
    by_name = {item["variant"]: item for item in results}
    if "k100" not in by_name or "k50" not in by_name:
        return None
    k100 = by_name["k100"]
    k50 = by_name["k50"]
    comparison = {
        "e2e_uninstrumented": _delta_summary(
            k100["e2e_uninstrumented"]["median_ms"],
            k50["e2e_uninstrumented"]["median_ms"],
        ),
        "stages": {},
    }
    for phase_name in SPARSE_PHASES + (
        "phase_sum",
        "unattributed",
        "instrumented_full_cuda",
    ):
        comparison["stages"][phase_name] = _delta_summary(
            k100["profiled"][phase_name]["median_ms"],
            k50["profiled"][phase_name]["median_ms"],
        )
    return comparison


def _delta_summary(reference_ms: float, candidate_ms: float) -> dict:
    delta_ms = candidate_ms - reference_ms
    return {
        "k100_ms": reference_ms,
        "k50_ms": candidate_ms,
        "k50_minus_k100_ms": delta_ms,
        "k50_minus_k100_fraction": (
            delta_ms / reference_ms if reference_ms != 0.0 else None
        ),
    }


def render_markdown(results: list, comparison: Optional[dict]) -> str:
    lines = [
        "## Uninstrumented end-to-end latency",
        "",
        "| Variant | Median of repeat medians (ms) | Min repeat (ms) | Max repeat (ms) | CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        summary = item["e2e_uninstrumented"]
        lines.append(
            "| {variant} | {median:.4f} | {minimum:.4f} | {maximum:.4f} | {cv:.2%} |".format(
                variant=item["variant"],
                median=summary["median_ms"],
                minimum=summary["min_ms"],
                maximum=summary["max_ms"],
                cv=summary["repeat_median_cv"],
            )
        )

    lines.extend(
        [
            "",
            "## CUDA-event stage diagnosis",
            "",
            "Instrumented stage times are diagnostic only; do not use them as the paper's headline latency.",
            "",
        ]
    )
    for item in results:
        lines.extend(
            [
                f"### {item['variant']}",
                "",
                "| Stage | Median (ms) | Share of instrumented full |",
                "|---|---:|---:|",
            ]
        )
        for phase_name in phases_for_variant(item["variant"]):
            summary = item["profiled"][phase_name]
            lines.append(
                f"| {phase_name} | {summary['median_ms']:.4f} | "
                f"{summary['share_of_instrumented_full']:.2%} |"
            )
        for total_name in ("phase_sum", "unattributed", "instrumented_full_cuda"):
            summary = item["profiled"][total_name]
            lines.append(f"| {total_name} | {summary['median_ms']:.4f} | — |")
        lines.append("")

    if comparison is not None:
        lines.extend(
            [
                "## K50 minus K100",
                "",
                "Positive values mean K50 is slower; negative values mean K50 saves time.",
                "",
                "| Stage | K100 (ms) | K50 (ms) | Delta (ms) | Delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        e2e = comparison["e2e_uninstrumented"]
        lines.append(_comparison_row("e2e_uninstrumented", e2e))
        for phase_name in SPARSE_PHASES + (
            "phase_sum",
            "unattributed",
            "instrumented_full_cuda",
        ):
            lines.append(_comparison_row(phase_name, comparison["stages"][phase_name]))
    return "\n".join(lines).rstrip() + "\n"


def _comparison_row(name: str, values: dict) -> str:
    fraction = values["k50_minus_k100_fraction"]
    rendered_fraction = "N/A" if fraction is None else f"{fraction:.2%}"
    return (
        f"| {name} | {values['k100_ms']:.4f} | {values['k50_ms']:.4f} | "
        f"{values['k50_minus_k100_ms']:+.4f} | {rendered_fraction} |"
    )


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("stage profiling requires a CUDA device")
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
        print(f"PROFILE variant={variant_name}", flush=True)
        model = build_variant(checkpoint, variant_name)
        parameters = equalize_autocast_weight_cache(model)
        model = model.to(device).eval()

        with torch.inference_mode(), amp_context(args.amp):
            output = None
            for _ in range(args.warmup_iters):
                output = model(model_input)
            del output
            synchronize(device)

            validate_profiled_forward(model, model_input, variant_name, device)

            e2e = benchmark_uninstrumented(model, model_input, args, device)
            profiled = benchmark_profiled(
                model,
                model_input,
                variant_name,
                args,
                device,
            )

        results.append(
            {
                "variant": variant_name,
                "router_stage": VARIANTS[variant_name]["router_stage"],
                "keep_ratio": VARIANTS[variant_name]["keep_ratio"],
                "parameters": parameters,
                "tokens": token_metadata(model, variant_name, args.image_size),
                "e2e_uninstrumented": e2e,
                "profiled": profiled,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device)

    comparison = compare_k50_k100(results)
    markdown = render_markdown(results, comparison)
    payload = {
        "metadata": {
            "script": "scripts/profile_coverage_router_stages.py",
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
            "autocast_weight_cache_enabled": True,
            "parameter_requires_grad_equalized_for_autocast_cache": True,
            "batch_size": args.batch_size,
            "num_people_per_image": args.num_people,
            "image_size": [args.image_size, args.image_size],
            "input_source": "fixed_seed_dummy_preloaded_on_device",
            "warmup_iters": args.warmup_iters,
            "e2e_iters": args.e2e_iters,
            "profile_iters": args.profile_iters,
            "repeats": args.repeats,
            "stage_events_synchronize_only_after_complete_forward": True,
            "instrumented_stage_times_are_diagnostic_only": True,
            "profiled_forward_output_checked_against_normal": True,
            "checkpoint_loading_and_image_h2d_excluded": True,
        },
        "results": results,
        "k50_vs_k100": comparison,
        "markdown": markdown,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f"Saved stage profile to {output_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
