import argparse
import csv
import importlib
import json
import platform
import time
from pathlib import Path


MODEL_VARIANTS = {
    "gf_baseline": {
        "dataset": "GazeFollow",
        "method": "Baseline (GF)",
        "module": "gazelle.model_v0",
        "factory": "gazelle_dinov3_vitb16",
        "factory_kwargs": {},
        "input_hw": (448, 448),
    },
    "gf_gazespot": {
        "dataset": "GazeFollow",
        "method": "GazeSpot (GF)",
        "module": "gazelle.model",
        "factory": "gazelle_dinov3_vitb16",
        "factory_kwargs": {"sasa": True, "ggsf": True, "aux": False},
        "input_hw": (512, 512),
    },
    "vat_baseline": {
        "dataset": "VAT",
        "method": "Baseline (VAT)",
        "module": "gazelle.model_v0",
        "factory": "gazelle_dinov3_vitb16_inout",
        "factory_kwargs": {},
        "input_hw": (448, 448),
    },
    "vat_gazespot": {
        "dataset": "VAT",
        "method": "GazeSpot (VAT)",
        "module": "gazelle.model",
        "factory": "gazelle_dinov3_vitb16_inout",
        "factory_kwargs": {"sasa": True, "ggsf": True, "aux": False},
        "input_hw": (512, 512),
    },
}

CSV_FIELDS = [
    "Dataset",
    "Method",
    "Input Size",
    "Batch Size",
    "Trainable Params",
    "Total Params",
    "Non-backbone Parameters (diagnostic)",
    "MACs",
    "FLOPs",
    "MACs/FLOPs",
    "Latency ms/img",
    "FPS",
    "Device",
    "Profiling Tool",
]


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to profile complexity/runtime. "
            "Run this script in the Gazelle experiment environment."
        ) from exc
    return torch


def require_thop():
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError(
            "THOP is required to compute MACs/FLOPs. Install thop in the "
            "Gazelle experiment environment before running this command."
        ) from exc
    return profile


def humanize_count(value):
    value = float(value)
    for suffix in ("", "K", "M", "G", "T"):
        if abs(value) < 1000.0 or suffix == "T":
            return f"{value:.2f}{suffix}"
        value /= 1000.0
    return f"{value:.2f}T"


def round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def freeze_backbone_for_reporting(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for param in backbone.parameters():
        param.requires_grad = False


def count_parameters(model):
    freeze_backbone_for_reporting(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_backbone_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("backbone")
    )
    return {
        "Trainable Params": int(trainable_params),
        "Total Params": int(total_params),
        "Non-backbone Parameters (diagnostic)": int(non_backbone_params),
    }


class ModelWrapper:
    def __init__(self, torch, model, batch_size):
        nn = torch.nn

        class _Wrapper(nn.Module):
            def __init__(self, wrapped_model, wrapped_batch_size):
                super().__init__()
                self.model = wrapped_model
                self.dummy_bboxes = [
                    [[0.3, 0.3, 0.6, 0.6]] for _ in range(wrapped_batch_size)
                ]

            def forward(self, images):
                return self.model({"images": images, "bboxes": self.dummy_bboxes})

        self.module = _Wrapper(model, batch_size)


def resolve_device(torch, requested_device):
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA timing was requested, but torch.cuda.is_available() is false. "
            "Refusing to report CPU numbers as RTX/CUDA latency or FPS."
        )
    return torch.device(requested_device)


def get_device_name(torch, device):
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    return platform.processor() or platform.machine() or "CPU"


def synchronize_if_cuda(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def import_factory(variant):
    module = importlib.import_module(variant["module"])
    return getattr(module, variant["factory"])


def build_model(variant):
    factory = import_factory(variant)
    model, _ = factory(**variant["factory_kwargs"])
    return model


def profile_macs(torch, profile, wrapped_model, dummy_image, batch_size):
    with torch.no_grad():
        macs_batch, _ = profile(wrapped_model, inputs=(dummy_image,), verbose=False)
    return float(macs_batch) / float(batch_size)


def measure_latency(torch, wrapped_model, dummy_image, device, warmup_iters, measure_iters, batch_size):
    with torch.inference_mode():
        for _ in range(warmup_iters):
            wrapped_model(dummy_image)
        synchronize_if_cuda(torch, device)

        start = time.perf_counter()
        for _ in range(measure_iters):
            wrapped_model(dummy_image)
        synchronize_if_cuda(torch, device)
        elapsed_s = time.perf_counter() - start

    images = measure_iters * batch_size
    return {
        "Latency ms/img": (elapsed_s * 1000.0) / images,
        "FPS": images / elapsed_s if elapsed_s > 0 else None,
    }


def profile_variant(torch, profile, variant_key, variant, args, device, device_name):
    model = build_model(variant)
    model.to(device)
    model.eval()
    freeze_backbone_for_reporting(model)

    batch_size = args.batch_size
    height, width = variant["input_hw"]
    dummy_image = torch.randn(batch_size, 3, height, width, device=device)
    wrapped_model = ModelWrapper(torch, model, batch_size).module.to(device)
    wrapped_model.eval()

    param_counts = count_parameters(model)
    macs = profile_macs(torch, profile, wrapped_model, dummy_image, batch_size)
    flops = macs * 2.0
    latency = measure_latency(
        torch,
        wrapped_model,
        dummy_image,
        device,
        args.warmup_iters,
        args.measure_iters,
        batch_size,
    )

    return {
        "Variant": variant_key,
        "Dataset": variant["dataset"],
        "Method": variant["method"],
        "Input Size": f"{height}x{width}",
        "Batch Size": batch_size,
        **param_counts,
        "MACs": int(macs),
        "FLOPs": int(flops),
        "MACs/FLOPs": f"{humanize_count(macs)} MACs / {humanize_count(flops)} FLOPs",
        "Latency ms/img": round_or_none(latency["Latency ms/img"]),
        "FPS": round_or_none(latency["FPS"]),
        "Device": device_name,
        "Profiling Tool": "THOP",
    }


def make_markdown_table(rows):
    headers = [
        "Method",
        "Input",
        "Trainable Params",
        "Total Params",
        "MACs/FLOPs",
        "Latency ms/img",
        "FPS",
        "Device",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {Method} | {Input Size} | {Trainable Params} | {Total Params} | "
            "{MACs/FLOPs} | {Latency ms/img} | {FPS} | {Device} |".format(**row)
        )
    return "\n".join(lines)


def write_outputs(output_path, payload, rows):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = path.with_suffix(".md")
    md_path.write_text(make_markdown_table(rows) + "\n", encoding="utf-8")
    return path, csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report Gazelle/GazeSpot complexity, parameter counts, latency, and FPS."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for profiling: cuda, cuda:0, cpu, or auto. Defaults to cuda.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--measure_iters", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_VARIANTS),
        default=list(MODEL_VARIANTS),
        help="Subset of model variants to profile.",
    )
    parser.add_argument(
        "--output",
        "--output_path",
        dest="output",
        default=None,
        help="Path for machine-readable JSON. CSV and Markdown sidecars are also written.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be non-negative.")
    if args.measure_iters <= 0:
        raise ValueError("--measure_iters must be positive.")


def main():
    args = parse_args()
    validate_args(args)

    torch = require_torch()
    profile = require_thop()
    device = resolve_device(torch, args.device)
    device_name = get_device_name(torch, device)

    rows = []
    for variant_key in args.models:
        variant = MODEL_VARIANTS[variant_key]
        print(
            f"Profiling {variant['method']} at {variant['input_hw'][0]}x"
            f"{variant['input_hw'][1]} on {device_name}..."
        )
        rows.append(profile_variant(torch, profile, variant_key, variant, args, device, device_name))

    payload = {
        "metadata": {
            "script": "scripts/compute_flops.py",
            "device_requested": args.device,
            "device": str(device),
            "device_name": device_name,
            "batch_size": args.batch_size,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "cuda_synchronized": device.type == "cuda",
            "macs_are_per_image": True,
            "flops_are_estimated_as_2x_macs": True,
        },
        "results": rows,
        "markdown_table": make_markdown_table(rows),
    }

    print("\n" + payload["markdown_table"])
    if args.output:
        json_path, csv_path, md_path = write_outputs(args.output, payload, rows)
        print(f"\nWrote JSON: {json_path}")
        print(f"Wrote CSV: {csv_path}")
        print(f"Wrote Markdown: {md_path}")
    else:
        print("\nNo --output provided; JSON/CSV/Markdown files were not written.")


if __name__ == "__main__":
    main()
