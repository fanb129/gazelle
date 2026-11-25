"""Matched multi-person timing for the same selected Router weights only."""

import math

import torch

from cscwd_router.code.stag import final_io as io
from cscwd_router.code.stag.final_multipeople import IndependentQueries
from scripts.train_coverage_router import stack_predictions

VARIANTS = ("k100", "independent", "union")


def build_variant(source, payload, variant):
    model, transform = io.build_router(source, payload, keep_ratio=1. if variant == "k100" else .5)
    if variant == "independent":
        model = IndependentQueries(model)
    return model.eval().requires_grad_(False), transform


@torch.inference_mode()
def trace(model, inputs, variant):
    seen = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, values, index=index: seen.append({"block": index,
            "batch": values[0].shape[0], "tokens": values[0].shape[1]})
    ) for index, block in enumerate(model.backbone.model.blocks)]
    try:
        with torch.autocast(inputs["images"].device.type, dtype=torch.float16,
                            enabled=inputs["images"].device.type == "cuda", cache_enabled=False):
            output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    people = sum(len(boxes) for boxes in inputs["bboxes"])
    copies = people if variant == "independent" else len(inputs["bboxes"])
    kept = 1024 if variant == "k100" else 512
    expected = [{"block": index, "batch": copies,
        "tokens": (1024 if index <= model.route_after_block else kept) + 5}
        for index in range(len(model.backbone.model.blocks))]
    if seen != expected or output["routing"].image_keep_indices.shape[1] != kept:
        raise ValueError("self-comparison timing graph differs from its fixed budget")
    maps, scores = stack_predictions(output)
    if maps.shape[0] != people or not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
        raise FloatingPointError("invalid multi-person timing predictions")
    return {"block_inputs": seen, "encoded_image_copies": copies, "input_images": len(inputs["bboxes"]),
        "queried_people": people, "kept_patches_per_copy": kept}


def aggregate(cells):
    if len(cells) != 6 or len({cell["gpu_uuid"] for cell in cells}) != 1:
        raise ValueError("self-comparison requires six cells on the same physical GPU")
    result = {}
    for n in (1, 2, 3):
        pair = [cell for cell in cells if cell["request"]["n"] == n]
        if len(pair) != 2 or {cell["request"]["order"] for cell in pair} != {"forward", "reverse"}:
            raise ValueError("each query count requires forward and reverse orders")
        common = [{k: v for k, v in cell["request"].items() if k != "order"} for cell in pair]
        if common[0] != common[1]:
            raise ValueError("forward/reverse timing requests differ")
        variants = {}
        for variant in VARIANTS:
            values = [cell["results"][variant]["timing"] for cell in pair]
            geomean = lambda key: math.sqrt(values[0][key] * values[1][key])
            fps = geomean("throughput_images_per_second_median")
            variants[variant] = {"latency_ms_per_frame": geomean("latency_ms_median"),
                "images_per_second": fps, "queries_per_second": n * fps,
                "peak_allocated_mib": max(value["peak_allocated_mib"] for value in values)}
        result[str(n)] = {"variants": variants,
            "union_speedup_vs_k100": variants["k100"]["latency_ms_per_frame"] / variants["union"]["latency_ms_per_frame"],
            "union_speedup_vs_independent": variants["independent"]["latency_ms_per_frame"] / variants["union"]["latency_ms_per_frame"]}
    return result


def markdown(timings):
    lines = ["# Selected-weight multi-person efficiency (VAT-B)", "",
        "K100 is our shared full-token control. Independent and union use K50. No official GazeLLE is timed.", "",
        "| N | Variant | ms/frame | images/s | queries/s | peak MiB |", "|---|---|---|---|---|---|"]
    for n, row in timings.items():
        for name, values in row["variants"].items():
            lines.append("| " + " | ".join([n, name] + [f"{values[key]:.6f}" for key in (
                "latency_ms_per_frame", "images_per_second", "queries_per_second", "peak_allocated_mib")]) + " |")
    return "\n".join(lines) + "\n"
