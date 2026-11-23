"""Official DINOv2 GazeLLE versus the selected DINOv3 K100/K50 system."""

import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

from PIL import Image
import torch

from cscwd_router.code.scripts.benchmark_router_dinov2 import gpu_snapshot, measure
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_sources import load_source

ROOT = original.ROOT
ORDERS = {"forward": ("official", "k100", "k50"), "reverse": ("k50", "k100", "official")}
PROTOCOL = "dinov3_vs_official_gazelle_efficiency_v1"


def source_hashes():
    names = ("cscwd_router/code/stag/dinov3_sources.py", "cscwd_router/code/stag/dinov3_efficiency.py",
             "cscwd_router/code/scripts/benchmark_router_dinov3.py",
             "cscwd_router/code/scripts/run_router_efficiency.py",
             "cscwd_router/code/scripts/benchmark_router_dinov2.py",
             "gazelle/backbone_v0_dinov2.py", "gazelle/model_v0_dinov2.py")
    values = {**original.source_hashes(), **{name: sha256(ROOT / name) for name in names}}
    digest = hashlib.sha256()
    for path in sorted((ROOT / "dinov2" / "dinov2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    values["native_dinov2_source_tree"] = digest.hexdigest()
    return values


def select_samples(data_path, batch_size, seed):
    annotation = Path(data_path) / "train_preprocessed.json"
    candidates = []
    for record in json.loads(annotation.read_text()):
        for index, head in enumerate(record["heads"]):
            if head["inout"] == 1:
                candidates.append({"path": record["path"], "head_index": index, "head_bbox": head["bbox_norm"]})
                break
    candidates.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item['path']}".encode()).hexdigest())
    samples = candidates[:batch_size]
    if len(samples) != batch_size:
        raise ValueError("not enough train RGB/GT head queries for the requested batch")
    for sample in samples:
        sample["image_sha256"] = sha256(Path(data_path) / sample["path"])
    return {"annotation_sha256": sha256(annotation), "samples": samples,
            "query_selection": "one_in_frame_GT_head_per_train_image", "people_per_image": 1}


def make_request(args):
    summary = json.loads(Path(args.source_summary).read_text())
    official_backbone = ROOT / "checkpoints" / f"dinov2_vit{summary['size']}14_pretrain.pth"
    return {"protocol": PROTOCOL, "dataset": summary["dataset"], "size": summary["size"],
            "source_summary": {"path": str(Path(args.source_summary).resolve()), "sha256": sha256(args.source_summary)},
            "router_checkpoint": summary["artifacts"]["best_test_checkpoint"],
            "router_backbone": summary["artifacts"]["backbone"],
            "official_checkpoint": {"path": str(Path(args.official_checkpoint).resolve()), "sha256": sha256(args.official_checkpoint)},
            "official_backbone": {"path": str(official_backbone.resolve()), "sha256": sha256(official_backbone)},
            "batch_size": args.batch_size, "order": args.order, "precision": args.precision,
            "seed": args.seed, "allow_closed_gf": args.allow_closed_gf,
            "settings": {key: getattr(args, key) for key in ("warmup", "latency_iters", "throughput_iters", "repeats")},
            "raw_input": select_samples(args.data_path, args.batch_size, args.seed), "sources": source_hashes()}


def build_variant(summary, payload, official_checkpoint, variant):
    if variant == "official":
        from gazelle.model_v0_dinov2 import get_gazelle_model
        suffix = "_inout" if summary["dataset"] == "vat" else ""
        model, transform = get_gazelle_model(f"gazelle_dinov2_vit{summary['size']}14{suffix}")
        state = torch.load(official_checkpoint, map_location="cpu", weights_only=False)
        original.restore_task(model, state)
    else:
        from gazelle.routing.model import get_coverage_router_model
        config = payload["model_config"]
        model, transform = get_coverage_router_model(config["model"], router_stage=config["router_stage"],
            route_after_block=config["route_after_block"], keep_ratio=1.0 if variant == "k100" else .5,
            router_hidden_dim=config["router_hidden_dim"], router_temperature=config["router_temperature"],
            escape_tokens=config["escape_tokens"], spatial_prior=config["spatial_prior"], fusion=config["fusion"])
        original.restore_task(model, payload["model_state"])
        if {**model.get_model_config(), "keep_ratio": .5} != config:
            raise ValueError("benchmark model differs from the evaluated checkpoint")
    return model.eval().requires_grad_(False), transform


def prepare_inputs(data_path, samples, transform, device):
    images = []
    for sample in samples:
        with Image.open(Path(data_path) / sample["path"]) as image:
            images.append(transform(image.convert("RGB")))
    return {"images": torch.stack(images).to(device), "bboxes": [[sample["head_bbox"]] for sample in samples]}


def expected_trace(model, batch, variant):
    native = model.backbone.model
    special = 1 if variant == "official" else model.backbone.prefix_token_count
    kept = 512 if variant == "k50" else 1024
    route = len(native.blocks) if variant == "official" else model.route_after_block
    return [{"block": index, "batch": batch, "tokens": (1024 if index <= route else kept) + special}
            for index in range(len(native.blocks))]


def trace_forward(model, inputs, variant, precision):
    trace = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, args, index=index: trace.append({"block": index, "batch": args[0].shape[0], "tokens": args[0].shape[1]})
    ) for index, block in enumerate(model.backbone.model.blocks)]
    try:
        with torch.inference_mode(), torch.autocast(inputs["images"].device.type,
                dtype=torch.float16, enabled=precision == "fp16", cache_enabled=False):
            output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    if trace != expected_trace(model, inputs["images"].shape[0], variant):
        raise RuntimeError(f"unexpected token graph for {variant}: {trace}")
    routing = output.get("routing")
    kept = 1024 if routing is None else routing.image_keep_indices.shape[1]
    if kept != (512 if variant == "k50" else 1024):
        raise RuntimeError("actual routing budget differs from the benchmark variant")
    for key in ("heatmap", "inout"):
        if output.get(key) is not None and not all(torch.isfinite(value).all() for value in output[key]):
            raise FloatingPointError(f"non-finite {key} in {variant}")
    return {"kept_patches": kept, "block_inputs": trace,
            "out_indices": getattr(model.backbone, "out_indices", [len(trace) - 1]),
            "image_size": list(inputs["images"].shape[-2:])}


def benchmark(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    if Path(args.output).exists():
        raise FileExistsError(f"benchmark output already exists: {args.output}")
    request = make_request(args)
    before = gpu_snapshot()
    summary, manifest, payload = load_source(args.source_summary, allow_closed_gf=args.allow_closed_gf)
    if sha256(ROOT / "checkpoints" / f"dinov3_vit{summary['size']}16_pretrain.pth") != request["router_backbone"]["sha256"]:
        raise ValueError("factory DINOv3 backbone differs from the evaluated source")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    for variant in ORDERS[args.order]:
        model, transform = build_variant(summary, payload, args.official_checkpoint, variant)
        model.cuda()
        inputs = prepare_inputs(args.data_path, request["raw_input"]["samples"], transform, "cuda")
        tokens = trace_forward(model, inputs, variant, args.precision)
        timing = measure(model, inputs, precision=args.precision, warmup=args.warmup,
            latency_iters=args.latency_iters, throughput_iters=args.throughput_iters, repeats=args.repeats)
        results[variant] = {"tokens": tokens, "timing": timing}
        print(json.dumps({"variant": variant, "batch": args.batch_size, **timing}), flush=True)
        del model, inputs
        gc.collect()
        torch.cuda.empty_cache()
    report = {"request": request, "variant_order": list(ORDERS[args.order]), "results": results,
              "source_metrics": summary["metrics"], "source_selection": summary["selection"],
              "source_closed_by_user": summary.get("closed_by_user", False),
              "environment": {"gpu_uuid": before["uuid"], "before": before, "after": gpu_snapshot(),
                              "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
              "timing_scope": {"full_forward": True, "preprocessing_and_H2D": False,
                               "inference_mode": True, "autocast_cache_enabled": False, "tf32": False},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    original.atomic_json(args.output, report)
    return report


def summarize(cells):
    first = cells[0]
    by_key = {(cell["request"]["batch_size"], cell["request"]["order"]): cell for cell in cells}
    batches = sorted({key[0] for key in by_key})
    if len(by_key) != len(cells) or set(by_key) != {(b, order) for b in batches for order in ORDERS}:
        raise ValueError("efficiency matrix needs both orders for each batch")
    common = {k: v for k, v in first["request"].items() if k not in ("batch_size", "order", "raw_input")}
    for cell in cells:
        current = {k: v for k, v in cell["request"].items() if k not in ("batch_size", "order", "raw_input")}
        if current != common or cell["environment"]["gpu_uuid"] != first["environment"]["gpu_uuid"]:
            raise ValueError("matrix settings, checkpoints, source or GPU differ")
    def geomean(values):
        return math.exp(sum(math.log(v) for v in values) / len(values))
    efficiency = {}
    for batch in batches:
        pair = [by_key[batch, order] for order in ORDERS]
        if pair[0]["request"]["raw_input"] != pair[1]["request"]["raw_input"]:
            raise ValueError("forward/reverse RGB and head queries differ")
        timings = [cell["results"] for cell in pair]
        efficiency[str(batch)] = {
            "variants": {variant: {
                "latency_ms_geomean": geomean([x[variant]["timing"]["latency_ms_median"] for x in timings]),
                "throughput_fps_geomean": geomean([x[variant]["timing"]["throughput_images_per_second_median"] for x in timings]),
                "peak_allocated_mib_max": max(x[variant]["timing"]["peak_allocated_mib"] for x in timings)} for variant in ORDERS["forward"]},
            "k50_speedup_vs_official": geomean([x["official"]["timing"]["latency_ms_median"] / x["k50"]["timing"]["latency_ms_median"] for x in timings]),
            "k50_speedup_vs_k100": geomean([x["k100"]["timing"]["latency_ms_median"] / x["k50"]["timing"]["latency_ms_median"] for x in timings]),
            "k50_throughput_ratio_vs_official": geomean([x["k50"]["timing"]["throughput_images_per_second_median"] / x["official"]["timing"]["throughput_images_per_second_median"] for x in timings])}
    return {"protocol": PROTOCOL, "dataset": common["dataset"], "size": common["size"],
            "gpu_uuid": first["environment"]["gpu_uuid"], "source_metrics": first["source_metrics"],
            "source_selection": first["source_selection"], "source_closed_by_user": first["source_closed_by_user"],
            "efficiency": efficiency}


def render_summary(report):
    lines = [f"# DINOv3 Router efficiency · {report['dataset']} / vit{report['size']}", "",
             "| Batch | Official ms | K100 ms | K50 ms | K50 speedup vs official | vs K100 | FPS ratio vs official |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for batch, item in report["efficiency"].items():
        v = item["variants"]
        lines.append(f"| {batch} | {v['official']['latency_ms_geomean']:.4f} | {v['k100']['latency_ms_geomean']:.4f} | "
                     f"{v['k50']['latency_ms_geomean']:.4f} | {item['k50_speedup_vs_official']:.4f} | "
                     f"{item['k50_speedup_vs_k100']:.4f} | {item['k50_throughput_ratio_vs_official']:.4f} |")
    return "\n".join(lines) + "\n"
