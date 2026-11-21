"""T4 end-to-end GPU inference matrix cell on real train RGB/head boxes.

Data loading, transforms, transfers and checkpoint loading are outside timing.
Each CLI invocation is one fresh process for a batch/order cell.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dinov2"))

from gazelle.model_v0_dinov2 import GazeLLE
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone
from cscwd_router.code.stag.checkpoint import read_checkpoint, restore_checkpoint, sha256
from cscwd_router.code.stag.main_results import BATCHES, ORDERS
from cscwd_router.code.stag.model_dinov2 import build_model
from cscwd_router.code.stag.pipeline import OfficialBackboneView, checkpoint_ref


def gpu_snapshot():
    selected = os.environ["CUDA_VISIBLE_DEVICES"]
    fields = ("uuid", "name", "driver_version", "pstate", "temperature.gpu", "memory.used", "utilization.gpu")
    output = subprocess.check_output(["nvidia-smi", "-i", selected, "--query-gpu=" + ",".join(fields),
                                      "--format=csv,noheader,nounits"], text=True)
    values = next(csv.reader(output.splitlines()))
    state = dict(zip(fields, [value.strip() for value in values]))
    processes = subprocess.check_output(["nvidia-smi", "-i", selected, "--query-compute-apps=pid",
                                         "--format=csv,noheader,nounits"], text=True)
    others = [int(value.strip()) for value in processes.splitlines() if value.strip().isdigit() and int(value.strip()) != os.getpid()]
    if others:
        raise RuntimeError(f"selected GPU has other compute processes: {others}")
    if "RTX 3090" not in state["name"]:
        raise RuntimeError(f"T4 efficiency requires RTX 3090, found {state['name']}")
    return state


def prepare_inputs(data_path, transform, batch_size, seed):
    """Select by a stable path hash; gaze coordinates are never read."""
    data_path = Path(data_path)
    annotation = data_path / "train_preprocessed.json"
    candidates = []
    for record in json.loads(annotation.read_text()):
        heads = [head for head in record["heads"] if head["inout"] == 1]
        if len(heads) == 1:
            candidates.append((record["path"], heads[0]["bbox_norm"]))
    candidates.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest())
    selected = candidates[:batch_size]
    if len(selected) != batch_size:
        raise ValueError("not enough single-head train images for the requested batch")
    images, boxes, samples = [], [], []
    for relative, box in selected:
        path = data_path / relative
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
        boxes.append([box])
        samples.append({"path": relative, "head_bbox": box, "image_sha256": sha256(path)})
    metadata = {"source": "gazefollow_train_rgb_and_gt_head_boxes", "annotation_sha256": sha256(annotation),
                "samples": samples, "image_size": 448, "people_per_image": 1}
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return {"images": torch.stack(images), "bboxes": boxes}, metadata


def build_variant(payload, variant, *, backbone=None):
    if backbone is None:
        backbone = RoutedDinoV2Backbone()
    if variant == "dense":
        config = payload["model_config"]
        model = GazeLLE(OfficialBackboneView(backbone.model), dim=config["decoder_dim"],
                       num_layers=config["decoder_layers"], in_size=(config["image_size"],) * 2)
        model.load_gazelle_state_dict(payload["decoder"])
    else:
        model = build_model(payload["model_config"], backbone=backbone)
        restore_checkpoint(model, payload)
        model.encoder.router.keep_ratio = {"k100": 1.0, "k50": 0.5}[variant]
    return model.eval().requires_grad_(False)


def native_backbone(model, variant):
    return model.backbone if variant == "dense" else model.encoder.backbone


def trace_forward(model, inputs, variant, precision):
    backbone = native_backbone(model, variant)
    trace = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, args, index=index: trace.append({"block": index, "batch": args[0].shape[0], "tokens": args[0].shape[1]})
    ) for index, block in enumerate(backbone.model.blocks)]
    with torch.inference_mode(), torch.autocast(inputs["images"].device.type, dtype=torch.float16, enabled=precision == "fp16", cache_enabled=False):
        output = model(inputs)
    for hook in hooks:
        hook.remove()
    routing = output.get("routing")
    kept = routing.image_keep_indices.shape[1] if routing is not None else 1024
    batch = inputs["images"].shape[0]
    expected_keep = 512 if variant == "k50" else 1024
    expected = [{"block": index, "batch": batch, "tokens": (1024 if index <= 5 else expected_keep) + 1}
                for index in range(12)]
    if kept != expected_keep or trace != expected:
        raise RuntimeError(f"unexpected token graph for {variant}: {trace}")
    if not torch.isfinite(torch.cat(output["heatmap"])).all():
        raise FloatingPointError(f"non-finite predictions in benchmark variant {variant}")
    return {"kept_patches": kept, "special_tokens": 1, "block_inputs": trace}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(model, inputs, *, precision, warmup, latency_iters, throughput_iters, repeats):
    latencies, repeat_medians, throughput = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16", cache_enabled=False):
        for _ in range(warmup):
            model(inputs)
        torch.cuda.synchronize()
        steady = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            gpu_snapshot()
            samples = []
            for _ in range(latency_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(inputs)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            latencies.extend(samples)
            repeat_medians.append(statistics.median(samples))
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(throughput_iters):
                model(inputs)
            torch.cuda.synchronize()
            throughput.append(throughput_iters * inputs["images"].shape[0] / (time.perf_counter() - start))
        peak = torch.cuda.max_memory_allocated()
    return {"latency_ms_median": statistics.median(latencies), "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_p95": percentile(latencies, 0.95), "latency_sample_count": len(latencies),
            "latency_repeat_medians_ms": repeat_medians,
            "throughput_images_per_second_median": statistics.median(throughput), "throughput_repeat_fps": throughput,
            "steady_allocated_mib": steady / 2**20, "peak_allocated_mib": peak / 2**20}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--sparse-checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=BATCHES, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--throughput-iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    if args.output.exists():
        raise FileExistsError(f"benchmark output already exists: {args.output}")
    before = gpu_snapshot()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dense, sparse = read_checkpoint(args.dense_checkpoint), read_checkpoint(args.sparse_checkpoint)
    for payload, stage in ((dense, "dense_official"), (sparse, "sparse_decoder")):
        if payload["stage"] != stage or payload["context"]["data"]["protocol"] != "full_train" or payload["checkpoint_role"] != "fixed_final":
            raise ValueError("benchmark requires fixed-final full-train official/K50 checkpoints")
        if sha256(ROOT / "checkpoints/dinov2_vitb14_pretrain.pth") != payload["context"]["external_weights"]["backbone"]["sha256"]:
            raise ValueError("external DINOv2 checkpoint SHA256 differs from training")
    inputs = metadata = None
    results = {}
    for variant in ORDERS[args.order]:
        model = build_variant(dense if variant == "dense" else sparse, variant).cuda()
        if inputs is None:
            inputs, metadata = prepare_inputs(args.data_path, native_backbone(model, variant).get_transform((448, 448)), args.batch_size, args.seed)
            inputs["images"] = inputs["images"].cuda()
        tokens = trace_forward(model, inputs, variant, args.precision)
        timing = measure(model, inputs, precision=args.precision, warmup=args.warmup,
                         latency_iters=args.latency_iters, throughput_iters=args.throughput_iters, repeats=args.repeats)
        results[variant] = {"tokens": tokens, "timing": timing}
        print(json.dumps({"variant": variant, "batch": args.batch_size, **timing}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    source_names = ("cscwd_router/code/scripts/benchmark_router_dinov2.py", "cscwd_router/code/stag/main_results.py",
                    "cscwd_router/code/stag/model_dinov2.py", "cscwd_router/code/stag/router_dinov2.py",
                    "cscwd_router/code/stag/pipeline.py", "cscwd_router/code/stag/checkpoint.py", "gazelle/utils.py",
                    "cscwd_router/code/stag/backbone_sparse_dinov2.py", "gazelle/backbone_v0_dinov2.py",
                    "gazelle/model_v0_dinov2.py", "gazelle/routing/router.py",
                    "dinov2/dinov2/models/vision_transformer.py", "dinov2/dinov2/layers/attention.py", "dinov2/dinov2/layers/block.py")
    report = {
        "task": "T4_efficiency", "protocol": "dinov2_main_efficiency_v1", "batch_size": args.batch_size,
        "order": args.order, "variant_order": list(ORDERS[args.order]), "precision": args.precision,
        "settings": {"warmup": args.warmup, "latency_iters": args.latency_iters, "throughput_iters": args.throughput_iters,
                     "repeats": args.repeats, "autocast_cache_enabled": False, "inference_mode": True, "tf32": False,
                     "image_size": 448, "people_per_image": 1, "transfers_and_preprocessing_timed": False},
        "checkpoints": {"dense_official": checkpoint_ref(args.dense_checkpoint), "sparse_decoder": checkpoint_ref(args.sparse_checkpoint)},
        "environment": {"gpu_uuid": before["uuid"], "gpu": before["name"], "driver": before["driver_version"],
                        "torch": str(torch.__version__), "python": platform.python_version(), "cuda": torch.version.cuda,
                        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "before": before, "after": gpu_snapshot()},
        "input": metadata, "results": results, "sources": {name: sha256(ROOT / name) for name in source_names},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "argv": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
