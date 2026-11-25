"""Time N=1/2/3 shared K100, independent K50 and union K50 on an idle GPU."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from cscwd_router.code.stag import final_io as io, final_multipeople as multi
from cscwd_router.code.stag import dinov3_efficiency as efficiency, self_multi_timing as timing
from cscwd_router.code.stag.engine import seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "data-path", "report-root", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--precision", choices=("fp16",), default="fp16")
    for name, default in (("warmup", 50), ("latency-iters", 200), ("throughput-iters", 500), ("repeats", 5), ("seed", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--n", choices=(1, 2, 3), type=int, help=argparse.SUPPRESS)
    parser.add_argument("--order", choices=("forward", "reverse"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def prepare(args):
    source, manifest, payload = io.verified_source(args.source_summary)
    if args.seed != 0 or source["dataset"] != "vat" or source["size"] != "b":
        raise ValueError("self multi-person timing uses the selected VAT-B model and seed 0")
    frame, selection, annotation = multi.timing_sample(args.data_path, args.seed)
    if annotation != manifest["data"]["train_annotation"]:
        raise ValueError("timing train annotation differs from the selected model")
    sample = {"path": frame["path"], "head_indices": selection["head_indices"],
        "head_bboxes": [frame["heads"][i]["bbox_norm"] for i in selection["head_indices"]],
        "image_sha256": io.reference(args.data_path / frame["path"])["sha256"], "annotation": annotation}
    request = {"protocol": "selected_weight_nested_multi_timing_v1", "source": io.reference(args.source_summary),
        "checkpoint": source["artifacts"]["best_test_checkpoint"], "backbone": source["artifacts"]["backbone"],
        "seed": args.seed, "precision": args.precision, "batch_images": 1, "sample": sample,
        "settings": {key: getattr(args, key) for key in ("warmup", "latency_iters", "throughput_iters", "repeats")},
        "sources": io.sources("cscwd_router/code/stag/self_multi_timing.py",
            "cscwd_router/code/scripts/run_self_multi_timing.py", "cscwd_router/code/stag/final_multipeople.py",
            "cscwd_router/code/data_prep/final_queries.py", "cscwd_router/code/stag/dinov3_efficiency.py",
            "cscwd_router/code/stag/dinov3_sources.py", "cscwd_router/code/scripts/benchmark_router_dinov2.py")}
    return source, payload, request


def cell_request(common, n, order):
    sample = common["sample"]
    return {**common, "n": n, "order": order, "sample": {**sample,
        "head_indices": sample["head_indices"][:n], "head_bboxes": sample["head_bboxes"][:n]}}


def cell_path(args, n, order):
    return args.report_root / f"n{n}_{order}.json"


def run_cell(args, prepared):
    source, payload, common = prepared
    request = cell_request(common, args.n, args.order)
    path = cell_path(args, args.n, args.order)
    old = io.existing(path, request, args.resume)
    if old:
        return old
    before = efficiency.gpu_snapshot()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variants = timing.VARIANTS if args.order == "forward" else tuple(reversed(timing.VARIANTS))
    results = {}
    for variant in variants:
        model, transform = timing.build_variant(source, payload, variant)
        model.cuda()
        sample = request["sample"]
        inputs = efficiency.prepare_inputs(args.data_path,
            [{"path": sample["path"], "head_bbox": sample["head_bboxes"][0]}], transform, "cuda")
        inputs["bboxes"] = [sample["head_bboxes"]]
        tokens = timing.trace(model, inputs, variant)
        measured = efficiency.measure(model, inputs, precision=args.precision, warmup=args.warmup,
            latency_iters=args.latency_iters, throughput_iters=args.throughput_iters, repeats=args.repeats)
        results[variant] = {"trace": tokens, "timing": measured}
        print(json.dumps({"n": args.n, "variant": variant, **measured}), flush=True)
        del model, inputs
        gc.collect()
        torch.cuda.empty_cache()
    report = {"request": request, "results": results, "variant_order": list(variants),
        "gpu_uuid": before["uuid"], "gpu_before": before, "gpu_after": efficiency.gpu_snapshot(),
        "runtime": io.runtime(args.device), "source_selection": source["selection"],
        "timing_scope": {"full_forward": True, "preprocessing_and_H2D": False,
            "inference_mode": True, "autocast_cache_enabled": False, "tf32": False}, "artifacts": {}}
    saved = io.finish(path, report, args.provenance, f"Selected-weight multi-person timing N={args.n} {args.order}")
    print(f"COMPLETE {path}", flush=True)
    return saved


def command_for(args, n, order):
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    for key in ("source_summary", "data_path", "report_root", "provenance", "device", "precision",
                "warmup", "latency_iters", "throughput_iters", "repeats", "seed"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    command += ["--n", str(n), "--order", order]
    if args.resume:
        command.append("--resume")
    return command


def run(args):
    prepared = prepare(args)
    if args.check_only:
        report = {"checked": True, "model_forward_executed": False, "timing_cells": 6,
            "variants": list(timing.VARIANTS), "request": prepared[2]}
        print(json.dumps(report), flush=True)
        return report
    if args.n is not None:
        return run_cell(args, prepared)
    efficiency.gpu_snapshot()
    for n in (1, 2, 3):
        for order in ("forward", "reverse"):
            subprocess.run(command_for(args, n, order), cwd=ROOT, check=True)
    cells = [io.existing(cell_path(args, n, order), cell_request(prepared[2], n, order), True)
             for n in (1, 2, 3) for order in ("forward", "reverse")]
    timings = timing.aggregate(cells)
    path = args.report_root / "summary.json"
    report = io.finish(path, {"request": prepared[2], "timing": timings,
        "artifacts": {f"n{n}_{order}": io.reference(cell_path(args, n, order))
            for n in (1, 2, 3) for order in ("forward", "reverse")}}, args.provenance,
        "Selected-weight multi-person efficiency summary")
    path.with_suffix(".md").write_text(timing.markdown(timings))
    print(f"COMPLETE {path}", flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
