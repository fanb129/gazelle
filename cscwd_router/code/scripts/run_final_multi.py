"""VAT common-frame N=1/2/3 evaluation; shared official, independent and union routes."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from cscwd_router.code.data_prep.final_queries import cohort
from cscwd_router.code.stag import final_io as io, final_multipeople as multi
from cscwd_router.code.stag import dinov3_efficiency as efficiency
from cscwd_router.code.stag.dinov3_run import atomic_json
from cscwd_router.code.stag.engine import seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "data-path", "official-checkpoint", "artifact-root", "report-root", "provenance"):
        p.add_argument("--" + name, type=Path, required=True)
    for name, default in (("batch-size", 4), ("workers", 4), ("seed", 0), ("warmup", 50),
                          ("latency-iters", 200), ("throughput-iters", 500), ("repeats", 5)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--task", choices=("all", "accuracy", "timing"), default="all")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--precision", choices=("fp16",), default="fp16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--cell-role", choices=("accuracy", "timing"), help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, choices=(1, 2, 3), help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=multi.VARIANTS, default="union", help=argparse.SUPPRESS)
    p.add_argument("--order", choices=("forward", "reverse"), default="forward", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def prepare(args):
    if args.seed != 0:
        raise ValueError("the final multi-query experiment uses seed 0")
    source, manifest, payload = io.verified_source(args.source_summary)
    if source["dataset"] != "vat" or source["size"] != "b":
        raise ValueError("the final multi-query experiment uses the selected VAT-B source")
    records, selection = cohort(args.data_path, args.seed)
    if selection["annotation"] != manifest["data"]["test_annotation"]:
        raise ValueError("multi-query test annotation differs from the selected source")
    if not records:
        raise ValueError("no real test frames have at least three GT heads")
    cohort_path = args.report_root / "cohort.json"
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != selection:
        raise ValueError("existing multi-query cohort differs")
    atomic_json(cohort_path, selection)
    frame, frame_selection, train_annotation = multi.timing_sample(args.data_path, args.seed)
    common = {"protocol": "VAT_real_nested_multi_query_v1", "source": io.reference(args.source_summary),
        "checkpoint": source["artifacts"]["best_test_checkpoint"], "backbone": source["artifacts"]["backbone"],
        "official_checkpoint": io.reference(args.official_checkpoint),
        "official_backbone": io.reference(ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"),
        "cohort": {k: v for k, v in selection.items() if k != "entries"}, "seed": args.seed,
        "sources": {**efficiency.source_hashes(), **io.sources(
            "cscwd_router/code/data_prep/final_queries.py", "cscwd_router/code/stag/final_multipeople.py",
            "cscwd_router/code/scripts/run_final_multi.py")}}
    sample = {"path": frame["path"], "head_indices": frame_selection["head_indices"],
        "head_bboxes": [frame["heads"][index]["bbox_norm"] for index in frame_selection["head_indices"]],
        "image_sha256": io.reference(args.data_path / frame["path"])["sha256"], "annotation": train_annotation}
    return source, payload, records, selection, common, sample


def request_for(args, common, sample, role, n, variant="union", order="forward"):
    if role == "accuracy":
        return {**common, "role": role, "n": n, "variant": variant, "precision": "fp32", "batch_size": args.batch_size}
    chosen = {**sample, "head_indices": sample["head_indices"][:n], "head_bboxes": sample["head_bboxes"][:n]}
    return {**common, "role": role, "n": n, "order": order, "sample": chosen, "precision": args.precision,
            "batch_images": 1, "settings": {key: getattr(args, key) for key in ("warmup", "latency_iters", "throughput_iters", "repeats")}}


def output_path(args, role, n, variant="union", order="forward"):
    return args.report_root / role / f"n{n}_{variant if role == 'accuracy' else order}.json"


def run_cell(args, prepared):
    source, payload, records, selection, common, sample = prepared
    role, n, variant, order = args.cell_role, args.n, args.variant, args.order
    request = request_for(args, common, sample, role, n, variant, order)
    path = output_path(args, role, n, variant, order)
    old = io.existing(path, request, args.resume)
    if old:
        return old
    seed_everything(args.seed)
    if role == "accuracy":
        model, transform = multi.build_variant(source, payload, args.official_checkpoint, variant)
        model.to(args.device)
        loader = multi.loader_for(args.data_path, records, selection, n, transform, args.batch_size, args.workers)
        rows = args.artifact_root / f"n{n}_{variant}_predictions.jsonl.gz"
        metrics = multi.evaluate(model, loader, args.device, rows)
        result = {"request": request, "metrics": metrics, "source_selection": source["selection"],
            "runtime": io.runtime(args.device), "artifacts": {"per_query_predictions": io.reference(rows)}}
    else:
        if args.device != "cuda":
            raise ValueError("timing requires an idle CUDA GPU")
        before = efficiency.gpu_snapshot()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        order_variants = multi.VARIANTS if order == "forward" else tuple(reversed(multi.VARIANTS))
        results = {}
        for selected in order_variants:
            model, transform = multi.build_variant(source, payload, args.official_checkpoint, selected)
            model.cuda()
            chosen = request["sample"]
            inputs = efficiency.prepare_inputs(args.data_path,
                [{"path": chosen["path"], "head_bbox": chosen["head_bboxes"][0]}], transform, "cuda")
            inputs["bboxes"] = [chosen["head_bboxes"]]
            tokens = multi.trace(model, inputs, selected)
            timing = efficiency.measure(model, inputs, precision="fp16", warmup=args.warmup,
                latency_iters=args.latency_iters, throughput_iters=args.throughput_iters, repeats=args.repeats)
            results[selected] = {"trace": tokens, "timing": timing}
            print(json.dumps({"n": n, "variant": selected, **timing}), flush=True)
            del model, inputs
            gc.collect()
            torch.cuda.empty_cache()
        result = {"request": request, "results": results, "variant_order": list(order_variants),
            "gpu_uuid": before["uuid"], "gpu_before": before, "gpu_after": efficiency.gpu_snapshot(),
            "runtime": io.runtime(args.device), "timing_scope": {"full_forward": True, "preprocessing_and_H2D": False,
                "inference_mode": True, "autocast_cache_enabled": False, "tf32": False}, "artifacts": {}}
    result = io.finish(path, result, args.provenance, f"Final multi-query {role} · N={n}")
    print(f"COMPLETE {path}", flush=True)
    return result


def command_for(args, role, n, variant="union", order="forward"):
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    for key in ("source_summary", "data_path", "official_checkpoint", "artifact_root", "report_root", "provenance",
                "batch_size", "workers", "seed", "warmup", "latency_iters", "throughput_iters", "repeats", "device", "precision"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    command += ["--cell-role", role, "--n", str(n), "--variant", variant, "--order", order]
    if args.resume:
        command.append("--resume")
    return command


def render_summary(quality, timings):
    lines = ["# Real VAT multi-query results", "", "| N | Variant | AUC | L2 | AP | GT point coverage | ms/frame | images/s | queries/s | peak MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in (1, 2, 3):
        for variant in multi.VARIANTS:
            q = quality.get(str(n), {}).get(variant, {})
            t = timings.get(str(n), {}).get(variant, {})
            values = [q.get(k) for k in ("auc", "l2", "inout_ap", "routing_gt_point_coverage")]
            values += [t.get(k) for k in ("latency_ms_per_frame", "images_per_second", "queries_per_second", "peak_allocated_mib")]
            lines.append("| " + " | ".join([str(n), variant] + ["—" if v is None else f"{v:.6f}" for v in values]) + " |")
    return "\n".join(lines) + "\n"


def run(args):
    prepared = prepare(args)
    if args.check_only:
        report = {"checked": True, "frames": prepared[3]["frames"], "selection_sha256": prepared[3]["selection_sha256"],
                  "quality_cells": 9, "timing_cells": 6, "model_forward_executed": False}
        print(json.dumps(report), flush=True)
        return report
    if args.cell_role:
        return run_cell(args, prepared)
    _, _, _, _, common, sample = prepared
    for n in (1, 2, 3):
        if args.task in ("all", "accuracy"):
            for variant in multi.VARIANTS:
                subprocess.run(command_for(args, "accuracy", n, variant), cwd=ROOT, check=True)
        if args.task in ("all", "timing"):
            for order in ("forward", "reverse"):
                subprocess.run(command_for(args, "timing", n, order=order), cwd=ROOT, check=True)
    quality, timing_cells, artifacts = {}, [], {"cohort": io.reference(args.report_root / "cohort.json")}
    for n in (1, 2, 3):
        for variant in multi.VARIANTS:
            path = output_path(args, "accuracy", n, variant)
            if path.exists():
                saved = io.existing(path, request_for(args, common, sample, "accuracy", n, variant), True)
                quality.setdefault(str(n), {})[variant] = saved["metrics"]
                artifacts[f"accuracy_n{n}_{variant}"] = io.reference(path)
        for order in ("forward", "reverse"):
            path = output_path(args, "timing", n, order=order)
            if path.exists():
                timing_cells.append(io.existing(path, request_for(args, common, sample, "timing", n, order=order), True))
                artifacts[f"timing_n{n}_{order}"] = io.reference(path)
    if timing_cells and len({cell["gpu_uuid"] for cell in timing_cells}) != 1:
        raise ValueError("multi-query timing cells used different physical GPUs")
    timings = multi.aggregate_timing(timing_cells) if len(timing_cells) == 6 else {}
    result = {"request": common, "requested_task": args.task, "quality": quality, "timing": timings,
              "artifacts": artifacts, "quality_query_counts": {str(n): prepared[3]["frames"] * n for n in (1, 2, 3)}}
    path = args.report_root / "summary.json"
    io.finish(path, result, args.provenance, "Final real VAT multi-query summary")
    path.with_suffix(".md").write_text(render_summary(quality, timings))
    print(f"COMPLETE {path}", flush=True)
    return result


if __name__ == "__main__":
    run(parse_args())
