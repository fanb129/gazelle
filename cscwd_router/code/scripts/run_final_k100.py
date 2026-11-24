"""Evaluate the four selected checkpoints at K100 once, without training or reselection."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.data_prep.train_test import make_full_test, make_loader
from cscwd_router.code.stag import final_io as io
from cscwd_router.code.stag import dinov3_efficiency as efficiency
from cscwd_router.code.stag.dinov3_engine import evaluate
from cscwd_router.code.stag.engine import seed_everything
from scripts.train_coverage_router import unpack_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries", type=Path, nargs="+", required=True)
    for name in ("gf-data-path", "vat-data-path", "artifact-root", "report-root", "provenance"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--allow-closed-gf", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def run(args, *, model_builder=io.build_router):
    if args.seed != 0:
        raise ValueError("the final evaluation uses seed 0")
    reports = []
    for path in args.summaries:
        source, manifest, payload = io.verified_source(path, args.allow_closed_gf)
        data_path = args.gf_data_path if source["dataset"] == "gazefollow" else args.vat_data_path
        annotation = io.reference(data_path / "test_preprocessed.json")
        if annotation != manifest["data"]["test_annotation"]:
            raise ValueError("evaluation annotation differs from the source test annotation")
        request = {"protocol": "selected_weights_K100_accuracy_v1", "source": io.reference(path),
            "checkpoint": source["artifacts"]["best_test_checkpoint"], "backbone": source["artifacts"]["backbone"],
            "test_annotation": annotation, "keep_ratio": 1., "precision": "fp32", "seed": args.seed,
            "batch_size": args.batch_size, "frame_sample_every": 1,
            "sources": io.sources("cscwd_router/code/scripts/run_final_k100.py", "cscwd_router/code/stag/dinov3_efficiency.py",
                                  "cscwd_router/code/stag/dinov3_sources.py")}
        output = args.report_root / source["dataset"] / f"vit{source['size']}" / "summary.json"
        if args.check_only:
            print(json.dumps({"checked": str(path), "request": request}), flush=True)
            continue
        old = io.existing(output, request, args.resume)
        if old:
            reports.append(old)
            continue
        seed_everything(args.seed)
        model, transform = model_builder(source, payload, keep_ratio=1.)
        model.to(args.device)
        dataset = make_full_test(source["dataset"], data_path, transform)
        loader = make_loader(dataset, args.batch_size, args.workers)
        inputs, *_ = unpack_batch(next(iter(loader)), args.device, training=False)
        trace = efficiency.trace_forward(model, inputs, "k100", "fp32")
        rows_path = args.artifact_root / source["dataset"] / f"vit{source['size']}" / "k100_predictions.jsonl.gz"
        metrics = evaluate(model, loader, args.device, dataset=source["dataset"], keep_ratio=1., rows_path=rows_path)
        baseline = source["metrics"]
        if (metrics["persons"], metrics["inframe_persons"]) != (baseline["persons"], baseline["inframe_persons"]):
            raise ValueError("K100 and saved K50 query counts differ")
        metric_names = ("auc", "avg_l2", "min_l2") if source["dataset"] == "gazefollow" else ("auc", "l2", "inout_ap")
        result = {"request": request, "source_selection": source["selection"], "K50_saved": baseline, "K100": metrics,
            "K100_minus_K50": {k: metrics[k] - baseline[k] for k in metric_names}, "trace": trace,
            "source_closed_by_user": source.get("closed_by_user", False), "runtime": io.runtime(args.device),
            "artifacts": {"per_query_predictions": io.reference(rows_path)}}
        reports.append(io.finish(output, result, args.provenance, f"Final K100 accuracy · {source['dataset']}/vit{source['size']}"))
        del model, payload, loader
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"COMPLETE {output}", flush=True)
    return reports


if __name__ == "__main__":
    run(parse_args())
