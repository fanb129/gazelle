"""Zero-shot GOO-Real evaluation of fixed GF Router and official GazeLLE weights."""

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import sklearn
import torch

from cscwd_router.code.data_prep.gooreal import audited_records
from cscwd_router.code.stag import final_io as io
from cscwd_router.code.stag import gooreal
from cscwd_router.code.stag.dinov3_efficiency import build_variant, source_hashes
from cscwd_router.code.stag.engine import seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "data-path", "official-checkpoint", "artifact-root", "report-root", "provenance"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=("k50", "official", "k100"), default=["k50", "official"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--allow-closed-gf", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def run(args):
    if args.seed != 0:
        raise ValueError("GOO-Real evaluation uses the fixed seed 0")
    source, manifest, payload = io.verified_source(args.source_summary, args.allow_closed_gf)
    if source["dataset"] != "gazefollow":
        raise ValueError("zero-shot GOO-Real uses GazeFollow-trained checkpoints only")
    records, audit = audited_records(args.data_path)
    native = ROOT / "checkpoints" / f"dinov2_vit{source['size']}14_pretrain.pth"
    hashes = {**source_hashes(), **io.sources("cscwd_router/code/data_prep/gooreal.py", "cscwd_router/code/stag/gooreal.py",
                       "cscwd_router/code/scripts/eval_gooreal_router.py", "scripts/eval_gooreal.py",
                       "data_prep/preprocess_gooreal.py", "gazelle/model_v0_dinov2.py", "gazelle/backbone_v0_dinov2.py")}
    cv2.setNumThreads(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for variant in args.variants:
        request = {"protocol": "GOO_Real_full_test_zero_shot_v1", "dataset": "gooreal", "size": source["size"],
            "variant": variant, "training_dataset": "gazefollow", "target_training": False,
            "target_checkpoint_selection": False, "source_summary": io.reference(args.source_summary),
            "checkpoint": io.reference(args.official_checkpoint) if variant == "official" else source["artifacts"]["best_test_checkpoint"],
            "backbone": io.reference(native) if variant == "official" else source["artifacts"]["backbone"],
            "annotation": io.reference(args.data_path / "gooreal_test_preprocessed.json"),
            "official_raw_annotation": io.reference(args.data_path / "testrealhumansNew.pickle"),
            "precision": "fp32", "seed": args.seed, "batch_size": args.batch_size,
            "metric_definition": gooreal.METRIC, "sources": hashes}
        output = args.report_root / f"vit{source['size']}" / variant / "summary.json"
        if args.check_only:
            print(json.dumps({"checked": True, "request": request, "annotation_audit": audit}), flush=True)
            continue
        if io.existing(output, request, args.resume):
            print(f"RESUME completed {output}", flush=True)
            continue
        seed_everything(args.seed)
        model, transform = (build_variant(source, payload, args.official_checkpoint, "official")
                            if variant == "official" else io.build_router(source, payload, .5 if variant == "k50" else 1.))
        model.to(args.device)
        directory = args.artifact_root / f"vit{source['size']}" / variant
        directory.mkdir(parents=True, exist_ok=True)
        rows, cases = directory / "predictions.jsonl.gz", directory / "visual_cases.npz"
        metrics, trace = gooreal.evaluate(model, transform, records, args, variant, rows, cases)
        report = {"request": request, "metrics": metrics, "annotation_audit": audit, "trace": trace,
            "source_selection": source["selection"] if variant != "official" else "official released GF weights",
            "visualization_selection": "first 8 paths sorted by SHA256(seed0 + path), independent of predictions",
            "runtime": {**io.runtime(args.device), "opencv": cv2.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__},
            "artifacts": {"per_query_predictions": io.reference(rows), "visual_cases": io.reference(cases)}}
        io.finish(output, report, args.provenance, f"GOO-Real zero-shot · vit{source['size']}/{variant}")
        print(json.dumps({"completed": str(output), "metrics": metrics}), flush=True)
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run(parse_args())
