"""Full-frame VAT evaluation of saved epochs for multi-metric checkpoint comparison."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.vat_epoch_selection import compare_epochs, render_comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "data-path", "run-dir", "report-dir", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True, help="One-based training epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs) < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    args.epochs = sorted(set(args.epochs))
    return args


def run(args):
    import torch
    from gazelle.routing.model import get_coverage_router_model
    from cscwd_router.code.data_prep.train_test import make_full_test, make_loader
    from cscwd_router.code.stag import dinov3_run as original
    from cscwd_router.code.stag.checkpoint import sha256
    from cscwd_router.code.stag.dinov3_engine import evaluate
    from cscwd_router.code.stag.dinov3_sources import load_source
    from cscwd_router.code.stag.engine import seed_everything

    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    source, manifest, baseline = load_source(args.source_summary)
    if source["dataset"] != "vat" or source["final_test_frame_rate"] != 1:
        raise ValueError("source must be a completed VAT full-frame result")
    if args.report_dir.resolve() == args.source_summary.parent.resolve():
        raise ValueError("comparison reports must use a new directory")
    annotation = args.data_path / "test_preprocessed.json"
    if sha256(annotation) != manifest["data"]["test_annotation"]["sha256"]:
        raise ValueError("test annotation differs from the source experiment")
    if sha256(ROOT / "checkpoints" / Path(source["artifacts"]["backbone"]["path"]).name) != source["artifacts"]["backbone"]["sha256"]:
        raise ValueError("factory backbone differs from the source experiment")
    history = json.loads(Path(source["artifacts"]["history"]["path"]).read_text())
    if sha256(source["artifacts"]["history"]["path"]) != source["artifacts"]["history"]["sha256"]:
        raise ValueError("source epoch history SHA256 differs")
    references = {}
    for epoch in args.epochs:
        item = source["artifacts"][f"epoch_{epoch - 1}"]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"epoch checkpoint SHA256 differs: {epoch}")
        payload = torch.load(item["path"], map_location="cpu", weights_only=False)
        if (payload["epoch"] != epoch - 1 or payload["model_config"] != source["model_config"]
                or payload["context_fingerprint"] != source["context_fingerprint"]
                or payload["test_metrics"] != history[epoch - 1]["test"]):
            raise ValueError(f"epoch metadata differs: {epoch}")
        state = payload["model_state"]
        if (set(state) != set(baseline["model_state"]) or any(
                value.shape != baseline["model_state"][key].shape for key, value in state.items())):
            raise ValueError(f"epoch task tensors differ: {epoch}")
        if epoch == source["selection"]["epoch"] + 1 and any(
                not torch.equal(value, baseline["model_state"][key]) for key, value in state.items()):
            raise ValueError("source full-test metrics do not match the saved selected epoch")
        references[str(epoch)] = item
    del payload
    source_names = ("cscwd_router/code/scripts/compare_vat_epochs.py",
                    "cscwd_router/code/stag/vat_epoch_selection.py", "cscwd_router/code/stag/dinov3_sources.py")
    context = {"format": "cscwd_router_vat_epoch_comparison_v1",
        "source_summary": {"path": str(args.source_summary.resolve()), "sha256": sha256(args.source_summary)},
        "source_context_fingerprint": source["context_fingerprint"], "model_config": source["model_config"],
        "backbone": source["artifacts"]["backbone"], "epoch_checkpoints": references,
        "test_annotation": {"path": str(annotation.resolve()), "sha256": sha256(annotation)},
        "frame_rate": 1, "precision": "fp32", "query_unit": "person", "seed": args.seed,
        "batch_size": args.batch_size, "workers": args.workers, "device": args.device,
        "sources": {**original.source_hashes(), **{name: sha256(ROOT / name) for name in source_names}}}
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    if args.check_only:
        dataset = make_full_test("vat", args.data_path, None)
        print(json.dumps({"checked_epochs_one_based": args.epochs, "persons": len(dataset),
                          "precision": "fp32", "frame_rate": 1, "context_fingerprint": fingerprint}))
        return
    manifest_path = args.report_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("existing comparison differs or needs --resume")
    runtime = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "python": platform.python_version(), "torch": str(torch.__version__),
               "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    original.atomic_json(manifest_path, {**context, "fingerprint": fingerprint, "runtime": runtime})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    model = loader = None
    rows = []
    for epoch in args.epochs:
        result_path = args.report_dir / f"epoch_{epoch:03d}.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if (row["context_fingerprint"] != fingerprint or sha256(row["predictions"]["path"]) != row["predictions"]["sha256"]):
                raise ValueError(f"cached evaluation differs: {epoch}")
        else:
            reuse = epoch == source["selection"]["epoch"] + 1
            if reuse:
                metrics, predictions = source["metrics"], source["artifacts"]["per_query_predictions"]
                if sha256(predictions["path"]) != predictions["sha256"]:
                    raise ValueError("source full-test predictions SHA256 differs")
            else:
                if model is None:
                    config = source["model_config"]
                    model, transform = get_coverage_router_model(config["model"], **{k: v for k, v in config.items() if k != "model"})
                    model.to(args.device).eval().requires_grad_(False)
                    loader = make_loader(make_full_test("vat", args.data_path, transform), args.batch_size, args.workers)
                checkpoint = torch.load(references[str(epoch)]["path"], map_location="cpu", weights_only=False)
                original.restore_task(model, checkpoint["model_state"])
                path = args.run_dir / f"epoch_{epoch:03d}.predictions.jsonl.gz"
                print(f"FULL_TEST epoch={epoch} frame_rate=1 keep=0.5 precision=fp32", flush=True)
                metrics = evaluate(model, loader, args.device, dataset="vat", keep_ratio=.5, rows_path=path)
                predictions = {"path": str(path.resolve()), "sha256": sha256(path)}
            if (metrics["persons"], metrics["inframe_persons"]) != (source["metrics"]["persons"], source["metrics"]["inframe_persons"]):
                raise ValueError("full-test query counts differ from the source")
            row = {"epoch_one_based": epoch, "context_fingerprint": fingerprint, "metrics": metrics,
                   "checkpoint": references[str(epoch)], "predictions": predictions, "reused_source_full_test": reuse}
            original.atomic_json(result_path, row)
        rows.append(row)
        print(json.dumps({"epoch_one_based": epoch, "metrics": row["metrics"]}), flush=True)
    comparison = compare_epochs(rows)
    report = {"completed": True, "format": context["format"], "context_fingerprint": fingerprint,
              "frame_rate": 1, "precision": "fp32", "runtime": runtime, **comparison}
    summary_path = args.report_dir / "summary.json"
    original.atomic_json(summary_path, report)
    summary_path.with_suffix(".md").write_text(render_comparison(comparison, title="VAT epoch comparison", frame_rate=1))
    with args.provenance.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"\n## VAT epoch comparison · vit{source['size']}\n\n- report: `{summary_path.resolve()}`; SHA256: `{sha256(summary_path)}`\n")
        for row in rows:
            for role in ("checkpoint", "predictions"):
                item = row[role]
                handle.write(f"- epoch {row['epoch_one_based']} {role}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    print(f"COMPLETE {summary_path}\nRecommended epoch (equal relative regret): {comparison['recommended_epoch_one_based']}", flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
