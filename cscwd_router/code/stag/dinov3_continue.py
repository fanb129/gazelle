"""Low-LR VAT continuation from a completed Router run, at fixed K50."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import torch

from gazelle.routing.model import get_coverage_router_model
from cscwd_router.code.data_prep.train_test import make_datasets, make_full_test, make_loader
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_engine import evaluate, train_epoch
from cscwd_router.code.stag.engine import seed_everything

ROOT = original.ROOT
FORMAT = "cscwd_router_dinov3_continuation_v1"


def source_hashes():
    return {**original.source_hashes(), **{
        name: sha256(ROOT / name) for name in (
            "cscwd_router/code/stag/dinov3_continue.py",
            "cscwd_router/code/scripts/continue_router_dinov3.py")}}


def probe_initialization(model, loader, device, parent, context, output):
    """Measure the actual parent dense decoder; no dense training is involved."""
    state = original.task_state(model)
    item = parent["artifacts"]["initialization"]
    original.initialize(model, torch.load(item["path"], map_location="cpu", weights_only=False), "decoder")
    model.router_stage = "support_pilot"
    dense = evaluate(model, loader, device, dataset="vat", keep_ratio=1.0)
    model.router_stage = "backbone_sparse"
    k100 = evaluate(model, loader, device, dataset="vat", keep_ratio=1.0)
    original.restore_task(model, state)
    differences = {key: k100[key] - dense[key] for key in ("auc", "l2", "inout_ap")}
    report = {"context_fingerprint": context["fingerprint"], "test_frame_rate": 6,
              "initialization_native_dense": dense, "initialization_routed_k100": k100,
              "routed_minus_dense": differences}
    original.atomic_json(output, report)
    if max(abs(value) for value in differences.values()) > 1e-8:
        raise ValueError("initialization native dense and routed K100 metrics differ; inspect probes.json")
    return report


def run(args, *, model_factory=get_coverage_router_model):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    parent_dir = Path(args.parent_report_root) / f"vit{args.size}"
    parent_path = parent_dir / "summary.json"
    parent = json.loads(parent_path.read_text())
    parent_manifest = json.loads((parent_dir / "manifest.json").read_text())
    if not parent["completed"] or parent["dataset"] != "vat" or parent["size"] != args.size:
        raise ValueError("parent must be a completed VAT experiment of the same size")
    if parent_manifest["recipe"]["seed"] != args.seed or parent["model_config"]["keep_ratio"] != 0.5:
        raise ValueError("parent must use the same base seed and K50")
    for role in ("initialization", "backbone", "decoder_resume_checkpoint", "best_test_checkpoint"):
        item = parent["artifacts"][role]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"parent artifact SHA mismatch: {role}")
    run_dir = Path(args.output_root) / f"vit{args.size}"
    report_dir = Path(args.report_root) / f"vit{args.size}"
    if not args.resume and any(path.exists() and any(path.iterdir()) for path in (run_dir, report_dir)):
        raise FileExistsError("continuation outputs are not empty; use --resume or new paths")
    parent_run = Path(parent["artifacts"]["decoder_resume_checkpoint"]["path"]).parent.parent
    if run_dir.resolve() == parent_run.resolve() or report_dir.resolve() == parent_dir.resolve():
        raise ValueError("continuation must use new output/report directories")
    seed_everything(args.seed)
    recipe = parent_manifest["recipe"]
    model, transform = model_factory(parent["model_config"]["model"], keep_ratio=0.5,
                                     router_hidden_dim=recipe["router_hidden_dim"],
                                     router_temperature=recipe["router_temperature"],
                                     escape_tokens=recipe["escape_tokens"])
    train, test, data = make_datasets("vat", args.data_path, transform, train_rate=6, test_rate=6)
    if data != parent_manifest["data"]:
        raise ValueError("continuation data differs from parent train/test views")
    context = {"parent_report": {"path": str(parent_path.resolve()), "sha256": sha256(parent_path)},
               "parent_fingerprint": parent["context_fingerprint"], "data": data,
               "recipe": {key: value for key, value in vars(args).items()
                          if key not in ("resume", "output_root", "report_root", "provenance")},
               "sources": source_hashes(), "torch_version": torch.__version__}
    context["fingerprint"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest_path, summary_path = report_dir / "manifest.json", report_dir / "summary.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != context["fingerprint"]:
            raise ValueError("continuation resume configuration or source differs")
    if args.resume and summary_path.exists():
        completed = json.loads(summary_path.read_text())
        for item in completed["artifacts"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"completed artifact SHA mismatch: {item['path']}")
        print(f"COMPLETE already verified: {summary_path}", flush=True)
        return completed
    parent_last = torch.load(parent["artifacts"]["decoder_resume_checkpoint"]["path"],
                             map_location="cpu", weights_only=False)
    if parent_last["selection"] != parent["selection"] or parent_last["context_fingerprint"] != parent["context_fingerprint"]:
        raise ValueError("parent last checkpoint and summary disagree")
    original.restore_task(model, parent_last["model_state"])
    device = torch.device(args.device)
    model.to(device)
    test_loader = make_loader(test, args.eval_batch_size, args.workers)
    optimizer_args = SimpleNamespace(decoder_lr=args.decoder_lr, inout_lr=args.inout_lr, weight_decay=0.0)
    optimizer = original.optimizer_for(model, "sparse_decoder", optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.additional_epochs)
    # A fresh low-LR optimizer is intentional: the parent's cosine schedule ended.
    selection = {**parent["selection"], "origin": "parent"}
    best_state = parent_last["best_model_state"]
    history, start = [], 0
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["context_fingerprint"] != context["fingerprint"]:
            raise ValueError("continuation checkpoint context differs")
        original.restore_task(model, last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        history, selection, best_state = last["history"], last["selection"], last["best_model_state"]
        start = last["additional_epoch"] + 1
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "cuda_version": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    original.atomic_json(manifest_path, {**context, "runtime": runtime})
    probes_path = report_dir / "probes.json"
    if not probes_path.exists():
        probe_initialization(model, test_loader, device, parent, context, probes_path)
    probes = json.loads(probes_path.read_text())
    if probes["context_fingerprint"] != context["fingerprint"]:
        raise ValueError("initialization probes have a different context")
    if max(abs(value) for value in probes["routed_minus_dense"].values()) > 1e-8:
        raise ValueError("initialization native dense and routed K100 metrics differ; inspect probes.json")
    offset = parent_last["epoch"] + 1
    history_path = report_dir / "history.json"
    for extra in range(start, args.additional_epochs):
        epoch = offset + extra
        seed_everything(args.seed + epoch)
        loader = make_loader(train, args.batch_size, args.workers, training=True, seed=args.seed + epoch)
        print(f"CONTINUE epoch={epoch + 1} additional={extra + 1}/{args.additional_epochs} keep=0.5", flush=True)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        trained = train_epoch(model, loader, optimizer, device, stage="sparse_decoder",
                              accumulation=args.grad_accumulation, clip_norm=args.clip_norm,
                              keep_ratio=0.5, budget_weight=0.0, log_every=args.log_every)
        tested = evaluate(model, test_loader, device, dataset="vat", keep_ratio=0.5)
        scheduler.step()
        history.append({"epoch": epoch, "additional_epoch": extra, "learning_rates": learning_rates,
                        "train": trained, "test": tested})
        if original.selection_better(tested, selection, "l2"):
            selection = {"epoch": epoch, "origin": "continuation", "metric": "l2", "metrics": tested,
                         "policy": "test_selected", "tie_breakers": ["auc_max", "earliest_epoch"]}
            best_state = original.task_state(model)
        original.atomic_torch(last_path, {"format": FORMAT, "epoch": epoch, "additional_epoch": extra,
            "model_config": model.get_model_config(), "model_state": original.task_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "selection": selection, "best_model_state": best_state, "history": history,
            "context_fingerprint": context["fingerprint"], "context": context})
        original.atomic_json(history_path, history)
        print(json.dumps(history[-1], allow_nan=False), flush=True)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    original.atomic_json(history_path, last["history"])
    original.restore_task(model, last["best_model_state"])
    best_path = run_dir / "best_test.pt"
    original.atomic_torch(best_path, {"format": FORMAT, "epoch": last["selection"]["epoch"],
        "selection": last["selection"], "model_config": model.get_model_config(),
        "model_state": last["best_model_state"], "context": context,
        "context_fingerprint": context["fingerprint"]})
    final_loader = make_loader(make_full_test("vat", args.data_path, transform), args.eval_batch_size, args.workers)
    rows_path = run_dir / "predictions.jsonl.gz"
    metrics = evaluate(model, final_loader, device, dataset="vat", keep_ratio=0.5, rows_path=rows_path)
    paths = {"parent_summary": parent_path, "parent_last": parent["artifacts"]["decoder_resume_checkpoint"]["path"],
             "parent_best": parent["artifacts"]["best_test_checkpoint"]["path"],
             "initialization": parent["artifacts"]["initialization"]["path"],
             "backbone": parent["artifacts"]["backbone"]["path"], "best_test_checkpoint": best_path,
             "decoder_resume_checkpoint": last_path, "per_query_predictions": rows_path, "probes": probes_path}
    artifacts = {role: {"path": str(Path(path).resolve()), "sha256": sha256(path)} for role, path in paths.items()}
    summary = {"completed": True, "format": FORMAT, "dataset": "vat", "size": args.size,
        "model_config": model.get_model_config(), "selection_policy": "test_selected",
        "selection": last["selection"], "selection_test_frame_rate": 6, "final_test_frame_rate": 1,
        "metrics": metrics, "additional_epochs": len(last["history"]), "parent_decoder_epochs": offset,
        "optimizer_policy": "fresh_low_lr_from_parent_last", "parent_best_included_in_selection": True,
        "context_fingerprint": context["fingerprint"], "artifacts": artifacts, "runtime": runtime}
    original.atomic_json(summary_path, summary)
    with Path(args.provenance).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"\n## DINOv3 K50 continuation · vat / {args.size}\n\n")
        handle.write(f"- report: `{summary_path}`; SHA256: `{sha256(summary_path)}`\n")
        for role, item in artifacts.items():
            handle.write(f"- {role}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    print(f"COMPLETE {summary_path}\n" + json.dumps(summary, indent=2), flush=True)
    return summary
