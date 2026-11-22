"""One B/L Router experiment, with atomic epoch recovery and test selection."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from gazelle.routing.model import get_coverage_router_model
from cscwd_router.code.data_prep.train_test import make_datasets, make_full_test, make_loader
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_engine import configure_stage, evaluate, train_epoch
from cscwd_router.code.stag.engine import seed_everything


ROOT = Path(__file__).resolve().parents[3]
FORMAT = "cscwd_router_dinov3_testselected_v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def task_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if not name.startswith("backbone.")}


def restore_task(model, state):
    required = {name for name in model.state_dict() if not name.startswith("backbone.")}
    if set(state) != required:
        raise ValueError(f"task checkpoint keys differ: missing={sorted(required - set(state))}, "
                         f"unexpected={sorted(set(state) - required)}")
    model.load_state_dict(state, strict=False)


def initialize(model, payload, kind):
    state = payload.get("model_state", payload)
    state = {name: value for name, value in state.items() if not name.startswith("backbone.")}
    required = {name for name in model.state_dict() if not name.startswith("backbone.")}
    if kind == "decoder":
        state = {name: value for name, value in state.items() if not name.startswith("router.")}
        required = {name for name in required if not name.startswith("router.")}
    if set(state) != required:
        raise ValueError(f"initialization keys differ for {kind}: "
                         f"missing={sorted(required - set(state))}, unexpected={sorted(set(state) - required)}")
    # Shape mismatches (e.g. B versus L, or single versus four layers) stay errors.
    model.load_state_dict(state, strict=False)
    return {"kind": kind, "loaded_tensors": len(state), "router_reinitialized": kind == "decoder"}


def selection_better(candidate, selection, metric):
    if selection is None:
        return True
    incumbent = selection["metrics"]
    return (candidate[metric], -candidate["auc"]) < (incumbent[metric], -incumbent["auc"])


def optimizer_for(model, stage, args):
    configure_stage(model, stage)
    groups = []
    if stage == "support":
        groups.append({"params": list(model.router.parameters()), "lr": args.router_lr})
    else:
        decoder, inout = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (inout if name.startswith(("inout_token.", "inout_head.")) else decoder).append(parameter)
        groups.append({"params": decoder, "lr": args.decoder_lr})
        if inout:
            groups.append({"params": inout, "lr": args.inout_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def stage_keep(stage, epoch, target):
    return 0.25 if stage == "support" else (1.0, (1 + target) / 2, target)[min(epoch, 2)]


def run_stage(model, train, test_loader, device, args, context, *, stage, epochs):
    stage_dir = Path(args.run_dir) / stage
    last_path = stage_dir / "last.pt"
    optimizer = optimizer_for(model, stage, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, selection, best_state, start_epoch = [], None, None, 0
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["context_fingerprint"] != context["fingerprint"] or last["stage"] != stage:
            raise ValueError("resume checkpoint does not match this experiment")
        restore_task(model, last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        history, selection, best_state = last["history"], last["selection"], last["best_model_state"]
        start_epoch = last["epoch"] + 1
        print(f"RESUME stage={stage} next_epoch={start_epoch}/{epochs}", flush=True)
    metric = "min_l2" if args.dataset == "gazefollow" else "l2"
    for epoch in range(start_epoch, epochs):
        epoch_seed = args.seed + epoch
        seed_everything(epoch_seed)
        loader = make_loader(train, args.batch_size, args.workers, training=True, seed=epoch_seed)
        active_keep = stage_keep(stage, epoch, args.keep_ratio)
        print(f"EPOCH stage={stage} epoch={epoch + 1}/{epochs} train_keep={active_keep}", flush=True)
        trained = train_epoch(model, loader, optimizer, device, stage=stage,
                              accumulation=1 if stage == "support" else args.grad_accumulation,
                              clip_norm=args.clip_norm, keep_ratio=active_keep,
                              budget_weight=args.budget_weight, log_every=args.log_every)
        trained["trainable_parameters"] = {
            group: sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and
                       (name.startswith(group + ".") if group in ("backbone", "router") else
                        not name.startswith(("backbone.", "router."))))
            for group in ("backbone", "router", "decoder_and_inout")}
        tested = evaluate(model, test_loader, device, dataset=args.dataset, keep_ratio=args.keep_ratio)
        scheduler.step()
        history.append({"epoch": epoch, "train": trained, "test": tested})
        if stage != "support" and selection_better(tested, selection, metric):
            selection = {"epoch": epoch, "metric": metric, "metrics": tested,
                         "policy": "test_selected", "tie_breakers": ["auc_max", "earliest_epoch"]}
            best_state = task_state(model)
        payload = {"format": FORMAT, "format_version": 1, "stage": stage, "epoch": epoch,
                   "model_config": model.get_model_config(), "model_state": task_state(model),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "selection": selection, "best_model_state": best_state, "history": history,
                   "context_fingerprint": context["fingerprint"], "context": context}
        # The atomic last checkpoint is the epoch commit point. Best state is
        # embedded so an interruption between file writes cannot select a later epoch.
        atomic_torch(last_path, payload)
        atomic_json(Path(args.report_dir) / f"{stage}_history.json", history)
        print(json.dumps(history[-1], allow_nan=False), flush=True)
    if not last_path.exists():
        raise RuntimeError(f"stage did not produce a checkpoint: {stage}")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    atomic_json(Path(args.report_dir) / f"{stage}_history.json", last["history"])
    selected = last["selection"]
    state = last["model_state"] if stage == "support" else last["best_model_state"]
    output = stage_dir / ("final.pt" if stage == "support" else "best_test.pt")
    if start_epoch < epochs or not output.exists():
        atomic_torch(output, {"format": FORMAT, "format_version": 1,
                             "model_config": model.get_model_config(), "model_state": state,
                             "stage": stage, "epoch": last["epoch"] if selected is None else selected["epoch"],
                             "selection": selected, "context": context,
                             "context_fingerprint": context["fingerprint"]})
    restore_task(model, state)
    return output, last["history"], selected


def source_hashes():
    names = ["gazelle/backbone.py", "gazelle/model.py", "gazelle/dataloader.py", "gazelle/utils.py",
             "gazelle/routing/backbone.py", "gazelle/routing/model.py", "gazelle/routing/router.py",
             "gazelle/routing/losses.py", "scripts/train_coverage_router.py",
             "cscwd_router/code/data_prep/gazefollow.py", "cscwd_router/code/stag/checkpoint.py",
             "cscwd_router/code/stag/engine.py",
             "cscwd_router/code/data_prep/train_test.py", "cscwd_router/code/stag/dinov3_engine.py",
             "cscwd_router/code/stag/dinov3_run.py", "cscwd_router/code/stag/dinov3_initializers.py",
             "cscwd_router/code/scripts/run_router_dinov3_queue.py",
             "cscwd_router/code/scripts/train_router_dinov3.py"]
    hashes = {name: sha256(ROOT / name) for name in names}
    native = hashlib.sha256()
    for path in sorted((ROOT / "dinov3").rglob("*.py")):
        native.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    hashes["native_dinov3_source_tree"] = native.hexdigest()
    return hashes


def register_provenance(path, summary_path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"\n## DINOv3 test-selected · {summary['dataset']} / {summary['size']}\n",
             f"- report: `{summary_path}`; SHA256: `{sha256(summary_path)}`\n",
             "- protocol: official train/test, no validation; FP32 accuracy; frozen backbone\n"]
    for role, item in summary["artifacts"].items():
        lines.append(f"- {role}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write("".join(lines))
        handle.flush()


def run(args, *, model_factory=get_coverage_router_model):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    run_dir, report_dir = Path(args.run_dir), Path(args.report_dir)
    if not args.resume and any(p.exists() and any(p.iterdir()) for p in (run_dir, report_dir)):
        raise FileExistsError("run/report directory is not empty; use --resume or a new output path")
    init_path = Path(args.init_checkpoint)
    init_sha = sha256(init_path)
    if init_sha != args.init_sha256:
        raise ValueError(f"initialization SHA256 mismatch: {init_path}")
    backbone_path = ROOT / "checkpoints" / f"dinov3_vit{args.size}16_pretrain.pth"
    backbone_sha = sha256(backbone_path)
    if backbone_sha != args.backbone_sha256:
        raise ValueError(f"backbone SHA256 mismatch: {backbone_path}")
    seed_everything(args.seed)
    model_name = f"gazelle_dinov3_vit{args.size}16" + ("_inout" if args.dataset == "vat" else "")
    model, transform = model_factory(model_name, keep_ratio=args.keep_ratio,
                                     router_hidden_dim=args.router_hidden_dim,
                                     router_temperature=args.router_temperature,
                                     escape_tokens=args.escape_tokens)
    initialization = initialize(model, torch.load(init_path, map_location="cpu", weights_only=False), args.init_kind)
    train, test, data = make_datasets(args.dataset, args.data_path, transform,
                                    train_rate=args.train_frame_rate, test_rate=args.test_frame_rate)
    recipe = {key: value for key, value in vars(args).items()
              if key not in ("resume", "run_dir", "report_dir", "provenance")}
    context = {"recipe": recipe, "model_config": model.get_model_config(), "data": data,
               "initialization": {**initialization, "path": str(init_path.resolve()), "sha256": init_sha},
               "backbone": {"path": str(backbone_path.resolve()), "sha256": backbone_sha},
               "sources": source_hashes(), "torch_version": torch.__version__}
    context["fingerprint"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    summary_path = report_dir / "summary.json"
    manifest_path = report_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != context["fingerprint"]:
            raise ValueError("resume manifest has a different configuration or source")
    if args.resume and summary_path.exists():
        completed = json.loads(summary_path.read_text())
        if completed["context_fingerprint"] != context["fingerprint"]:
            raise ValueError("completed experiment has a different configuration or source")
        for item in completed["artifacts"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"completed artifact SHA mismatch: {item['path']}")
        print(f"COMPLETE already verified: {summary_path}", flush=True)
        return completed
    device = torch.device(args.device)
    model.to(device)
    test_loader = make_loader(test, args.eval_batch_size, args.workers)
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "cuda_version": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    atomic_json(manifest_path, {**context, "runtime": runtime})
    print(json.dumps({"dataset": args.dataset, "size": args.size, "initialization": initialization,
                      "route_after_block": model.route_after_block, "data": data, "runtime": runtime}), flush=True)
    artifacts = {"initialization": context["initialization"], "backbone": context["backbone"]}
    if args.init_kind == "decoder":
        support_path, _, _ = run_stage(model, train, test_loader, device, args, context,
                                       stage="support", epochs=args.support_epochs)
        artifacts["support_checkpoint"] = {"path": str(support_path.resolve()), "sha256": sha256(support_path)}
        support_last = support_path.with_name("last.pt")
        artifacts["support_resume_checkpoint"] = {"path": str(support_last.resolve()), "sha256": sha256(support_last)}
    best_path, history, selection = run_stage(model, train, test_loader, device, args, context,
                                             stage="sparse_decoder", epochs=args.decoder_epochs)
    if args.dataset == "vat":
        final_test = make_full_test(args.dataset, args.data_path, transform)
        final_loader = make_loader(final_test, args.eval_batch_size, args.workers)
    else:
        final_loader = test_loader
    rows_path = run_dir / "predictions.jsonl.gz"
    final_metrics = evaluate(model, final_loader, device, dataset=args.dataset,
                             keep_ratio=args.keep_ratio, rows_path=rows_path)
    artifacts.update({"best_test_checkpoint": {"path": str(best_path.resolve()), "sha256": sha256(best_path)},
                      "decoder_resume_checkpoint": {"path": str(best_path.with_name("last.pt").resolve()),
                                                    "sha256": sha256(best_path.with_name("last.pt"))},
                      "per_query_predictions": {"path": str(rows_path.resolve()), "sha256": sha256(rows_path)}})
    summary = {"completed": True, "format": FORMAT, "dataset": args.dataset, "size": args.size,
               "model_config": model.get_model_config(), "selection_policy": "test_selected",
               "selection": selection, "selection_test_frame_rate": data["test_frame_sample_every"],
               "final_test_frame_rate": 1, "metrics": final_metrics, "artifacts": artifacts,
               "context_fingerprint": context["fingerprint"], "runtime": runtime,
               "training_epochs": {"support": 0 if args.init_kind == "support" else args.support_epochs,
                                   "sparse_decoder": len(history)}}
    atomic_json(summary_path, summary)
    register_provenance(args.provenance, summary_path, summary)
    print(f"COMPLETE {summary_path}\n" + json.dumps(summary, indent=2), flush=True)
    return summary
