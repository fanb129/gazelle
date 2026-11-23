"""Transfer the complete GF Router and heatmap decoder to VAT at K50."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch.nn import functional as F

from gazelle.routing.losses import coverage_router_loss
from gazelle.routing.model import get_coverage_router_model
from scripts.train_coverage_router import stack_predictions, unpack_batch
from cscwd_router.code.data_prep.train_test import make_datasets, make_full_test, make_loader
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256, state_sha256
from cscwd_router.code.stag.dinov3_engine import evaluate
from cscwd_router.code.stag.engine import seed_everything
from cscwd_router.code.stag.dinov3_sources import load_source

ROOT = original.ROOT
FORMAT = "cscwd_router_gf_to_vat_v1"


def is_inout(name):
    return name.startswith(("inout_token.", "inout_head."))


def parameter_group(name):
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("router."):
        return "router"
    return "inout" if is_inout(name) else "decoder"


def transfer_state(model, gf_state):
    """Require all shared tensors, leaving only new VAT in/out tensors random."""
    target = original.task_state(model)
    shared = {name for name in target if not is_inout(name)}
    if set(gf_state) != shared:
        raise ValueError(f"GF transfer keys differ: missing={sorted(shared - set(gf_state))}, "
                         f"unexpected={sorted(set(gf_state) - shared)}")
    model.load_state_dict(gf_state, strict=False)
    loaded = original.task_state(model)
    if any(not torch.equal(loaded[name], gf_state[name]) for name in shared):
        raise ValueError("shared VAT tensors differ from GF after transfer")
    return {"loaded_gf_tensors": len(gf_state), "new_inout_keys": sorted(set(target) - shared),
            "shared_tensors_identical": True,
            "gf_router_sha256": state_sha256({k: v for k, v in gf_state.items() if k.startswith("router.")}),
            "gf_decoder_sha256": state_sha256({k: v for k, v in gf_state.items() if not k.startswith("router.")}),
            "new_inout_sha256": state_sha256({k: v for k, v in loaded.items() if is_inout(k)}),
            "initial_task_sha256": state_sha256(loaded)}


def configure(model, router_active):
    model.train()
    model.router_stage = "backbone_sparse"
    model.router.keep_ratio = 0.5
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("backbone.") and
                                 (router_active or not name.startswith("router.")))
    model.backbone.eval()
    model.router.train(router_active)


def optimizer_for(model, args):
    groups = {"router": [], "decoder": [], "inout": []}
    for name, parameter in model.named_parameters():
        if not name.startswith("backbone."):
            groups[parameter_group(name)].append(parameter)
    return torch.optim.AdamW([{"params": params, "name": name, "lr": getattr(args, name + "_lr")}
                             for name, params in groups.items()], weight_decay=0.0)


def train_epoch(model, loader, optimizer, device, args, *, router_active):
    configure(model, router_active)
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("loss", "heatmap_loss", "inout_loss", "router_loss")}
    persons = updates = 0
    started = time.monotonic()
    for index, batch in enumerate(loader):
        inputs, _, _, flags, _, _, targets = unpack_batch(batch, device, training=True)
        output = model(inputs)
        maps, inout = stack_predictions(output)
        valid = flags.bool()
        heatmap = F.binary_cross_entropy(maps[valid].float(), targets[valid].float()) if valid.any() else maps.sum() * 0
        classification = F.binary_cross_entropy(inout.float(), flags.float())
        router_loss = coverage_router_loss(output["routing"], targets, inout=flags,
                                           budget_weight=args.budget_weight, budget_target=0.5)["total"]
        loss = heatmap + classification + (args.router_loss_weight * router_loss if router_active else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite transfer loss at batch {index}")
        group_start = index // args.grad_accumulation * args.grad_accumulation
        group_size = min(args.grad_accumulation, len(loader) - group_start)
        (loss / group_size).backward()
        if (index + 1) % args.grad_accumulation == 0 or index + 1 == len(loader):
            if args.clip_norm:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], args.clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        count = len(flags)
        for key, value in zip(totals, (loss, heatmap, classification, router_loss)):
            totals[key] += value.item() * count
        persons += count
        if index % args.log_every == 0:
            print(f"TRAIN_TRANSFER batch={index}/{len(loader)} loss={loss.item():.6f} "
                  f"router_active={router_active} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    parameters = dict.fromkeys(("backbone", "router", "decoder", "inout"), 0)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters[parameter_group(name)] += parameter.numel()
    return {**{key: value / persons for key, value in totals.items()}, "persons": persons,
            "optimizer_steps": updates, "seconds": time.monotonic() - started,
            "router_active": router_active, "train_keep_ratio": 0.5, "trainable_parameters": parameters}


def source_hashes():
    return {**original.source_hashes(), **{name: sha256(ROOT / name) for name in (
        "cscwd_router/code/stag/dinov3_transfer.py", "cscwd_router/code/scripts/transfer_gf_router_to_vat.py",
        "cscwd_router/code/stag/dinov3_sources.py")}}


def run(args, *, model_factory=get_coverage_router_model):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    gf_path = Path(args.gf_summary)
    gf, gf_manifest, payload = load_source(gf_path, allow_closed_gf=args.allow_closed_gf)
    gf_manifest_path = gf_path.with_name("manifest.json")
    if (gf["dataset"] != "gazefollow" or gf["model_config"]["keep_ratio"] != 0.5
            or gf["model_config"]["router_stage"] != "backbone_sparse"):
        raise ValueError("source must be an evaluated K50 GF Router experiment")
    if gf_manifest["fingerprint"] != gf["context_fingerprint"] or gf_manifest["model_config"] != gf["model_config"]:
        raise ValueError("GF manifest and summary disagree")
    size = gf["size"]
    if gf["model_config"]["model"] != f"gazelle_dinov3_vit{size}16":
        raise ValueError("GF source model name and size disagree")
    checkpoint = gf["artifacts"]["best_test_checkpoint"]
    backbone = gf["artifacts"]["backbone"]
    if sha256(checkpoint["path"]) != checkpoint["sha256"] or sha256(backbone["path"]) != backbone["sha256"]:
        raise ValueError("GF source checkpoint or backbone SHA256 mismatch")
    if sha256(ROOT / "checkpoints" / f"dinov3_vit{size}16_pretrain.pth") != backbone["sha256"]:
        raise ValueError("factory backbone differs from GF source")
    run_dir, report_dir = Path(args.run_dir), Path(args.report_dir)
    source_dir = Path(checkpoint["path"]).parent
    if run_dir.resolve() in (source_dir.resolve(), source_dir.parent.resolve()) or report_dir.resolve() == gf_path.parent.resolve():
        raise ValueError("transfer must use new output/report directories")
    if not args.resume and any(path.exists() and any(path.iterdir()) for path in (run_dir, report_dir)):
        raise FileExistsError("transfer outputs are not empty; use --resume or new directories")
    if payload["selection"] != gf["selection"] or payload["context_fingerprint"] != gf["context_fingerprint"]:
        raise ValueError("GF source checkpoint and summary disagree")
    seed_everything(args.seed)
    recipe = gf_manifest["recipe"]
    model, transform = model_factory(gf["model_config"]["model"] + "_inout", keep_ratio=0.5,
                                     router_hidden_dim=recipe["router_hidden_dim"],
                                     router_temperature=recipe["router_temperature"], escape_tokens=recipe["escape_tokens"])
    if {**model.get_model_config(), "model": gf["model_config"]["model"]} != gf["model_config"]:
        raise ValueError("VAT Router configuration differs from GF source")
    initialization = transfer_state(model, payload["model_state"])
    train, test, data = make_datasets("vat", args.data_path, transform,
                                    train_rate=args.train_frame_rate, test_rate=args.test_frame_rate)
    context = {"gf_summary": {"path": str(gf_path.resolve()), "sha256": sha256(gf_path)},
        "gf_manifest": {"path": str(gf_manifest_path.resolve()), "sha256": sha256(gf_manifest_path)},
        "gf_checkpoint": checkpoint, "backbone": backbone, "initialization": initialization,
        "gf_source_status": {"completed": gf["completed"], "closed_by_user": gf.get("closed_by_user", False),
                             "training_epochs": gf["training_epochs"]},
        "data": data, "sources": source_hashes(), "torch_version": torch.__version__,
        "recipe": {key: value for key, value in vars(args).items() if key not in ("resume", "run_dir", "report_dir", "provenance")}}
    context["fingerprint"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest_path, summary_path = report_dir / "manifest.json", report_dir / "summary.json"
    if args.resume and manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != context["fingerprint"]:
        raise ValueError("transfer resume configuration or source differs")
    if args.resume and summary_path.exists():
        completed = json.loads(summary_path.read_text())
        for item in completed["artifacts"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"completed artifact SHA mismatch: {item['path']}")
        print(f"COMPLETE already verified: {summary_path}", flush=True)
        return completed
    device = torch.device(args.device)
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_version": torch.version.cuda,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    original.atomic_json(manifest_path, {**context, "runtime": runtime})
    original.atomic_json(report_dir / "initialization.json", {**initialization, "context_fingerprint": context["fingerprint"]})
    initial_path = run_dir / "initial.pt"
    if not initial_path.exists():
        original.atomic_torch(initial_path, {"format": FORMAT, "epoch": -1, "model_config": model.get_model_config(),
            "model_state": original.task_state(model), "context_fingerprint": context["fingerprint"], "context": context})
    model.to(device)
    test_loader = make_loader(test, args.eval_batch_size, args.workers)
    initial_test_path = report_dir / "initial_test.json"
    if not initial_test_path.exists():
        initial_test = evaluate(model, test_loader, device, dataset="vat", keep_ratio=0.5)
        original.atomic_json(initial_test_path, {"context_fingerprint": context["fingerprint"],
            "inout_initialization": "random_seeded", "checkpoint_selection_candidate": False, "metrics": initial_test})
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, selection, best_state, start = [], None, None, 0
    last_path, history_path = run_dir / "last.pt", report_dir / "history.json"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["context_fingerprint"] != context["fingerprint"]:
            raise ValueError("transfer checkpoint context differs")
        original.restore_task(model, last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        history, selection, best_state, start = last["history"], last["selection"], last["best_model_state"], last["epoch"] + 1
    for epoch in range(start, args.epochs):
        seed_everything(args.seed + epoch)
        router_active = epoch >= args.router_warmup_epochs and args.router_lr > 0
        print(f"TRANSFER epoch={epoch + 1}/{args.epochs} router_active={router_active} keep=0.5", flush=True)
        loader = make_loader(train, args.batch_size, args.workers, training=True, seed=args.seed + epoch)
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        trained = train_epoch(model, loader, optimizer, device, args, router_active=router_active)
        tested = evaluate(model, test_loader, device, dataset="vat", keep_ratio=0.5)
        scheduler.step()
        history.append({"epoch": epoch, "learning_rates": rates, "train": trained, "test": tested})
        state = original.task_state(model)
        if original.selection_better(tested, selection, "l2"):
            selection = {"epoch": epoch, "metric": "l2", "metrics": tested, "policy": "test_selected",
                         "tie_breakers": ["auc_max", "earliest_epoch"]}
            best_state = state
        compact = {"format": FORMAT, "epoch": epoch, "model_config": model.get_model_config(),
                   "model_state": state, "context_fingerprint": context["fingerprint"], "context": context}
        original.atomic_torch(run_dir / f"epoch_{epoch}.pt", {**compact, "test_metrics": tested})
        original.atomic_torch(last_path, {**compact, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "selection": selection, "best_model_state": best_state})
        original.atomic_json(history_path, history)
        print(json.dumps(history[-1], allow_nan=False), flush=True)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    original.atomic_json(history_path, last["history"])
    original.restore_task(model, last["best_model_state"])
    best_path = run_dir / "best_test.pt"
    original.atomic_torch(best_path, {"format": FORMAT, "epoch": last["selection"]["epoch"], "selection": last["selection"],
        "model_config": model.get_model_config(), "model_state": last["best_model_state"],
        "context_fingerprint": context["fingerprint"], "context": context})
    rows_path = run_dir / "predictions.jsonl.gz"
    full_loader = make_loader(make_full_test("vat", args.data_path, transform), args.eval_batch_size, args.workers)
    metrics = evaluate(model, full_loader, device, dataset="vat", keep_ratio=0.5, rows_path=rows_path)
    paths = {"gf_summary": gf_path, "gf_manifest": gf_manifest_path, "gf_checkpoint": checkpoint["path"],
        "backbone": backbone["path"], "initial_checkpoint": initial_path, "best_test_checkpoint": best_path,
        "decoder_resume_checkpoint": last_path, "per_query_predictions": rows_path,
        "manifest": manifest_path, "history": history_path,
        "initialization_report": report_dir / "initialization.json", "initial_test_report": initial_test_path}
    paths.update({f"epoch_{epoch}": run_dir / f"epoch_{epoch}.pt" for epoch in range(args.epochs)})
    artifacts = {role: {"path": str(Path(path).resolve()), "sha256": sha256(path)} for role, path in paths.items()}
    summary = {"completed": True, "format": FORMAT, "dataset": "vat", "size": size,
        "model_config": model.get_model_config(), "selection_policy": "test_selected", "selection": last["selection"],
        "selection_test_frame_rate": args.test_frame_rate, "final_test_frame_rate": 1, "metrics": metrics,
        "epochs": len(last["history"]), "router_warmup_epochs": args.router_warmup_epochs,
        "initialization": initialization, "artifacts": artifacts, "runtime": runtime,
        "context_fingerprint": context["fingerprint"], "gf_source_status": context["gf_source_status"]}
    original.atomic_json(summary_path, summary)
    with Path(args.provenance).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"\n## GF Router to VAT · vit{size}\n\n- report: `{summary_path}`; SHA256: `{sha256(summary_path)}`\n")
        for role, item in artifacts.items():
            handle.write(f"- {role}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    print(f"COMPLETE {summary_path}\n" + json.dumps(summary, indent=2), flush=True)
    return summary
