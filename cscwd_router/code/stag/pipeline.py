"""Fixed-final 3+5+5 recipe with an identical official decoder branch start."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from gazelle.backbone_v0_dinov2 import DinoV2Backbone
from gazelle.model_v0_dinov2 import GazeLLE
from cscwd_router.code.stag.checkpoint import clone_state, read_checkpoint, restore_checkpoint, save_checkpoint, sha256, state_sha256
from cscwd_router.code.stag.engine import evaluate, seed_everything, train_epoch
from cscwd_router.code.stag.model_dinov2 import build_model


SUPPORT_EPOCHS = 3
SPARSE_KEEP_SCHEDULE = (1.0, 0.75, 0.5, 0.5, 0.5)
MATCHED_EPOCHS = 5


def epoch_seed(seed, stage, epoch):
    # Both decoder branches receive the same shuffle, augmentation and dropout RNG.
    return seed + (0 if stage == "support" else 10000) + epoch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class OfficialBackboneView(DinoV2Backbone):
    """Reuse loaded weights with the original dense forward method."""

    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model


@torch.inference_mode()
def initial_equivalence(model, batch, device):
    model.eval()
    batch = batch.to(device)
    reference = GazeLLE(
        OfficialBackboneView(model.encoder.backbone.model), dim=model.config["decoder_dim"],
        num_layers=model.config["decoder_layers"], in_size=(model.config["image_size"],) * 2,
    ).to(device).eval().requires_grad_(False)
    reference.load_gazelle_state_dict(clone_state(model.decoder))
    expected = torch.cat(reference(batch.inputs())["heatmap"])
    checks = {}
    for mode in ("dense", "sparse"):
        actual = torch.cat(model(batch.inputs(), mode=mode)["heatmap"])
        checks[mode] = {"equal": torch.equal(actual, expected),
                        "max_abs_diff": (actual - expected).abs().max().item(),
                        "mismatched_values": (actual != expected).sum().item(),
                        "finite": bool(torch.isfinite(actual).all())}
    return checks


def module_hashes(model):
    return {"decoder": state_sha256(model.decoder.state_dict()), "router": state_sha256(model.encoder.router.state_dict())}


def checkpoint_ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def run_pipeline(model, data, device, *, run_dir, report_dir, context, router_lr=1e-3,
                 decoder_lr=1e-4, weight_decay=0.0, budget_weight=0.05, seed=3106, log_every=50, resume=False):
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    run_dir.mkdir(parents=True, exist_ok=resume)
    report_dir.mkdir(parents=True, exist_ok=True)
    recipe = {"support_epochs": SUPPORT_EPOCHS, "support_keep_ratio": 0.25,
              "sparse_keep_schedule": list(SPARSE_KEEP_SCHEDULE), "matched_dense_epochs": MATCHED_EPOCHS,
              "router_lr": router_lr, "decoder_lr": decoder_lr, "weight_decay": weight_decay,
              "budget_weight": budget_weight, "seed": seed, "selection": "fixed_final_epoch"}
    identity = {"model": model.config, "data": data.metadata, "recipe": recipe,
                "train_config": context["train_config"], "sources": context["sources"],
                "external_weights": context["external_weights"]}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    context = {**context, "recipe": recipe, "data": data.metadata, "signature": signature}
    manifest_path = report_dir / "manifest.json"
    if resume:
        if json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("resume requires the same source, data split, weights and training configuration")
    else:
        with manifest_path.open("x") as stream:
            stream.write(json.dumps(context, indent=2) + "\n")

    baseline_path = run_dir / "dense_official/final.pt"
    example = next(iter(data.train_loader(seed)))
    if resume:
        baseline = read_checkpoint(baseline_path)
        restore_checkpoint(model, baseline)
        checks = json.loads((report_dir / "initial_equivalence.json").read_text())
    else:
        checks = initial_equivalence(model, example, device)
        write_json(report_dir / "initial_equivalence.json", checks)
        if not all(value["equal"] and value["finite"] for value in checks.values()):
            raise ValueError("initial dense/K100 predictions do not exactly match the official decoder")
        model.set_stage("dense_official")
        baseline = save_checkpoint(baseline_path, model, stage="dense_official", epoch=-1, context=context)
    original_decoder = baseline["decoder"]
    original_hash = state_sha256(original_decoder)
    validation = data.validation_loader()
    stages = []
    schedules = (("support", (0.25,) * SUPPORT_EPOCHS),
                 ("sparse_decoder", SPARSE_KEEP_SCHEDULE), ("matched_dense", (1.0,) * MATCHED_EPOCHS))
    for stage, schedule in schedules:
        final_path = run_dir / stage / "final.pt"
        last_path = run_dir / stage / "last.pt"
        if resume and final_path.exists():
            payload = read_checkpoint(final_path)
            if payload["context"]["signature"] != signature:
                raise ValueError("completed stage belongs to a different experiment")
            restore_checkpoint(model, payload)
            stage_report = payload["metrics"]
        else:
            if stage != "support":
                model.decoder.load_state_dict(original_decoder, strict=True)
            model.set_stage(stage)
            starts = module_hashes(model)
            learning_rate = router_lr if stage == "support" else decoder_lr
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                           lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(schedule), eta_min=1e-7)
            history, start_epoch = [], 0
            if resume and last_path.exists():
                payload = read_checkpoint(last_path)
                if payload["context"]["signature"] != signature:
                    raise ValueError("partial stage belongs to a different experiment")
                restore_checkpoint(model, payload, optimizer=optimizer, scheduler=scheduler)
                history, starts = payload["metrics"]["history"], payload["metrics"]["start_hashes"]
                start_epoch = payload["epoch"] + 1
            for epoch in range(start_epoch, len(schedule)):
                active_seed = epoch_seed(seed, stage, epoch)
                seed_everything(active_seed)
                model.encoder.router.keep_ratio = schedule[epoch]
                learning_rate = optimizer.param_groups[0]["lr"]
                metrics = train_epoch(model, data.train_loader(active_seed), optimizer, device,
                                       budget_weight=budget_weight, log_every=log_every)
                scheduler.step()
                row = {"epoch": epoch, "epoch_seed": active_seed, "keep_ratio": schedule[epoch],
                       "learning_rate": learning_rate, **metrics}
                history.append(row)
                print(json.dumps({"stage": stage, **row}), flush=True)
                save_checkpoint(last_path, model, stage=stage, epoch=epoch, context=context,
                                metrics={"history": history, "start_hashes": starts}, optimizer=optimizer, scheduler=scheduler)
                write_json(report_dir / f"{stage}_history.json", history)
            ends = module_hashes(model)
            mode = "sparse" if stage == "sparse_decoder" else "dense"
            validation_metrics = evaluate(model, validation, device, mode=mode) if validation is not None and stage != "support" else None
            stage_report = {
                "stage": stage, "epochs": len(history), "history": history, "start_hashes": starts, "end_hashes": ends,
                "decoder_changed": starts["decoder"] != ends["decoder"], "router_changed": starts["router"] != ends["router"],
                "optimizer_steps": sum(row["optimizer_steps"] for row in history),
                "training_persons_seen": sum(row["persons"] for row in history), "validation": validation_metrics,
                "backbone_trainable_parameters": sum(p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad),
                "backbone_parameters_with_grad": sum(p.grad is not None for p in model.encoder.backbone.parameters()),
            }
            save_checkpoint(final_path, model, stage=stage, epoch=len(schedule) - 1, context=context,
                            metrics=stage_report, optimizer=optimizer, scheduler=scheduler)
        stage_report = {**stage_report, "checkpoint": checkpoint_ref(final_path)}
        write_json(report_dir / f"{stage}_summary.json", stage_report)
        stages.append(stage_report)

    # Exercise both final checkpoint paths, with the same frozen backbone object.
    roundtrips = []
    for stage in ("sparse_decoder", "matched_dense"):
        payload = read_checkpoint(run_dir / stage / "final.pt")
        restore_checkpoint(model, payload)
        model.eval()
        fresh = build_model(model.config, backbone=model.encoder.backbone).to(device)
        restore_checkpoint(fresh, payload)
        fresh.eval()
        inputs = example.to(device).inputs()
        mode = "sparse" if stage == "sparse_decoder" else "dense"
        with torch.inference_mode():
            before = torch.cat(model(inputs, mode=mode)["heatmap"])
            after = torch.cat(fresh(inputs, mode=mode)["heatmap"])
        roundtrips.append({"stage": stage, "equal": torch.equal(before, after),
                           "max_abs_diff": (before - after).abs().max().item(), "finite": bool(torch.isfinite(after).all())})

    support, sparse, matched = stages
    engineering = {
        "initial_equivalence": all(item["equal"] and item["finite"] for item in checks.values()),
        "support_updates_router_only": support["router_changed"] and not support["decoder_changed"],
        "sparse_updates_decoder_only": sparse["decoder_changed"] and not sparse["router_changed"],
        "matched_updates_decoder_only": matched["decoder_changed"] and not matched["router_changed"],
        "same_official_decoder_start": sparse["start_hashes"]["decoder"] == matched["start_hashes"]["decoder"] == original_hash,
        "matched_optimizer_budget": sparse["optimizer_steps"] == matched["optimizer_steps"],
        "matched_training_sample_budget": sparse["training_persons_seen"] == matched["training_persons_seen"],
        "matched_epoch_seeds": [row["epoch_seed"] for row in sparse["history"]] == [row["epoch_seed"] for row in matched["history"]],
        "backbone_frozen": all(stage["backbone_trainable_parameters"] == stage["backbone_parameters_with_grad"] == 0 for stage in stages),
        "checkpoint_roundtrips": all(row["equal"] and row["finite"] for row in roundtrips),
    }
    result = {"task": "T3", "protocol": data.metadata["protocol"], "completed": True, "context": context,
              "baseline_checkpoint": checkpoint_ref(baseline_path), "initial_equivalence": checks,
              "stages": stages, "checkpoint_roundtrips": roundtrips, "engineering_checks": engineering,
              "passed": all(engineering.values())}
    write_json(report_dir / "summary.json", result)
    return result
