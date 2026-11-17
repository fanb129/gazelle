"""R4: train one causal refiner under the fixed compact development budget."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from prepare_development import artifact
from train_temporal_joint import training_inputs, train_epoch
from stag.cache import file_sha256, write_json
from stag.compact_protocol import (COMPACT_BUDGET, PLAN_VERSION, make_candidate,
                                   pareto_front, select_checkpoint, validate_budget)
from stag.compact_validation import evaluate_compact_model
from stag.history_model import HistoryValidityRefiner
from stag.temporal_model import CausalJointRefiner


MODEL_IDS = ("ordinary_gru", "history_validity", "always_keep")


def build_model(model_id, feature_channels, hidden_dim, temperature):
    if model_id == "ordinary_gru":
        return CausalJointRefiner(feature_channels, hidden_dim)
    if model_id in ("history_validity", "always_keep"):
        mode = "learned" if model_id == "history_validity" else "always_keep"
        return HistoryValidityRefiner(feature_channels, hidden_dim, mode, temperature)
    raise ValueError(f"Unknown compact model: {model_id}")


def shared_initialization_sha256(model):
    """Fingerprint common R2 layers before optimization, independently of R3 extras."""
    digest = hashlib.sha256()
    common = ("feature_projection.", "gru.", "heatmap_residual.", "inout_residual.")
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(common):
            digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_compact_training(feature_cache, feature_report, r1_report, r1_review, plan,
                         output_dir, model_id="ordinary_gru", epochs=5,
                         learning_rate=0.0001, lambda_state=0.1, hidden_dim=128,
                         clip_frames=16, batch_size=8, seed=0, weight_decay=0.01,
                         gradient_clip_norm=1.0, temperature=0.1, device="cuda:0"):
    config = {"model_id": model_id, "epochs": epochs, "learning_rate": learning_rate,
              "lambda_state": lambda_state, "hidden_dim": hidden_dim,
              "clip_frames": clip_frames, "batch_size": batch_size, "seed": seed,
              "weight_decay": weight_decay, "gradient_clip_norm": gradient_clip_norm,
              "temperature": temperature, "optimizer": "AdamW", "device": str(device),
              "precision": "float32; no_autocast; tf32_disabled",
              "history": "own_predicted_outputs; persistent_slots; detach_only_at_TBPTT_chunks",
              "initialization": "fresh_new_module; shared_seed_initialization; no_R2_resume",
              "selection": "precision_eligible_lexicographic_validation_only; no_test",
              "early_stopping": False}
    validate_budget(config)
    if PLAN_VERSION not in Path(plan).read_text():
        raise ValueError("Plan does not contain the compact protocol version")
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown compact model: {model_id}")
    r1, _, feature_index, features, prediction_index, arrays, tracks, sources = training_inputs(
        feature_cache, feature_report, r1_report, r1_review)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(model_id, feature_index["contract"]["channels"], hidden_dim, temperature).to(device)
    config["feature_channels"] = model.feature_channels
    shared_hash = shared_initialization_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_reports = {name: artifact(path) for name, path in (
        ("feature_report", feature_report), ("r1_report", r1_report),
        ("r1_review", r1_review), ("plan", plan))}
    write_json(output_dir / "training_config.json", {
        "plan_version": PLAN_VERSION, "plan": source_reports["plan"], "config": config,
        "sources": sources, "feature_contract": feature_index["contract"],
        "shared_initialization_sha256": shared_hash})
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    baseline = r1["baseline"]["validation"]
    history, candidates = [], []
    for epoch in range(1, epochs + 1):
        print(f"R4 {model_id} epoch {epoch}/{epochs}", flush=True)
        training = train_epoch(model, optimizer, tracks["train"], arrays, feature_index,
                               features, device, clip_frames, batch_size, seed + epoch - 1,
                               lambda_state)
        validation = evaluate_compact_model(model, tracks["validation"], arrays,
            feature_index, features, output_dir / f"validation_epoch_{epoch:03d}",
            prediction_index["person_frames"], device, clip_frames, batch_size,
            {"sources": sources, "config": config, "plan_version": PLAN_VERSION,
             "plan_sha256": source_reports["plan"]["sha256"], "epoch": epoch})
        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "model_id": model_id, "plan_version": PLAN_VERSION,
                    "plan_sha256": source_reports["plan"]["sha256"],
                    "feature_contract": feature_index["contract"], "sources": sources}, checkpoint_path)
        history.append({"epoch": epoch, "training": training, "validation": validation,
                        "checkpoint": artifact(checkpoint_path)})
        candidate = make_candidate(model_id, epoch, validation, baseline)
        candidates.append(candidate)
        write_json(output_dir / "training_history.json", history)
        print(f"Epoch {epoch}: train_loss={training['total_loss']:.6f} "
              f"validation={validation['standard']} "
              f"precision_eligible={candidate['precision_eligible']}", flush=True)
    last_path = output_dir / "last.pt"
    shutil.copyfile(history[-1]["checkpoint"]["path"], last_path)
    selected = select_checkpoint(candidates)
    selection = {"status": "no_precision_eligible_checkpoint", "epoch": None,
                 "candidate_id": None, "checkpoint": None}
    if selected is not None:
        selected_path = output_dir / "selected.pt"
        item = next(item for item in history if item["epoch"] == selected["epoch"])
        shutil.copyfile(item["checkpoint"]["path"], selected_path)
        selection = {"status": "selected", "epoch": selected["epoch"],
                     "candidate_id": selected["candidate_id"], "checkpoint": artifact(selected_path)}
    changed = {name: not torch.equal(initial[name], parameter.detach().cpu())
               for name, parameter in model.named_parameters()}
    checks = {"train_frames_match_manifest": all(item["training"]["valid_frames"] ==
                r1["split"]["partitions"]["train"]["counts"]["person_frames"] for item in history),
              "validation_frames_match_manifest": all(item["validation"]["standard"]["person_frames"] ==
                r1["split"]["partitions"]["validation"]["counts"]["person_frames"] for item in history),
              "all_planned_epochs_completed": [item["epoch"] for item in history] == list(range(1, epochs + 1)),
              "heatmap_head_updated": any(value for name, value in changed.items() if name.startswith("heatmap_residual")),
              "state_head_updated": any(value for name, value in changed.items() if name.startswith("inout_residual")),
              "only_new_module_parameters": not any(name.startswith("backbone") for name in changed),
              "frozen_features_read_only": not features.flags.writeable,
              "prediction_arrays_read_only": all(not values.flags.writeable for values in arrays.values())}
    if model_id != "ordinary_gru":
        checks["history_gate_updated"] = any(value for name, value in changed.items() if name.startswith("history_gate"))
        checks["all_epochs_have_history_diagnostics"] = all(item["validation"]["diagnostics"]["passed"] for item in history)
    if not all(checks.values()):
        raise ValueError(f"R4 compact training verification failed: {checks}")
    report = {"task": "R4_compact_training", "model_id": model_id, "passed": True,
              "stage_status": "awaiting_development_review", "plan_version": PLAN_VERSION,
              "plan": source_reports["plan"], "config": config, "sources": sources,
              "feature_contract": feature_index["contract"], "source_reports": source_reports,
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
              "shared_initialization_sha256": shared_hash, "history": history,
              "candidates": candidates, "selection": selection, "last_checkpoint": artifact(last_path),
              "baseline_validation": baseline, "checks": checks,
              "pareto_candidate_ids": pareto_front([
                  make_candidate("single_frame", 0, baseline, baseline), *candidates]),
              "interpretation": "single_seed_development_screening; not_test_or_robustness_evidence"}
    write_json(output_dir / "selection.json", selection)
    return report


def summary_markdown(report):
    lines = ["# R4 compact causal refinement training", "",
             f"- Model: `{report['model_id']}`; plan: `{report['plan_version']}`.",
             f"- Parameters: `{report['parameter_count']}`; fresh initialization, no test data.",
             f"- Selection: `{json.dumps(report['selection'])}`.", "",
             "| Epoch | AUC | L2 | In/Out AP | Accurate stable | State error | Joint gaze response |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def value(number):
        return "null" if number is None else f"{number:.6f}"
    for item in report["history"]:
        v = item["validation"]
        metrics = [v["standard"][name] for name in ("auc", "l2", "inout_ap")]
        metrics.extend((v["temporal"]["stability"]["accurate_stable_ratio"],
                        v["state"]["state_error"]["error_rate"],
                        v["temporal"]["response"]["gaze_shift"]["response_rate"]))
        lines.append(f"| {item['epoch']} | " + " | ".join(map(value, metrics)) + " |")
    lines.extend(("", "All epochs and all joint/state metrics are retained in the JSON report.",
                  "This is a single-seed development run, not a test or statistical significance result."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    for name in ("feature-cache", "feature-report", "r1-report", "r1-review", "plan",
                 "output-dir", "report", "summary", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name, default in COMPACT_BUDGET.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_compact_training(args.feature_cache, args.feature_report, args.r1_report,
        args.r1_review, args.plan, args.output_dir, args.model_id, args.epochs,
        args.learning_rate, args.lambda_state, args.hidden_dim, args.clip_frames,
        args.batch_size, args.seed, args.weight_decay, args.gradient_clip_norm,
        args.temperature, args.device)
    paths = [Path(__file__), Path(__file__).with_name("train_temporal_joint.py"),
             Path(__file__).with_name("prepare_development.py"),
             Path(__file__).with_name("eval_temporal.py"),
             REPO_ROOT / "gazelle/utils.py", *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
             ("compact_protocol", "compact_validation", "temporal_model", "temporal_data",
              "history_model", "history_diagnostics", "feature_cache", "cache", "metrics", "state_metrics"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
                   "elapsed_seconds": time.perf_counter() - started,
                   "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                               "device_name": torch.cuda.get_device_name(torch.device(args.device))
                                   if torch.device(args.device).type == "cuda" else "CPU"}})
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report))
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R4 compact training: {report['model_id']}\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Plan: `{report['plan']['path']}`; version: `{PLAN_VERSION}`; SHA256: `{report['plan']['sha256']}`.\n"
            f"- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Last checkpoint: `{report['last_checkpoint']['path']}`; SHA256: `{report['last_checkpoint']['sha256']}`.\n")
        if report["selection"]["checkpoint"] is not None:
            selected = report["selection"]["checkpoint"]
            stream.write(f"- Selected epoch: `{report['selection']['epoch']}`; checkpoint: `{selected['path']}`; SHA256: `{selected['sha256']}`.\n")
        for item in report["history"]:
            v, cp = item["validation"], item["checkpoint"]
            stream.write(f"- Epoch {item['epoch']}: checkpoint `{cp['path']}` SHA256 `{cp['sha256']}`; "
                f"validation `{v['cache']['path']}` index SHA256 `{v['cache']['index_sha256']}`.\n")
            if v["diagnostics"] is not None:
                d = v["diagnostics"]
                stream.write(f"  - History diagnostics: `{d['path']}`; index SHA256: `{d['index_sha256']}`.\n")
    print(f"R4 compact training completed; model: {report['model_id']}; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
