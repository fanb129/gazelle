"""D4: train one current-frame-only joint refiner with the original per-frame loss."""

import argparse
from datetime import datetime, timezone
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
from train_compact_joint import shared_initialization_sha256
from stag.current_frame_model import (MODEL_ID, CONTROL_VERSION, CurrentFrameRefiner, parameter_usage)
from train_temporal_joint import train_epoch
from stag.cache import file_sha256, write_json
from stag.official_data import official_training_inputs
from stag.official_evaluation import evaluate_official_baseline, evaluate_official_model
from stag.official_protocol import (PROTOCOL_VERSION, TRAINING_BUDGET, make_candidate,
                                    select_checkpoint, validate_budget)


INPUT_NAMES = ("train_feature_cache", "train_feature_report", "test_feature_cache",
               "test_feature_report", "train_prediction_report", "test_prediction_report")


def run_current_frame_training(train_feature_cache, train_feature_report, test_feature_cache,
                          test_feature_report, train_prediction_report, test_prediction_report,
                          plan, output_dir, epochs=5,
                          learning_rate=0.0001, lambda_state=0.1, hidden_dim=128,
                          clip_frames=16, batch_size=8, seed=0, weight_decay=0.01,
                          gradient_clip_norm=1.0, temperature=0.1, device="cuda:0"):
    model_id = MODEL_ID
    config = {"model_id": model_id, "epochs": epochs, "learning_rate": learning_rate,
              "lambda_state": lambda_state, "hidden_dim": hidden_dim,
              "clip_frames": clip_frames, "batch_size": batch_size, "seed": seed,
              "weight_decay": weight_decay, "gradient_clip_norm": gradient_clip_norm,
              "temperature": temperature, "optimizer": "AdamW", "device": str(device),
              "precision": "float32; no_autocast; tf32_disabled",
              "history": "none; state_reset_every_frame; outputs_not_reused",
              "initialization": "fresh_new_module; shared_seed_initialization; no_R2_or_R4_resume",
              "selection": "test_auc_desc_l2_asc_inout_ap_desc_epoch_asc; no_precision_gate",
              "train_split": "train", "selection_split": "test",
              "test_used_for_checkpoint_selection": True, "has_validation_split": False,
              "early_stopping": False, "control_version": CONTROL_VERSION,
              "output_calibration": "none", "temporal_loss": "none",
              "objective": "heatmap_MSE + lambda_state * state_BCE"}
    validate_budget(config)
    if (PROTOCOL_VERSION not in Path(plan).read_text()
            or CONTROL_VERSION not in Path(plan).read_text()):
        raise ValueError("Plan does not contain the official train/test and current-frame control versions")
    inputs = official_training_inputs(train_feature_cache, train_feature_report,
        test_feature_cache, test_feature_report, train_prediction_report, test_prediction_report)
    train, test = inputs["train"], inputs["test"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = CurrentFrameRefiner(inputs["feature_contract"]["channels"], hidden_dim).to(device)
    usage = parameter_usage(model)
    config["feature_channels"] = model.feature_channels
    shared_hash = shared_initialization_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # Cache directories are identified by their completed indexes; source reports
    # retain the large arrays' exact paths and SHA256 values.
    source_reports = {name: artifact(path) for name, path in (
        ("train_feature_cache", Path(train_feature_cache) / "index.json"),
        ("train_feature_report", train_feature_report),
        ("test_feature_cache", Path(test_feature_cache) / "index.json"),
        ("test_feature_report", test_feature_report),
        ("train_prediction_report", train_prediction_report),
        ("test_prediction_report", test_prediction_report), ("plan", plan))}
    write_json(output_dir / "training_config.json", {
        "protocol_version": PROTOCOL_VERSION, "control_version": CONTROL_VERSION,
        "plan": source_reports["plan"], "parameter_usage": usage,
        "config": config, "sources": inputs["sources"],
        "feature_contract": inputs["feature_contract"],
        "shared_initialization_sha256": shared_hash})
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    baseline = evaluate_official_baseline(test["tracks"], test["arrays"], output_dir)
    history, candidates = [], []
    for epoch in range(1, epochs + 1):
        print(f"R4 current-frame {model_id} epoch {epoch}/{epochs}", flush=True)
        training = train_epoch(model, optimizer, train["tracks"], train["arrays"],
            train["feature_index"], train["features"], device, clip_frames, batch_size,
            seed + epoch - 1, lambda_state)
        evaluation = evaluate_official_model(model, test["tracks"], test["arrays"],
            test["feature_index"], test["features"], output_dir / f"test_epoch_{epoch:03d}",
            test["prediction_index"]["person_frames"], device, clip_frames, batch_size,
            {"sources": inputs["sources"], "config": config,
             "protocol_version": PROTOCOL_VERSION, "control_version": CONTROL_VERSION,
             "plan_sha256": source_reports["plan"]["sha256"], "epoch": epoch})
        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "model_id": model_id, "protocol_version": PROTOCOL_VERSION,
                    "control_version": CONTROL_VERSION,
                    "plan_sha256": source_reports["plan"]["sha256"],
                    "feature_contract": inputs["feature_contract"],
                    "sources": inputs["sources"]}, checkpoint_path)
        history.append({"epoch": epoch, "training": training, "test": evaluation,
                        "checkpoint": artifact(checkpoint_path)})
        candidates.append(make_candidate(model_id, epoch, evaluation))
        write_json(output_dir / "training_history.json", history)
        print(f"Epoch {epoch}: train_loss={training['total_loss']:.6f} "
              f"test={evaluation['standard']}", flush=True)
    last_path = output_dir / "last.pt"
    shutil.copyfile(history[-1]["checkpoint"]["path"], last_path)
    selected = select_checkpoint(candidates)
    selected_path = output_dir / "selected.pt"
    item = next(item for item in history if item["epoch"] == selected["epoch"])
    shutil.copyfile(item["checkpoint"]["path"], selected_path)
    selection = {"status": "selected", "epoch": selected["epoch"],
                 "candidate_id": selected["candidate_id"], "checkpoint": artifact(selected_path)}
    changed = {name: not torch.equal(initial[name], value.detach().cpu())
               for name, value in model.named_parameters()}
    checks = {
        "all_official_train_frames_used_each_epoch": all(item["training"]["valid_frames"] ==
            train["prediction_index"]["person_frames"] for item in history),
        "all_official_test_frames_evaluated_each_epoch": all(item["test"]["standard"]["person_frames"] ==
            test["prediction_index"]["person_frames"] for item in history),
        "all_planned_epochs_completed": [item["epoch"] for item in history] == list(range(1, epochs + 1)),
        "heatmap_head_updated": any(value for name, value in changed.items() if name.startswith("heatmap_residual")),
        "state_head_updated": any(value for name, value in changed.items() if name.startswith("inout_residual")),
        "only_new_module_parameters": not any(name.startswith("backbone") for name in changed),
        "frozen_features_read_only": all(not part["features"].flags.writeable for part in (train, test)),
        "prediction_arrays_read_only": all(not values.flags.writeable for part in (train, test)
                                             for values in part["arrays"].values()),
        "test_selection_explicitly_recorded": config["test_used_for_checkpoint_selection"],
        "no_validation_partition_used": not config["has_validation_split"],
        "all_test_epochs_have_no_history_diagnostics": all(
            item["test"]["diagnostics"] is None for item in history),
    }
    if not all(checks.values()):
        raise ValueError(f"Official train/test training verification failed: {checks}")
    coverage = {split: {"tracks": len(part["tracks"]),
                       "person_frames": part["prediction_index"]["person_frames"],
                       "unique_image_frames": len(part["feature_index"]["image_paths"])}
                for split, part in (("train", train), ("test", test))}
    report = {"task": "R4_official_training", "model_id": model_id, "passed": True,
              "stage_status": "awaiting_result_review", "protocol_version": PROTOCOL_VERSION,
              "control_version": CONTROL_VERSION, "parameter_usage": usage,
              "plan": source_reports["plan"], "config": config, "sources": inputs["sources"],
              "feature_contract": inputs["feature_contract"], "source_reports": source_reports,
              "parameter_count": sum(value.numel() for value in model.parameters()),
              "shared_initialization_sha256": shared_hash, "coverage": coverage,
              "history": history, "candidates": candidates, "selection": selection,
              "last_checkpoint": artifact(last_path), "baseline_test": baseline, "checks": checks,
              "test_used_for_checkpoint_selection": True,
              "independent_held_out_test_estimate": False,
              "interpretation": "single_seed_current_frame_control; official_test_used_for_epoch_selection; not_an_independent_held_out_test_estimate"}
    write_json(output_dir / "selection.json", selection)
    return report


def summary_markdown(report):
    lines = ["# R4-D4 current-frame-only joint refinement", "",
             f"- Model: `{report['model_id']}`; protocol: `{report['protocol_version']}`.",
             f"- Control: `{report['control_version']}`; fresh state at every frame.",
             "- Original per-frame MSE/BCE; no D3 temporal loss or D2 calibration.",
             "- Same nominal layers as ordinary GRU; structurally unused history weights are documented in parameter_usage.",
             "- Training: all official train tracks; no validation partition.",
             "- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.",
             "- Test is used for checkpoint selection; this is not an independent held-out test estimate.",
             f"- Selected: `{json.dumps(report['selection'])}`.", "",
             "| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def value(number):
        return "null" if number is None else f"{number:.6f}"
    for item in report["history"]:
        test = item["test"]
        metrics = [test["standard"][name] for name in ("auc", "l2", "inout_ap")]
        metrics.extend((test["temporal"]["stability"]["accurate_stable_ratio"],
                        test["state"]["state_error"]["error_rate"],
                        test["temporal"]["response"]["gaze_shift"]["response_rate"]))
        lines.append(f"| {item['epoch']} | " + " | ".join(map(value, metrics)) + " |")
    lines.extend(("", "All metrics in the selected result come from the same epoch.",
                  "All epoch checkpoints, complete test predictions and diagnostics remain on the server."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES, "plan", "output_dir", "report", "summary", "provenance"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name, default in TRAINING_BUDGET.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_current_frame_training(*(getattr(args, name) for name in INPUT_NAMES),
        args.plan, args.output_dir, args.epochs, args.learning_rate,
        args.lambda_state, args.hidden_dim, args.clip_frames, args.batch_size, args.seed,
        args.weight_decay, args.gradient_clip_norm, args.temperature, args.device)
    paths = [Path(__file__), *(Path(__file__).with_name(name + ".py") for name in
             ("train_compact_joint", "train_temporal_joint", "prepare_development", "eval_temporal")),
             REPO_ROOT / "gazelle/utils.py", *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
             ("current_frame_model", "official_protocol", "official_data", "official_evaluation", "compact_protocol",
              "compact_validation", "temporal_model", "temporal_data", "history_model",
              "history_diagnostics", "feature_cache", "cache", "metrics", "state_metrics"))]
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
        stream.write(f"\n## R4-D4 current-frame train/test: {report['model_id']}\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Protocol: `{PROTOCOL_VERSION}`; control: `{CONTROL_VERSION}`; plan SHA256: `{report['plan']['sha256']}`.\n"
            f"- Test used for epoch selection: `true`; independent held-out estimate: `false`.\n"
            f"- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Last checkpoint: `{report['last_checkpoint']['path']}`; SHA256: `{report['last_checkpoint']['sha256']}`.\n"
            f"- Selected epoch: `{report['selection']['epoch']}`; checkpoint: `{report['selection']['checkpoint']['path']}`; SHA256: `{report['selection']['checkpoint']['sha256']}`.\n"
            f"- Single-frame test details: `{report['baseline_test']['details']['path']}`; SHA256: `{report['baseline_test']['details']['sha256']}`.\n")
        for item in report["history"]:
            evaluation, checkpoint = item["test"], item["checkpoint"]
            stream.write(f"- Epoch {item['epoch']}: checkpoint `{checkpoint['path']}` SHA256 `{checkpoint['sha256']}`; "
                f"test `{evaluation['cache']['path']}` index SHA256 `{evaluation['cache']['index_sha256']}`.\n")
    print(f"R4 current-frame training completed; model: {report['model_id']}; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
