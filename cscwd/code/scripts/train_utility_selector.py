"""D6: frozen candidate banks, strong controls and one cost-supervised history gate."""

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
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from compare_compact_joint import artifact, paired_details
from compare_state_supervised import _snapshot, _deltas
from stag.cache import file_sha256, write_json
from stag.official_data import official_training_inputs
from stag.official_protocol import PROTOCOL_VERSION, make_candidate, select_checkpoint
from stag.selector_bank import build_selector_bank, load_selector_bank
from stag.selector_evaluation import evaluate_selector_choices
from stag.utility_selector import (VERSION, MODEL_ID, BANK_CANDIDATES, ANCHOR_INDEX,
                                  HISTORY_INDEX, UtilitySelector, cost_weighted_bce, hard_choices)


INPUT_NAMES = ("train_feature_cache", "train_feature_report", "test_feature_cache",
               "test_feature_report", "train_prediction_report", "test_prediction_report")
SELECTOR_BUDGET = {"epochs": 5, "learning_rate": .0001, "weight_decay": .01,
                   "hidden_dim": 64, "batch_size": 256, "seed": 0, "gradient_clip_norm": 1.}


def training_targets(bank):
    frames = [frame for track in bank["index"]["tracks"] for frame in track["frames"]]
    return (np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames],
                       dtype=np.float64),
            np.asarray([frame["inout"] == 1 for frame in frames], dtype=bool))


def train_selector_epoch(model, optimizer, bank, gt, inside, device, batch_size, seed, clip_norm):
    model.train()
    totals = {"person_frames": 0, "supervised_frames": 0, "prefer_history_frames": 0,
              "tie_frames": 0, "weighted_bce_sum": 0., "weight_sum": 0.}
    order = np.random.default_rng(seed).permutation(len(bank["inout"]))
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        observations = torch.tensor(np.array(bank["observations"][rows]), device=device)
        xy = torch.tensor(np.array(bank["candidate_xy"][rows]), device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = cost_weighted_bce(model(observations), xy[:, ANCHOR_INDEX], xy[:, HISTORY_INDEX],
            torch.tensor(gt[rows], dtype=torch.float64, device=device),
            torch.tensor(inside[rows], device=device),
            torch.tensor(np.array(bank["history_available"][rows]), device=device))
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        totals["person_frames"] += len(rows)
        for key in ("supervised_frames", "prefer_history_frames", "tie_frames"):
            totals[key] += int(losses[key].detach().cpu())
        for key in ("weighted_bce_sum", "weight_sum"):
            totals[key] += float(losses[key].detach().cpu())
    totals["loss"] = totals["weighted_bce_sum"] / max(totals["weight_sum"], 1e-12)
    return totals


def infer_selector(model, bank, device, batch_size):
    """Return logits and hard choices from one forward per frame, without GT."""
    model.eval()
    logits = np.empty(len(bank["inout"]), dtype=np.float32)
    choices = np.empty(len(logits), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(logits), batch_size):
            stop = min(start + batch_size, len(logits))
            values = model(torch.tensor(np.array(bank["observations"][start:stop]), device=device))
            selected = hard_choices(values,
                torch.tensor(np.array(bank["history_available"][start:stop]), device=device))
            logits[start:stop] = values.cpu().numpy()
            choices[start:stop] = selected.cpu().numpy()
    return logits, choices


def unchanged_state(reference, evaluation):
    return (evaluation["standard"]["inout_ap"] == reference["standard"]["inout_ap"]
            and evaluation["state"] == reference["state"]
            and evaluation["temporal"]["response"]["leave_frame"]
                == reference["temporal"]["response"]["leave_frame"])


def run_selector_training(train_feature_cache, train_feature_report, test_feature_cache,
                          test_feature_report, train_prediction_report, test_prediction_report,
                          plan, output_dir, epochs=5, learning_rate=.0001, weight_decay=.01,
                          hidden_dim=64, batch_size=256, seed=0, gradient_clip_norm=1., device="cuda:0"):
    config = dict(zip(SELECTOR_BUDGET, (epochs, learning_rate, weight_decay, hidden_dim,
                                      batch_size, seed, gradient_clip_norm)))
    if config != SELECTOR_BUDGET:
        raise ValueError(f"Utility selector budget mismatch: expected {SELECTOR_BUDGET}")
    if VERSION not in Path(plan).read_text():
        raise ValueError("Plan does not contain the utility selector version")
    inputs = official_training_inputs(train_feature_cache, train_feature_report, test_feature_cache,
                                     test_feature_report, train_prediction_report, test_prediction_report)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_record = artifact(plan)
    source_reports = {name: artifact(path) for name, path in zip(INPUT_NAMES,
        (Path(train_feature_cache) / "index.json", train_feature_report,
         Path(test_feature_cache) / "index.json", test_feature_report,
         train_prediction_report, test_prediction_report))}
    config.update({"version": VERSION, "model_id": MODEL_ID,
        "objective": "abs(L2_C-L2_H)_weighted_BCE; target=L2_H<L2_C; no_class_reweighting",
        "anchor": "current_grid", "history_candidate": "transport_previous",
        "inout": "raw_unchanged", "history_feedback": "raw_only; gate_outputs_not_reused",
        "precision": "float32; no_autocast; tf32_disabled", "device": str(device),
        "selection": "test_auc_desc_l2_asc_inout_ap_desc_epoch_asc",
        "test_used_for_checkpoint_selection": True, "has_validation_split": False})
    records, banks = {}, {}
    for split in ("train", "test"):
        print(f"Preparing frozen selector bank: {split}", flush=True)
        records[split] = build_selector_bank(inputs[split], output_dir / "banks" / split,
            inputs["sources"], {"split": split, "version": VERSION,
                                "plan_sha256": plan_record["sha256"]}, device=device)
        banks[split] = load_selector_bank(records[split])
    train, test = banks["train"], banks["test"]
    baseline_metadata = {"version": VERSION, "sources": inputs["sources"],
                         "plan_sha256": plan_record["sha256"], "evaluation_split": "test"}
    baselines = {}
    for column, name in enumerate(BANK_CANDIDATES):
        print(f"Evaluating fixed control: {name}", flush=True)
        baselines[name] = evaluate_selector_choices(test, np.full(len(test["inout"]), column),
            output_dir / "baselines" / name, {**baseline_metadata, "model_id": name, "epoch": 0})
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    channels = train["index"]["feature_channels"]
    model = UtilitySelector(channels, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    gt, inside = training_targets(train)
    anchor_error = np.linalg.norm(train["candidate_xy"][:, ANCHOR_INDEX].astype(np.float64) - gt, axis=1)
    history_error = np.linalg.norm(train["candidate_xy"][:, HISTORY_INDEX].astype(np.float64) - gt, axis=1)
    advantage = anchor_error - history_error
    supervised = inside & train["history_available"]
    expected_targets = {"supervised_frames": int(supervised.sum()),
        "prefer_history_frames": int((supervised & (advantage > 0)).sum()),
        "tie_frames": int((supervised & (advantage == 0)).sum()),
        "weight_sum": float((np.abs(advantage) * supervised).sum())}
    model_dir = output_dir / "selector"
    model_dir.mkdir()
    write_json(model_dir / "training_config.json", {"config": config, "plan": plan_record,
        "source_reports": source_reports, "sources": inputs["sources"], "banks": records})
    history, candidates = [], []
    for epoch in range(1, epochs + 1):
        print(f"Utility selector epoch {epoch}/{epochs}", flush=True)
        training = train_selector_epoch(model, optimizer, train, gt, inside, device,
                                       batch_size, seed + epoch - 1, gradient_clip_norm)
        logits, choices = infer_selector(model, test, device, batch_size)
        evaluation = evaluate_selector_choices(test, choices, model_dir / f"test_epoch_{epoch:03d}",
            {**baseline_metadata, "model_id": MODEL_ID, "epoch": epoch, "config": config}, logits=logits)
        checkpoint = model_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "model_id": MODEL_ID, "version": VERSION,
            "epoch": epoch, "config": config, "feature_channels": channels,
            "plan_sha256": plan_record["sha256"], "sources": inputs["sources"],
            "banks": records}, checkpoint)
        history.append({"epoch": epoch, "training": training, "test": evaluation,
                        "checkpoint": artifact(checkpoint)})
        candidates.append(make_candidate(MODEL_ID, epoch, evaluation))
        write_json(model_dir / "training_history.json", history)
        print(f"Epoch {epoch}: weighted_BCE={training['loss']:.6f}; "
              f"supervised={training['supervised_frames']}; test={evaluation['standard']}", flush=True)
    winner = select_checkpoint(candidates)
    selected_item = history[winner["epoch"] - 1]
    selected_path, last_path = model_dir / "selected.pt", model_dir / "last.pt"
    shutil.copyfile(selected_item["checkpoint"]["path"], selected_path)
    shutil.copyfile(history[-1]["checkpoint"]["path"], last_path)
    selection = {"epoch": winner["epoch"], "candidate_id": winner["candidate_id"],
                 "checkpoint": artifact(selected_path)}
    write_json(model_dir / "selection.json", selection)
    selected = _snapshot(MODEL_ID, winner["epoch"], selected_item["test"])
    baseline_rows = {name: _snapshot(name, 0, evaluation) for name, evaluation in baselines.items()}
    selected_details = json.loads(Path(selected_item["test"]["details"]["path"]).read_text())
    paired = {name: paired_details(name, MODEL_ID,
        json.loads(Path(evaluation["details"]["path"]).read_text()), selected_details)
        for name, evaluation in baselines.items()}
    expected_supervised = expected_targets["supervised_frames"]
    state_checks = {"fixed_state:" + name: unchanged_state(baselines["current"], evaluation)
                    for name, evaluation in baselines.items()}
    state_checks.update({f"fixed_state:epoch_{item['epoch']}": unchanged_state(
        baselines["current"], item["test"]) for item in history})
    checks = {**state_checks,
        "complete_train_each_epoch": all(h["training"]["person_frames"] == len(train["inout"]) for h in history),
        "supervision_current_in_frame_and_history_only": all(
            h["training"]["supervised_frames"] == expected_supervised for h in history),
        "fixed_cost_targets_and_weights_reproduced_each_epoch": all(
            all(h["training"][key] == expected_targets[key] for key in
                ("prefer_history_frames", "tie_frames"))
            and np.isclose(h["training"]["weight_sum"], expected_targets["weight_sum"], rtol=1e-12, atol=1e-12)
            for h in history),
        "complete_test_each_epoch": all(h["test"]["standard"]["person_frames"] == len(test["inout"]) for h in history),
        "all_five_epochs_retained": [h["epoch"] for h in history] == [1, 2, 3, 4, 5],
        "MLP_parameters_updated": any(not torch.equal(initial[n], v.detach().cpu())
                                      for n, v in model.named_parameters()),
        "selected_checkpoint_matches_chosen_epoch": file_sha256(selected_path) == selected_item["checkpoint"]["sha256"],
        "last_checkpoint_matches_last_epoch": file_sha256(last_path) == history[-1]["checkpoint"]["sha256"],
        "bank_arrays_read_only": all(not bank[key].flags.writeable for bank in banks.values()
            for key in ("observations", "heatmaps", "candidate_xy", "inout", "history_available")),
        "no_history_forces_current_anchor": all(np.all(np.load(h["test"]["selection"]["choices"]["path"],
            allow_pickle=False)[~test["history_available"]] == ANCHOR_INDEX) for h in history),
    }
    if not all(checks.values()):
        raise ValueError(f"Utility selector verification failed: {checks}")
    return {"task": "R4_utility_selector_training", "version": VERSION, "model_id": MODEL_ID,
        "protocol_version": PROTOCOL_VERSION, "passed": True, "stage_status": "awaiting_result_review",
        "config": config, "plan": plan_record, "sources": inputs["sources"],
        "source_reports": source_reports, "feature_contract": inputs["feature_contract"], "banks": records,
        "parameter_count": sum(v.numel() for v in model.parameters()),
        "coverage": {split: record["coverage"] for split, record in records.items()},
        "expected_train_supervised_frames": expected_supervised,
        "expected_train_cost_targets": expected_targets, "baselines": baselines,
        "history": history, "candidates": candidates, "selection": selection,
        "last_checkpoint": artifact(last_path), "selected": selected,
        "delta_vs_baselines": {name: _deltas(row, selected) for name, row in baseline_rows.items()},
        "paired_response": paired, "checks": checks,
        "test_used_for_checkpoint_selection": True, "independent_held_out_test_estimate": False,
        "interpretation": "single_seed_cost_supervised_hard_history_selection; compare_against_grid_and_grid_EMA; AP_and_state_fixed"}


def summary_markdown(report):
    lines = ["# R4-D6 cost-supervised history selector", "",
        "- Hard choice: current_grid or one-hop visual history; original InOut unchanged.",
        "- Complete official train, five fresh epochs; complete test every epoch.",
        "- Test used for epoch selection: AUC descending, L2 ascending, AP descending, earlier epoch.",
        f"- Selected epoch: {report['selection']['epoch']}; parameters: {report['parameter_count']}.",
        "- This is not an independent held-out estimate. No GT/oracle result feeds candidate memory.", "",
        "| Model | Epoch | AUC | L2 | AP | Accurate stable | Gaze responses | History choices |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [(name, 0, value) for name, value in report["baselines"].items()]
    rows += [(MODEL_ID, h["epoch"], h["test"]) for h in report["history"]]
    for name, epoch, value in rows:
        standard, temporal = value["standard"], value["temporal"]
        lines.append(f"| {name} | {epoch} | {standard['auc']:.9f} | {standard['l2']:.9f} | "
            f"{standard['inout_ap']:.9f} | {temporal['stability']['accurate_stable_ratio']} | "
            f"{temporal['response']['gaze_shift']['responded_events']} | "
            f"{value['selection']['choice_counts'].get('transport_previous', 0)} |")
    lines.extend(("", "All selected metrics come from one epoch. All five checkpoints and full outputs remain on server.",
        "JSON preserves cost-loss totals, complete fixed controls, all-epoch metrics, deltas and paired events.",
        "Compare stability and response jointly against current_grid and grid_ema_075 before attributing a history benefit."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES, "plan", "output_dir", "report", "summary", "selection_manifest", "provenance"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name, default in SELECTOR_BUDGET.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_selector_training(*(getattr(args, name) for name in INPUT_NAMES), args.plan,
        args.output_dir, *(getattr(args, name) for name in SELECTOR_BUDGET), device=args.device)
    paths = [Path(__file__), *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
        ("utility_selector", "selector_bank", "selector_evaluation", "history_candidates", "temporal_model",
         "cache", "feature_cache", "official_data", "metrics", "state_metrics", "official_protocol", "compact_protocol")),
         *(Path(__file__).with_name(name + ".py") for name in ("compare_compact_joint", "compare_state_supervised", "eval_temporal"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "code_sha256": {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths},
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device_name":
            torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == "cuda" else "CPU"}})
    write_json(args.report, report)
    args.summary.write_text(summary_markdown(report))
    write_json(args.selection_manifest, {"version": VERSION, "protocol_version": PROTOCOL_VERSION,
        "selected": report["selected"], "selection": report["selection"], "report": artifact(args.report),
        "test_used_for_checkpoint_selection": True, "independent_held_out_test_estimate": False})
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R4-D6 cost-supervised utility selector\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Plan SHA256: `{report['plan']['sha256']}`; test used for checkpoint selection.\n"
            f"- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Summary: `{args.summary}`; SHA256: `{file_sha256(args.summary)}`.\n"
            f"- Selection: `{args.selection_manifest}`; SHA256: `{file_sha256(args.selection_manifest)}`.\n"
            f"- Selected epoch: `{report['selection']['epoch']}`; checkpoint: `{report['selection']['checkpoint']}`.\n"
            f"- Last checkpoint: `{report['last_checkpoint']}`.\n")
        for split, record in report["banks"].items():
            stream.write(f"- {split} bank: `{record['path']}`; index SHA256: `{record['index_sha256']}`.\n")
        for item in report["history"]:
            stream.write(f"- Epoch {item['epoch']}: checkpoint `{item['checkpoint']}`; test cache `{item['test']['cache']}`.\n")
    print(f"R4 utility selector completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
