"""Compare the D4 current-frame control with fixed official temporal references."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from compare_compact_joint import artifact, paired_details, verify_artifact
from compare_state_supervised import _deltas, _evaluation_summary, _selected_evaluation, _snapshot
from eval_temporal import verify_completed_cache
from stag.cache import file_sha256, write_json
from stag.official_protocol import (PROTOCOL_VERSION, SELECTION_RULE, TRAINING_BUDGET,
                                    make_candidate, select_checkpoint, validate_budget)


MODEL_ID = "current_frame"
CONTROL_VERSION = "r4_current_frame_v1"
HISTORY_POLICY = "none; state_reset_every_frame; outputs_not_reused"


def _validate_control_config(run, plan):
    config = run["config"]
    if (run["task"] != "R4_official_training" or not run["passed"]
            or not all(run["checks"].values()) or run["model_id"] != MODEL_ID
            or run["protocol_version"] != PROTOCOL_VERSION
            or run.get("control_version") != CONTROL_VERSION
            or config.get("control_version") != CONTROL_VERSION
            or config["model_id"] != MODEL_ID or config["history"] != HISTORY_POLICY
            or config.get("output_calibration") != "none"
            or config.get("temporal_loss") != "none"):
        raise ValueError("Comparison requires the completed fixed D4 current-frame control")
    validate_budget(config)
    if (config["train_split"] != "train" or config["selection_split"] != "test"
            or config["has_validation_split"] or not config["test_used_for_checkpoint_selection"]):
        raise ValueError("Current-frame control must retain official train/test epoch selection")
    text = Path(plan).read_text()
    if PROTOCOL_VERSION not in text or CONTROL_VERSION not in text:
        raise ValueError("Comparison requires the official D4 control plan")


def verify_current_test(run, item, test_counts):
    """Verify the selected stateless output without fabricating a GRU identity."""
    evaluation = item["test"]
    cache = evaluation["cache"]
    verify_completed_cache(cache["path"], {"passed": run["passed"], "cache": cache})
    index = json.loads((Path(cache["path"]) / "index.json").read_text())
    metadata = index["metadata"]
    if (not index["passed"] or index["person_frames"] != test_counts["person_frames"]
            or index["track_count"] != test_counts["tracks"]
            or metadata["sources"] != run["sources"] or metadata["config"] != run["config"]
            or metadata["protocol_version"] != PROTOCOL_VERSION
            or metadata.get("control_version") != CONTROL_VERSION
            or metadata["plan_sha256"] != run["plan"]["sha256"]
            or metadata["epoch"] != item["epoch"] or metadata["evaluation_split"] != "test"
            or any(not track["track_id"].startswith("test/") for track in index["tracks"])
            or evaluation["diagnostics"] is not None):
        raise ValueError("Selected current-frame test cache metadata or coverage differs from its run")
    return json.loads(verify_artifact(evaluation["details"]).read_text())


def _verify_current_run(run, plan):
    plan_record = artifact(plan)
    if run["plan"]["sha256"] != plan_record["sha256"]:
        raise ValueError("Current-frame training plan SHA256 differs from the supplied plan")
    for record in run["source_reports"].values():
        verify_artifact(record)
    if run["source_reports"]["plan"]["sha256"] != plan_record["sha256"]:
        raise ValueError("Current-frame source plan identity differs")
    for name, digest in run["code_sha256"].items():
        if file_sha256(REPO_ROOT / name) != digest:
            raise ValueError(f"Current-frame training source code SHA256 mismatch: {name}")
    for name in ("train_prediction_report", "test_prediction_report", "train_feature_report", "test_feature_report"):
        if run["sources"][name + "_sha256"] != run["source_reports"][name]["sha256"]:
            raise ValueError("Current-frame training sources differ from their supplied reports")
    counts = {split: json.loads(verify_artifact(run["source_reports"][split + "_prediction_report"]).read_text())["cache"]
              for split in ("train", "test")}
    if any(run["coverage"][split] != {key: counts[split][key]
            for key in ("tracks", "person_frames", "unique_image_frames")} for split in counts):
        raise ValueError("Current-frame reported coverage differs from source split counts")
    history = run["history"]
    if [item["epoch"] for item in history] != list(range(1, TRAINING_BUDGET["epochs"] + 1)):
        raise ValueError("Current-frame run does not contain every planned epoch")
    candidates = [make_candidate(MODEL_ID, item["epoch"], item["test"]) for item in history]
    if candidates != run["candidates"]:
        raise ValueError("Current-frame candidates differ from recomputed test metrics")
    for item in history:
        verify_artifact(item["checkpoint"])
        if (item["training"]["valid_frames"] != counts["train"]["person_frames"]
                or item["test"]["standard"]["person_frames"] != counts["test"]["person_frames"]
                or item["test"]["diagnostics"] is not None):
            raise ValueError("Current-frame epoch coverage or stateless diagnostics differ")
    verify_artifact(run["last_checkpoint"])
    if run["last_checkpoint"]["sha256"] != history[-1]["checkpoint"]["sha256"]:
        raise ValueError("Current-frame last checkpoint differs from the final epoch")
    winner = select_checkpoint(candidates)
    selection = run["selection"]
    item = history[winner["epoch"] - 1]
    if (selection["status"] != "selected" or selection["epoch"] != winner["epoch"]
            or selection["candidate_id"] != winner["candidate_id"]
            or selection["checkpoint"]["sha256"] != item["checkpoint"]["sha256"]):
        raise ValueError("Current-frame selected checkpoint differs from AUC/L2/AP ranking")
    verify_artifact(selection["checkpoint"])
    baseline_details = json.loads(verify_artifact(run["baseline_test"]["details"]).read_text())
    selected_details = verify_current_test(run, item, counts["test"])
    return candidates, winner, baseline_details, selected_details


def compare_current_frame(run_report, plan, reference_comparison):
    """Keep the complete control curve and matched-epoch temporal references."""
    run = json.loads(Path(run_report).read_text())
    _validate_control_config(run, plan)
    candidates, winner, baseline_details, selected_details = _verify_current_run(run, plan)
    baseline = _evaluation_summary(run["baseline_test"])
    reference = json.loads(Path(reference_comparison).read_text())
    if (reference["task"] != "R4_official_comparison" or not reference["passed"]
            or not all(reference["checks"].values()) or reference["protocol_version"] != PROTOCOL_VERSION
            or reference.get("supervision_version") is not None
            or reference["baseline_test"] != baseline
            or any(reference["shared_contract"][key] != run[key]
                   for key in ("sources", "feature_contract", "shared_initialization_sha256"))
            or any(reference["shared_contract"]["config"][key] != run["config"][key]
                   for key in TRAINING_BUDGET)):
        raise ValueError("Reference and D4 control differ in sources, initialization, budget or baseline")
    selected_item = run["history"][winner["epoch"] - 1]
    current = _snapshot(MODEL_ID, winner["epoch"], _evaluation_summary(selected_item["test"]))
    baseline_snapshot = _snapshot("single_frame", 0, baseline)
    old_selected, same_epoch, parameter_counts = {}, {}, {MODEL_ID: run["parameter_count"]}
    for model_id in ("ordinary_gru", "history_validity"):
        old = json.loads(verify_artifact(reference["source_reports"][model_id]).read_text())
        if (old["task"] != "R4_official_training" or not old["passed"]
                or not all(old["checks"].values()) or old["model_id"] != model_id
                or old["protocol_version"] != PROTOCOL_VERSION
                or old["sources"] != run["sources"] or old["feature_contract"] != run["feature_contract"]
                or old["shared_initialization_sha256"] != run["shared_initialization_sha256"]
                or _evaluation_summary(old["baseline_test"]) != baseline
                or any(old["config"][key] != run["config"][key] for key in TRAINING_BUDGET)):
            raise ValueError("Temporal reference report differs from the declared sources or budget")
        if model_id == "ordinary_gru" and old["parameter_count"] != run["parameter_count"]:
            raise ValueError("D4 stored parameter count differs from the ordinary GRU structure")
        evaluation, epoch = _selected_evaluation(old, reference, model_id)
        old_selected[model_id] = _snapshot(model_id, epoch, evaluation)
        if [item["epoch"] for item in old["history"]] != [item["epoch"] for item in run["history"]]:
            raise ValueError("Matched-epoch reference history differs from the control budget")
        same_epoch[model_id] = []
        for previous, control in zip(old["history"], run["history"]):
            original = _snapshot(model_id, previous["epoch"], _evaluation_summary(previous["test"]))
            candidate = _snapshot(MODEL_ID, control["epoch"], _evaluation_summary(control["test"]))
            same_epoch[model_id].append({"epoch": control["epoch"], "reference": original,
                "current_frame": candidate, "delta_current_frame_minus_reference": _deltas(original, candidate)})
        parameter_counts[model_id] = old["parameter_count"]
    rows = [make_candidate("single_frame", 0, baseline), *candidates]
    return {"task": "R4_current_frame_comparison", "passed": True, "stage_status": "awaiting_result_review",
        "protocol_version": PROTOCOL_VERSION, "control_version": CONTROL_VERSION,
        "plan": artifact(plan), "source_reports": {"current_frame": artifact(run_report),
            "reference_comparison": artifact(reference_comparison)},
        "sources": run["sources"], "feature_contract": run["feature_contract"],
        "coverage": run["coverage"], "candidate_rows": rows, "selected": winner,
        "selection": run["selection"], "baseline_test": baseline,
        "current_selected": current, "parameter_counts": parameter_counts,
        "parameter_usage": run.get("parameter_usage"),
        "parameter_interpretation": "same_stored_GRU_structure; zero_history_changes_effective_parameter_usage; not_equal_effective_capacity",
        "paired_response": {"current_frame_vs_single_frame": paired_details("single_frame", MODEL_ID,
            baseline_details, selected_details)},
        "reference_round": {"comparison_report": artifact(reference_comparison),
            "source_reports": reference["source_reports"], "selected": old_selected,
            "delta_current_selected_vs_reference_selected": {
                name: _deltas(previous, current) for name, previous in old_selected.items()},
            "same_epoch": same_epoch,
            "reference_artifact_verification": "hashed_small_reports_and_selection_identities; previous_NPY_PT_and_event_details_not_reopened"},
        "delta_vs_single_frame": _deltas(baseline_snapshot, current),
        "selection_rule": SELECTION_RULE, "selection_split": "test",
        "test_selected": True, "test_used_for_checkpoint_selection": True,
        "test_labels_used_for_gradient": False, "independent_test_estimate": False,
        "output_state_calibration": "none", "temporal_supervision": "none",
        "checks": {"fixed_current_frame_control_verified": True,
            "all_epoch_candidates_selection_and_artifacts_verified": True,
            "selected_current_frame_test_cache_and_event_pair_verified": True,
            "reference_sources_budget_initializer_and_stored_structure_match": True,
            "reference_selected_and_all_matched_epoch_rows_retained": True},
        "interpretation": "single_seed_current_frame_attribution_control; selected_and_matched_epoch_comparisons; not_history_utility_upper_bound"}


def selection_manifest(report):
    keys = ("protocol_version", "control_version", "plan", "source_reports", "sources", "coverage",
            "selected", "selection", "selection_rule", "selection_split", "test_selected",
            "test_used_for_checkpoint_selection", "test_labels_used_for_gradient", "independent_test_estimate")
    return {"task": "R4_current_frame_test_selected_checkpoint_manifest",
            "status": "ready_for_reporting_test_selected_protocol", **{key: report[key] for key in keys}}


def summary_markdown(report):
    lines = ["# R4-D4 current-frame attribution control", "",
        f"- Control: `{report['control_version']}`; original per-frame loss; no D2 calibration or D3 temporal loss.",
        "- History is cleared at every frame; only current observations enter the model.",
        "- Full official test is evaluated every epoch and used for complete-epoch selection.",
        "- Selection: AUC descending, L2 ascending, AP descending, earlier epoch; not an independent held-out estimate.", "",
        "| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["candidate_rows"]:
        metrics = row["metrics"]
        values = [metrics[key] for key in ("auc", "l2", "inout_ap", "accurate_stable_ratio",
                                           "state_error_rate", "joint_gaze_shift_response_rate")]
        formatted = ["null" if value is None else f"{value:.6f}" for value in values]
        lines.append(f"| {row['model_id']} | {row['epoch']} | " + " | ".join(formatted)
                     + f" | {row['candidate_id'] == report['selected']['candidate_id']} |")
    lines.extend(("", "## Selected outputs", "",
        "| Model | Epoch | AUC | L2 | AP | Accurate stable windows | State errors | False flips | Joint gaze responses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in [*report["reference_round"]["selected"].values(), report["current_selected"]]:
        metrics, counts = row["metrics"], row["counts"]
        values = [metrics[key] for key in ("auc", "l2", "inout_ap")]
        columns = ["null" if value is None else f"{value:.6f}" for value in values]
        columns.extend(str(counts[key]) for key in ("accurate_stable_windows", "state_errors", "stable_state_flips",
                                                   "joint_gaze_shift_responded"))
        lines.append(f"| {row['model_id']} | {row['epoch']} | " + " | ".join(columns) + " |")
    lines.extend(("", "All five matched-epoch comparisons and complete metric/count deltas are retained in JSON.",
        "Stored parameter count matches ordinary GRU, while effective parameter usage differs with zero history.",
        "Prior temporal rows use hashed original reports; old large artifacts are not reopened."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-report", "plan", "reference-comparison", "report", "summary", "selection-manifest", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare_current_frame(args.run_report, args.plan, args.reference_comparison)
    paths = [Path(__file__), *(Path(__file__).with_name(name + ".py") for name in
        ("compare_compact_joint", "compare_state_supervised", "eval_temporal")),
        *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in ("cache", "official_protocol", "compact_protocol"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
        "elapsed_seconds": time.perf_counter() - started})
    write_json(args.report, report)
    manifest = selection_manifest(report)
    manifest["comparison_report"] = artifact(args.report)
    manifest["comparison_code_sha256"] = report["code_sha256"]
    write_json(args.selection_manifest, manifest)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report))
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R4-D4 current-frame comparison\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Reference comparison: `{args.reference_comparison}`; SHA256: `{file_sha256(args.reference_comparison)}`.\n")
        for name, path in (("Report", args.report), ("Summary", args.summary), ("Selection manifest", args.selection_manifest)):
            stream.write(f"- {name}: `{path}`; SHA256: `{file_sha256(path)}`.\n")
    print(f"R4 current-frame comparison completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
