"""Compare the fixed D3 state-supervision runs with the original official round."""

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

from compare_compact_joint import artifact, verify_artifact
from compare_official_joint import (compare_runs, selection_manifest as official_selection_manifest,
                                    summary_markdown as official_summary_markdown)
from stag.cache import file_sha256, write_json
from stag.official_protocol import PROTOCOL_VERSION, TRAINING_BUDGET, make_candidate, select_checkpoint


SUPERVISION_VERSION = "r4_state_supervision_v1"
SUPERVISION_SETTINGS = {"lambda_time": 0.1, "response_frames": 3}
MODEL_IDS = ("ordinary_gru", "history_validity")


def _evaluation_summary(evaluation):
    return {key: evaluation[key] for key in ("standard", "temporal", "state")}


def _selected_evaluation(run, comparison, model_id):
    """Recompute selection from small reports; do not reopen old large artifacts."""
    candidates = [make_candidate(model_id, item["epoch"], item["test"]) for item in run["history"]]
    winner = select_checkpoint(candidates)
    selection = run["selection"]
    if (candidates != run["candidates"] or winner != comparison["selected"][model_id]
            or selection != comparison["per_model_selection"][model_id]
            or selection["epoch"] != winner["epoch"]
            or selection["candidate_id"] != winner["candidate_id"]):
        raise ValueError("Selected epoch differs from its comparison or recomputed test ranking")
    item = next(item for item in run["history"] if item["epoch"] == winner["epoch"])
    if item["checkpoint"]["sha256"] != selection["checkpoint"]["sha256"]:
        raise ValueError("Selected checkpoint identity differs from its epoch record")
    return _evaluation_summary(item["test"]), winner["epoch"]


def _snapshot(model_id, epoch, evaluation):
    temporal, state = evaluation["temporal"], evaluation["state"]
    counts = {
        "accurate_stable_windows": temporal["stability"]["accurate_stable_frames"],
        "eligible_stability_windows": temporal["stability"]["eligible_windows"],
        "state_errors": state["state_error"]["errors"],
        "in_frame_state_errors": state["state_error"]["by_gt_state"]["in_frame"]["errors"],
        "out_of_frame_state_errors": state["state_error"]["by_gt_state"]["out_of_frame"]["errors"],
        "stable_state_flips": state["stable_state_flips"]["flips"],
        **{"joint_" + kind + "_responded": temporal["response"][kind]["responded_events"]
           for kind in ("enter_frame", "leave_frame", "gaze_shift")},
        **{"state_" + kind + "_responded": state["response"][kind]["responded_events"]
           for kind in ("enter_frame", "leave_frame")},
    }
    return {"model_id": model_id, "epoch": epoch, "summary": evaluation,
            "metrics": make_candidate(model_id, epoch, evaluation)["metrics"], "counts": counts}


def _deltas(reference, candidate):
    return {group: {key: value - reference[group][key]
                   if value is not None and reference[group][key] is not None else None
                   for key, value in candidate[group].items()}
            for group in ("metrics", "counts")}


def compare_supervised_runs(run_reports, plan, previous_comparison):
    """Keep all D3 epochs and add like-for-like selected-output comparisons."""
    if SUPERVISION_VERSION not in Path(plan).read_text():
        raise ValueError("Comparison requires the D3 state-supervision plan")
    runs = {}
    for path in run_reports:
        run = json.loads(Path(path).read_text())
        runs[run["model_id"]] = run
    for run in runs.values():
        config = run["config"]
        if (run.get("supervision_version") != SUPERVISION_VERSION
                or config.get("supervision_version") != SUPERVISION_VERSION
                or any(config.get(key) != value for key, value in SUPERVISION_SETTINGS.items())):
            raise ValueError("D3 runs must share the fixed state-supervision version and settings")
    report = compare_runs(run_reports, plan)
    previous = json.loads(Path(previous_comparison).read_text())
    if (previous["task"] != "R4_official_comparison" or not previous["passed"]
            or not all(previous["checks"].values())
            or previous["protocol_version"] != PROTOCOL_VERSION
            or previous.get("supervision_version") is not None):
        raise ValueError("Previous comparison must be the completed original official round")
    if (any(previous["shared_contract"][key] != report["shared_contract"][key]
            for key in ("sources", "feature_contract", "shared_initialization_sha256"))
            or previous["baseline_test"] != report["baseline_test"]
            or any(previous["shared_contract"]["config"][key] != report["shared_contract"]["config"][key]
                   for key in TRAINING_BUDGET)):
        raise ValueError("Previous and D3 rounds differ in sources, frozen features, initialization, budget or baseline")
    old_selected, new_selected = {}, {}
    for model_id in MODEL_IDS:
        old_run = json.loads(verify_artifact(previous["source_reports"][model_id]).read_text())
        if (old_run["task"] != "R4_official_training" or not old_run["passed"]
                or not all(old_run["checks"].values()) or old_run["model_id"] != model_id
                or old_run["protocol_version"] != PROTOCOL_VERSION
                or old_run["sources"] != report["shared_contract"]["sources"]
                or old_run["feature_contract"] != report["shared_contract"]["feature_contract"]
                or old_run["parameter_count"] != runs[model_id]["parameter_count"]
                or _evaluation_summary(old_run["baseline_test"]) != previous["baseline_test"]):
            raise ValueError("Previous model report differs from the declared original sources or architecture")
        evaluation, epoch = _selected_evaluation(old_run, previous, model_id)
        old_selected[model_id] = _snapshot(model_id, epoch, evaluation)
        evaluation, epoch = _selected_evaluation(runs[model_id], report, model_id)
        new_selected[model_id] = _snapshot(model_id, epoch, evaluation)
    baseline = _snapshot("single_frame", 0, report["baseline_test"])
    report.update({
        "supervision_version": SUPERVISION_VERSION,
        "supervision_settings": SUPERVISION_SETTINGS,
        "previous_round": {
            "comparison_report": artifact(previous_comparison),
            "source_reports": previous["source_reports"],
            "comparison_scope": "original_uncalibrated_outputs; no_D2_state_calibration",
            "single_frame": baseline, "previous_selected": old_selected, "current_selected": new_selected,
            "delta_vs_previous_same_model": {name: _deltas(old_selected[name], new_selected[name]) for name in MODEL_IDS},
            "delta_vs_single_frame": {name: _deltas(baseline, new_selected[name]) for name in MODEL_IDS},
            "delta_convention": "D3_candidate_minus_reference; rates_are_absolute_ratio_differences",
            "previous_artifact_verification": "small_run_reports_and_selection_identities_only; previous_NPY_PT_and_event_details_not_reopened",
        },
    })
    report["checks"].update({"fixed_state_supervision_settings": True,
        "previous_round_sources_budget_initializer_and_baseline_match": True,
        "previous_selected_small_reports_and_selection_identities_verified": True})
    return report


def selection_manifest(report):
    manifest = official_selection_manifest(report)
    manifest.update({"supervision_version": report["supervision_version"],
                     "supervision_settings": report["supervision_settings"],
                     "previous_comparison": report["previous_round"]["comparison_report"],
                     "output_state_calibration": "none"})
    return manifest


def summary_markdown(report):
    lines = [official_summary_markdown(report).replace(
        "# R4 official train/test comparison", "# R4-D3 state-supervision comparison", 1).rstrip(),
        "", f"- State supervision: `{report['supervision_version']}`; fixed settings: `{json.dumps(report['supervision_settings'])}`.",
        "- The following rows use original output scores; no D2 calibration is applied.",
        "- Each model's complete row comes from its own test-selected epoch.", "",
        "## Selected outputs before and after state supervision", "",
        "| Round | Model | Epoch | AUC | L2 | AP | Accurate stable | State errors | False flips | Joint enter | Joint leave | Joint gaze |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    previous = report["previous_round"]
    rows = [("single_frame", previous["single_frame"])]
    rows.extend((stage, row) for stage, key in (("previous", "previous_selected"), ("D3", "current_selected"))
                for row in previous[key].values())
    for stage, row in rows:
        metrics, counts = row["metrics"], row["counts"]
        values = [metrics[key] for key in ("auc", "l2", "inout_ap", "accurate_stable_ratio")]
        formatted = ["null" if value is None else f"{value:.6f}" for value in values]
        formatted.extend(str(counts[key]) for key in ("state_errors", "stable_state_flips", "joint_enter_frame_responded",
                                                     "joint_leave_frame_responded", "joint_gaze_shift_responded"))
        lines.append(f"| {stage} | {row['model_id']} | {row['epoch']} | " + " | ".join(formatted) + " |")
    lines.extend(("", "Rate/count deltas against the same model's previous selected epoch and the single-frame baseline are in JSON.",
                  "Previous large caches/checkpoints are not reopened; prior rows are traced to their hashed small reports."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, nargs="+", required=True)
    for name in ("plan", "previous-comparison", "report", "summary", "selection-manifest", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare_supervised_runs(args.run_report, args.plan, args.previous_comparison)
    paths = [Path(__file__), *(Path(__file__).with_name(name + ".py") for name in
             ("compare_official_joint", "compare_compact_joint", "eval_temporal")),
             *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
             ("official_protocol", "compact_protocol", "cache"))]
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
        stream.write(f"\n## R4-D3 state-supervision comparison\n\n"
            f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n"
            f"- Previous comparison: `{args.previous_comparison}`; SHA256: `{file_sha256(args.previous_comparison)}`.\n")
        for name, path in (("Report", args.report), ("Summary", args.summary), ("Selection manifest", args.selection_manifest)):
            stream.write(f"- {name}: `{path}`; SHA256: `{file_sha256(path)}`.\n")
        stream.write("- Full official test evaluated each epoch and used for selection; original outputs, no D2 calibration.\n")
    print(f"R4 state-supervised comparison completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
