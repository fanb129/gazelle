"""R4: compare completed compact runs and record a development selection snapshot."""

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

from eval_temporal import paired_response_summary, verify_completed_cache
from stag.cache import file_sha256, write_json
from stag.compact_protocol import (COMPACT_BUDGET, PLAN_VERSION, make_candidate,
                                   pareto_front, select_checkpoint, validate_budget)


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def verify_artifact(record):
    path = Path(record["path"])
    if (file_sha256(path) != record["sha256"]
            or path.stat().st_size != record["size_bytes"]):
        raise ValueError(f"Artifact SHA256 or size mismatch: {path}")
    return path


def pair_events(reference, candidate, state_only=False):
    """Align source identities, independent of each cache's global offsets."""
    def key(event):
        return (event["track_id"], event["type"],
                event["event_frame_index"], event["window_frames"])
    first, second = sorted(reference, key=key), sorted(candidate, key=key)
    if [key(event) for event in first] != [key(event) for event in second]:
        raise ValueError("Response event identities or windows differ")
    result = paired_response_summary(first, second)
    for item in result.values():
        item["reference_only_responded_events"] = item.pop("single_frame_only_responded_events")
        item["candidate_only_responded_events"] = item.pop("stag_only_responded_events")
    if state_only:
        del result["gaze_shift"]
    return result


def paired_details(reference_id, candidate_id, reference, candidate):
    return {"reference_model": reference_id, "candidate_model": candidate_id,
            "joint": pair_events(reference["joint"]["events"], candidate["joint"]["events"]),
            "state": pair_events(reference["state"]["events"], candidate["state"]["events"], True)}


def verify_selected_validation(run, item, validation_counts):
    validation = item["validation"]
    cache = validation["cache"]
    verify_completed_cache(cache["path"], {"passed": run["passed"], "cache": cache})
    index = json.loads((Path(cache["path"]) / "index.json").read_text())
    expected_metadata = {"sources": run["sources"], "config": run["config"],
                         "plan_version": PLAN_VERSION, "plan_sha256": run["plan"]["sha256"],
                         "epoch": item["epoch"]}
    if (index["metadata"] != expected_metadata or not index["passed"]
            or index["person_frames"] != validation_counts["person_frames"]
            or index["track_count"] != validation_counts["tracks"]):
        raise ValueError("Selected validation cache metadata or coverage differs from its run")
    diagnostics = validation["diagnostics"]
    if diagnostics is not None:
        directory = Path(diagnostics["path"])
        if file_sha256(directory / "index.json") != diagnostics["index_sha256"]:
            raise ValueError("Selected history diagnostic index SHA256 mismatch")
        diagnostic_index = json.loads((directory / "index.json").read_text())
        if (not diagnostics["passed"] or not diagnostic_index["passed"]
                or diagnostic_index["metadata"] != expected_metadata
                or diagnostic_index["person_frames"] != index["person_frames"]
                or diagnostic_index["files"] != diagnostics["files"]):
            raise ValueError("Selected history diagnostics differ from validation metadata")
        for record in diagnostics["files"].values():
            verify_artifact({**record, "path": str(directory / record["path"])})
    elif run["model_id"] != "ordinary_gru":
        raise ValueError("Selected history model has no diagnostic cache")
    return json.loads(verify_artifact(validation["details"]).read_text())


def compare_runs(run_reports, r1_report, plan):
    r1 = json.loads(Path(r1_report).read_text())
    plan_record = artifact(plan)
    if not r1["passed"] or PLAN_VERSION not in Path(plan).read_text():
        raise ValueError("Comparison requires the accepted R1 data and compact plan")
    baseline = r1["baseline"]["validation"]
    baseline_row = make_candidate("single_frame", 0, baseline, baseline)
    baseline_details = json.loads(verify_artifact(r1["artifacts"]["details"]).read_text())["validation"]
    runs, report_records, rows, selected, selected_details = {}, {}, [baseline_row], {}, {}
    shared = None
    for report_path in run_reports:
        run = json.loads(Path(report_path).read_text())
        model_id = run["model_id"]
        if model_id in runs:
            raise ValueError(f"Duplicate model report: {model_id}")
        if (run["task"] != "R4_compact_training" or not run["passed"]
                or not all(run["checks"].values()) or run["plan_version"] != PLAN_VERSION):
            raise ValueError("Comparison requires completed compact training reports")
        validate_budget(run["config"])
        if run["config"]["model_id"] != model_id or run["baseline_validation"] != baseline:
            raise ValueError("Training model or baseline differs from its supplied report")
        if run["plan"]["sha256"] != plan_record["sha256"]:
            raise ValueError("Training plan SHA256 differs from the supplied compact plan")
        for name, record in run["source_reports"].items():
            verify_artifact(record)
        if (run["source_reports"]["plan"]["sha256"] != plan_record["sha256"]
                or run["source_reports"]["r1_report"]["sha256"] != file_sha256(r1_report)):
            raise ValueError("Training source reports do not match R1 and the supplied plan")
        expected_sources = {"r1_report_sha256": file_sha256(r1_report),
                            "r1_review_sha256": run["source_reports"]["r1_review"]["sha256"],
                            "split_manifest_sha256": r1["artifacts"]["split_manifest"]["sha256"],
                            "prediction_index_sha256": r1["cache"]["index_sha256"]}
        if run["sources"] != expected_sources:
            raise ValueError("Training development sources differ from accepted R1")
        for name, digest in run["code_sha256"].items():
            if file_sha256(REPO_ROOT / name) != digest:
                raise ValueError(f"Training source code SHA256 mismatch: {name}")
        contract = {"sources": run["sources"], "feature_contract": run["feature_contract"],
                    "feature_report_sha256": run["source_reports"]["feature_report"]["sha256"],
                    "code_sha256": run["code_sha256"], "git_commit": run["git_commit"],
                    "shared_initialization_sha256": run["shared_initialization_sha256"],
                    "config": {key: value for key, value in run["config"].items() if key != "model_id"}}
        if shared is None:
            shared = contract
        elif shared != contract:
            raise ValueError("Compact runs do not share sources, budget, code and initialization")
        history = run["history"]
        if [item["epoch"] for item in history] != list(range(1, COMPACT_BUDGET["epochs"] + 1)):
            raise ValueError("Compact run does not contain every planned epoch")
        candidates = [make_candidate(model_id, item["epoch"], item["validation"], baseline)
                      for item in history]
        if candidates != run["candidates"]:
            raise ValueError("Reported candidates differ from recomputed validation selection")
        if run["pareto_candidate_ids"] != pareto_front([baseline_row, *candidates]):
            raise ValueError("Reported per-model Pareto set differs from all epoch results")
        for item in history:
            verify_artifact(item["checkpoint"])
            if (item["training"]["valid_frames"] != r1["split"]["partitions"]["train"]["counts"]["person_frames"]
                    or item["validation"]["standard"]["person_frames"] != r1["split"]["partitions"]["validation"]["counts"]["person_frames"]):
                raise ValueError("Training or validation coverage differs from the manifest")
        verify_artifact(run["last_checkpoint"])
        if run["last_checkpoint"]["sha256"] != history[-1]["checkpoint"]["sha256"]:
            raise ValueError("Last checkpoint does not match the final epoch")
        winner = select_checkpoint(candidates)
        choice = run["selection"]
        if winner is None:
            if choice != {"status": "no_precision_eligible_checkpoint", "epoch": None,
                          "candidate_id": None, "checkpoint": None}:
                raise ValueError("Reported selection must retain no eligible checkpoint")
            selected[model_id] = None
        else:
            item = next(item for item in history if item["epoch"] == winner["epoch"])
            if (choice["status"] != "selected" or choice["epoch"] != winner["epoch"]
                    or choice["candidate_id"] != winner["candidate_id"]
                    or choice["checkpoint"]["sha256"] != item["checkpoint"]["sha256"]):
                raise ValueError("Reported checkpoint selection differs from validation ranking")
            verify_artifact(choice["checkpoint"])
            selected[model_id] = winner
            selected_details[model_id] = verify_selected_validation(
                run, item, r1["split"]["partitions"]["validation"]["counts"])
        runs[model_id], report_records[model_id] = run, artifact(report_path)
        rows.extend(candidates)
    if not {"ordinary_gru", "history_validity"}.issubset(runs):
        raise ValueError("First-round comparison requires ordinary_gru and history_validity")
    paired = {f"{name}_vs_single_frame": paired_details("single_frame", name,
                  baseline_details, details) for name, details in selected_details.items()}
    paired["history_validity_vs_ordinary_gru"] = (
        paired_details("ordinary_gru", "history_validity", selected_details["ordinary_gru"],
                       selected_details["history_validity"])
        if all(name in selected_details for name in ("ordinary_gru", "history_validity")) else None)
    for name in runs:
        paired.setdefault(f"{name}_vs_single_frame", None)
    reference = baseline_row["metrics"]
    deltas = {row["candidate_id"]: {key: value - reference[key]
              if value is not None and reference[key] is not None else None
              for key, value in row["metrics"].items()} for row in rows}
    preference = select_checkpoint([baseline_row, *(row for row in selected.values() if row is not None)])
    return {"task": "R4_compact_comparison", "passed": True,
            "stage_status": "awaiting_development_review", "plan_version": PLAN_VERSION,
            "plan": plan_record, "source_reports": {"r1_report": artifact(r1_report), **report_records},
            "shared_contract": shared, "candidate_rows": rows, "selected": selected,
            "per_model_selection": {name: run["selection"] for name, run in runs.items()},
            "validation_preference": preference, "pareto_candidate_ids": pareto_front(rows),
            "baseline_validation": baseline, "paired_response": paired,
            "deltas_vs_single_frame": deltas,
            "delta_convention": "candidate_minus_single_frame; rates_are_absolute_ratio_differences",
            "parameter_counts": {name: run["parameter_count"] for name, run in runs.items()},
            "checks": {"sources_budget_code_and_initializer_match": True,
                       "all_epoch_candidates_and_selections_recomputed": True,
                       "all_checkpoint_artifacts_verified": True,
                       "selected_prediction_and_diagnostic_artifacts_verified": True,
                       "selected_event_identities_and_windows_match": True},
            "test_configuration_frozen": False,
            "interpretation": "single_seed_development_comparison; selection_snapshot_requires_review; not_test_or_significance_evidence"}


def selection_manifest(report):
    return {"task": "R4_development_selection_snapshot", "plan_version": report["plan_version"],
            "plan": report["plan"], "source_reports": report["source_reports"],
            "shared_contract": report["shared_contract"],
            "per_model_selection": report["per_model_selection"],
            "validation_preference": report["validation_preference"],
            "pareto_candidate_ids": report["pareto_candidate_ids"],
            "status": "awaiting_development_review", "test_configuration_frozen": False}


def summary_markdown(report):
    lines = ["# R4 compact validation comparison", "",
             f"- Plan: `{report['plan_version']}`; single seed, all planned epochs retained.",
             "- Status: development snapshot awaiting review; test configuration is not frozen.", "",
             "| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Eligible | Selected |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    selected = {row["candidate_id"] for row in report["selected"].values() if row is not None}
    for row in report["candidate_rows"]:
        values = [row["metrics"][key] for key in ("auc", "l2", "inout_ap", "accurate_stable_ratio",
                                                  "state_error_rate", "joint_gaze_shift_response_rate")]
        formatted = ["null" if value is None else f"{value:.6f}" for value in values]
        lines.append(f"| {row['model_id']} | {row['epoch']} | " + " | ".join(formatted)
                     + f" | {row['precision_eligible']} | {row['candidate_id'] in selected} |")
    lines.extend(("", f"- Pareto IDs: `{json.dumps(report['pareto_candidate_ids'])}`.",
                  f"- Validation preference: `{json.dumps(report['validation_preference'])}`.",
                  "- Deltas, all state/flip/response metrics and common-event paired delays are in the JSON.",
                  "- A selected checkpoint or Pareto membership does not by itself establish method effectiveness."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, nargs="+", required=True)
    for name in ("r1-report", "plan", "report", "summary", "selection-manifest", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare_runs(args.run_report, args.r1_report, args.plan)
    paths = [Path(__file__), Path(__file__).with_name("eval_temporal.py"),
             REPO_ROOT / "cscwd/code/stag/compact_protocol.py", REPO_ROOT / "cscwd/code/stag/cache.py"]
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
        stream.write(f"\n## R4 compact validation comparison\n\n"
                     f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
                     f"- Command: `{report['command']}`.\n")
        for name, path in (("Report", args.report), ("Summary", args.summary),
                           ("Development selection snapshot", args.selection_manifest)):
            stream.write(f"- {name}: `{path}`; SHA256: `{file_sha256(path)}`.\n")
        stream.write("- Test configuration frozen: `false`; awaiting development review.\n")
    print(f"R4 compact comparison completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
