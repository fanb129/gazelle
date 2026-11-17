"""Compare full-train/per-epoch-test runs under the disclosed test selection rule."""

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
from eval_temporal import verify_completed_cache
from stag.cache import file_sha256, write_json
from stag.official_protocol import (PROTOCOL_VERSION, SELECTION_RULE, TRAINING_BUDGET,
                                    make_candidate, select_checkpoint, validate_budget)


def verify_selected_test(run, item, test_counts):
    evaluation = item["test"]
    cache = evaluation["cache"]
    verify_completed_cache(cache["path"], {"passed": run["passed"], "cache": cache})
    index = json.loads((Path(cache["path"]) / "index.json").read_text())
    metadata = index["metadata"]
    if (not index["passed"] or index["person_frames"] != test_counts["person_frames"]
            or index["track_count"] != test_counts["tracks"]
            or metadata["sources"] != run["sources"] or metadata["config"] != run["config"]
            or metadata["protocol_version"] != PROTOCOL_VERSION
            or metadata["plan_sha256"] != run["plan"]["sha256"]
            or metadata["epoch"] != item["epoch"] or metadata["evaluation_split"] != "test"
            or any(not track["track_id"].startswith("test/") for track in index["tracks"])):
        raise ValueError("Selected test cache metadata or coverage differs from its run")
    diagnostics = evaluation["diagnostics"]
    if diagnostics is not None:
        directory = Path(diagnostics["path"])
        if file_sha256(directory / "index.json") != diagnostics["index_sha256"]:
            raise ValueError("Selected history diagnostic index SHA256 mismatch")
        diagnostic_index = json.loads((directory / "index.json").read_text())
        if (not diagnostics["passed"] or not diagnostic_index["passed"]
                or diagnostic_index["metadata"] != metadata
                or diagnostic_index["person_frames"] != index["person_frames"]
                or diagnostic_index["files"] != diagnostics["files"]):
            raise ValueError("Selected history diagnostics differ from test metadata")
        for record in diagnostics["files"].values():
            verify_artifact({**record, "path": str(directory / record["path"])})
    elif run["model_id"] != "ordinary_gru":
        raise ValueError("Selected history model has no diagnostic cache")
    return json.loads(verify_artifact(evaluation["details"]).read_text())


def compare_runs(run_reports, plan):
    plan_record = artifact(plan)
    if PROTOCOL_VERSION not in Path(plan).read_text():
        raise ValueError("Comparison requires the official train/test protocol plan")
    runs, report_records, selected, selected_details = {}, {}, {}, {}
    rows, shared, baseline, baseline_details = [], None, None, None
    for report_path in run_reports:
        run = json.loads(Path(report_path).read_text())
        model_id = run["model_id"]
        if model_id in runs:
            raise ValueError(f"Duplicate model report: {model_id}")
        if (run["task"] != "R4_official_training" or not run["passed"]
                or not all(run["checks"].values()) or run["protocol_version"] != PROTOCOL_VERSION):
            raise ValueError("Comparison requires completed official train/test reports")
        config = run["config"]
        validate_budget(config)
        if (config["model_id"] != model_id or config["train_split"] != "train"
                or config["selection_split"] != "test" or config["has_validation_split"]
                or not config["test_used_for_checkpoint_selection"]):
            raise ValueError("Run does not use the declared official train/test selection protocol")
        if run["plan"]["sha256"] != plan_record["sha256"]:
            raise ValueError("Training plan SHA256 differs from the supplied official plan")
        for record in run["source_reports"].values():
            verify_artifact(record)
        if run["source_reports"]["plan"]["sha256"] != plan_record["sha256"]:
            raise ValueError("Training source plan differs from the supplied official plan")
        for name, digest in run["code_sha256"].items():
            if file_sha256(REPO_ROOT / name) != digest:
                raise ValueError(f"Training source code SHA256 mismatch: {name}")
        source_hashes = {name: record["sha256"] for name, record in run["source_reports"].items()}
        for name in ("train_prediction_report", "test_prediction_report",
                     "train_feature_report", "test_feature_report"):
            if run["sources"][name + "_sha256"] != source_hashes[name]:
                raise ValueError("Training source identities differ from the supplied reports")
        contract = {"sources": run["sources"], "feature_contract": run["feature_contract"],
                    "source_report_sha256": source_hashes,
                    "code_sha256": run["code_sha256"], "git_commit": run["git_commit"],
                    "shared_initialization_sha256": run["shared_initialization_sha256"],
                    "config": {key: value for key, value in config.items() if key != "model_id"}}
        run_baseline = {name: run["baseline_test"][name] for name in ("standard", "temporal", "state")}
        if shared is None:
            shared, baseline = contract, run_baseline
            rows.append(make_candidate("single_frame", 0, baseline))
            baseline_details = json.loads(verify_artifact(run["baseline_test"]["details"]).read_text())
        elif shared != contract or baseline != run_baseline:
            raise ValueError("Official runs do not share sources, budget, code, initialization and baseline")
        else:
            verify_artifact(run["baseline_test"]["details"])
        train_source = json.loads(Path(run["source_reports"]["train_prediction_report"]["path"]).read_text())
        test_source = json.loads(Path(run["source_reports"]["test_prediction_report"]["path"]).read_text())
        train_counts, test_counts = train_source["cache"], test_source["cache"]
        history = run["history"]
        if [item["epoch"] for item in history] != list(range(1, TRAINING_BUDGET["epochs"] + 1)):
            raise ValueError("Official run does not contain every planned epoch")
        candidates = [make_candidate(model_id, item["epoch"], item["test"]) for item in history]
        if candidates != run["candidates"]:
            raise ValueError("Reported candidates differ from recomputed test metrics")
        for item in history:
            verify_artifact(item["checkpoint"])
            if (item["training"]["valid_frames"] != train_counts["person_frames"]
                    or item["test"]["standard"]["person_frames"] != test_counts["person_frames"]):
                raise ValueError("Training or test coverage differs from the official source cache")
        verify_artifact(run["last_checkpoint"])
        if run["last_checkpoint"]["sha256"] != history[-1]["checkpoint"]["sha256"]:
            raise ValueError("Last checkpoint does not match the final epoch")
        winner = select_checkpoint(candidates)
        choice = run["selection"]
        item = history[winner["epoch"] - 1]
        if (choice["status"] != "selected" or choice["epoch"] != winner["epoch"]
                or choice["candidate_id"] != winner["candidate_id"]
                or choice["checkpoint"]["sha256"] != item["checkpoint"]["sha256"]):
            raise ValueError("Reported checkpoint selection differs from test AUC/L2/AP ranking")
        verify_artifact(choice["checkpoint"])
        selected[model_id] = winner
        selected_details[model_id] = verify_selected_test(run, item, test_counts)
        runs[model_id], report_records[model_id] = run, artifact(report_path)
        rows.extend(candidates)
    if not {"ordinary_gru", "history_validity"}.issubset(runs):
        raise ValueError("First-round comparison requires ordinary_gru and history_validity")
    paired = {f"{name}_vs_single_frame": paired_details("single_frame", name,
              baseline_details, details) for name, details in selected_details.items()}
    paired["history_validity_vs_ordinary_gru"] = paired_details("ordinary_gru", "history_validity",
        selected_details["ordinary_gru"], selected_details["history_validity"])
    reference = rows[0]["metrics"]
    deltas = {row["candidate_id"]: {key: value - reference[key]
              if value is not None and reference[key] is not None else None
              for key, value in row["metrics"].items()} for row in rows}
    return {"task": "R4_official_comparison", "passed": True,
            "stage_status": "awaiting_result_review", "protocol_version": PROTOCOL_VERSION,
            "plan": plan_record, "source_reports": report_records, "shared_contract": shared,
            "candidate_rows": rows, "selected": selected,
            "per_model_selection": {name: run["selection"] for name, run in runs.items()},
            "test_preference": select_checkpoint(rows), "baseline_test": baseline,
            "paired_response": paired, "deltas_vs_single_frame": deltas,
            "delta_convention": "candidate_minus_single_frame; rates_are_absolute_ratio_differences",
            "parameter_counts": {name: run["parameter_count"] for name, run in runs.items()},
            "selection_rule": SELECTION_RULE, "selection_split": "test",
            "test_selected": True, "test_used_for_checkpoint_selection": True,
            "test_labels_used_for_gradient": False, "independent_test_estimate": False,
            "checks": {"sources_budget_code_and_initializer_match": True,
                       "all_epoch_candidates_and_selections_recomputed": True,
                       "all_checkpoint_artifacts_verified": True,
                       "selected_prediction_and_diagnostic_artifacts_verified": True,
                       "selected_event_identities_and_windows_match": True},
            "interpretation": "single_seed_test_selected_epoch; test_used_each_epoch_and_for_checkpoint_selection; not_independent_heldout_or_significance_evidence"}


def selection_manifest(report):
    keys = ("protocol_version", "plan", "source_reports", "shared_contract", "per_model_selection",
            "test_preference", "selection_rule", "selection_split", "test_selected",
            "test_used_for_checkpoint_selection", "test_labels_used_for_gradient", "independent_test_estimate")
    return {"task": "R4_official_test_selected_checkpoint_manifest",
            "status": "ready_for_reporting_test_selected_protocol", **{key: report[key] for key in keys}}


def summary_markdown(report):
    lines = ["# R4 official train/test comparison", "",
             f"- Protocol: `{report['protocol_version']}`; all planned epochs retained.",
             "- Train: full official train; evaluation and epoch selection: full official test.",
             "- Selection: highest test AUC; ties use lower L2, higher AP, then earlier epoch.",
             "- Test is used for checkpoint selection; these are not an independent held-out estimate.", "",
             "| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Selected |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    selected = {row["candidate_id"] for row in report["selected"].values()}
    for row in report["candidate_rows"]:
        values = [row["metrics"][key] for key in ("auc", "l2", "inout_ap", "accurate_stable_ratio",
                                                  "state_error_rate", "joint_gaze_shift_response_rate")]
        formatted = ["null" if value is None else f"{value:.6f}" for value in values]
        lines.append(f"| {row['model_id']} | {row['epoch']} | " + " | ".join(formatted)
                     + f" | {row['candidate_id'] in selected} |")
    lines.extend(("", f"- Overall test preference: `{json.dumps(report['test_preference'])}`.",
                  "- All output columns come from each selected checkpoint's same epoch.",
                  "- Deltas, state/flip/response metrics and paired common-event delays are retained in JSON."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, nargs="+", required=True)
    for name in ("plan", "report", "summary", "selection-manifest", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare_runs(args.run_report, args.plan)
    paths = [Path(__file__), Path(__file__).with_name("compare_compact_joint.py"),
             Path(__file__).with_name("eval_temporal.py"),
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
        stream.write(f"\n## R4 official test-selected comparison\n\n"
                     f"- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
                     f"- Command: `{report['command']}`.\n")
        for name, path in (("Report", args.report), ("Summary", args.summary),
                           ("Test-selected checkpoint manifest", args.selection_manifest)):
            stream.write(f"- {name}: `{path}`; SHA256: `{file_sha256(path)}`.\n")
        stream.write("- Full official test evaluated after every epoch and used for checkpoint selection; no independent held-out claim.\n")
    print(f"R4 official comparison completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
