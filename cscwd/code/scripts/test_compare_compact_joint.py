"""Synthetic end-to-end R4 comparison, source identity and paired events."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import test_train_temporal_joint as r2_fixtures
from compare_compact_joint import (compare_runs, pair_events, selection_manifest,
                                   verify_selected_validation)
from stag.cache import file_sha256, write_json
from stag.compact_protocol import make_candidate, pareto_front


class CompactComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = r2_fixtures.JointTrainingTests()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.fixture.setUp()
        cls.root = cls.fixture.root
        cls.plan = cls.root / "plan.md"
        cls.plan.write_text("# Synthetic plan\n\nr4_compact_v2\n")
        cls.reports = []
        for model in ("ordinary_gru", "history_validity"):
            report = cls.root / f"{model}.json"
            command = [sys.executable, str(Path(__file__).with_name("train_compact_joint.py")),
                "--model-id", model, "--feature-cache", str(cls.fixture.feature_dir),
                "--feature-report", str(cls.fixture.feature_report),
                "--r1-report", str(cls.fixture.r1_path), "--r1-review", str(cls.fixture.review_path),
                "--plan", str(cls.plan), "--output-dir", str(cls.root / model), "--device", "cpu",
                "--report", str(report), "--summary", str(cls.root / f"{model}.md"),
                "--provenance", str(cls.root / "provenance.md")]
            result = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            if result.returncode:
                raise AssertionError(result.stderr + result.stdout)
            cls.reports.append(report)

    def compare(self, paths=None):
        return compare_runs(self.reports if paths is None else paths,
                            self.fixture.r1_path, self.plan)

    def altered_report(self, mutate):
        value = json.loads(self.reports[1].read_text())
        mutate(value)
        path = self.root / "altered_history.json"
        write_json(path, value)
        return [self.reports[0], path]

    def test_all_epochs_and_development_snapshot_are_retained(self):
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["candidate_rows"]), 11)
        for model in ("ordinary_gru", "history_validity"):
            self.assertEqual([row["epoch"] for row in report["candidate_rows"]
                              if row["model_id"] == model], [1, 2, 3, 4, 5])
        self.assertFalse(report["test_configuration_frozen"])
        self.assertFalse(selection_manifest(report)["test_configuration_frozen"])
        for name, selected in report["selected"].items():
            if selected is not None:
                self.assertTrue(selected["precision_eligible"])
                self.assertIsNotNone(report["paired_response"][f"{name}_vs_single_frame"])
        self.assertEqual(report["parameter_counts"]["history_validity"]
                         - report["parameter_counts"]["ordinary_gru"], 25155)

    def test_pairing_uses_original_frame_identity_and_common_responders(self):
        def event(frame, global_index, status, delay):
            return {"track_id": "train/video/clip/person", "type": "enter_frame",
                    "event_frame_index": frame, "event_global_index": global_index,
                    "window_frames": 8, "status": status, "response_delay_frames": delay}
        reference = [event(100, 500, "responded", 4), event(120, 520, "responded", 1)]
        candidate = [event(120, 20, "censored", None), event(100, 0, "responded", 2)]
        result = pair_events(reference, candidate, state_only=True)
        self.assertNotIn("gaze_shift", result)
        self.assertEqual(result["all"]["both_responded_events"], 1)
        self.assertEqual(result["all"]["reference_only_responded_events"], 1)
        self.assertEqual(result["all"]["candidate_only_responded_events"], 0)
        self.assertEqual(result["all"]["common_responded_delay_delta_frames"]["mean"], -2)

    def test_changed_event_windows_cannot_be_paired(self):
        event = {"track_id": "p", "type": "gaze_shift", "event_frame_index": 10,
                 "event_global_index": 50, "window_frames": 5,
                 "status": "responded", "response_delay_frames": 0}
        with self.assertRaisesRegex(ValueError, "identities or windows"):
            pair_events([event], [{**event, "window_frames": 6}])

    def test_history_validation_artifacts_and_events_are_verified_even_if_not_selected(self):
        # Exercise history artifact verification independently of synthetic AUC eligibility.
        run = json.loads(self.reports[1].read_text())
        r1 = json.loads(self.fixture.r1_path.read_text())
        item = run["history"][-1]
        details = verify_selected_validation(run, item,
            r1["split"]["partitions"]["validation"]["counts"])
        baseline = json.loads(Path(r1["artifacts"]["details"]["path"]).read_text())["validation"]
        paired = pair_events(baseline["joint"]["events"], details["joint"]["events"])
        self.assertEqual(paired["all"]["events"], len(baseline["joint"]["events"]))
        diagnostic = Path(item["validation"]["diagnostics"]["path"]) / "index.json"
        original = diagnostic.read_bytes()
        try:
            diagnostic.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "diagnostic index SHA256"):
                verify_selected_validation(run, item,
                    r1["split"]["partitions"]["validation"]["counts"])
        finally:
            diagnostic.write_bytes(original)

    def test_mixed_initialization_or_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.compare(self.altered_report(lambda r: r.update(shared_initialization_sha256="different")))
        def change_source(r):
            r["sources"]["prediction_index_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "development sources"):
            self.compare(self.altered_report(change_source))

    def test_reported_candidate_and_selection_are_recomputed(self):
        def change_candidate(r):
            r["candidates"][0]["metrics"]["auc"] += 0.01
        with self.assertRaisesRegex(ValueError, "recomputed validation selection"):
            self.compare(self.altered_report(change_candidate))
        def change_selection(r):
            r["selection"]["epoch"] = 999
        with self.assertRaisesRegex(ValueError, "selection"):
            self.compare(self.altered_report(change_selection))

    def test_checkpoint_file_tampering_is_detected(self):
        run = json.loads(self.reports[1].read_text())
        path = Path(run["history"][0]["checkpoint"]["path"])
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Artifact SHA256"):
                self.compare()
        finally:
            path.write_bytes(original)

    def test_no_eligible_model_keeps_single_frame_and_no_test_freeze(self):
        paths = []
        for source in self.reports:
            run = json.loads(source.read_text())
            baseline = run["baseline_validation"]
            for item in run["history"]:
                item["validation"]["standard"]["auc"] = baseline["standard"]["auc"] - 0.1
            run["candidates"] = [make_candidate(run["model_id"], item["epoch"],
                                  item["validation"], baseline) for item in run["history"]]
            run["pareto_candidate_ids"] = pareto_front([
                make_candidate("single_frame", 0, baseline, baseline), *run["candidates"]])
            run["selection"] = {"status": "no_precision_eligible_checkpoint", "epoch": None,
                                "candidate_id": None, "checkpoint": None}
            path = self.root / f"controlled_no_eligible_{run['model_id']}.json"
            write_json(path, run)
            paths.append(path)
        report = self.compare(paths)
        self.assertEqual(report["validation_preference"]["model_id"], "single_frame")
        self.assertTrue(all(row is None for row in report["selected"].values()))
        self.assertTrue(all(row is None for row in report["paired_response"].values()))
        self.assertFalse(report["test_configuration_frozen"])

    def test_comparison_cli_publishes_hashed_report_and_snapshot(self):
        report, summary, manifest = (self.root / name for name in (
            "comparison.json", "comparison.md", "selection_manifest.json"))
        command = [sys.executable, str(Path(__file__).with_name("compare_compact_joint.py")),
            "--run-report", *(str(p) for p in self.reports),
            "--r1-report", str(self.fixture.r1_path), "--plan", str(self.plan),
            "--report", str(report), "--summary", str(summary),
            "--selection-manifest", str(manifest), "--provenance", str(self.root / "provenance.md")]
        result = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("R4 compact comparison completed", result.stdout)
        snapshot = json.loads(manifest.read_text())
        self.assertEqual(snapshot["comparison_report"]["sha256"], file_sha256(report))
        self.assertFalse(snapshot["test_configuration_frozen"])
        self.assertIn("ordinary_gru | 5", summary.read_text())
        self.assertIn("history_validity | 5", summary.read_text())
        self.assertIn(file_sha256(manifest), (self.root / "provenance.md").read_text())


if __name__ == "__main__":
    unittest.main()
