"""Actual synthetic train/test CLI runs, selected artifacts and comparison output."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_official_joint import compare_runs, selection_manifest, verify_selected_test
from stag.cache import file_sha256, write_json
from stag.official_protocol import PROTOCOL_VERSION, select_checkpoint
from test_official_data import INPUT_NAMES, make_official_fixture


class OfficialComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.fixture = make_official_fixture(cls.root / "sources")
        cls.plan = cls.root / "plan.md"
        cls.plan.write_text(f"# Synthetic official protocol\n\n{PROTOCOL_VERSION}\n")
        cls.reports = []
        for model in ("ordinary_gru", "history_validity"):
            report = cls.root / f"{model}.json"
            command = [sys.executable, str(Path(__file__).with_name("train_official_joint.py")),
                "--model-id", model, "--plan", str(cls.plan), "--output-dir", str(cls.root / model),
                "--device", "cpu", "--report", str(report), "--summary", str(cls.root / f"{model}.md"),
                "--provenance", str(cls.root / "provenance.md")]
            for name in INPUT_NAMES:
                command.extend(("--" + name.replace("_", "-"), str(cls.fixture[name])))
            process = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            if process.returncode:
                raise AssertionError(process.stderr + process.stdout)
            cls.reports.append(report)

    def compare(self, paths=None):
        return compare_runs(self.reports if paths is None else paths, self.plan)

    def altered_report(self, mutate):
        report = json.loads(self.reports[1].read_text())
        mutate(report)
        path = self.root / "altered_history.json"
        write_json(path, report)
        return [self.reports[0], path]

    def test_all_test_epochs_and_explicit_test_selected_protocol_are_reported(self):
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["candidate_rows"]), 11)
        self.assertTrue(report["test_selected"])
        self.assertTrue(report["test_used_for_checkpoint_selection"])
        self.assertFalse(report["test_labels_used_for_gradient"])
        self.assertFalse(report["independent_test_estimate"])
        self.assertEqual(report["test_preference"], select_checkpoint(report["candidate_rows"]))
        for model in ("ordinary_gru", "history_validity"):
            rows = [row for row in report["candidate_rows"] if row["model_id"] == model]
            self.assertEqual([row["epoch"] for row in rows], [1, 2, 3, 4, 5])
            self.assertEqual(report["selected"][model], select_checkpoint(rows))
            self.assertIsNotNone(report["paired_response"][model + "_vs_single_frame"])
            self.assertTrue(all("precision_eligible" not in row for row in rows))
        manifest = selection_manifest(report)
        self.assertEqual(manifest["status"], "ready_for_reporting_test_selected_protocol")
        self.assertEqual(manifest["selection_split"], "test")

    def test_mixed_initialization_or_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.compare(self.altered_report(lambda r: r.update(shared_initialization_sha256="different")))
        def change_source(report):
            report["sources"]["test_prediction_report_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "source identities"):
            self.compare(self.altered_report(change_source))

    def test_reported_candidates_and_selection_are_independently_recomputed(self):
        def change_candidate(report):
            report["candidates"][0]["metrics"]["auc"] += .01
        with self.assertRaisesRegex(ValueError, "recomputed test metrics"):
            self.compare(self.altered_report(change_candidate))
        with self.assertRaisesRegex(ValueError, "selection"):
            self.compare(self.altered_report(lambda r: r["selection"].update(epoch=999)))

    def test_old_validation_or_changed_training_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "official train/test selection"):
            self.compare(self.altered_report(lambda r: r["config"].update(selection_split="validation")))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.compare(self.altered_report(lambda r: r["history"][0]["training"].update(valid_frames=12)))

    def test_checkpoint_tampering_is_detected(self):
        run = json.loads(self.reports[1].read_text())
        path = Path(run["history"][0]["checkpoint"]["path"])
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Artifact SHA256"):
                self.compare()
        finally:
            path.write_bytes(original)

    def test_selected_test_prediction_and_history_diagnostics_are_verified(self):
        run = json.loads(self.reports[1].read_text())
        item = run["history"][run["selection"]["epoch"] - 1]
        counts = json.loads(self.fixture["test_prediction_report"].read_text())["cache"]
        details = verify_selected_test(run, item, counts)
        self.assertEqual(set(details), {"joint", "state"})
        path = Path(item["test"]["cache"]["path"]) / "inout.npy"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Cache SHA256"):
                verify_selected_test(run, item, counts)
        finally:
            path.write_bytes(original)
        diagnostic = Path(item["test"]["diagnostics"]["path"]) / "index.json"
        original = diagnostic.read_bytes()
        try:
            diagnostic.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "diagnostic index SHA256"):
                verify_selected_test(run, item, counts)
        finally:
            diagnostic.write_bytes(original)

    def test_duplicate_or_missing_primary_model_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate model"):
            self.compare([self.reports[0], self.reports[0]])
        with self.assertRaisesRegex(ValueError, "requires ordinary_gru and history_validity"):
            self.compare([self.reports[1]])

    def test_comparison_cli_publishes_all_rows_and_hashed_test_selection_manifest(self):
        report, summary, manifest = (self.root / name for name in (
            "comparison.json", "comparison.md", "selection.json"))
        command = [sys.executable, str(Path(__file__).with_name("compare_official_joint.py")),
            "--run-report", *(str(path) for path in self.reports), "--plan", str(self.plan),
            "--report", str(report), "--summary", str(summary), "--selection-manifest", str(manifest),
            "--provenance", str(self.root / "provenance.md")]
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertIn("R4 official comparison completed", process.stdout)
        selected = json.loads(manifest.read_text())
        self.assertTrue(selected["test_used_for_checkpoint_selection"])
        self.assertFalse(selected["independent_test_estimate"])
        self.assertEqual(selected["comparison_report"]["sha256"], file_sha256(report))
        self.assertIn("ordinary_gru | 5", summary.read_text())
        self.assertIn("history_validity | 5", summary.read_text())
        self.assertIn("not an independent held-out estimate", summary.read_text())
        self.assertIn(file_sha256(manifest), (self.root / "provenance.md").read_text())


if __name__ == "__main__":
    unittest.main()
