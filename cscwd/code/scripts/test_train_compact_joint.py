"""Synthetic fixed-budget R4 training, artifact selection and CLI integration."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_train_temporal_joint as r2_fixtures
from train_compact_joint import (build_model, run_compact_training,
                                 shared_initialization_sha256)
from stag.cache import file_sha256, load_cache
from stag.compact_protocol import COMPACT_BUDGET, PLAN_VERSION
from stag.compact_validation import evaluate_compact_model


SCRIPT = Path(__file__).with_name("train_compact_joint.py")


class CompactTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = r2_fixtures.JointTrainingTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        self.plan = self.root / "compact_plan.md"
        self.plan.write_text(f"# Synthetic compact plan\n\nVersion: {PLAN_VERSION}\n")
        self.r1 = json.loads(self.fixture.r1_path.read_text())

    def run_training(self, model_id="ordinary_gru", name="run", **kwargs):
        return run_compact_training(self.fixture.feature_dir, self.fixture.feature_report,
            self.fixture.r1_path, self.fixture.review_path, self.plan,
            self.root / name, model_id=model_id, device="cpu", **kwargs)

    def test_five_epochs_both_models_share_initialization_and_preserve_sources(self):
        paths = [*self.fixture.feature_dir.iterdir(), *self.fixture.fixture["cache_dir"].iterdir()]
        original = {str(path): file_sha256(path) for path in paths}
        reports = [self.run_training(model_id, model_id)
                   for model_id in ("ordinary_gru", "history_validity")]
        self.assertEqual(reports[0]["shared_initialization_sha256"],
                         reports[1]["shared_initialization_sha256"])
        self.assertLess(reports[0]["parameter_count"], reports[1]["parameter_count"])
        for report in reports:
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual([h["epoch"] for h in report["history"]], [1, 2, 3, 4, 5])
            self.assertEqual(report["baseline_validation"], self.r1["baseline"]["validation"])
            self.assertEqual({key: report["config"][key] for key in COMPACT_BUDGET}, COMPACT_BUDGET)
            for item in report["history"]:
                self.assertEqual(item["training"]["valid_frames"],
                                 self.r1["split"]["partitions"]["train"]["counts"]["person_frames"])
                index, arrays = load_cache(item["validation"]["cache"]["path"])
                self.assertEqual(index["person_frames"],
                                 self.r1["split"]["partitions"]["validation"]["counts"]["person_frames"])
                self.assertTrue(np.isfinite(arrays["heatmap"]).all())
                self.assertEqual(item["checkpoint"]["sha256"], file_sha256(item["checkpoint"]["path"]))
                if report["model_id"] == "ordinary_gru":
                    self.assertIsNone(item["validation"]["diagnostics"])
                else:
                    self.assertTrue(item["validation"]["diagnostics"]["passed"])
        self.assertEqual(original, {str(path): file_sha256(path) for path in paths})

    def test_selected_checkpoint_follows_best_eligible_epoch_not_last_epoch(self):
        # Controlled synthetic validation values exercise the checkpoint controller.
        ranking = iter((0.4, 0.9, 0.7, 0.6, 0.5))
        def controlled(*args, **kwargs):
            result = evaluate_compact_model(*args, **kwargs)
            result["standard"] = copy.deepcopy(self.r1["baseline"]["validation"]["standard"])
            result["temporal"]["stability"]["accurate_stable_ratio"] = next(ranking)
            return result
        with patch("train_compact_joint.evaluate_compact_model", side_effect=controlled):
            report = self.run_training()
        self.assertEqual(report["selection"]["epoch"], 2)
        selected = torch.load(report["selection"]["checkpoint"]["path"], weights_only=True, map_location="cpu")
        last = torch.load(report["last_checkpoint"]["path"], weights_only=True, map_location="cpu")
        self.assertEqual(selected["epoch"], 2)
        self.assertEqual(last["epoch"], 5)
        self.assertEqual(report["selection"]["checkpoint"]["sha256"], report["history"][1]["checkpoint"]["sha256"])
        self.assertEqual(report["last_checkpoint"]["sha256"], report["history"][-1]["checkpoint"]["sha256"])

    def test_no_precision_eligible_checkpoint_is_reported_without_fallback_selection(self):
        def controlled(*args, **kwargs):
            result = evaluate_compact_model(*args, **kwargs)
            result["standard"]["auc"] = self.r1["baseline"]["validation"]["standard"]["auc"] - 0.1
            return result
        with patch("train_compact_joint.evaluate_compact_model", side_effect=controlled):
            report = self.run_training()
        self.assertEqual(report["selection"]["status"], "no_precision_eligible_checkpoint")
        self.assertIsNone(report["selection"]["checkpoint"])
        self.assertFalse((self.root / "run" / "selected.pt").exists())
        self.assertTrue(Path(report["last_checkpoint"]["path"]).exists())
        self.assertTrue(report["passed"])

    def test_budget_or_plan_mismatch_does_not_start_training(self):
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            self.run_training(epochs=1)
        self.assertFalse((self.root / "run").exists())
        self.plan.write_text("# Different experimental plan\n")
        with self.assertRaisesRegex(ValueError, "protocol version"):
            self.run_training()
        self.assertFalse((self.root / "run").exists())

    def test_shared_initialization_fingerprint_is_seed_sensitive_and_excludes_new_controls(self):
        hashes = []
        for model_id in ("ordinary_gru", "history_validity", "always_keep"):
            torch.manual_seed(0)
            model = build_model(model_id, 4, 128, 0.1)
            hashes.append(shared_initialization_sha256(model))
        self.assertEqual(len(set(hashes)), 1)
        torch.manual_seed(1)
        self.assertNotEqual(hashes[0], shared_initialization_sha256(build_model("ordinary_gru", 4, 128, 0.1)))

    def test_cli_reports_real_artifacts_and_checkpoint_reloads_validation_predictions(self):
        report_path, summary, provenance = (self.root / name for name in ("report.json", "summary.md", "provenance.md"))
        provenance.write_text("# Synthetic R4 provenance\n")
        command = [sys.executable, str(SCRIPT), "--model-id", "history_validity",
            "--feature-cache", str(self.fixture.feature_dir), "--feature-report", str(self.fixture.feature_report),
            "--r1-report", str(self.fixture.r1_path), "--r1-review", str(self.fixture.review_path),
            "--plan", str(self.plan), "--output-dir", str(self.root / "cli"), "--device", "cpu",
            "--report", str(report_path), "--summary", str(summary), "--provenance", str(provenance)]
        for name, value in COMPACT_BUDGET.items():
            command.extend(("--" + name.replace("_", "-"), str(value)))
        process = subprocess.run(command, capture_output=True, text=True,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("R4 compact training completed", process.stdout)
        report = json.loads(report_path.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["runtime"]["device_name"], "CPU")
        self.assertIn("single-seed development", summary.read_text())
        self.assertIn(file_sha256(report_path), provenance.read_text())
        self.assertEqual(report["source_reports"]["feature_report"]["sha256"], file_sha256(self.fixture.feature_report))
        for name, digest in report["code_sha256"].items():
            self.assertEqual(digest, file_sha256(Path(__file__).resolve().parents[3] / name))
        cp = torch.load(report["last_checkpoint"]["path"], weights_only=True, map_location="cpu")
        self.assertEqual(cp["epoch"], 5)
        self.assertEqual(cp["config"]["initialization"], "fresh_new_module; shared_seed_initialization; no_R2_resume")
        model = build_model("history_validity", 4, 128, 0.1)
        model.load_state_dict(cp["model"])
        model.eval()
        index, arrays = load_cache(report["history"][-1]["validation"]["cache"]["path"])
        source_index, source_arrays = load_cache(self.fixture.fixture["cache_dir"])
        by_id = {track["track_id"]: track for track in source_index["tracks"]}
        track = by_id[index["tracks"][0]["track_id"]]
        feature_index = json.loads((self.fixture.feature_dir / "index.json").read_text())
        features = np.load(self.fixture.feature_dir / "scene_features.npy", mmap_mode="r")
        frame = track["frames"][0]
        feature_position = feature_index["image_paths"].index(frame["path"])
        with torch.no_grad():
            heatmap, score, _ = model.step(
                torch.tensor(features[feature_position:feature_position+1]),
                torch.tensor(np.array(source_arrays["heatmap"][track["start"]:track["start"]+1])),
                torch.tensor(np.array(source_arrays["inout"][track["start"]:track["start"]+1])),
                torch.tensor([frame["bbox_norm"]]))
        np.testing.assert_allclose(heatmap[0].numpy(), arrays["heatmap"][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(score[0].numpy(), arrays["inout"][0], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
