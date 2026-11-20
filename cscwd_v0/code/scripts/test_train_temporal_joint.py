"""Synthetic R1-to-R2 training, validation-cache and checkpoint integration."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import test_prepare_development as r1_fixtures
from prepare_development import run_preparation
from train_temporal_joint import run_smoke_training
from cache_frozen_features import feature_sources
from stag.cache import file_sha256, load_cache, write_json
from stag.feature_cache import unique_image_records, write_feature_cache
from stag.temporal_model import CausalJointRefiner


SCRIPT = Path(__file__).with_name("train_temporal_joint.py")


class JointTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = r1_fixtures.DevelopmentPreparationTests().make_fixture(self.root / "fixture")
        f = self.fixture
        r1 = run_preparation(f["data_path"], f["cache_dir"], f["cache_report"],
            f["recovery_report"], f["reference_report"], self.root / "r1", self.root / "previews",
            cases_per_category=1, preview_count=0)
        self.r1_path = self.root / "r1_report.json"
        write_json(self.r1_path, r1)
        self.review_path = self.root / "r1_review.json"
        write_json(self.review_path, {"status": "accepted_for_R2_preparation",
                    "source_report": {"sha256": file_sha256(self.r1_path)}})
        index, _ = load_cache(f["cache_dir"])
        records = unique_image_records(index["tracks"])
        rng = np.random.default_rng(3)
        features = rng.normal(size=(len(records), 4, 2, 2)).astype(np.float32)
        contract = {"layer_index": 11, "channels": 4, "grid": [2, 2], "storage_dtype": "float16"}
        sources = {"r1_report_sha256": file_sha256(self.r1_path),
                   "r1_review_sha256": file_sha256(self.review_path),
                   "split_manifest_sha256": r1["artifacts"]["split_manifest"]["sha256"],
                   "prediction_index_sha256": r1["cache"]["index_sha256"]}
        self.feature_dir = self.root / "features"
        cache = write_feature_cache([record["path"] for record in records], [features],
                                     self.feature_dir, contract, sources)
        self.feature_report = self.root / "feature_report.json"
        write_json(self.feature_report, {"mode": "cache", "passed": True, "cache": cache, "sources": sources})

    def run_training(self, output):
        return run_smoke_training(self.feature_dir, self.feature_report, self.r1_path,
            self.review_path, output, epochs=1, hidden_dim=8, clip_frames=4, batch_size=2, device="cpu")

    def test_train_both_heads_and_reopen_full_validation_predictions_and_checkpoint(self):
        before = {path: file_sha256(path) for path in self.feature_dir.iterdir()}
        report = self.run_training(self.root / "training")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        checkpoint = torch.load(report["checkpoint"]["path"], map_location="cpu", weights_only=True)
        model = CausalJointRefiner(4, 8)
        model.load_state_dict(checkpoint["model"])
        val = report["history"][-1]["validation"]
        index, arrays = load_cache(val["cache"]["path"])
        r1 = json.loads(self.r1_path.read_text())
        self.assertEqual(index["person_frames"], r1["split"]["partitions"]["validation"]["counts"]["person_frames"])
        self.assertTrue(np.isfinite(arrays["heatmap"]).all())
        self.assertTrue(all(t["track_id"].startswith("train/") for t in index["tracks"]))
        self.assertIn("source_global_start", index["tracks"][0])
        self.assertEqual(before, {path: file_sha256(path) for path in self.feature_dir.iterdir()})
        self.assertEqual(report["config"]["selection"], "none; fixed_smoke_configuration; retain_last_epoch; no_test")

    def test_tampered_feature_file_cannot_start_training(self):
        array = np.load(self.feature_dir / "scene_features.npy", mmap_mode="r+")
        array[0, 0, 0, 0] += 1
        array.flush()
        with self.assertRaisesRegex(ValueError, "feature file SHA256"):
            self.run_training(self.root / "not_created")
        self.assertFalse((self.root / "not_created").exists())

    def test_mismatched_manual_review_cannot_start_training(self):
        write_json(self.review_path, {"status": "accepted_for_R2_preparation",
                   "source_report": {"sha256": "wrong"}})
        with self.assertRaisesRegex(ValueError, "accepted R1 review"):
            self.run_training(self.root / "not_created")

    def test_feature_probe_sources_reuse_the_accepted_prediction_and_split_identity(self):
        f = self.fixture
        r1, run, index, arrays, coverage, split, sources = feature_sources(
            self.r1_path, self.review_path, f["cache_report"], f["recovery_report"], f["reference_report"])
        self.assertEqual(coverage, f["coverage"])
        self.assertEqual(sources["prediction_index_sha256"], r1["cache"]["index_sha256"])
        self.assertEqual(sources["split_manifest_sha256"], r1["artifacts"]["split_manifest"]["sha256"])
        self.assertFalse(arrays["heatmap"].flags.writeable)
        self.assertEqual(len(index["tracks"]), sum(part["counts"]["tracks"] for part in split["partitions"].values()))

    def test_feature_probe_rejects_a_prediction_report_outside_the_accepted_R1_chain(self):
        f = self.fixture
        changed = json.loads(f["cache_report"].read_text())
        changed["model"] = "changed"
        write_json(f["cache_report"], changed)
        with self.assertRaisesRegex(ValueError, "Prediction report"):
            feature_sources(self.r1_path, self.review_path, f["cache_report"],
                            f["recovery_report"], f["reference_report"])

    def test_cli_records_real_synthetic_training_artifacts_and_runtime(self):
        report, summary, provenance = (self.root / name for name in ("smoke.json", "smoke.md", "provenance.md"))
        provenance.write_text("# Synthetic provenance\n")
        command = [sys.executable, str(SCRIPT), "--feature-cache", str(self.feature_dir),
            "--feature-report", str(self.feature_report), "--r1-report", str(self.r1_path),
            "--r1-review", str(self.review_path), "--output-dir", str(self.root / "cli_train"),
            "--epochs", "1", "--hidden-dim", "8", "--clip-frames", "4", "--batch-size", "2",
            "--learning-rate", "0.0001", "--lambda-state", "0.1", "--seed", "0", "--device", "cpu",
            "--report", str(report), "--summary", str(summary), "--provenance", str(provenance)]
        import os
        result = subprocess.run(command, capture_output=True, text=True,
                                env={**os.environ, "OMP_NUM_THREADS": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        restored = json.loads(report.read_text())
        self.assertIn("R2 joint smoke completed", result.stdout)
        self.assertTrue(restored["passed"])
        self.assertIn(str(report), provenance.read_text())
        self.assertEqual(restored["checkpoint"]["sha256"], file_sha256(restored["checkpoint"]["path"]))
        self.assertIn("not a final", summary.read_text())


if __name__ == "__main__":
    unittest.main()
