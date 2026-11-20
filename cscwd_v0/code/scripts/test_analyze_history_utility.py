"""Synthetic train-only history analysis, artifact mapping and causal isolation."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import analyze_history_utility as analysis
from test_official_data import make_official_fixture
from stag.cache import file_sha256
from stag.metrics import TemporalConfig, evaluate_temporal
from stag.official_data import _load_split
from stag.state_metrics import evaluate_state_temporal


VERSION = "r4_history_utility_v1"
CANDIDATES = ["current", "current_grid", "previous_raw", "ema_075", "transport_previous"]


class HistoryUtilityAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.fixture = make_official_fixture(cls.root / "sources")
        # The real CLI must work without any official test artifacts available.
        for path in (cls.root / "sources").glob("test_*"):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        cls.plan = cls.root / "plan.md"
        cls.plan.write_text((analysis.REPO_ROOT / "cscwd/r4_history_utility_plan.md").read_text())
        cls.original = {path: file_sha256(path) for path in (cls.root / "sources").rglob("*")
                        if path.is_file()}
        cls.report_path = cls.root / "result.json"
        cls.summary_path = cls.root / "result.md"
        cls.provenance = cls.root / "provenance.md"
        cls.output_dir = cls.root / "cli_outputs"
        command = [sys.executable, str(Path(analysis.__file__)),
            "--feature-cache", str(cls.fixture["train_feature_cache"]),
            "--feature-report", str(cls.fixture["train_feature_report"]),
            "--prediction-report", str(cls.fixture["train_prediction_report"]),
            "--plan", str(cls.plan), "--output-dir", str(cls.output_dir), "--device", "cpu",
            "--report", str(cls.report_path), "--summary", str(cls.summary_path),
            "--provenance", str(cls.provenance)]
        cls.process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if cls.process.returncode:
            raise AssertionError(cls.process.stdout + cls.process.stderr)
        cls.report = json.loads(cls.report_path.read_text())

    def load_train(self):
        return _load_split("train", self.fixture["train_feature_cache"],
            self.fixture["train_feature_report"], self.fixture["train_prediction_report"])

    def run_analysis(self, destination, fixture=None, plan=None):
        sources = fixture or self.fixture
        return analysis.run_history_utility(sources["train_feature_cache"],
            sources["train_feature_report"], sources["train_prediction_report"],
            plan or self.plan, self.root / destination, device="cpu")

    @staticmethod
    def array(report, name):
        return np.load(report["artifacts"][name]["path"], allow_pickle=False)

    def test_actual_cli_completes_with_no_test_files_and_publishes_sources(self):
        report = self.report
        self.assertEqual(self.process.returncode, 0)
        self.assertEqual(report["task"], "R4_history_utility")
        self.assertEqual(report["utility_version"], VERSION)
        self.assertEqual(report["evaluation_split"], "train")
        self.assertFalse(report["training"])
        self.assertFalse(report["test_accessed"])
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["summary"])
        self.assertEqual(set(report["source_reports"]), {"prediction", "features", "plan"})
        self.assertFalse(list((self.root / "sources").glob("test_*")))
        for record in report["source_reports"].values():
            self.assertEqual(record["sha256"], file_sha256(record["path"]))
        for path, digest in report["code_sha256"].items():
            self.assertEqual(digest, file_sha256(analysis.REPO_ROOT / path))
        self.assertIn(file_sha256(self.report_path), self.provenance.read_text())
        self.assertIn(VERSION, self.summary_path.read_text())
        self.assertEqual(self.original, {path: file_sha256(path) for path in self.original})

    def test_artifacts_reproduce_current_coordinates_and_complete_row_mapping(self):
        report = self.report
        part, _, _ = self.load_train()
        self.assertEqual(set(report["artifacts"]), {"index", "candidate_xy", "history_available",
            "errors", "group_codes", "oracle_choice", "details"})
        for record in report["artifacts"].values():
            self.assertEqual(record["sha256"], file_sha256(record["path"]))
            self.assertEqual(record["size_bytes"], Path(record["path"]).stat().st_size)
        index = json.loads(Path(report["artifacts"]["index"]["path"]).read_text())
        self.assertEqual(index["candidate_names"], CANDIDATES)
        self.assertEqual(index["person_frames"], 24)
        self.assertEqual(index["tracks"], part["tracks"])
        self.assertEqual(index["sources"], report["sources"])
        self.assertEqual(index["artifacts"],
            {name: record for name, record in report["artifacts"].items() if name != "index"})
        candidates = self.array(report, "candidate_xy")
        available = self.array(report, "history_available").astype(bool)
        self.assertEqual(candidates.shape, (24, 5, 2))
        self.assertEqual(available.shape, (24,))
        np.testing.assert_array_equal(candidates[:, 0], part["arrays"]["argmax_xy"])
        self.assertEqual(np.flatnonzero(~available).tolist(), [0, 12])
        for column in (2, 3, 4):
            np.testing.assert_array_equal(candidates[~available, column], candidates[~available, 0])
        self.assertEqual(self.array(report, "errors").shape, (24, 5))
        self.assertEqual(self.array(report, "group_codes").shape, (24,))
        self.assertEqual(self.array(report, "oracle_choice").shape, (24,))
        state = evaluate_state_temporal(part["tracks"], part["arrays"]["inout"])["summary"]
        leave = evaluate_temporal(part["tracks"], part["arrays"]["argmax_xy"],
            part["arrays"]["inout"], TemporalConfig())["summary"]["response"]["leave_frame"]
        for candidate in report["summary"]["candidates"].values():
            self.assertEqual(candidate["state"], state)
            self.assertEqual(candidate["temporal"]["response"]["leave_frame"], leave)

    def test_gt_changes_scoring_but_not_candidates_or_history_availability(self):
        part, prediction, features = self.load_train()
        changed = {**part, "tracks": copy.deepcopy(part["tracks"])}
        changed["prediction_index"] = {**part["prediction_index"], "tracks": changed["tracks"]}
        for track in changed["tracks"]:
            for frame in track["frames"]:
                if frame["inout"]:
                    frame["gazex_norm"], frame["gazey_norm"] = [.98], [.98]
        with patch.object(analysis, "_load_split", return_value=(changed, prediction, features)) as load:
            report = self.run_analysis("changed_gt")
        self.assertEqual(load.call_args.args[0], "train")
        for name in ("candidate_xy", "history_available"):
            np.testing.assert_array_equal(self.array(self.report, name), self.array(report, name))
        self.assertFalse(np.array_equal(self.array(self.report, "errors"),
                                       self.array(report, "errors"), equal_nan=True))

    def test_future_heatmaps_and_features_cannot_change_earlier_candidates(self):
        part, prediction, feature_report = self.load_train()
        changed = {**part, "arrays": {**part["arrays"],
            "heatmap": np.array(part["arrays"]["heatmap"]),
            "argmax_xy": np.array(part["arrays"]["argmax_xy"])},
            "features": np.array(part["features"])}
        earlier = []
        future_paths = set()
        for track in part["tracks"]:
            for offset, frame in enumerate(track["frames"]):
                position = track["start"] + offset
                if frame["frame_index"] >= 9:
                    changed["arrays"]["heatmap"][position] = .01
                    changed["arrays"]["heatmap"][position, 3, 60] = .99
                    changed["arrays"]["argmax_xy"][position] = [60 / 64, 3 / 64]
                    future_paths.add(frame["path"])
                else:
                    earlier.append(position)
        for position, path in enumerate(part["feature_index"]["image_paths"]):
            if path in future_paths:
                changed["features"][position] = -changed["features"][position]
        changed["features"].setflags(write=False)
        changed["arrays"]["heatmap"].setflags(write=False)
        changed["arrays"]["argmax_xy"].setflags(write=False)
        with patch.object(analysis, "_load_split", return_value=(changed, prediction, feature_report)):
            report = self.run_analysis("changed_future")
        np.testing.assert_array_equal(self.array(self.report, "candidate_xy")[earlier],
                                      self.array(report, "candidate_xy")[earlier])
        np.testing.assert_array_equal(self.array(self.report, "history_available"),
                                      self.array(report, "history_available"))

    def test_tampered_features_or_predictions_are_rejected(self):
        for kind in ("features", "predictions"):
            with self.subTest(kind=kind):
                fixture = make_official_fixture(self.root / ("tampered_" + kind))
                if kind == "features":
                    path = fixture["train_feature_cache"] / "scene_features.npy"
                else:
                    prediction = json.loads(fixture["train_prediction_report"].read_text())
                    path = Path(prediction["cache"]["path"]) / "heatmap.npy"
                values = np.load(path, mmap_mode="r+")
                values.flat[0] += .125
                values.flush()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.run_analysis("rejected_" + kind, fixture)

    def test_repeated_output_directory_is_rejected_without_changing_artifacts(self):
        original = {Path(record["path"]): record["sha256"]
                    for record in self.report["artifacts"].values()}
        with self.assertRaises(FileExistsError):
            self.run_analysis("cli_outputs")
        self.assertEqual(original, {path: file_sha256(path) for path in original})

    def test_wrong_plan_version_is_rejected(self):
        plan = self.root / "wrong_plan.md"
        plan.write_text("# Different analysis plan\n")
        with self.assertRaisesRegex(ValueError, "plan|version|protocol|Plan"):
            self.run_analysis("wrong_plan", plan=plan)


if __name__ == "__main__":
    unittest.main()
