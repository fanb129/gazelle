"""Synthetic complete-track R4 prediction and same-forward diagnostic checks."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

import test_train_temporal_joint as r2_fixtures
from train_temporal_joint import evaluate_model, training_inputs
from stag.cache import file_sha256, load_cache
from stag.compact_validation import evaluate_compact_model
from stag.history_diagnostics import CONTROL_COLUMNS
from stag.history_model import HistoryValidityRefiner
from stag.temporal_data import iter_track_batches
from stag.temporal_model import CausalJointRefiner


class CountingHistoryRefiner(HistoryValidityRefiner):
    def __init__(self):
        super().__init__(4, 8)
        self.forward_calls = 0

    def step_with_diagnostics(self, *args, **kwargs):
        self.forward_calls += 1
        return super().step_with_diagnostics(*args, **kwargs)


class CompactValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = r2_fixtures.JointTrainingTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        (self.r1, _, self.feature_index, self.features, self.prediction_index,
         self.arrays, self.tracks, self.sources) = training_inputs(
            self.fixture.feature_dir, self.fixture.feature_report,
            self.fixture.r1_path, self.fixture.review_path)

    def run_validation(self, model, name, clip_frames=5, batch_size=3, tracks=None):
        return evaluate_compact_model(model, self.tracks["validation"] if tracks is None else tracks,
            self.arrays, self.feature_index, self.features, self.root / name,
            self.prediction_index["person_frames"], "cpu", clip_frames, batch_size,
            {"sources": self.sources, "model": "synthetic_only"})

    def expose_history(self, model):
        with torch.no_grad():
            model.heatmap_residual.weight.fill_(0.05)
            model.inout_residual.weight.fill_(0.1)
            if isinstance(model, HistoryValidityRefiner):
                model.correspondence_gain.fill_(0.2)

    def test_both_models_match_existing_validation_outputs_and_metric_definitions(self):
        for model_id, cls in (("gru", CausalJointRefiner), ("history", HistoryValidityRefiner)):
            with self.subTest(model=model_id):
                torch.manual_seed(51)
                model = cls(4, 8)
                self.expose_history(model)
                expected = evaluate_model(model, self.tracks["validation"], self.arrays,
                    self.feature_index, self.features, self.root / f"old_{model_id}",
                    self.prediction_index["person_frames"], "cpu", 5, 3, {})
                actual = self.run_validation(model, f"new_{model_id}")
                for family in ("standard", "temporal", "state"):
                    self.assertEqual(actual[family], expected[family])
                old_index, old_arrays = load_cache(expected["cache"]["path"])
                index, arrays = load_cache(actual["cache"]["path"])
                for name in arrays:
                    np.testing.assert_array_equal(arrays[name], old_arrays[name])
                self.assertEqual(index["tracks"], old_index["tracks"])
                self.assertEqual(index["task"], "R4_joint_validation")
                if model_id == "gru":
                    self.assertIsNone(actual["diagnostics"])
                    self.assertFalse((self.root / f"new_{model_id}" / "history_diagnostics").exists())
                else:
                    self.assertTrue(actual["diagnostics"]["passed"])

    def test_history_diagnostics_are_emitted_by_one_prediction_forward_per_step(self):
        model = CountingHistoryRefiner()
        self.expose_history(model)
        expected_calls = sum(batch["valid"].shape[1] for batch in iter_track_batches(
            self.tracks["validation"], self.arrays, self.feature_index, self.features, 5, 3))
        result = self.run_validation(model, "one_forward")
        self.assertEqual(model.forward_calls, expected_calls)
        diag_dir = Path(result["diagnostics"]["path"])
        index = json.loads((diag_dir / "index.json").read_text())
        self.assertEqual(index["control_columns"], list(CONTROL_COLUMNS))
        source = np.load(diag_dir / "source_global_indices.npy")
        controls = np.load(diag_dir / "controls.npy")
        expected_sources = [position for track in self.tracks["validation"]
                            for position in range(track["start"], track["stop"])]
        np.testing.assert_array_equal(source, expected_sources)
        self.assertEqual(controls.shape, (len(expected_sources), 8))
        self.assertNotIn(-1, source)
        # Compare controls against an independent chronological replay, not a
        # second evaluation hidden inside the production function.
        records, state = {}, None
        with torch.no_grad():
            for batch in iter_track_batches(self.tracks["validation"], self.arrays,
                    self.feature_index, self.features, 5, 3):
                for step in range(batch["valid"].shape[1]):
                    _, _, state, diag = model.step_with_diagnostics(
                        batch["features"][:, step], batch["raw_heatmap"][:, step],
                        batch["raw_inout"][:, step], batch["bboxes"][:, step], state,
                        batch["reset"][:, step])
                    for slot in batch["valid"][:, step].nonzero().flatten():
                        position = int(batch["global_indices"][slot, step])
                        records[position] = [diag[name][slot].item() for name in CONTROL_COLUMNS]
        np.testing.assert_array_equal(controls, np.array([records[int(i)] for i in source], dtype=np.float32))

    def test_changing_chunks_preserves_predictions_and_aligned_history_controls(self):
        torch.manual_seed(67)
        model = HistoryValidityRefiner(4, 8)
        self.expose_history(model)
        results = [self.run_validation(model, f"chunks_{size}", size, batch)
                   for size, batch in ((4, 2), (5, 3))]
        caches = [load_cache(result["cache"]["path"])[1] for result in results]
        for name in caches[0]:
            np.testing.assert_allclose(caches[0][name], caches[1][name], atol=1e-6, rtol=1e-5)
        for name in ("controls", "correspondence_xy", "source_global_indices"):
            values = [np.load(Path(result["diagnostics"]["path"]) / f"{name}.npy") for result in results]
            np.testing.assert_allclose(values[0], values[1], atol=1e-6, rtol=1e-5)

    def test_ground_truth_changes_scoring_but_not_predictions_or_history_controls(self):
        model = HistoryValidityRefiner(4, 8)
        self.expose_history(model)
        altered = copy.deepcopy(self.tracks["validation"])
        for track in altered:
            for frame in track["frames"]:
                frame["inout"] = 0
                frame["gazex_norm"] = [-1.]
                frame["gazey_norm"] = [-1.]
        original = self.run_validation(model, "labels_original")
        changed = self.run_validation(model, "labels_changed", tracks=altered)
        self.assertEqual(changed["standard"], {"auc": None, "l2": None, "inout_ap": None,
            "person_frames": original["standard"]["person_frames"], "localization_frames": 0})
        one, two = (load_cache(r["cache"]["path"])[1] for r in (original, changed))
        for name in one:
            np.testing.assert_array_equal(one[name], two[name])
        for name in ("controls", "correspondence_xy"):
            one = np.load(Path(original["diagnostics"]["path"]) / f"{name}.npy")
            two = np.load(Path(changed["diagnostics"]["path"]) / f"{name}.npy")
            np.testing.assert_array_equal(one, two)

    def test_exported_artifacts_have_matching_hashes_without_changing_sources(self):
        source_paths = [*self.fixture.feature_dir.iterdir(), *self.fixture.fixture["cache_dir"].iterdir()]
        before = {str(path): file_sha256(path) for path in source_paths}
        model = HistoryValidityRefiner(4, 8)
        result = self.run_validation(model, "artifacts")
        for cache in (result["cache"], result["diagnostics"]):
            directory = Path(cache["path"])
            self.assertEqual(cache["index_sha256"], file_sha256(directory / "index.json"))
            for record in cache["files"].values():
                self.assertEqual(record["sha256"], file_sha256(directory / record["path"]))
        self.assertEqual(result["details"]["sha256"], file_sha256(result["details"]["path"]))
        self.assertEqual(before, {str(path): file_sha256(path) for path in source_paths})
        self.assertFalse(self.features.flags.writeable)
        self.assertTrue(all(not values.flags.writeable for values in self.arrays.values()))


if __name__ == "__main__":
    unittest.main()
