"""Synthetic R3 learning, complete-track validation and diagnostic integration."""

import copy
import json
import unittest

import numpy as np
import torch

import test_train_temporal_joint as r2_fixtures
from train_temporal_joint import evaluate_model, forward_chunk, training_inputs, train_epoch
from stag.cache import file_sha256, load_cache
from stag.history_diagnostics import CONTROL_COLUMNS, HistoryDiagnosticWriter
from stag.history_model import HistoryValidityRefiner
from stag.temporal_data import iter_track_batches
from stag.temporal_model import detach_state


class HistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = r2_fixtures.JointTrainingTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        (self.r1, _, self.feature_index, self.features, self.prediction_index,
         self.arrays, self.tracks, self.sources) = training_inputs(
            self.fixture.feature_dir, self.fixture.feature_report,
            self.fixture.r1_path, self.fixture.review_path)
        torch.manual_seed(81)
        self.model = HistoryValidityRefiner(feature_channels=4, hidden_dim=8)

    def batches(self, tracks, clip_frames=5, batch_size=3):
        return iter_track_batches(tracks, self.arrays, self.feature_index,
                                  self.features, clip_frames, batch_size)

    def expose_history(self):
        # Identity output initialization otherwise hides state dependence.
        with torch.no_grad():
            self.model.heatmap_residual.weight.fill_(0.05)
            self.model.inout_residual.weight.fill_(0.1)
            self.model.correspondence_gain.fill_(0.2)

    def test_full_epoch_updates_both_heads_and_preserves_frozen_cache(self):
        paths = [*self.fixture.feature_dir.iterdir(),
                 *self.fixture.fixture["cache_dir"].iterdir()]
        original_files = {str(path): file_sha256(path) for path in paths}
        initial = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        training = train_epoch(self.model, optimizer, self.tracks["train"], self.arrays,
            self.feature_index, self.features, "cpu", clip_frames=5, batch_size=3,
            seed=0, lambda_state=0.1)
        counts = self.r1["split"]["partitions"]["train"]["counts"]
        self.assertEqual(training["valid_frames"], counts["person_frames"])
        self.assertEqual(training["inframe_frames"],
                         self.r1["baseline"]["train"]["standard"]["localization_frames"])
        self.assertTrue(np.isfinite(training["total_loss"]))
        changed = {name: not torch.equal(initial[name], value)
                   for name, value in self.model.named_parameters()}
        for prefix in ("heatmap_residual", "inout_residual", "history_gate"):
            self.assertTrue(any(value for name, value in changed.items() if name.startswith(prefix)))
        self.assertEqual(original_files, {str(path): file_sha256(path) for path in paths})
        self.assertFalse(self.features.flags.writeable)
        self.assertTrue(all(not array.flags.writeable for array in self.arrays.values()))

    def test_full_validation_covers_manifest_once_and_is_chunk_independent(self):
        self.expose_history()
        output = []
        for clip_frames, batch_size in ((4, 2), (5, 3)):
            result = evaluate_model(self.model, self.tracks["validation"], self.arrays,
                self.feature_index, self.features, self.root / f"validation_{clip_frames}",
                self.prediction_index["person_frames"], "cpu", clip_frames, batch_size,
                {"sources": self.sources, "model": "synthetic_R3"})
            index, arrays = load_cache(result["cache"]["path"])
            expected = self.r1["split"]["partitions"]["validation"]["counts"]
            self.assertEqual(index["person_frames"], expected["person_frames"])
            self.assertEqual(index["track_count"], expected["tracks"])
            self.assertTrue(index["checks"]["all_validation_frames_written_once"])
            self.assertEqual({track["track_id"] for track in index["tracks"]},
                             {track["track_id"] for track in self.tracks["validation"]})
            for source, compact in zip(self.tracks["validation"], index["tracks"]):
                self.assertEqual((compact["source_global_start"], compact["source_global_stop"]),
                                 (source["start"], source["stop"]))
                self.assertEqual(compact["stop"] - compact["start"], len(source["frames"]))
            self.assertTrue(np.isfinite(arrays["heatmap"]).all())
            output.append({name: np.array(values) for name, values in arrays.items()})
        for name in output[0]:
            np.testing.assert_allclose(output[0][name], output[1][name], atol=1e-6, rtol=1e-5)

    def test_checkpoint_and_predicted_memory_resume_the_same_next_frame(self):
        self.expose_history()
        self.model.eval()
        batch = next(self.batches(self.tracks["validation"]))
        with torch.no_grad():
            _, _, history = forward_chunk(self.model, batch, "cpu", None)
        path = self.root / "history_checkpoint.pt"
        torch.save({"model": self.model.state_dict(), "history": detach_state(history)}, path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored = HistoryValidityRefiner(feature_channels=4, hidden_dim=8)
        restored.load_state_dict(checkpoint["model"])
        restored.eval()
        self.assertTrue(any(name.startswith("history_gate") for name in checkpoint["model"]))
        self.assertEqual(checkpoint["history"]["available"].dtype, torch.bool)
        inputs = (batch["features"][:, 0], batch["raw_heatmap"][:, 0],
                  batch["raw_inout"][:, 0], batch["bboxes"][:, 0])
        with torch.no_grad():
            expected = self.model.step_with_diagnostics(*inputs, history)
            actual = restored.step_with_diagnostics(*inputs, checkpoint["history"])
        for position in (0, 1):
            torch.testing.assert_close(actual[position], expected[position], atol=0, rtol=0)
        for key in expected[2]:
            torch.testing.assert_close(actual[2][key], expected[2][key], atol=0, rtol=0)
        for key in expected[3]:
            torch.testing.assert_close(actual[3][key], expected[3][key], atol=0, rtol=0)

    def test_changed_labels_do_not_change_observations_resets_or_predicted_history(self):
        self.expose_history()
        self.model.eval()
        altered = copy.deepcopy(self.tracks["validation"])
        for track in altered:
            for frame in track["frames"]:
                frame["inout"] = 1 - frame["inout"]
                frame["gazex_norm"] = [0.73] if frame["inout"] else [-1.0]
                frame["gazey_norm"] = [0.18] if frame["inout"] else [-1.0]
        original_state, altered_state = None, None
        for original, changed in zip(self.batches(self.tracks["validation"]), self.batches(altered)):
            self.assertFalse(torch.equal(original["target_inout"], changed["target_inout"]))
            for key in ("features", "raw_heatmap", "raw_inout", "bboxes", "valid", "reset", "global_indices"):
                torch.testing.assert_close(original[key], changed[key], atol=0, rtol=0)
            with torch.no_grad():
                original_maps, original_scores, original_state = forward_chunk(
                    self.model, original, "cpu", original_state)
                changed_maps, changed_scores, altered_state = forward_chunk(
                    self.model, changed, "cpu", altered_state)
            torch.testing.assert_close(original_maps, changed_maps, atol=0, rtol=0)
            torch.testing.assert_close(original_scores, changed_scores, atol=0, rtol=0)
            for key in original_state:
                torch.testing.assert_close(original_state[key], altered_state[key], atol=0, rtol=0)

    def test_diagnostics_map_to_each_valid_source_frame_and_exclude_padding(self):
        state, records, padding_seen = None, {}, False
        output = self.root / "diagnostics"
        metadata = {"sources": self.sources, "model": "synthetic_R3"}
        writer = HistoryDiagnosticWriter(output, self.tracks["validation"],
                                         self.prediction_index["person_frames"], metadata)
        self.model.eval()
        with torch.no_grad():
            for batch in self.batches(self.tracks["validation"]):
                for step in range(batch["valid"].shape[1]):
                    _, _, state, diagnostics = self.model.step_with_diagnostics(
                        batch["features"][:, step], batch["raw_heatmap"][:, step],
                        batch["raw_inout"][:, step], batch["bboxes"][:, step], state,
                        batch["reset"][:, step])
                    valid = batch["valid"][:, step]
                    writer.write(batch["global_indices"][:, step], valid, diagnostics)
                    padding_seen |= not bool(valid.all())
                    for slot in valid.nonzero().flatten().tolist():
                        source = int(batch["global_indices"][slot, step])
                        self.assertNotIn(source, records)
                        self.assertGreaterEqual(source, 0)
                        records[source] = {key: value[slot].clone() for key, value in diagnostics.items()}
                        self.assertTrue(torch.isfinite(diagnostics["correspondence_xy"][slot]).all())
                        if batch["reset"][slot, step]:
                            self.assertFalse(bool(diagnostics["history_available"][slot]))
                            self.assertEqual(float(diagnostics["history_validity"][slot]), 0)
        expected = {index for track in self.tracks["validation"]
                    for index in range(track["start"], track["stop"])}
        self.assertEqual(set(records), expected)
        self.assertTrue(padding_seen)
        self.assertNotIn(-1, records)
        manifest = writer.finalize()
        index = json.loads((output / "index.json").read_text())
        self.assertTrue(manifest["passed"])
        self.assertEqual(index["metadata"], metadata)
        self.assertEqual(index["control_columns"], list(CONTROL_COLUMNS))
        self.assertEqual(manifest["index_sha256"], file_sha256(output / "index.json"))
        sources = np.load(output / "source_global_indices.npy", mmap_mode="r")
        controls = np.load(output / "controls.npy", mmap_mode="r")
        xy = np.load(output / "correspondence_xy.npy", mmap_mode="r")
        self.assertEqual(set(sources.tolist()), expected)
        self.assertEqual(controls.shape, (len(expected), 8))
        self.assertEqual(controls.dtype, np.float32)
        self.assertEqual(xy.shape, (len(expected), 2))
        for position, source in enumerate(sources):
            np.testing.assert_array_equal(controls[position],
                [records[int(source)][name].item() for name in CONTROL_COLUMNS])
            np.testing.assert_array_equal(xy[position], records[int(source)]["correspondence_xy"].numpy())
        for original, compact in zip(self.tracks["validation"], index["tracks"]):
            self.assertEqual(compact["frame_indices"], [f["frame_index"] for f in original["frames"]])
            np.testing.assert_array_equal(sources[compact["start"]:compact["stop"]],
                                          np.arange(original["start"], original["stop"]))
        for item in manifest["files"].values():
            self.assertEqual(item["sha256"], file_sha256(output / item["path"]))

    def diagnostic_step(self):
        batch = next(self.batches(self.tracks["validation"]))
        with torch.no_grad():
            _, _, _, diagnostics = self.model.step_with_diagnostics(
                batch["features"][:, 0], batch["raw_heatmap"][:, 0],
                batch["raw_inout"][:, 0], batch["bboxes"][:, 0],
                reset=batch["reset"][:, 0])
        return batch["global_indices"][:, 0], batch["valid"][:, 0], diagnostics

    def test_diagnostic_writer_cannot_publish_incomplete_tracks(self):
        writer = HistoryDiagnosticWriter(self.root / "incomplete", self.tracks["validation"],
                                         self.prediction_index["person_frames"], {})
        writer.write(*self.diagnostic_step())
        with self.assertRaisesRegex(ValueError, "missing source frames"):
            writer.finalize()
        self.assertFalse((self.root / "incomplete" / "index.json").exists())

    def test_diagnostic_writer_rejects_repeated_frames_within_or_across_steps(self):
        source, valid, diagnostics = self.diagnostic_step()
        writer = HistoryDiagnosticWriter(self.root / "repeated", self.tracks["validation"],
                                         self.prediction_index["person_frames"], {})
        writer.write(source, valid, diagnostics)
        with self.assertRaisesRegex(ValueError, "written twice"):
            writer.write(source, valid, diagnostics)
        fresh = HistoryDiagnosticWriter(self.root / "same_step", self.tracks["validation"],
                                        self.prediction_index["person_frames"], {})
        repeated = torch.tensor([int(source[0]), int(source[0])])
        with self.assertRaisesRegex(ValueError, "written twice"):
            fresh.write(repeated, torch.ones(2, dtype=torch.bool),
                        {name: value[[0, 0]] for name, value in diagnostics.items()})

    def test_diagnostic_writer_rejects_cache_or_partition_misalignment(self):
        writer = HistoryDiagnosticWriter(self.root / "misaligned", self.tracks["validation"],
                                         self.prediction_index["person_frames"], {})
        _, _, diagnostics = self.diagnostic_step()
        one = {name: value[:1] for name, value in diagnostics.items()}
        for source in (-1, self.prediction_index["person_frames"], self.tracks["train"][0]["start"]):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "outside"):
                writer.write(np.array([source]), np.array([True]), one)


if __name__ == "__main__":
    unittest.main()
