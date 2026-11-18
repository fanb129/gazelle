"""Synthetic frozen train replay, causal history and source-row checks."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code/scripts"))

from test_official_data import make_official_fixture
from stag.cache import file_sha256
from stag.calibration_rollout import cache_frozen_train_scores
from stag.history_model import HistoryValidityRefiner
from stag.official_data import _load_split
from stag.temporal_model import CausalJointRefiner


class RecurrentProbe(nn.Module):
    """A scalar recurrence whose resets can be checked independently by hand."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.75))
        self.forward_contract = []

    def step(self, features, raw_heatmap, raw_inout, bboxes, state=None, reset=None):
        previous = torch.zeros_like(raw_inout) if state is None else state["inout"]
        previous = previous.masked_fill(reset, 0)
        score = torch.sigmoid(raw_inout + self.gain * previous)
        self.forward_contract.append((self.training, torch.is_grad_enabled(), self.gain.requires_grad))
        return raw_heatmap, score, {"inout": score}


class CalibrationRolloutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fixture = make_official_fixture(self.root)
        self.part, _, _ = _load_split("train", fixture["train_feature_cache"],
            fixture["train_feature_report"], fixture["train_prediction_report"])
        self.metadata = {"source_prediction_index": {"path": str(self.root / "train_predictions/index.json"),
            "sha256": file_sha256(self.root / "train_predictions/index.json")}}

    def run_replay(self, model, name, part=None, clip_frames=3, batch_size=2):
        result = cache_frozen_train_scores(model, self.part if part is None else part,
            self.root / name, "cpu", clip_frames, batch_size, self.metadata)
        return result, np.load(Path(result["path"]) / "inout.npy")

    def test_replay_uses_own_past_predictions_and_resets_at_person_or_frame_gap(self):
        part = {**self.part, "tracks": copy.deepcopy(self.part["tracks"])}
        for frame in part["tracks"][0]["frames"][6:]:
            frame["frame_index"] += 10
        part["tracks"].reverse()  # Iterator order differs from original cache rows.
        model = RecurrentProbe()
        result, scores = self.run_replay(model, "probe", part, clip_frames=2, batch_size=1)
        expected = np.empty_like(scores)
        for track in part["tracks"]:
            previous, previous_frame = torch.tensor(0.), None
            for offset, frame in enumerate(track["frames"]):
                if previous_frame is None or frame["frame_index"] != previous_frame + 1:
                    previous = torch.tensor(0.)
                row = track["start"] + offset
                previous = torch.sigmoid(torch.tensor(float(part["arrays"]["inout"][row])) + .75 * previous)
                expected[row] = previous.item()
                previous_frame = frame["frame_index"]
        np.testing.assert_array_equal(scores, expected)
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(contract == (False, False, False) for contract in model.forward_contract))

    def test_chunk_sizes_preserve_both_real_models_scores(self):
        for name, cls in (("gru", CausalJointRefiner), ("history", HistoryValidityRefiner)):
            with self.subTest(model=name):
                torch.manual_seed(17)
                model = cls(4, 8)
                with torch.no_grad():
                    model.inout_residual.weight.fill_(.2)
                    model.heatmap_residual.weight.fill_(.05)
                _, short = self.run_replay(model, name + "_short", clip_frames=2, batch_size=2)
                _, long = self.run_replay(model, name + "_long", clip_frames=12, batch_size=2)
                np.testing.assert_array_equal(short, long)

    def test_ground_truth_changes_do_not_change_any_train_score(self):
        changed = {**self.part, "tracks": copy.deepcopy(self.part["tracks"])}
        for track in changed["tracks"]:
            for frame in track["frames"]:
                frame.update(inout=1-frame["inout"], gazex_norm=[.9], gazey_norm=[.1])
        torch.manual_seed(19)
        model = HistoryValidityRefiner(4, 8)
        with torch.no_grad():
            model.inout_residual.weight.fill_(.3)
        _, before = self.run_replay(model, "before_labels")
        _, after = self.run_replay(model, "after_labels", changed)
        np.testing.assert_array_equal(before, after)

    def test_score_cache_hashes_source_order_and_model_weights_remain_unchanged(self):
        source_paths = [*self.root.glob("train_features/*"), *self.root.glob("train_predictions/*")]
        before_sources = {str(path): file_sha256(path) for path in source_paths}
        model = RecurrentProbe()
        before_weights = {name: value.clone() for name, value in model.state_dict().items()}
        result, scores = self.run_replay(model, "artifacts")
        index_path = Path(result["path"]) / "index.json"
        index = json.loads(index_path.read_text())
        self.assertEqual(result["index_sha256"], file_sha256(index_path))
        self.assertEqual(result["files"]["inout"]["sha256"], file_sha256(Path(result["path"]) / "inout.npy"))
        self.assertEqual(index["metadata"], self.metadata)
        self.assertEqual(index["row_order"], "source_prediction_index_global_indices")
        self.assertEqual(index["person_frames"], len(scores))
        self.assertEqual(index["track_slices"], [{key: track[key] for key in ("track_id", "start", "stop")}
            for track in self.part["tracks"]])
        self.assertEqual({p.name for p in Path(result["path"]).iterdir()}, {"index.json", "inout.npy"})
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before_weights[name]))
        self.assertEqual(before_sources, {str(path): file_sha256(path) for path in source_paths})


if __name__ == "__main__":
    unittest.main()
