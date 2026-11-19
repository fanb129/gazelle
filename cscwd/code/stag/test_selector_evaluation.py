"""Whole-map candidate selection, original metrics and lossless output records."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_temporal import verify_completed_cache
from stag.cache import argmax_points, file_sha256, load_cache
from stag.metrics import TemporalConfig, evaluate_predictions, evaluate_temporal
from stag.selector_evaluation import evaluate_selector_choices
from stag.state_metrics import evaluate_state_temporal


NAMES = ["current", "current_grid", "ema_075", "grid_ema_075", "transport_previous"]


def make_bank(count=12):
    rng = np.random.default_rng(53)
    maps = rng.uniform(0, .1, size=(count, 5, 64, 64)).astype(np.float32)
    for row in range(count):
        for column in range(5):
            maps[row, column, 8 + column * 5, 12 + row % 16] = .2 + column
    labels = np.asarray(([0, 1, 1, 1, 0, 0] * (count // 6 + 1))[:count])
    scores = np.where(labels, .8, .3).astype(np.float32)
    middle = count // 2
    tracks = []
    for person, (start, stop) in enumerate(((0, middle), (middle, count))):
        tracks.append({"track_id": f"test/video/clip/person{person}", "start": start, "stop": stop,
                       "frames": [{"frame_index": offset + 1,
                                   "path": f"video/clip/{offset:06d}.jpg",
                                   "inout": int(labels[row]), "gazex_norm": [.3], "gazey_norm": [.4]}
                                  for offset, row in enumerate(range(start, stop))]})
    xy = argmax_points(maps.reshape(-1, 64, 64)).reshape(count, 5, 2)
    observations = np.zeros((count, 23), dtype=np.float32)
    history = np.ones(count, dtype=bool)
    history[[0, middle]] = False
    for value in (maps, scores, xy, observations, history):
        value.setflags(write=False)
    return {"index": {"tracks": tracks, "candidate_names": NAMES}, "heatmaps": maps,
            "observations": observations, "inout": scores, "candidate_xy": xy,
            "history_available": history}


class SelectorEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank = make_bank()
        self.choices = np.arange(12, dtype=np.int64) % 5
        self.metadata = {"evaluation_split": "test", "epoch": 2, "source": "synthetic_bank"}

    def evaluate(self, name="evaluation", choices=None, logits=None, bank=None):
        return evaluate_selector_choices(bank or self.bank,
            self.choices if choices is None else choices, self.root / name, self.metadata, logits)

    def test_selection_preserves_complete_maps_raw_scores_and_original_row_order(self):
        before = {name: hashlib.sha256(value.tobytes()).hexdigest()
                  for name, value in self.bank.items() if isinstance(value, np.ndarray)}
        result = self.evaluate()
        index, arrays = load_cache(result["cache"]["path"])
        selected = self.bank["heatmaps"][np.arange(12), self.choices]
        np.testing.assert_array_equal(arrays["heatmap"], selected)
        np.testing.assert_array_equal(arrays["inout"], self.bank["inout"])
        np.testing.assert_array_equal(arrays["argmax_xy"], self.bank["candidate_xy"][np.arange(12), self.choices])
        self.assertEqual(index["tracks"], self.bank["index"]["tracks"])
        self.assertEqual(index["metadata"], self.metadata)
        self.assertEqual(index["task"], "R4_selector_evaluation")
        self.assertTrue(all(result["cache"]["checks"].values()))
        self.assertEqual(before, {name: hashlib.sha256(self.bank[name].tobytes()).hexdigest() for name in before})
        self.assertTrue(all(not a.flags.writeable for a in arrays.values()))

    def test_original_standard_temporal_and_state_evaluators_match_exactly(self):
        result = self.evaluate()
        selected = self.bank["heatmaps"][np.arange(12), self.choices]
        points = argmax_points(selected)
        tracks, scores = self.bank["index"]["tracks"], self.bank["inout"]
        config = TemporalConfig()
        self.assertEqual(result["standard"], evaluate_predictions(tracks, selected, scores))
        temporal = evaluate_temporal(tracks, points, scores, config)
        state = evaluate_state_temporal(tracks, scores, config)
        self.assertEqual(result["temporal"], temporal["summary"])
        self.assertEqual(result["state"], state["summary"])
        details = json.loads(Path(result["details"]["path"]).read_text())
        self.assertEqual(details["joint"], {k: temporal[k] for k in ("tracks", "events")})
        self.assertEqual(details["state"], {k: state[k] for k in ("tracks", "events")})

    def test_cache_files_hashes_value_digests_and_original_verifier_are_valid(self):
        result = self.evaluate()
        verify_completed_cache(result["cache"]["path"], {"passed": True, "cache": result["cache"]})
        index, arrays = load_cache(result["cache"]["path"])
        self.assertEqual(index["files"], result["cache"]["files"])
        for name, item in result["cache"]["files"].items():
            path = Path(result["cache"]["path"]) / item["path"]
            self.assertEqual(item["sha256"], file_sha256(path))
            self.assertEqual(item["values_sha256"], hashlib.sha256(arrays[name].tobytes()).hexdigest())
            self.assertEqual(item["shape"], list(arrays[name].shape))
            self.assertEqual(item["dtype"], "float32")
            self.assertEqual(item["size_bytes"], path.stat().st_size)
        self.assertEqual(result["details"]["sha256"], file_sha256(result["details"]["path"]))
        self.assertIsNone(result["diagnostics"])

    def test_same_forward_logits_and_choices_are_saved_without_recomputation(self):
        logits = np.linspace(-2, 2, 12, dtype=np.float32)
        result = self.evaluate(logits=logits)
        selection = result["selection"]
        for name, expected in (("choices", self.choices), ("logits", logits)):
            record = selection[name]
            np.testing.assert_array_equal(np.load(record["path"], allow_pickle=False), expected)
            self.assertEqual(record["sha256"], file_sha256(record["path"]))
        self.assertEqual(selection["choice_counts"], {name: int(np.sum(self.choices == column))
                         for column, name in enumerate(NAMES)})
        index, _ = load_cache(result["cache"]["path"])
        self.assertEqual(index["selection"], selection)

    def test_constant_baseline_needs_no_logits_and_uses_full_anchor_map(self):
        choices = np.ones(12, dtype=np.int8)
        result = self.evaluate(choices=choices)
        _, arrays = load_cache(result["cache"]["path"])
        np.testing.assert_array_equal(arrays["heatmap"], self.bank["heatmaps"][:, 1])
        self.assertEqual(result["selection"]["choice_counts"]["current_grid"], 12)
        self.assertIsNone(result["selection"]["logits"])
        self.assertFalse((Path(result["cache"]["path"]) / "selector_logits.npy").exists())

    def test_chunk_boundary_covers_every_frame_and_preserves_first_argmax_on_ties(self):
        bank = make_bank(131)
        maps = np.array(bank["heatmaps"])
        maps[128, 4] = .25
        xy = np.array(bank["candidate_xy"])
        xy[128, 4] = [0., 0.]
        maps.setflags(write=False)
        xy.setflags(write=False)
        bank = {**bank, "heatmaps": maps, "candidate_xy": xy}
        choices = np.full(131, 4, dtype=np.int64)
        result = self.evaluate(bank=bank, choices=choices)
        _, arrays = load_cache(result["cache"]["path"])
        np.testing.assert_array_equal(arrays["heatmap"], maps[:, 4])
        np.testing.assert_array_equal(arrays["argmax_xy"][128], [0., 0.])
        self.assertEqual(result["cache"]["person_frames"], 131)

    def test_bad_choice_or_logit_contract_is_rejected_before_creating_output(self):
        for choices in (np.zeros(12), np.zeros(11, dtype=int), np.full(12, -1), np.full(12, 5)):
            with self.subTest(choices=choices.dtype):
                with self.assertRaisesRegex(ValueError, "choices"):
                    self.evaluate(choices=choices)
        for logits in (np.ones(11), np.full(12, np.nan)):
            with self.subTest(logits=logits.shape):
                with self.assertRaisesRegex(ValueError, "logits"):
                    self.evaluate(logits=logits)
        self.assertFalse((self.root / "evaluation").exists())

    def test_mismatching_bank_coordinates_and_repeated_output_are_rejected(self):
        bad_xy = np.array(self.bank["candidate_xy"])
        bad_xy[0, 0] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "selected_coordinates_match_bank"):
            self.evaluate(name="bad", bank={**self.bank, "candidate_xy": bad_xy})
        result = self.evaluate()
        before = file_sha256(Path(result["cache"]["path"]) / "index.json")
        with self.assertRaises(FileExistsError):
            self.evaluate()
        self.assertEqual(before, file_sha256(Path(result["cache"]["path"]) / "index.json"))


if __name__ == "__main__":
    unittest.main()
