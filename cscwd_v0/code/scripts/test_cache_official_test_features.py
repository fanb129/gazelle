"""Synthetic test-cache stream, frozen FP32 agreement and source identity checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from cache_official_test_features import (cache_test_features, probe_test_batch,
    test_feature_sources as feature_sources, test_image_records as image_records)
from test_official_data import make_official_fixture
from stag.cache import file_sha256, load_cache, write_json
from stag.feature_cache import load_feature_cache


class SyntheticBackbone(nn.Module):
    out_indices = [11]

    def forward(self, images):
        values = images[:, :1, :2, :2] / 10
        return [values.repeat(1, 4, 1, 1)]


class SyntheticFrozenModel(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        self.backbone = SyntheticBackbone()
        self.register_buffer("maps", torch.from_numpy(np.array(arrays["heatmap"])))
        self.register_buffer("scores", torch.from_numpy(np.array(arrays["inout"])))
        self.eval()

    def forward_from_features(self, inputs, features):
        maps, scores = [], []
        for image, boxes in zip(inputs["images"], inputs["bboxes"]):
            frame = int(image[0, 0, 0])
            positions = torch.tensor([frame + person * 12 for person in range(len(boxes))])
            maps.append(self.maps[positions])
            scores.append(self.scores[positions])
        return {"heatmap": maps, "inout": scores}


class OfficialTestFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = make_official_fixture(self.root / "fixture")
        run = json.loads(self.fixture["test_prediction_report"].read_text())
        self.index, self.arrays = load_cache(run["cache"]["path"])
        self.records = image_records(self.index["tracks"])
        self.images = torch.stack([torch.full((3, 2, 2), float(i)) for i in range(12)])
        self.loader = [(self.images[:5], self.records[:5]), (self.images[5:], self.records[5:])]
        self.contract = json.loads(self.fixture["train_feature_report"].read_text())["contract"]
        self.model = SyntheticFrozenModel(self.arrays)

    def test_all_test_images_cached_with_original_predictions_retained(self):
        prediction_dir = Path(json.loads(self.fixture["test_prediction_report"].read_text())["cache"]["path"])
        before = {p.name: file_sha256(p) for p in prediction_dir.iterdir()}
        _, _, _, _, sources = feature_sources(
            self.fixture["test_prediction_report"], self.fixture["train_feature_report"])
        destination = self.root / "official_test_features"
        result = cache_test_features(self.model, self.loader, self.arrays,
            [r["path"] for r in self.records], self.contract, sources, destination, "cpu")
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["probe"]["images"], 5)
        self.assertEqual(result["probe"]["person_frames"], 10)
        self.assertEqual(result["probe"]["heatmap_max_abs_error"], 0)
        index, values = load_feature_cache(destination)
        self.assertEqual(index["task"], "R4_official_test_features")
        self.assertEqual(index["official_split"], "test")
        self.assertEqual(index["sources"], sources)
        self.assertEqual(values.shape, (12, 4, 2, 2))
        self.assertEqual(values.dtype, np.float16)
        self.assertFalse(values.flags.writeable)
        expected = self.model.backbone(self.images)[-1].numpy().astype(np.float16)
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(file_sha256(destination / "index.json"), result["cache"]["index_sha256"])
        self.assertEqual(before, {p.name: file_sha256(p) for p in prediction_dir.iterdir()})

    def test_probe_compares_all_people_with_source_global_indices(self):
        images, records = self.loader[0]
        result = probe_test_batch(self.model, images, records, self.arrays, self.contract, "cpu")
        self.assertTrue(result["passed"])
        self.assertEqual(result["person_frames"], 2 * len(records))
        self.assertEqual(result["argmax_changed_people"], 0)

    def test_wrong_native_contract_is_rejected_before_creating_cache(self):
        contract = {**self.contract, "grid": [1, 1]}
        destination = self.root / "wrong_contract"
        with self.assertRaisesRegex(ValueError, "native feature contract"):
            cache_test_features(self.model, self.loader, self.arrays,
                [r["path"] for r in self.records], contract, {}, destination, "cpu")
        self.assertFalse(destination.exists())

    def test_prediction_disagreement_is_rejected_before_creating_cache(self):
        self.model.scores[0] = .4
        destination = self.root / "wrong_predictions"
        with self.assertRaisesRegex(ValueError, "probe failed"):
            cache_test_features(self.model, self.loader, self.arrays,
                [r["path"] for r in self.records], self.contract, {}, destination, "cpu")
        self.assertFalse(destination.exists())

    def test_sources_require_same_checkpoint_as_train_contract(self):
        path = self.fixture["train_feature_report"]
        report = json.loads(path.read_text())
        report["sources"]["checkpoint_sha256"] = "different"
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "differs from the retained train features"):
            feature_sources(self.fixture["test_prediction_report"], path)

    def test_cli_exposes_complete_official_test_cache_arguments(self):
        script = Path(__file__).with_name("cache_official_test_features.py")
        process = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        for argument in ("--prediction-report", "--train-feature-report", "--checkpoint", "--output-dir"):
            self.assertIn(argument, process.stdout)


if __name__ == "__main__":
    unittest.main()
