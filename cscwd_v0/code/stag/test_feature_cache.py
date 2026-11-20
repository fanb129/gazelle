"""Synthetic native-feature contracts and current-frame probe checks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cscwd/code/scripts"))
sys.path.insert(0, str(ROOT / "cscwd/code"))

from cache_frozen_features import probe_features, verify_probe
from stag.cache import file_sha256
from stag.feature_cache import (feature_contract, load_feature_cache, select_probe_records,
                                unique_image_records, write_feature_cache)


class FakeBackbone(nn.Module):
    out_indices = [2, 5, 8, 11]

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)

    def forward(self, images):
        last = self.projection(F.adaptive_avg_pool2d(images, (4, 4)))
        return [last * factor for factor in (0.25, 0.5, 0.75, 1.0)]


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()

    def forward_from_features(self, inputs, pyramid):
        maps, scores = [], []
        for index, boxes in enumerate(inputs["bboxes"]):
            image_map = torch.sigmoid(F.interpolate(pyramid[-1][index:index + 1].mean(1, keepdim=True),
                (64, 64), mode="bilinear", align_corners=False)).squeeze(1)
            maps.append(image_map.repeat(len(boxes), 1, 1))
            scores.append(image_map.mean().repeat(len(boxes)))
        return {"heatmap": maps, "inout": scores}


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(4)
        self.model = FakeModel().eval().requires_grad_(False)
        self.images = torch.rand(2, 3, 12, 12)
        self.records = [{"path": f"images/video/clip/{i}.jpg", "people": [
            {"track_id": "train/video/clip/p0", "global_index": i,
             "bbox_norm": [0.01, 0.01, 0.03, 0.03], "frame_index": i}]} for i in range(2)]
        with torch.no_grad():
            self.pyramid = self.model.backbone(self.images)
            out = self.model.forward_from_features(
                {"bboxes": [[r["people"][0]["bbox_norm"]] for r in self.records]}, self.pyramid)
        self.predictions = {"heatmap": torch.cat(out["heatmap"]).numpy(),
                            "inout": torch.cat(out["inout"]).numpy()}

    def test_fp16_cache_reopens_exact_quantized_values_with_original_image_order(self):
        raw = self.pyramid[-1].numpy()
        contract = feature_contract(self.pyramid[-1], 11)
        report = write_feature_cache([r["path"] for r in self.records], [raw[:1], raw[1:]],
                                     self.root / "cache", contract, {"origin": "synthetic"})
        index, values = load_feature_cache(self.root / "cache")
        np.testing.assert_array_equal(values, raw.astype(np.float16))
        self.assertFalse(values.flags.writeable)
        self.assertEqual(index["image_paths"], [r["path"] for r in self.records])
        self.assertEqual(report["index_sha256"], file_sha256(self.root / "cache/index.json"))
        self.assertEqual(report["files"]["scene_features"]["sha256"], file_sha256(self.root / "cache/scene_features.npy"))

    def test_incomplete_stream_does_not_publish_completed_index(self):
        with self.assertRaisesRegex(ValueError, "Incomplete feature cache"):
            write_feature_cache(["a", "b"], [self.pyramid[-1][:1].numpy()], self.root / "short",
                                 feature_contract(self.pyramid[-1], 11), {})
        self.assertFalse((self.root / "short/index.json").exists())

    def test_probe_measures_native_shape_tiny_heads_and_base_agreement(self):
        report = probe_features(self.model, [(self.images, self.records)], self.predictions,
                                 10, self.root / "future_cache", "cpu")
        self.assertTrue(report["passed"])
        self.assertEqual(report["contract"]["grid"], [4, 4])
        self.assertEqual(report["storage"]["estimated_feature_bytes"], 10 * 4 * 4 * 4 * 2)
        self.assertEqual(report["head_grid_coverage"]["heads_with_no_patch_center_inside"], 2)
        self.assertEqual(report["base_prediction_comparison"]["argmax_changed_people"], 0)
        self.assertGreater(report["fp16_quantization"]["max_abs_error"], 0)
        self.assertFalse((self.root / "future_cache").exists())

    def test_probe_rejects_changed_base_scores_and_trainable_model(self):
        bad = {**self.predictions, "inout": self.predictions["inout"] + 0.1}
        self.model.requires_grad_(True)
        report = probe_features(self.model, [(self.images, self.records)], bad,
                                 2, self.root / "future", "cpu")
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["fp32_inout_agrees_with_R1"])
        self.assertFalse(report["checks"]["model_all_parameters_frozen"])

    def test_probe_contract_requires_identical_sources_and_code(self):
        report = {"passed": True, "sources": {"data": "sha"}, "code_sha256": {"file": "sha"},
                  "contract": {"grid": [4, 4]}}
        self.assertEqual(verify_probe(report, report["sources"], report["code_sha256"]), report["contract"])
        with self.assertRaisesRegex(ValueError, "sources|identities|differ"):
            verify_probe(report, {"data": "changed"}, report["code_sha256"])

    def test_image_deduplication_and_probe_selection_do_not_use_gaze_labels(self):
        tracks = [{"track_id": f"train/video/clip/p{i}", "start": i * 2, "frames": [
            {"path": f"images/video/clip/{j}.jpg", "bbox_norm": [0.01, 0.01, 0.03, 0.03],
             "frame_index": j, "gazex_norm": [0.1], "inout": 1} for j in range(2)]} for i in range(2)]
        records = unique_image_records(tracks)
        changed = copy.deepcopy(tracks)
        for track in changed:
            for frame in track["frames"]:
                frame.update(gazex_norm=[-1], inout=0)
        self.assertEqual(records, unique_image_records(changed))
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0]["people"]), 2)
        selected = select_probe_records(records, [tracks[0]["track_id"]],
                    [{"track_id": tracks[0]["track_id"], "anchor_global_index": 1}], limit=1)
        self.assertEqual(selected[0]["path"], "images/video/clip/1.jpg")

    def test_offline_reader_does_not_import_torch(self):
        write_feature_cache([r["path"] for r in self.records], [self.pyramid[-1].numpy()],
                             self.root / "cache", feature_contract(self.pyramid[-1], 11), {})
        code = "import sys; sys.path.insert(0,sys.argv[1]); from stag.feature_cache import load_feature_cache; load_feature_cache(sys.argv[2]); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "cscwd/code"),
                                 str(self.root / "cache")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
