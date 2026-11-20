"""Synthetic official train/test cache identities, coverage and read-only loading."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from stag.cache import file_sha256, write_json, write_prediction_cache
from stag.feature_cache import write_feature_cache
from stag.official_data import official_training_inputs


INPUT_NAMES = ("train_feature_cache", "train_feature_report", "test_feature_cache",
               "test_feature_report", "train_prediction_report", "test_prediction_report")


def make_official_fixture(root):
    """Small distinct official splits; no R1 split or review file is created."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint, backbone = root / "epoch_7.pt", root / "backbone.pth"
    checkpoint.write_bytes(b"synthetic GazeSpot identity")
    backbone.write_bytes(b"synthetic backbone identity")
    contract = {"layer_index": 11, "channels": 4, "grid": [2, 2],
                "storage_dtype": "float16", "source_dtype": "torch.float32",
                "layout": "NCHW", "spatial_pooling": "none",
                "image_unit": "unique_current_RGB", "learned_projection_cached": False}
    identity = {"model": "gazelle_dinov3_vitb16_inout",
                "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
                "backbone_checkpoint": {"path": str(backbone), "sha256": file_sha256(backbone)},
                "protocol": {"modality": "RGB", "boxes": "T1_current_frame_raw_ground_truth",
                    "spatial_prior": "ggsf", "fusion": "sasa", "aux": False,
                    "precision": "float32; no_autocast; tf32_disabled", "temporal_context": "none",
                    "labels_used_in_forward": False, "all_parameters_frozen": True, "training": False}}
    fixture = {"checkpoint": checkpoint, "backbone": backbone, "data_path": root / "data"}
    rng = np.random.default_rng(4)
    for split in ("train", "test"):
        sequence_path = f"images/{split}/video/clip"
        tracks = []
        maps, scores = [], []
        for person in range(2):
            frames = []
            for frame_index in range(1, 13):
                inside = int(frame_index <= 6 or frame_index >= 10)
                x, y = (16, 16) if frame_index <= 3 else (44, 40)
                frames.append({"path": f"{sequence_path}/{frame_index:08d}.jpg",
                    "frame_index": frame_index, "person_id": f"p{person}",
                    "bbox_norm": [0.1 + person * .4, .1, .3 + person * .4, .4],
                    "gazex_norm": [x / 64] if inside else [-1.0],
                    "gazey_norm": [y / 64] if inside else [-1.0], "inout": inside})
                heatmap = np.zeros((64, 64), dtype=np.float32)
                heatmap[y, x] = .9
                maps.append(heatmap)
                scores.append(.8 if frame_index == 8 else .2 if frame_index == 2 else .9 if inside else .1)
            tracks.append({"track_id": f"{split}/video/clip/p{person}", "person_id": f"p{person}",
                           "frames": frames, "source_csv": f"annotations/{split}/video/clip/p{person}.txt"})
        sequences = [{"path": sequence_path, "width": 160, "height": 120, "tracks": tracks}]
        track_json = root / f"{split}_tracks.json"
        write_json(track_json, sequences)
        metadata = {**identity, "track_json": {"path": str(track_json), "sha256": file_sha256(track_json)}}
        prediction_dir = root / f"{split}_predictions"
        cache = write_prediction_cache(sequences, [(np.stack(maps), np.asarray(scores, dtype=np.float32))],
                                       prediction_dir, metadata)
        prediction_report = root / f"{split}_predictions.json"
        write_json(prediction_report, {"task": "T2", "passed": True, "cache": cache, **metadata})
        image_paths = sorted({frame["path"] for track in tracks for frame in track["frames"]})
        features = rng.normal(size=(len(image_paths), 4, 2, 2)).astype(np.float32)
        sources = {"prediction_index_sha256": cache["index_sha256"],
                   "checkpoint_sha256": identity["checkpoint"]["sha256"],
                   "backbone_checkpoint_sha256": identity["backbone_checkpoint"]["sha256"]}
        if split == "test":
            sources.update({"prediction_report_sha256": file_sha256(prediction_report),
                            "train_feature_report_sha256": file_sha256(fixture["train_feature_report"])})
        feature_dir = root / f"{split}_features"
        feature_cache = write_feature_cache(image_paths, [features], feature_dir, contract, sources)
        feature_report = root / f"{split}_features.json"
        write_json(feature_report, {"task": "R2_features" if split == "train" else "R4_official_test_features",
            "mode": "cache", "official_split": split, "passed": True, "contract": contract,
            "sources": sources, "cache": feature_cache,
            "runtime": {"torch": "synthetic", "transform": "synthetic"}})
        fixture[f"{split}_feature_cache"] = feature_dir
        fixture[f"{split}_feature_report"] = feature_report
        fixture[f"{split}_prediction_report"] = prediction_report
    return fixture


class OfficialDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = make_official_fixture(self.root)

    def load(self):
        return official_training_inputs(*(self.fixture[name] for name in INPUT_NAMES))

    def test_all_official_tracks_loaded_without_development_manifest(self):
        loaded = self.load()
        self.assertEqual(set(loaded), {"train", "test", "sources", "feature_contract"})
        for split in ("train", "test"):
            data = loaded[split]
            self.assertEqual(len(data["tracks"]), 2)
            self.assertEqual(data["prediction_index"]["person_frames"], 24)
            self.assertTrue(all(t["track_id"].startswith(split + "/") for t in data["tracks"]))
            self.assertFalse(data["features"].flags.writeable)
            self.assertTrue(all(not v.flags.writeable for v in data["arrays"].values()))
        self.assertFalse(list(self.root.glob('*split*')))
        self.assertFalse(list(self.root.glob('*review*')))
        json.dumps(loaded["sources"], allow_nan=False)

    def test_mismatched_checkpoint_identity_cannot_load(self):
        report_path = self.fixture["test_prediction_report"]
        report = json.loads(report_path.read_text())
        report["checkpoint"]["sha256"] = "different"
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "different sources"):
            self.load()

    def test_changed_feature_bytes_are_rejected(self):
        path = self.fixture["test_feature_cache"] / "scene_features.npy"
        values = np.load(path, mmap_mode="r+")
        values[0, 0, 0, 0] += 1
        values.flush()
        with self.assertRaisesRegex(ValueError, "feature file SHA256"):
            self.load()

    def test_wrong_split_prefix_is_rejected_even_with_updated_index_report(self):
        report_path = self.fixture["test_prediction_report"]
        report = json.loads(report_path.read_text())
        path = Path(report["cache"]["path"]) / "index.json"
        index = json.loads(path.read_text())
        index["tracks"][0]["track_id"] = "train/video/clip/p0"
        write_json(path, index)
        report["cache"]["index_sha256"] = file_sha256(path)
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "official test"):
            self.load()

    def test_changed_prediction_values_are_rejected(self):
        report = json.loads(self.fixture["test_prediction_report"].read_text())
        path = Path(report["cache"]["path"]) / "inout.npy"
        values = np.load(path, mmap_mode="r+")
        values[0] = .4
        values.flush()
        with self.assertRaisesRegex(ValueError, "prediction file SHA256"):
            self.load()


if __name__ == "__main__":
    unittest.main()
