"""Synthetic T2 tests: actual tensor inference and NPY recovery, without DINO/VAT."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from cache_single_frame import TrackFrameDataset, collate_frames, load_frozen_checkpoint, prediction_batches
from stag.cache import argmax_points, file_sha256, iter_cached_tracks, load_cache, write_prediction_cache


def transform_image(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255


class TinyGazeModel(torch.nn.Module):
    """Exercise GazeLLE's list-per-image output contract with real torch tensors."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)
        self.calls = 0
        self.seen_keys = []
        self.inference_flags = []

    def load_gazelle_state_dict(self, state):
        combined = self.state_dict()
        combined.update(state)
        self.load_state_dict(combined)

    def forward(self, inputs):
        self.calls += 1
        self.seen_keys.append(set(inputs))
        self.inference_flags.append((self.training, torch.is_grad_enabled()))
        pixels = inputs["images"].mean(dim=(1, 2, 3))
        positions = torch.tensor([boxes[0][0] for boxes in inputs["bboxes"]])
        peaks = ((pixels * 255 + positions * 100).round().long() % 4096)
        heatmaps = torch.zeros(len(pixels), 4096, dtype=torch.float32)
        heatmaps[torch.arange(len(pixels)), peaks] = 0.875
        inout = torch.sigmoid(self.head(pixels[:, None])).flatten()
        return {"heatmap": list(heatmaps.reshape(-1, 64, 64).split(1)),
                "inout": list(inout.split(1))}


class SingleFrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequences = []
        for clip, indices in (("clip_a", [1, 3]), ("clip_b", [1, 2, 3])):
            sequence_path = f"images/video_a/{clip}"
            image_dir = self.root / sequence_path
            image_dir.mkdir(parents=True)
            frames = []
            for index in indices:
                Image.new("RGB", (8, 8), (index * 30, 20, 10)).save(image_dir / f"{index:08d}.png")
                frames.append({
                    "path": f"{sequence_path}/{index:08d}.png", "frame_index": index,
                    "person_id": "007", "bbox_norm": [index / 10, 0.1, 0.7, 0.8],
                    "gazex_norm": [0.5], "gazey_norm": [0.4], "inout": int(index != 3),
                })
            self.sequences.append({
                "path": sequence_path, "width": 8, "height": 8,
                "tracks": [{"person_id": "007", "track_id": f"test/video_a/{clip}/007",
                            "source_csv": f"annotations/test/video_a/{clip}/007.txt", "frames": frames}],
            })
        self.checkpoint = self.root / "epoch_7.pt"
        torch.save({"head.weight": torch.tensor([[0.75]]), "head.bias": torch.tensor([-0.25])}, self.checkpoint)
        self.model = TinyGazeModel()
        self.original_backbone = copy.deepcopy(self.model.backbone.state_dict())
        load_frozen_checkpoint(self.model, self.checkpoint)

    def batches(self, sequences=None, batch_size=3):
        dataset = TrackFrameDataset(self.sequences if sequences is None else sequences, self.root, transform_image)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_frames)
        return prediction_batches(self.model, loader, torch.device("cpu"))

    def test_track_cache_lossless_mapping_and_offline_recovery(self):
        live = list(self.batches())
        expected_heatmaps = np.concatenate([batch[0] for batch in live])
        expected_inout = np.concatenate([batch[1] for batch in live])
        calls = self.model.calls
        result = write_prediction_cache(self.sequences, iter(live), self.root / "cache", {"model": "synthetic"})
        index, arrays = load_cache(self.root / "cache")
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual((result["tracks"], result["person_frames"], result["unique_image_frames"]), (2, 5, 5))
        np.testing.assert_array_equal(arrays["heatmap"], expected_heatmaps)
        np.testing.assert_array_equal(arrays["inout"], expected_inout)
        np.testing.assert_array_equal(arrays["argmax_xy"], argmax_points(expected_heatmaps))
        self.assertFalse(arrays["heatmap"].flags.writeable)
        recovered = list(iter_cached_tracks(index, arrays))
        self.assertEqual([(track["start"], track["stop"]) for track, _ in recovered], [(0, 2), (2, 5)])
        self.assertEqual([track["track_id"] for track, _ in recovered],
                         ["test/video_a/clip_a/007", "test/video_a/clip_b/007"])
        self.assertEqual([frame["frame_index"] for frame in recovered[0][0]["frames"]], [1, 3])
        for (track, values), source in zip(recovered, self.sequences):
            self.assertEqual(track["frames"], source["tracks"][0]["frames"])
            self.assertEqual(len(values["heatmap"]), len(track["frames"]))
        for artifact in result["files"].values():
            self.assertEqual(artifact["sha256"], file_sha256(self.root / "cache" / artifact["path"]))
        self.assertEqual(self.model.calls, calls)

    def test_frozen_eval_contract_and_current_frame_only_inputs(self):
        list(self.batches())
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))
        self.assertTrue(all(flags == (False, False) for flags in self.model.inference_flags))
        self.assertTrue(all(keys == {"images", "bboxes"} for keys in self.model.seen_keys))
        for key, value in self.model.backbone.state_dict().items():
            torch.testing.assert_close(value, self.original_backbone[key], rtol=0, atol=0)
        torch.testing.assert_close(self.model.head.weight, torch.tensor([[0.75]]), rtol=0, atol=0)

    def test_labels_never_change_predictions(self):
        original = list(self.batches())
        changed = copy.deepcopy(self.sequences)
        for sequence in changed:
            for frame in sequence["tracks"][0]["frames"]:
                frame.update(gazex_norm=[999.0], gazey_norm=[-999.0], inout=1 - frame["inout"])
        altered = list(self.batches(changed))
        for before, after in zip(original, altered):
            for expected, actual in zip(before, after):
                np.testing.assert_array_equal(expected, actual)

    def test_future_changes_and_batch_boundaries_preserve_prefix(self):
        full = list(self.batches(batch_size=5))[0]
        prefix = copy.deepcopy(self.sequences[:1])
        prefix[0]["tracks"][0]["frames"] = prefix[0]["tracks"][0]["frames"][:1]
        short = list(self.batches(prefix, batch_size=1))[0]
        self.sequences[0]["tracks"][0]["frames"][1]["bbox_norm"] = [0.9, 0.9, 1.0, 1.0]
        future = self.root / self.sequences[0]["tracks"][0]["frames"][1]["path"]
        Image.new("RGB", (8, 8), (255, 255, 255)).save(future)
        changed = list(self.batches(batch_size=2))[0]
        for original, truncated, altered in zip(full, short, changed):
            np.testing.assert_array_equal(original[:1], truncated)
            np.testing.assert_array_equal(original[:1], altered[:1])

    def test_wrong_checkpoint_heads_are_rejected(self):
        torch.save({"head.weight": torch.zeros(1, 1)}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "Checkpoint head keys mismatch"):
            load_frozen_checkpoint(self.model, self.checkpoint)

    def test_incomplete_cache_does_not_publish_index(self):
        predictions = list(self.batches())[:1]
        with self.assertRaisesRegex(ValueError, "Incomplete cache"):
            write_prediction_cache(self.sequences, predictions, self.root / "partial", {})
        self.assertFalse((self.root / "partial/index.json").exists())

    def test_lossy_precision_is_rejected(self):
        heatmaps, inout = list(self.batches())[0]
        with self.assertRaisesRegex(ValueError, "must be float32"):
            write_prediction_cache(self.sequences, [(heatmaps.astype(np.float16), inout)], self.root / "half", {})

    def test_argmax_coordinates_and_tie_rule(self):
        heatmaps = np.zeros((3, 64, 64), dtype=np.float32)
        heatmaps[1, 63, 63] = 1
        heatmaps[2, 20, 10] = heatmaps[2, 30, 40] = 1
        np.testing.assert_array_equal(argmax_points(heatmaps), [[0, 0], [63 / 64, 63 / 64], [10 / 64, 20 / 64]])

    def test_offline_reader_imports_no_torch(self):
        write_prediction_cache(self.sequences, self.batches(), self.root / "cache", {})
        source = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from stag.cache import load_cache, iter_cached_tracks; "
            "index, arrays = load_cache(sys.argv[2]); "
            "print(sum(len(values['heatmap']) for _, values in iter_cached_tracks(index, arrays))); "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", source,
                                 str(Path(__file__).resolve().parents[1]), str(self.root / "cache")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "5")


if __name__ == "__main__":
    unittest.main()
