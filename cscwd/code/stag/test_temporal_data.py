"""Synthetic checks for state-preserving chunks and VAT supervision targets."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from gazelle.utils import get_heatmap
from stag.temporal_data import contiguous_segments, iter_track_batches


class TemporalDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tracks = []
        start = 0
        definitions = (("a", [1, 2, 3, 4, 5, 6, 7, 20, 21, 22]),
                       ("b", [1, 2, 3, 4]), ("c", [1, 2, 3, 4, 5]))
        for person, indices in definitions:
            frames = []
            for position, frame_index in enumerate(indices):
                inside = int(position % 4 != 2)
                frames.append({
                    "path": f"images/video/clip/{frame_index:08d}.jpg",
                    "frame_index": frame_index, "person_id": person,
                    "bbox_norm": [0.1, 0.2, 0.3, 0.4],
                    "gazex_norm": [0.25] if inside else [-1.0],
                    "gazey_norm": [0.5] if inside else [-1.0], "inout": inside,
                })
            self.tracks.append({"track_id": f"train/video/clip/{person}",
                                "start": start, "stop": start + len(frames), "frames": frames})
            start += len(frames)
        image_paths = sorted({frame["path"] for track in self.tracks for frame in track["frames"]})
        self.feature_index = {"image_paths": image_paths,
                              "contract": {"channels": 3, "grid": [2, 2], "storage_dtype": "float16"}}
        feature_path = Path(self.temporary.name) / "features.npy"
        np.save(feature_path, np.arange(len(image_paths) * 12, dtype=np.float16).reshape(-1, 3, 2, 2))
        self.features = np.load(feature_path, mmap_mode="r")
        self.arrays = {"heatmap": np.arange(start * 64 * 64, dtype=np.float32).reshape(start, 64, 64),
                       "inout": np.linspace(0.1, 0.9, start, dtype=np.float32)}

    def batches(self, **kwargs):
        return list(iter_track_batches(self.tracks, self.arrays, self.feature_index, self.features,
                                       clip_frames=3, batch_size=2, **kwargs))

    def test_all_valid_person_frames_appear_once_with_global_prediction_slices(self):
        seen = []
        for batch in self.batches():
            indices = batch["global_indices"][batch["valid"]]
            seen.extend(indices.tolist())
            torch.testing.assert_close(batch["raw_heatmap"][batch["valid"]],
                                       torch.from_numpy(self.arrays["heatmap"][indices.numpy()]))
            torch.testing.assert_close(batch["raw_inout"][batch["valid"]],
                                       torch.from_numpy(self.arrays["inout"][indices.numpy()]))
        self.assertEqual(sorted(seen), list(range(len(self.arrays["inout"]))))
        self.assertEqual(len(seen), len(set(seen)))

    def test_slots_continue_without_chunk_reset_and_reset_at_gap_or_new_person(self):
        previous = [None, None]
        continuations = 0
        observed_segment_starts = set()
        for batch in self.batches():
            for slot, identity in enumerate(batch["track_ids"]):
                valid = batch["valid"][slot]
                if identity is None:
                    continue
                frames = batch["frame_indices"][slot][valid].tolist()
                self.assertTrue(all(right - left == 1 for left, right in zip(frames, frames[1:])))
                begins = bool(batch["reset"][slot, 0])
                if begins:
                    observed_segment_starts.add((identity, frames[0]))
                else:
                    self.assertEqual(previous[slot][0], identity)
                    self.assertEqual(previous[slot][1] + 1, frames[0])
                    continuations += 1
                self.assertFalse(batch["reset"][slot, 1:].any())
                previous[slot] = (identity, frames[-1])
        self.assertGreater(continuations, 0)
        self.assertEqual(observed_segment_starts,
                         {(self.tracks[0]["track_id"], 1), (self.tracks[0]["track_id"], 20),
                          (self.tracks[1]["track_id"], 1), (self.tracks[2]["track_id"], 1)})

    def test_padding_is_masked_and_has_no_cache_index_or_label(self):
        batches = self.batches()
        self.assertTrue(any((~batch["valid"]).any() for batch in batches))
        for batch in batches:
            padded = ~batch["valid"]
            self.assertTrue((batch["global_indices"][padded] == -1).all())
            self.assertTrue((batch["frame_indices"][padded] == -1).all())
            self.assertFalse(batch["reset"][padded].any())
            for key in ("features", "raw_heatmap", "raw_inout", "bboxes",
                        "target_heatmap", "target_inout"):
                self.assertTrue((batch[key][padded] == 0).all(), key)

    def test_seeded_segment_shuffle_is_reproducible_without_reordering_its_frames(self):
        left = self.batches(seed=13)
        right = self.batches(seed=13)
        for a, b in zip(left, right):
            self.assertEqual(a["track_ids"], b["track_ids"])
            self.assertTrue(torch.equal(a["global_indices"], b["global_indices"]))
        canonical = [(track["track_id"], start, stop)
                     for track, start, stop in contiguous_segments(self.tracks)]
        shuffled = [(track["track_id"], start, stop)
                    for track, start, stop in contiguous_segments(self.tracks, seed=13)]
        self.assertCountEqual(canonical, shuffled)
        self.assertNotEqual(canonical, shuffled)
        for batch in left:
            for frames, valid in zip(batch["frame_indices"], batch["valid"]):
                sequence = frames[valid].tolist()
                self.assertTrue(all(y - x == 1 for x, y in zip(sequence, sequence[1:])))

    def test_people_sharing_an_image_read_identical_frozen_scene_features(self):
        seen = {}
        for batch in self.batches():
            self.assertEqual(batch["features"].dtype, torch.float16)
            for slot, identity in enumerate(batch["track_ids"]):
                for position in torch.nonzero(batch["valid"][slot], as_tuple=False).flatten().tolist():
                    frame_index = int(batch["frame_indices"][slot, position])
                    feature = batch["features"][slot, position]
                    if frame_index in seen:
                        self.assertTrue(torch.equal(feature, seen[frame_index]))
                    seen[frame_index] = feature.clone()
        self.assertFalse(self.features.flags.writeable)

    def test_ground_truth_changes_only_targets_not_inputs_or_reset_schedule(self):
        originals = self.batches()
        changed_tracks = copy.deepcopy(self.tracks)
        for track in changed_tracks:
            for frame in track["frames"]:
                frame.update(inout=1-frame["inout"], gazex_norm=[0.75], gazey_norm=[0.1])
        changed = list(iter_track_batches(changed_tracks, self.arrays, self.feature_index,
                                         self.features, clip_frames=3, batch_size=2))
        for before, after in zip(originals, changed):
            for key in ("features", "raw_heatmap", "raw_inout", "bboxes", "valid", "reset",
                        "global_indices", "frame_indices"):
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertTrue(any(not torch.equal(before["target_inout"], after["target_inout"])
                            for before, after in zip(originals, changed)))

    def test_heatmaps_match_original_vat_gaussian_at_interior_and_boundaries(self):
        tracks = copy.deepcopy(self.tracks)
        coordinates = ((0.0, 0.0), (1.0, 1.0), (0.01, 0.99), (0.25, 0.5), (0.999, 0.01))
        for track in tracks:
            for position, frame in enumerate(track["frames"]):
                x, y = coordinates[position % len(coordinates)]
                frame.update(inout=1, gazex_norm=[x], gazey_norm=[y])
        batches = iter_track_batches(tracks, self.arrays, self.feature_index, self.features,
                                     clip_frames=3, batch_size=2)
        frames = {track["start"] + position: frame for track in tracks
                  for position, frame in enumerate(track["frames"])}
        for batch in batches:
            for slot, position in torch.nonzero(batch["valid"], as_tuple=False).tolist():
                frame = frames[int(batch["global_indices"][slot, position])]
                expected = get_heatmap(frame["gazex_norm"][0], frame["gazey_norm"][0], 64, 64)
                torch.testing.assert_close(batch["target_heatmap"][slot, position], expected, rtol=0, atol=0)

    def test_outside_targets_are_zero_and_yielded_buffers_do_not_modify_sources(self):
        original_predictions = {key: value.copy() for key, value in self.arrays.items()}
        original_features = np.array(self.features)
        for batch in self.batches():
            outside = batch["valid"] & (batch["target_inout"] == 0)
            self.assertTrue((batch["target_heatmap"][outside] == 0).all())
            batch["features"].zero_()
            batch["raw_heatmap"].zero_()
            batch["raw_inout"].zero_()
        for key in self.arrays:
            np.testing.assert_array_equal(self.arrays[key], original_predictions[key])
        np.testing.assert_array_equal(self.features, original_features)


if __name__ == "__main__":
    unittest.main()
