"""R1 development split checks using synthetic cache-index tracks."""

import copy
import unittest

from development_split import build_development_split


def make_track(video, clip="clip_a", person="p00", indices=(1, 2, 4), split="train"):
    return {
        "track_id": f"{split}/{video}/{clip}/{person}",
        "sequence_path": f"images/{video}/{clip}",
        "source_csv": f"annotations/{split}/{video}/{clip}/{person}.txt",
        "start": 0,
        "stop": len(indices),
        "frames": [{"path": f"images/{video}/{clip}/{index:08d}.jpg",
                    "frame_index": index, "inout": 1,
                    "gazex_norm": [0.5], "gazey_norm": [0.5]}
                   for index in indices],
    }


class DevelopmentSplitTests(unittest.TestCase):
    def setUp(self):
        self.tracks = [make_track(f"video_{index}") for index in range(10)]
        self.tracks.extend([make_track("video_0", "clip_b"),
                            make_track("video_0", "clip_a", "p01")])

    def test_video_clips_and_people_stay_together_with_exact_counts(self):
        manifest = build_development_split(self.tracks)
        self.assertTrue(manifest["passed"])
        self.assertTrue(all(manifest["checks"].values()))
        self.assertEqual(manifest["totals"], {"tracks": 12, "person_frames": 36,
                                             "unique_image_frames": 33,
                                             "annotation_videos": 10, "source_video_groups": 10})
        self.assertEqual(len(manifest["partitions"]["validation"]["group_ids"]), 2)
        group = manifest["groups"]["video_0"]
        self.assertEqual(len(group["track_ids"]), 3)
        partition_ids = manifest["partitions"][group["partition"]]["track_ids"]
        self.assertTrue(set(group["track_ids"]) <= set(partition_ids))
        self.assertEqual(group["counts"], {"tracks": 3, "person_frames": 9,
                                            "unique_image_frames": 6})
        self.assertIn("not_verified_independent_recording_id",
                      manifest["grouping"]["default_semantics"])

    def test_seeded_split_is_independent_of_input_order(self):
        forward = build_development_split(self.tracks, seed=97)
        reverse = build_development_split(list(reversed(self.tracks)), seed=97)
        self.assertEqual(forward, reverse)
        changed_seed = build_development_split(self.tracks, seed=98)
        self.assertNotEqual(forward["partitions"]["validation"]["group_ids"],
                            changed_seed["partitions"]["validation"]["group_ids"])

    def test_split_reuses_stable_hash_holdout_and_round_half_up(self):
        from gazelle.data_splits import grouped_holdout_split

        manifest = build_development_split(self.tracks, validation_fraction=0.25, seed=17)
        sorted_tracks = sorted(self.tracks, key=lambda track: track["track_id"])
        assignment = grouped_holdout_split([track["track_id"].split("/")[1]
                                            for track in sorted_tracks],
                                           validation_fraction=0.25, seed=17)
        expected = [sorted_tracks[index]["track_id"] for index in assignment.validation_indices]
        self.assertEqual(manifest["partitions"]["validation"]["track_ids"], expected)
        self.assertEqual(manifest["grouping"]["assignment"], assignment.metadata())
        self.assertEqual(len(manifest["partitions"]["validation"]["group_ids"]), 3)

    def test_gaze_labels_never_affect_partition_and_input_is_unchanged(self):
        original = copy.deepcopy(self.tracks)
        manifest = build_development_split(self.tracks)
        self.assertEqual(original, self.tracks)
        changed = copy.deepcopy(self.tracks)
        for track in changed:
            for frame in track["frames"]:
                frame["inout"] = 0
                frame["gazex_norm"] = [-1]
                frame["gazey_norm"] = [-1]
        self.assertEqual(manifest, build_development_split(changed))

    def test_explicit_source_mapping_merges_annotation_directories(self):
        mapping = {"video_0": "source_a", "video_1": "source_a"}
        manifest = build_development_split(self.tracks, source_video_map=mapping)
        self.assertEqual(manifest["totals"]["source_video_groups"], 9)
        self.assertEqual(manifest["groups"]["source_a"]["annotation_videos"],
                         ["video_0", "video_1"])
        self.assertEqual(manifest["grouping"]["provided_source_video_map"], mapping)
        partition = manifest["groups"]["source_a"]["partition"]
        selected = manifest["partitions"][partition]["track_ids"]
        expected = [track["track_id"] for track in self.tracks
                    if track["track_id"].split("/")[1] in mapping]
        self.assertTrue(set(expected) <= set(selected))

    def test_image_overlap_across_partitions_is_rejected(self):
        manifest = build_development_split(self.tracks)
        train_id = manifest["partitions"]["train"]["track_ids"][0]
        validation_id = manifest["partitions"]["validation"]["track_ids"][0]
        changed = copy.deepcopy(self.tracks)
        by_id = {track["track_id"]: track for track in changed}
        by_id[validation_id]["frames"][0]["path"] = by_id[train_id]["frames"][0]["path"]
        with self.assertRaisesRegex(ValueError, "image_paths_disjoint.*False"):
            build_development_split(changed)

    def test_test_tracks_cannot_enter_development_split(self):
        with self.assertRaisesRegex(ValueError, "accepts train tracks only"):
            build_development_split(self.tracks + [make_track("test_video", split="test")])

    def test_insufficient_groups_and_invalid_fraction_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two source video groups"):
            build_development_split([make_track("only_video")])
        with self.assertRaisesRegex(ValueError, "at least two source video groups"):
            build_development_split([make_track("a"), make_track("b")],
                                    source_video_map={"a": "same", "b": "same"})
        for fraction in (0, 1):
            with self.assertRaisesRegex(ValueError, "validation_fraction"):
                build_development_split(self.tracks, validation_fraction=fraction)

    def test_rounding_keeps_both_partitions_and_does_not_drop_gap_frames(self):
        tracks = [make_track("a", indices=(1, 4, 99)), make_track("b", indices=(3, 8))]
        for fraction in (0.01, 0.99):
            manifest = build_development_split(tracks, validation_fraction=fraction)
            self.assertEqual(manifest["totals"]["person_frames"], 5)
            self.assertEqual([len(manifest["partitions"][name]["track_ids"])
                              for name in ("train", "validation")], [1, 1])
            self.assertTrue(manifest["checks"]["person_frames_preserved"])

    def test_unknown_mapping_and_duplicate_tracks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown annotation videos"):
            build_development_split(self.tracks, source_video_map={"typo_video": "source_a"})
        with self.assertRaisesRegex(ValueError, "Duplicate track_id"):
            build_development_split(self.tracks + [self.tracks[0]])


if __name__ == "__main__":
    unittest.main()
