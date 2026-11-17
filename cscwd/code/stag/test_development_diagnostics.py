"""R1 diagnostic sampling checks using synthetic tracks and RGB images."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from development_diagnostics import CATEGORY_ORDER, render_development_case, select_development_cases
from metrics import TemporalConfig


def fixture(specifications, offset=0):
    tracks, points, scores = [], [[0.0, 0.0]] * offset, [0.1] * offset
    for person, clip, indices, labels, targets in specifications:
        start = len(points)
        frames = []
        for index, label, target in zip(indices, labels, targets):
            frames.append({
                "path": f"images/video/{clip}/{index:08d}.jpg", "frame_index": index,
                "bbox_norm": [0.1, 0.1, 0.4, 0.5], "inout": label,
                "gazex_norm": [target[0]], "gazey_norm": [target[1]],
            })
            points.append(target if label else [0.5, 0.5])
            scores.append(0.9 if label else 0.1)
        tracks.append({"track_id": f"train/video/{clip}/{person}",
                       "sequence_path": f"images/video/{clip}", "frames": frames,
                       "start": start, "stop": len(points)})
    return tracks, np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def stable(person, clip, length=6):
    return person, clip, list(range(1, length + 1)), [1] * length, [[0.4, 0.4]] * length


class DevelopmentDiagnosticTests(unittest.TestCase):
    def select(self, values, **kwargs):
        return select_development_cases(*values, TemporalConfig(), **kwargs)

    def dynamic(self):
        labels = [1] * 5 + [0, 0] + [1] * 3
        targets = [[0.2, 0.2]] * 2 + [[0.7, 0.7]] * 3 + [[-1, -1]] * 2 + [[0.4, 0.4]] * 3
        values = fixture([("p0", "dynamic", list(range(1, 11)), labels, targets)], offset=7)
        values[1][8] = [0.9, 0.9]
        return values

    def test_proxy_categories_are_separate_from_manual_labels(self):
        result = self.select(self.dynamic())
        self.assertEqual(tuple(result["counts"]), CATEGORY_ORDER)
        self.assertEqual(result["counts"]["gaze_shift_proxy"]["candidates"], 1)
        self.assertEqual(result["counts"]["state_transition"]["candidates"], 2)
        self.assertEqual(result["counts"]["prediction_jump"]["candidates"], 1)
        self.assertEqual(result["counts"]["stable_reference"]["candidates"], 2)
        for case in result["selected"]:
            self.assertEqual(case["review_status"], "unreviewed")
            self.assertEqual(case["labels"], [])
            self.assertEqual(case["notes"], "")
            self.assertGreaterEqual(case["anchor_global_index"], 7)
        json.dumps(result, allow_nan=False)

    def test_prediction_jump_requires_stable_gt_and_does_not_require_inout_score(self):
        values = fixture([stable("p0", "a")])
        values[1][1] = [0.9, 0.9]
        values[2][:] = 0.1
        self.assertEqual(self.select(values)["counts"]["prediction_jump"]["candidates"], 2)
        for frame in values[0][0]["frames"][1:]:
            frame["gazex_norm"] = [0.9]
            frame["gazey_norm"] = [0.9]
        self.assertEqual(self.select(values)["counts"]["prediction_jump"]["candidates"], 1)

    def test_seeded_sampling_is_reproducible_and_input_order_independent(self):
        values = fixture([stable("p0", f"clip_{number}") for number in range(12)])
        expected = self.select(values, cases_per_category=3)
        reordered = (list(reversed(values[0])), values[1], values[2])
        self.assertEqual(self.select(reordered, cases_per_category=3), expected)
        self.assertEqual(self.select(values, cases_per_category=3), expected)
        alternate = self.select(values, cases_per_category=3, seed=20261118)
        self.assertNotEqual([case["case_id"] for case in expected["selected"]],
                            [case["case_id"] for case in alternate["selected"]])

    def test_each_category_samples_at_most_one_person_per_clip(self):
        values = fixture([stable("p0", "a"), stable("p1", "a"), stable("p0", "b")])
        result = self.select(values, cases_per_category=3)
        self.assertEqual(result["counts"]["stable_reference"],
                         {"candidates": 3, "candidate_clips": 2, "selected": 2})
        cases = [case for case in result["selected"] if case["category"] == "stable_reference"]
        self.assertEqual(len({case["sequence_path"] for case in cases}), len(cases))

    def test_gap_is_not_an_event_and_context_never_crosses_gap(self):
        indices = [1, 2, 7, 8, 9, 15, 16, 17]
        values = fixture([("p0", "gaps", indices, [1] * 8, [[0.4, 0.4]] * 8)], offset=4)
        values[1][6:9] = [0.9, 0.9]
        result = self.select(values)
        self.assertEqual(result["counts"]["prediction_jump"]["candidates"], 0)
        self.assertEqual(result["counts"]["stable_reference"]["candidates"], 2)
        for case in result["selected"]:
            frames = values[0][0]["frames"][case["context_start"] - 4:case["context_stop"] - 4]
            self.assertTrue(all(right["frame_index"] - left["frame_index"] == 1
                                for left, right in zip(frames, frames[1:])))
            self.assertEqual(case["frame_paths"], [frame["path"] for frame in frames])

    def test_test_tracks_are_rejected_and_empty_categories_remain_empty(self):
        values = fixture([("p0", "out", [1, 2], [0, 0], [[-1, -1]] * 2)])
        result = self.select(values)
        self.assertEqual(result["selected"], [])
        self.assertTrue(all(count == {"candidates": 0, "candidate_clips": 0, "selected": 0}
                            for count in result["counts"].values()))
        values[0][0]["track_id"] = "test/video/out/p0"
        with self.assertRaisesRegex(ValueError, "train tracks only"):
            self.select(values)

    def test_sampling_preserves_track_labels_and_prediction_arrays(self):
        values = self.dynamic()
        before = copy.deepcopy(values)
        self.select(values)
        self.assertEqual(values[0], before[0])
        np.testing.assert_array_equal(values[1], before[1])
        np.testing.assert_array_equal(values[2], before[2])

    def test_rgb_render_keeps_sources_readonly_and_snapshots_prioritize_anchor(self):
        values = self.dynamic()
        result = self.select(values)
        case = next(case for case in result["selected"] if case["category"] == "gaze_shift_proxy")
        track, points, scores = values[0][0], values[1], values[2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frame in track["frames"]:
                path = root / frame["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 180), (160, 130, 100)).save(path)
            hashes = {frame["path"]: hashlib.sha256((root / frame["path"]).read_bytes()).hexdigest()
                      for frame in track["frames"]}
            metadata = render_development_case(case, track, points, scores, root,
                                                root / "preview.png", snapshots=3)
            self.assertEqual(metadata["snapshot_global_indices"],
                             [case["anchor_global_index"] - 1, case["anchor_global_index"],
                              case["anchor_global_index"] + 1])
            with Image.open(metadata["path"]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (metadata["width"], metadata["height"]))
                self.assertGreater(len(image.getcolors(image.width * image.height)), 3)
            for name, expected in hashes.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
            state_case = next(case for case in result["selected"] if case["category"] == "state_transition")
            self.assertTrue(Path(render_development_case(
                state_case, track, points, scores, root, root / "state.png")["path"]).exists())
            with self.assertRaises(FileNotFoundError):
                render_development_case(case, track, points, scores, root / "missing", root / "missing.png")


if __name__ == "__main__":
    unittest.main()
