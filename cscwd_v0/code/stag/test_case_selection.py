"""Synthetic checks for T6 deterministic case selection and complete failures."""

import copy
import json
import unittest

import numpy as np

from case_selection import CATEGORY_ORDER, METHODS, select_cases
from metrics import TemporalConfig


def fixture(specifications):
    tracks, points, scores = [], [], []
    offset = 0
    for name, indices, clip in specifications:
        frames = [{"path": f"images/video/{clip}/{index:08d}.jpg", "frame_index": index,
                   "inout": 1, "gazex_norm": [0.5], "gazey_norm": [0.5], "bbox_norm": [0, 0, 0.2, 0.2]}
                  for index in indices]
        tracks.append({"track_id": name, "start": offset, "stop": offset + len(indices),
                       "sequence_path": f"images/video/{clip}", "frames": frames})
        points.extend([[0.5, 0.5]] * len(indices))
        scores.extend([0.9] * len(indices))
        offset += len(indices)
    arrays = {method: {"argmax_xy": np.asarray(points, dtype=np.float32).copy(),
                       "inout": np.asarray(scores, dtype=np.float32).copy()}
              for method in METHODS}
    details = {method: {"events": [], "tracks": []} for method in METHODS}
    return {"tracks": tracks, "person_frames": offset}, arrays, details


def add_event(details, track, position, statuses, delays=None):
    delays = {} if delays is None else delays
    for method in METHODS:
        status = statuses.get(method, "responded")
        details[method]["events"].append({
            "track_id": track["track_id"], "event_global_index": track["start"] + position,
            "event_frame_index": track["frames"][position]["frame_index"], "type": "gaze_shift",
            "window_frames": 4, "status": status,
            "response_delay_frames": delays.get(method, 0) if status == "responded" else None,
        })


class CaseSelectionTests(unittest.TestCase):
    def select(self, values, **kwargs):
        return select_cases(*values, TemporalConfig(), **kwargs)

    def event_fixture(self):
        values = fixture([(name, list(range(1, 7)), name) for name in ("z", "a", "f1", "f2", "f3", "f4", "f5", "lag")])
        index, _, details = values
        for track in index["tracks"][:2]:
            add_event(details, track, 1, {"ema_alpha_0p25": "censored"})
        for track in index["tracks"][2:7]:
            add_event(details, track, 1, {"stag": "censored"})
        add_event(details, index["tracks"][-1], 1, {}, {"single_frame": 0, "stag": 2})
        return values

    def test_all_single_frame_only_failures_survive_group_limit(self):
        result = self.select(self.event_fixture(), cases_per_group=1)
        self.assertEqual(result["counts"]["single_frame_only"], {"candidates": 5, "selected": 5})
        self.assertEqual(result["counts"]["stag_over_ema"], {"candidates": 2, "selected": 1})
        self.assertEqual(result["candidates"]["stag_over_ema"][0]["track_id"], "a")
        failure_ids = [case["track_id"] for case in result["selected"] if case["category"] == "single_frame_only"]
        self.assertEqual(failure_ids, ["f1", "f2", "f3", "f4", "f5"])
        self.assertEqual(result["counts"]["stag_lag"]["candidates"], 1)

    def test_order_is_independent_of_track_and_event_container_order(self):
        values = self.event_fixture()
        original = self.select(values, cases_per_group=1)
        shuffled = list(copy.deepcopy(values))
        shuffled[0]["tracks"].reverse()
        shuffled[2] = {method: shuffled[2][method] for method in reversed(METHODS)}
        for method in METHODS:
            shuffled[2][method]["events"].reverse()
        self.assertEqual(self.select(shuffled, cases_per_group=1), original)
        self.assertEqual(list(original["candidates"]), list(CATEGORY_ORDER))
        case_ids = [case["case_id"] for group in original["candidates"].values() for case in group]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(all("/" not in identity for identity in case_ids))

    def test_context_stays_inside_track_and_original_frame_segment(self):
        values = fixture([("gap", [1, 2, 7, 8, 9], "a"), ("next", [1, 2, 3], "b")])
        index, _, details = values
        add_event(details, index["tracks"][0], 1, {"stag": "censored"})
        add_event(details, index["tracks"][0], 3, {"stag": "censored"})
        add_event(details, index["tracks"][1], 1, {"stag": "censored"})
        cases = self.select(values)["candidates"]["single_frame_only"]
        self.assertEqual([(case["context_start"], case["context_stop"]) for case in cases], [(0, 2), (2, 5), (5, 8)])
        bounded = self.select(values, context_before=0, context_after=0)["candidates"]["single_frame_only"]
        self.assertTrue(all(case["context_start"] == case["anchor_global_index"]
                            and case["context_stop"] == case["anchor_global_index"] + 1 for case in bounded))

    def stable_fixture(self):
        values = fixture([("stable", list(range(1, 10)), "a")])
        values[1]["single_frame"]["argmax_xy"][:, 0] = [0.4, 0.6, 0.4, 0.4, 0.4, 0.4, 0.6, 0.4, 0.4]
        return values

    def test_stable_gain_compresses_consecutive_hit_windows_into_episodes(self):
        result = self.select(self.stable_fixture())
        candidates = result["candidates"]["stable_gain"]
        self.assertEqual([case["anchor_global_index"] for case in candidates], [2, 6])
        self.assertTrue(all(case["event"] is None for case in candidates))
        self.assertEqual(result["counts"]["stable_gain"], {"candidates": 2, "selected": 2})

    def test_stable_gain_requires_truth_stability_accuracy_and_predicted_inframe(self):
        values = self.stable_fixture()
        values[0]["tracks"][0]["frames"][1]["gazex_norm"] = [0.53]
        self.assertEqual([case["anchor_global_index"] for case in self.select(values)["candidates"]["stable_gain"]], [6])
        values = self.stable_fixture()
        values[1]["stag"]["inout"][1] = 0.5
        self.assertEqual([case["anchor_global_index"] for case in self.select(values)["candidates"]["stable_gain"]], [6])
        values = self.stable_fixture()
        values[1]["stag"]["argmax_xy"][:] = [0.8, 0.8]
        self.assertEqual(self.select(values)["candidates"]["stable_gain"], [])
        values = self.stable_fixture()
        values[0]["tracks"][0]["frames"][6]["inout"] = 0
        self.assertEqual([case["anchor_global_index"] for case in self.select(values)["candidates"]["stable_gain"]], [2])

    def test_crowd_counts_unique_annotated_people_and_one_earliest_frame_per_clip(self):
        values = fixture([(name, [1, 2, 3], "a") for name in ("a", "b", "c", "d")]
                         + [(name, [5, 6], "b") for name in ("e", "f", "g", "h")])
        result = self.select(values, min_heads=4)
        crowded = result["candidates"]["crowded"]
        self.assertEqual([(case["track_id"], case["anchor_frame_index"], case["head_count"])
                          for case in crowded], [("a", 1, 4), ("e", 5, 4)])
        values = fixture([(name, [1, 2], "a") for name in ("a", "b", "c")])
        values[0]["tracks"].append(copy.deepcopy(values[0]["tracks"][0]))
        self.assertEqual(self.select(values, min_heads=4)["candidates"]["crowded"], [])

    def test_short_people_and_empty_groups_are_not_substituted(self):
        values = fixture([("a", [1, 2], "a"), ("b", [1, 2], "b")])
        values[1]["single_frame"]["argmax_xy"][:, 0] = [0.4, 0.6, 0.4, 0.6]
        result = self.select(values)
        self.assertEqual(result["selected"], [])
        self.assertTrue(all(not group for group in result["candidates"].values()))
        self.assertTrue(all(count == {"candidates": 0, "selected": 0} for count in result["counts"].values()))
        json.dumps(result, allow_nan=False)

    def test_inputs_are_unchanged(self):
        values = self.event_fixture()
        original = copy.deepcopy(values)
        self.select(values)
        self.assertEqual(values[0], original[0])
        self.assertEqual(values[2], original[2])
        for method in METHODS:
            for key in values[1][method]:
                np.testing.assert_array_equal(values[1][method][key], original[1][method][key])


if __name__ == "__main__":
    unittest.main()
