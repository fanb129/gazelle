"""Hand-computed offline history grouping, source alignment and associations."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.history_diagnosis import _pearson, summarize_history_controls
from stag.history_diagnostics import CONTROL_COLUMNS


def fixture():
    descriptions = [([0, 1, 2, 3, 5, 6, 7, 8], [0, 1, 1, 1, 1, 1, 1, 0]),
                    ([2, 3, 4], [1, 1, 1])]
    tracks, start = [], 0
    for number, (indices, labels) in enumerate(descriptions):
        frames = [{"frame_index": index, "inout": label,
                   "gazex_norm": [.5 if number == 0 and index == 3 else .2],
                   "gazey_norm": [.2]} for index, label in zip(indices, labels)]
        tracks.append({"track_id": f"test/video/clip/person_{number}", "start": start,
                       "stop": start + len(frames), "frames": frames})
        start += len(frames)
    points = np.asarray([[f["gazex_norm"][0], f["gazey_norm"][0]]
                         for track in tracks for f in track["frames"]])
    scores = np.asarray([.9 if f["inout"] else .1 for track in tracks for f in track["frames"]])
    controls = np.full((start, len(CONTROL_COLUMNS)), .5)
    g = np.asarray([0, .1, .2, .3, 0, .5, .6, .7, 0, .9, 1])
    controls[:, CONTROL_COLUMNS.index("history_validity")] = g
    controls[:, CONTROL_COLUMNS.index("history_available")] = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    index = {"control_columns": list(CONTROL_COLUMNS), "source_person_frames": start,
             "person_frames": start, "tracks": [
                 {"track_id": t["track_id"], "start": t["start"], "stop": t["stop"],
                  "source_global_start": t["start"], "source_global_stop": t["stop"],
                  "frame_indices": [f["frame_index"] for f in t["frames"]]} for t in tracks]}
    arrays = {"controls": controls, "correspondence_xy": points + [.1, 0],
              "source_global_indices": np.arange(start)}
    return tracks, points, scores, points + np.column_stack((g * .01, np.zeros(start))), scores + g * .1, index, arrays


class HistoryDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()

    def test_hand_computed_group_counts_control_quantiles_and_gate_fractions(self):
        result = summarize_history_controls(*self.inputs)
        groups = result["groups"]
        self.assertEqual(result["person_frames"], 11)
        self.assertEqual(groups["GT_inframe"]["frames"], 9)
        self.assertEqual(groups["GT_out_of_frame"]["frames"], 2)
        self.assertEqual(groups["GT_stable_inframe_pair_endpoints"]["frames"], 5)
        self.assertEqual(groups["history_available"]["frames"], 8)
        self.assertEqual(groups["no_history"]["frames"], 3)
        g = groups["all"]["controls"]["history_validity"]
        self.assertAlmostEqual(g["mean"], 4.3 / 11)
        self.assertEqual(g["p10"], 0)
        self.assertEqual(g["p50"], .3)
        self.assertEqual(g["p90"], .9)
        fractions = groups["all"]["gate_fractions"]["history_validity"]
        self.assertEqual(fractions["low_fraction"], 4 / 11)
        self.assertEqual(fractions["high_fraction"], 2 / 11)

    def test_history_absence_is_separate_from_validity_and_correspondence_statistics(self):
        result = summarize_history_controls(*self.inputs)
        self.assertEqual(result["groups"]["all"]["validity_on_available_history"]["gate_fractions"]["low_fraction"], 1 / 8)
        correspondence = result["groups"]["all"]["correspondence_on_available_history"]
        self.assertEqual(correspondence["to_raw_prediction"]["count"], 8)
        self.assertEqual(correspondence["to_GT_inframe"]["count"], 7)
        self.assertAlmostEqual(correspondence["to_raw_prediction"]["mean"], .1)
        empty = result["groups"]["no_history"]["correspondence_on_available_history"]["to_raw_prediction"]
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["mean"])

    def test_proxy_event_neighborhoods_use_original_frames_and_do_not_cross_gaps_or_tracks(self):
        result = summarize_history_controls(*self.inputs, event_radius=3)
        self.assertEqual(result["proxy_event_counts"], {"enter_frame": 1, "leave_frame": 1, "gaze_shift": 1})
        for group in ("near_enter_frame", "near_leave_frame", "near_gaze_shift"):
            self.assertEqual(result["groups"][group]["frames"], 4)
        center = summarize_history_controls(*self.inputs, event_radius=0)
        for group in ("near_enter_frame", "near_leave_frame", "near_gaze_shift"):
            self.assertEqual(center["groups"][group]["frames"], 1)

    def test_source_row_permutation_preserves_every_summary_without_modifying_inputs(self):
        expected = summarize_history_controls(*self.inputs)
        inputs = list(self.inputs)
        order = np.asarray([8, 3, 2, 6, 1, 10, 0, 7, 4, 9, 5])
        arrays = {name: values[order].copy() for name, values in inputs[-1].items()}
        for values in arrays.values():
            values.flags.writeable = False
        original = {name: values.tobytes() for name, values in arrays.items()}
        inputs[-1] = arrays
        self.assertEqual(summarize_history_controls(*inputs), expected)
        self.assertEqual(original, {name: values.tobytes() for name, values in arrays.items()})

    def test_correlations_are_hand_computed_and_constant_or_short_groups_are_null(self):
        result = summarize_history_controls(*self.inputs)
        associations = result["groups"]["history_available"]["correction_associations_on_available_history"]
        self.assertAlmostEqual(associations["history_validity"]["coordinate_delta_l2"]["pearson"], 1)
        self.assertAlmostEqual(associations["history_validity"]["score_absolute_delta"]["pearson"], 1)
        self.assertIsNone(associations["refresh"]["score_absolute_delta"]["pearson"])
        short = summarize_history_controls(*self.inputs, event_radius=0)
        short = short["groups"]["near_leave_frame"]["correction_associations_on_available_history"]
        self.assertEqual(short["history_validity"]["coordinate_delta_l2"]["count"], 1)
        self.assertIsNone(short["history_validity"]["coordinate_delta_l2"]["pearson"])
        self.assertIsNone(_pearson(np.arange(7), np.full(7, .1, dtype=np.float32))["pearson"])
        self.assertIsNone(_pearson(np.full(7, .1), np.arange(7))["pearson"])
        json.dumps(result, allow_nan=False)

    def test_duplicate_or_missing_source_rows_are_rejected(self):
        inputs = copy.deepcopy(list(self.inputs))
        inputs[-1]["source_global_indices"][0] = inputs[-1]["source_global_indices"][1]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize_history_controls(*inputs)
        inputs = copy.deepcopy(list(self.inputs))
        inputs[-1] = {key: value[:-1] for key, value in inputs[-1].items()}
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize_history_controls(*inputs)

    def test_track_frame_mapping_and_control_columns_are_verified(self):
        inputs = copy.deepcopy(list(self.inputs))
        inputs[-2]["tracks"][0]["frame_indices"][0] = 99
        with self.assertRaisesRegex(ValueError, "track/frame mapping"):
            summarize_history_controls(*inputs)
        inputs = copy.deepcopy(list(self.inputs))
        inputs[-2]["control_columns"][0] = "unknown"
        with self.assertRaisesRegex(ValueError, "control columns"):
            summarize_history_controls(*inputs)

    def test_empty_event_groups_are_explicit_and_radius_must_be_nonnegative(self):
        inputs = copy.deepcopy(list(self.inputs))
        for track in inputs[0]:
            for frame in track["frames"]:
                frame.update(inout=1, gazex_norm=[.2])
        result = summarize_history_controls(*inputs)
        for group in ("near_enter_frame", "near_leave_frame", "near_gaze_shift"):
            self.assertEqual(result["groups"][group]["frames"], 0)
            self.assertIsNone(result["groups"][group]["controls"]["refresh"]["mean"])
        with self.assertRaisesRegex(ValueError, "radius"):
            summarize_history_controls(*self.inputs, event_radius=-1)


if __name__ == "__main__":
    unittest.main()
