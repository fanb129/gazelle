"""Hand-computed state-only metrics, independent of heatmaps and gaze motion."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.metrics import TemporalConfig
from stag.state_metrics import evaluate_state_temporal, state_temporal_protocol


def make_track(labels, indices=None, start=0, name="person_a"):
    indices = list(range(1, len(labels) + 1)) if indices is None else indices
    return {"track_id": name, "start": start, "stop": start + len(labels),
            "frames": [{"frame_index": index, "inout": label}
                       for index, label in zip(indices, labels)]}


class StateTemporalMetricTests(unittest.TestCase):
    def evaluate(self, labels, scores, indices=None, config=None):
        return evaluate_state_temporal([make_track(labels, indices)], np.asarray(scores),
                                       config or TemporalConfig())

    def test_all_frames_errors_and_strict_threshold_are_split_by_gt_state(self):
        result = self.evaluate([1, 0, 1, 0, 1], [0.9, 0.5, 0.5, 0.8, 0.7])
        errors = result["summary"]["state_error"]
        self.assertEqual(errors["person_frames"], 5)
        self.assertEqual(errors["errors"], 2)
        self.assertEqual(errors["error_rate"], 0.4)
        self.assertEqual(errors["by_gt_state"]["in_frame"],
                         {"person_frames": 3, "errors": 1, "error_rate": 1 / 3})
        self.assertEqual(errors["by_gt_state"]["out_of_frame"],
                         {"person_frames": 2, "errors": 1, "error_rate": 0.5})

    def test_flip_mask_excludes_true_state_transitions_and_reports_class_counts(self):
        result = self.evaluate([1, 1, 1, 0, 0, 0], [0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        flips = result["summary"]["stable_state_flips"]
        self.assertEqual(flips["eligible_pairs"], 4)
        self.assertEqual(flips["flips"], 4)
        self.assertEqual(flips["flip_rate"], 1)
        self.assertEqual(flips["by_gt_state"]["in_frame"]["eligible_pairs"], 2)
        self.assertEqual(flips["by_gt_state"]["out_of_frame"]["flips"], 2)

    def test_gap_and_people_are_not_bridged_and_counts_are_frame_weighted(self):
        first = make_track([1, 1, 0], indices=[10, 11, 20])
        second = make_track([0, 0, 0, 0], start=3, name="person_b")
        result = evaluate_state_temporal([first, second],
                                         np.asarray([0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.9]))
        self.assertEqual(result["events"], [])
        self.assertEqual(result["summary"]["state_error"]["person_frames"], 7)
        self.assertEqual(result["summary"]["state_error"]["errors"], 3)
        flips = result["summary"]["stable_state_flips"]
        self.assertEqual(flips["eligible_pairs"], 4)
        self.assertEqual(flips["flips"], 4)
        self.assertEqual(result["tracks"][0]["stable_state_flips"]["eligible_pairs"], 1)
        self.assertEqual(result["tracks"][1]["stable_state_flips"]["eligible_pairs"], 3)

    def test_state_response_requires_no_heatmap_or_gaze_coordinate_fields(self):
        result = self.evaluate([0, 1, 1, 1, 1, 0, 0, 0],
                               [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        enter, leave = result["events"]
        self.assertEqual((enter["type"], leave["type"]), ("enter_frame", "leave_frame"))
        self.assertEqual((enter["response_delay_frames"], leave["response_delay_frames"]), (0, 0))
        self.assertEqual(result["summary"]["response"]["all"]["response_rate"], 1)
        self.assertNotIn("gaze_shift", result["summary"]["response"])

    def test_gaze_displacement_does_not_create_or_cut_state_response_events(self):
        track = make_track([0, 1, 1, 1, 1, 1], indices=list(range(100, 106)))
        for position, frame in enumerate(track["frames"]):
            frame["gazex_norm"] = [float(position % 2)]
            frame["gazey_norm"] = [float(position % 2)]
        scores = np.asarray([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        result = evaluate_state_temporal([track], scores)
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["window_frames"], 5)
        self.assertEqual(event["event_frame_index"], 101)
        self.assertEqual(event["response_delay_frames"], 1)
        self.assertEqual(event["status"], "responded")
        for frame in track["frames"]:
            frame["gazex_norm"] = [-100.0]
            frame["gazey_norm"] = [100.0]
        self.assertEqual(result, evaluate_state_temporal([track], scores))

    def test_next_state_event_truncates_window_and_short_events_are_excluded(self):
        result = self.evaluate([0, 1, 1, 0, 0, 0], [0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
        enter, leave = result["events"]
        self.assertEqual(enter["window_frames"], 2)
        self.assertEqual(enter["status"], "insufficient_followup")
        self.assertEqual(leave["window_frames"], 3)
        self.assertEqual(leave["status"], "responded")
        summary = result["summary"]["response"]["all"]
        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["assessable_events"], 1)
        self.assertEqual(summary["insufficient_followup_events"], 1)
        self.assertEqual(summary["response_rate"], 1)

    def test_gap_track_end_and_horizon_bound_windows(self):
        gap = self.evaluate([0, 1, 1, 1, 1, 1], [0.1, 0.9, 0.9, 0.9, 0.9, 0.9],
                             indices=[1, 2, 3, 7, 8, 9])
        self.assertEqual(gap["events"][0]["window_frames"], 2)
        self.assertEqual(gap["events"][0]["status"], "insufficient_followup")
        short = self.evaluate([0, 1, 1], [0.1, 0.9, 0.9])
        self.assertEqual(short["events"][0]["status"], "insufficient_followup")
        horizon = self.evaluate([0, 1, 1, 1, 1, 1], [0.1, 0.1, 0.9, 0.9, 0.9, 0.9],
                                 config=TemporalConfig(max_response_frames=3))
        self.assertEqual(horizon["events"][0]["window_frames"], 3)
        self.assertEqual(horizon["events"][0]["status"], "censored")
        self.assertIsNone(horizon["events"][0]["response_delay_frames"])

    def test_censored_events_remain_in_response_rate_denominator(self):
        first = make_track([1, 0, 0, 0], name="responded")
        second = make_track([1, 0, 0, 0], start=4, name="censored")
        result = evaluate_state_temporal([first, second],
                                         np.asarray([0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]))
        summary = result["summary"]["response"]["leave_frame"]
        self.assertEqual(summary["assessable_events"], 2)
        self.assertEqual(summary["responded_events"], 1)
        self.assertEqual(summary["censored_events"], 1)
        self.assertEqual(summary["response_rate"], 0.5)
        self.assertEqual(summary["responded_delay_frames"]["mean"], 0)

    def test_false_flip_rate_does_not_hide_constant_wrong_states(self):
        result = self.evaluate([1] * 5, [0.1] * 5)
        self.assertEqual(result["summary"]["stable_state_flips"]["flip_rate"], 0)
        self.assertEqual(result["summary"]["state_error"]["error_rate"], 1)

    def test_empty_denominators_and_protocol_are_json_serializable(self):
        result = self.evaluate([1], [0.9])
        self.assertIsNone(result["summary"]["stable_state_flips"]["flip_rate"])
        self.assertIsNone(result["summary"]["state_error"]["by_gt_state"]["out_of_frame"]["error_rate"])
        self.assertIsNone(result["summary"]["response"]["all"]["response_rate"])
        config = TemporalConfig(hold_frames=2, max_response_frames=7)
        protocol = state_temporal_protocol(config)
        self.assertEqual(protocol["version"], "state_temporal_v1")
        self.assertEqual(protocol["hold_frames"], 2)
        self.assertEqual(protocol["max_response_frames"], 7)
        text = json.dumps({"protocol": protocol, "result": result}, allow_nan=False)
        self.assertIn('"mean": null', text)
        self.assertNotIn("NaN", text)
        empty = evaluate_state_temporal([], np.asarray([]))
        self.assertIsNone(empty["summary"]["state_error"]["error_rate"])

    def test_scores_and_metadata_are_unchanged(self):
        track = make_track([0, 1, 1, 1])
        scores = np.asarray([0.1, 0.9, 0.9, 0.9])
        before_scores = scores.copy()
        before_track = json.loads(json.dumps(track))
        evaluate_state_temporal([track], scores)
        np.testing.assert_array_equal(scores, before_scores)
        self.assertEqual(track, before_track)


if __name__ == "__main__":
    unittest.main()
