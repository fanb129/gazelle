"""Synthetic temporal-metric checks with hand-computed trajectories and events."""

import json
import unittest

import numpy as np

from metrics import TemporalConfig, evaluate_temporal, temporal_protocol


def make_track(gt, labels=None, indices=None, start=0, name="person_a"):
    labels = [1] * len(gt) if labels is None else labels
    indices = list(range(1, len(gt) + 1)) if indices is None else indices
    return {"track_id": name, "start": start, "stop": start + len(gt),
            "frames": [{"frame_index": index, "inout": label,
                        "gazex_norm": [point[0]], "gazey_norm": [point[1]]}
                       for point, label, index in zip(gt, labels, indices)]}


class TemporalMetricTests(unittest.TestCase):
    def evaluate(self, gt, predicted, labels=None, indices=None, scores=None, config=None):
        track = make_track(gt, labels, indices)
        scores = np.ones(len(gt)) if scores is None else np.asarray(scores)
        return evaluate_temporal([track], np.asarray(predicted), scores, config or TemporalConfig())

    def test_jitter_and_gt_use_identical_eligible_pair_mask(self):
        gt = [[0, 0], [0.1, 0], [-1, -1], [0.7, 0], [0.9, 0]]
        predicted = [[0, 0], [0.2, 0], [0, 1], [0.4, 0], [0.7, 0]]
        result = self.evaluate(gt, predicted, labels=[1, 1, 0, 1, 1])
        jitter = result["summary"]["jitter"]
        self.assertEqual(jitter["pairs"], 2)
        self.assertAlmostEqual(jitter["predicted"]["mean"], 0.25)
        self.assertAlmostEqual(jitter["ground_truth"]["mean"], 0.15)
        self.assertAlmostEqual(jitter["predicted"]["median"], 0.25)
        self.assertAlmostEqual(jitter["predicted"]["p90"], 0.29)

    def test_gaps_and_people_are_not_bridged_and_aggregate_pairs(self):
        first = make_track([[0, 0], [0.1, 0], [1, 0]], indices=[10, 11, 20])
        second = make_track([[0, 0], [0.2, 0], [0.4, 0], [0.6, 0]], start=3, name="person_b")
        predicted = np.array([[0, 0], [0.1, 0], [1, 0], [0, 0], [0.2, 0], [0.4, 0], [0.6, 0]])
        result = evaluate_temporal([first, second], predicted, np.ones(7), TemporalConfig())
        self.assertEqual(result["summary"]["jitter"]["pairs"], 4)
        self.assertAlmostEqual(result["summary"]["jitter"]["predicted"]["mean"], 0.175)
        self.assertEqual(result["tracks"][0]["stability"]["eligible_windows"], 0)
        self.assertEqual(result["tracks"][1]["stability"]["eligible_windows"], 2)
        self.assertEqual([event["event_global_index"] for event in result["events"]], [4, 5, 6])

    def test_stability_window_denominator_and_accurate_stability(self):
        result = self.evaluate([[0.5, 0.5]] * 5, [[0, 0]] * 5)
        stability = result["summary"]["stability"]
        self.assertEqual(stability["eligible_windows"], 3)
        self.assertEqual(stability["stable_predicted_frames"], 3)
        self.assertEqual(stability["stable_gt_frames"], 3)
        self.assertEqual(stability["stable_ratio"], 1)
        self.assertEqual(stability["accurate_stable_ratio"], 0)
        result = self.evaluate([[0.5, 0.5]] * 5, [[0.5, 0.5]] * 5,
                               scores=[0.9, 0.9, 0.5, 0.9, 0.9])
        self.assertEqual(result["summary"]["stability"]["accurate_stable_frames"], 0)

    def test_stability_uses_consecutive_inframe_windows(self):
        gt = [[0, 0]] * 8
        result = self.evaluate(gt, gt, labels=[1, 1, 0, 1, 1, 1, 1, 1],
                               indices=[1, 2, 3, 4, 5, 6, 9, 10])
        self.assertEqual(result["summary"]["stability"]["eligible_windows"], 1)
        self.assertEqual(result["summary"]["stability"]["accurate_stable_frames"], 1)

    def test_response_can_be_immediate_or_delayed_in_original_frames(self):
        gt = [[0, 0]] + [[0.5, 0]] * 6
        immediate = self.evaluate(gt, gt, indices=list(range(100, 107)))
        delayed = self.evaluate(gt, [[0, 0]] * 3 + [[0.5, 0]] * 4,
                                indices=list(range(100, 107)))
        self.assertEqual(immediate["events"][0]["response_delay_frames"], 0)
        event = delayed["events"][0]
        self.assertEqual(event["event_frame_index"], 101)
        self.assertEqual(event["event_global_index"], 1)
        self.assertEqual(event["response_delay_frames"], 2)
        self.assertEqual(event["status"], "responded")
        self.assertEqual(delayed["summary"]["response"]["all"]["response_rate"], 1)

    def test_unresolved_events_are_censored_and_stay_in_denominator(self):
        gt = [[0, 0]] + [[0.5, 0]] * 5
        first = make_track(gt, name="responded")
        second = make_track(gt, start=6, name="unresolved")
        predictions = np.array(gt + [[0, 0]] * 6)
        result = evaluate_temporal([first, second], predictions, np.ones(12), TemporalConfig())
        response = result["summary"]["response"]["gaze_shift"]
        self.assertEqual(response["events"], 2)
        self.assertEqual(response["assessable_events"], 2)
        self.assertEqual(response["responded_events"], 1)
        self.assertEqual(response["censored_events"], 1)
        self.assertEqual(response["response_rate"], 0.5)
        self.assertEqual(response["responded_delay_frames"]["mean"], 0)
        self.assertIsNone(result["events"][1]["response_delay_frames"])

    def test_next_event_cuts_window_and_short_followup_is_explicit(self):
        gt = [[0, 0], [0.5, 0], [0.5, 0], [0.9, 0], [0.9, 0], [0.9, 0]]
        result = self.evaluate(gt, gt)
        first, second = result["events"]
        self.assertEqual(first["window_frames"], 2)
        self.assertEqual(first["status"], "insufficient_followup")
        self.assertEqual(second["window_frames"], 3)
        self.assertEqual(second["status"], "responded")
        summary = result["summary"]["response"]["all"]
        self.assertEqual(summary["insufficient_followup_events"], 1)
        self.assertEqual(summary["assessable_events"], 1)

    def test_gap_track_end_and_horizon_bound_response_window(self):
        gt = [[0, 0]] + [[0.5, 0]] * 6
        gap = self.evaluate(gt, gt, indices=[1, 2, 3, 7, 8, 9, 10])
        self.assertEqual(gap["events"][0]["window_frames"], 2)
        self.assertEqual(gap["events"][0]["status"], "insufficient_followup")
        short_track = self.evaluate(gt[:3], gt[:3])
        self.assertEqual(short_track["events"][0]["status"], "insufficient_followup")
        horizon = self.evaluate(gt, [[0, 0]] * 3 + [[0.5, 0]] * 4,
                                config=TemporalConfig(max_response_frames=3))
        self.assertEqual(horizon["events"][0]["window_frames"], 3)
        self.assertEqual(horizon["events"][0]["status"], "censored")

    def test_leave_uses_status_and_enter_also_requires_localization(self):
        gt = [[0, 0]] + [[-1, -1]] * 3 + [[0.5, 0]] * 4
        predicted = [[0, 0]] * 8
        scores = [0.9, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
        result = self.evaluate(gt, predicted, labels=[1, 0, 0, 0, 1, 1, 1, 1], scores=scores)
        leave, enter = result["events"]
        self.assertEqual(leave["type"], "leave_frame")
        self.assertEqual(leave["response_delay_frames"], 0)
        self.assertEqual(enter["type"], "enter_frame")
        self.assertEqual(enter["status"], "censored")

    def test_shift_threshold_is_strict_and_response_radius_is_inclusive(self):
        gt = [[0, 0], [0.25, 0], [0.25, 0], [0.25, 0]]
        no_event = self.evaluate(gt, gt, config=TemporalConfig(shift_threshold=0.25))
        self.assertEqual(no_event["events"], [])
        gt = [[0, 0]] + [[0.5, 0]] * 3
        result = self.evaluate(gt, [[0, 0]] + [[0.375, 0]] * 3,
                               config=TemporalConfig(response_radius=0.125))
        self.assertEqual(result["events"][0]["status"], "responded")

    def test_empty_statistics_are_json_null_and_protocol_is_serializable(self):
        result = self.evaluate([[0, 0]], [[0, 0]])
        self.assertIsNone(result["summary"]["jitter"]["predicted"]["mean"])
        self.assertIsNone(result["summary"]["stability"]["stable_ratio"])
        self.assertIsNone(result["summary"]["response"]["all"]["response_rate"])
        text = json.dumps({"protocol": temporal_protocol(TemporalConfig()), "result": result}, allow_nan=False)
        self.assertIn('"mean": null', text)
        self.assertNotIn("NaN", text)


if __name__ == "__main__":
    unittest.main()
