"""Hand-computed utility groups, fixed-state scoring and candidate-set oracle."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.history_utility import (CANDIDATE_NAMES, GROUP_NAMES, ORACLE_NAMES,
                                 analyze_history_utility)
from stag.metrics import TemporalConfig, evaluate_temporal
from stag.state_metrics import evaluate_state_temporal


def make_track(labels, start=0, gt=None, indices=None, name="train/video/clip/person"):
    gt = [[0., 0.]] * len(labels) if gt is None else gt
    indices = range(1, len(labels) + 1) if indices is None else indices
    return {"track_id": name, "start": start, "stop": start + len(labels),
            "frames": [{"frame_index": index, "inout": label,
                        "gazex_norm": [point[0]], "gazey_norm": [point[1]]}
                       for label, point, index in zip(labels, gt, indices)]}


def history_mask(tracks):
    mask = np.zeros(sum(len(t["frames"]) for t in tracks), dtype=bool)
    for track in tracks:
        indices = [frame["frame_index"] for frame in track["frames"]]
        mask[track["start"]:track["stop"]] = np.r_[False, np.diff(indices) == 1]
    return mask


class HistoryUtilityTests(unittest.TestCase):
    def example(self):
        tracks = [make_track([1] * 5)]
        xy = np.zeros((5, len(CANDIDATE_NAMES), 2))
        xy[:, :, 0] = np.asarray([0., 0., .2, 0., .2])[:, None]
        xy[:, CANDIDATE_NAMES.index("previous_raw"), 0] = [0., 0., 0., .2, .2]
        scores = np.asarray([.9, .9, .5, .9, .9])
        return tracks, xy, scores, history_mask(tracks)

    def test_four_quadrants_gain_and_raw_out_blocking_are_hand_counted(self):
        result = analyze_history_utility(*self.example())
        summary = result["summary"]
        utility = summary["candidates"]["previous_raw"]["utility"]
        self.assertEqual(utility["frames"], 4)
        self.assertEqual(utility["localization"], {
            "both_correct": 1, "current_only_correct": 1,
            "history_only_correct": 1, "both_wrong": 1})
        self.assertEqual(utility["fixed_inout_joint"], {
            "both_correct": 1, "current_only_correct": 1,
            "history_only_correct": 0, "both_wrong": 2})
        self.assertEqual(utility["rescue_blocked_by_raw_out"], 1)
        self.assertEqual(utility["net_rescue_minus_harm"], 0)
        self.assertEqual([utility[key] for key in
                          ("candidate_better", "candidate_equal", "candidate_worse")], [1, 2, 1])
        self.assertAlmostEqual(utility["l2_gain"]["mean"], 0.)
        self.assertEqual(summary["coverage"]["utility_frames"], 4)
        self.assertEqual(summary["candidates"]["current"]["utility"]["localization"], {
            "both_correct": 2, "current_only_correct": 0,
            "history_only_correct": 0, "both_wrong": 2})
        self.assertTrue(all(summary["checks"].values()))

    def test_group_boundaries_enter_and_gap_priority_partition_frames(self):
        specifications = [
            ([0, 1], [[-1., -1.], [0., 0.]], [1, 2]),
            ([1, 1], [[0., 0.], [.02, 0.]], [1, 2]),
            ([1, 1], [[0., 0.], [.15, 0.]], [1, 2]),
            ([1, 1], [[0., 0.], [.150001, 0.]], [1, 2]),
            ([1, 0, 1], [[0., 0.], [-1., -1.], [0., 0.]], [1, 3, 4]),
            ([1, 0], [[0., 0.], [-1., -1.]], [1, 2]),
        ]
        tracks, start = [], 0
        for number, (labels, gt, indices) in enumerate(specifications):
            tracks.append(make_track(labels, start, gt, indices, name=f"train/{number}"))
            start += len(labels)
        result = analyze_history_utility(tracks, np.zeros((start, 5, 2)),
                                         np.full(start, .9), history_mask(tracks))
        self.assertEqual(result["arrays"]["group_codes"].tolist(),
                         [0, 2, 0, 3, 0, 4, 0, 5, 0, 0, 2, 0, 1])
        coverage = result["summary"]["coverage"]
        self.assertEqual(coverage["group_counts"], dict(zip(GROUP_NAMES, [7, 1, 2, 1, 1, 1])))
        self.assertEqual(coverage["utility_frames"], 5)
        groups = result["summary"]["candidates"]["previous_raw"]["groups"]
        self.assertEqual([groups[name]["frames"] for name in GROUP_NAMES], [0, 0, 2, 1, 1, 1])
        self.assertTrue(np.isnan(result["arrays"]["errors"][[0, 9, 12]]).all())

    def test_oracle_excludes_grid_control_and_ties_prefer_current(self):
        tracks, xy, scores, mask = self.example()
        xy[:, 1] = 0.  # This reference would also rescue the final error.
        result = analyze_history_utility(tracks, xy, scores, mask)
        oracle = result["summary"]["oracle"]
        self.assertEqual(oracle["candidate_names"], list(ORACLE_NAMES))
        self.assertEqual(result["arrays"]["oracle_choice"].tolist(), [0, 0, 2, 0, 0])
        self.assertEqual(oracle["choice_counts"], {
            "current": 4, "previous_raw": 1, "ema_075": 0, "transport_previous": 0})
        self.assertAlmostEqual(oracle["l2"], .04)
        self.assertAlmostEqual(oracle["current_l2"], .08)
        self.assertEqual(oracle["current_wrong_frames"], 2)
        self.assertEqual(oracle["rescued_current_errors"], 1)
        self.assertEqual(oracle["utility_history_available"]["fixed_inout_joint_rescues"], 0)
        self.assertEqual(oracle["utility_history_available"]["rescue_blocked_by_raw_out"], 1)
        self.assertNotIn("temporal", oracle)

    def test_existing_temporal_and_state_functions_are_reused_exactly(self):
        tracks = [make_track([0, 1, 1, 1, 1, 0, 0, 0],
                             gt=[[0., 0.]] * 4 + [[.3, 0.]] * 4)]
        xy = np.zeros((8, 5, 2))
        xy[3:, 2, 0] = .3
        scores = np.asarray([.1, .5, .9, .9, .9, .1, .2, .1])
        config = TemporalConfig(hold_frames=2)
        result = analyze_history_utility(tracks, xy, scores, history_mask(tracks), config)
        state = evaluate_state_temporal(tracks, scores, config)
        for column, name in enumerate(CANDIDATE_NAMES):
            actual = result["summary"]["candidates"][name]
            temporal = evaluate_temporal(tracks, xy[:, column], scores, config)
            self.assertEqual(actual["temporal"], temporal["summary"])
            self.assertEqual(actual["state"], state["summary"])
            self.assertEqual(result["details"]["candidates"][name]["events"], temporal["events"])
            self.assertNotIn("auc", actual)
            self.assertNotIn("inout_ap", actual)
        self.assertEqual(result["details"]["state"]["events"], state["events"])
        self.assertTrue(result["summary"]["checks"]["candidate_leave_responses_equal_current"])

    def test_localization_radius_is_inclusive_and_state_half_is_out(self):
        tracks = [make_track([1, 1, 1])]
        xy = np.zeros((3, 5, 2))
        xy[:, :, 0] = .1
        xy[2, 2, 0] = .100001
        result = analyze_history_utility(tracks, xy, [.9, .5, .500001], history_mask(tracks))
        utility = result["summary"]["candidates"]["previous_raw"]["utility"]
        self.assertEqual(utility["localization"]["both_correct"], 1)
        self.assertEqual(utility["localization"]["current_only_correct"], 1)
        self.assertEqual(utility["fixed_inout_joint"]["both_wrong"], 1)
        self.assertEqual(utility["fixed_inout_joint"]["current_only_correct"], 1)

    def test_all_gt_out_and_empty_utility_groups_have_explicit_empty_values(self):
        tracks = [make_track([0, 0, 0], gt=[[-1., -1.]] * 3)]
        result = analyze_history_utility(tracks, np.zeros((3, 5, 2)),
                                         np.full(3, .2), history_mask(tracks))
        summary = result["summary"]
        self.assertEqual(summary["coverage"]["utility_frames"], 0)
        self.assertEqual(summary["inout_ap"], 0.)
        self.assertIsNone(summary["oracle"]["l2"])
        self.assertEqual(result["arrays"]["oracle_choice"].tolist(), [-1, -1, -1])
        for candidate in summary["candidates"].values():
            self.assertIsNone(candidate["l2"])
            self.assertEqual(candidate["utility"]["frames"], 0)
            self.assertIsNone(candidate["utility"]["l2_gain"]["mean"])
        self.assertTrue(np.isnan(result["arrays"]["errors"]).all())

    def test_metadata_history_and_start_gap_fallback_are_verified(self):
        tracks, xy, scores, mask = self.example()
        with self.assertRaisesRegex(ValueError, "History availability differs"):
            analyze_history_utility(tracks, xy, scores, ~mask)
        xy[0, 2, 0] = .5
        with self.assertRaisesRegex(ValueError, "fallback_to_current_without_history"):
            analyze_history_utility(tracks, xy, scores, mask)

    def test_inputs_remain_read_only_and_details_are_plain_json(self):
        tracks, xy, scores, mask = self.example()
        before_tracks, before_xy, before_scores, before_mask = copy.deepcopy(tracks), xy.copy(), scores.copy(), mask.copy()
        for values in (xy, scores, mask):
            values.flags.writeable = False
        result = analyze_history_utility(tracks, xy, scores, mask)
        self.assertEqual(tracks, before_tracks)
        np.testing.assert_array_equal(xy, before_xy)
        np.testing.assert_array_equal(scores, before_scores)
        np.testing.assert_array_equal(mask, before_mask)
        json.dumps(result["summary"], allow_nan=False)
        json.dumps(result["details"], allow_nan=False)

    def test_scoring_import_does_not_require_torch(self):
        code_root = str(Path(__file__).resolve().parents[1])
        script = ("import builtins, sys\n"
                  f"sys.path.insert(0, {code_root!r})\n"
                  "original = builtins.__import__\n"
                  "def guarded(name, *args, **kwargs):\n"
                  "    if name == 'torch' or name.startswith('torch.'):\n"
                  "        raise AssertionError('Torch import in scoring')\n"
                  "    return original(name, *args, **kwargs)\n"
                  "builtins.__import__ = guarded\n"
                  "from stag.history_utility import analyze_history_utility\n")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
