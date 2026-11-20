"""Hand-computed cached-output replacement effects and T4 protocol agreement."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.cache_diagnosis import diagnose_prediction_pair
from stag.metrics import TemporalConfig, evaluate_temporal
from stag.state_metrics import evaluate_state_temporal


def track(labels, indices=None, start=0, name="test/video/clip/person", gt=None):
    indices = indices if indices is not None else list(range(1, len(labels) + 1))
    gt = gt if gt is not None else [[0., 0.]] * len(labels)
    return {"track_id": name, "start": start, "stop": start + len(labels),
            "frames": [{"frame_index": i, "inout": label,
                        "gazex_norm": [point[0]], "gazey_norm": [point[1]]}
                       for i, label, point in zip(indices, labels, gt)]}


class CachedOutputDiagnosisTests(unittest.TestCase):
    def diagnose(self, raw_xy=None, raw_scores=None, refined_xy=None, refined_scores=None):
        raw_xy = np.zeros((3, 2)) if raw_xy is None else np.asarray(raw_xy)
        refined_xy = raw_xy.copy() if refined_xy is None else np.asarray(refined_xy)
        raw_scores = np.full(3, .9) if raw_scores is None else np.asarray(raw_scores)
        refined_scores = raw_scores.copy() if refined_scores is None else np.asarray(refined_scores)
        return diagnose_prediction_pair([track([1, 1, 1])], raw_xy, raw_scores,
                                        refined_xy, refined_scores)

    def assert_counts(self, result, expected):
        attribution = result["attribution"]
        self.assertEqual(attribution["accurate_stable_counts"], dict(zip(("M00", "M10", "M01", "M11"), expected)))
        self.assertEqual(attribution["total_delta"]["window_count"], expected[3] - expected[0])
        self.assertEqual(sum(attribution[k]["window_count"] for k in
                             ("coordinate_effect_at_raw_score", "score_effect_at_raw_xy", "interaction")),
                         attribution["total_delta"]["window_count"])
        self.assertEqual(sum(attribution["shapley"][k]["window_count"] for k in ("coordinate", "score")),
                         attribution["total_delta"]["window_count"])
        for matrix, variant in attribution["matrix_definition"].items():
            self.assertEqual(attribution["accurate_stable_counts"][matrix],
                             result["variants"][variant]["temporal"]["stability"]["accurate_stable_frames"])

    def test_coordinate_only_loss_has_localization_failure_and_zero_score_effect(self):
        result = self.diagnose(refined_xy=[[.5, 0]] * 3)
        self.assert_counts(result, [1, 0, 1, 0])
        attr = result["attribution"]
        self.assertEqual(attr["coordinate_effect_at_raw_score"]["window_count"], -1)
        self.assertEqual(attr["score_effect_at_raw_xy"]["window_count"], 0)
        self.assertEqual(attr["interaction"]["window_count"], 0)
        self.assertEqual(attr["window_transitions"]["lost_refined_failure_masks"]["2"], 1)

    def test_score_only_loss_preserves_coordinates_and_has_state_failure(self):
        result = self.diagnose(refined_scores=[.9, .5, .9])
        self.assert_counts(result, [1, 1, 0, 0])
        attr = result["attribution"]
        self.assertEqual(attr["coordinate_effect_at_raw_score"]["window_count"], 0)
        self.assertEqual(attr["score_effect_at_raw_xy"]["window_count"], -1)
        self.assertEqual(attr["window_transitions"]["lost_refined_failure_masks"]["4"], 1)

    def test_two_repairs_needed_create_positive_interaction_and_half_window_shapley(self):
        result = self.diagnose(raw_xy=[[.5, 0]] * 3, raw_scores=[.1] * 3,
                               refined_xy=[[0., 0.]] * 3, refined_scores=[.9] * 3)
        self.assert_counts(result, [0, 0, 0, 1])
        attr = result["attribution"]
        self.assertEqual(attr["interaction"]["window_count"], 1)
        self.assertEqual(attr["shapley"]["coordinate"]["window_count"], .5)
        self.assertEqual(attr["shapley"]["score"]["window_count"], .5)
        self.assertEqual(attr["window_transitions"]["gained_raw_failure_masks"]["6"], 1)

    def test_coordinate_repair_cancelled_by_score_has_negative_interaction(self):
        result = self.diagnose(raw_xy=[[.5, 0]] * 3, refined_xy=[[0., 0.]] * 3,
                               refined_scores=[.1] * 3)
        self.assert_counts(result, [0, 1, 0, 0])
        attr = result["attribution"]
        self.assertEqual(attr["interaction"]["window_count"], -1)
        self.assertEqual(attr["shapley"]["coordinate"]["window_count"], .5)
        self.assertEqual(attr["shapley"]["score"]["window_count"], -.5)

    def test_three_overlapping_failures_are_one_mutually_exclusive_bit_pattern(self):
        result = self.diagnose(refined_xy=[[0., 0.], [.2, 0.], [.4, 0.]], refined_scores=[.9, .1, .9])
        self.assert_counts(result, [1, 0, 0, 0])
        transitions = result["attribution"]["window_transitions"]
        self.assertEqual(transitions["lost"], 1)
        self.assertEqual(transitions["lost_refined_failure_masks"]["7"], 1)
        self.assertEqual(sum(transitions["lost_refined_failure_masks"].values()), transitions["lost"])

    def test_lost_and_gained_windows_keep_track_boundaries_and_net_delta(self):
        tracks = [track([1]*3, name="first"), track([1]*3, start=3, name="second")]
        raw_xy = np.asarray([[0., 0.]]*3 + [[.5, 0.]]*3)
        refined_xy = np.asarray([[.5, 0.]]*3 + [[0., 0.]]*3)
        result = diagnose_prediction_pair(tracks, raw_xy, np.full(6, .9), refined_xy, np.full(6, .9))
        self.assert_counts(result, [1, 1, 1, 1])
        attr = result["attribution"]
        self.assertEqual(attr["eligible_windows"], 2)
        self.assertEqual(attr["window_transitions"]["lost"], 1)
        self.assertEqual(attr["window_transitions"]["gained"], 1)
        self.assertEqual(attr["total_delta"]["ratio"], 0.)

    def test_gap_out_of_frame_and_short_people_do_not_create_windows(self):
        tracks = [track([1]*5, indices=[1, 2, 10, 11, 12]),
                  track([1]*2, start=5, name="short"),
                  track([1, 0, 1], start=7, name="mixed")]
        result = diagnose_prediction_pair(tracks, np.zeros((10, 2)), np.full(10, .9),
                                          np.zeros((10, 2)), np.full(10, .9))
        self.assert_counts(result, [1, 1, 1, 1])
        self.assertEqual(result["attribution"]["eligible_windows"], 1)
        self.assertEqual(result["variants"]["raw_xy_raw_score"]["standard"]["localization_frames"], 9)

    def test_endpoint_and_cross_metrics_reuse_exact_old_protocol(self):
        tracks = [track([0, 1, 1, 1, 1, 0, 0, 0], gt=[[0, 0]]*4 + [[.3, 0]]*4)]
        raw_xy = np.zeros((8, 2)); refined_xy = np.asarray([[0., 0.]]*3 + [[.3, 0.]]*5)
        raw = np.asarray([.1, .9, .8, .7, .9, .2, .1, .1]); refined = np.asarray([.1, .2, .9, .9, .9, .1, .2, .1])
        config = TemporalConfig(hold_frames=2)
        result = diagnose_prediction_pair(tracks, raw_xy, raw, refined_xy, refined, config)
        for name, xy, scores in (("raw_xy_raw_score", raw_xy, raw),
                                 ("refined_xy_raw_score", refined_xy, raw),
                                 ("raw_xy_refined_score", raw_xy, refined),
                                 ("refined_xy_refined_score", refined_xy, refined)):
            item = result["variants"][name]
            self.assertEqual(item["temporal"], evaluate_temporal(tracks, xy, scores, config)["summary"])
            self.assertEqual(item["state"], evaluate_state_temporal(tracks, scores, config)["summary"])
            self.assertNotIn("auc", item["standard"])
            self.assertEqual(item["standard"]["inout_ap"], average_precision_score([0,1,1,1,1,0,0,0], scores))
        self.assertEqual(result["details"]["raw_xy_raw_score"]["joint"]["events"],
                         evaluate_temporal(tracks, raw_xy, raw, config)["events"])
        self.assertEqual(result["details"]["refined_xy_refined_score"]["state"]["events"],
                         evaluate_state_temporal(tracks, refined, config)["events"])
        self.assert_counts(result, [result["variants"][v]["temporal"]["stability"]["accurate_stable_frames"]
                                    for v in ("raw_xy_raw_score", "refined_xy_raw_score", "raw_xy_refined_score", "refined_xy_refined_score")])

    def test_prediction_changes_include_strict_threshold_and_gt_class_corrections(self):
        tracks = [track([1, 1, 0, 0, 1, 0])]
        raw_xy = np.zeros((6, 2)); refined_xy = raw_xy.copy(); refined_xy[0,0] = .1; refined_xy[3,0] = .2
        result = diagnose_prediction_pair(tracks, raw_xy, np.asarray([.5,.9,.5,.9,.1,.1]),
            refined_xy, np.asarray([.6,.5,.4,.2,.5,.9]))
        changes = result["prediction_changes"]
        self.assertEqual(changes["all"]["coordinates_changed"], 2)
        self.assertAlmostEqual(changes["all"]["coordinate_displacement"]["mean"], .05)
        self.assertEqual(changes["all"]["crossed_to_in_frame"], 2)
        self.assertEqual(changes["all"]["crossed_to_out_of_frame"], 2)
        for name in ("gt_in_frame", "gt_out_of_frame"):
            self.assertEqual(changes[name]["state_errors_corrected"], 1)
            self.assertEqual(changes[name]["state_errors_introduced"], 1)
        self.assertEqual(changes["all"]["state_errors_corrected"], 2)
        self.assertEqual(changes["all"]["state_errors_introduced"], 2)

    def test_only_track_slices_are_scored_and_arrays_and_labels_stay_unchanged(self):
        tracks = [track([1, 1, 1], start=2)]
        xy=np.asarray([[99.,99.]]*2 + [[0.,0.]]*3 + [[99.,99.]])
        raw=np.asarray([.1,.1,.9,.9,.9,.1]); refined=raw.copy()
        for values in (xy,raw,refined): values.setflags(write=False)
        before=copy.deepcopy(tracks)
        result=diagnose_prediction_pair(tracks,xy,raw,xy,refined)
        self.assertEqual(result["variants"]["raw_xy_raw_score"]["standard"]["person_frames"],3)
        self.assertEqual(result["variants"]["raw_xy_raw_score"]["standard"]["l2"],0.)
        self.assertEqual(result["prediction_changes"]["all"]["person_frames"],3)
        self.assertEqual(tracks,before)
        self.assertEqual(xy[0,0],99.)

    def test_no_eligible_window_has_null_ratios_and_json_serializes(self):
        result=diagnose_prediction_pair([track([1,0])],np.zeros((2,2)),np.asarray([.9,.1]),
                                        np.zeros((2,2)),np.asarray([.8,.2]))
        self.assert_counts(result,[0,0,0,0])
        self.assertEqual(result["attribution"]["eligible_windows"],0)
        self.assertIsNone(result["attribution"]["total_delta"]["ratio"])
        self.assertIn('"ratio": null',json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    unittest.main()
