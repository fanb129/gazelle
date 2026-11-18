"""Numerical and fitting checks for the fixed train-only calibration control."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.state_calibration import (BIAS_BOUNDS, apply_state_bias, fit_state_bias,
                                    ranking_checks)


class StateCalibrationTests(unittest.TestCase):
    def test_recovers_known_log_odds_offset_from_empirical_prevalence(self):
        scores = np.full(100, .2, dtype=np.float32)
        labels = np.r_[np.ones(60), np.zeros(40)]
        result = fit_state_bias(scores, labels)
        expected = np.log(.6 / .4) - np.log(float(scores[0]) / (1 - float(scores[0])))
        self.assertAlmostEqual(result["bias"], expected, places=12)
        self.assertAlmostEqual(result["optimizer"]["derivative_per_all_frames"], 0, places=14)
        self.assertLess(result["train_after"]["full_nll"], result["train_before"]["full_nll"])
        self.assertEqual(result["optimizer"]["status"], "interior_optimum")

    def test_heterogeneous_scores_obey_full_sample_mean_stationarity(self):
        scores = np.asarray([.01, .1, .2, .4, .7, .8, .9, .99], np.float32)
        labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
        result = fit_state_bias(scores, labels)
        calibrated = apply_state_bias(scores, result["bias"])
        self.assertAlmostEqual(calibrated.mean(), labels.mean(), places=14)
        self.assertLessEqual(result["train_after"]["variable_nll"], result["train_before"]["variable_nll"])

    def test_extreme_float32_scores_endpoints_and_ties_preserve_ranking(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        last = np.nextafter(np.float32(1), np.float32(0))
        before_last = np.nextafter(last, np.float32(0))
        scores = np.asarray([0, tiny, 2*tiny, 1e-30, .01, .5, .5,
                             .75, before_last, last, 1, 1], np.float32)
        labels = np.arange(len(scores)) % 2
        for bias in (*BIAS_BOUNDS, 0.0, -3.25, 2.75):
            with self.subTest(bias=bias):
                result = apply_state_bias(scores, bias)
                self.assertEqual(result.dtype, np.float64)
                self.assertEqual(result[0], 0)
                self.assertEqual(result[-1], 1)
                self.assertTrue(all(ranking_checks(scores, result, labels)["checks"].values()))

    def test_dense_float32_near_one_does_not_saturate_into_ties(self):
        # Adjacent float32 values near one are the hardest case for positive b.
        bits = np.arange(0x3f7f0000, 0x3f800001, dtype=np.uint32)
        scores = bits.view(np.float32)
        for bias in BIAS_BOUNDS:
            with self.subTest(bias=bias):
                calibrated = apply_state_bias(scores, bias)
                self.assertTrue((np.diff(calibrated) > 0).all())

    def test_zero_bias_is_exact_identity_and_mapping_does_not_mutate_input(self):
        scores = np.asarray([0, .01, .2, .5, .99, 1], np.float32)
        copy = scores.copy()
        np.testing.assert_array_equal(apply_state_bias(scores, 0), scores.astype(np.float64))
        np.testing.assert_array_equal(scores, copy)

    def test_fixed_wrong_endpoints_are_explicit_infinite_constant(self):
        scores = np.asarray([0, 0, 1, 1, .2, .2, .2, .2], np.float32)
        labels = np.asarray([0, 1, 0, 1, 1, 1, 0, 0])
        result = fit_state_bias(scores, labels)
        for key in ("train_before", "train_after"):
            self.assertIsNone(result[key]["full_nll"])
            self.assertFalse(result[key]["full_nll_finite"])
            self.assertEqual(result[key]["fixed_endpoint_frames"], 4)
            self.assertEqual(result[key]["uncalibratable_endpoint_errors"], 2)
        self.assertAlmostEqual(result["train_after"]["variable_nll"], .5*np.log(2), places=14)
        self.assertAlmostEqual(result["bias"], np.log(4), places=6)

    def test_correct_endpoints_have_zero_loss_and_do_not_change_fit(self):
        scores = np.asarray([.1, .3, .7, .8], np.float32)
        labels = np.asarray([0, 0, 1, 0])
        original = fit_state_bias(scores, labels)
        extended = fit_state_bias(np.r_[scores, 0, 1], np.r_[labels, 0, 1])
        self.assertAlmostEqual(original["bias"], extended["bias"], places=14)
        self.assertTrue(extended["train_after"]["full_nll_finite"])
        self.assertAlmostEqual(extended["train_after"]["full_nll"],
                               original["train_after"]["full_nll"] * 4 / 6, places=14)

    def test_all_endpoint_input_leaves_bias_at_zero(self):
        result = fit_state_bias(np.asarray([0, 1, 0, 1]), np.asarray([0, 1, 1, 0]))
        self.assertEqual(result["bias"], 0)
        self.assertEqual(result["optimizer"]["status"], "no_adjustable_scores")
        self.assertEqual(result["train_after"]["variable_nll"], 0)

    def test_single_class_interior_scores_hit_declared_bound(self):
        scores = np.asarray([.2, .5, .8], np.float32)
        for label, expected_bias, status in ((0, -16, "lower_bound"), (1, 16, "upper_bound")):
            with self.subTest(label=label):
                result = fit_state_bias(scores, np.full(3, label))
                self.assertEqual(result["bias"], expected_bias)
                self.assertEqual(result["optimizer"]["status"], status)
                self.assertLess(result["train_after"]["full_nll"], result["train_before"]["full_nll"])

    def test_ranking_checks_detect_new_ties_and_broken_existing_ties(self):
        original = np.asarray([.1, .2, .2, .3], np.float32)
        labels = np.asarray([0, 1, 0, 1])
        new_ties = ranking_checks(original, np.asarray([.1, .1, .1, .3]), labels)
        self.assertFalse(new_ties["checks"]["strict_order_preserved"])
        broken_ties = ranking_checks(original, np.asarray([.1, .2, .21, .3]), labels)
        self.assertFalse(broken_ties["checks"]["original_ties_preserved"])


if __name__ == "__main__":
    unittest.main()
