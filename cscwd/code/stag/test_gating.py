"""Synthetic STAG checks: formula, state reset, causal prefixes, and switching."""

import unittest

import numpy as np

from gating import smooth_track


def peak_maps(points):
    heatmaps = np.zeros((len(points), 64, 64), dtype=np.float32)
    for index, (row, column) in enumerate(points):
        heatmaps[index, row, column] = 1
    return heatmaps


class CausalGatingTests(unittest.TestCase):
    def setUp(self):
        self.heatmaps = peak_maps([(0, 0), (0, 32), (0, 0), (32, 32), (0, 32)])
        self.inout = np.array([0.8, 0.7, 0.2, 0.3, 0.9], dtype=np.float32)
        self.boxes = np.array([[0.1, 0.1, 0.3, 0.3]] * 5, dtype=np.float32)
        self.indices = np.arange(1, 6)
        self.weights = [-2, 20, 2, 10]

    def run_track(self, **changes):
        inputs = dict(heatmaps=self.heatmaps, inout=self.inout, bboxes=self.boxes,
                      frame_indices=self.indices, weights=self.weights)
        inputs.update(changes)
        return smooth_track(**inputs)

    def test_first_frame_initialization_and_output_types(self):
        result = self.run_track()
        np.testing.assert_array_equal(result["heatmap"][0], self.heatmaps[0])
        np.testing.assert_array_equal(result["signals"][0], [0, 0, 0])
        self.assertEqual(result["alpha"][0], 1)
        self.assertTrue(result["reset"][0])
        self.assertFalse(result["reset"][1:].any())
        for name in ("heatmap", "alpha", "signals"):
            self.assertEqual(result[name].dtype, np.dtype("float32"))
        self.assertEqual(result["reset"].dtype, np.dtype("bool"))

    def test_manual_gate_and_ema_formula(self):
        boxes = self.boxes[:2].copy()
        boxes[1] += np.array([0.3, 0.4, 0.3, 0.4], dtype=np.float32)
        result = self.run_track(heatmaps=peak_maps([(0, 0), (32, 0)]),
                                inout=[0.9, 0.1], bboxes=boxes,
                                frame_indices=[1, 2], weights=[-2, 1, 2, 3])
        expected_alpha = 1 / (1 + np.exp(-2))
        np.testing.assert_allclose(result["signals"][1], [0.5, 1, 0.5], rtol=1e-6)
        self.assertAlmostEqual(float(result["alpha"][1]), expected_alpha, places=6)
        self.assertAlmostEqual(float(result["heatmap"][1, 32, 0]), expected_alpha, places=6)
        self.assertAlmostEqual(float(result["heatmap"][1, 0, 0]), 1 - expected_alpha, places=6)

    def test_prefix_is_unchanged_by_future_rows(self):
        original = self.run_track()
        prefix = self.run_track(heatmaps=self.heatmaps[:3], inout=self.inout[:3],
                                bboxes=self.boxes[:3], frame_indices=self.indices[:3])
        future_heatmaps = self.heatmaps.copy()
        future_heatmaps[3:] = peak_maps([(63, 63), (62, 62)])
        future_inout = self.inout.copy()
        future_inout[3:] = [0, 1]
        future_boxes = self.boxes.copy()
        future_boxes[3:] = [0.7, 0.7, 0.9, 0.9]
        altered = self.run_track(heatmaps=future_heatmaps, inout=future_inout, bboxes=future_boxes)
        for key in original:
            np.testing.assert_array_equal(original[key][:3], prefix[key])
            np.testing.assert_array_equal(original[key][:3], altered[key][:3])

    def test_gap_resets_history_from_current_frame(self):
        result = self.run_track(frame_indices=[1, 2, 6, 7, 8])
        independent = self.run_track(heatmaps=self.heatmaps[2:], inout=self.inout[2:],
                                     bboxes=self.boxes[2:], frame_indices=[6, 7, 8])
        np.testing.assert_array_equal(result["reset"], [True, False, True, False, False])
        np.testing.assert_array_equal(result["heatmap"][2], self.heatmaps[2])
        self.assertEqual(result["alpha"][2], 1)
        for key in result:
            np.testing.assert_array_equal(result[key][2:], independent[key])

    def test_transition_uses_strict_threshold_on_raw_probabilities(self):
        result = self.run_track(inout=[0.5, 0.5, 0.500001, 0.5, 0.49])
        np.testing.assert_array_equal(result["signals"][:, 1], [0, 0, 1, 1, 0])

    def test_heat_shift_uses_previous_smoothed_map(self):
        result = self.run_track(weights=[np.log(0.1 / 0.9), 0, 0, 0])
        self.assertEqual(np.unravel_index(result["heatmap"][1].argmax(), (64, 64)), (0, 0))
        self.assertEqual(result["signals"][2, 2], 0)
        self.assertEqual(result["signals"][1, 2], 0.5)

    def test_inputs_are_not_modified(self):
        originals = [value.copy() for value in (self.heatmaps, self.inout, self.boxes, self.indices)]
        self.run_track()
        for actual, expected in zip((self.heatmaps, self.inout, self.boxes, self.indices), originals):
            np.testing.assert_array_equal(actual, expected)

    def test_positive_shift_weight_enables_fast_switch_response(self):
        fixed = self.run_track(weights=[-2, 0, 0, 0])
        gated = self.run_track(weights=[-2, 0, 0, 20])
        self.assertLess(fixed["alpha"][1], 0.5)
        self.assertGreater(gated["alpha"][1], 0.99)
        self.assertEqual(np.unravel_index(fixed["heatmap"][1].argmax(), (64, 64)), (0, 0))
        self.assertEqual(np.unravel_index(gated["heatmap"][1].argmax(), (64, 64)), (0, 32))

    def test_sigmoid_is_finite_for_large_positive_and_negative_scores(self):
        for bias, expected in ((-10000, 0), (10000, 1)):
            with self.subTest(bias=bias), np.errstate(over="raise", invalid="raise"):
                result = self.run_track(weights=[bias, 0, 0, 0])
                self.assertTrue(np.isfinite(result["heatmap"]).all())
                np.testing.assert_array_equal(result["alpha"][1:], expected)


if __name__ == "__main__":
    unittest.main()
