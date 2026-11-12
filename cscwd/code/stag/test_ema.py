"""Synthetic checks for fixed-alpha causal smoothing and its STAG control."""

import unittest

import numpy as np

from ema import smooth_ema_track
from gating import smooth_track


class FixedEMATests(unittest.TestCase):
    def setUp(self):
        self.heatmaps = np.random.default_rng(17).random((6, 64, 64), dtype=np.float32)
        self.indices = np.arange(10, 16)

    def test_manual_recurrence_and_output_types(self):
        heatmaps = np.stack([np.full((64, 64), value, dtype=np.float32) for value in (0, 1, 0)])
        result = smooth_ema_track(heatmaps, [1, 2, 3], 0.25)
        np.testing.assert_array_equal(result["heatmap"][:, 0, 0], [0, 0.25, 0.1875])
        np.testing.assert_array_equal(result["alpha"], [1, 0.25, 0.25])
        np.testing.assert_array_equal(result["reset"], [True, False, False])
        self.assertEqual(result["heatmap"].dtype, np.dtype("float32"))
        self.assertEqual(result["alpha"].dtype, np.dtype("float32"))
        self.assertEqual(result["reset"].dtype, np.dtype("bool"))

    def test_future_changes_do_not_change_prefix(self):
        full = smooth_ema_track(self.heatmaps, self.indices, 0.25)
        prefix = smooth_ema_track(self.heatmaps[:3], self.indices[:3], 0.25)
        future = self.heatmaps.copy()
        future[3:] = 0
        altered = smooth_ema_track(future, self.indices, 0.25)
        for key in full:
            np.testing.assert_array_equal(full[key][:3], prefix[key])
            np.testing.assert_array_equal(full[key][:3], altered[key][:3])

    def test_gap_resets_without_bridging_old_history(self):
        indices = [1, 2, 3, 9, 10, 11]
        result = smooth_ema_track(self.heatmaps, indices, 0.5)
        independent = smooth_ema_track(self.heatmaps[3:], indices[3:], 0.5)
        np.testing.assert_array_equal(result["reset"], [True, False, False, True, False, False])
        for key in result:
            np.testing.assert_array_equal(result[key][3:], independent[key])

    def test_separate_tracks_start_with_their_own_observation(self):
        first = smooth_ema_track(self.heatmaps[:3], [1, 2, 3], 0.25)
        second = smooth_ema_track(self.heatmaps[3:], [1, 2, 3], 0.25)
        np.testing.assert_array_equal(first["heatmap"][0], self.heatmaps[0])
        np.testing.assert_array_equal(second["heatmap"][0], self.heatmaps[3])
        self.assertEqual(second["alpha"][0], 1)
        self.assertTrue(second["reset"][0])

    def test_inputs_are_not_modified(self):
        original_maps, original_indices = self.heatmaps.copy(), self.indices.copy()
        smooth_ema_track(self.heatmaps, self.indices, 0.75)
        np.testing.assert_array_equal(self.heatmaps, original_maps)
        np.testing.assert_array_equal(self.indices, original_indices)

    def test_alpha_zero_keeps_segment_start_heatmap(self):
        result = smooth_ema_track(self.heatmaps, [1, 2, 3, 9, 10, 11], 0)
        np.testing.assert_array_equal(result["heatmap"][:3], np.repeat(self.heatmaps[:1], 3, axis=0))
        np.testing.assert_array_equal(result["heatmap"][3:], np.repeat(self.heatmaps[3:4], 3, axis=0))

    def test_alpha_one_preserves_all_single_frame_heatmaps(self):
        result = smooth_ema_track(self.heatmaps, self.indices, 1)
        np.testing.assert_array_equal(result["heatmap"], self.heatmaps)

    def test_bias_matched_alpha_is_exactly_stag_without_gate_terms(self):
        alpha = np.exp(-np.logaddexp(0.0, 2.0))
        indices = [1, 2, 3, 9, 10, 11]
        result = smooth_ema_track(self.heatmaps, indices, alpha)
        gated = smooth_track(self.heatmaps, [0.9, 0.1, 0.8, 0.2, 0.6, 0.4],
                             [[0.1, 0.1, 0.3, 0.3]] * 6, indices, [-2, 0, 0, 0])
        for key in result:
            np.testing.assert_array_equal(result[key], gated[key])

    def test_invalid_alpha_is_rejected(self):
        for alpha in (-0.1, 1.1, np.nan):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "alpha must be in"):
                smooth_ema_track(self.heatmaps, self.indices, alpha)


if __name__ == "__main__":
    unittest.main()
