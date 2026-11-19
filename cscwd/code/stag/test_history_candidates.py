"""Causal raw-history candidates, directed transport and original coordinates."""

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.ema import smooth_ema_track
from stag.history_candidates import (CANDIDATE_NAMES, VERSION,
                                     maps_to_coordinates, step_candidates)


class HistoryCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_fixed_interfaces_fp32_maps_and_raw_only_state(self):
        self.assertEqual(tuple(inspect.signature(step_candidates).parameters),
                         ("current_heatmap", "current_features", "frame_index", "state"))
        self.assertEqual(VERSION, "r4_history_utility_v1")
        raw = torch.rand(64, 64, dtype=torch.float64)
        features = torch.rand(3, 2, 2, dtype=torch.float16)
        maps, state, available = step_candidates(raw, features, 12)
        self.assertFalse(available)
        self.assertEqual(tuple(maps), CANDIDATE_NAMES)
        self.assertEqual(set(state), {"previous_heatmap", "previous_features", "ema", "frame_index"})
        for value in maps.values():
            self.assertEqual(value.shape, (64, 64))
            self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.equal(state["previous_heatmap"], raw.float()))
        self.assertTrue(torch.equal(state["previous_features"], features.float()))

    def test_start_and_gap_fall_back_to_current_for_all_history_candidates(self):
        features = torch.randn(3, 2, 2)
        old = torch.rand(64, 64)
        _, state, _ = step_candidates(old, features, 8)
        raw = torch.rand(64, 64)
        for previous in (None, state):
            maps, next_state, available = step_candidates(raw, features, 11, previous)
            self.assertFalse(available)
            for name in ("previous_raw", "ema_075", "transport_previous"):
                self.assertTrue(torch.equal(maps[name], raw))
            self.assertEqual(next_state["frame_index"], 11)
            self.assertTrue(torch.equal(next_state["ema"], raw))

    def test_new_person_cannot_inherit_previous_person_state(self):
        features = torch.randn(3, 2, 2)
        raw = torch.rand(64, 64)
        _, state, _ = step_candidates(torch.ones(64, 64), features, 1)
        inherited, _, inherited_available = step_candidates(raw, features, 2, state)
        fresh, _, fresh_available = step_candidates(raw, features, 2, None)
        self.assertTrue(inherited_available)
        self.assertFalse(fresh_available)
        self.assertFalse(torch.equal(inherited["previous_raw"], fresh["previous_raw"]))
        self.assertTrue(torch.equal(fresh["previous_raw"], raw))

    def test_ema_matches_original_numpy_implementation_including_gaps(self):
        heatmaps = torch.rand(6, 64, 64)
        indices = np.array([3, 4, 5, 9, 10, 11])
        expected = smooth_ema_track(heatmaps.numpy(), indices, .75)["heatmap"]
        state = None
        actual = []
        for raw, index in zip(heatmaps, indices):
            maps, state, available = step_candidates(raw, torch.eye(4).reshape(4, 2, 2), int(index), state)
            actual.append(maps["ema_075"].numpy())
            self.assertEqual(available, index not in (3, 9))
        np.testing.assert_array_equal(np.stack(actual), expected)

    def test_source_to_target_transport_moves_mass_to_permuted_feature(self):
        # The previous left feature reappears on the current right. All mass
        # starts left; a mistaken softmax direction would violate this result.
        previous_features = torch.tensor([[[1., 0.]], [[0., 1.]]])
        current_features = previous_features.flip(-1)
        previous_map = torch.zeros(64, 64)
        previous_map[:, :32] = 1
        _, state, _ = step_candidates(previous_map, previous_features, 1)
        maps, _, available = step_candidates(torch.rand(64, 64), current_features, 2, state)
        weights = torch.tensor([0., 10.]).softmax(dim=0).reshape(1, 1, 1, 2)
        expected = F.interpolate(weights, (64, 64), mode="bilinear", align_corners=False)[0, 0]
        expected = expected / expected.sum()
        self.assertTrue(available)
        torch.testing.assert_close(maps["transport_previous"], expected, atol=1e-10, rtol=1e-5)
        # Bilinear upsampling spreads a ramp across the patch boundary.
        self.assertGreater(maps["transport_previous"][:, 32:].sum().item(), .85)
        self.assertAlmostEqual(maps["transport_previous"].sum().item(), 1., places=6)

    def test_transport_uses_source_normalization_for_many_to_one_matches(self):
        # Both source patches favor current patch 0, with different certainty.
        # A transpose/target-normalized average is a different operation.
        source = torch.tensor([[[1., .6]], [[0., .8]]])
        target = torch.tensor([[[1., -1.]], [[0., 0.]]])
        raw = torch.zeros(64, 64)
        raw[:, :32], raw[:, 32:] = .25, .75
        _, state, _ = step_candidates(raw, source, 1)
        maps, _, _ = step_candidates(torch.zeros(64, 64), target, 2, state)
        correspondence = torch.tensor([[10., -10.], [6., -6.]]).softmax(dim=1)
        mass = torch.tensor([.25, .75]) @ correspondence
        expected = F.interpolate(mass.reshape(1, 1, 1, 2), (64, 64),
                                 mode="bilinear", align_corners=False)[0, 0]
        expected /= expected.sum()
        torch.testing.assert_close(maps["transport_previous"], expected, atol=1e-10, rtol=1e-5)

    def test_grid_control_has_prescribed_downsample_and_no_history(self):
        raw = torch.zeros(64, 64)
        raw[0, 0] = 1
        maps, _, _ = step_candidates(raw, torch.ones(2, 2, 2), 1)
        grid = torch.tensor([[[[1 / 1024, 0.], [0., 0.]]]])
        expected = F.interpolate(grid, (64, 64), mode="bilinear", align_corners=False)[0, 0]
        self.assertTrue(torch.equal(maps["current"], raw))
        torch.testing.assert_close(maps["current_grid"], expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(maps["current_grid"], raw))

    def test_transport_does_not_feed_back_and_state_is_not_an_input_alias(self):
        features = torch.eye(4).reshape(4, 2, 2)
        raw1, raw2 = torch.rand(64, 64), torch.rand(64, 64)
        _, state1, _ = step_candidates(raw1, features, 1)
        maps2, state2, _ = step_candidates(raw2, features.flip(-1), 2, state1)
        untouched_raw2 = raw2.clone()
        self.assertTrue(torch.equal(state2["previous_heatmap"], untouched_raw2))
        maps2["transport_previous"].fill_(100)
        raw2.zero_()
        features.zero_()
        maps3, _, _ = step_candidates(torch.rand(64, 64), torch.ones(4, 2, 2), 3, state2)
        self.assertTrue(torch.equal(maps3["previous_raw"], untouched_raw2))
        self.assertTrue(torch.isfinite(maps3["transport_previous"]).all())
        self.assertAlmostEqual(maps3["transport_previous"].sum().item(), 1., places=6)

    def test_future_changes_never_change_previous_candidates(self):
        heatmaps = torch.rand(5, 64, 64)
        features = torch.rand(5, 3, 2, 2)
        def rollout(raws, observed_features):
            state = None
            result = []
            for index, (raw, feature) in enumerate(zip(raws, observed_features)):
                maps, state, _ = step_candidates(raw, feature, index, state)
                result.append({name: value.clone() for name, value in maps.items()})
            return result
        original = rollout(heatmaps, features)
        changed_heatmaps, changed_features = heatmaps.clone(), features.clone()
        changed_heatmaps[3:] = torch.rand_like(changed_heatmaps[3:])
        changed_features[3:] = torch.rand_like(changed_features[3:])
        changed = rollout(changed_heatmaps, changed_features)
        for before, after in zip(original[:3], changed[:3]):
            for name in CANDIDATE_NAMES:
                self.assertTrue(torch.equal(before[name], after[name]))

    def test_coordinates_preserve_order_first_tie_and_original_grid(self):
        maps = {}
        for index, name in enumerate(reversed(CANDIDATE_NAMES)):
            value = torch.zeros(64, 64)
            value[index + 1, 2 * index + 3] = 1
            value[60, 60] = 1
            maps[name] = value
        actual = maps_to_coordinates(maps)
        expected = np.asarray([[(2 * index + 3) / 64, (index + 1) / 64]
                               for index in reversed(range(len(CANDIDATE_NAMES)))], dtype=np.float32)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_zero_maps_and_zero_features_remain_finite(self):
        raw, features = torch.zeros(64, 64), torch.zeros(3, 2, 2)
        _, state, _ = step_candidates(raw, features, 1)
        maps, _, available = step_candidates(raw, features, 2, state)
        self.assertTrue(available)
        for value in maps.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(torch.count_nonzero(value).item(), 0)
        np.testing.assert_array_equal(maps_to_coordinates(maps), np.zeros((5, 2), dtype=np.float32))

    def test_native_fp16_feature_grid_runs_in_fp32(self):
        features = torch.randn(768, 32, 32, dtype=torch.float16)
        _, state, _ = step_candidates(torch.rand(64, 64), features, 1)
        maps, next_state, available = step_candidates(torch.rand(64, 64), features, 2, state)
        self.assertTrue(available)
        self.assertEqual(next_state["previous_features"].dtype, torch.float32)
        self.assertEqual(maps["transport_previous"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(maps["transport_previous"]).all())
        self.assertAlmostEqual(maps["transport_previous"].sum().item(), 1., places=6)
        self.assertEqual(maps_to_coordinates(maps).shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
