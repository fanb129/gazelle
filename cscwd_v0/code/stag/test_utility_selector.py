"""D6 observations, raw-only memory, hard outputs and cost-sensitive supervision."""

import copy
import inspect
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.ema import smooth_ema_track
from stag.history_candidates import step_candidates
from stag.utility_selector import (ANCHOR_INDEX, BANK_CANDIDATES, HISTORY_INDEX,
    MODEL_ID, SCALAR_NAMES, VERSION, UtilitySelector, cost_weighted_bce,
    hard_choices, step_selector_inputs)


class UtilitySelectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(89)
        self.box = torch.tensor([0., 0., 1., 1.])

    def inputs(self, channels=3, grid=2):
        return (torch.rand(64, 64), torch.randn(channels, grid, grid),
                torch.tensor(.7), self.box.clone())

    def test_fixed_contract_and_label_free_step_signature(self):
        self.assertEqual(VERSION, "r4_utility_selector_v1")
        self.assertEqual(MODEL_ID, "utility_selector")
        self.assertEqual(BANK_CANDIDATES, ("current", "current_grid", "ema_075",
                                         "grid_ema_075", "transport_previous"))
        self.assertEqual((ANCHOR_INDEX, HISTORY_INDEX), (1, 4))
        self.assertEqual(len(SCALAR_NAMES), 17)
        self.assertEqual(tuple(inspect.signature(step_selector_inputs).parameters),
                         ("raw_heatmap", "features", "raw_inout", "bbox", "frame_index", "state"))
        maps, observation, available, state = step_selector_inputs(*self.inputs(), 1)
        self.assertEqual(tuple(maps), BANK_CANDIDATES)
        self.assertEqual(observation.shape, (6 * 3 + 17,))
        self.assertEqual(observation.dtype, torch.float32)
        self.assertFalse(available)
        self.assertEqual(set(state), {"candidates", "grid_ema", "previous_person", "previous_scene",
                                     "previous_region", "previous_inout", "previous_bbox"})
        self.assertFalse(observation.requires_grad)

    def test_initial_model_is_anchor_and_choices_preserve_whole_maps(self):
        model = UtilitySelector(3)
        logits = model(torch.randn(4, 35))
        self.assertTrue(torch.equal(logits, torch.zeros(4)))
        self.assertTrue(torch.equal(hard_choices(logits, torch.ones(4, dtype=torch.bool)),
                                    torch.full((4,), 1, dtype=torch.long)))
        choices = hard_choices(torch.tensor([-1., 0., 1., 1.]), torch.tensor([1, 1, 1, 0]))
        self.assertEqual(choices.dtype, torch.long)
        self.assertEqual(choices.tolist(), [1, 1, 4, 1])
        # Candidate amplitudes differ deliberately: selection is a full map,
        # never a soft interpolation between response and probability scales.
        bank = torch.stack([torch.full((4, 64, 64), float(i + 1)) for i in range(5)], dim=1)
        selected = bank[torch.arange(4), choices]
        self.assertEqual(selected[:, 0, 0].tolist(), [2., 2., 5., 2.])
        for row, choice in enumerate(choices):
            self.assertTrue(torch.equal(selected[row], bank[row, choice]))

    def test_d5_candidates_unchanged_and_both_emas_match_fixed_controls(self):
        raws = torch.rand(5, 64, 64)
        frames = np.array([1, 2, 3, 8, 9])
        features = torch.eye(4).reshape(4, 2, 2)
        state, d5_state, grid_maps, raw_ema, grid_ema = None, None, [], [], []
        for raw, frame in zip(raws, frames):
            maps, _, _, state = step_selector_inputs(raw, features, torch.tensor(.6), self.box, int(frame), state)
            expected, d5_state, _ = step_candidates(raw, features, int(frame), d5_state)
            for name in ("current", "current_grid", "ema_075", "transport_previous"):
                self.assertTrue(torch.equal(maps[name], expected[name]))
            grid_maps.append(maps["current_grid"].numpy())
            raw_ema.append(maps["ema_075"].numpy())
            grid_ema.append(maps["grid_ema_075"].numpy())
        np.testing.assert_array_equal(np.stack(raw_ema), smooth_ema_track(raws.numpy(), frames, .75)["heatmap"])
        np.testing.assert_array_equal(np.stack(grid_ema), smooth_ema_track(np.stack(grid_maps), frames, .75)["heatmap"])

    def test_tokens_and_all_scalar_positions_have_hand_computed_values(self):
        previous_features = torch.zeros(2, 2, 2)
        previous_features[0] = 1
        current_features = torch.zeros_like(previous_features)
        current_features[1] = 1
        raw = torch.ones(64, 64)
        previous_box = torch.tensor([0., 0., .5, .5])
        current_box = torch.tensor([.5, .5, 1., 1.])
        _, _, _, state = step_selector_inputs(raw, previous_features, torch.tensor(.2), previous_box, 1)
        _, observation, available, _ = step_selector_inputs(raw, current_features, torch.tensor(.8), current_box, 2, state)
        self.assertTrue(available)
        expected_tokens = torch.tensor([[0., 1.], [0., 1.], [1., 0.],
                                        [0., 1.], [0., 1.], [1., 0.]])
        torch.testing.assert_close(observation[:12].reshape(6, 2), expected_tokens)
        expected_scalars = torch.tensor([.8, .2, .6, math.sqrt(.5), 0., 1., 0.,
            1., 1., 1 / 4096, 1 / 4096, 0., 0., 0., 0., 0., 1.])
        torch.testing.assert_close(observation[12:], expected_scalars, atol=1e-7, rtol=1e-6)

    def test_region_tokens_use_fractional_head_pool_and_normalized_map_weights(self):
        features = torch.tensor([[[1., 0.]], [[0., 1.]]])
        raw = torch.zeros(64, 64)
        raw[:, :32] = 1
        _, observation, _, _ = step_selector_inputs(raw, features, torch.tensor(.7),
                                                   torch.tensor([0., 0., .5, 1.]), 1)
        tokens = observation[:12].reshape(6, 2)
        torch.testing.assert_close(tokens[0], torch.tensor([1., 1.]) / math.sqrt(2))
        torch.testing.assert_close(tokens[1], torch.tensor([1., 0.]))
        torch.testing.assert_close(tokens[3], F.normalize(torch.tensor([.875, .125]), dim=0))
        torch.testing.assert_close(tokens[4], torch.tensor([1., 0.]))

    def test_no_history_has_zero_past_tokens_and_scalars(self):
        _, observation, available, _ = step_selector_inputs(*self.inputs(), 1)
        self.assertFalse(available)
        tokens = observation[:18].reshape(6, 3)
        self.assertTrue(torch.equal(tokens[2], torch.zeros(3)))
        self.assertTrue(torch.equal(tokens[5], torch.zeros(3)))
        scalars = dict(zip(SCALAR_NAMES, observation[18:]))
        for name in ("previous_raw_inout", "inout_absolute_change", "bbox_center_displacement",
                     "person_cosine", "history_region_previous_region_cosine",
                     "anchor_previous_raw_displacement", "history_previous_raw_displacement",
                     "scene_cosine", "history_available"):
            self.assertEqual(scalars[name].item(), 0.)

    def test_person_and_gap_resets_match_fresh_observations_and_controls(self):
        _, _, _, state = step_selector_inputs(*self.inputs(), 1)
        current = self.inputs()
        fresh_maps, fresh_obs, _, _ = step_selector_inputs(*current, 4)
        gap_maps, gap_obs, available, _ = step_selector_inputs(*current, 4, state)
        self.assertFalse(available)
        self.assertTrue(torch.equal(gap_obs, fresh_obs))
        for name in BANK_CANDIDATES:
            self.assertTrue(torch.equal(fresh_maps[name], gap_maps[name]))
        # A new person uses None even when original frame numbers are adjacent.
        _, new_obs, new_available, _ = step_selector_inputs(*current, 2, None)
        self.assertFalse(new_available)
        self.assertTrue(torch.equal(new_obs, fresh_obs))

    def test_future_inputs_do_not_change_past_maps_or_observations(self):
        frames = [self.inputs() for _ in range(5)]
        def rollout(inputs):
            state, rows = None, []
            for index, current in enumerate(inputs):
                maps, observation, _, state = step_selector_inputs(*current, index, state)
                rows.append(({name: value.clone() for name, value in maps.items()}, observation.clone()))
            return rows
        original = rollout(frames)
        changed = rollout(frames[:3] + [self.inputs(), self.inputs()])
        for (maps, observation), (after_maps, after_observation) in zip(original[:3], changed[:3]):
            self.assertTrue(torch.equal(observation, after_observation))
            for name in BANK_CANDIDATES:
                self.assertTrue(torch.equal(maps[name], after_maps[name]))

    def test_output_choices_maps_and_input_aliases_cannot_update_memory(self):
        inputs = self.inputs()
        maps, observation, _, state = step_selector_inputs(*inputs, 1)
        expected_state = copy.deepcopy(state)
        for value in maps.values():
            value.fill_(99)
        observation.fill_(99)
        for value in inputs:
            value.zero_()
        # Choices are consumed externally; neither they nor mutated returned
        # maps can write through the raw observation memory.
        current = self.inputs()
        actual_maps, actual_observation, _, _ = step_selector_inputs(*current, 2, state)
        expected_maps, expected_observation, _, _ = step_selector_inputs(*current, 2, expected_state)
        self.assertTrue(torch.equal(actual_observation, expected_observation))
        for name in BANK_CANDIDATES:
            self.assertTrue(torch.equal(actual_maps[name], expected_maps[name]))

    def test_native_shape_fp32_observations_and_parameter_count(self):
        inputs = (torch.rand(64, 64), torch.randn(768, 32, 32, dtype=torch.float16),
                  torch.tensor(.4), self.box)
        _, _, _, state = step_selector_inputs(*inputs, 1)
        maps, observation, available, _ = step_selector_inputs(*inputs, 2, state)
        self.assertTrue(available)
        self.assertEqual(observation.shape, (4625,))
        self.assertEqual(observation.dtype, torch.float32)
        self.assertTrue(torch.isfinite(observation).all())
        model = UtilitySelector(768)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 296129)
        self.assertEqual(model(observation[None]).shape, (1,))
        for value in maps.values():
            self.assertEqual(value.dtype, torch.float32)

    def test_hand_computed_weighted_loss_masks_ties_and_gradient(self):
        logits = torch.tensor([.7, -.3, 1., 20.], requires_grad=True)
        anchor = torch.zeros(4, 2)
        history = torch.tensor([[.5, 0.], [.5, 0.], [0., 0.], [1., 0.]])
        gt = torch.tensor([[.5, 0.], [0., 0.], [0., 0.], [1., 0.]], dtype=torch.float64)
        result = cost_weighted_bce(logits, anchor, history, gt,
                                   torch.tensor([1, 1, 1, 0]), torch.ones(4, dtype=torch.bool))
        expected = (.5 * F.softplus(-logits[0]) + .5 * F.softplus(logits[1])).double()
        torch.testing.assert_close(result["weighted_bce_sum"], expected)
        self.assertEqual(result["weight_sum"].item(), 1.)
        self.assertEqual(result["supervised_frames"].item(), 3)
        self.assertEqual(result["prefer_history_frames"].item(), 1)
        self.assertEqual(result["tie_frames"].item(), 1)
        self.assertEqual(result["loss"].dtype, torch.float64)
        result["loss"].backward()
        expected_gradient = torch.tensor([(torch.sigmoid(logits[0]).item() - 1) / 2,
                                           torch.sigmoid(logits[1]).item() / 2, 0., 0.])
        torch.testing.assert_close(logits.grad, expected_gradient)

    def test_costs_are_not_rebalanced_between_classes(self):
        logits = torch.zeros(2, requires_grad=True)
        anchor = torch.zeros(2, 2)
        history = torch.tensor([[.75, 0.], [.25, 0.]])
        gt = torch.tensor([[.75, 0.], [0., 0.]], dtype=torch.float64)
        result = cost_weighted_bce(logits, anchor, history, gt,
                                   torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        result["loss"].backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.375, .125]))

    def test_error_advantage_keeps_float64_ground_truth_precision(self):
        logits = torch.zeros(1, requires_grad=True)
        result = cost_weighted_bce(logits, torch.tensor([[0., 0.]]), torch.tensor([[1., 0.]]),
            torch.tensor([[.5000000001, 0.]], dtype=torch.float64),
            torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
        self.assertEqual(result["prefer_history_frames"].item(), 1)
        self.assertEqual(result["tie_frames"].item(), 0)
        self.assertGreater(result["weight_sum"].item(), 0.)
        self.assertLess(result["weight_sum"].item(), 1e-9)
        result["loss"].backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.5]))

    def test_all_tie_or_unsupervised_batch_is_differentiable_zero(self):
        for available in (torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
            logits = torch.randn(3, requires_grad=True)
            xy = torch.rand(3, 2)
            result = cost_weighted_bce(logits, xy, xy, torch.rand(3, 2).double(),
                                       torch.ones(3, dtype=torch.bool), available)
            self.assertEqual(result["loss"].item(), 0.)
            result["loss"].backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros(3)))

    def test_cost_supervision_can_update_both_mlp_layers(self):
        model = UtilitySelector(3)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        observations = torch.randn(16, 35)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        anchor, history = torch.zeros(16, 2), torch.ones(16, 2) * .5
        gt = history.double()
        for _ in range(2):
            optimizer.zero_grad()
            result = cost_weighted_bce(model(observations), anchor, history, gt,
                torch.ones(16, dtype=torch.bool), torch.ones(16, dtype=torch.bool))
            result["loss"].backward()
            optimizer.step()
        for name, value in model.named_parameters():
            self.assertFalse(torch.equal(value, initial[name]), name)


if __name__ == "__main__":
    unittest.main()
