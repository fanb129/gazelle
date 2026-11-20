"""Causal predicted-region memory and visual-history control checks."""

import inspect
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.history_model import (
    HistoryValidityRefiner, normalized_map_weights, target_correspondence,
)
from stag.temporal_model import detach_state, joint_loss


class HistoryValidityRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = HistoryValidityRefiner(feature_channels=4, hidden_dim=8)
        self.features = torch.randn(2, 4, 3, 4)
        self.raw = torch.rand(2, 64, 64) * 0.8 + 0.1
        self.scores = torch.tensor([0.8, 0.2])
        self.boxes = torch.tensor([[0.01, 0.02, 0.03, 0.04], [0.4, 0.2, 0.8, 0.7]])

    def controls(self, validity, refresh):
        """Exercise control paths without claiming learned event recognition."""
        with torch.no_grad():
            self.model.history_gate[-1].weight.zero_()
            self.model.history_gate[-1].bias.copy_(torch.tensor([validity, refresh]))

    def nontrivial_heads(self):
        with torch.no_grad():
            self.model.heatmap_residual.weight.fill_(0.1)
            self.model.inout_residual.weight.fill_(0.2)
            self.model.correspondence_gain.fill_(0.5)

    def sequence(self, features, state=None, raw=None, scores=None):
        raw = self.raw if raw is None else raw
        scores = self.scores if scores is None else scores
        maps, states = [], []
        for current in features:
            heatmap, score, state = self.model.step(current, raw, scores, self.boxes, state)
            maps.append(heatmap)
            states.append(score)
        return torch.stack(maps), torch.stack(states), state

    def test_identity_initialization_and_first_frame_has_no_history(self):
        heatmap, score, state, diag = self.model.step_with_diagnostics(
            self.features.half(), self.raw, self.scores, self.boxes)
        torch.testing.assert_close(heatmap, self.raw, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(score, self.scores, atol=1e-6, rtol=1e-6)
        self.assertEqual(heatmap.shape, (2, 64, 64))
        self.assertEqual(heatmap.dtype, torch.float32)
        self.assertEqual(state["target"].shape, (2, 4))
        self.assertEqual(state["person"].shape, (2, 4))
        self.assertEqual(state["available"].dtype, torch.bool)
        self.assertTrue(state["available"].all())
        self.assertFalse(diag["history_available"].any())
        self.assertTrue((diag["history_validity"] == 0).all())
        self.assertTrue((diag["write_strength"] == 1).all())
        self.assertTrue((diag["correspondence_map"] == 0).all())
        self.assertTrue((diag["correspondence_xy"] == 0).all())
        self.assertFalse(any("backbone" in name for name, _ in self.model.named_parameters()))

    def test_region_readout_uses_nonnegative_unit_sum_weights(self):
        heatmap = torch.tensor([[[-1., 3.], [1., -2.]]])
        weights = normalized_map_weights(heatmap, (2, 2))
        torch.testing.assert_close(weights, torch.tensor([[[0., 0.75], [0.25, 0.]]]))
        torch.testing.assert_close(weights.sum(dim=(-2, -1)), torch.ones(1))
        self.assertTrue((weights >= 0).all())
        self.assertTrue((normalized_map_weights(torch.zeros_like(heatmap), (2, 2)) == 0).all())

    def test_descriptor_is_from_the_model_predicted_region_in_frozen_space(self):
        heatmap, _, state = self.model.step(self.features, self.raw, self.scores, self.boxes)
        weights = normalized_map_weights(heatmap, self.features.shape[-2:])
        expected = (self.features * weights[:, None]).sum(dim=(-2, -1))
        torch.testing.assert_close(state["target"], expected)
        self.assertEqual(state["target"].shape[1], self.features.shape[1])
        self.assertNotEqual(state["target"].shape[1], self.model.hidden_dim)

    def test_correspondence_follows_a_translated_visual_patch(self):
        target = torch.tensor([[1., 0., 0., 0.]])
        first = torch.zeros(1, 4, 3, 4)
        first[:, 1] = 1
        first[0, :, 0, 0] = target[0]
        second = torch.zeros_like(first)
        second[:, 1] = 1
        second[0, :, 2, 3] = target[0]
        one, cosine_one = target_correspondence(first, target)
        two, cosine_two = target_correspondence(second, target)
        self.assertEqual(one.flatten(1).argmax(dim=1).item(), 0)
        self.assertEqual(two.flatten(1).argmax(dim=1).item(), 11)
        torch.testing.assert_close(one.sum(dim=(-2, -1)), torch.ones(1))
        torch.testing.assert_close(two.sum(dim=(-2, -1)), torch.ones(1))
        self.assertEqual(cosine_one[0, 0, 0].item(), 1.)
        self.assertEqual(cosine_two[0, 2, 3].item(), 1.)
        scaled, _ = target_correspondence(second * 7, target * 3)
        torch.testing.assert_close(two, scaled)

    def test_correspondence_can_move_heatmap_support_to_the_current_location(self):
        self.controls(100., -100.)
        with torch.no_grad():
            self.model.correspondence_gain.fill_(2.)
        old_features = torch.zeros_like(self.features)
        old_features[:, 0] = 1
        flat = torch.full_like(self.raw, 0.5)
        _, _, old = self.model.step(old_features, flat, self.scores, self.boxes)
        current = torch.zeros_like(self.features)
        current[:, 1] = 1
        current[:, :, 2, 3] = torch.tensor([1., 0., 0., 0.])
        heatmap, _, _, diag = self.model.step_with_diagnostics(
            current, flat, self.scores, self.boxes, old)
        expected_xy = torch.tensor([[3.5 / 4, 2.5 / 3]]).expand(2, -1)
        torch.testing.assert_close(diag["correspondence_xy"], expected_xy)
        positions = heatmap.flatten(1).argmax(dim=1)
        self.assertTrue(((positions % 64).float() / 64 >= 0.75).all())
        self.assertTrue(((positions // 64).float() / 64 >= 2 / 3).all())
        self.assertTrue((heatmap.amax(dim=(-2, -1)) > 0.5).all())

    def test_controlled_retention_refresh_and_release_are_distinct_paths(self):
        previous = torch.zeros_like(self.features)
        previous[:, 0] = 1
        current = torch.zeros_like(self.features)
        current[:, 1] = 1
        _, _, old = self.model.step(previous, self.raw, self.scores, self.boxes)
        current_target = torch.tensor([[0., 1., 0., 0.]]).expand(2, -1)
        self.controls(100., -100.)
        _, _, kept, keep_diag = self.model.step_with_diagnostics(
            current, self.raw, self.scores, self.boxes, old)
        torch.testing.assert_close(kept["target"], old["target"], atol=1e-6, rtol=0)
        self.assertTrue((keep_diag["history_validity"] == 1).all())
        self.assertTrue((keep_diag["write_strength"] < 1e-6).all())
        self.controls(100., 100.)
        _, _, refreshed, refresh_diag = self.model.step_with_diagnostics(
            current, self.raw, self.scores, self.boxes, old)
        torch.testing.assert_close(refreshed["target"], current_target, atol=1e-6, rtol=0)
        self.assertTrue((refresh_diag["history_validity"] == 1).all())
        self.assertTrue((refresh_diag["write_strength"] == 1).all())
        self.controls(-100., -100.)
        _, _, released, release_diag = self.model.step_with_diagnostics(
            current, self.raw, self.scores, self.boxes, old)
        torch.testing.assert_close(released["target"], current_target, atol=1e-6, rtol=0)
        self.assertTrue((release_diag["history_validity"] == 0).all())
        self.assertTrue((release_diag["write_strength"] == 1).all())

    def test_zero_validity_removes_all_old_prediction_paths(self):
        self.nontrivial_heads()
        self.controls(-100., 0.)
        _, _, one = self.sequence(torch.randn(3, 2, 4, 3, 4))
        _, _, two = self.sequence(torch.randn(3, 2, 4, 3, 4) * 2,
                                  raw=self.raw.flip(-1), scores=1 - self.scores)
        fresh = self.model.step(self.features, self.raw, self.scores, self.boxes)
        with_one = self.model.step(self.features, self.raw, self.scores, self.boxes, one)
        with_two = self.model.step(self.features, self.raw, self.scores, self.boxes, two)
        for output in (0, 1):
            torch.testing.assert_close(with_one[output], fresh[output], rtol=0, atol=0)
            torch.testing.assert_close(with_two[output], fresh[output], rtol=0, atol=0)
        for name in fresh[2]:
            torch.testing.assert_close(with_one[2][name], fresh[2][name], rtol=0, atol=0)
            torch.testing.assert_close(with_two[2][name], fresh[2][name], rtol=0, atol=0)

    def test_always_keep_ablation_preserves_correspondence_despite_low_gate(self):
        self.controls(-100., 0.)
        self.nontrivial_heads()
        always = HistoryValidityRefiner(4, 8, history_mode="always_keep")
        always.load_state_dict(self.model.state_dict())
        _, _, old = self.model.step(self.features, self.raw, self.scores, self.boxes)
        learned = self.model.step_with_diagnostics(
            self.features, self.raw, self.scores, self.boxes, old)
        kept = always.step_with_diagnostics(self.features, self.raw, self.scores, self.boxes, old)
        self.assertTrue((learned[3]["history_validity"] == 0).all())
        self.assertTrue((kept[3]["history_validity"] == 1).all())
        torch.testing.assert_close(learned[3]["correspondence_map"], kept[3]["correspondence_map"])
        self.assertFalse(torch.equal(learned[0], kept[0]))

    def test_history_availability_is_metadata_not_an_inout_threshold(self):
        self.controls(100., -100.)
        scores = torch.zeros_like(self.scores)
        _, _, old, first_diag = self.model.step_with_diagnostics(
            self.features, self.raw, scores, self.boxes)
        _, _, state, second_diag = self.model.step_with_diagnostics(
            self.features, self.raw, scores, self.boxes, old)
        self.assertFalse(first_diag["history_available"].any())
        self.assertTrue(second_diag["history_available"].all())
        self.assertTrue(state["available"].all())
        self.assertTrue((second_diag["history_validity"] == 1).all())

    def test_future_features_do_not_change_a_predicted_prefix(self):
        self.nontrivial_heads()
        features = torch.randn(5, 2, 4, 3, 4)
        changed = features.clone()
        changed[3:] = torch.randn_like(changed[3:]) * 3
        original = self.sequence(features)
        modified = self.sequence(changed)
        torch.testing.assert_close(original[0][:3], modified[0][:3], rtol=0, atol=0)
        torch.testing.assert_close(original[1][:3], modified[1][:3], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[0][3:], modified[0][3:]))

    def test_detached_chunk_state_preserves_continuous_inference(self):
        self.nontrivial_heads()
        features = torch.randn(6, 2, 4, 3, 4)
        continuous = self.sequence(features)
        first = self.sequence(features[:2])
        detached = detach_state(first[2])
        second = self.sequence(features[2:], state=detached)
        torch.testing.assert_close(torch.cat((first[0], second[0])), continuous[0], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat((first[1], second[1])), continuous[1], rtol=0, atol=0)
        for name, value in second[2].items():
            torch.testing.assert_close(value, continuous[2][name], rtol=0, atol=0)
        self.assertTrue(all(not value.requires_grad for value in detached.values()))

    def test_reset_clears_every_memory_field_for_only_the_selected_person(self):
        self.nontrivial_heads()
        _, _, old = self.sequence(torch.randn(3, 2, 4, 3, 4))
        copy = {name: value.clone() for name, value in old.items()}
        result = self.model.step(self.features, self.raw, self.scores, self.boxes, old,
                                 reset=torch.tensor([True, False]))
        fresh = self.model.step(self.features, self.raw, self.scores, self.boxes)
        continued = self.model.step(self.features, self.raw, self.scores, self.boxes, old)
        for output in (0, 1):
            torch.testing.assert_close(result[output][0], fresh[output][0], rtol=0, atol=0)
            torch.testing.assert_close(result[output][1], continued[output][1], rtol=0, atol=0)
        for name in old:
            torch.testing.assert_close(result[2][name][0], fresh[2][name][0], rtol=0, atol=0)
            torch.testing.assert_close(result[2][name][1], continued[2][name][1], rtol=0, atol=0)
            torch.testing.assert_close(old[name], copy[name], rtol=0, atol=0)

    def test_one_persons_memory_cannot_change_another_person(self):
        self.nontrivial_heads()
        _, _, old = self.sequence(torch.randn(3, 2, 4, 3, 4))
        changed = {name: value.clone() for name, value in old.items()}
        changed["hidden"][0] += 2
        changed["target"][0] = changed["target"][0].flip(0)
        changed["person"][0] = changed["person"][0].flip(0)
        changed["heatmap"][0] = 1 - changed["heatmap"][0]
        changed["inout"][0] = 1 - changed["inout"][0]
        original = self.model.step(self.features, self.raw, self.scores, self.boxes, old)
        modified = self.model.step(self.features, self.raw, self.scores, self.boxes, changed)
        for output in (0, 1):
            torch.testing.assert_close(original[output][1], modified[output][1], rtol=0, atol=0)
        for name in old:
            torch.testing.assert_close(original[2][name][1], modified[2][name][1], rtol=0, atol=0)

    def test_forward_contract_has_no_gaze_or_state_labels(self):
        forbidden = {"target_heatmap", "target_inout", "labels", "gaze", "ground_truth"}
        for method in (self.model.step, self.model.step_with_diagnostics):
            self.assertFalse(forbidden & set(inspect.signature(method).parameters))
        _, _, state = self.model.step(self.features, self.raw, self.scores, self.boxes)
        self.assertEqual(set(state), {"hidden", "heatmap", "inout", "target", "person", "available"})

    def test_tiny_head_boxes_and_endpoint_scores_remain_finite(self):
        self.nontrivial_heads()
        scores = torch.tensor([0., 1.])
        heatmap, inout, state, diag = self.model.step_with_diagnostics(
            self.features.half(), self.raw, scores, self.boxes)
        for value in (heatmap, inout, *state.values(), *diag.values()):
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(((heatmap >= 0) & (heatmap <= 1)).all())
        self.assertTrue(((inout >= 0) & (inout <= 1)).all())

    def test_two_updates_train_both_heads_and_history_control(self):
        features = torch.randn(3, 2, 4, 3, 4)
        before = features.clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            maps, scores, _ = self.sequence(features)
            loss = joint_loss(maps, scores, torch.full_like(maps, 0.9),
                              torch.ones_like(scores), torch.ones_like(scores, dtype=torch.bool), 1.)
            loss["total"].backward()
            optimizer.step()
        for parameter in (self.model.heatmap_residual.weight, self.model.inout_residual.weight,
                          self.model.feature_projection.weight, self.model.gru.weight_ih,
                          self.model.history_gate[-1].weight, self.model.correspondence_gain):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad)
        torch.testing.assert_close(features, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
