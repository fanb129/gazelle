"""Causal history, fractional pooling and masked joint-learning checks."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.temporal_model import CausalJointRefiner, bbox_pool, detach_state, joint_loss


class CausalJointRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = CausalJointRefiner(feature_channels=4, hidden_dim=8)
        self.features = torch.randn(2, 4, 3, 4)
        self.raw = torch.rand(2, 64, 64) * 0.8 + 0.1
        self.scores = torch.tensor([0.8, 0.2])
        self.boxes = torch.tensor([[0.01, 0.02, 0.03, 0.04], [0.4, 0.2, 0.8, 0.7]])

    def unzero_heads(self):
        # A nontrivial head exposes temporal dependence hidden by identity init.
        with torch.no_grad():
            self.model.heatmap_residual.weight.fill_(0.1)
            self.model.inout_residual.weight.fill_(0.2)

    def run_sequence(self, features, raw=None, scores=None, boxes=None, state=None):
        raw = self.raw if raw is None else raw
        scores = self.scores if scores is None else scores
        boxes = self.boxes if boxes is None else boxes
        maps, states = [], []
        for current in features:
            heatmap, score, state = self.model.step(current, raw, scores, boxes, state)
            maps.append(heatmap)
            states.append(score)
        return torch.stack(maps), torch.stack(states), state

    def test_identity_initialization_shapes_range_and_float32(self):
        heatmap, score, state = self.model.step(self.features.half(), self.raw, self.scores, self.boxes)
        self.assertEqual(heatmap.shape, (2, 64, 64))
        self.assertEqual(score.shape, (2,))
        self.assertEqual(state["hidden"].shape, (2, 8))
        self.assertEqual(heatmap.dtype, torch.float32)
        torch.testing.assert_close(heatmap, self.raw, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(score, self.scores, atol=1e-6, rtol=1e-6)
        self.assertTrue(((heatmap >= 0) & (heatmap <= 1)).all())
        self.assertGreater(sum(p.numel() for p in self.model.parameters()), 0)
        self.assertFalse(any("backbone" in name for name, _ in self.model.named_parameters()))

    def test_prefix_is_unchanged_when_future_observations_change(self):
        self.unzero_heads()
        sequence = torch.randn(5, 2, 4, 3, 4)
        changed = sequence.clone()
        changed[3:] = torch.randn_like(changed[3:]) * 100
        first = self.run_sequence(sequence)
        second = self.run_sequence(changed)
        torch.testing.assert_close(first[0][:3], second[0][:3], rtol=0, atol=0)
        torch.testing.assert_close(first[1][:3], second[1][:3], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0][3:], second[0][3:]))

    def test_chunk_state_matches_continuous_per_frame_inference(self):
        self.unzero_heads()
        sequence = torch.randn(6, 2, 4, 3, 4)
        continuous = self.run_sequence(sequence)
        first = self.run_sequence(sequence[:2])
        second = self.run_sequence(sequence[2:], state=detach_state(first[2]))
        torch.testing.assert_close(torch.cat((first[0], second[0])), continuous[0], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat((first[1], second[1])), continuous[1], rtol=0, atol=0)

    def test_reset_clears_only_the_selected_person(self):
        self.unzero_heads()
        _, _, old = self.run_sequence(torch.randn(3, 2, 4, 3, 4))
        retained = {key: value.clone() for key, value in old.items()}
        result = self.model.step(self.features, self.raw, self.scores, self.boxes, old,
                                 reset=torch.tensor([True, False]))
        fresh = self.model.step(self.features, self.raw, self.scores, self.boxes)
        continued = self.model.step(self.features, self.raw, self.scores, self.boxes, old)
        for output in (0, 1):
            torch.testing.assert_close(result[output][0], fresh[output][0], rtol=0, atol=0)
            torch.testing.assert_close(result[output][1], continued[output][1], rtol=0, atol=0)
        for name in old:
            torch.testing.assert_close(old[name], retained[name], rtol=0, atol=0)

    def test_batch_people_do_not_share_hidden_state(self):
        self.unzero_heads()
        _, _, state = self.run_sequence(torch.randn(3, 2, 4, 3, 4))
        changed = {key: value.clone() for key, value in state.items()}
        for value in changed.values():
            value[0].add_(100)
        original = self.model.step(self.features, self.raw, self.scores, self.boxes, state)
        modified = self.model.step(self.features, self.raw, self.scores, self.boxes, changed)
        torch.testing.assert_close(original[0][1], modified[0][1], rtol=0, atol=0)
        torch.testing.assert_close(original[1][1], modified[1][1], rtol=0, atol=0)

    def test_fractional_pool_keeps_tiny_head_and_weights_patch_boundaries(self):
        features = torch.tensor([[[[2., 10.], [20., 30.]]]])
        tiny = torch.tensor([[0.01, 0.01, 0.02, 0.02]])
        torch.testing.assert_close(bbox_pool(features, tiny), torch.tensor([[2.]]))
        crossing = torch.tensor([[0.4, 0., 0.6, 0.5]])
        torch.testing.assert_close(bbox_pool(features, crossing), torch.tensor([[6.]]))
        full = torch.tensor([[0., 0., 1., 1.]])
        torch.testing.assert_close(bbox_pool(features, full), features.mean(dim=(-2, -1)))

    def test_history_contains_model_outputs_and_detach_does_not_change_values(self):
        self.unzero_heads()
        heatmap, score, state = self.model.step(self.features, self.raw, self.scores, self.boxes)
        self.assertIs(state["heatmap"], heatmap)
        self.assertIs(state["inout"], score)
        detached = detach_state(state)
        for key, value in state.items():
            self.assertTrue(value.requires_grad)
            self.assertFalse(detached[key].requires_grad)
            torch.testing.assert_close(detached[key], value, rtol=0, atol=0)
        self.assertIsNone(detach_state(None))

    def test_joint_loss_masks_outside_and_padding_targets(self):
        maps = torch.full((1, 3, 2, 2), 0.5, requires_grad=True)
        scores = torch.tensor([[0.8, 0.25, 0.7]], requires_grad=True)
        target = torch.tensor([[[[1., 1.], [1., 1.]],
                                [[float('nan')]*2]*2, [[float('nan')]*2]*2]])
        labels = torch.tensor([[1., 0., -1.]])
        valid = torch.tensor([[True, True, False]])
        losses = joint_loss(maps, scores, target, labels, valid, lambda_state=0.1)
        torch.testing.assert_close(losses["heatmap"], torch.tensor(0.25))
        expected_state = -(torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.75))) / 2
        torch.testing.assert_close(losses["state"], expected_state)
        losses["total"].backward()
        self.assertTrue((maps.grad[0, 0] != 0).all())
        self.assertTrue((maps.grad[0, 1:] == 0).all())
        self.assertNotEqual(scores.grad[0, 1].item(), 0)
        self.assertEqual(scores.grad[0, 2].item(), 0)

    def test_no_inframe_or_valid_frames_produces_differentiable_zero(self):
        maps = torch.rand(2, 2, 2, requires_grad=True)
        scores = torch.tensor([0.3, 0.6], requires_grad=True)
        labels = torch.zeros(2)
        target = torch.full_like(maps, float('nan'))
        losses = joint_loss(maps, scores, target, labels, torch.ones(2, dtype=torch.bool), 1.)
        self.assertEqual(losses["heatmap"].item(), 0)
        self.assertGreater(losses["state"].item(), 0)
        empty = joint_loss(maps, scores, target, labels, torch.zeros(2, dtype=torch.bool), 1.)
        self.assertEqual(empty["total"].item(), 0)
        empty["total"].backward()
        self.assertIsNotNone(maps.grad)
        self.assertIsNotNone(scores.grad)

    def test_two_heads_learn_without_changing_frozen_input_features(self):
        features = self.features.clone()
        before = features.clone()
        original_maps, original_scores, _ = self.model.step(features, self.raw, self.scores, self.boxes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            heatmap, scores, _ = self.model.step(features, self.raw, self.scores, self.boxes)
            losses = joint_loss(heatmap, scores, torch.full_like(heatmap, 0.9),
                                torch.ones(2), torch.ones(2, dtype=torch.bool), 1.)
            losses["total"].backward()
            optimizer.step()
        heatmap, scores, _ = self.model.step(features, self.raw, self.scores, self.boxes)
        self.assertFalse(torch.equal(heatmap, original_maps))
        self.assertFalse(torch.equal(scores, original_scores))
        self.assertIsNotNone(self.model.feature_projection.weight.grad)
        self.assertGreater(self.model.feature_projection.weight.grad.abs().sum().item(), 0)
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad)
        torch.testing.assert_close(features, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
