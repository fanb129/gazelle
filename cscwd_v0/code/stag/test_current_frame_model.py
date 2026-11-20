"""Current-only outputs, initialization identity and structural weight usage."""

import copy
from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code/scripts"))

from stag.current_frame_model import (CONTROL_VERSION, MODEL_ID, CurrentFrameRefiner,
                                      parameter_usage)
from stag.temporal_model import CausalJointRefiner, joint_loss
from train_compact_joint import shared_initialization_sha256


class CurrentFrameModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.model = CurrentFrameRefiner(4, 8).eval()
        # Nonzero heads make history-independence checks exercise the refiner,
        # rather than just its zero-initialized identity output.
        with torch.no_grad():
            self.model.heatmap_residual.weight.normal_(std=.2)
            self.model.inout_residual.weight.normal_(std=.2)
        self.inputs = (
            torch.randn(2, 4, 3, 3),
            torch.rand(2, 64, 64) * .8 + .1,
            torch.tensor([.3, .8]),
            torch.tensor([[.1, .1, .4, .5], [.5, .3, .9, .9]]),
        )

    def test_shared_initialization_and_all_parameter_shapes_match_ordinary_gru(self):
        torch.manual_seed(0)
        control = CurrentFrameRefiner(4, 8)
        torch.manual_seed(0)
        recurrent = CausalJointRefiner(4, 8)
        self.assertEqual(MODEL_ID, "current_frame")
        self.assertEqual(CONTROL_VERSION, "r4_current_frame_v1")
        self.assertEqual(control.state_dict().keys(), recurrent.state_dict().keys())
        self.assertEqual(shared_initialization_sha256(control), shared_initialization_sha256(recurrent))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, recurrent.state_dict()[name]))

    def test_supplied_state_and_reset_are_ignored_and_no_state_is_returned(self):
        poison = {"hidden": torch.randn(2, 8) * 100,
                  "heatmap": torch.rand(2, 64, 64), "inout": torch.tensor([1., 0.])}
        with torch.no_grad():
            baseline = self.model.step(*self.inputs)
            contaminated = self.model.step(*self.inputs, poison, torch.tensor([False, False]))
            reset = self.model.step(*self.inputs, poison, torch.tensor([True, True]))
        for output in (baseline, contaminated, reset):
            self.assertIsNone(output[2])
        for position in (0, 1):
            torch.testing.assert_close(baseline[position], contaminated[position], rtol=0, atol=0)
            torch.testing.assert_close(baseline[position], reset[position], rtol=0, atol=0)

    def test_current_outputs_equal_an_ordinary_gru_fresh_step(self):
        recurrent = CausalJointRefiner(4, 8).eval()
        recurrent.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            expected = recurrent.step(*self.inputs, state=None, reset=None)
            actual = self.model.step(*self.inputs, state=expected[2])
            recurrent_history = recurrent.step(*self.inputs, state=expected[2])
        for position in (0, 1):
            torch.testing.assert_close(actual[position], expected[position], rtol=0, atol=0)
        self.assertIsNone(actual[2])
        self.assertFalse(torch.equal(actual[1], recurrent_history[1]))

    def test_frame_order_prior_inputs_and_chunk_boundaries_do_not_change_outputs(self):
        generator = torch.Generator().manual_seed(45)
        sequence = [
            (torch.randn(2, 4, 3, 3, generator=generator),
             torch.rand(2, 64, 64, generator=generator) * .8 + .1,
             torch.rand(2, generator=generator) * .8 + .1,
             self.inputs[3])
            for _ in range(7)
        ]
        def run(order, chunk_size):
            outputs, state = {}, None
            with torch.no_grad():
                for start in range(0, len(order), chunk_size):
                    for index in order[start:start + chunk_size]:
                        heatmap, score, state = self.model.step(*sequence[index], state=state)
                        outputs[index] = heatmap, score
                    self.assertIsNone(state)
            return outputs
        expected = run(list(range(7)), 7)
        for order, chunk_size in ((list(range(7)), 1), (list(range(7)), 3), (list(reversed(range(7))), 2)):
            with self.subTest(order=order, chunk_size=chunk_size):
                actual = run(order, chunk_size)
                for index in order:
                    for position in (0, 1):
                        torch.testing.assert_close(actual[index][position], expected[index][position], rtol=0, atol=0)
        # A different prefix must not change the final frame's output.
        with torch.no_grad():
            self.model.step(*(value * .5 for value in sequence[0]))
            final = self.model.step(*sequence[-1])
        for position in (0, 1):
            torch.testing.assert_close(final[position], expected[6][position], rtol=0, atol=0)

    def test_structural_zero_weights_have_zero_task_gradient_but_bias_and_heads_work(self):
        model = copy.deepcopy(self.model).train()
        heatmap, score, state = model.step(*self.inputs)
        self.assertIsNone(state)
        target = torch.zeros_like(heatmap)
        target[:, 12, 19] = 1
        loss = joint_loss(heatmap, score, target, torch.tensor([1., 0.]),
                          torch.ones(2, dtype=torch.bool), .1)
        loss["total"].backward()
        usage = parameter_usage(model)
        groups = usage["structural_zero_weight_groups"]
        columns = groups["gru.weight_ih.previous_heatmap_columns"]
        previous_map = slice(columns["input_columns_start"], columns["input_columns_stop_exclusive"])
        previous_score = groups["gru.weight_ih.previous_inout_column"]["input_column"]
        self.assertEqual(torch.count_nonzero(model.gru.weight_hh.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(model.gru.weight_ih.grad[:, previous_map]).item(), 0)
        self.assertEqual(torch.count_nonzero(model.gru.weight_ih.grad[:, previous_score]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.gru.weight_ih.grad[:, :columns["input_columns_start"]]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.gru.bias_hh.grad[2 * model.hidden_dim:]).item(), 0)
        for head in (model.heatmap_residual, model.inout_residual):
            self.assertGreater(torch.count_nonzero(head.weight.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(model.feature_projection.weight.grad).item(), 0)
        self.assertFalse(usage["effective_parameter_use_matches_recurrent_model"])
        self.assertEqual(usage["structural_zero_weight_elements"], 3 * 8 * 8 + 3 * 8 * 65)

    def test_zero_task_gradient_does_not_mean_no_adamw_decay_or_output_dependency(self):
        model = copy.deepcopy(self.model).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        before = model.gru.weight_hh.detach().clone()
        heatmap, score, _ = model.step(*self.inputs)
        (heatmap.mean() + score.mean()).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.gru.weight_hh))
        model.eval()
        usage = parameter_usage(model)
        group = usage["structural_zero_weight_groups"]["gru.weight_ih.previous_heatmap_columns"]
        previous_map = slice(group["input_columns_start"], group["input_columns_stop_exclusive"])
        previous_score = usage["structural_zero_weight_groups"]["gru.weight_ih.previous_inout_column"]["input_column"]
        with torch.no_grad():
            reference = model.step(*self.inputs)
            model.gru.weight_hh.mul_(1000)
            model.gru.weight_ih[:, previous_map].mul_(1000)
            model.gru.weight_ih[:, previous_score].mul_(1000)
            changed = model.step(*self.inputs)
        for position in (0, 1):
            torch.testing.assert_close(reference[position], changed[position], rtol=0, atol=0)

    def test_native_frozen_feature_shape_and_retained_parameter_count(self):
        torch.manual_seed(0)
        model = CurrentFrameRefiner(768, 128).eval()
        with torch.no_grad():
            heatmap, score, state = model.step(torch.randn(1, 768, 32, 32),
                torch.rand(1, 64, 64) * .8 + .1, torch.tensor([.7]),
                torch.tensor([[.1, .1, .4, .5]]))
        self.assertEqual(heatmap.shape, (1, 64, 64))
        self.assertEqual(score.shape, (1,))
        self.assertTrue(torch.isfinite(heatmap).all() and torch.isfinite(score).all())
        self.assertIsNone(state)
        usage = parameter_usage(model)
        self.assertEqual(usage["full_parameter_count"], 347522)
        self.assertEqual(usage["structural_zero_weight_elements"], 74112)


if __name__ == "__main__":
    unittest.main()
