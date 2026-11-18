"""Hand-counted temporal state losses, causal carries and actual epoch updates."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code/scripts"))

from stag.history_model import HistoryValidityRefiner
from stag.official_data import _load_split
from stag.state_supervision import (_forward_chunk, state_temporal_loss,
                                    train_supervised_epoch)
from stag.temporal_data import iter_track_batches
from stag.temporal_model import CausalJointRefiner
from test_official_data import make_official_fixture


def tensors(scores, labels, valid=None, reset=None):
    scores = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor(labels, dtype=torch.float32)
    valid = torch.ones_like(labels, dtype=torch.bool) if valid is None else torch.tensor(valid, dtype=torch.bool)
    reset = torch.zeros_like(valid) if reset is None else torch.tensor(reset, dtype=torch.bool)
    return scores, labels, valid, reset


class StateTemporalLossTests(unittest.TestCase):
    def test_hand_computed_stable_L1_and_independently_normalized_transition_BCE(self):
        args = tensors([[.8, .7, .2, .1, .3, .9, .8]], [[1, 1, 0, 0, 0, 1, 1]])
        loss, _ = state_temporal_loss(*args)
        self.assertEqual(loss["stable_pairs"].item(), 4)
        self.assertEqual(loss["transition_frames"].item(), 5)
        torch.testing.assert_close(loss["stable_sum"], torch.tensor(.5))
        torch.testing.assert_close(loss["stable"], torch.tensor(.125))
        expected = -torch.log(torch.tensor([.8, .9, .7, .9, .8])).sum()
        torch.testing.assert_close(loss["transition_sum"], expected)
        torch.testing.assert_close(loss["transition"], expected / 5)

    def test_transition_targets_do_not_reach_before_the_observed_flip(self):
        args = tensors([[.4, .5, .6, .7, .8]], [[0, 0, 0, 1, 1]])
        loss, _ = state_temporal_loss(*args)
        self.assertEqual(loss["transition_frames"].item(), 2)
        loss["transition"].backward()
        torch.testing.assert_close(args[0].grad[:, :3], torch.zeros(1, 3), rtol=0, atol=0)
        self.assertTrue((args[0].grad[:, 3:] < 0).all())

    def test_transition_window_is_three_frames_and_restarts_on_a_new_flip(self):
        args = tensors([[.2] * 8], [[0, 0, 1, 1, 0, 0, 0, 0]])
        loss, carry = state_temporal_loss(*args)
        # Flip at 2 selects 2,3; flip at 4 restarts and selects 4,5,6.
        self.assertEqual(loss["transition_frames"].item(), 5)
        self.assertEqual(carry["remaining"].item(), 0)
        self.assertEqual(loss["stable_pairs"].item(), 5)

    def test_reset_prevents_cross_person_or_gap_pairs_and_ends_pending_response(self):
        args = tensors([[.1, .9, .2, .8]], [[0, 1, 0, 1]], reset=[[True, False, True, False]])
        loss, _ = state_temporal_loss(*args)
        self.assertEqual(loss["stable_pairs"].item(), 0)
        # Frame 2 is a new segment, so only frames 1 and 3 are flip responses.
        self.assertEqual(loss["transition_frames"].item(), 2)
        expected = -torch.log(torch.tensor([.9, .8])).mean()
        torch.testing.assert_close(loss["transition"], expected)

    def test_constant_predictions_on_constant_labels_have_zero_temporal_losses(self):
        args = tensors([[.7, .7, .7]], [[1, 1, 1]])
        loss, _ = state_temporal_loss(*args)
        self.assertEqual(loss["stable_pairs"].item(), 2)
        self.assertEqual(loss["transition_frames"].item(), 0)
        self.assertEqual(loss["stable"].item(), 0)
        self.assertEqual(loss["transition"].item(), 0)
        (loss["stable"] + loss["transition"]).backward()
        self.assertIsNotNone(args[0].grad)

    def test_empty_masks_are_differentiable_and_padding_never_bridges_frames(self):
        args = tensors([[.1, .9, .7]], [[0, 1, 1]], valid=[[True, False, True]])
        loss, carry = state_temporal_loss(*args)
        self.assertEqual(loss["stable_pairs"].item(), 0)
        self.assertEqual(loss["transition_frames"].item(), 0)
        (loss["stable"] + loss["transition"]).backward()
        torch.testing.assert_close(args[0].grad, torch.zeros_like(args[0]), rtol=0, atol=0)
        empty_args = tensors([[.3, .2]], [[0, 0]], valid=[[False, False]])
        empty, carry = state_temporal_loss(*empty_args, carry=carry)
        self.assertEqual(empty["stable"].item(), 0)
        self.assertEqual(empty["transition"].item(), 0)
        self.assertFalse(carry["valid"].any())
        self.assertEqual(carry["remaining"].sum().item(), 0)

    def test_chunk_boundaries_preserve_all_pairs_response_windows_and_numerators(self):
        scores, labels, valid, reset = tensors(
            [[.7, .6, .2, .3, .4, .8, .7], [.2, .4, .8, .9, .3, .4, .6]],
            [[1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0, 0]],
            valid=[[True] * 7, [True] * 5 + [False] * 2])
        full, _ = state_temporal_loss(scores, labels, valid, reset)
        for chunk_size in (1, 2, 3, 5):
            with self.subTest(chunk_size=chunk_size):
                carry = None
                sums = {key: 0. for key in ("stable_sum", "transition_sum", "stable_pairs", "transition_frames")}
                for first in range(0, scores.shape[1], chunk_size):
                    selection = (slice(None), slice(first, first + chunk_size))
                    loss, carry = state_temporal_loss(scores[selection], labels[selection],
                        valid[selection], reset[selection], carry)
                    for key in sums:
                        sums[key] += loss[key].item()
                for key, value in sums.items():
                    self.assertAlmostEqual(value, full[key].item(), places=6)

    def test_cross_chunk_previous_prediction_is_detached_but_current_has_gradient(self):
        first_args = tensors([[.3]], [[1]])
        _, carry = state_temporal_loss(*first_args)
        self.assertFalse(carry["score"].requires_grad)
        next_args = tensors([[.7]], [[1]])
        second, _ = state_temporal_loss(*next_args, carry=carry)
        second["stable"].backward()
        self.assertIsNone(first_args[0].grad)
        self.assertEqual(next_args[0].grad.item(), 1)

    def test_inside_chunk_stable_pair_backpropagates_both_adjacent_predictions(self):
        args = tensors([[.3, .7]], [[1, 1]])
        loss, _ = state_temporal_loss(*args)
        loss["stable"].backward()
        torch.testing.assert_close(args[0].grad, torch.tensor([[-1., 1.]]), rtol=0, atol=0)


class StateSupervisedEpochTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fixture = make_official_fixture(self.root)
        self.part, _, _ = _load_split("train", fixture["train_feature_cache"],
            fixture["train_feature_report"], fixture["train_prediction_report"])

    def train(self, model, clip_frames=3, learning_rate=0.01):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        return train_supervised_epoch(model, optimizer, self.part["tracks"], self.part["arrays"],
            self.part["feature_index"], self.part["features"], "cpu", clip_frames, 2, 0, .1, .1, 3)

    def test_actual_epochs_update_both_heads_and_preserve_read_only_inputs(self):
        before_features = self.part["features"].copy()
        before_arrays = {name: values.copy() for name, values in self.part["arrays"].items()}
        for cls in (CausalJointRefiner, HistoryValidityRefiner):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(7)
                model = cls(4, 8)
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                stats = self.train(model)
                self.assertEqual(stats["valid_frames"], 24)
                self.assertEqual(stats["inframe_frames"], 18)
                self.assertEqual(stats["stable_pairs"], 18)
                self.assertEqual(stats["transition_frames"], 12)
                self.assertEqual(stats["chunks"], 4)
                self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                    if name.startswith("heatmap_residual")))
                self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                    if name.startswith("inout_residual")))
                self.assertAlmostEqual(stats["total_loss"], stats["heatmap_loss"] + .1 * (
                    stats["state_loss"] + .1 * (stats["stable_loss"] + stats["transition_loss"])), places=12)
                self.assertFalse(any(name.startswith("backbone") for name, _ in model.named_parameters()))
        np.testing.assert_array_equal(self.part["features"], before_features)
        for name, before in before_arrays.items():
            np.testing.assert_array_equal(self.part["arrays"][name], before)

    def test_zero_update_chunk_sizes_keep_epoch_counts_and_pooled_loss_terms(self):
        torch.manual_seed(11)
        initial = CausalJointRefiner(4, 8)
        with torch.no_grad():
            initial.inout_residual.weight.fill_(.2)
        results = [self.train(copy.deepcopy(initial), clip_frames=size, learning_rate=0)
                   for size in (1, 2, 3, 7, 12)]
        for result in results[1:]:
            for key in ("valid_frames", "inframe_frames", "stable_pairs", "transition_frames"):
                self.assertEqual(result[key], results[0][key])
            for key in ("heatmap_loss", "state_loss", "stable_loss", "transition_loss", "total_loss"):
                self.assertAlmostEqual(result[key], results[0][key], places=6)

    def test_GT_changes_loss_supervision_without_changing_forward_inputs_or_predictions(self):
        changed = copy.deepcopy(self.part["tracks"])
        for track in changed:
            for frame in track["frames"]:
                frame["inout"] = 1-frame["inout"]
        original_batch = next(iter_track_batches(self.part["tracks"], self.part["arrays"],
            self.part["feature_index"], self.part["features"], 12, 2))
        changed_batch = next(iter_track_batches(changed, self.part["arrays"],
            self.part["feature_index"], self.part["features"], 12, 2))
        torch.manual_seed(31)
        model = HistoryValidityRefiner(4, 8)
        with torch.no_grad():
            model.inout_residual.weight.fill_(.3)
        first = _forward_chunk(model, original_batch, "cpu", None)
        second = _forward_chunk(model, changed_batch, "cpu", None)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        for name in first[2]:
            torch.testing.assert_close(first[2][name], second[2][name], rtol=0, atol=0)
        original_loss, _ = state_temporal_loss(first[1], original_batch["target_inout"],
            original_batch["valid"], original_batch["reset"])
        changed_loss, _ = state_temporal_loss(second[1], changed_batch["target_inout"],
            changed_batch["valid"], changed_batch["reset"])
        self.assertNotEqual(original_loss["transition"].item(), changed_loss["transition"].item())


if __name__ == "__main__":
    unittest.main()
