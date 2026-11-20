"""Synthetic checks for the trainable causal adapter and frozen heads."""

import tempfile
import unittest
from pathlib import Path

import torch

from cscwd.code.stag.model import FrozenTaskHeads, TemporalModel, detach_state


class TemporalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_real_shape_identity_frozen_heads_and_gradients(self):
        heads = FrozenTaskHeads()
        model = TemporalModel(heads).train()
        x, z = torch.randn(1, 256, 32, 32), torch.randn(1, 256)
        original_heatmap, original_inout = heads(x, z)
        heatmap, inout, state = model.forward_step(x, z)
        self.assertEqual(tuple(heatmap.shape), (1, 64, 64))
        self.assertTrue(torch.equal(heatmap, original_heatmap))
        self.assertTrue(torch.equal(inout, original_inout))
        self.assertFalse(any(module.training for module in heads.modules()))
        self.assertFalse(any(parameter.requires_grad for parameter in heads.parameters()))
        before = {name: value.clone() for name, value in heads.state_dict().items()}
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad()
            heatmap, inout, state = model.forward_step(x, z)
            loss = heatmap.square().mean() + 0.1 * inout.square().mean()
            loss.backward()
            projections = ("spatial_output", "token_output") if step == 0 else (
                "spatial_input", "token_input", "cell", *projections
            )
            for prefix in projections:
                gradients = [parameter.grad for name, parameter in model.adapter.named_parameters()
                             if name.startswith(prefix)]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in heads.parameters()))
        self.assertTrue(all(torch.equal(value, before[name])
                            for name, value in heads.state_dict().items()))

    def test_causality_chunk_detach_and_slot_reset(self):
        model = TemporalModel(FrozenTaskHeads(feature_channels=8), hidden_channels=3,
                              feature_channels=8).eval()
        torch.nn.init.normal_(model.adapter.spatial_output.weight, std=0.1)
        torch.nn.init.normal_(model.adapter.token_output.weight, std=0.1)
        xs, zs = torch.randn(4, 2, 8, 3, 3), torch.randn(4, 2, 8)

        def rollout(x_values, z_values, detach_at=None):
            state = None
            outputs = []
            for time, (x, z) in enumerate(zip(x_values, z_values)):
                if time == detach_at:
                    state = detach_state(state)
                    self.assertTrue(all(not value.requires_grad for value in state))
                heatmap, inout, state = model.forward_step(x, z, state)
                outputs.append((heatmap, inout))
            return outputs, state

        full, state = rollout(xs, zs)
        chunked, _ = rollout(xs, zs, detach_at=2)
        changed_xs, changed_zs = xs.clone(), zs.clone()
        changed_xs[2:], changed_zs[2:] = 100, -100
        changed, _ = rollout(changed_xs, changed_zs)
        for time in range(4):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(full[time], chunked[time])))
            if time < 2:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(full[time], changed[time])))
        with_reset = model.forward_step(xs[0], zs[0], state, torch.tensor([True, False]))
        fresh = model.forward_step(xs[0], zs[0])
        continued = model.forward_step(xs[0], zs[0], state)
        for output in range(2):
            self.assertTrue(torch.equal(with_reset[output][0], fresh[output][0]))
            self.assertTrue(torch.equal(with_reset[output][1], continued[output][1]))
        for reset_value, fresh_value in zip(with_reset[2], fresh[2]):
            self.assertTrue(torch.equal(reset_value[0], fresh_value[0]))

    def test_heads_round_trip(self):
        heads = FrozenTaskHeads()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.pt"
            torch.save({"heatmap_head": heads.heatmap_head.state_dict(),
                        "inout_head": heads.inout_head.state_dict()}, path)
            loaded = FrozenTaskHeads.from_checkpoint(path)
            self.assertTrue(all(torch.equal(value, loaded.state_dict()[name])
                                for name, value in heads.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
