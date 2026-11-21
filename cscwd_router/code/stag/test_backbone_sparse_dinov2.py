"""Synthetic CPU checks using the actual official DINOv2 implementation.

Put a DINOv2 checkout on PYTHONPATH; no checkpoint or dataset is needed.
The small random model keeps the official patch, position, block and norm code.
"""

import io
import unittest
from unittest.mock import patch

import torch
from dinov2.models.vision_transformer import DinoVisionTransformer

from cscwd_router.code.stag.backbone_sparse_dinov2 import (
    RoutedDinoV2Backbone,
    gather_patch_tokens,
    scatter_patch_tokens,
)


class SparseDinoV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3106)
        torch.set_num_threads(1)
        self.model = DinoVisionTransformer(
            img_size=56, patch_size=14, embed_dim=24, depth=12,
            num_heads=3, mlp_ratio=2, init_values=1.0, block_chunks=0,
        ).eval()
        self.backbone = RoutedDinoV2Backbone(model=self.model).eval()

    def test_k100_is_exactly_official_forward_features(self):
        # Cover position interpolation, a rectangular grid and multiple images.
        with torch.inference_mode():
            for batch_size, height, width in ((1, 56, 56), (2, 42, 70), (2, 84, 84)):
                images = torch.randn(batch_size, 3, height, width)
                expected = self.model.forward_features(images)["x_norm_patchtokens"]
                for route_after_block in (0, 5, 10):
                    with self.subTest(shape=tuple(images.shape), route=route_after_block):
                        actual = self.backbone(images, route_after_block=route_after_block).flatten(2).transpose(1, 2)
                        self.assertTrue(torch.equal(actual, expected))

    def test_k100_at_the_full_448_input_grid(self):
        with torch.inference_mode():
            images = torch.randn(1, 3, 448, 448)
            expected = self.model.forward_features(images)["x_norm_patchtokens"]
            actual = self.backbone(images)
            self.assertEqual(tuple(actual.shape), (1, 24, 32, 32))
            self.assertTrue(torch.equal(actual.flatten(2).transpose(1, 2), expected))

    def test_prefix_is_the_exact_block5_output(self):
        with torch.inference_mode():
            images = torch.randn(2, 3, 56, 56)
            expected = self.model.prepare_tokens_with_masks(images)
            for block in self.model.blocks[:6]:
                expected = block(expected)
            state = self.backbone.forward_prefix(images)
            self.assertTrue(torch.equal(state.tokens, expected))
            route_map = self.backbone.tokens_to_map(state.tokens, state.height, state.width)
            self.assertTrue(torch.equal(route_map.flatten(2).transpose(1, 2), self.model.norm(expected)[:, 1:]))

    def test_sparse_suffix_keeps_original_positions_and_early_fill(self):
        with torch.inference_mode():
            state = self.backbone.forward_prefix(torch.randn(2, 3, 56, 56))
            before = state.tokens.clone()
            indices = torch.tensor([[0, 3, 8, 15], [1, 4, 10, 14]])
            # Independent indexing includes CLS and keeps each image's positions.
            positions = torch.cat((torch.zeros(2, 1, dtype=torch.long), indices + 1), dim=1)
            expected_sparse = torch.stack([state.tokens[b, positions[b]] for b in range(2)])
            suffix_inputs = []
            hook = self.model.blocks[6].register_forward_pre_hook(lambda _module, args: suffix_inputs.append(args[0].clone()))
            actual = self.backbone.forward_suffix(state, indices).flatten(2).transpose(1, 2)
            hook.remove()
            self.assertTrue(torch.equal(suffix_inputs[0], expected_sparse))
            for block in self.model.blocks[6:]:
                expected_sparse = block(expected_sparse)
            expected_dense = state.tokens.clone()
            expected_dense[:, 0] = expected_sparse[:, 0]
            for b in range(2):
                expected_dense[b, indices[b] + 1] = expected_sparse[b, 1:]
            self.assertTrue(torch.equal(actual, self.model.norm(expected_dense)[:, 1:]))
            early = self.model.norm(state.tokens)[:, 1:]
            for b in range(2):
                dropped = torch.ones(16, dtype=torch.bool)
                dropped[indices[b]] = False
                self.assertTrue(torch.equal(actual[b, dropped], early[b, dropped]))
            self.assertTrue(torch.equal(state.tokens, before))

    def test_suffix_really_receives_k_patches_and_cls_in_both_paths(self):
        for keep_count in (8, 16):
            lengths = []
            hooks = [block.register_forward_pre_hook(lambda _module, args: lengths.append(args[0].shape[1]))
                     for block in self.model.blocks]
            indices = torch.arange(keep_count).expand(2, -1)
            with torch.inference_mode():
                self.backbone(torch.randn(2, 3, 56, 56), indices)
            for hook in hooks:
                hook.remove()
            self.assertEqual(lengths, [17] * 6 + [keep_count + 1] * 6)

    def test_gather_scatter_preserve_per_image_values_and_gradients(self):
        early = torch.randn(2, 16, 4, requires_grad=True)
        deep = torch.randn(2, 4, 4, requires_grad=True)
        indices = torch.tensor([[0, 3, 8, 15], [1, 4, 10, 14]])
        gathered = gather_patch_tokens(early, indices)
        for b in range(2):
            self.assertTrue(torch.equal(gathered[b], early[b, indices[b]]))
        filled = scatter_patch_tokens(early, deep, indices)
        filled.sum().backward()
        expected_early_grad = torch.ones_like(early)
        for b in range(2):
            expected_early_grad[b, indices[b]] = 0
        self.assertTrue(torch.equal(early.grad, expected_early_grad))
        self.assertTrue(torch.equal(deep.grad, torch.ones_like(deep)))

    def test_feature_gradients_reach_the_input_through_the_split(self):
        self.backbone.requires_grad_(False)
        images = torch.randn(1, 3, 56, 56, requires_grad=True)
        features = self.backbone(images, torch.tensor([[0, 3, 8, 15]]))
        features[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_official_state_dict_layout_and_round_trip(self):
        expected = {"model." + name: value for name, value in self.model.state_dict().items()}
        self.assertEqual(set(self.backbone.state_dict()), set(expected))
        with io.BytesIO() as checkpoint:
            torch.save(self.backbone.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = torch.load(checkpoint, weights_only=True)
        self.backbone.load_state_dict(restored, strict=True)
        self.assertTrue(all(torch.equal(value, expected[name]) for name, value in restored.items()))
        self.assertEqual(self.backbone.get_dimension(), 24)
        self.assertEqual(self.backbone.get_out_size((448, 448)), (32, 32))

    def test_server_check_case_uses_the_unmodified_official_decoder(self):
        from gazelle.backbone_v0_dinov2 import DinoV2Backbone
        from gazelle.model_v0_dinov2 import GazeLLE
        from cscwd_router.code.scripts.check_sparse_dinov2 import check_case

        # Replace only checkpoint I/O; both official forward methods run as-is.
        with patch("torch.hub.load", return_value=self.model), patch("torch.load", return_value=self.model.state_dict()):
            official = DinoV2Backbone("dinov2_vitb14")
        decoder = GazeLLE(official, dim=32, num_layers=1, in_size=(56, 56)).eval()
        case = check_case(official, self.backbone, decoder, batch_size=2, precision="fp32",
                          device=torch.device("cpu"), image_size=56, route_after_block=5)
        self.assertTrue(case["passed"])
        self.assertEqual(case["k100_official_decoder"]["mismatched_values"], 0)
        self.assertEqual(case["k50_suffix_lengths"], [9])


if __name__ == "__main__":
    unittest.main()
