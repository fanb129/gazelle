"""T2 structural and gradient checks; no local support fitting is performed."""

import unittest

import torch
from dinov2.models.vision_transformer import DinoVisionTransformer

from gazelle.routing.losses import coverage_router_loss
from gazelle.routing.router import CoverageAwareSpatialRouter
from cscwd_router.code.data_prep.synthetic_support import make_synthetic_support_batch
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone
from cscwd_router.code.stag.router_dinov2 import GazeConditionedDinoV2Encoder


class DinoV2RouterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3106)
        torch.set_num_threads(1)
        self.model = DinoVisionTransformer(
            img_size=56, patch_size=14, embed_dim=24, depth=12,
            num_heads=3, mlp_ratio=2, init_values=1.0, block_chunks=0,
        ).eval()
        self.encoder = GazeConditionedDinoV2Encoder(
            RoutedDinoV2Backbone(model=self.model), hidden_dim=16, keep_ratio=0.5, escape_tokens=0,
        )
        self.batch = make_synthetic_support_batch(image_size=56)

    def test_existing_router_receives_normalized_block5_features(self):
        self.assertIs(type(self.encoder.router), CoverageAwareSpatialRouter)
        state, expected = self.encoder.forward_prefix(self.batch.images)
        captured = []
        hook = self.encoder.router.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        routing = self.encoder.predict_support(self.batch.images, self.batch.bboxes)
        hook.remove()
        self.assertTrue(torch.equal(captured[0], expected))
        self.assertEqual(tuple(captured[0].shape), (2, 24, 4, 4))
        self.assertFalse(state.tokens.requires_grad)
        self.assertFalse(captured[0].requires_grad)
        self.assertTrue(routing.support_probs.requires_grad)
        self.assertEqual(routing.person_to_image.tolist(), [0, 0, 1])

    def test_multi_person_union_uses_one_fixed_budget_suffix_per_image(self):
        trace = []
        hooks = [block.register_forward_pre_hook(lambda _module, args: trace.append(tuple(args[0].shape[:2])))
                 for block in self.model.blocks]
        output = self.encoder(self.batch.images, self.batch.bboxes)
        for hook in hooks:
            hook.remove()
        self.assertEqual(trace, [(2, 17)] * 6 + [(2, 9)] * 6)
        self.assertEqual(tuple(output.features.shape), (2, 24, 4, 4))
        routing = output.routing
        self.assertEqual(tuple(routing.support_probs.shape), (3, 4, 4))
        self.assertTrue(torch.equal(routing.image_union_probs[0], routing.support_probs[:2].amax(0)))
        self.assertTrue(torch.equal(routing.image_union_probs[1], routing.support_probs[2]))
        self.assertEqual(routing.image_hard_masks.flatten(1).sum(1).tolist(), [8, 8])
        self.assertTrue(torch.equal(routing.image_keep_indices, routing.image_keep_indices.sort(1).values))
        self.assertEqual(routing.actual_keep_ratio, 0.5)

    def test_support_only_skips_suffix_and_keeps_backbone_frozen_in_train_mode(self):
        self.encoder.train()
        self.assertTrue(self.encoder.router.training)
        self.assertFalse(any(module.training for module in self.encoder.backbone.modules()))
        self.assertFalse(any(p.requires_grad for p in self.encoder.backbone.parameters()))
        calls = []
        hooks = [block.register_forward_pre_hook(lambda _module, args: calls.append(1)) for block in self.model.blocks[6:]]
        self.encoder.predict_support(self.batch.images, self.batch.bboxes)
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, [])

    def test_k100_still_matches_official_forward_features(self):
        self.encoder.router.keep_ratio = 1.0
        with torch.no_grad():
            output = self.encoder(self.batch.images, self.batch.bboxes)
            expected = self.model.forward_features(self.batch.images)["x_norm_patchtokens"]
        self.assertTrue(torch.equal(output.features.flatten(2).transpose(1, 2), expected))
        self.assertEqual(output.routing.image_keep_indices.tolist(), [list(range(16))] * 2)

    def test_unselected_grid_uses_the_same_block5_features(self):
        _, early = self.encoder.forward_prefix(self.batch.images)
        output = self.encoder(self.batch.images, self.batch.bboxes)
        dropped = ~output.routing.image_hard_masks
        actual = output.features.permute(0, 2, 3, 1)[dropped]
        expected = early.permute(0, 2, 3, 1)[dropped]
        self.assertTrue(torch.equal(actual, expected))

    def test_person_permutation_preserves_the_learned_union_and_route(self):
        torch.nn.init.normal_(self.encoder.router.score_head[-1].weight, std=0.1)
        first = self.encoder.predict_support(self.batch.images, self.batch.bboxes)
        swapped_boxes = [list(reversed(boxes)) for boxes in self.batch.bboxes]
        second = self.encoder.predict_support(self.batch.images, swapped_boxes)
        self.assertTrue(torch.equal(first.support_probs[[1, 0, 2]], second.support_probs))
        self.assertTrue(torch.equal(first.image_union_probs, second.image_union_probs))
        self.assertTrue(torch.equal(first.image_keep_indices, second.image_keep_indices))

    def test_zero_initialized_support_has_a_finite_coverage_gradient(self):
        routing = self.encoder.predict_support(self.batch.images, self.batch.bboxes)
        losses = coverage_router_loss(routing, self.batch.target_heatmaps)
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        gradient = self.encoder.router.score_head[-1].weight.grad
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.encoder.backbone.parameters()))

    def test_scene_head_and_geometry_branches_are_all_connected(self):
        torch.nn.init.normal_(self.encoder.router.score_head[-1].weight, std=0.1)
        images = self.batch.images.requires_grad_(True)
        routing = self.encoder.predict_support(images, self.batch.bboxes)
        coverage_router_loss(routing, self.batch.target_heatmaps)["total"].backward()
        for module in (self.encoder.router.scene_projection, self.encoder.router.head_projection,
                       self.encoder.router.geometry_projection):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        self.assertIsNone(images.grad)

    def test_missing_head_box_is_rejected_instead_of_using_a_placeholder(self):
        _, features = self.encoder.forward_prefix(self.batch.images)
        with self.assertRaisesRegex(ValueError, "actual head bboxes"):
            self.encoder.route_features(features, [[None], self.batch.bboxes[1]])

    def test_synthetic_fixture_is_rgb_and_person_targets_are_aligned(self):
        self.assertEqual(tuple(self.batch.images.shape), (2, 3, 56, 56))
        self.assertEqual(tuple(self.batch.target_heatmaps.shape), (3, 64, 64))
        self.assertEqual([len(boxes) for boxes in self.batch.bboxes], [2, 1])
        again = make_synthetic_support_batch(image_size=56)
        self.assertTrue(torch.equal(self.batch.images, again.images))
        self.assertTrue(torch.equal(self.batch.target_heatmaps, again.target_heatmaps))
        maxima_x = self.batch.target_heatmaps.flatten(1).argmax(1) % 64
        self.assertGreater(maxima_x[0].item(), maxima_x[1].item())

    def test_server_route_checks_cover_all_three_budgets(self):
        from cscwd_router.code.scripts.check_router_dinov2 import check_routes

        _, early = self.encoder.forward_prefix(self.batch.images)
        cases = check_routes(self.encoder, self.batch.images, self.batch.bboxes, early)
        self.assertTrue(all(case["passed"] for case in cases))
        self.assertEqual([case["keep_count"] for case in cases], [4, 8, 16])
        self.assertEqual(self.encoder.router.keep_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
