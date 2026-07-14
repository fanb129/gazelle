from __future__ import annotations

import unittest

import torch
from torch import nn

from AAAIAlchemyModels.fusion import CrossLayerAttentionResidual, ResidualRefinement
from AAAIAlchemyModels.model import AlchemyGazeLLE
from gazelle.model import GazeLLE


class DummyBackbone(nn.Module):
    def __init__(self, channels: int = 8, feature_size: int = 2):
        super().__init__()
        self.channels = channels
        self.feature_size = feature_size
        self.projection = nn.Conv2d(3, channels, kernel_size=1)
        self.out_indices = [2, 5, 8, 11]

    def get_dimension(self):
        return self.channels

    def get_out_size(self, _in_size):
        return self.feature_size, self.feature_size

    def forward(self, images):
        base = torch.nn.functional.adaptive_avg_pool2d(
            self.projection(images), (self.feature_size, self.feature_size)
        )
        return [base + float(index) / 10.0 for index in range(4)]


def build_legacy(inout: bool = True):
    return GazeLLE(
        DummyBackbone(),
        inout=inout,
        dim=32,
        num_layers=1,
        in_size=(32, 32),
        out_size=(8, 8),
        use_sasa=True,
        use_ggsf=True,
        dropout=0.0,
    )


def build_alchemy(refinement: str, inout: bool = True):
    return AlchemyGazeLLE(
        DummyBackbone(),
        refinement=refinement,
        inout=inout,
        dim=32,
        num_layers=1,
        in_size=(32, 32),
        out_size=(8, 8),
        refinement_width=16,
        attention_dim=16,
        attention_heads=4,
        dropout=0.0,
    )


def sample_input():
    return {
        "images": torch.randn(2, 3, 16, 16),
        "bboxes": [
            [[0.1, 0.1, 0.3, 0.4], [0.5, 0.2, 0.8, 0.5]],
            [[0.2, 0.3, 0.4, 0.6]],
        ],
    }


class FusionUnitTests(unittest.TestCase):
    def test_synthetic_branch_shapes_and_zero_outputs(self):
        features = [torch.randn(3, 8, 4, 4) for _ in range(4)]
        branches = (
            ResidualRefinement(8, 16, width=12),
            CrossLayerAttentionResidual(8, 16, attention_dim=16, num_heads=4),
        )
        for branch in branches:
            output = branch(features)
            self.assertEqual(tuple(output.shape), (3, 16, 4, 4))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_branch_rejects_invalid_hierarchy(self):
        branch = ResidualRefinement(8, 16, width=12)
        with self.assertRaisesRegex(ValueError, "expected 4"):
            branch([torch.randn(2, 8, 4, 4) for _ in range(3)])


class ModelIntegrationTests(unittest.TestCase):
    def test_r0_shapes_for_multiple_people(self):
        torch.manual_seed(1)
        model = build_alchemy("none").eval()
        output = model(sample_input())
        self.assertEqual([tuple(item.shape) for item in output["heatmap"]], [(2, 8, 8), (1, 8, 8)])
        self.assertEqual([tuple(item.shape) for item in output["inout"]], [(2,), (1,)])
        self.assertEqual(tuple(output["layer_weights"].shape), (3, 4))

    def test_zero_init_equivalence_to_historical_sasa_ggsf(self):
        torch.manual_seed(7)
        legacy = build_legacy().eval()
        state = legacy.get_gazelle_state_dict(include_backbone=False)
        batch = sample_input()
        legacy_output = legacy(batch)

        for refinement in ("none", "residual_refinement", "cross_layer_attention"):
            candidate = build_alchemy(refinement).eval()
            # Real runs reload the same frozen DINOv3 weights outside the task
            # checkpoint; mirror that invariant for the synthetic backbone.
            candidate.backbone.load_state_dict(legacy.backbone.state_dict())
            report = candidate.load_alchemy_checkpoint(state)
            self.assertEqual(report["shared_base_coverage"], 1.0)
            self.assertEqual(report["unexpected"], [])
            self.assertEqual(report["incompatible_shapes"], {})
            candidate_output = candidate(batch)
            for expected, actual in zip(legacy_output["heatmap"], candidate_output["heatmap"]):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            for expected, actual in zip(legacy_output["inout"], candidate_output["inout"]):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                candidate_output["layer_weights"], legacy_output["layer_weights"], rtol=0.0, atol=0.0
            )

    def test_zero_init_branches_receive_gradient(self):
        for refinement in ("residual_refinement", "cross_layer_attention"):
            torch.manual_seed(11)
            model = build_alchemy(refinement).train()
            output = model(sample_input())
            loss = sum(item.mean() for item in output["heatmap"]) + sum(
                item.mean() for item in output["inout"]
            )
            loss.backward()
            gradient = model.fusion_refiner.output_projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_checkpoint_report_identifies_only_new_branch_as_expected_missing(self):
        legacy = build_legacy().eval()
        candidate = build_alchemy("residual_refinement").eval()
        report = candidate.load_alchemy_checkpoint(legacy.get_gazelle_state_dict(False))
        self.assertEqual(report["shared_base_coverage"], 1.0)
        self.assertTrue(report["new_branch_missing"])
        self.assertEqual(report["missing"], report["new_branch_missing"])

    def test_gazefollow_to_vat_transfer_coverage_excludes_only_inout_head(self):
        source = build_alchemy("residual_refinement", inout=False).eval()
        target = build_alchemy("residual_refinement", inout=True).eval()
        report = target.load_alchemy_checkpoint(source.get_gazelle_state_dict(False))
        self.assertEqual(report["cross_dataset_shared_coverage"], 1.0)
        self.assertTrue(report["missing"])
        self.assertTrue(
            all(key.startswith(("inout_token.", "inout_head.")) for key in report["missing"])
        )


if __name__ == "__main__":
    unittest.main()
