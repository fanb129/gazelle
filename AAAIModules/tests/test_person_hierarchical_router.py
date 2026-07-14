import torch
import torch.nn as nn

from AAAIModules.person_hierarchical_gazelle import PersonHierarchicalGazeLLE
from AAAIModules.person_hierarchical_router import PersonConditionedHierarchicalRouter


class DummyBackbone(nn.Module):
    def __init__(self, channels=8, height=4, width=4):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.out_indices = [2, 5, 8, 11]
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        base = images.mean(dim=1, keepdim=True)
        base = torch.nn.functional.interpolate(base, (self.height, self.width))
        return [base.repeat(1, self.channels, 1, 1) + index for index in range(4)]

    def get_dimension(self):
        return self.channels

    def get_out_size(self, _in_size):
        return self.height, self.width


def test_router_produces_person_weights_that_sum_to_one():
    router = PersonConditionedHierarchicalRouter(8, dropout=0.0)
    features = [torch.randn(2, 8, 4, 4) for _ in range(4)]
    weighted, weights, metadata = router(
        features,
        [[[0.0, 0.0, 0.4, 0.5], [0.5, 0.5, 1.0, 1.0]], [[0.2, 0.1, 0.8, 0.9]]],
    )
    assert weights.shape == (3, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))
    assert len(weighted) == 4
    assert weighted[0].shape == (3, 8, 4, 4)
    assert metadata["models_query_interaction"] is False


def test_router_starts_from_measured_global_prior():
    prior = (0.0340173, 0.0974448, 0.2007198, 0.6678180)
    router = PersonConditionedHierarchicalRouter(8, dropout=0.0, prior_weights=prior).eval()
    features = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    with torch.no_grad():
        _, weights, metadata = router(features, [[[0.1, 0.1, 0.5, 0.6]]])
    expected = torch.tensor(prior) / sum(prior)
    assert torch.allclose(weights[0], expected, atol=1e-6)
    assert metadata["routing_decomposition"] == "task_global_prior_plus_person_residual"


def test_synthetic_model_smoke_without_dino_weights():
    backbone = DummyBackbone()
    model = PersonHierarchicalGazeLLE(
        backbone,
        inout=True,
        dim=32,
        num_layers=1,
        in_size=(64, 64),
        out_size=(16, 16),
        router_hidden_dim=16,
        router_dropout=0.0,
    ).eval()
    inputs = {
        "images": torch.randn(2, 3, 64, 64),
        "bboxes": [[[0.1, 0.1, 0.4, 0.5], [0.5, 0.2, 0.8, 0.7]], [[0.2, 0.2, 0.6, 0.6]]],
    }
    with torch.no_grad():
        outputs = model(inputs)
    assert backbone.calls == 1
    assert [item.shape for item in outputs["heatmap"]] == [(2, 16, 16), (1, 16, 16)]
    assert [item.shape for item in outputs["inout"]] == [(2,), (1,)]
    assert outputs["layer_weights"].shape == (3, 4)
    assert len(outputs["layer_weights_split"]) == 2
    assert outputs["metadata"]["shared_scene_encoding"] is True
    assert outputs["metadata"]["models_query_interaction"] is False


def test_base_checkpoint_compatibility_keeps_new_router_parameters_missing():
    source = PersonHierarchicalGazeLLE(
        DummyBackbone(), dim=32, num_layers=1, in_size=(64, 64), out_size=(16, 16),
        router_hidden_dim=16,
    )
    historical = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("layer_router.")
    }
    target = PersonHierarchicalGazeLLE(
        DummyBackbone(), dim=32, num_layers=1, in_size=(64, 64), out_size=(16, 16),
        router_hidden_dim=16,
    )
    report = target.load_base_checkpoint(historical)
    assert report["loaded"]
    assert report["incompatible_shapes"] == []
    assert report["unexpected"] == []
    assert report["missing"]
    assert all(key.startswith("layer_router.") for key in report["missing"])


def test_legacy_sasa_ggsf_keys_can_be_explicitly_ignored():
    model = PersonHierarchicalGazeLLE(
        DummyBackbone(), dim=32, num_layers=1, in_size=(64, 64), out_size=(16, 16),
        router_hidden_dim=16,
    )
    historical = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("layer_router.")
    }
    historical["sasa.fake_weight"] = torch.ones(1)
    historical["ggsf.fake_weight"] = torch.ones(1)
    report = model.load_base_checkpoint(historical, allow_legacy_sasa_ggsf=True)
    assert report["unexpected"] == []
    assert report["ignored_source_keys"] == ["ggsf.fake_weight", "sasa.fake_weight"]
