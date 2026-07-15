import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from AAAISelectiveGaze.models.layer_probe import (
    DEFAULT_LAYERS,
    FixedLayerProbes,
    LayerGazeProbe,
    count_parameters,
    freeze_module,
)


def _features(batch=2, channels=8, height=6, width=7):
    return {
        layer: torch.randn(batch, channels, height, width)
        for layer in DEFAULT_LAYERS
    }


def test_layer_probe_shape_range_and_gradient():
    probe = LayerGazeProbe(in_channels=8, hidden_channels=4)
    feature = torch.randn(3, 8, 6, 7, requires_grad=True)
    head_map = torch.zeros(3, 6, 7)
    head_map[:, 1:3, 2:4] = 1

    logits = probe(feature, head_map)
    heatmap = probe.predict_heatmap(feature, head_map)
    assert logits.shape == (3, 64, 64)
    assert heatmap.shape == (3, 64, 64)
    assert torch.all((heatmap >= 0) & (heatmap <= 1))

    logits.mean().backward()
    assert feature.grad is not None
    assert all(parameter.grad is not None for parameter in probe.parameters())


def test_fixed_probes_use_default_layers_and_independent_parameters():
    probes = FixedLayerProbes(in_channels=8, hidden_channels=4)
    assert probes.layers == DEFAULT_LAYERS
    assert len(probes.probes) == 4

    projection_weights = [
        probes.probes[str(layer)].projection.weight for layer in DEFAULT_LAYERS
    ]
    assert len({weight.data_ptr() for weight in projection_weights}) == 4
    original = projection_weights[1].detach().clone()
    with torch.no_grad():
        projection_weights[0].add_(1)
    assert torch.equal(projection_weights[1], original)


def test_fixed_probes_mapping_sequence_shapes_and_all_probe_gradients():
    probes = FixedLayerProbes(in_channels=8, hidden_channels=4)
    features = _features()
    head_map = torch.zeros(2, 6, 7)
    outputs = probes(features, head_map)
    sequence_outputs = probes([features[layer] for layer in DEFAULT_LAYERS], head_map)

    assert list(outputs) == list(DEFAULT_LAYERS)
    assert all(output.shape == (2, 64, 64) for output in outputs.values())
    assert all(
        torch.allclose(outputs[layer], sequence_outputs[layer])
        for layer in DEFAULT_LAYERS
    )

    sum(output.mean() for output in outputs.values()).backward()
    for layer in DEFAULT_LAYERS:
        assert all(
            parameter.grad is not None
            for parameter in probes.probes[str(layer)].parameters()
        )


def test_fixed_probes_validate_layer_count_keys_and_shapes():
    probes = FixedLayerProbes(in_channels=8, hidden_channels=4)
    features = _features()
    head_map = torch.zeros(2, 6, 7)

    with pytest.raises(ValueError, match="expected 4 feature tensors"):
        probes(list(features.values())[:3], head_map)
    with pytest.raises(ValueError, match="exactly match"):
        probes({2: features[2], 5: features[5], 8: features[8]}, head_map)
    bad = dict(features)
    bad[11] = torch.randn(2, 8, 5, 7)
    with pytest.raises(ValueError, match=r"share \[N, H, W\]"):
        probes(bad, head_map)
    with pytest.raises(ValueError, match="spatial shapes differ"):
        probes(features, torch.zeros(2, 5, 7))


def test_freeze_and_parameter_count_helpers():
    module = LayerGazeProbe(in_channels=8, hidden_channels=4)
    assert count_parameters(module) > 0
    assert count_parameters(module, trainable_only=True) == count_parameters(module)
    returned = freeze_module(module)
    assert returned is module
    assert not module.training
    assert count_parameters(module, trainable_only=True) == 0
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_synthetic_smoke_cli_writes_checkpoint_and_summary(tmp_path):
    output_dir = tmp_path / "probe-smoke"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "AAAISelectiveGaze.scripts.train_layer_probes",
            "--synthetic-smoke",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-channels",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    checkpoint = torch.load(output_dir / "layer_probes.pt", map_location="cpu")
    summary = json.loads((output_dir / "summary.json").read_text())
    assert checkpoint["layers"] == list(DEFAULT_LAYERS)
    assert summary["status"] == "ok"
    assert summary["synthetic_smoke"] is True
    assert summary["backbone_loaded"] is False
    assert summary["parameter_count"] > 0
    assert len(summary["history"]) == 1
