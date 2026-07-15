import pytest


torch = pytest.importorskip("torch")
from torch import nn

from AAAISelectiveGaze.models.layer_probe import FixedLayerProbes
from AAAISelectiveGaze.models.selective_gazelle import SelectiveGazelle


class FakeBackbone(nn.Module):
    out_indices = [2, 5, 8, 11]

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        base = images.mean(dim=1, keepdim=True)[:, :, :4, :4] * self.weight
        return [base.repeat(1, 3, 1, 1) + index for index in range(4)]


class FakePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.decoder = nn.Parameter(torch.ones(()))

    def get_input_head_maps(self, bboxes):
        output = []
        for image_boxes in bboxes:
            maps = []
            for _ in image_boxes:
                item = torch.zeros(4, 4)
                item[0:2, 0:2] = 1
                maps.append(item)
            output.append(torch.stack(maps))
        return output

    def forward(self, inputs):
        features = self.backbone.forward(inputs["images"])
        counts = [len(items) for items in inputs["bboxes"]]
        heatmaps = [torch.zeros(count, 64, 64) + self.decoder for count in counts]
        return {"heatmap": heatmaps, "inout": None, "feature_count": len(features)}


def test_wrapper_reuses_one_backbone_pass_and_splits_probe_outputs():
    predictor = FakePredictor()
    probes = FixedLayerProbes(in_channels=3, hidden_channels=4)
    model = SelectiveGazelle(predictor, probes)
    output = model(
        {
            "images": torch.randn(2, 3, 4, 4),
            "bboxes": [
                [[0.0, 0.0, 0.5, 0.5]],
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
            ],
        }
    )

    assert predictor.backbone.calls == 1
    assert list(output["probe_heatmap"]) == [2, 5, 8, 11]
    assert output["probe_heatmap"][2][0].shape == (1, 64, 64)
    assert output["probe_heatmap"][2][1].shape == (2, 64, 64)
    assert all(not parameter.requires_grad for parameter in predictor.parameters())


def test_wrapper_rejects_mismatched_layer_contract():
    with pytest.raises(ValueError, match="do not match"):
        SelectiveGazelle(
            FakePredictor(),
            FixedLayerProbes(in_channels=3, layers=(1, 2, 3, 4)),
            layers=(2, 5, 8, 11),
        )
