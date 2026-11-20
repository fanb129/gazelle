"""Small synthetic end-to-end checks; no VAT data or pretrained weights."""

import json

import numpy as np
import torch
from torch import nn

from cscwd.code.data_prep.task_cache import FrozenTaskFeatureExtractor, extract_task_cache, sha256
from cscwd.code.data_prep.test_task_cache import SyntheticGazeSpot, make_synthetic_tracks, transform
from cscwd.code.scripts import train_convlstm
from cscwd.code.stag.data import chunk_rows
from cscwd.code.stag.engine import VATMetrics, checkpoint_rank, task_loss
from cscwd.code.stag.model import FrozenTaskHeads, TemporalModel


def test_chunk_order_coverage_gap_and_person_resets():
    tracks = []
    row = 0
    for person, numbers in (("a", [1, 2, 3, 4, 9, 10]), ("b", [7]), ("c", [1, 2, 3])):
        frames = [{"row": row + offset, "frame_index": number}
                  for offset, number in enumerate(numbers)]
        tracks.append({"track_id": person, "frames": frames})
        row += len(numbers)
    batches = list(chunk_rows(tracks, batch_size=2, clip_frames=2))
    visited = [int(index) for rows, _ in batches for index in rows[rows >= 0]]
    assert sorted(visited) == list(range(row))
    assert len(set(visited)) == row
    # Slot 0 keeps person a across chunks; the original frame gap is at a boundary.
    assert batches[1][0][0].tolist() == [2, 3]
    assert batches[1][1][0].tolist() == [False, False]
    assert batches[2][1][0].tolist() == [True, False]
    # Slot 1 switches from b to c only between chunks and resets its memory.
    assert batches[1][0][1].tolist() == [7, 8]
    assert bool(batches[1][1][1, 0])
    assert all(reset[rows < 0].all() for rows, reset in batches)


def test_loss_metrics_population_and_epoch_selection():
    heatmaps = torch.full((1, 2, 64, 64), 0.2, requires_grad=True)
    scores = torch.full((1, 2), 0.001, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([[True, True]])
    target = torch.zeros_like(heatmaps)
    loss, mse, bce = task_loss(heatmaps, scores, target, labels, valid)
    torch.testing.assert_close(mse, torch.tensor(0.04))
    torch.testing.assert_close(loss, mse + 0.1 * bce)
    loss.backward()
    assert heatmaps.grad[0, 1].count_nonzero() == 0
    assert scores.grad.count_nonzero() == 2
    outside_loss, outside_mse, _ = task_loss(
        heatmaps, scores, target, torch.zeros_like(labels), valid,
    )
    assert torch.isfinite(outside_loss) and outside_mse == 0
    accumulator = VATMetrics()
    accumulator.update(heatmaps[0], scores[0], labels[0], np.array([[0.5, 0.5], [-1, -1]]))
    result = accumulator.compute()
    assert result["gt_in_frame"] == 1 and result["person_frames"] == 2
    candidates = [
        ({"auc": 0.8, "l2": 0.2, "inout_ap": 0.7}, 3),
        ({"auc": 0.8, "l2": 0.2, "inout_ap": 0.7}, 4),
        ({"auc": 0.8, "l2": 0.2, "inout_ap": 0.6}, 2),
        ({"auc": 0.8, "l2": 0.3, "inout_ap": 0.9}, 1),
        ({"auc": 0.7, "l2": 0.1, "inout_ap": 0.9}, 1),
    ]
    assert sorted(candidates, key=lambda item: checkpoint_rank(*item), reverse=True) == candidates


def test_actual_gazespot_path_head_input_identity():
    from gazelle.model import GazeLLE

    class SyntheticBackbone(nn.Module):
        def get_dimension(self):
            return 4

        def get_out_size(self, in_size):
            return 32, 32

        def forward(self, images):
            feature = images.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)
            return [feature * scale for scale in (0.5, 1.0, 1.5, 2.0)]

    torch.manual_seed(0)
    torch.set_num_threads(1)
    base = GazeLLE(SyntheticBackbone(), inout=True, use_sasa=True, use_ggsf=True)
    base.eval()
    images, boxes = torch.randn(1, 3, 32, 32), [[[0.1, 0.2, 0.4, 0.6]]]
    with torch.no_grad():
        original = base({"images": images, "bboxes": boxes})
    extractor = FrozenTaskFeatureExtractor(base)
    features = extractor(images, boxes)
    extractor.close()
    temporal = TemporalModel(FrozenTaskHeads(base.heatmap_head, base.inout_head))
    heatmap, inout, _ = temporal.forward_step(features["x"], features["z"])
    torch.testing.assert_close(heatmap, torch.cat(original["heatmap"]), rtol=0, atol=0)
    torch.testing.assert_close(inout, torch.cat(original["inout"]), rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_synthetic_cache_train_test_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    base = SyntheticGazeSpot()
    checkpoint = tmp_path / "synthetic_base.pt"
    torch.save(base.state_dict(), checkpoint)
    caches = {}
    for split in ("train", "test"):
        tracks = make_synthetic_tracks(tmp_path, split)
        caches[split] = extract_task_cache(
            base, transform, tmp_path, tracks, checkpoint, tmp_path / split,
            split, device="cpu", batch_size=2, num_workers=0,
        )
    # Force one-frame chunks to exercise state carry and detachment in the engine.
    calls = []

    class RecordingTemporalModel(TemporalModel):
        def forward_step(self, x, z, state=None, reset=None):
            if self.training:
                calls.append(None if state is None else [value.requires_grad for value in state])
            return super().forward_step(x, z, state, reset)

    monkeypatch.setattr(train_convlstm, "TemporalModel", RecordingTemporalModel)
    output, reports = tmp_path / "checkpoints", tmp_path / "reports"
    provenance = tmp_path / "RESULT_PROVENANCE.md"
    comparison = train_convlstm.run_training(
        tmp_path / "train", tmp_path / "test", output, reports, provenance,
        train_convlstm.TrainingConfig(epochs=2, clip_frames=1, batch_size=2,
                                     hidden_channels=3, device="cpu", log_every=100),
        command="synthetic-check-only",
    )
    assert calls == [None, [False, False], None, [False, False]]
    history = json.loads((reports / "training_history.json").read_text())
    assert [entry["epoch"] for entry in history["epochs"]] == [1, 2]
    assert all(entry["train"]["person_frames"] == entry["test"]["person_frames"] == 3
               and entry["test"]["gt_in_frame"] == 1 for entry in history["epochs"])
    selected = max(history["epochs"], key=lambda item: checkpoint_rank(item["test"], item["epoch"]))
    assert comparison["temporal"]["epoch"] == selected["epoch"] == history["selected_epoch"]
    weights = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
    assert weights["epoch"] == selected["epoch"]
    assert weights["metrics"] == selected
    assert torch.load(output / "last.pt", map_location="cpu", weights_only=True)["epoch"] == 2
    restored = TemporalModel(FrozenTaskHeads(), hidden_channels=3)
    restored.load_state_dict(weights["model"], strict=True)
    assert all(not parameter.requires_grad for parameter in restored.heads.parameters())
    artifacts = json.loads((reports / "artifacts.json").read_text())
    for artifact in [*artifacts["checkpoints"], artifacts["selected"], artifacts["last"]]:
        assert sha256(artifact["path"]) == artifact["sha256"]
        assert artifact["sha256"] in provenance.read_text()
    assert weights["protocol"]["test_used_for_checkpoint_selection"]
    assert (reports / "comparison.md").is_file()
