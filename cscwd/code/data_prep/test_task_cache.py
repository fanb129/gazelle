"""Synthetic cache checks: head-input fidelity, frozen model and row mapping."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from cscwd.code.data_prep.task_cache import (
    FrozenTaskFeatureExtractor, TaskFeatureCache, extract_task_cache, sha256,
)


class SyntheticGazeSpot(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 256, 1)
        self.heatmap_head = nn.Sequential(nn.ConvTranspose2d(256, 256, 2, 2),
                                          nn.Conv2d(256, 1, 1, bias=False), nn.Sigmoid())
        self.inout_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, inputs):
        # Any GT key reaching the model is a test failure.
        assert set(inputs) == {"images", "bboxes"}
        scenes = self.backbone(inputs["images"])
        per_image = [torch.stack([scenes[i] + float(bbox[0]) for bbox in boxes])
                     for i, boxes in enumerate(inputs["bboxes"])]
        x = torch.cat(per_image)
        z = x.mean(dim=(-2, -1))
        heatmap = self.heatmap_head(x).squeeze(1)
        inout = self.inout_head(z).squeeze(-1)
        sizes = [len(boxes) for boxes in inputs["bboxes"]]
        return {"heatmap": list(heatmap.split(sizes)), "inout": list(inout.split(sizes))}


def transform(image):
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)


def make_synthetic_tracks(root, split="train"):
    root = Path(root)
    tracks = []
    for person, ids in (("b", [3]), ("a", [3, 1])):
        frames = []
        for number in ids:
            path = f"images/video/clip/{number:08d}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (number * 30, 40, 50)).save(root / path)
            frames.append({"path": path, "frame_index": number, "person_id": person,
                           "bbox_norm": [0.1 if person == "a" else 0.5, 0.1, 0.7, 0.9],
                           "gazex_norm": [0.5 if number == 1 else -1.0],
                           "gazey_norm": [0.5 if number == 1 else -1.0],
                           "inout": int(number == 1)})
        tracks.append({"track_id": f"{split}/video/clip/{person}", "person_id": person,
                       "source_csv": f"annotations/{split}/video/clip/{person}.txt", "frames": frames})
    path = root / f"{split}_tracks.json"
    path.write_text(json.dumps([{"tracks": tracks}]))
    return path


def test_frozen_extractor_matches_original_head_inputs():
    torch.manual_seed(0)
    model = SyntheticGazeSpot().eval()
    images = torch.rand(2, 3, 32, 32)
    boxes = [[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.2, 0.7, 0.8]]]
    original = model({"images": images, "bboxes": boxes})
    extractor = FrozenTaskFeatureExtractor(model)
    extractor.train()
    result = extractor(images, boxes)
    torch.testing.assert_close(result["heatmap"], torch.cat(original["heatmap"]), rtol=0, atol=0)
    torch.testing.assert_close(result["inout"], torch.cat(original["inout"]), rtol=0, atol=0)
    torch.testing.assert_close(model.heatmap_head(result["x"]).squeeze(1), result["heatmap"], rtol=0, atol=0)
    torch.testing.assert_close(model.inout_head(result["z"]).squeeze(-1), result["inout"], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in extractor.modules())
    assert not result["x"].requires_grad and not result["z"].requires_grad


def test_cache_roundtrip_and_person_rows(tmp_path):
    torch.manual_seed(0)
    tracks_path = make_synthetic_tracks(tmp_path)
    checkpoint = tmp_path / "synthetic.pt"
    model = SyntheticGazeSpot()
    torch.save(model.state_dict(), checkpoint)
    output = tmp_path / "cache"
    cache = extract_task_cache(model, transform, tmp_path, tracks_path, checkpoint, output,
                               "train", device="cpu", batch_size=2, num_workers=0)
    assert isinstance(cache, TaskFeatureCache) and len(cache) == 3
    assert cache.index["num_images"] == 2
    assert [track["track_id"] for track in cache.index["tracks"]] == ["train/video/clip/a", "train/video/clip/b"]
    assert [frame["frame_index"] for frame in cache.index["tracks"][0]["frames"]] == [1, 3]
    extractor = FrozenTaskFeatureExtractor(model)
    for track in cache.index["tracks"]:
        for frame in track["frames"]:
            with Image.open(tmp_path / frame["path"]) as image:
                image_tensor = transform(image).unsqueeze(0)
            expected = extractor(image_tensor, [[frame["bbox_norm"]]])
            torch.testing.assert_close(torch.from_numpy(cache.x[frame["row"]].copy()), expected["x"][0])
            torch.testing.assert_close(torch.from_numpy(cache.z[frame["row"]].copy()), expected["z"][0])
    assert cache.x.dtype == cache.z.dtype == np.float32
    for metadata in cache.index["arrays"].values():
        assert sha256(output / metadata["path"]) == metadata["sha256"]
    heads = torch.load(output / "heads.pt", map_location="cpu", weights_only=True)
    assert set(heads) == {"heatmap_head", "inout_head"}
