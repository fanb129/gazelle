"""Frame-grouped VideoAttentionTarget data loading for COTB."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image

from AAAICOTB.annotations import load_sequences, sequence_id
from gazelle.utils import get_heatmap


def _jitter_bbox(bbox: Sequence[float], scale: float) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    width, height = x2 - x1, y2 - y1
    values = [
        x1 + random.uniform(-scale, scale) * width,
        y1 + random.uniform(-scale, scale) * height,
        x2 + random.uniform(-scale, scale) * width,
        y2 + random.uniform(-scale, scale) * height,
    ]
    values[0], values[2] = sorted((max(0.0, min(1.0, values[0])), max(0.0, min(1.0, values[2]))))
    values[1], values[3] = sorted((max(0.0, min(1.0, values[1])), max(0.0, min(1.0, values[3]))))
    # Preserve a non-empty head prompt even under extreme jitter near a border.
    if values[2] - values[0] < 1e-4:
        values[0] = min(values[0], 1.0 - 1e-4)
        values[2] = values[0] + 1e-4
    if values[3] - values[1] < 1e-4:
        values[1] = min(values[1], 1.0 - 1e-4)
        values[3] = values[1] + 1e-4
    return values


class VATFrameDataset(torch.utils.data.Dataset):
    """Return one scene and all annotated observer queries in that scene."""

    def __init__(
        self,
        data_path: str | Path,
        annotation_path: str | Path,
        transform,
        sequence_indices: Iterable[int] | None = None,
        frame_sample_every: int = 1,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
        bbox_jitter_probability: float = 0.5,
        bbox_jitter_scale: float = 0.2,
    ) -> None:
        if frame_sample_every < 1:
            raise ValueError("frame_sample_every must be >= 1")
        self.data_path = Path(data_path)
        self.annotation_path = Path(annotation_path)
        self.transform = transform
        self.sequences = load_sequences(annotation_path)
        allowed = set(range(len(self.sequences))) if sequence_indices is None else set(sequence_indices)
        self.augment = augment
        self.horizontal_flip_probability = horizontal_flip_probability
        self.bbox_jitter_probability = bbox_jitter_probability
        self.bbox_jitter_scale = bbox_jitter_scale
        self.items: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(self.sequences):
            if sequence_index not in allowed:
                continue
            frames = sequence.get("frames", [])
            for frame_index in range(0, len(frames), frame_sample_every):
                if frames[frame_index].get("heads"):
                    self.items.append((sequence_index, frame_index))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        sequence_index, frame_index = self.items[index]
        sequence = self.sequences[sequence_index]
        frame = sequence["frames"][frame_index]
        heads = copy.deepcopy(frame.get("heads", []))
        image = Image.open(self.data_path / frame["path"]).convert("RGB")

        if self.augment and random.random() < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for head in heads:
                x1, y1, x2, y2 = map(float, head["bbox_norm"])
                head["bbox_norm"] = [1.0 - x2, y1, 1.0 - x1, y2]
                if int(head.get("inout", 0)) == 1:
                    head["gazex_norm"] = [1.0 - float(value) for value in head["gazex_norm"]]

        # Pair eligibility must use the unjittered annotation boxes; otherwise
        # augmentation could make a duplicate track pass the duplicate-IoU gate.
        binding_bboxes = [list(map(float, head["bbox_norm"])) for head in heads]
        if self.augment and self.bbox_jitter_scale > 0:
            for head in heads:
                if random.random() < self.bbox_jitter_probability:
                    head["bbox_norm"] = _jitter_bbox(head["bbox_norm"], self.bbox_jitter_scale)

        bboxes = [list(map(float, head["bbox_norm"])) for head in heads]
        inout = torch.tensor([int(head.get("inout", 0)) for head in heads], dtype=torch.bool)
        targets = []
        heatmaps = []
        for head, is_in in zip(heads, inout.tolist()):
            if is_in:
                x = float(head["gazex_norm"][0])
                y = float(head["gazey_norm"][0])
            else:
                x = y = -1.0
            targets.append((x, y))
            heatmaps.append(get_heatmap(x, y, 64, 64))

        return {
            "image": self.transform(image),
            "bboxes": bboxes,
            "binding_bboxes": binding_bboxes,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "heatmaps": torch.stack(heatmaps),
            "inout": inout,
            "path": frame["path"],
            "sequence_id": sequence_id(sequence, frame),
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "height": int(sequence.get("height", 0)),
            "width": int(sequence.get("width", 0)),
        }


def collate_frames(batch: Sequence[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "bboxes": [item["bboxes"] for item in batch],
        "binding_bboxes": [item["binding_bboxes"] for item in batch],
        "targets": [item["targets"] for item in batch],
        "heatmaps": [item["heatmaps"] for item in batch],
        "inout": [item["inout"] for item in batch],
        "paths": [item["path"] for item in batch],
        "sequence_ids": [item["sequence_id"] for item in batch],
        "sequence_indices": [item["sequence_index"] for item in batch],
        "frame_indices": [item["frame_index"] for item in batch],
        "heights": [item["height"] for item in batch],
        "widths": [item["width"] for item in batch],
    }
