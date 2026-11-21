"""GazeFollow views and split provenance, using the existing data protocol."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import posixpath
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazelle.data_splits import grouped_holdout_split
from gazelle.dataloader import GazeDataset, GazeFollowImageDataset, collate_fn, collate_gazefollow_images
from cscwd_router.code.data_prep.synthetic_support import make_synthetic_support_batch


@dataclass
class GazeBatch:
    images: torch.Tensor
    bboxes: list
    gazex: list
    gazey: list
    inout: torch.Tensor
    heights: list
    widths: list
    heatmaps: torch.Tensor | None = None

    def to(self, device):
        return GazeBatch(self.images.to(device), self.bboxes, self.gazex, self.gazey, self.inout.to(device),
                         self.heights, self.widths, self.heatmaps.to(device) if self.heatmaps is not None else None)

    def inputs(self):
        return {"images": self.images, "bboxes": self.bboxes}


def collate_people(samples):
    images, boxes, xs, ys, flags, heights, widths, targets = collate_fn(samples)
    return GazeBatch(images, [[box] for box in boxes], xs, ys, flags, heights, widths, targets)


def collate_images(samples):
    return GazeBatch(*collate_gazefollow_images(samples))


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def synthetic_batch(batch_size, image_size, seed):
    fixture = make_synthetic_support_batch(batch_size, image_size, seed=seed)
    maxima = fixture.target_heatmaps.flatten(1).argmax(1)
    xs = [[float(index % 64) / 64] for index in maxima]
    ys = [[float(index // 64) / 64] for index in maxima]
    count = len(xs)
    return GazeBatch(fixture.images, fixture.bboxes, xs, ys, torch.ones(count),
                     [image_size] * count, [image_size] * count, fixture.target_heatmaps)


@dataclass
class ExperimentData:
    train_dataset: object
    validation_dataset: object
    fixture: GazeBatch | None
    metadata: dict
    batch_size: int
    eval_batch_size: int
    workers: int

    def train_loader(self, epoch_seed):
        if self.fixture is not None:
            return [self.fixture]
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_people,
                          num_workers=self.workers, worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(epoch_seed), persistent_workers=False)

    def validation_loader(self):
        if self.fixture is not None:
            return [self.fixture]
        if self.validation_dataset is None:
            return None
        return DataLoader(self.validation_dataset, batch_size=self.eval_batch_size, shuffle=False,
                          collate_fn=collate_images, num_workers=self.workers)


def make_data(
    *, protocol, data_path, transform, batch_size, eval_batch_size, workers,
    seed=3106, split_seed=3106, val_fraction=0.1, image_size=448, augment=True,
):
    if protocol == "synthetic":
        fixture = synthetic_batch(batch_size, image_size, seed)
        metadata = {"protocol": "synthetic", "official_test_accessed": False, "seed": seed,
                    "image_size": image_size, "train_persons": len(fixture.gazex), "train_images": batch_size}
        return ExperimentData(None, None, fixture, metadata, batch_size, eval_batch_size, workers)

    source = Path(data_path) / "train_preprocessed.json"
    records = json.loads(source.read_text())
    keys = [posixpath.normpath(record["path"].strip().replace("\\", "/")) for record in records]
    if protocol == "holdout":
        split = grouped_holdout_split(keys, validation_fraction=val_fraction, seed=split_seed)
        train_indices = split.train_indices
        validation = GazeFollowImageDataset(data_path, "train", transform, records=records, image_indices=split.validation_indices)
        split_info = split.metadata()
    elif protocol == "full_train":
        train_indices, validation = tuple(range(len(records))), None
        assignment = "\n".join(f"{index}\0{key}\0train" for index, key in enumerate(keys))
        split_info = {"assignment_fingerprint": hashlib.sha256(assignment.encode()).hexdigest(), "validation_fraction": 0.0}
    else:
        raise ValueError(f"unknown data protocol: {protocol}")
    train = GazeDataset("gazefollow", data_path, "train", transform, records=records,
                        image_indices=train_indices, augment=augment, return_heatmap=True)
    metadata = {
        **split_info, "protocol": protocol, "official_test_accessed": False,
        "annotation_path": str(source.resolve()), "annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "group_key": "normalized_image_path", "train_records": len(train_indices), "train_persons": len(train),
        "train_images": len({keys[index] for index in train_indices}),
        "validation_images": len(validation) if validation is not None else 0,
        "validation_persons": validation.person_count if validation is not None else 0,
        "training_query_unit": "person", "validation_query_unit": "image_with_all_heads", "augment": augment,
    }
    return ExperimentData(train, validation, None, metadata, batch_size, eval_batch_size, workers)
