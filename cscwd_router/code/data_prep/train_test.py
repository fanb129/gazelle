"""Official train/test views; no held-out training records or validation loader."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gazelle.dataloader import GazeDataset, collate_fn
from cscwd_router.code.data_prep.gazefollow import seed_worker
from cscwd_router.code.stag.checkpoint import sha256


def make_datasets(dataset, data_path, transform, *, train_rate=6, test_rate=6):
    name = "gazefollow" if dataset == "gazefollow" else "videoattentiontarget"
    rates = (1, 1) if dataset == "gazefollow" else (train_rate, test_rate)
    train = GazeDataset(name, str(data_path), "train", transform,
                        in_frame_only=dataset == "gazefollow", sample_rate=rates[0],
                        augment=True, return_heatmap=True)
    test = GazeDataset(name, str(data_path), "test", transform,
                       in_frame_only=dataset == "gazefollow", sample_rate=rates[1],
                       augment=False, return_heatmap=False)
    metadata = {"strategy": "official_train_test", "selection_policy": "test_selected",
                "validation_fraction": 0.0, "train_split": "train", "evaluation_split": "test",
                "query_unit": "person", "train_persons": len(train), "test_persons": len(test),
                "train_frame_sample_every": rates[0], "test_frame_sample_every": rates[1]}
    for split in ("train", "test"):
        path = Path(data_path) / f"{split}_preprocessed.json"
        metadata[f"{split}_annotation"] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return train, test, metadata


def make_loader(dataset, batch_size, workers, *, training=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      collate_fn=collate_fn, num_workers=workers, pin_memory=True,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False)


def make_full_test(dataset, data_path, transform):
    name = "gazefollow" if dataset == "gazefollow" else "videoattentiontarget"
    return GazeDataset(name, str(data_path), "test", transform,
                       in_frame_only=dataset == "gazefollow", sample_rate=1,
                       augment=False, return_heatmap=False)


def query_identifier(dataset, index):
    image_index, head_index = dataset.data_idxs[index]
    return {"path": dataset.data[image_index]["path"], "head_index": head_index}
