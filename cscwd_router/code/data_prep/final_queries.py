"""Real VAT frames with deterministic nested GT-person queries."""

import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from gazelle.dataloader import load_data_vat
from cscwd_router.code.stag.checkpoint import sha256


def selected_heads(record, seed, maximum=3):
    indices = range(len(record["heads"]))
    return sorted(indices, key=lambda index: hashlib.sha256(
        f"{seed}\0{record['path']}\0{index}".encode()).hexdigest())[:maximum]


def cohort(data_path, seed=0, split="test", minimum_heads=3):
    annotation = Path(data_path) / f"{split}_preprocessed.json"
    records = [r for r in load_data_vat(annotation, sample_rate=1) if len(r["heads"]) >= minimum_heads]
    entries = [{"path": r["path"], "head_indices": selected_heads(r, seed)} for r in records]
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("multi-person evaluation requires one GT record per real image path")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return records, {"annotation": {"path": str(annotation.resolve()), "sha256": sha256(annotation)},
        "split": split, "seed": seed, "minimum_heads": minimum_heads, "frames": len(records),
        "selection": "nested_GT_queries_by_seed_path_head_index", "selection_sha256": digest, "entries": entries}


class VATQueryImages(Dataset):
    def __init__(self, data_path, records, manifest, count, transform):
        self.path, self.data, self.transform = Path(data_path), records, transform
        self.count = count
        self.head_indices = [entry["head_indices"][:count] for entry in manifest["entries"]]
        self.data_idxs = [(i, h) for i, indices in enumerate(self.head_indices) for h in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        heads = [record["heads"][h] for h in self.head_indices[index]]
        with Image.open(self.path / record["path"]) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)
        return (tensor, [copy.deepcopy(h["bbox_norm"]) for h in heads],
                [h["gazex_norm"] for h in heads], [h["gazey_norm"] for h in heads],
                torch.tensor([h["inout"] for h in heads]), [height] * len(heads), [width] * len(heads))
