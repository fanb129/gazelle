"""One native-grid frozen feature tensor per unique RGB image.

The spatial cache explicitly stores FP16 features. Original FP32 heatmaps and
in/out scores remain in the R1 prediction cache and are never rewritten here.
Reading this module requires NumPy only, without importing the visual model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import file_sha256, write_json


def unique_image_records(tracks):
    """Collect all person references to an image in deterministic path order."""
    grouped = {}
    for track in tracks:
        if not track["track_id"].startswith("train/"):
            raise ValueError("R2 development features accept official train tracks only")
        for position, frame in enumerate(track["frames"], start=track["start"]):
            grouped.setdefault(frame["path"], []).append({
                "track_id": track["track_id"], "global_index": position,
                "bbox_norm": frame["bbox_norm"], "frame_index": frame["frame_index"],
            })
    return [{"path": path, "people": sorted(people, key=lambda item: item["global_index"])}
            for path, people in sorted(grouped.items())]


def select_probe_records(records, train_track_ids, selected_cases, limit=32, seed=20261117):
    """Use train-only case anchors, small heads, then label-independent hashing."""
    train_track_ids = set(train_track_ids)
    available = [record for record in records
                 if any(person["track_id"] in train_track_ids for person in record["people"])]
    by_path = {record["path"]: record for record in available}
    by_position = {person["global_index"]: record["path"] for record in available
                   for person in record["people"] if person["track_id"] in train_track_ids}
    ordered, reasons = [], {}

    def take(path, reason):
        if path not in reasons:
            ordered.append(path)
            reasons[path] = []
        reasons[path].append(reason)

    for case in selected_cases:
        if case["track_id"] in train_track_ids:
            take(by_position[case["anchor_global_index"]], "R1_training_case_anchor")
    smallest = sorted(available, key=lambda record: (
        min((person["bbox_norm"][2] - person["bbox_norm"][0]) *
            (person["bbox_norm"][3] - person["bbox_norm"][1])
            for person in record["people"] if person["track_id"] in train_track_ids), record["path"]))
    for record in smallest[:8]:
        take(record["path"], "small_training_head")
    for record in sorted(available, key=lambda item: (
            hashlib.sha256(f"{seed}\0{item['path']}".encode()).hexdigest(), item["path"])):
        take(record["path"], "seeded_image_hash")
    chosen = ordered[:limit]
    return [{**by_path[path], "probe_reasons": reasons[path]} for path in chosen]


def feature_contract(feature, layer_index):
    """Describe the actual last-layer tensor, without assuming a grid size."""
    _, channels, height, width = feature.shape
    return {"layer_index": int(layer_index), "channels": int(channels),
            "grid": [int(height), int(width)], "storage_dtype": "float16",
            "source_dtype": str(feature.dtype), "layout": "NCHW",
            "spatial_pooling": "none", "image_unit": "unique_current_RGB",
            "learned_projection_cached": False}


def write_feature_cache(image_paths, batches, cache_dir, contract, sources):
    """Stream FP32 tensors to FP16 NPY and verify the quantized values exactly."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    shape = (len(image_paths), contract["channels"], *contract["grid"])
    path = cache_dir / "scene_features.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    live = hashlib.sha256()
    start = 0
    for batch in batches:
        if tuple(batch.shape[1:]) != tuple(shape[1:]):
            raise ValueError("Backbone feature shape differs from the accepted probe contract")
        converted = np.asarray(batch, dtype=np.float16)
        if not np.isfinite(converted).all():
            raise ValueError("Frozen features are non-finite or overflow FP16 storage")
        stop = start + len(converted)
        if stop > len(image_paths):
            raise ValueError("Feature stream contains more images than its manifest")
        values[start:stop] = converted
        live.update(converted.tobytes(order="C"))
        start = stop
    if start != len(image_paths):
        raise ValueError(f"Incomplete feature cache: {start} of {len(image_paths)} unique images")
    values.flush()
    del values
    print("Verifying stored FP16 feature values", flush=True)
    restored = np.load(path, mmap_mode="r", allow_pickle=False)
    disk = hashlib.sha256()
    for start in range(0, len(image_paths), 8):
        disk.update(restored[start:start + 8].tobytes(order="C"))
    if live.digest() != disk.digest():
        raise ValueError("FP16 feature cache readback differs from the quantized stream")
    print("Hashing completed scene feature file", flush=True)
    files = {"scene_features": {"path": path.name, "shape": list(shape), "dtype": "float16",
                                 "sha256": file_sha256(path), "values_sha256": disk.hexdigest(),
                                 "size_bytes": path.stat().st_size}}
    index = {"schema_version": 1, "task": "R2", "image_paths": list(image_paths),
             "contract": contract, "files": files, "sources": sources,
             "checks": {"all_unique_images_cached": True, "fp16_exact_readback": True},
             "passed": True}
    write_json(cache_dir / "index.json", index)
    return {"path": str(cache_dir.resolve()), "index_sha256": file_sha256(cache_dir / "index.json"),
            "images": len(image_paths), "contract": contract, "files": files,
            "checks": index["checks"], "passed": True}


def load_feature_cache(cache_dir):
    """Read a completed index and native-grid FP16 features without Torch."""
    cache_dir = Path(cache_dir)
    index = json.loads((cache_dir / "index.json").read_text())
    item = index["files"]["scene_features"]
    features = np.load(cache_dir / item["path"], mmap_mode="r", allow_pickle=False)
    if list(features.shape) != item["shape"] or str(features.dtype) != item["dtype"]:
        raise ValueError("Feature NPY header differs from its completed index")
    return index, features
