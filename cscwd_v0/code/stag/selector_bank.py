"""Stream fixed selector inputs to verified, read-only per-frame NPY banks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import argmax_points, file_sha256, write_json


VERSION = "r4_utility_selector_v1"
FILES = {name: name + ".npy" for name in (
    "heatmaps", "observations", "inout", "candidate_xy", "history_available")}


def _schema(person_frames, channels):
    return {
        "heatmaps": ((person_frames, 5, 64, 64), np.float32),
        "observations": ((person_frames, 6 * channels + 17), np.float32),
        "inout": ((person_frames,), np.float32),
        "candidate_xy": ((person_frames, 5, 2), np.float32),
        "history_available": ((person_frames,), np.bool_),
    }


def build_selector_bank(part, output_dir, sources, metadata, device="cuda:0"):
    """Generate from observations only, resetting memory at every track start.

    The already verified official split supplies global row order. Candidate
    generation never reads gaze labels; labels are retained only in the index
    for subsequent training supervision and offline evaluation.
    """
    import torch

    from .utility_selector import BANK_CANDIDATES, step_selector_inputs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    count = part["prediction_index"]["person_frames"]
    channels = part["feature_index"]["contract"]["channels"]
    schema = _schema(count, channels)
    arrays = {name: np.lib.format.open_memmap(output_dir / FILES[name], mode="w+",
        dtype=dtype, shape=shape) for name, (shape, dtype) in schema.items()}
    live = {name: hashlib.sha256() for name in arrays}
    feature_rows = {path: row for row, path in enumerate(part["feature_index"]["image_paths"])}
    written = 0
    with torch.no_grad():
        for number, track in enumerate(part["tracks"], start=1):
            state = None
            for position, frame in enumerate(track["frames"], start=track["start"]):
                heatmap = torch.tensor(np.array(part["arrays"]["heatmap"][position]),
                    dtype=torch.float32, device=device)
                feature = torch.tensor(np.array(part["features"][feature_rows[frame["path"]]]),
                    dtype=torch.float32, device=device)
                score = torch.tensor(part["arrays"]["inout"][position].item(),
                    dtype=torch.float32, device=device)
                bbox = torch.tensor(frame["bbox_norm"], dtype=torch.float32, device=device)
                maps, observation, available, state = step_selector_inputs(
                    heatmap, feature, score, bbox, frame["frame_index"], state)
                maps = torch.stack([maps[name] for name in BANK_CANDIDATES]).cpu().numpy()
                values = {"heatmaps": maps,
                    "observations": observation.cpu().numpy(),
                    "inout": np.asarray(score.cpu().item(), dtype=np.float32),
                    "candidate_xy": argmax_points(maps),
                    "history_available": np.asarray(available, dtype=np.bool_)}
                for name, value in values.items():
                    arrays[name][position] = value
                    live[name].update(value.tobytes(order="C"))
                written += 1
            print(f"Selector bank {metadata['split']} track {number}/{len(part['tracks'])}; "
                  f"frames {written}/{count}", flush=True)
    for array in arrays.values():
        array.flush()
    del arrays

    restored = {name: np.load(output_dir / filename, mmap_mode="r", allow_pickle=False)
                for name, filename in FILES.items()}
    disk = {name: hashlib.sha256() for name in restored}
    finite, raw_maps, raw_scores, raw_points, reconstructed = True, True, True, True, True
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        for name, values in restored.items():
            chunk = values[start:stop]
            disk[name].update(chunk.tobytes(order="C"))
            finite &= bool(np.isfinite(chunk).all())
        maps = restored["heatmaps"][start:stop]
        raw_maps &= np.array_equal(maps[:, 0], part["arrays"]["heatmap"][start:stop])
        raw_scores &= np.array_equal(restored["inout"][start:stop], part["arrays"]["inout"][start:stop])
        raw_points &= np.array_equal(restored["candidate_xy"][start:stop, 0],
                                     part["arrays"]["argmax_xy"][start:stop])
        points = argmax_points(maps.reshape(-1, 64, 64)).reshape(-1, 5, 2)
        reconstructed &= np.array_equal(points, restored["candidate_xy"][start:stop])
    checks = {
        "all_official_frames_written": written == count,
        "all_values_finite": bool(finite),
        "raw_current_heatmap_exact": bool(raw_maps),
        "raw_current_inout_exact": bool(raw_scores),
        "raw_current_coordinates_exact": bool(raw_points),
        "all_candidate_coordinates_reconstructed": bool(reconstructed),
        **{name + "_exact_readback": live[name].digest() == disk[name].digest()
           for name in restored},
    }
    if not all(checks.values()):
        raise ValueError(f"Selector bank verification failed: {checks}")
    files = {name: {"path": FILES[name], "shape": list(shape), "dtype": str(np.dtype(dtype)),
        "sha256": file_sha256(output_dir / FILES[name]),
        "values_sha256": disk[name].hexdigest(),
        "size_bytes": (output_dir / FILES[name]).stat().st_size}
        for name, (shape, dtype) in schema.items()}
    coverage = {"tracks": len(part["tracks"]), "person_frames": count,
                "unique_image_frames": len(part["feature_index"]["image_paths"])}
    index = {"task": "R4_selector_bank", "schema_version": 1, "version": VERSION,
        "tracks": part["tracks"], "feature_channels": channels,
        "candidate_names": list(BANK_CANDIDATES), "sources": sources,
        "metadata": metadata, "coverage": coverage, "files": files,
        "checks": checks, "passed": True}
    write_json(output_dir / "index.json", index)
    return {"path": str(output_dir.resolve()), "index_sha256": file_sha256(output_dir / "index.json"),
            "files": files, "passed": True, "coverage": coverage}


def load_selector_bank(record, verify=True):
    """Open read-only arrays; verification hashes files with bounded memory."""
    directory = Path(record["path"])
    index_path = directory / "index.json"
    if verify and file_sha256(index_path) != record["index_sha256"]:
        raise ValueError("Selector bank index SHA256 mismatch")
    index = json.loads(index_path.read_text())
    if (not record["passed"] or not index["passed"] or index["version"] != VERSION
            or index["coverage"] != record["coverage"] or index["files"] != record["files"]):
        raise ValueError("Selector bank index differs from its completed record")
    schema = _schema(index["coverage"]["person_frames"], index["feature_channels"])
    arrays = {}
    for name, (shape, dtype) in schema.items():
        item = index["files"][name]
        path = directory / item["path"]
        if verify and (file_sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]):
            raise ValueError(f"Selector bank file SHA256 or size mismatch: {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (list(values.shape) != item["shape"] or str(values.dtype) != item["dtype"]
                or values.shape != shape or values.dtype != np.dtype(dtype)):
            raise ValueError(f"Selector bank NPY header differs from its contract: {name}")
        arrays[name] = values
    return {"index": index, **arrays}
