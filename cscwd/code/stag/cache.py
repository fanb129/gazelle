"""Lossless float32 single-frame predictions, stored in contiguous track order.

The server-only index contains each track's [start, stop) slice and original
frame records. Reading a track requires NumPy, without importing the model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_FILES = {"heatmap": "heatmap.npy", "inout": "inout.npy", "argmax_xy": "argmax_xy.npy"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")


def argmax_points(heatmaps):
    """Return normalized (x, y), matching gazelle.utils.vat_l2's /64 rule."""
    height, width = heatmaps.shape[-2:]
    indices = heatmaps.reshape(len(heatmaps), -1).argmax(axis=1)
    return np.column_stack((indices % width / width, indices // width / height)).astype(np.float32)


def index_tracks(sequences):
    tracks = []
    start = 0
    for sequence in sequences:
        for track in sequence["tracks"]:
            stop = start + len(track["frames"])
            tracks.append({
                **track, "sequence_path": sequence["path"],
                "width": sequence["width"], "height": sequence["height"],
                "start": start, "stop": stop,
            })
            start = stop
    return tracks, start


def load_cache(cache_dir):
    """Load the completed index and read-only arrays, without model inference."""
    cache_dir = Path(cache_dir)
    with (cache_dir / "index.json").open() as stream:
        index = json.load(stream)
    arrays = {name: np.load(cache_dir / filename, mmap_mode="r", allow_pickle=False)
              for name, filename in ARRAY_FILES.items()}
    return index, arrays


def iter_cached_tracks(index, arrays):
    """Yield (track metadata, prediction views) in the native track order."""
    for track in index["tracks"]:
        selection = slice(track["start"], track["stop"])
        yield track, {name: values[selection] for name, values in arrays.items()}


def write_prediction_cache(sequences, predictions, cache_dir, metadata):
    """Write streamed batches and verify exact recovery from reopened NPY files.

    predictions yields (heatmap[B,64,64], inout[B]) in sequence/track/frame order.
    The index is published only after full coverage and lossless readback pass.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    tracks, expected_frames = index_tracks(sequences)
    shapes = {"heatmap": (expected_frames, 64, 64), "inout": (expected_frames,),
              "argmax_xy": (expected_frames, 2)}
    arrays = {name: np.lib.format.open_memmap(cache_dir / ARRAY_FILES[name], mode="w+",
                                             dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    live_digests = {name: hashlib.sha256() for name in arrays}
    start = 0
    for heatmaps, inout in predictions:
        if heatmaps.dtype != np.float32 or inout.dtype != np.float32:
            raise ValueError("Predictions must be float32; lossy cache conversion is disabled")
        if heatmaps.shape[1:] != (64, 64) or inout.shape != (len(heatmaps),):
            raise ValueError("Expected heatmap[B,64,64] and inout[B]")
        if not np.isfinite(heatmaps).all() or not np.isfinite(inout).all():
            raise ValueError("Model predictions contain non-finite values")
        stop = start + len(heatmaps)
        batch = {"heatmap": heatmaps, "inout": inout, "argmax_xy": argmax_points(heatmaps)}
        for name, values in batch.items():
            arrays[name][start:stop] = values
            live_digests[name].update(values.tobytes(order="C"))
        start = stop
    written_frames = start
    if written_frames != expected_frames:
        raise ValueError(f"Incomplete cache: received {written_frames} of {expected_frames} person frames")
    for values in arrays.values():
        values.flush()
    del arrays

    # Hash raw values with a different chunking than inference, after reopening.
    disk_arrays = {name: np.load(cache_dir / filename, mmap_mode="r", allow_pickle=False)
                   for name, filename in ARRAY_FILES.items()}
    disk_digests = {name: hashlib.sha256() for name in disk_arrays}
    points_match = True
    for start in range(0, expected_frames, 128):
        stop = min(start + 128, expected_frames)
        for name, values in disk_arrays.items():
            disk_digests[name].update(values[start:stop].tobytes(order="C"))
        points_match &= np.array_equal(argmax_points(disk_arrays["heatmap"][start:stop]),
                                       disk_arrays["argmax_xy"][start:stop])
    checks = {
        "all_person_frames_cached": written_frames == expected_frames,
        "all_tracks_covered": tracks[-1]["stop"] == expected_frames,
        **{f"{name}_exact_readback": live_digests[name].digest() == disk_digests[name].digest()
           for name in disk_arrays},
        "argmax_reconstructed_from_heatmap": bool(points_match),
    }
    if not all(checks.values()):
        raise ValueError(f"Cache readback verification failed: {checks}")
    files = {name: {"path": filename, "shape": list(shapes[name]), "dtype": "float32",
                    "sha256": file_sha256(cache_dir / filename),
                    "values_sha256": disk_digests[name].hexdigest()}
             for name, filename in ARRAY_FILES.items()}
    index = {
        "schema_version": 1, "task": "T2", "metadata": metadata,
        "person_frames": expected_frames, "track_count": len(tracks),
        "layout": "contiguous_sequence_track_native_frame_order",
        "argmax_convention": "xy; x=column/64, y=row/64; first_flat_index_on_ties",
        "files": files, "tracks": tracks, "checks": checks, "passed": True,
    }
    write_json(cache_dir / "index.json", index)
    return {
        "path": str(cache_dir.resolve()), "index_sha256": file_sha256(cache_dir / "index.json"),
        "files": files, "tracks": len(tracks), "person_frames": expected_frames,
        "unique_image_frames": len({frame["path"] for track in tracks for frame in track["frames"]}),
        "checks": checks, "passed": True,
    }
