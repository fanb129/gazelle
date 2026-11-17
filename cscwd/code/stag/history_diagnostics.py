"""Compact per-frame R3 controls aligned with complete prediction tracks."""

from pathlib import Path

import numpy as np

from stag.cache import file_sha256, write_json


CONTROL_COLUMNS = (
    "history_validity", "refresh", "write_strength", "match_peak",
    "match_entropy", "agreement", "person_change", "history_available",
)


def _numpy(value):
    return value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()


class HistoryDiagnosticWriter:
    """Store valid person-frame diagnostics once, in the supplied track order.

    ``write`` receives one recurrent step: source indices/valid flags [B],
    diagnostic scalars [B] and correspondence coordinates [B,2]. Predictions
    and controls may still be GPU tensors; no computation graph is retained.
    """

    def __init__(self, output_dir, tracks, source_frames, metadata):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.source_frames = source_frames
        self.metadata = metadata
        self.destination = np.full(source_frames, -1, dtype=np.int64)
        self.tracks = []
        count = 0
        for track in tracks:
            stop = count + len(track["frames"])
            self.destination[track["start"]:track["stop"]] = np.arange(count, stop)
            self.tracks.append({
                "track_id": track["track_id"], "start": count, "stop": stop,
                "source_global_start": track["start"], "source_global_stop": track["stop"],
                "frame_indices": [frame["frame_index"] for frame in track["frames"]],
            })
            count = stop
        self.person_frames = count
        self.seen = np.zeros(count, dtype=bool)
        specifications = {
            "controls": (np.float32, (count, len(CONTROL_COLUMNS))),
            "correspondence_xy": (np.float32, (count, 2)),
            "source_global_indices": (np.int64, (count,)),
        }
        self.arrays = {name: np.lib.format.open_memmap(
            self.output_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            for name, (dtype, shape) in specifications.items()}
        for track in self.tracks:
            self.arrays["source_global_indices"][track["start"]:track["stop"]] = np.arange(
                track["source_global_start"], track["source_global_stop"])

    def write(self, global_indices, valid, diagnostics):
        valid = _numpy(valid).astype(bool, copy=False)
        source = _numpy(global_indices)[valid]
        if np.any((source < 0) | (source >= self.source_frames)):
            raise ValueError("Diagnostic source frame is outside the input cache")
        positions = self.destination[source]
        if np.any(positions < 0):
            raise ValueError("Diagnostic source frame is outside the supplied tracks")
        if self.seen[positions].any() or len(np.unique(positions)) != len(positions):
            raise ValueError("Diagnostic source frame written twice")
        controls = np.stack([_numpy(diagnostics[name]) for name in CONTROL_COLUMNS], axis=-1)
        self.arrays["controls"][positions] = controls[valid]
        self.arrays["correspondence_xy"][positions] = _numpy(diagnostics["correspondence_xy"])[valid]
        self.seen[positions] = True

    def finalize(self):
        if not self.seen.all():
            raise ValueError("Diagnostic cache is missing source frames")
        files = {}
        for name, values in self.arrays.items():
            values.flush()
            path = self.output_dir / f"{name}.npy"
            files[name] = {"path": path.name, "shape": list(values.shape),
                           "dtype": str(values.dtype), "sha256": file_sha256(path),
                           "size_bytes": path.stat().st_size}
        index_path = self.output_dir / "index.json"
        write_json(index_path, {
            "schema_version": 1, "task": "R3_history_diagnostics", "metadata": self.metadata,
            "source_person_frames": self.source_frames, "person_frames": self.person_frames,
            "track_count": len(self.tracks), "tracks": self.tracks,
            "control_columns": list(CONTROL_COLUMNS), "files": files,
            "checks": {"all_person_frames_written_once": True}, "passed": True,
        })
        return {"path": str(self.output_dir.resolve()), "person_frames": self.person_frames,
                "index_sha256": file_sha256(index_path), "files": files, "passed": True}
