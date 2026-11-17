"""Complete-track R4 validation with controls from the same causal forward."""

from pathlib import Path

import numpy as np
import torch

from stag.cache import ARRAY_FILES, argmax_points, file_sha256, write_json
from stag.history_diagnostics import HistoryDiagnosticWriter
from stag.history_model import HistoryValidityRefiner
from stag.metrics import TemporalConfig, evaluate_predictions, evaluate_temporal
from stag.state_metrics import evaluate_state_temporal
from stag.temporal_data import iter_track_batches


def _artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def evaluate_compact_model(model, tracks, arrays, feature_index, features, output_dir,
                           source_frames, device, clip_frames, batch_size, metadata):
    """Save every validation frame once, carrying same-person state across chunks.

    History diagnostics use the exact forward that produced each stored gaze
    prediction; ordinary GRU validation creates no synthetic history controls.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    compact_tracks, count = [], 0
    destination = np.full(source_frames, -1, dtype=np.int64)
    for track in tracks:
        stop = count + len(track["frames"])
        compact_tracks.append({**track, "source_global_start": track["start"],
                               "source_global_stop": track["stop"], "start": count, "stop": stop})
        destination[track["start"]:track["stop"]] = np.arange(count, stop)
        count = stop
    shapes = {"heatmap": (count, 64, 64), "inout": (count,), "argmax_xy": (count, 2)}
    predicted = {name: np.lib.format.open_memmap(output_dir / ARRAY_FILES[name], mode="w+",
                    dtype=np.float32, shape=shape) for name, shape in shapes.items()}
    writer = (HistoryDiagnosticWriter(output_dir / "history_diagnostics", tracks,
                                     source_frames, metadata)
              if isinstance(model, HistoryValidityRefiner) else None)
    seen = np.zeros(count, dtype=bool)
    state = None
    model.eval()
    with torch.no_grad():
        for batch in iter_track_batches(tracks, arrays, feature_index, features,
                                         clip_frames, batch_size):
            heatmaps, scores = [], []
            for step in range(batch["valid"].shape[1]):
                inputs = (batch["features"][:, step].to(device=device, dtype=torch.float32),
                          batch["raw_heatmap"][:, step].to(device),
                          batch["raw_inout"][:, step].to(device),
                          batch["bboxes"][:, step].to(device), state,
                          batch["reset"][:, step].to(device))
                if writer is None:
                    heatmap, score, state = model.step(*inputs)
                else:
                    heatmap, score, state, diagnostic = model.step_with_diagnostics(*inputs)
                    writer.write(batch["global_indices"][:, step], batch["valid"][:, step], diagnostic)
                heatmaps.append(heatmap)
                scores.append(score)
            valid = batch["valid"].numpy()
            positions = destination[batch["global_indices"].numpy()[valid]]
            if (np.any(positions < 0) or seen[positions].any()
                    or len(np.unique(positions)) != len(positions)):
                raise ValueError("Validation frame is outside its manifest or written twice")
            maps = torch.stack(heatmaps, dim=1).cpu().numpy()[valid]
            predicted["heatmap"][positions] = maps
            predicted["inout"][positions] = torch.stack(scores, dim=1).cpu().numpy()[valid]
            predicted["argmax_xy"][positions] = argmax_points(maps)
            seen[positions] = True
    if not seen.all():
        raise ValueError("Validation predictions do not cover every manifest frame")
    for values in predicted.values():
        values.flush()
    files = {name: {"path": ARRAY_FILES[name], "shape": list(shapes[name]), "dtype": "float32",
                    "sha256": file_sha256(output_dir / ARRAY_FILES[name])} for name in predicted}
    diagnostics = writer.finalize() if writer is not None else None
    write_json(output_dir / "index.json", {
        "schema_version": 1, "task": "R4_joint_validation", "metadata": metadata,
        "person_frames": count, "track_count": len(compact_tracks), "tracks": compact_tracks,
        "files": files, "diagnostics": diagnostics, "passed": True,
        "checks": {"all_validation_frames_written_once": True},
    })
    config = TemporalConfig()
    temporal = evaluate_temporal(compact_tracks, predicted["argmax_xy"], predicted["inout"], config)
    state_result = evaluate_state_temporal(compact_tracks, predicted["inout"], config)
    details_path = output_dir / "validation_details.json"
    write_json(details_path, {"joint": {key: temporal[key] for key in ("tracks", "events")},
                               "state": {key: state_result[key] for key in ("tracks", "events")}})
    labels = [frame["inout"] for track in compact_tracks for frame in track["frames"]]
    standard = (evaluate_predictions(compact_tracks, predicted["heatmap"], predicted["inout"])
                if any(labels) else {"auc": None, "l2": None, "inout_ap": None,
                                     "person_frames": len(labels), "localization_frames": 0})
    return {"standard": standard, "temporal": temporal["summary"], "state": state_result["summary"],
            "details": _artifact(details_path), "diagnostics": diagnostics,
            "cache": {"path": str(output_dir.resolve()), "files": files,
                      "index_sha256": file_sha256(output_dir / "index.json"), "passed": True}}
