"""Frozen causal train replay for output-only state calibration."""

from pathlib import Path

import numpy as np
import torch

from stag.cache import file_sha256, write_json
from stag.temporal_data import iter_track_batches


def cache_frozen_train_scores(model, train_part, output_dir, device,
                              clip_frames=16, batch_size=8, metadata=None):
    """Keep original predicted history and write scores in source-cache order.

    Only the returned score is saved. The model's unmodified heatmap and state
    still drive its next step; no calibration parameter or label is passed to
    the model. State persists across chunks and resets follow the shared
    iterator's person/contiguous-segment boundaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    count = train_part["prediction_index"]["person_frames"]
    scores_path = output_dir / "inout.npy"
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32, shape=(count,))
    seen = np.zeros(count, dtype=bool)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.eval().requires_grad_(False)
    state = None
    batches = 0
    with torch.no_grad():
        for batch in iter_track_batches(train_part["tracks"], train_part["arrays"],
                train_part["feature_index"], train_part["features"], clip_frames, batch_size):
            chunk_scores = []
            for step in range(batch["valid"].shape[1]):
                _, score, state = model.step(
                    batch["features"][:, step].to(device=device, dtype=torch.float32),
                    batch["raw_heatmap"][:, step].to(device),
                    batch["raw_inout"][:, step].to(device),
                    batch["bboxes"][:, step].to(device), state,
                    batch["reset"][:, step].to(device))
                chunk_scores.append(score)
            valid = batch["valid"].numpy()
            positions = batch["global_indices"].numpy()[valid]
            if seen[positions].any() or len(np.unique(positions)) != len(positions):
                raise ValueError("Frozen train replay writes a source frame twice")
            scores[positions] = torch.stack(chunk_scores, dim=1).cpu().numpy()[valid]
            seen[positions] = True
            batches += 1
            if batches % 100 == 0:
                print(f"Frozen train score replay: {int(seen.sum())}/{count} frames", flush=True)
    checks = {
        "all_train_frames_written_once": bool(seen.all()),
        "parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "model_eval": not model.training,
        "parameters_and_buffers_unchanged": all(torch.equal(original[name], value.detach().cpu())
            for name, value in model.state_dict().items()),
        "prediction_arrays_read_only": all(not values.flags.writeable
            for values in train_part["arrays"].values()),
        "features_read_only": not train_part["features"].flags.writeable,
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen train replay verification failed: {checks}")
    scores.flush()
    files = {"inout": {"path": "inout.npy", "shape": [count], "dtype": "float32",
                        "sha256": file_sha256(scores_path)}}
    write_json(output_dir / "index.json", {
        "schema_version": 1, "task": "R4_state_calibration_train_scores",
        "metadata": metadata or {}, "official_split": "train",
        "row_order": "source_prediction_index_global_indices",
        "person_frames": count, "track_count": len(train_part["tracks"]),
        "track_slices": [{key: track[key] for key in ("track_id", "start", "stop")}
                         for track in train_part["tracks"]],
        "files": files, "checks": checks, "passed": True,
        "history": "original_unmodified_model_predictions; no_calibration_feedback",
    })
    print(f"Frozen train score replay completed: {count} frames", flush=True)
    return {"path": str(output_dir.resolve()), "index_sha256": file_sha256(output_dir / "index.json"),
            "files": files, "person_frames": count, "tracks": len(train_part["tracks"]),
            "batches": batches, "checks": checks, "passed": True}
