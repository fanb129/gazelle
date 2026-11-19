"""Lossless whole-candidate evaluation with unchanged raw in/out scores."""

import hashlib
from pathlib import Path

import numpy as np

from .cache import ARRAY_FILES, argmax_points, file_sha256, load_cache, write_json
from .metrics import TemporalConfig, evaluate_predictions, evaluate_temporal
from .state_metrics import evaluate_state_temporal


def _artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def evaluate_selector_choices(bank, choices, output_dir, metadata, logits=None):
    """Save exact selected maps and score them using the existing VAT metrics.

    ``choices`` contains bank candidate column numbers for every source row.
    Optional logits come from the same forward that determined those choices;
    this function performs no model inference. All output arrays are FP32,
    while choice/logit artifacts preserve the supplied values and dtypes.
    """
    choices = np.asarray(choices)
    maps, scores = bank["heatmaps"], bank["inout"]
    count, candidate_count = maps.shape[:2]
    if (choices.shape != (count,) or not np.issubdtype(choices.dtype, np.integer)
            or np.any(choices < 0) or np.any(choices >= candidate_count)):
        raise ValueError("Selector choices must be one valid integer candidate column per frame")
    if maps.dtype != np.float32 or scores.dtype != np.float32:
        raise ValueError("Selector evaluation requires lossless float32 maps and in/out scores")
    if logits is not None:
        logits = np.asarray(logits)
        if (logits.shape != (count,) or not np.issubdtype(logits.dtype, np.floating)
                or not np.isfinite(logits).all()):
            raise ValueError("Selector logits must be one finite floating-point value per frame")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shapes = {"heatmap": (count, 64, 64), "inout": (count,), "argmax_xy": (count, 2)}
    arrays = {name: np.lib.format.open_memmap(output_dir / ARRAY_FILES[name], mode="w+",
                                             dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    live = {name: hashlib.sha256() for name in arrays}
    selected_coordinates_match = True
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        positions = np.arange(start, stop)
        selected = maps[positions, choices[start:stop]]
        if not np.isfinite(selected).all() or not np.isfinite(scores[start:stop]).all():
            raise ValueError("Selected candidate predictions contain non-finite values")
        batch = {"heatmap": selected, "inout": scores[start:stop],
                 "argmax_xy": argmax_points(selected)}
        selected_coordinates_match &= np.array_equal(
            batch["argmax_xy"], bank["candidate_xy"][positions, choices[start:stop]])
        for name, values in batch.items():
            arrays[name][start:stop] = values
            live[name].update(values.tobytes(order="C"))
    for values in arrays.values():
        values.flush()
    del arrays
    restored = {name: np.load(output_dir / filename, mmap_mode="r", allow_pickle=False)
                for name, filename in ARRAY_FILES.items()}
    disk = {name: hashlib.sha256() for name in restored}
    argmax_matches = True
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        for name, values in restored.items():
            disk[name].update(values[start:stop].tobytes(order="C"))
        argmax_matches &= np.array_equal(argmax_points(restored["heatmap"][start:stop]),
                                        restored["argmax_xy"][start:stop])
    choices_path = output_dir / "choices.npy"
    np.save(choices_path, choices, allow_pickle=False)
    logits_path = output_dir / "selector_logits.npy"
    if logits is not None:
        np.save(logits_path, logits, allow_pickle=False)
    tracks = bank["index"]["tracks"]
    checks = {
        "all_person_frames_written_once": len(restored["heatmap"]) == count,
        "all_tracks_cover_original_rows": tracks[0]["start"] == 0 and tracks[-1]["stop"] == count,
        **{f"{name}_exact_readback": live[name].digest() == disk[name].digest()
           for name in restored},
        "argmax_reconstructed_from_heatmap": bool(argmax_matches),
        "selected_coordinates_match_bank": bool(selected_coordinates_match),
        "raw_inout_preserved_exactly": np.array_equal(restored["inout"], scores),
        "choices_preserved_exactly": np.array_equal(np.load(choices_path, allow_pickle=False), choices),
        "provided_logits_preserved_exactly": logits is None or np.array_equal(
            np.load(logits_path, allow_pickle=False), logits),
    }
    if not all(checks.values()):
        raise ValueError(f"Selector evaluation verification failed: {checks}")
    names = bank["index"]["candidate_names"]
    selection = {"choices": _artifact(choices_path),
                 "logits": _artifact(logits_path) if logits is not None else None,
                 "choice_counts": {name: int(np.sum(choices == column))
                                   for column, name in enumerate(names)}}
    files = {name: {"path": filename, "shape": list(shapes[name]), "dtype": "float32",
                    "sha256": file_sha256(output_dir / filename),
                    "values_sha256": disk[name].hexdigest(),
                    "size_bytes": (output_dir / filename).stat().st_size}
             for name, filename in ARRAY_FILES.items()}
    split = metadata.get("evaluation_split", "test")
    write_json(output_dir / "index.json", {
        "schema_version": 1, "task": "R4_selector_evaluation", "evaluation_split": split,
        "metadata": metadata, "person_frames": count, "track_count": len(tracks),
        "tracks": tracks, "files": files, "selection": selection, "diagnostics": None,
        "layout": "original_source_track_global_row_order",
        "argmax_convention": "xy; x=column/64, y=row/64; first_flat_index_on_ties",
        "checks": checks, "passed": True})
    # Use the original loader to score exactly the published prediction cache.
    _, predicted = load_cache(output_dir)
    config = TemporalConfig()
    temporal = evaluate_temporal(tracks, predicted["argmax_xy"], predicted["inout"], config)
    state = evaluate_state_temporal(tracks, predicted["inout"], config)
    details_path = output_dir / "evaluation_details.json"
    write_json(details_path, {"joint": {key: temporal[key] for key in ("tracks", "events")},
                              "state": {key: state[key] for key in ("tracks", "events")}})
    return {"evaluation_split": split,
            "standard": evaluate_predictions(tracks, predicted["heatmap"], predicted["inout"]),
            "temporal": temporal["summary"], "state": state["summary"],
            "details": _artifact(details_path), "selection": selection, "diagnostics": None,
            "cache": {"path": str(output_dir.resolve()),
                      "index_sha256": file_sha256(output_dir / "index.json"),
                      "files": files, "checks": checks, "passed": True,
                      "tracks": len(tracks), "person_frames": count,
                      "unique_image_frames": len({frame["path"] for track in tracks for frame in track["frames"]})}}
