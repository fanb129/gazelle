"""Versioned, per-person prediction-cache schema for the pilot.

Nested arrays are kept as regular Python lists in memory.  For Parquet output,
``probe_heatmaps`` is JSON encoded because Arrow's inferred struct fields cannot
reliably represent arbitrary layer keys across files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CACHE_SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "sample_id",
    "dataset",
    "split",
    "image_path",
    "person_index",
    "final_heatmap",
    "probe_heatmaps",
    "gt_gaze",
    "gt_inout",
    "bbox",
    "checkpoint_hash",
)


def checkpoint_sha256(path: str | Path | None) -> str | None:
    """Return a checkpoint content hash, or ``None`` when no path is supplied."""

    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sample_id(dataset: str, image_path: str, person_index: int) -> str:
    """Build a stable, filesystem-safe ID without exposing absolute data paths."""

    raw = f"{dataset}\0{image_path}\0{int(person_index)}".encode("utf-8")
    return f"{dataset}-{hashlib.sha1(raw).hexdigest()[:20]}"


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _tolist(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tolist(item) for item in value]
    return value


def _heatmap(value: Any, field: str) -> list[list[float]]:
    array = np.asarray(_tolist(value), dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"{field} must be a 2-D heatmap, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array.tolist()


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and convert one cache record into JSON/Arrow-safe values."""

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"prediction record is missing required fields: {missing}")

    result = {str(key): _tolist(value) for key, value in record.items()}
    result["schema_version"] = CACHE_SCHEMA_VERSION
    result["sample_id"] = str(result["sample_id"])
    result["dataset"] = str(result["dataset"])
    result["split"] = str(result["split"])
    result["image_path"] = str(result["image_path"])
    result["person_index"] = int(result["person_index"])
    result["final_heatmap"] = _heatmap(result["final_heatmap"], "final_heatmap")

    probes = result["probe_heatmaps"]
    if not isinstance(probes, Mapping) or len(probes) < 2:
        raise ValueError("probe_heatmaps must map at least two layer IDs to heatmaps")
    result["probe_heatmaps"] = {
        str(layer): _heatmap(heatmap, f"probe_heatmaps[{layer}]")
        for layer, heatmap in probes.items()
    }

    bbox = np.asarray(result["bbox"], dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("bbox must contain four finite normalized coordinates")
    result["bbox"] = bbox.tolist()

    gaze = np.asarray(result["gt_gaze"], dtype=np.float64)
    if gaze.size == 0:
        gaze = gaze.reshape(0, 2)
    elif gaze.ndim == 1 and gaze.shape == (2,):
        gaze = gaze.reshape(1, 2)
    if gaze.ndim != 2 or gaze.shape[1] != 2 or not np.isfinite(gaze).all():
        raise ValueError("gt_gaze must have shape [num_annotations, 2]")
    result["gt_gaze"] = gaze.tolist()
    result["gt_inout"] = int(result["gt_inout"])
    if result["gt_inout"] not in (0, 1):
        raise ValueError("gt_inout must be 0 or 1")
    result["checkpoint_hash"] = (
        None if result["checkpoint_hash"] is None else str(result["checkpoint_hash"])
    )
    if "inout_probability" in result and result["inout_probability"] is not None:
        result["inout_probability"] = float(result["inout_probability"])
    return result


def validate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_record(record) for record in records]
    ids = [record["sample_id"] for record in normalized]
    if len(set(ids)) != len(ids):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate sample_id values: {duplicates[:5]}")
    if not normalized:
        raise ValueError("prediction cache cannot be empty")
    return normalized


def save_prediction_cache(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Save records to ``.parquet``, ``.json``, ``.jsonl``, or ``.csv``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_records(records)
    suffix = destination.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with destination.open("w", encoding="utf-8") as handle:
            if suffix == ".json":
                json.dump(normalized, handle, indent=2, sort_keys=True)
                handle.write("\n")
            else:
                for record in normalized:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        return destination

    import pandas as pd

    rows = []
    nested = {"final_heatmap", "probe_heatmaps", "gt_gaze", "bbox"}
    for record in normalized:
        rows.append(
            {
                key: json.dumps(value, separators=(",", ":")) if key in nested else value
                for key, value in record.items()
            }
        )
    frame = pd.DataFrame(rows)
    if suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    elif suffix == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError(f"unsupported prediction-cache extension: {suffix}")
    return destination


def load_prediction_cache(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    else:
        import pandas as pd

        if suffix == ".parquet":
            frame = pd.read_parquet(source)
        elif suffix == ".csv":
            frame = pd.read_csv(source)
        else:
            raise ValueError(f"unsupported prediction-cache extension: {suffix}")
        rows = frame.to_dict(orient="records")
        for row in rows:
            for field in ("final_heatmap", "probe_heatmaps", "gt_gaze", "bbox"):
                row[field] = json.loads(row[field])
    return validate_records(rows)


def heatmap_peak(heatmap: Any) -> tuple[float, float]:
    array = np.asarray(heatmap, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("heatmap must be a finite 2-D array")
    y, x = np.unravel_index(int(np.argmax(array)), array.shape)
    return x / float(array.shape[1]), y / float(array.shape[0])


def derive_risk_targets(
    record: Mapping[str, Any], failure_l2_threshold: float = 0.15
) -> dict[str, float | int | None]:
    """Derive pilot targets without using any additional annotation.

    Localization uses the nearest valid annotation, matching the usual
    multi-annotation GazeFollow ``min_l2`` protocol. Out-of-frame records have no
    localization target.
    """

    normalized = normalize_record(record)
    gt_inout = normalized["gt_inout"]
    localization_l2: float | None = None
    localization_failure: int | None = None
    if gt_inout == 1:
        gaze = np.asarray(normalized["gt_gaze"], dtype=np.float64)
        valid = gaze[(gaze[:, 0] >= 0) & (gaze[:, 1] >= 0)]
        if valid.size == 0:
            raise ValueError("in-frame record has no valid gaze annotation")
        pred = np.asarray(heatmap_peak(normalized["final_heatmap"]))
        localization_l2 = float(np.linalg.norm(valid - pred[None, :], axis=1).min())
        localization_failure = int(localization_l2 > failure_l2_threshold)

    probability = normalized.get("inout_probability")
    visibility_failure: int | None = None
    visibility_brier: float | None = None
    if probability is not None:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("inout_probability must lie in [0, 1]")
        visibility_failure = int((probability >= 0.5) != bool(gt_inout))
        visibility_brier = float((probability - gt_inout) ** 2)
    return {
        "localization_l2": localization_l2,
        "localization_failure": localization_failure,
        "visibility_failure": visibility_failure,
        "visibility_brier": visibility_brier,
    }


def attach_risk_targets(
    records: Sequence[Mapping[str, Any]], failure_l2_threshold: float = 0.15
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = normalize_record(record)
        item.update(derive_risk_targets(item, failure_l2_threshold))
        output.append(item)
    return output


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "attach_risk_targets",
    "checkpoint_sha256",
    "derive_risk_targets",
    "heatmap_peak",
    "load_prediction_cache",
    "make_sample_id",
    "normalize_record",
    "save_prediction_cache",
    "validate_records",
]
