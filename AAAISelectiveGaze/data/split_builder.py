"""Deterministic, group-safe dataset splits for SelectiveGaze.

GazeFollow is split by image (``path`` by default).  VideoAttentionTarget
(VAT) is split by complete sequence, never by frame.  The functions in this
module intentionally operate on the preprocessed JSON structures used by the
existing Gazelle dataloader so that the generated files remain drop-in
compatible with it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


SPLIT_NAMES = (
    "base_train",
    "base_val",
    "risk_train",
    "risk_calibration",
)

_GAZEFOLLOW_PATH_KEYS = ("path", "image_path", "image", "filename")
_VAT_SEQUENCE_KEYS = ("sequence_id", "sequence", "video_id", "video", "path")


@dataclass(frozen=True)
class SplitRatios:
    """Ratios for the four disjoint development splits."""

    base_train: float = 0.85
    base_val: float = 0.05
    risk_train: float = 0.05
    risk_calibration: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "base_train": self.base_train,
            "base_val": self.base_val,
            "risk_train": self.risk_train,
            "risk_calibration": self.risk_calibration,
        }

    def validate(self) -> None:
        values = self.as_dict()
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Split ratios must be finite and non-negative")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Split ratios must sum to 1.0; got "
                f"{sum(values.values()):.12g}"
            )


def _normalise_dataset(dataset: str) -> str:
    name = dataset.strip().lower().replace("-", "").replace("_", "")
    if name in {"gazefollow", "gf"}:
        return "gazefollow"
    if name in {"vat", "videoattentiontarget"}:
        return "vat"
    raise ValueError("dataset must be 'gazefollow' or 'vat'")


def _json_scalar(value: Any, *, label: str) -> str:
    if value is None or isinstance(value, (dict, list)):
        raise ValueError(f"{label} must be a non-null JSON scalar")
    return str(value)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return key, record[key]
    raise KeyError(f"None of the required keys are present: {', '.join(keys)}")


def _group_value(
    record: Mapping[str, Any],
    *,
    dataset: str,
    group_key: str,
    nested_sequence: bool,
) -> str:
    if group_key in record and record[group_key] not in (None, ""):
        return _json_scalar(record[group_key], label=group_key)

    # The repository's legacy VAT preprocessing uses ``path`` for the
    # sequence identity, while the documented CLI calls it ``sequence_id``.
    # Only apply this aliasing to a sequence container; flat frame lists must
    # provide a real sequence/group key so that we cannot silently leak video.
    if dataset == "vat" and nested_sequence and group_key == "sequence_id":
        _, value = _first_present(record, _VAT_SEQUENCE_KEYS)
        return _json_scalar(value, label="VAT sequence identity")

    if dataset == "gazefollow" and group_key == "path":
        _, value = _first_present(record, _GAZEFOLLOW_PATH_KEYS)
        return _json_scalar(value, label="GazeFollow image path")

    raise KeyError(f"Record is missing required group key {group_key!r}")


def _allocate_group_counts(number_of_groups: int, ratios: SplitRatios) -> dict[str, int]:
    """Allocate whole groups with Hamilton rounding.

    When there are at least as many groups as non-zero splits, every non-zero
    split receives a group.  This makes small synthetic/pilot datasets useful
    without ever breaking the group boundary.
    """

    ratio_map = ratios.as_dict()
    exact = {name: number_of_groups * ratio_map[name] for name in SPLIT_NAMES}
    counts = {name: int(math.floor(exact[name])) for name in SPLIT_NAMES}
    remaining = number_of_groups - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1

    positive = [name for name in SPLIT_NAMES if ratio_map[name] > 0]
    if number_of_groups >= len(positive):
        for empty_name in (name for name in positive if counts[name] == 0):
            donors = [name for name in positive if counts[name] > 1]
            if not donors:
                break
            donor = max(
                donors,
                key=lambda name: (counts[name] - exact[name], -SPLIT_NAMES.index(name)),
            )
            counts[donor] -= 1
            counts[empty_name] += 1

    assert sum(counts.values()) == number_of_groups
    return counts


def _add_person_sample_ids(
    item: MutableMapping[str, Any], *, dataset: str, group_id: str
) -> None:
    frames = item.get("frames") if isinstance(item.get("frames"), list) else [item]
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, MutableMapping):
            raise ValueError("Every frame must be a JSON object")
        frame_identity = str(frame.get("path", frame_index))
        heads = frame.get("heads", [])
        if not isinstance(heads, list):
            raise ValueError("Frame 'heads' must be a list")
        for head_index, head in enumerate(heads):
            if not isinstance(head, MutableMapping):
                raise ValueError("Every head annotation must be a JSON object")
            if "sample_id" not in head:
                person_identity = head.get("head_id", head_index)
                raw_id = f"{dataset}|{group_id}|{frame_identity}|{person_identity}"
                digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
                head["sample_id"] = f"{dataset}:{digest}"


def _sample_ids(item: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    if "sample_id" in item:
        result.append(_json_scalar(item["sample_id"], label="sample_id"))
    frames = item.get("frames") if isinstance(item.get("frames"), list) else [item]
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        if frame is not item and "sample_id" in frame:
            result.append(_json_scalar(frame["sample_id"], label="sample_id"))
        heads = frame.get("heads", [])
        if isinstance(heads, list):
            for head in heads:
                if isinstance(head, Mapping) and "sample_id" in head:
                    result.append(_json_scalar(head["sample_id"], label="sample_id"))
    return result


def _image_ids(item: Mapping[str, Any]) -> list[str]:
    frames = item.get("frames") if isinstance(item.get("frames"), list) else [item]
    return [
        str(frame["path"])
        for frame in frames
        if isinstance(frame, Mapping) and frame.get("path") not in (None, "")
    ]


def _validate_identity_ownership(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    sample_owner: dict[str, str] = {}
    image_owner: dict[str, str] = {}
    for group_id, items in grouped.items():
        for item in items:
            for sample_id in _sample_ids(item):
                previous = sample_owner.setdefault(sample_id, group_id)
                if previous != group_id:
                    raise ValueError(
                        f"sample_id {sample_id!r} occurs in multiple groups: "
                        f"{previous!r} and {group_id!r}"
                    )
            for image_id in _image_ids(item):
                previous = image_owner.setdefault(image_id, group_id)
                if previous != group_id:
                    raise ValueError(
                        f"image path {image_id!r} occurs in multiple groups: "
                        f"{previous!r} and {group_id!r}"
                    )


def build_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    ratios: SplitRatios | None = None,
    group_key: str | None = None,
    seed: int = 3106,
    add_sample_ids: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build four deterministic group-disjoint splits.

    Returns ``(splits, manifest)`` without writing files.  VAT sequence-list
    inputs remain sequence lists in every output; frames are never flattened.
    """

    dataset = _normalise_dataset(dataset)
    ratios = ratios or SplitRatios()
    ratios.validate()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("Input JSON must be a list of records")

    default_group_key = "path" if dataset == "gazefollow" else "sequence_id"
    group_key = group_key or default_group_key
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    resolved_group_key = group_key

    for index, source_item in enumerate(records):
        if not isinstance(source_item, Mapping):
            raise ValueError(f"Record {index} is not a JSON object")
        nested_sequence = dataset == "vat" and isinstance(source_item.get("frames"), list)
        if dataset == "vat" and not nested_sequence and group_key == "sequence_id":
            # Flat VAT records are accepted only when an explicit/recognised
            # sequence key is present; image paths are not safe substitutes.
            try:
                actual_key, value = _first_present(source_item, _VAT_SEQUENCE_KEYS[:-1])
            except KeyError as exc:
                raise KeyError(
                    "Flat VAT input requires sequence_id (or sequence/video_id/video); "
                    "do not split frames by image path"
                ) from exc
            group_id = _json_scalar(value, label=actual_key)
            resolved_group_key = actual_key
        else:
            group_id = _group_value(
                source_item,
                dataset=dataset,
                group_key=group_key,
                nested_sequence=nested_sequence,
            )
            if (
                dataset == "vat"
                and nested_sequence
                and group_key == "sequence_id"
                and "sequence_id" not in source_item
            ):
                resolved_group_key = "path (sequence_id alias)"

        item = copy.deepcopy(dict(source_item))
        if add_sample_ids:
            _add_person_sample_ids(item, dataset=dataset, group_id=group_id)
        grouped[group_id].append(item)

    _validate_identity_ownership(grouped)

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    counts = _allocate_group_counts(len(group_ids), ratios)

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_groups: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    cursor = 0
    for split_name in SPLIT_NAMES:
        selected = group_ids[cursor : cursor + counts[split_name]]
        cursor += counts[split_name]
        split_groups[split_name] = selected
        for group_id in selected:
            splits[split_name].extend(grouped[group_id])

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": seed,
        "requested_group_key": group_key,
        "resolved_group_key": resolved_group_key,
        "ratios": ratios.as_dict(),
        "total_records": len(records),
        "total_groups": len(group_ids),
        "splits": {
            name: {
                "file": f"{name}.json",
                "num_records": len(splits[name]),
                "num_groups": len(split_groups[name]),
                "num_samples": sum(len(_sample_ids(item)) for item in splits[name]),
                "group_ids": split_groups[name],
            }
            for name in SPLIT_NAMES
        },
        "leakage_check": {
            "group_intersections_empty": all(
                set(split_groups[left]).isdisjoint(split_groups[right])
                for i, left in enumerate(SPLIT_NAMES)
                for right in SPLIT_NAMES[i + 1 :]
            ),
            "sample_intersections_empty": all(
                set().union(*(_sample_ids(item) for item in splits[left])).isdisjoint(
                    set().union(*(_sample_ids(item) for item in splits[right]))
                )
                for i, left in enumerate(SPLIT_NAMES)
                for right in SPLIT_NAMES[i + 1 :]
            ),
            "image_intersections_empty": all(
                set().union(*(_image_ids(item) for item in splits[left])).isdisjoint(
                    set().union(*(_image_ids(item) for item in splits[right]))
                )
                for i, left in enumerate(SPLIT_NAMES)
                for right in SPLIT_NAMES[i + 1 :]
            ),
        },
    }
    return splits, manifest


def write_splits(
    input_json: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    ratios: SplitRatios | None = None,
    group_key: str | None = None,
    seed: int = 3106,
) -> dict[str, Any]:
    """Read preprocessed data, build splits, and write JSON plus manifest."""

    input_path = Path(input_json)
    raw_bytes = input_path.read_bytes()
    records = json.loads(raw_bytes)
    splits, manifest = build_splits(
        records,
        dataset=dataset,
        ratios=ratios,
        group_key=group_key,
        seed=seed,
    )
    manifest["input"] = {
        "path": str(input_path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        (destination / f"{split_name}.json").write_text(
            json.dumps(splits[split_name], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["SPLIT_NAMES", "SplitRatios", "build_splits", "write_splits"]
