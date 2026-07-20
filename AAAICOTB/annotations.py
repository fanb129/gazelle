"""Torch-free VAT annotation loading and sequence split helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def load_sequences(annotation_path: str | Path) -> list[dict]:
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("VAT annotation must be a list of sequence dictionaries")
    return payload


def sequence_id(sequence: dict, frame: dict) -> str:
    return str(sequence.get("path") or Path(frame["path"]).parent)


def split_sequence_indices(
    sequences: Sequence[dict], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if len(sequences) < 2:
        raise ValueError("at least two VAT sequences are required for a train/validation split")
    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    validation_count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])
