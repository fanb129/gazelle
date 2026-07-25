"""Deterministic grouped dataset splits used by formal experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence


@dataclass(frozen=True)
class GroupedHoldoutSplit:
    """A deterministic record split whose groups never cross partitions."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_group_count: int
    validation_group_count: int
    source_group_fingerprint: str
    train_group_fingerprint: str
    validation_group_fingerprint: str
    assignment_fingerprint: str
    validation_fraction: float
    seed: int

    def metadata(self) -> dict:
        return {
            "strategy": "stable_sha256_group_holdout_v1",
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "train_record_count": len(self.train_indices),
            "validation_record_count": len(self.validation_indices),
            "train_group_count": self.train_group_count,
            "validation_group_count": self.validation_group_count,
            "source_group_fingerprint": self.source_group_fingerprint,
            "train_group_fingerprint": self.train_group_fingerprint,
            "validation_group_fingerprint": self.validation_group_fingerprint,
            "assignment_fingerprint": self.assignment_fingerprint,
        }


def _stable_group_score(group: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).digest()


def _fingerprint(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def grouped_holdout_split(
    group_keys: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> GroupedHoldoutSplit:
    """Split record indices by a stable hash of their group keys.

    Records sharing the same key always stay together. Sorting by SHA256 rather
    than using Python's process-randomized ``hash`` makes the split identical
    across machines and Python versions.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(group_keys) < 2:
        raise ValueError("at least two records are required for a holdout split")

    normalized_keys = tuple(str(key) for key in group_keys)
    unique_groups = sorted(set(normalized_keys))
    if len(unique_groups) < 2:
        raise ValueError("at least two distinct groups are required for a holdout split")

    ordered_groups = sorted(
        unique_groups,
        key=lambda group: (_stable_group_score(group, int(seed)), group),
    )
    validation_group_count = int(
        math.floor(len(ordered_groups) * float(validation_fraction) + 0.5)
    )
    validation_group_count = max(
        1, min(len(ordered_groups) - 1, validation_group_count)
    )
    validation_groups = frozenset(ordered_groups[:validation_group_count])
    train_groups = frozenset(set(unique_groups) - validation_groups)

    train_indices = tuple(
        index
        for index, group in enumerate(normalized_keys)
        if group not in validation_groups
    )
    validation_indices = tuple(
        index
        for index, group in enumerate(normalized_keys)
        if group in validation_groups
    )
    if not train_indices or not validation_indices:
        raise RuntimeError("grouped holdout produced an empty partition")

    source_group_fingerprint = _fingerprint(
        [f"{index}\0{group}" for index, group in enumerate(normalized_keys)]
    )
    train_group_fingerprint = _fingerprint(sorted(train_groups))
    validation_group_fingerprint = _fingerprint(sorted(validation_groups))
    assignment_fingerprint = _fingerprint(
        [
            f"{index}\0{group}\0"
            f"{'validation' if group in validation_groups else 'train'}"
            for index, group in enumerate(normalized_keys)
        ]
    )
    return GroupedHoldoutSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_group_count=len(unique_groups) - validation_group_count,
        validation_group_count=validation_group_count,
        source_group_fingerprint=source_group_fingerprint,
        train_group_fingerprint=train_group_fingerprint,
        validation_group_fingerprint=validation_group_fingerprint,
        assignment_fingerprint=assignment_fingerprint,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
