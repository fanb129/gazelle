"""Torch-free geometry helpers used by annotation audit and COTB loss."""

from __future__ import annotations

import itertools
import math
from typing import Sequence


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def cluster_targets(
    points: Sequence[tuple[float, float]], radius: float
) -> tuple[list[int], list[tuple[float, float]]]:
    """Single-link cluster near-identical gaze points into shared targets."""

    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in itertools.combinations(range(len(points)), 2):
        if math.dist(points[first], points[second]) <= radius:
            union(first, second)

    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(points)):
        root = find(index)
        root_to_label.setdefault(root, len(root_to_label))
        labels.append(root_to_label[root])

    centers: list[tuple[float, float]] = []
    for label in range(len(root_to_label)):
        members = [points[index] for index, value in enumerate(labels) if value == label]
        centers.append(
            (
                sum(point[0] for point in members) / len(members),
                sum(point[1] for point in members) / len(members),
            )
        )
    return labels, centers
