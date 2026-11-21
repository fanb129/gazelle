"""Deterministic synthetic RGB/head/target fixture for T2 engineering checks."""

from dataclasses import dataclass

import torch


@dataclass
class SyntheticSupportBatch:
    images: torch.Tensor
    bboxes: list[list[tuple[float, float, float, float]]]
    target_heatmaps: torch.Tensor


def make_synthetic_support_batch(
    batch_size: int = 2, image_size: int = 448, heatmap_size: int = 64, seed: int = 3106,
) -> SyntheticSupportBatch:
    """Mirror each head center horizontally to give distinct person targets.

    This fixture tests optimization and query alignment; it is not gaze data.
    Targets stay separate from the encoder's RGB/head-box input interface.
    """
    images = torch.randn(batch_size, 3, image_size, image_size, generator=torch.Generator().manual_seed(seed))
    bboxes = []
    heatmaps = []
    grid = (torch.arange(heatmap_size, dtype=torch.float32) + 0.5) / heatmap_size
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    for image_index in range(batch_size):
        boxes = ([(0.10, 0.10, 0.25, 0.30), (0.65, 0.15, 0.85, 0.35)]
                 if image_index % 2 == 0 else [(0.25, 0.10, 0.45, 0.30)])
        bboxes.append(boxes)
        for xmin, ymin, xmax, ymax in boxes:
            target_x = 1.0 - (xmin + xmax) * 0.5
            target_y = 0.65 + (ymin + ymax) * 0.25
            heatmaps.append(torch.exp(-((xx - target_x).square() + (yy - target_y).square()) / (2 * 0.04 ** 2)))
    return SyntheticSupportBatch(images, bboxes, torch.stack(heatmaps))
