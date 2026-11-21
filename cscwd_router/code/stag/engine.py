"""Router-only/decoder-only updates and the repository's GazeFollow metrics."""

import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from gazelle.routing.losses import coverage_router_loss
from gazelle.utils import gazefollow_auc, gazefollow_l2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def train_epoch(model, loader, optimizer, device, *, budget_weight=0.05, log_every=50):
    model.train()
    totals = {}
    persons = images = updates = 0
    for index, cpu_batch in enumerate(loader):
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if model.stage == "support":
            routing = model.predict_support(batch.inputs())
            terms = coverage_router_loss(routing, batch.heatmaps, inout=batch.inout,
                                          budget_weight=budget_weight, entropy_weight=0.0)
        else:
            mode = "sparse" if model.stage == "sparse_decoder" else "dense"
            predictions = model(batch.inputs(), mode=mode)
            terms = {"total": F.binary_cross_entropy(torch.cat(predictions["heatmap"]).float(), batch.heatmaps.float())}
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError(f"non-finite loss: stage={model.stage}, batch={index}")
        terms["total"].backward()
        optimizer.step()
        count = len(batch.gazex)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value.item() * count
        persons += count
        images += batch.images.shape[0]
        updates += 1
        if index % log_every == 0:
            print(f"stage={model.stage} batch={index} loss={terms['total'].item():.6f}", flush=True)
    return {"losses": {key: value / persons for key, value in totals.items()},
            "persons": persons, "images": images, "optimizer_steps": updates}


@torch.inference_mode()
def evaluate(model, loader, device, *, mode):
    """Mean over people; AUC at original resolution and all annotated gaze points."""
    model.eval()
    rows = []
    image_count = selected_points = total_points = 0
    strata = {name: [] for name in ("single", "two", "three_plus")}
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        predictions = model(batch.inputs(), mode=mode)
        heatmaps = torch.cat(predictions["heatmap"]).float().cpu()
        head_counts = [len(boxes) for boxes in batch.bboxes for _ in boxes]
        routing = predictions["routing"]
        hard_masks = routing.image_hard_masks.cpu() if routing is not None else None
        mapping = routing.person_to_image.cpu() if routing is not None else None
        for index, heatmap in enumerate(heatmaps):
            avg_l2, min_l2 = gazefollow_l2(heatmap, batch.gazex[index], batch.gazey[index])
            row = [float(gazefollow_auc(heatmap, batch.gazex[index], batch.gazey[index], batch.heights[index], batch.widths[index])),
                   float(avg_l2), float(min_l2)]
            rows.append(row)
            count = head_counts[index]
            strata["single" if count == 1 else "two" if count == 2 else "three_plus"].append(row)
            for x, y in zip(batch.gazex[index], batch.gazey[index]):
                if routing is None:
                    selected_points += 1
                else:
                    height, width = hard_masks.shape[-2:]
                    ix, iy = min(int(x * width), width - 1), min(int(y * height), height - 1)
                    selected_points += int(hard_masks[mapping[index], iy, ix])
                total_points += 1
        image_count += batch.images.shape[0]

    def summarize(values):
        if not values:
            return {"persons": 0, "auc": None, "avg_l2": None, "min_l2": None}
        means = np.asarray(values).mean(0).tolist()
        if not all(math.isfinite(value) for value in means):
            raise FloatingPointError("non-finite GazeFollow metrics")
        return dict(zip(("auc", "avg_l2", "min_l2"), means), persons=len(values))

    return {**summarize(rows), "images": image_count, "query_unit": "image_with_all_heads", "precision": "fp32",
            "routing_gt_point_coverage": selected_points / total_points,
            "head_count_strata": {name: summarize(values) for name, values in strata.items()}}
