"""Frozen DINOv3 support/decoder updates and official gaze metrics."""

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from gazelle.routing.losses import coverage_router_loss
from gazelle.utils import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2
from scripts.train_coverage_router import (
    coordinate_heatmaps, exact_point_coverage, stack_predictions, unpack_batch,
)
from cscwd_router.code.data_prep.train_test import query_identifier


def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.router_stage = "backbone_sparse"
    if stage == "support":
        model.eval()
        model.router.train()
        for parameter in model.router.parameters():
            parameter.requires_grad_(True)
    else:
        model.train()
        model.router.eval()
        for name, parameter in model.named_parameters():
            if not name.startswith(("backbone.", "router.")):
                parameter.requires_grad_(True)
    model.backbone.eval()


def support_forward(model, inputs):
    # All supported route points are among the native intermediate outputs.
    with torch.no_grad():
        state = model.backbone.forward_prefix(inputs["images"], model.route_after_block)
        features = state.dense_features[model.route_after_block]
    return model.router(features, inputs["bboxes"])


def train_epoch(model, loader, optimizer, device, *, stage, accumulation, clip_norm,
                keep_ratio, budget_weight, log_every):
    configure_stage(model, stage)
    model.router.keep_ratio = keep_ratio
    optimizer.zero_grad(set_to_none=True)
    loss_sum = persons = updates = 0
    started = time.monotonic()
    for index, batch in enumerate(loader):
        inputs, _, _, flags, _, _, targets = unpack_batch(batch, device, training=True)
        if stage == "support":
            routing = support_forward(model, inputs)
            loss = coverage_router_loss(routing, targets, inout=flags,
                                        budget_weight=budget_weight, budget_target=keep_ratio)["total"]
        else:
            output = model(inputs)
            heatmaps, inout = stack_predictions(output)
            valid = flags.bool()
            loss = F.binary_cross_entropy(heatmaps[valid].float(), targets[valid].float()) if valid.any() else heatmaps.sum() * 0
            if inout is not None:
                loss = loss + F.binary_cross_entropy(inout.float(), flags.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: stage={stage}, batch={index}")
        start = (index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - start)
        (loss / group_size).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            if clip_norm:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        count = len(flags)
        loss_sum += loss.item() * count
        persons += count
        if index % log_every == 0:
            elapsed = time.monotonic() - started
            print(f"TRAIN stage={stage} batch={index}/{len(loader)} loss={loss.item():.6f} "
                  f"keep={keep_ratio:.3f} elapsed_s={elapsed:.1f}", flush=True)
    return {"loss": loss_sum / persons, "persons": persons, "optimizer_steps": updates,
            "seconds": time.monotonic() - started, "train_keep_ratio": keep_ratio}


@torch.inference_mode()
def evaluate(model, loader, device, *, dataset, keep_ratio, rows_path=None):
    model.eval()
    model.router.keep_ratio = keep_ratio
    values = []
    flags_all, scores_all = [], []
    hard_sum = soft_sum = valid_total = point_hits = point_total = 0
    persons = 0
    stream = None
    if rows_path is not None:
        rows_path = Path(rows_path)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        stream = gzip.open(str(rows_path) + ".tmp", "wt", encoding="utf-8")
    try:
        for batch_index, batch in enumerate(loader):
            inputs, xs, ys, flags, heights, widths, _ = unpack_batch(batch, device, training=False)
            output = model(inputs)
            maps, scores = stack_predictions(output)
            maps = maps.float().cpu()
            targets = coordinate_heatmaps(xs, ys, flags).to(device)
            routing = output["routing"]
            terms = coverage_router_loss(routing, targets, inout=flags,
                                         coverage_weight=0, budget_weight=0, entropy_weight=0)
            count = int(flags.bool().sum().item())
            hard_sum += terms["hard_coverage"].item() * count
            soft_sum += terms["soft_coverage"].item() * count
            valid_total += count
            recall, points = exact_point_coverage(routing, xs, ys, flags)
            point_hits += (recall or 0) * points
            point_total += points
            flags_list = flags.cpu().int().tolist()
            score_list = scores.float().cpu().tolist() if scores is not None else [None] * len(flags_list)
            for i, heatmap in enumerate(maps):
                row = {"sample_index": persons + i, "in_frame": flags_list[i], "inout_score": score_list[i]}
                if flags_list[i]:
                    if dataset == "gazefollow":
                        avg, minimum = gazefollow_l2(heatmap, xs[i], ys[i])
                        metrics = [float(gazefollow_auc(heatmap, xs[i], ys[i], heights[i], widths[i])), float(avg), float(minimum)]
                    else:
                        metrics = [float(vat_auc(heatmap, xs[i][0], ys[i][0])), float(vat_l2(heatmap, xs[i][0], ys[i][0]))]
                    values.append(metrics)
                    row.update(zip(("auc", "avg_l2", "min_l2") if dataset == "gazefollow" else ("auc", "l2"), metrics))
                if stream is not None:
                    row.update(query_identifier(loader.dataset, persons + i))
                    row.update({"gazex": list(xs[i]), "gazey": list(ys[i])})
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            flags_all.extend(flags_list)
            scores_all.extend(score_list)
            persons += len(flags_list)
            if batch_index % 200 == 0:
                print(f"TEST batch={batch_index}/{len(loader)} persons={persons} keep={keep_ratio}", flush=True)
    finally:
        if stream is not None:
            stream.close()
    if rows_path is not None:
        Path(str(rows_path) + ".tmp").replace(rows_path)
    means = np.asarray(values).mean(0).tolist()
    if not all(math.isfinite(v) for v in means):
        raise FloatingPointError("non-finite evaluation metrics")
    result = dict(zip(("auc", "avg_l2", "min_l2") if dataset == "gazefollow" else ("auc", "l2"), means))
    if dataset == "vat":
        result["inout_ap"] = float(average_precision_score(flags_all, scores_all))
    result.update({"persons": persons, "inframe_persons": valid_total, "precision": "fp32",
                   "query_unit": "person", "keep_ratio": keep_ratio,
                   "routing_gt_point_coverage": point_hits / point_total,
                   "routing_hard_coverage": hard_sum / valid_total,
                   "routing_soft_coverage": soft_sum / valid_total})
    return result
