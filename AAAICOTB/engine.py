"""Shared training loss and raw VAT evaluation engine for COTB."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from AAAICOTB.binding import BindingConfig, compute_frame_binding


def _head_size(bbox: Sequence[float]) -> float:
    return math.sqrt(
        max(0.0, float(bbox[2]) - float(bbox[0]))
        * max(0.0, float(bbox[3]) - float(bbox[1]))
    )


def compute_batch_loss(
    output: dict,
    batch: dict,
    device: torch.device,
    binding_config: BindingConfig,
    bind_weight: float,
    inout_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_heatmaps = torch.cat(output["heatmap"], dim=0)
    target_heatmaps = torch.cat([item.to(device) for item in batch["heatmaps"]], dim=0)
    inout = torch.cat([item.to(device) for item in batch["inout"]], dim=0)
    if inout.any():
        heatmap_loss = F.binary_cross_entropy(predicted_heatmaps[inout], target_heatmaps[inout])
    else:
        heatmap_loss = predicted_heatmaps.sum() * 0.0

    predicted_inout = torch.cat(output["inout"], dim=0)
    inout_loss = F.binary_cross_entropy(predicted_inout, inout.float())

    binding_loss_sum = predicted_heatmaps.sum() * 0.0
    pair_count = 0
    for frame_index, frame_heatmaps in enumerate(output["heatmap"]):
        result = compute_frame_binding(
            frame_heatmaps,
            batch["targets"][frame_index].to(device),
            batch["inout"][frame_index].to(device),
            batch["binding_bboxes"][frame_index],
            binding_config,
        )
        if result["pair_count"]:
            binding_loss_sum = binding_loss_sum + result["loss"] * result["pair_count"]
            pair_count += result["pair_count"]
    binding_loss = binding_loss_sum / pair_count if pair_count else binding_loss_sum
    total = heatmap_loss + inout_loss_weight * inout_loss + bind_weight * binding_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "inout_loss": float(inout_loss.detach().cpu()),
        "binding_loss": float(binding_loss.detach().cpu()),
        "binding_pairs": float(pair_count),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    binding_config: BindingConfig,
    bind_weight: float,
    inout_loss_weight: float,
    log_every: int,
    max_batches: int | None = None,
) -> dict[str, float | None]:
    model.train()
    # The VFM is frozen; retaining evaluation behavior avoids stochastic
    # backbone changes while the task head is optimized.
    model.backbone.eval()
    rows: list[dict[str, float]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model({"images": batch["images"].to(device), "bboxes": batch["bboxes"]})
        loss, row = compute_batch_loss(
            output, batch, device, binding_config, bind_weight, inout_loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append(row)
        if batch_index % log_every == 0:
            print(
                f"train batch={batch_index}/{len(loader)} total={row['loss']:.6f} "
                f"heatmap={row['heatmap_loss']:.6f} inout={row['inout_loss']:.6f} "
                f"binding={row['binding_loss']:.6f} pairs={int(row['binding_pairs'])}",
                flush=True,
            )
    keys = ("loss", "heatmap_loss", "inout_loss", "binding_loss")
    summary: dict[str, float | None] = {
        key: float(np.mean([row[key] for row in rows])) if rows else None for key in keys
    }
    summary["binding_pairs_seen"] = float(sum(row["binding_pairs"] for row in rows))
    summary["batch_count"] = float(len(rows))
    return summary


def _bootstrap_mean(
    rows: Sequence[dict],
    field: str,
    iterations: int,
    seed: int,
    cluster_field: str = "sequence_id",
) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    clusters = sorted({str(row[cluster_field]) for row in rows if row.get(field) is not None})
    result: dict[str, float | int | None] = {
        "value": float(np.mean(values)) if values else None,
        "row_count": len(values),
        "cluster_count": len(clusters),
        "ci95_low": None,
        "ci95_high": None,
    }
    if iterations <= 0 or not values or len(clusters) < 2:
        return result
    by_cluster = {
        cluster: np.asarray(
            [float(row[field]) for row in rows if str(row.get(cluster_field)) == cluster and row.get(field) is not None],
            dtype=np.float64,
        )
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled = np.concatenate([by_cluster[clusters[index]] for index in selected])
        samples.append(float(sampled.mean()))
    result["ci95_low"], result["ci95_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return result


def _binding_subsets(pair_rows: Sequence[dict]) -> dict[str, list[dict]]:
    return {
        "overall": list(pair_rows),
        "far": [row for row in pair_rows if float(row["target_separation"]) >= 0.30],
        "crowd_ge4": [row for row in pair_rows if int(row["people_count"]) >= 4],
        "crowd_ge5": [row for row in pair_rows if int(row["people_count"]) >= 5],
        "far_crowd_ge4": [
            row
            for row in pair_rows
            if float(row["target_separation"]) >= 0.30 and int(row["people_count"]) >= 4
        ],
    }


def summarize_evaluation(
    person_rows: Sequence[dict],
    pair_rows: Sequence[dict],
    query_rows: Sequence[dict],
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    from sklearn.metrics import average_precision_score

    inframe = [row for row in person_rows if int(row["inout"]) == 1]
    labels = [int(row["inout"]) for row in person_rows]
    scores = [float(row["inout_score"]) for row in person_rows]
    standard = {
        "query_count": len(person_rows),
        "inframe_count": len(inframe),
        "auc": float(np.mean([row["auc"] for row in inframe])) if inframe else None,
        "l2": float(np.mean([row["l2"] for row in inframe])) if inframe else None,
        "inout_ap": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None,
    }
    binding = {}
    for offset, (name, rows) in enumerate(_binding_subsets(pair_rows).items()):
        binding[name] = {
            "swap_error": _bootstrap_mean(rows, "swap_error", bootstrap_iterations, seed + offset),
            "diagonal_accuracy": _bootstrap_mean(rows, "diagonal_correct", bootstrap_iterations, seed + 100 + offset),
            "diag_margin": _bootstrap_mean(rows, "diag_margin", bootstrap_iterations, seed + 200 + offset),
        }
    query_binding = {
        "ownership_accuracy": _bootstrap_mean(query_rows, "ownership_correct", bootstrap_iterations, seed + 300),
        "own_target_rank": _bootstrap_mean(query_rows, "own_target_rank", bootstrap_iterations, seed + 301),
        "ownership_margin": _bootstrap_mean(query_rows, "ownership_margin", bootstrap_iterations, seed + 302),
    }
    return {"standard": standard, "binding": binding, "query_binding": query_binding}


def evaluate_loader(
    model,
    loader,
    device: torch.device,
    binding_config: BindingConfig,
    bootstrap_iterations: int = 1000,
    seed: int = 3106,
    max_batches: int | None = None,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    from gazelle.utils import vat_auc, vat_l2

    model.eval()
    person_rows: list[dict] = []
    pair_rows: list[dict] = []
    query_rows: list[dict] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            output = model({"images": batch["images"].to(device), "bboxes": batch["bboxes"]})
            for frame_index, frame_heatmaps in enumerate(output["heatmap"]):
                targets = batch["targets"][frame_index]
                inout = batch["inout"][frame_index]
                inout_scores = output["inout"][frame_index]
                path = batch["paths"][frame_index]
                seq = batch["sequence_ids"][frame_index]
                people_count = len(frame_heatmaps)
                binding_result = compute_frame_binding(
                    frame_heatmaps,
                    targets.to(device),
                    inout.to(device),
                    batch["binding_bboxes"][frame_index],
                    binding_config,
                )
                for row in binding_result["pairs"]:
                    first_size = _head_size(batch["binding_bboxes"][frame_index][row["query_i"]])
                    second_size = _head_size(batch["binding_bboxes"][frame_index][row["query_j"]])
                    pair_rows.append(
                        {
                            "sequence_id": seq,
                            "path": path,
                            "people_count": people_count,
                            "mean_head_size": (first_size + second_size) / 2.0,
                            **row,
                        }
                    )
                for row in binding_result["queries"]:
                    query_rows.append(
                        {
                            "sequence_id": seq,
                            "path": path,
                            "people_count": people_count,
                            "head_size": _head_size(batch["binding_bboxes"][frame_index][row["query_index"]]),
                            **row,
                        }
                    )
                for query_index, heatmap in enumerate(frame_heatmaps):
                    is_in = int(inout[query_index].item())
                    row = {
                        "sequence_id": seq,
                        "path": path,
                        "query_index": query_index,
                        "people_count": people_count,
                        "head_size": _head_size(batch["binding_bboxes"][frame_index][query_index]),
                        "inout": is_in,
                        "target_x": float(targets[query_index, 0]),
                        "target_y": float(targets[query_index, 1]),
                        "inout_score": float(inout_scores[query_index].detach().cpu()),
                        "auc": None,
                        "l2": None,
                    }
                    if is_in:
                        cpu_heatmap = heatmap.detach().cpu()
                        row["auc"] = float(vat_auc(cpu_heatmap, row["target_x"], row["target_y"]))
                        row["l2"] = float(vat_l2(cpu_heatmap, row["target_x"], row["target_y"]))
                    person_rows.append(row)
    summary = summarize_evaluation(person_rows, pair_rows, query_rows, bootstrap_iterations, seed)
    return summary, person_rows, pair_rows, query_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(
    output_dir: Path,
    report: dict,
    person_rows: Sequence[dict],
    pair_rows: Sequence[dict],
    query_rows: Sequence[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "persons.csv", person_rows)
    write_csv(output_dir / "pairs.csv", pair_rows)
    write_csv(output_dir / "queries.csv", query_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
