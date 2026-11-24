"""Matched real-image multi-query accuracy and complete-forward timing."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from gazelle.dataloader import collate_gazefollow_images
from gazelle.utils import vat_auc, vat_l2
from cscwd_router.code.data_prep.final_queries import VATQueryImages, cohort
from cscwd_router.code.data_prep.train_test import query_identifier
from cscwd_router.code.stag import final_io as io, dinov3_efficiency as efficiency
from scripts.train_coverage_router import unpack_batch, stack_predictions, exact_point_coverage

VARIANTS = ("official", "independent", "union")


class IndependentQueries(nn.Module):
    """Batch independent copies of the image; no shared person-union route."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def router(self):
        return self.model.router

    @property
    def backbone(self):
        return self.model.backbone

    @property
    def route_after_block(self):
        return self.model.route_after_block

    def forward(self, inputs):
        images = torch.cat([image.unsqueeze(0).expand(len(boxes), -1, -1, -1)
                            for image, boxes in zip(inputs["images"], inputs["bboxes"])], dim=0)
        boxes = [[box] for group in inputs["bboxes"] for box in group]
        return self.model({"images": images, "bboxes": boxes})


def build_variant(source, payload, official_checkpoint, variant):
    if variant == "official":
        return efficiency.build_variant(source, payload, official_checkpoint, "official")
    model, transform = io.build_router(source, payload)
    return (IndependentQueries(model) if variant == "independent" else model), transform


def loader_for(data_path, records, manifest, n, transform, batch_size, workers):
    dataset = VATQueryImages(data_path, records, manifest, n, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      collate_fn=collate_gazefollow_images, pin_memory=True)


@torch.inference_mode()
def evaluate(model, loader, device, rows_path):
    model.eval()
    values, flags_all, scores_all = [], [], []
    persons = point_hits = point_total = 0
    rows_path = Path(rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(rows_path) + ".tmp", "wt", encoding="utf-8") as stream:
        for batch_index, batch in enumerate(loader):
            inputs, xs, ys, flags, _, _, _ = unpack_batch(batch, device, training=False)
            output = model(inputs)
            maps, scores = stack_predictions(output)
            if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
                raise FloatingPointError("non-finite multi-query predictions")
            maps, scores = maps.float().cpu(), scores.float().cpu().tolist()
            flags_list = flags.cpu().int().tolist()
            if output.get("routing") is not None:
                recall, count = exact_point_coverage(output["routing"], xs, ys, flags)
                point_hits += (recall or 0) * count
                point_total += count
            for offset, (heatmap, flag, score) in enumerate(zip(maps, flags_list, scores)):
                index = persons + offset
                row = {"sample_index": index, **query_identifier(loader.dataset, index),
                       "in_frame": flag, "inout_score": score, "gazex": xs[offset], "gazey": ys[offset]}
                if flag:
                    a, l2 = float(vat_auc(heatmap, xs[offset][0], ys[offset][0])), float(vat_l2(heatmap, xs[offset][0], ys[offset][0]))
                    values.append([a, l2])
                    row.update(auc=a, l2=l2)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            persons += len(flags_list)
            flags_all.extend(flags_list)
            scores_all.extend(scores)
            if batch_index % 100 == 0:
                print(f"MULTI TEST batch={batch_index}/{len(loader)} persons={persons}", flush=True)
    Path(str(rows_path) + ".tmp").replace(rows_path)
    means = np.asarray(values).mean(0).tolist() if values else [None, None]
    return {"auc": means[0], "l2": means[1], "inout_ap": float(average_precision_score(flags_all, scores_all)),
            "persons": persons, "inframe_persons": sum(flags_all), "frames": len(loader.dataset),
            "queries_per_frame": loader.dataset.count, "query_unit": "person", "precision": "fp32",
            "routing_gt_point_coverage": point_hits / point_total if point_total else None}


def timing_sample(data_path, seed):
    records, manifest = cohort(data_path, seed, split="train")
    index = min(range(len(records)), key=lambda i: hashlib.sha256(
        f"{seed}\0{records[i]['path']}".encode()).hexdigest())
    return records[index], manifest["entries"][index], manifest["annotation"]


def trace(model, inputs, variant):
    seen = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, values, index=index: seen.append({"block": index, "batch": values[0].shape[0], "tokens": values[0].shape[1]})
    ) for index, block in enumerate(model.backbone.model.blocks)]
    try:
        with torch.inference_mode(), torch.autocast(inputs["images"].device.type, dtype=torch.float16,
                                                   enabled=inputs["images"].device.type == "cuda", cache_enabled=False):
            output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    copies = sum(len(boxes) for boxes in inputs["bboxes"]) if variant == "independent" else len(inputs["bboxes"])
    expected = [{"block": i, "batch": copies,
                 "tokens": 1025 if variant == "official" else 1029 if i <= model.route_after_block else 517}
                for i in range(len(model.backbone.model.blocks))]
    if seen != expected:
        raise RuntimeError(f"unexpected multi-query token graph: {seen}")
    if output.get("routing") is not None and output["routing"].image_keep_indices.shape[1] != 512:
        raise ValueError("multi-query routing changed the fixed K50 budget")
    maps, scores = stack_predictions(output)
    if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite timing trace predictions")
    return {"block_inputs": seen, "encoded_image_copies": copies, "input_images": inputs["images"].shape[0],
            "queried_people": sum(len(boxes) for boxes in inputs["bboxes"]),
            "kept_patches_per_copy": 1024 if variant == "official" else 512}


def aggregate_timing(cells):
    result = {}
    for n in (1, 2, 3):
        pair = [cell for cell in cells if cell["request"]["n"] == n]
        if len(pair) != 2 or {cell["request"]["order"] for cell in pair} != {"forward", "reverse"}:
            raise ValueError("multi-query timing needs both orders for each N")
        if pair[0]["request"]["sample"] != pair[1]["request"]["sample"]:
            raise ValueError("multi-query timing samples differ between orders")
        variants = {}
        for variant in VARIANTS:
            values = [cell["results"][variant]["timing"] for cell in pair]
            ms = math.exp(sum(math.log(v["latency_ms_median"]) for v in values) / 2)
            fps = math.exp(sum(math.log(v["throughput_images_per_second_median"]) for v in values) / 2)
            variants[variant] = {"latency_ms_per_frame": ms, "images_per_second": fps,
                "queries_per_second": n * fps, "peak_allocated_mib": max(v["peak_allocated_mib"] for v in values)}
        result[str(n)] = variants
    return result
