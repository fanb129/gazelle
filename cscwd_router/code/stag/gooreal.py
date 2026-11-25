"""GOO benchmark metrics and fixed-checkpoint cross-dataset inference."""

import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

from cscwd_router.code.data_prep.gooreal import GoorealTest, collate


METRIC = {
    "auc": "per-query ROC AUC after cv2 INTER_LINEAR resize to 5x5; one GT cell; macro mean",
    "l2": "argmax column/heatmap_width, row/heatmap_height; normalized Euclidean distance; macro mean",
    "reference": "https://github.com/upeee/GOO-GAZE2021/blob/main/gazefollowing/evaluate.py",
    "ap": "not applicable: all test gaze points are in-frame",
}


def query_metrics(heatmap, x, y):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if not np.isfinite(heatmap).all():
        raise FloatingPointError("non-finite GOO-Real heatmap")
    target = np.zeros((5, 5), dtype=np.int32)
    target[min(int(y * 5), 4), min(int(x * 5), 4)] = 1
    coarse = cv2.resize(heatmap, (5, 5), interpolation=cv2.INTER_LINEAR)
    row, col = np.unravel_index(heatmap.argmax(), heatmap.shape)
    point = [float(col / heatmap.shape[1]), float(row / heatmap.shape[0])]
    return {"auc": float(roc_auc_score(target.ravel(), coarse.ravel())),
            "l2": float(np.hypot(point[0] - x, point[1] - y)), "prediction": point}


def visualization_indices(records, seed=0, count=8):
    return sorted(range(len(records)), key=lambda i: hashlib.sha256(
        f"{seed}\0{records[i]['path']}".encode()).hexdigest())[:count]


@torch.inference_mode()
def evaluate(model, transform, records, args, variant, rows_path, cases_path):
    loader = DataLoader(GoorealTest(records, args.data_path, transform), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, collate_fn=collate,
                        pin_memory=args.device == "cuda")
    selected = visualization_indices(records, args.seed)
    heatmaps = {}
    totals = np.zeros(2, dtype=np.float64)
    count = covered = 0
    token_trace = None
    Path(rows_path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(rows_path, "wt") as handle:
        for batch_index, (images, indices) in enumerate(loader):
            boxes = [[records[i]["heads"][0]["bbox_norm"]] for i in indices]
            output = model({"images": images.to(args.device), "bboxes": boxes})
            heat = torch.cat(output["heatmap"]).float().cpu().numpy()
            routing = output.get("routing")
            masks = routing.image_hard_masks.cpu().numpy() if routing is not None else None
            if token_trace is None:
                token_trace = {"input_shape": list(images.shape[1:]), "heatmap_shape": list(heat.shape[1:]),
                               "kept_spatial_tokens": int(masks[0].sum()) if masks is not None else 1024}
                if token_trace["kept_spatial_tokens"] != (512 if variant == "k50" else 1024):
                    raise RuntimeError("actual GOO-Real token budget differs from requested variant")
            for j, i in enumerate(indices):
                record = records[i]
                head = record["heads"][0]
                x, y = head["gazex_norm"][0], head["gazey_norm"][0]
                metrics = query_metrics(heat[j], x, y)
                kept = None if masks is None else bool(masks[j, min(int(y*32), 31), min(int(x*32), 31)])
                row = {"index": i, "path": record["path"], "bbox_norm": head["bbox_norm"],
                       "gaze": [x, y], **metrics, "gt_point_kept": kept}
                handle.write(json.dumps(row) + "\n")
                totals += [metrics["auc"], metrics["l2"]]
                count += 1
                covered += int(kept) if kept is not None else 0
                if i in selected:
                    heatmaps[i] = heat[j]
            if batch_index % 50 == 0:
                print(json.dumps({"variant": variant, "processed": count, "total": len(records)}), flush=True)
    if count != len(records):
        raise RuntimeError("GOO-Real inference did not cover the full official test split")
    np.savez_compressed(cases_path, indices=np.asarray(selected), heatmaps=np.stack([heatmaps[i] for i in selected]))
    return {"images": count, "persons": count, "auc": float(totals[0]/count), "l2": float(totals[1]/count),
            "routing_gt_point_coverage": None if variant == "official" else covered/count}, token_trace
