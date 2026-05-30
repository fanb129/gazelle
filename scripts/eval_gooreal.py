import argparse
import csv
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


class SchemaError(ValueError):
    pass


def validate_records(records):
    if not isinstance(records, list):
        raise SchemaError(f"Preprocessed JSON must be a list, got {type(records).__name__}")
    for idx, item in enumerate(records):
        for field in ["path", "heads", "width", "height"]:
            if field not in item:
                raise SchemaError(f"Item {idx} is missing field {field!r}")
        if not isinstance(item["heads"], list) or not item["heads"]:
            raise SchemaError(f"Item {idx} must contain at least one head")
        for head_idx, head in enumerate(item["heads"]):
            for field in ["bbox_norm", "gazex_norm", "gazey_norm", "inout"]:
                if field not in head:
                    raise SchemaError(f"Item {idx} head {head_idx} is missing field {field!r}")
            if "gooreal_index" in head and (
                "gooreal_coord_width" not in head or "gooreal_coord_height" not in head
            ):
                raise SchemaError(
                    f"Item {idx} head {head_idx} looks like stale GOO-real preprocessing. "
                    "Regenerate it with data_prep/preprocess_gooreal.py so coordinates use the 640x480 GOO-real canvas."
                )
            bbox = head["bbox_norm"]
            if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1) or not (0 <= bbox[1] < bbox[3] <= 1):
                raise SchemaError(f"Item {idx} head {head_idx} has invalid bbox_norm: {bbox}")
    return True


def synthetic_records():
    return [
        {
            "path": "finalrealdatasetImgsV3Sparsed/8/cam1/cam00001_img00524.jpg",
            "heads": [
                {
                    "bbox": [156.0, 115.0, 212.0, 188.0],
                    "bbox_norm": [156.0 / 640.0, 115.0 / 480.0, 212.0 / 640.0, 188.0 / 480.0],
                    "gazex": [382.0],
                    "gazey": [329.0],
                    "gazex_norm": [382.0 / 640.0],
                    "gazey_norm": [329.0 / 480.0],
                    "inout": 1,
                    "head_id": 0,
                    "gooreal_index": 0,
                    "gooreal_coord_width": 640.0,
                    "gooreal_coord_height": 480.0,
                }
            ],
            "num_heads": 1,
            "width": 1920,
            "height": 1080,
        }
    ]


class GoorealImageStore:
    def __init__(self, data_path, zip_path=None, zip_cache_dir=None):
        self.data_path = data_path
        self.zip_path = zip_path or os.path.join(data_path, "gooreal.zip")
        self.zip_cache_dir = zip_cache_dir or os.path.join(data_path, ".gooreal_zip_cache")
        self._inner_zips = {}

    def open(self, rel_path):
        normalized = str(rel_path).replace("\\", "/").lstrip("/")
        fs_path = os.path.join(self.data_path, normalized)
        if os.path.exists(fs_path):
            return Image.open(fs_path).convert("RGB")

        zip_file = self._open_inner_zip_for_path(normalized)
        try:
            with zip_file.open(normalized) as file:
                return Image.open(io.BytesIO(file.read())).convert("RGB")
        except KeyError as exc:
            raise FileNotFoundError(
                f"Image {normalized!r} was not found under {self.data_path!r} or inside the nested GOO-real zip."
            ) from exc

    def close(self):
        for zip_file in self._inner_zips.values():
            zip_file.close()
        self._inner_zips.clear()

    def _open_inner_zip_for_path(self, rel_path):
        prefix = rel_path.split("/", 1)[0]
        inner_name = f"{prefix}.zip"
        if inner_name not in self._inner_zips:
            inner_path = self._ensure_inner_zip(inner_name)
            self._inner_zips[inner_name] = zipfile.ZipFile(inner_path)
        return self._inner_zips[inner_name]

    def _ensure_inner_zip(self, inner_name):
        direct_path = os.path.join(self.data_path, inner_name)
        if os.path.exists(direct_path):
            return direct_path

        cached_path = os.path.join(self.zip_cache_dir, inner_name)
        if os.path.exists(cached_path):
            return cached_path

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(
                f"Neither extracted image files nor GOO-real zip were found. Missing zip: {self.zip_path}"
            )

        os.makedirs(self.zip_cache_dir, exist_ok=True)
        print(f"Extracting nested image archive {inner_name} to {cached_path}")
        with zipfile.ZipFile(self.zip_path) as outer_zip:
            if inner_name not in outer_zip.namelist():
                raise FileNotFoundError(
                    f"Nested archive {inner_name!r} not found in {self.zip_path!r}. "
                    f"Available entries: {outer_zip.namelist()}"
                )
            with outer_zip.open(inner_name) as source, open(cached_path, "wb") as target:
                shutil.copyfileobj(source, target)
        return cached_path


class GoorealDataset:
    def __init__(self, records, image_store, transform_base, transform_spot):
        self.records = records
        self.image_store = image_store
        self.transform_base = transform_base
        self.transform_spot = transform_spot

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = self.image_store.open(item["path"])
        image_base = self.transform_base(image)
        image_spot = self.transform_spot(image)
        bboxes = [head["bbox_norm"] for head in item["heads"]]
        gazex = [head["gazex_norm"] for head in item["heads"]]
        gazey = [head["gazey_norm"] for head in item["heads"]]
        return image_base, image_spot, bboxes, gazex, gazey, item["height"], item["width"], item["path"]


def collate(batch):
    import torch

    image_base, image_spot, bboxes, gazex, gazey, heights, widths, paths = zip(*batch)
    return (
        torch.stack(image_base),
        torch.stack(image_spot),
        list(bboxes),
        list(gazex),
        list(gazey),
        list(heights),
        list(widths),
        list(paths),
    )


def summarize(values):
    return float(np.mean(values)) if values else None


def gooreal_auc(heatmap, gt_gazex, gt_gazey):
    from sklearn.metrics import roc_auc_score

    if isinstance(gt_gazex, (list, tuple, np.ndarray)):
        gt_gazex = gt_gazex[0]
    if isinstance(gt_gazey, (list, tuple, np.ndarray)):
        gt_gazey = gt_gazey[0]

    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    resized = np.asarray(
        Image.fromarray(heatmap).resize((5, 5), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    target = np.zeros((5, 5), dtype=np.int32)
    x = int(float(gt_gazex) * 5)
    y = int(float(gt_gazey) * 5)
    x = max(0, min(4, x))
    y = max(0, min(4, y))
    target[y, x] = 1
    return roc_auc_score(target.reshape(-1), resized.reshape(-1))


def write_results(output_path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote JSON results to {output_path}")


def write_csv(csv_path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows = []
    for method, metrics in payload["results"].items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    fieldnames = ["method", "auc", "l2", "avg_l2", "min_l2"]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV results to {csv_path}")


def load_records(json_path, max_items=None):
    with open(json_path, "r") as file:
        records = json.load(file)
    validate_records(records)
    return records[:max_items] if max_items is not None else records


def run_schema_check(json_path=None):
    if json_path:
        records = load_records(json_path)
        source = json_path
    else:
        records = synthetic_records()
        validate_records(records)
        source = "synthetic sample"
    num_heads = sum(len(item["heads"]) for item in records)
    print(f"Schema check OK: {source} ({len(records)} images, {num_heads} heads)")


def run_dry_run(args):
    if args.json_path:
        records = load_records(args.json_path, max_items=args.max_items)
        source = args.json_path
    else:
        records = synthetic_records()
        source = "synthetic sample"
    num_heads = sum(len(item["heads"]) for item in records)
    print("Dry run OK.")
    print(f"Source: {source}")
    print(f"Images: {len(records)}")
    print(f"Heads: {num_heads}")
    print("No models or checkpoints were loaded.")


def evaluate(args):
    import torch
    from tqdm import tqdm

    from gazelle.model import gazelle_dinov3_vitb16 as gazelle_spot
    from gazelle.model_v0 import gazelle_dinov3_vitb16 as gazelle_baseline
    from gazelle.utils import gazefollow_l2

    if not args.json_path:
        raise SystemExit("--json_path is required for evaluation")
    if not args.base_ckpt or not args.spot_ckpt:
        raise SystemExit("--base_ckpt and --spot_ckpt are required for evaluation")

    records = load_records(args.json_path, max_items=args.max_items)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Running on {device}")
    print(f"Loaded {len(records)} GOO-real images from {args.json_path}")

    model_base, transform_base = gazelle_baseline()
    model_base.load_gazelle_state_dict(torch.load(args.base_ckpt, map_location="cpu", weights_only=True))
    model_base.to(device).eval()

    model_spot, transform_spot = gazelle_spot(sasa=True, ggsf=True, aux=False)
    model_spot.load_gazelle_state_dict(torch.load(args.spot_ckpt, map_location="cpu", weights_only=True))
    model_spot.to(device).eval()

    image_store = GoorealImageStore(args.data_path, zip_path=args.zip_path, zip_cache_dir=args.zip_cache_dir)
    dataset = GoorealDataset(records, image_store, transform_base, transform_spot)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    metrics = {
        "Baseline (DINOv3 last-layer)": {"auc": [], "l2": [], "avg_l2": [], "min_l2": []},
        "GazeSpot": {"auc": [], "l2": [], "avg_l2": [], "min_l2": []},
    }

    try:
        with torch.no_grad():
            for images_base, images_spot, bboxes, gazex, gazey, heights, widths, _paths in tqdm(
                dataloader, desc="Evaluating GOO-real"
            ):
                out_base = model_base({"images": images_base.to(device), "bboxes": bboxes})
                out_spot = model_spot({"images": images_spot.to(device), "bboxes": bboxes})

                for i in range(images_base.shape[0]):
                    for j in range(len(bboxes[i])):
                        auc_b = gooreal_auc(out_base["heatmap"][i][j], gazex[i][j], gazey[i][j])
                        avg_l2_b, min_l2_b = gazefollow_l2(out_base["heatmap"][i][j], gazex[i][j], gazey[i][j])
                        metrics["Baseline (DINOv3 last-layer)"]["auc"].append(auc_b)
                        metrics["Baseline (DINOv3 last-layer)"]["l2"].append(avg_l2_b)
                        metrics["Baseline (DINOv3 last-layer)"]["avg_l2"].append(avg_l2_b)
                        metrics["Baseline (DINOv3 last-layer)"]["min_l2"].append(min_l2_b)

                        auc_s = gooreal_auc(out_spot["heatmap"][i][j], gazex[i][j], gazey[i][j])
                        avg_l2_s, min_l2_s = gazefollow_l2(out_spot["heatmap"][i][j], gazex[i][j], gazey[i][j])
                        metrics["GazeSpot"]["auc"].append(auc_s)
                        metrics["GazeSpot"]["l2"].append(avg_l2_s)
                        metrics["GazeSpot"]["avg_l2"].append(avg_l2_s)
                        metrics["GazeSpot"]["min_l2"].append(min_l2_s)
    finally:
        image_store.close()

    results = {
        method: {name: summarize(values) for name, values in method_metrics.items()}
        for method, method_metrics in metrics.items()
    }
    payload = {
        "dataset": "GOO-real",
        "json_path": args.json_path,
        "data_path": args.data_path,
        "num_images": len(records),
        "num_heads": sum(len(item["heads"]) for item in records),
        "metrics": ["auc", "l2", "avg_l2", "min_l2"],
        "results": results,
    }

    print(json.dumps(payload["results"], indent=2))
    if args.output:
        write_results(args.output, payload)
    if args.csv_output:
        write_csv(args.csv_output, payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline and GazeSpot on preprocessed GOO-real.")
    parser.add_argument("--data_path", type=str, default="/newhome/fb/dataset/gooreal_data")
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--base_ckpt", type=str, default=None)
    parser.add_argument("--spot_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv_output", type=str, default=None)
    parser.add_argument("--zip_path", type=str, default=None)
    parser.add_argument("--zip_cache_dir", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--schema_check", action="store_true", help="Validate preprocessed JSON without loading models.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned evaluation inputs without loading models.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.schema_check:
        run_schema_check(args.json_path)
        return
    if args.dry_run:
        run_dry_run(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
