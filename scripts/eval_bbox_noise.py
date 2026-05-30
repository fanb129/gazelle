import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gazelle.eval_utils import perturb_normalized_bbox


METHOD_BASELINE = "Baseline (DINOv3 last-layer)"
METHOD_SPOT = "GazeSpot"


class SchemaError(ValueError):
    pass


def synthetic_vat_records():
    return [
        {
            "path": "images/synthetic/000001.jpg",
            "width": 640,
            "height": 480,
            "heads": [
                {
                    "bbox_norm": [0.2, 0.2, 0.35, 0.45],
                    "gazex_norm": [0.75],
                    "gazey_norm": [0.55],
                    "inout": 1,
                },
                {
                    "bbox_norm": [0.55, 0.18, 0.7, 0.4],
                    "gazex_norm": [-1.0],
                    "gazey_norm": [-1.0],
                    "inout": 0,
                },
            ],
        }
    ]


def synthetic_gazefollow_records():
    return [
        {
            "path": "test/synthetic/000001.jpg",
            "width": 640,
            "height": 480,
            "heads": [
                {
                    "bbox_norm": [0.22, 0.18, 0.34, 0.42],
                    "gazex_norm": [0.64, 0.66],
                    "gazey_norm": [0.48, 0.5],
                    "inout": 1,
                }
            ],
        }
    ]


def resolve_json_path(data_path, json_path):
    if not json_path:
        return None
    return json_path if os.path.isabs(json_path) else os.path.join(data_path, json_path)


def flatten_vat_records(sequences):
    if not isinstance(sequences, list):
        raise SchemaError(f"VAT JSON must be a list, got {type(sequences).__name__}")
    records = []
    for seq_idx, seq in enumerate(sequences):
        if isinstance(seq, dict) and "frames" in seq:
            for frame_idx, frame in enumerate(seq["frames"]):
                item = dict(frame)
                item.setdefault("width", seq.get("width"))
                item.setdefault("height", seq.get("height"))
                records.append(item)
        elif isinstance(seq, dict) and "heads" in seq:
            records.append(seq)
        else:
            raise SchemaError(f"VAT item {seq_idx} is neither a sequence with frames nor a frame with heads")
    return records


def validate_records(records, dataset):
    if not isinstance(records, list):
        raise SchemaError(f"{dataset} records must be a list, got {type(records).__name__}")
    for idx, item in enumerate(records):
        for field in ["path", "heads"]:
            if field not in item:
                raise SchemaError(f"{dataset} item {idx} is missing field {field!r}")
        if dataset == "gazefollow":
            for field in ["width", "height"]:
                if field not in item:
                    raise SchemaError(f"GazeFollow item {idx} is missing field {field!r}")
        if not isinstance(item["heads"], list) or not item["heads"]:
            raise SchemaError(f"{dataset} item {idx} must contain at least one head")
        for head_idx, head in enumerate(item["heads"]):
            for field in ["bbox_norm", "gazex_norm", "gazey_norm"]:
                if field not in head:
                    raise SchemaError(f"{dataset} item {idx} head {head_idx} is missing field {field!r}")
            if dataset == "vat" and "inout" not in head:
                raise SchemaError(f"VAT item {idx} head {head_idx} is missing field 'inout'")
            perturb_normalized_bbox(head["bbox_norm"], 0, np.random.default_rng(0))
    return records


def load_records(dataset, data_path, json_path=None, max_items=None):
    resolved = resolve_json_path(data_path, json_path)
    if resolved is None:
        records = synthetic_vat_records() if dataset == "vat" else synthetic_gazefollow_records()
    else:
        with open(resolved, "r") as file:
            payload = json.load(file)
        records = flatten_vat_records(payload) if dataset == "vat" else payload
    validate_records(records, dataset)
    return records[:max_items] if max_items is not None else records


def summarize(values):
    return float(np.mean(values)) if values else None


def summarize_metrics(raw_metrics):
    return {name: summarize(values) for name, values in raw_metrics.items()}


class BBoxNoiseDataset:
    def __init__(self, records, data_path, transform_base, transform_spot, jitter_level, seed):
        self.records = records
        self.data_path = data_path
        self.transform_base = transform_base
        self.transform_spot = transform_spot
        self.jitter_level = jitter_level
        self.perturbed_bboxes = self._build_perturbed_bboxes(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = item["path"] if os.path.isabs(item["path"]) else os.path.join(self.data_path, item["path"])
        image = Image.open(image_path).convert("RGB")
        image_base = self.transform_base(image)
        image_spot = self.transform_spot(image)
        bboxes = self.perturbed_bboxes[idx]
        gazex = [head["gazex_norm"] for head in item["heads"]]
        gazey = [head["gazey_norm"] for head in item["heads"]]
        inout = [head.get("inout", 1) for head in item["heads"]]
        return image_base, image_spot, bboxes, gazex, gazey, inout, item.get("height"), item.get("width"), item["path"]

    def _build_perturbed_bboxes(self, seed):
        rng = np.random.default_rng(seed)
        return [
            [
                perturb_normalized_bbox(head["bbox_norm"], self.jitter_level, rng)
                for head in item["heads"]
            ]
            for item in self.records
        ]


def collate(batch):
    import torch

    image_base, image_spot, bboxes, gazex, gazey, inout, heights, widths, paths = zip(*batch)
    return (
        torch.stack(image_base),
        torch.stack(image_spot),
        list(bboxes),
        list(gazex),
        list(gazey),
        list(inout),
        list(heights),
        list(widths),
        list(paths),
    )


def run_schema_check(args):
    records = load_records(args.dataset, args.data_path, args.json_path, args.max_items)
    head_count = sum(len(item["heads"]) for item in records)
    source = resolve_json_path(args.data_path, args.json_path) or "synthetic sample"
    print(f"Schema check OK: {args.dataset} {source} ({len(records)} images/frames, {head_count} heads)")


def run_dry_run(args):
    records = load_records(args.dataset, args.data_path, args.json_path, args.max_items)
    head_count = sum(len(item["heads"]) for item in records)
    print("Dry run OK.")
    print(f"Dataset: {args.dataset}")
    print(f"Source: {resolve_json_path(args.data_path, args.json_path) or 'synthetic sample'}")
    print(f"Jitter levels: {format_jitter_levels(args.jitter_levels)}")
    print(f"Seed: {args.seed}")
    print(f"Images/frames: {len(records)}")
    print(f"Heads: {head_count}")
    print("No models or checkpoints were loaded.")


def format_jitter_levels(levels):
    return [int(level) if float(level).is_integer() else float(level) for level in levels]


def metric_names_for_dataset(dataset):
    if dataset == "vat":
        return ["auc", "l2", "inout_ap"]
    return ["auc", "avg_l2", "min_l2"]


def load_models(args, device):
    import torch

    if not args.base_ckpt or not args.spot_ckpt:
        raise SystemExit("--base_ckpt and --spot_ckpt are required for evaluation")
    if args.dataset == "vat":
        from gazelle.model import gazelle_dinov3_vitb16_inout as gazelle_spot
        from gazelle.model_v0 import gazelle_dinov3_vitb16_inout as gazelle_baseline

        model_base, transform_base = gazelle_baseline()
        model_spot, transform_spot = gazelle_spot(sasa=True, ggsf=True, aux=False)
    else:
        from gazelle.model import gazelle_dinov3_vitb16 as gazelle_spot
        from gazelle.model_v0 import gazelle_dinov3_vitb16 as gazelle_baseline

        model_base, transform_base = gazelle_baseline()
        model_spot, transform_spot = gazelle_spot(sasa=True, ggsf=True, aux=False)

    model_base.load_gazelle_state_dict(torch.load(args.base_ckpt, map_location="cpu", weights_only=True))
    model_spot.load_gazelle_state_dict(torch.load(args.spot_ckpt, map_location="cpu", weights_only=True))
    model_base.to(device).eval()
    model_spot.to(device).eval()
    return model_base, transform_base, model_spot, transform_spot


def evaluate(args):
    import torch
    from sklearn.metrics import average_precision_score
    from tqdm import tqdm

    from gazelle.utils import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2

    if not args.json_path:
        raise SystemExit("--json_path is required for evaluation")

    records = load_records(args.dataset, args.data_path, args.json_path, args.max_items)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model_base, transform_base, model_spot, transform_spot = load_models(args, device)
    print(f"Running bbox-noise evaluation on {device}")
    print(f"Loaded {len(records)} {args.dataset} images/frames from {resolve_json_path(args.data_path, args.json_path)}")

    output_rows = []
    raw_results = {}

    with torch.no_grad():
        for jitter_level in args.jitter_levels:
            dataset = BBoxNoiseDataset(records, args.data_path, transform_base, transform_spot, jitter_level, args.seed)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate,
                num_workers=args.num_workers,
            )
            metrics = {
                METHOD_BASELINE: {name: [] for name in metric_names_for_dataset(args.dataset)},
                METHOD_SPOT: {name: [] for name in metric_names_for_dataset(args.dataset)},
            }
            inout_gts = []
            inout_preds_base = []
            inout_preds_spot = []

            for images_base, images_spot, bboxes, gazex, gazey, inout, heights, widths, _paths in tqdm(
                dataloader,
                desc=f"Evaluating {args.dataset} jitter={jitter_level:g}%",
            ):
                out_base = model_base({"images": images_base.to(device), "bboxes": bboxes})
                out_spot = model_spot({"images": images_spot.to(device), "bboxes": bboxes})

                for i in range(images_base.shape[0]):
                    for j in range(len(bboxes[i])):
                        if args.dataset == "vat":
                            if inout[i][j] == 1:
                                metrics[METHOD_BASELINE]["auc"].append(
                                    vat_auc(out_base["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0])
                                )
                                metrics[METHOD_BASELINE]["l2"].append(
                                    vat_l2(out_base["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0])
                                )
                                metrics[METHOD_SPOT]["auc"].append(
                                    vat_auc(out_spot["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0])
                                )
                                metrics[METHOD_SPOT]["l2"].append(
                                    vat_l2(out_spot["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0])
                                )
                            inout_gts.append(inout[i][j])
                            inout_preds_base.append(out_base["inout"][i][j].item())
                            inout_preds_spot.append(out_spot["inout"][i][j].item())
                        else:
                            metrics[METHOD_BASELINE]["auc"].append(
                                gazefollow_auc(out_base["heatmap"][i][j], gazex[i][j], gazey[i][j], heights[i], widths[i])
                            )
                            avg_l2_b, min_l2_b = gazefollow_l2(out_base["heatmap"][i][j], gazex[i][j], gazey[i][j])
                            metrics[METHOD_BASELINE]["avg_l2"].append(avg_l2_b)
                            metrics[METHOD_BASELINE]["min_l2"].append(min_l2_b)

                            metrics[METHOD_SPOT]["auc"].append(
                                gazefollow_auc(out_spot["heatmap"][i][j], gazex[i][j], gazey[i][j], heights[i], widths[i])
                            )
                            avg_l2_s, min_l2_s = gazefollow_l2(out_spot["heatmap"][i][j], gazex[i][j], gazey[i][j])
                            metrics[METHOD_SPOT]["avg_l2"].append(avg_l2_s)
                            metrics[METHOD_SPOT]["min_l2"].append(min_l2_s)

            if args.dataset == "vat":
                metrics[METHOD_BASELINE]["inout_ap"].append(average_precision_score(inout_gts, inout_preds_base))
                metrics[METHOD_SPOT]["inout_ap"].append(average_precision_score(inout_gts, inout_preds_spot))

            jitter_key = f"{jitter_level:g}%"
            raw_results[jitter_key] = {
                METHOD_BASELINE: summarize_metrics(metrics[METHOD_BASELINE]),
                METHOD_SPOT: summarize_metrics(metrics[METHOD_SPOT]),
            }
            for method, method_metrics in raw_results[jitter_key].items():
                row = {"dataset": args.dataset, "jitter": jitter_key, "method": method}
                row.update(method_metrics)
                output_rows.append(row)

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "data_path": args.data_path,
            "json_path": resolve_json_path(args.data_path, args.json_path),
            "checkpoints": {
                "baseline": args.base_ckpt,
                "gazespot": args.spot_ckpt,
            },
            "seed": args.seed,
            "jitter_levels": format_jitter_levels(args.jitter_levels),
            "perturbation": {
                "box_format": "normalized [xmin, ymin, xmax, ymax]",
                "translation": "uniform offsets in [-jitter, jitter] times box width/height",
                "scale": "uniform independent width/height scale factors in [1-jitter, 1+jitter]",
                "clip": "[0, 1]",
                "min_size": 1e-4,
                "zero_jitter": "original boxes are used without perturbation",
            },
            "sample_count": len(records),
            "head_count": sum(len(item["heads"]) for item in records),
            "metric_names": metric_names_for_dataset(args.dataset),
        },
        "results": raw_results,
        "rows": output_rows,
    }

    print(json.dumps(raw_results, indent=2))
    if args.output:
        write_json(args.output, payload)
    if args.csv_output:
        write_csv(args.csv_output, output_rows, metric_names_for_dataset(args.dataset))


def write_json(output_path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote JSON results to {output_path}")


def write_csv(csv_path, rows, metric_names):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["dataset", "jitter", "method"] + metric_names
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV results to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate head-box noise robustness for VAT/Crowd or GazeFollow.")
    parser.add_argument("--dataset", choices=["vat", "gazefollow"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--base_ckpt", type=str, default=None)
    parser.add_argument("--spot_ckpt", type=str, default=None)
    parser.add_argument("--jitter_levels", type=float, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv_output", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--schema_check", action="store_true", help="Validate annotation JSON without loading models.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned evaluation inputs without loading models.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.schema_check:
        run_schema_check(args)
        return
    if args.dry_run:
        run_dry_run(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
