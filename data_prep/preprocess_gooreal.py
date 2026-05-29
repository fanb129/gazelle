import argparse
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np


SPLIT_CONFIGS = {
    "test": ("testrealhumansNew.pickle", "finalrealdatasetImgsV3"),
    "test_sparse": ("testrealhumansSparsedNew.pickle", "finalrealdatasetImgsV3Sparsed"),
    "val": ("valrealhumansNew.pickle", "finalrealdatasetImgsV3"),
    "oneshot": ("oneshotrealhumansNew.pickle", "finalrealdatasetImgsV3"),
}


class SchemaError(ValueError):
    pass


def normalize_path(path):
    return str(path).replace("\\", "/").lstrip("/")


def as_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Field {field_name!r} must be numeric, got {value!r}") from exc


def clamp(value, low, high):
    return max(low, min(high, value))


def fix_bbox_order(bbox):
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return [xmin, ymin, xmax, ymax]


def clip_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = fix_bbox_order(bbox)
    xmin = clamp(xmin, 0.0, float(width))
    ymin = clamp(ymin, 0.0, float(height))
    xmax = clamp(xmax, 0.0, float(width))
    ymax = clamp(ymax, 0.0, float(height))
    if xmax <= xmin or ymax <= ymin:
        raise SchemaError(f"Invalid bbox after clipping: {[xmin, ymin, xmax, ymax]}")
    return [xmin, ymin, xmax, ymax]


def make_fallback_head_bbox(hx, hy, width, height):
    size = max(16.0, min(float(width), float(height)) * 0.04)
    half = size / 2.0
    xmin = clamp(hx - half, 0.0, float(width) - 1.0)
    ymin = clamp(hy - half, 0.0, float(height) - 1.0)
    xmax = clamp(hx + half, xmin + 1.0, float(width))
    ymax = clamp(hy + half, ymin + 1.0, float(height))
    return [xmin, ymin, xmax, ymax]


def point_in_bbox(x, y, bbox):
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax


def bbox_center_distance(x, y, bbox):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5


def select_head_bbox(record, warnings):
    width = as_float(record.get("width"), "width")
    height = as_float(record.get("height"), "height")
    hx = as_float(record.get("hx"), "hx")
    hy = as_float(record.get("hy"), "hy")

    ann = record.get("ann")
    if not isinstance(ann, dict):
        raise SchemaError("Field 'ann' must be a dict")
    if "bboxes" not in ann:
        raise SchemaError("Field 'ann.bboxes' is missing")

    bboxes = np.asarray(ann["bboxes"], dtype=float)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4 or len(bboxes) == 0:
        raise SchemaError(f"Field 'ann.bboxes' must have shape (N, 4), got {bboxes.shape}")

    labels = np.asarray(ann.get("labels", []))
    candidates = []
    for idx, bbox in enumerate(bboxes):
        try:
            clipped = clip_bbox(bbox, width, height)
        except SchemaError:
            continue
        label = int(labels[idx]) if idx < len(labels) else None
        contains_head = point_in_bbox(hx, hy, clipped)
        distance = bbox_center_distance(hx, hy, clipped)
        candidates.append((idx, label, clipped, contains_head, distance))

    if not candidates:
        warnings.append("No valid boxes in ann.bboxes; using synthetic box around hx/hy.")
        return make_fallback_head_bbox(hx, hy, width, height), "synthetic_hxhy"

    label_25 = [item for item in candidates if item[1] == 25 and item[3]]
    if label_25:
        chosen = min(label_25, key=lambda item: item[4])
        return chosen[2], f"label25_contains_hxhy:index={chosen[0]}"

    containing = [item for item in candidates if item[3]]
    if containing:
        chosen = min(containing, key=lambda item: item[4])
        warnings.append(
            f"No label-25 head box contained hx/hy; using containing box index {chosen[0]}."
        )
        return chosen[2], f"contains_hxhy:index={chosen[0]}"

    chosen = min(candidates, key=lambda item: item[4])
    if chosen[4] <= max(width, height) * 0.08:
        warnings.append(
            f"No box contained hx/hy; using nearest box index {chosen[0]} at distance {chosen[4]:.2f}."
        )
        return chosen[2], f"nearest_hxhy:index={chosen[0]}"

    warnings.append("No plausible head box found; using synthetic box around hx/hy.")
    return make_fallback_head_bbox(hx, hy, width, height), "synthetic_hxhy"


def convert_record(record, index, image_prefix):
    required = ["filename", "width", "height", "gaze_cx", "gaze_cy", "hx", "hy", "ann"]
    missing = [field for field in required if field not in record]
    if missing:
        raise SchemaError(f"Record {index} is missing fields: {', '.join(missing)}")

    width = int(as_float(record["width"], "width"))
    height = int(as_float(record["height"], "height"))
    if width <= 0 or height <= 0:
        raise SchemaError(f"Record {index} has invalid image size: {width}x{height}")

    gaze_x = as_float(record["gaze_cx"], "gaze_cx")
    gaze_y = as_float(record["gaze_cy"], "gaze_cy")
    if not (0 <= gaze_x <= width and 0 <= gaze_y <= height):
        raise SchemaError(f"Record {index} has out-of-image gaze point: ({gaze_x}, {gaze_y})")

    warnings = []
    bbox = select_head_bbox(record, warnings)
    bbox_pixels, bbox_source = bbox
    bbox_pixels = clip_bbox(bbox_pixels, width, height)

    filename = normalize_path(record["filename"])
    image_path = normalize_path(Path(image_prefix) / filename) if image_prefix else filename

    head = {
        "bbox": bbox_pixels,
        "bbox_norm": [
            bbox_pixels[0] / float(width),
            bbox_pixels[1] / float(height),
            bbox_pixels[2] / float(width),
            bbox_pixels[3] / float(height),
        ],
        "gazex": [gaze_x],
        "gazey": [gaze_y],
        "gazex_norm": [gaze_x / float(width)],
        "gazey_norm": [gaze_y / float(height)],
        "inout": 1,
        "head_id": 0,
        "gooreal_index": index,
        "gooreal_bbox_source": bbox_source,
        "gooreal_gaze_idx": record.get("gazeIdx"),
        "gooreal_gaze_item": record.get("gaze_item"),
        "gooreal_occluded": bool(record.get("occluded", False)),
    }

    item = {
        "path": image_path,
        "heads": [head],
        "num_heads": 1,
        "width": width,
        "height": height,
        "gooreal_source_filename": filename,
    }
    if warnings:
        item["gooreal_warnings"] = warnings
    return item


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GOO-real pickle not found: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    if not isinstance(data, list):
        raise SchemaError(f"GOO-real pickle must contain a list, got {type(data).__name__}")
    return data


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
            bbox = head["bbox_norm"]
            if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1) or not (0 <= bbox[1] < bbox[3] <= 1):
                raise SchemaError(f"Item {idx} head {head_idx} has invalid bbox_norm: {bbox}")
    return True


def inspect_pickle(path, max_items=1):
    data = load_pickle(path)
    print(f"Loaded {len(data)} records from {path}")
    for idx, record in enumerate(data[:max_items]):
        print(f"Record {idx}: type={type(record).__name__}")
        if not isinstance(record, dict):
            continue
        print(f"  keys={list(record.keys())}")
        ann = record.get("ann")
        if isinstance(ann, dict):
            print(f"  ann.keys={list(ann.keys())}")
            bboxes = np.asarray(ann.get("bboxes", []))
            labels = np.asarray(ann.get("labels", []))
            print(f"  ann.bboxes.shape={bboxes.shape}")
            print(f"  ann.labels.shape={labels.shape}")
        for field in ["filename", "width", "height", "gaze_cx", "gaze_cy", "hx", "hy", "gazeIdx", "gaze_item"]:
            print(f"  {field}={record.get(field)!r}")


def dry_run_sample():
    sample = {
        "filename": "8\\cam1\\cam00001_img00524.jpg",
        "width": 1920,
        "height": 1080,
        "ann": {
            "bboxes": np.array(
                [
                    [378.0, 316.0, 392.0, 340.0],
                    [156.0, 115.0, 212.0, 188.0],
                ]
            ),
            "labels": np.array([17, 25]),
        },
        "gaze_item": 17,
        "gazeIdx": 0,
        "gaze_cx": 382,
        "gaze_cy": 329,
        "hx": 199,
        "hy": 156,
        "seg": np.zeros((0, 2), dtype=np.float32),
        "occluded": False,
    }
    converted = [convert_record(sample, 0, "finalrealdatasetImgsV3Sparsed")]
    validate_records(converted)
    return converted


def check_image_paths(records, data_path, zip_path=None, max_missing=10):
    missing = []
    for item in records:
        fs_path = os.path.join(data_path, item["path"])
        if os.path.exists(fs_path):
            continue
        missing.append(item["path"])
        if len(missing) >= max_missing:
            break
    if missing:
        hint = ""
        if zip_path and os.path.exists(zip_path):
            hint = " Images may still be available inside the nested GOO-real zip for eval_gooreal.py."
        raise FileNotFoundError(
            "Missing image paths under data_path: " + ", ".join(missing) + "." + hint
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GOO-real annotations for GazeSpot evaluation.")
    parser.add_argument("--data_path", type=str, default="/newhome/fb/dataset/gooreal_data")
    parser.add_argument("--split", choices=sorted(SPLIT_CONFIGS), default="test")
    parser.add_argument("--pickle_path", type=str, default=None)
    parser.add_argument("--image_prefix", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--inspect", action="store_true", help="Print pickle keys and shapes without writing JSON.")
    parser.add_argument("--schema_check", action="store_true", help="Validate an existing preprocessed JSON file.")
    parser.add_argument("--dry_run", action="store_true", help="Run conversion on a synthetic sample without dataset access.")
    parser.add_argument("--check_images", action="store_true", help="Require converted image files to exist under data_path.")
    return parser.parse_args()


def main():
    args = parse_args()
    default_pickle, default_prefix = SPLIT_CONFIGS[args.split]
    pickle_path = args.pickle_path or os.path.join(args.data_path, default_pickle)
    image_prefix = args.image_prefix if args.image_prefix is not None else default_prefix

    if args.dry_run:
        converted = dry_run_sample()
        print("Dry run OK. Example converted item:")
        print(json.dumps(converted[0], indent=2))
        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w") as file:
                json.dump(converted, file, indent=2)
            print(f"Wrote dry-run JSON to {args.output_json}")
        return

    if args.schema_check:
        if not args.output_json:
            raise SystemExit("--schema_check requires --output_json pointing to a preprocessed JSON file")
        with open(args.output_json, "r") as file:
            records = json.load(file)
        validate_records(records)
        print(f"Schema check OK: {args.output_json} ({len(records)} items)")
        return

    if args.inspect:
        inspect_pickle(pickle_path)
        return

    if not args.output_json:
        raise SystemExit("--output_json is required unless --dry_run, --schema_check, or --inspect is used")

    raw_records = load_pickle(pickle_path)
    if args.max_items is not None:
        raw_records = raw_records[: args.max_items]

    converted = []
    warning_count = 0
    for idx, record in enumerate(raw_records):
        try:
            item = convert_record(record, idx, image_prefix)
        except SchemaError as exc:
            raise SchemaError(f"Failed to convert record {idx}: {exc}") from exc
        warning_count += len(item.get("gooreal_warnings", []))
        converted.append(item)

    validate_records(converted)
    if args.check_images:
        check_image_paths(converted, args.data_path, os.path.join(args.data_path, "gooreal.zip"))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as file:
        json.dump(converted, file)

    print(f"Wrote {len(converted)} GOO-real items to {args.output_json}")
    print(f"Split: {args.split}")
    print(f"Pickle: {pickle_path}")
    print(f"Image prefix: {image_prefix}")
    print(f"Items with conversion warnings: {warning_count}")


if __name__ == "__main__":
    main()
