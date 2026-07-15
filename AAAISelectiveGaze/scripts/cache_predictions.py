"""Cache final and fixed-probe predictions using a frozen Gazelle checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _create_gooreal_image_store(data_root: Path, zip_path=None, zip_cache_dir=None):
    """Construct the repository's nested-zip-aware GOO-Real image store."""

    from scripts.eval_gooreal import GoorealImageStore

    return GoorealImageStore(
        str(data_root),
        zip_path=zip_path,
        zip_cache_dir=zip_cache_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gazefollow", "vat", "gooreal"), required=True)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--zip-path",
        default=None,
        help="GOO-Real outer zip; defaults to <data-path>/gooreal.zip.",
    )
    parser.add_argument(
        "--zip-cache-dir",
        default=None,
        help="Directory for extracted nested GOO-Real image archives.",
    )
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    parser.add_argument("--fusion", default="raw_concat")
    parser.add_argument("--spatial-prior", default="none")
    parser.add_argument("--selected-layers", default=None)
    parser.add_argument("--output", required=True)
    return parser


def _load_frames(path: Path, dataset: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("preprocessed JSON must contain a list")
    frames: list[dict[str, Any]] = []
    if dataset == "vat" and payload and "frames" in payload[0]:
        for sequence_index, sequence in enumerate(payload):
            sequence_id = str(
                sequence.get("sequence_id", sequence.get("path", f"sequence-{sequence_index}"))
            )
            for frame in sequence.get("frames", []):
                item = dict(frame)
                item["sequence_id"] = sequence_id
                frames.append(item)
    else:
        frames = [dict(frame) for frame in payload]
    return [frame for frame in frames if frame.get("heads")]


def _checkpoint_payload(torch, path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("cache_predictions requires the server PyTorch environment") from exc

    from torch.utils.data import DataLoader, Dataset

    from AAAISelectiveGaze.data.prediction_cache import (
        checkpoint_sha256,
        make_sample_id,
        save_prediction_cache,
    )
    from AAAISelectiveGaze.models.layer_probe import FixedLayerProbes
    from AAAISelectiveGaze.models.selective_gazelle import SelectiveGazelle
    from gazelle.model import get_gazelle_model

    json_path = Path(args.json_path)
    frames = _load_frames(json_path, args.dataset)
    data_root = Path(args.data_path)
    split_name = args.split or json_path.stem.replace("_preprocessed", "")

    image_store = None
    effective_num_workers = args.num_workers
    if args.dataset == "gooreal":
        image_store = _create_gooreal_image_store(
            data_root,
            args.zip_path,
            args.zip_cache_dir,
        )
        # GoorealImageStore lazily opens an inner zip and owns open file handles.
        # Keeping reads in the main process avoids duplicate/racing extraction and
        # makes missing archive members surface with their original traceback.
        if effective_num_workers != 0:
            print(
                "GOO-Real nested-zip input detected; overriding "
                f"--num-workers {effective_num_workers} with 0."
            )
            effective_num_workers = 0

    predictor, transform = get_gazelle_model(
        args.model,
        spatial_prior=args.spatial_prior,
        fusion=args.fusion,
        selected_layers=args.selected_layers,
    )
    base_payload = _checkpoint_payload(torch, args.base_checkpoint)
    if isinstance(base_payload, dict) and "state_dict" in base_payload:
        base_payload = base_payload["state_dict"]
    predictor.load_gazelle_state_dict(base_payload)

    probe_payload = _checkpoint_payload(torch, args.probe_checkpoint)
    required_metadata = ("state_dict", "layers", "in_channels", "hidden_channels", "output_size")
    missing = [key for key in required_metadata if key not in probe_payload]
    if missing:
        raise ValueError(f"probe checkpoint is missing metadata: {missing}")
    probes = FixedLayerProbes(
        in_channels=probe_payload["in_channels"],
        layers=probe_payload["layers"],
        hidden_channels=probe_payload["hidden_channels"],
        output_size=tuple(probe_payload["output_size"]),
    )
    probes.load_state_dict(probe_payload["state_dict"])
    model = SelectiveGazelle(predictor, probes, layers=probe_payload["layers"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    class FrameDataset(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            frame = frames[index]
            if image_store is not None:
                image = image_store.open(frame["path"])
            else:
                image_path = data_root / frame["path"]
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"image {frame['path']!r} was not found under {str(data_root)!r}"
                    )
                image = Image.open(image_path).convert("RGB")
            return transform(image), frame

    def collate(items):
        images, metadata = zip(*items)
        return torch.stack(images), list(metadata)

    loader = DataLoader(
        FrameDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        collate_fn=collate,
    )
    base_hash = checkpoint_sha256(args.base_checkpoint)
    probe_hash = checkpoint_sha256(args.probe_checkpoint)
    records: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for images, metadata in loader:
                bboxes = [[head.get("bbox_norm") for head in frame["heads"]] for frame in metadata]
                outputs = model({"images": images.to(device), "bboxes": bboxes})
                for image_index, frame in enumerate(metadata):
                    for person_index, head in enumerate(frame["heads"]):
                        xs = head.get("gazex_norm", [])
                        ys = head.get("gazey_norm", [])
                        gt_gaze = [[float(x), float(y)] for x, y in zip(xs, ys)]
                        inout_output = outputs.get("inout")
                        inout_probability = None
                        if inout_output is not None:
                            inout_probability = float(inout_output[image_index][person_index].cpu())
                        record = {
                            "sample_id": head.get(
                                "sample_id",
                                make_sample_id(args.dataset, frame["path"], person_index),
                            ),
                            "dataset": args.dataset,
                            "split": split_name,
                            "image_path": frame["path"],
                            "sequence_id": frame.get("sequence_id"),
                            "person_index": person_index,
                            "final_heatmap": outputs["heatmap"][image_index][person_index].cpu(),
                            "probe_heatmaps": {
                                str(layer): layer_outputs[image_index][person_index].cpu()
                                for layer, layer_outputs in outputs["probe_heatmap"].items()
                            },
                            "inout_probability": inout_probability,
                            "gt_gaze": gt_gaze,
                            "gt_inout": int(head.get("inout", 1)),
                            "bbox": head["bbox_norm"],
                            "checkpoint_hash": base_hash,
                            "probe_checkpoint_hash": probe_hash,
                        }
                        records.append(record)
    finally:
        if image_store is not None:
            image_store.close()
    output = save_prediction_cache(records, args.output)
    print(f"Saved {len(records)} per-person predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
