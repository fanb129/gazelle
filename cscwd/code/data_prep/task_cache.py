"""FP32 post-Transformer person features from the frozen GazeSpot-B model.

The model sees only RGB and current-frame ground-truth head boxes. Labels are
kept in the row index for supervision/scoring and never passed to the model.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class FrozenTaskFeatureExtractor(nn.Module):
    """Capture the original heads' exact inputs without rewriting GazeSpot."""

    def __init__(self, model):
        super().__init__()
        self.model = model.float().requires_grad_(False)
        self.features = {}
        self.hooks = [
            model.heatmap_head.register_forward_pre_hook(self._capture("x")),
            model.inout_head.register_forward_pre_hook(self._capture("z")),
        ]
        self.eval()

    def _capture(self, name):
        def capture(module, inputs):
            self.features[name] = inputs[0].detach().float()
        return capture

    def train(self, mode=True):
        return super().train(False)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()

    @torch.no_grad()
    def forward(self, images, bboxes):
        self.features = {}
        outputs = self.model({"images": images.float(), "bboxes": bboxes})
        return {**self.features, "heatmap": torch.cat(outputs["heatmap"]),
                "inout": torch.cat(outputs["inout"])}


def index_tracks(tracks_path, split, data_root):
    """Flatten the existing native-person track JSON with explicit array rows."""
    with Path(tracks_path).open() as stream:
        sequences = json.load(stream)
    tracks = sorted((track for sequence in sequences for track in sequence["tracks"]),
                    key=lambda track: track["track_id"])
    ids = [track["track_id"] for track in tracks]
    if len(set(ids)) != len(ids) or any(not value.startswith(split + "/") for value in ids):
        raise ValueError("Track IDs must be unique and belong to the requested official split")
    indexed = []
    row = 0
    for track in tracks:
        frames = sorted(track["frames"], key=lambda frame: frame["frame_index"])
        frame_ids = [int(frame["frame_index"]) for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"Duplicate original frame number in {track['track_id']}")
        mapped = []
        start = row
        for frame in frames:
            if int(Path(frame["path"]).stem) != int(frame["frame_index"]):
                raise ValueError(f"Original frame number disagrees with image path: {frame['path']}")
            mapped.append({
                "row": row, "frame_index": int(frame["frame_index"]),
                "path": frame["path"], "image_path": str((Path(data_root) / frame["path"]).resolve()),
                "person_id": str(track["person_id"]), "bbox_norm": frame["bbox_norm"],
                "gazex_norm": frame["gazex_norm"], "gazey_norm": frame["gazey_norm"],
                "inout": int(frame["inout"]),
            })
            row += 1
        indexed.append({"track_id": track["track_id"], "person_id": str(track["person_id"]),
                        "source_csv": track["source_csv"], "start": start, "stop": row,
                        "frames": mapped})
    if not row:
        raise ValueError("The requested split has no person frames")
    return indexed, row


class PersonImageDataset(Dataset):
    """Group all queried persons per image, running the backbone once per RGB."""

    def __init__(self, tracks, data_root, transform):
        people = defaultdict(list)
        for track in tracks:
            for frame in track["frames"]:
                people[frame["path"]].append((frame["row"], frame["bbox_norm"]))
        self.images = sorted(people.items())
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, people = self.images[index]
        with Image.open(self.data_root / path) as image:
            rgb = self.transform(image.convert("RGB"))
        return rgb, [bbox for row, bbox in people], [row for row, bbox in people]


def collate_images(batch):
    images, bboxes, rows = zip(*batch)
    return torch.stack(images), list(bboxes), [row for group in rows for row in group]


class TaskFeatureCache:
    """Read-only mmap access; labels and row-to-track mappings live in index."""

    def __init__(self, root):
        self.root = Path(root)
        with (self.root / "index.json").open() as stream:
            self.index = json.load(stream)
        self.x = np.load(self.root / self.index["arrays"]["x"]["path"], mmap_mode="r")
        self.z = np.load(self.root / self.index["arrays"]["z"]["path"], mmap_mode="r")
        n = self.index["num_frames"]
        if (self.x.shape != (n, 256, 32, 32) or self.z.shape != (n, 256)
                or self.x.dtype != np.float32 or self.z.dtype != np.float32):
            raise ValueError("Task cache must contain aligned FP32 X[N,256,32,32] and z[N,256]")

    def __len__(self):
        return self.index["num_frames"]


def extract_task_cache(model, transform, data_root, tracks_path, checkpoint, output_dir,
                       split, device="cuda:0", batch_size=4, num_workers=4,
                       source_metadata=None):
    """Write a fresh split cache; index.json is published only after completion."""
    data_root, tracks_path = Path(data_root).resolve(), Path(tracks_path).resolve()
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    tracks, count = index_tracks(tracks_path, split, data_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    x = np.lib.format.open_memmap(output_dir / "x.npy", mode="w+", dtype=np.float32,
                                  shape=(count, 256, 32, 32))
    z = np.lib.format.open_memmap(output_dir / "z.npy", mode="w+", dtype=np.float32,
                                  shape=(count, 256))
    extractor = FrozenTaskFeatureExtractor(model).to(device)
    dataset = PersonImageDataset(tracks, data_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=collate_images,
                        pin_memory=str(device).startswith("cuda"))
    completed = 0
    for batch_index, (images, bboxes, rows) in enumerate(loader):
        features = extractor(images.to(device, non_blocking=True), bboxes)
        x[rows] = features["x"].cpu().numpy()
        z[rows] = features["z"].cpu().numpy()
        completed += len(rows)
        if batch_index % 100 == 0:
            print(f"CACHE {split}: {completed}/{count} person frames", flush=True)
    extractor.close()
    del features, extractor
    x.flush()
    z.flush()
    del x, z
    torch.save({"heatmap_head": {key: value.cpu() for key, value in model.heatmap_head.state_dict().items()},
                "inout_head": {key: value.cpu() for key, value in model.inout_head.state_dict().items()}},
               output_dir / "heads.pt")
    print(f"CACHE {split}: hashing completed artifacts", flush=True)
    arrays = {name: {"path": f"{name}.npy", "sha256": sha256(output_dir / f"{name}.npy"),
                     "dtype": "float32", "shape": [count, *shape]}
              for name, shape in (("x", [256, 32, 32]), ("z", [256]))}
    index = {
        "schema_version": 1, "feature_type": "post_person_transformer_original_head_inputs",
        "split": split, "num_frames": count, "num_images": len(dataset),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source": {"data_root": str(data_root), "tracks_path": str(tracks_path),
                   "tracks_sha256": sha256(tracks_path), "checkpoint": str(checkpoint),
                   "checkpoint_sha256": sha256(checkpoint), **(source_metadata or {})},
        "protocol": {"input": "RGB_and_current_frame_raw_ground_truth_head_box",
                     "augmentation": "none", "frozen_model_mode": "eval", "precision": "float32",
                     "frame_sampling": "all_official_person_frames", "gt_usage": "supervision_and_scoring_only"},
        "arrays": arrays, "heads": {"path": "heads.pt", "sha256": sha256(output_dir / "heads.pt")},
        "tracks": tracks,
    }
    write_json(output_dir / "index.json", index)
    print(f"CACHE {split}: complete: {count} person frames -> {output_dir}", flush=True)
    return TaskFeatureCache(output_dir)


def build_cache(data_root, tracks_path, checkpoint, output_dir, split, device="cuda:0",
                batch_size=4, num_workers=4):
    """Build the production GazeSpot-B cache from the supplied VAT checkpoint."""
    from gazelle.model import gazelle_dinov3_vitb16_inout

    model, transform = gazelle_dinov3_vitb16_inout(sasa=True, ggsf=True, aux=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {key for key in model.state_dict() if not key.startswith("backbone.")}
    provided = {key for key in state if not key.startswith("backbone.")}
    if provided != expected:
        raise ValueError(f"GazeSpot decoder checkpoint mismatch: missing={sorted(expected - provided)}, "
                         f"unexpected={sorted(provided - expected)}")
    # Full backbone is initialized by the existing factory from its local file.
    model.load_state_dict({**model.state_dict(), **state}, strict=True)
    backbone_path = REPO_ROOT / "checkpoints/dinov3_vitb16_pretrain.pth"
    source = {"model": "gazelle_dinov3_vitb16_inout", "fusion": "sasa", "spatial_prior": "ggsf",
              "backbone_checkpoint": str(backbone_path), "backbone_sha256": sha256(backbone_path)}
    return extract_task_cache(model, transform, data_root, tracks_path, checkpoint, output_dir,
                              split, device, batch_size, num_workers, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    build_cache(**vars(args))


if __name__ == "__main__":
    # The module can also be launched directly from the repository root.
    sys.path.insert(0, str(REPO_ROOT))
    main()
