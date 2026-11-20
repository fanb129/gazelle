"""T2: cache frozen RGB GazeSpot VAT predictions in native person-track order."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from stag.cache import file_sha256, write_json, write_prediction_cache


class TrackFrameDataset(Dataset):
    """Expose only current RGB and head boxes to the single-frame model."""

    def __init__(self, sequences, data_path, transform):
        self.records = [(frame["path"], frame["bbox_norm"])
                        for sequence in sequences for track in sequence["tracks"]
                        for frame in track["frames"]]
        self.data_path = Path(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, bbox = self.records[index]
        with Image.open(self.data_path / path) as image:
            image = self.transform(image.convert("RGB"))
        return image, bbox


def collate_frames(batch):
    images, bboxes = zip(*batch)
    return torch.stack(images), [[bbox] for bbox in bboxes]


def load_frozen_checkpoint(model, checkpoint):
    """Use the existing head-only checkpoint contract, requiring all head keys."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {key for key in model.state_dict() if not key.startswith("backbone")}
    supplied = set(state)
    if expected != supplied:
        raise ValueError(f"Checkpoint head keys mismatch: missing={sorted(expected - supplied)}, "
                         f"unexpected={sorted(supplied - expected)}")
    model.load_gazelle_state_dict(state)
    model.requires_grad_(False)
    model.eval()


def prediction_batches(model, loader, device):
    with torch.inference_mode():
        for batch_number, (images, bboxes) in enumerate(loader, start=1):
            output = model({"images": images.to(device), "bboxes": bboxes})
            heatmaps = torch.cat(output["heatmap"], dim=0).cpu().numpy()
            inout = torch.cat(output["inout"], dim=0).cpu().numpy()
            yield heatmaps, inout
            if batch_number % 100 == 0:
                print(f"Cached {batch_number}/{len(loader)} batches", flush=True)


def record_provenance(path, report, report_path):
    cache = report["cache"]
    lines = [
        "", "## T2 单帧预测缓存（服务器运行）", "",
        f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
        f"- 模型：`{report['model']}`，GGSF + SASA，冻结推理；checkpoint："
        f"`{report['checkpoint']['path']}`；SHA256：`{report['checkpoint']['sha256']}`。",
        f"- 骨干预训练：`{report['backbone_checkpoint']['path']}`；"
        f"SHA256：`{report['backbone_checkpoint']['sha256']}`。",
        f"- 输入 track JSON：`{report['track_json']['path']}`；SHA256：`{report['track_json']['sha256']}`。",
        f"- 命令：`{report['command']}`。",
        f"- 覆盖：{cache['tracks']} tracks / {cache['unique_image_frames']} 图像帧 / "
        f"{cache['person_frames']} 人物帧；float32 热图、画内概率和 argmax 坐标原样读回通过。",
        f"- 小报告：`{report_path}`；缓存目录（服务器保留）：`{cache['path']}`。",
        f"- 缓存 index.json SHA256：`{cache['index_sha256']}`。",
    ]
    for artifact in cache["files"].values():
        lines.append(f"- 大文件：`{Path(cache['path']) / artifact['path']}`；SHA256：`{artifact['sha256']}`。")
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--track-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["gazelle_dinov3_vitb16_inout", "gazelle_dinov3_vitl16_inout"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="New server-only output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t2_single_frame_cache.json")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    started = time.perf_counter()
    with args.track_json.open() as stream:
        sequences = json.load(stream)
    backbone_name = args.model.removeprefix("gazelle_").removesuffix("_inout")
    backbone_checkpoint = REPO_ROOT / "checkpoints" / f"{backbone_name}_pretrain.pth"
    report = {
        "task": "T2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": file_sha256(Path(__file__)),
        "cache_code_sha256": file_sha256(REPO_ROOT / "cscwd/code/stag/cache.py"),
        "command": shlex.join([sys.executable, *sys.argv]), "model": args.model,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": file_sha256(args.checkpoint)},
        "backbone_checkpoint": {"path": str(backbone_checkpoint), "sha256": file_sha256(backbone_checkpoint)},
        "track_json": {"path": str(args.track_json.resolve()), "sha256": file_sha256(args.track_json)},
        "protocol": {"modality": "RGB", "boxes": "T1_current_frame_raw_ground_truth",
                     "spatial_prior": "ggsf", "fusion": "sasa", "aux": False,
                     "precision": "float32; no_autocast; tf32_disabled",
                     "temporal_context": "none", "labels_used_in_forward": False,
                     "all_parameters_frozen": True, "training": False},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "device": args.device, "batch_size": args.batch_size, "num_workers": args.num_workers},
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    report["runtime"]["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    from gazelle.model import get_gazelle_model

    model, transform = get_gazelle_model(args.model, use_sasa=True, use_ggsf=True, use_aux=False)
    load_frozen_checkpoint(model, args.checkpoint)
    model.to(device)
    dataset = TrackFrameDataset(sequences, args.data_path, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_frames)
    report["cache"] = write_prediction_cache(
        sequences, prediction_batches(model, loader, device), args.cache_dir,
        {key: report[key] for key in ("model", "checkpoint", "backbone_checkpoint", "track_json", "protocol")},
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["passed"] = report["cache"]["passed"]
    write_json(args.report, report)
    record_provenance(args.provenance, report, args.report.resolve())
    print(f"T2 verification passed: {report['cache']['tracks']} tracks, "
          f"{report['cache']['person_frames']} person frames; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
