"""Train on all official train frames and score all official test frames per epoch."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import random
import shlex
import shutil
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from cscwd.code.data_prep.task_cache import TaskFeatureCache, sha256, write_json
from cscwd.code.stag.data import CachedChunks
from cscwd.code.stag.engine import checkpoint_rank, evaluate, train_epoch
from cscwd.code.stag.model import FrozenTaskHeads, TemporalModel


PROTOCOL = {
    "training_split": "complete_official_train",
    "evaluation_split": "complete_official_test_every_epoch",
    "validation_split": None,
    "checkpoint_selection": "test_auc_desc_then_l2_asc_then_inout_ap_desc_then_epoch_asc",
    "test_used_for_checkpoint_selection": True,
    "reported_columns": "one_selected_epoch_no_cross_epoch_column_mixing",
    "localization_population": "all_GT_in_frame_person_frames_regardless_of_predicted_inout",
    "inout_ap_population": "all_valid_person_frames",
    "loss": "heatmap_MSE_GT_in_frame_plus_0.1_inout_BCE_all_valid",
    "state": "causal_track_order_reset_on_person_change_or_original_frame_gap_detach_across_chunks",
    "inputs": "frozen_post_person_transformer_RGB_current_raw_GT_head_box_features_only",
}


@dataclass
class TrainingConfig:
    seed: int = 0
    epochs: int = 10
    clip_frames: int = 16
    batch_size: int = 4
    hidden_channels: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    device: str = "cuda:0"
    log_every: int = 100


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cache_metadata(cache):
    return {
        "path": str(cache.root.resolve()), "index_sha256": sha256(cache.root / "index.json"),
        "split": cache.index["split"], "num_frames": len(cache),
        "source": cache.index["source"], "arrays": cache.index["arrays"],
        "heads": cache.index["heads"],
    }


def validate_caches(train_cache, test_cache):
    if train_cache.index["split"] != "train" or test_cache.index["split"] != "test":
        raise ValueError("Training requires complete official train and test caches")
    train_source, test_source = train_cache.index["source"], test_cache.index["source"]
    for key in ("checkpoint_sha256", "backbone_sha256"):
        if train_source.get(key) != test_source.get(key):
            raise ValueError(f"Train/test cache source mismatch: {key}")
    train_heads = FrozenTaskHeads.from_checkpoint(train_cache.root / train_cache.index["heads"]["path"])
    test_heads = FrozenTaskHeads.from_checkpoint(test_cache.root / test_cache.index["heads"]["path"])
    if any(not torch.equal(value, test_heads.state_dict()[key])
           for key, value in train_heads.state_dict().items()):
        raise ValueError("Train/test caches must use the same frozen original heads")
    return train_heads


def append_provenance(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(text)


def begin_provenance(path, metadata, report_dir, output_dir):
    lines = [f"\n## {report_dir.name} / {metadata['created_at_utc']}\n",
             f"\n- Code commit: `{metadata['git_commit']}`.",
             f"\n- Command: `{metadata['command']}`.",
             f"\n- CUDA_VISIBLE_DEVICES: `{metadata['cuda_visible_devices']}`; internal device: `{metadata['config']['device']}`.",
             f"\n- Reports: `{report_dir}`; checkpoints: `{output_dir}`.",
             "\n- Protocol: complete official train; complete official test every epoch; no validation. "
             "Test selects one epoch by AUC descending, L2 ascending, AP descending, earlier epoch.\n",
             "\n| Artifact | Server path | SHA256 |\n|---|---|---|\n"]
    source = metadata["caches"]["train"]["source"]
    lines.append(f"| Base checkpoint | `{source['checkpoint']}` | `{source['checkpoint_sha256']}` |\n")
    if "backbone_checkpoint" in source:
        lines.append(f"| Backbone | `{source['backbone_checkpoint']}` | `{source['backbone_sha256']}` |\n")
    for split, cache in metadata["caches"].items():
        root = Path(cache["path"])
        lines.append(f"| {split} tracks | `{cache['source']['tracks_path']}` | `{cache['source']['tracks_sha256']}` |\n")
        lines.append(f"| {split} index | `{root / 'index.json'}` | `{cache['index_sha256']}` |\n")
        for name, artifact in {**cache["arrays"], "heads": cache["heads"]}.items():
            lines.append(f"| {split} {name} | `{root / artifact['path']}` | `{artifact['sha256']}` |\n")
    append_provenance(path, "".join(lines))


def run_training(train_cache, test_cache, output_dir, report_dir, provenance,
                 config=None, command=None):
    config = config or TrainingConfig()
    if min(config.epochs, config.clip_frames, config.batch_size, config.hidden_channels,
           config.log_every) <= 0:
        raise ValueError("Epochs, clip frames, batch size, hidden channels and log interval must be positive")
    output_dir, report_dir = Path(output_dir).resolve(), Path(report_dir).resolve()
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Use fresh output and report directories; existing experiments are never overwritten")
    seed_everything(config.seed)
    train_cache, test_cache = TaskFeatureCache(train_cache), TaskFeatureCache(test_cache)
    heads = validate_caches(train_cache, test_cache)
    model = TemporalModel(heads, hidden_channels=config.hidden_channels).to(config.device)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": command or shlex.join([sys.executable, *sys.argv]), "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "config": asdict(config), "protocol": PROTOCOL,
        "output_dir": str(output_dir), "report_dir": str(report_dir),
        "trainable_parameters": sum(parameter.numel() for parameter in model.adapter.parameters()),
        "caches": {"train": cache_metadata(train_cache), "test": cache_metadata(test_cache)},
    }
    write_json(report_dir / "run_metadata.json", metadata)
    begin_provenance(provenance, metadata, report_dir, output_dir)

    def test_chunks():
        return CachedChunks(test_cache, config.batch_size, config.clip_frames)

    print("Scoring frozen GazeSpot-B on the complete official test cache", flush=True)
    baseline = evaluate(model, test_chunks(), config.device, baseline=True, log_every=config.log_every)
    write_json(report_dir / "baseline_metrics.json", baseline)
    history = []
    artifacts = []
    selected_epoch = None
    selected_metrics = None
    best_rank = None
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}: complete official train", flush=True)
        train_metrics = train_epoch(
            model, CachedChunks(train_cache, config.batch_size, config.clip_frames,
                                shuffle=True, seed=config.seed + epoch - 1),
            optimizer, config.device, config.gradient_clip_norm, config.log_every,
        )
        print(f"Epoch {epoch}: complete official test", flush=True)
        metrics = evaluate(model, test_chunks(), config.device, log_every=config.log_every)
        record = {"epoch": epoch, "train": train_metrics, "test": metrics}
        if (train_metrics["person_frames"] != len(train_cache)
                or metrics["person_frames"] != len(test_cache)):
            raise RuntimeError("A full epoch must visit every official person frame exactly once")
        history.append(record)
        checkpoint_path = output_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "schema_version": 1, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": asdict(config),
            "metrics": record, "protocol": PROTOCOL, "metadata": metadata,
        }, checkpoint_path)
        checkpoint_sha = sha256(checkpoint_path)
        artifacts.append({"epoch": epoch, "path": str(checkpoint_path), "sha256": checkpoint_sha})
        shutil.copyfile(checkpoint_path, output_dir / "last.pt")
        rank = checkpoint_rank(metrics, epoch)
        if best_rank is None or rank > best_rank:
            best_rank, selected_epoch, selected_metrics = rank, epoch, metrics
            shutil.copyfile(checkpoint_path, output_dir / "selected.pt")
        write_json(report_dir / f"epoch_{epoch:02d}_metrics.json", record)
        write_json(report_dir / "training_history.json", {
            "protocol": PROTOCOL, "baseline": baseline, "epochs": history,
            "selected_epoch": selected_epoch, "selected_metrics": selected_metrics,
        })
        write_json(report_dir / "artifacts.json", {
            "checkpoints": artifacts, "last": {**artifacts[-1], "path": str(output_dir / "last.pt")},
            "selected": {**artifacts[selected_epoch - 1], "path": str(output_dir / "selected.pt")},
        })
        append_provenance(provenance, f"| Epoch {epoch} | `{checkpoint_path}` | `{checkpoint_sha}` |\n")
        print(f"Epoch {epoch}: AUC={metrics['auc']:.6f} L2={metrics['l2']:.6f} "
              f"AP={metrics['inout_ap']:.6f}; selected_epoch={selected_epoch}", flush=True)
    comparison = {
        "protocol": PROTOCOL, "baseline": {"model": "GazeSpot-B", **baseline},
        "temporal": {"model": "GazeSpot-B + ConvLSTM", "epoch": selected_epoch, **selected_metrics},
        "delta_temporal_minus_baseline": {
            key: selected_metrics[key] - baseline[key] for key in ("auc", "l2", "inout_ap")
        },
    }
    write_json(report_dir / "comparison.json", comparison)
    selected_sha = artifacts[selected_epoch - 1]["sha256"]
    append_provenance(provenance,
                      f"| Selected (epoch {selected_epoch}) | `{output_dir / 'selected.pt'}` | `{selected_sha}` |\n"
                      f"| Last (epoch {config.epochs}) | `{output_dir / 'last.pt'}` | `{artifacts[-1]['sha256']}` |\n")
    lines = ["# GazeSpot-B + ConvLSTM / official VAT\n\n",
             "Complete official train/test; no validation. **Test participates in checkpoint selection.**\n\n",
             "| Model | Epoch | AUC ↑ | L2 ↓ | InOut AP ↑ |\n|---|---|---|---|---|\n",
             f"| GazeSpot-B | frozen | {baseline['auc']:.6f} | {baseline['l2']:.6f} | {baseline['inout_ap']:.6f} |\n",
             f"| GazeSpot-B + ConvLSTM | {selected_epoch} | {selected_metrics['auc']:.6f} | "
             f"{selected_metrics['l2']:.6f} | {selected_metrics['inout_ap']:.6f} |\n",
             "\nAUC descending → L2 ascending → AP descending → earlier epoch; all temporal columns use that epoch.\n"]
    (report_dir / "comparison.md").write_text("".join(lines))
    print(f"Completed. Selected epoch {selected_epoch}; small reports: {report_dir}", flush=True)
    return comparison


def add_training_args(parser):
    defaults = TrainingConfig()
    for name in ("seed", "epochs", "clip_frames", "batch_size", "hidden_channels", "log_every"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--provenance", default=str(REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md"))


def config_from_args(args):
    return TrainingConfig(**{key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    add_training_args(parser)
    args = parser.parse_args()
    run_training(args.train_cache, args.test_cache, args.output_dir, args.report_dir,
                 args.provenance, config_from_args(args))


if __name__ == "__main__":
    main()
