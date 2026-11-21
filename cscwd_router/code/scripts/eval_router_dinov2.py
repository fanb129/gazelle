"""Evaluate one fixed checkpoint with the existing GazeFollow AUC/L2 protocol."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dinov2"))

from gazelle.dataloader import GazeFollowImageDataset
from cscwd_router.code.data_prep.gazefollow import collate_images, make_data, synthetic_batch
from cscwd_router.code.stag.checkpoint import read_checkpoint, restore_checkpoint, sha256
from cscwd_router.code.stag.engine import evaluate
from cscwd_router.code.stag.model_dinov2 import build_model


def validate_evaluation_request(payload, split):
    protocol = payload["context"]["data"]["protocol"]
    required_protocol = {"synthetic": "synthetic", "holdout": "holdout", "official_test": "full_train"}[split]
    if protocol != required_protocol:
        raise ValueError(f"{split} evaluation requires a {required_protocol} checkpoint")
    if split == "official_test":
        epochs = {"dense_official": -1, "sparse_decoder": 4, "matched_dense": 4}
        if (payload["checkpoint_role"] != "fixed_final" or payload["stage"] not in epochs
                or payload["epoch"] != epochs[payload["stage"]]):
            raise ValueError("official test requires a fixed-final dense/K50/matched checkpoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("synthetic", "holdout", "official_test"), required=True)
    parser.add_argument("--data-path", type=Path, default=Path("/newhome/fb/dataset/gazefollow_extended"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    payload = read_checkpoint(args.checkpoint)
    validate_evaluation_request(payload, args.split)
    actual_backbone = ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"
    if sha256(actual_backbone) != payload["context"]["external_weights"]["backbone"]["sha256"]:
        raise ValueError("external backbone checkpoint SHA256 does not match training")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = build_model(payload["model_config"]).to(device)
    restore_checkpoint(model, payload)
    transform = model.encoder.backbone.get_transform((448, 448))
    config = payload["context"]["train_config"]
    if args.split == "synthetic":
        loader = [synthetic_batch(args.batch_size, model.config["image_size"], config["seed"])]
        data_info = {"protocol": "synthetic", "official_test_accessed": False}
    elif args.split == "holdout":
        data = make_data(protocol="holdout", data_path=args.data_path, transform=transform,
                         batch_size=config["batch_size"], eval_batch_size=args.batch_size, workers=args.workers,
                         seed=config["seed"], split_seed=config["split_seed"], val_fraction=config["val_fraction"],
                         augment=not config["no_augment"])
        expected = payload["context"]["data"]
        if any(data.metadata[key] != expected[key] for key in ("annotation_sha256", "assignment_fingerprint")):
            raise ValueError("holdout annotation or image assignment differs from training")
        loader, data_info = data.validation_loader(), data.metadata
    else:
        dataset = GazeFollowImageDataset(args.data_path, "test", transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_images, num_workers=args.workers)
        annotation = args.data_path / "test_preprocessed.json"
        data_info = {"protocol": "official_test", "official_test_accessed": True,
                     "annotation_path": str(annotation.resolve()), "annotation_sha256": sha256(annotation)}
    mode = "sparse" if payload["stage"] == "sparse_decoder" else "dense"
    metrics = evaluate(model, loader, device, mode=mode)
    result = {
        "task": "T3_evaluation", "stage": payload["stage"], "mode": mode, "split": args.split,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "data": data_info, "metrics": metrics, "training_signature": payload["context"]["signature"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "argv": sys.argv, "batch_size": args.batch_size, "torch": str(torch.__version__), "precision": "fp32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("a") as stream:
        stream.write(f"\n- evaluation ({args.split}, {payload['stage']}): `{args.output.resolve()}`; SHA256: `{sha256(args.output)}`\n")
        stream.write(f"- input checkpoint: `{args.checkpoint.resolve()}`; SHA256: `{result['checkpoint']['sha256']}`\n")
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
