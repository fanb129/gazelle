"""Run the fixed 3+5+5 recipe, or its synthetic T3 engineering smoke."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dinov2"))

from cscwd_router.code.data_prep.gazefollow import make_data
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.engine import seed_everything
from cscwd_router.code.stag.model_dinov2 import build_model
from cscwd_router.code.stag.pipeline import checkpoint_ref, run_pipeline, write_json


SOURCE_NAMES = (
    "gazelle/backbone_v0_dinov2.py", "gazelle/model_v0_dinov2.py", "gazelle/dataloader.py",
    "gazelle/utils.py", "gazelle/data_splits.py", "gazelle/routing/router.py", "gazelle/routing/losses.py",
    "cscwd_router/code/stag/backbone_sparse_dinov2.py", "cscwd_router/code/stag/router_dinov2.py",
    "cscwd_router/code/stag/model_dinov2.py", "cscwd_router/code/stag/engine.py",
    "cscwd_router/code/stag/checkpoint.py", "cscwd_router/code/stag/pipeline.py",
    "cscwd_router/code/stag/test_training_dinov2.py",
    "cscwd_router/code/data_prep/synthetic_support.py", "cscwd_router/code/data_prep/gazefollow.py",
    "cscwd_router/code/scripts/train_router_dinov2.py", "cscwd_router/code/scripts/eval_router_dinov2.py",
    "dinov2/dinov2/models/vision_transformer.py", "dinov2/dinov2/layers/block.py", "dinov2/dinov2/layers/attention.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("synthetic", "holdout", "full_train"), required=True)
    parser.add_argument("--data-path", type=Path, default=Path("/newhome/fb/dataset/gazefollow_extended"))
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--split-seed", type=int, default=3106)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--router-hidden-dim", type=int, default=256)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--escape-tokens", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the last completed epoch without repeating updates.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    if args.protocol == "full_train" and args.val_fraction != 0.0:
        raise ValueError("full_train requires --val-fraction 0")
    unit_checks = None
    if args.protocol == "synthetic":
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
            "cscwd_router.code.stag.test_backbone_sparse_dinov2", "cscwd_router.code.stag.test_router_dinov2",
            "cscwd_router.code.stag.test_training_dinov2",
        ))
        checked = unittest.TextTestRunner(verbosity=2).run(suite)
        unit_checks = {"run": checked.testsRun, "failures": len(checked.failures), "errors": len(checked.errors), "skipped": len(checked.skipped)}
        if not checked.wasSuccessful() or checked.skipped:
            raise RuntimeError("synthetic structural tests must pass before the training smoke")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = build_model({"router_hidden_dim": args.router_hidden_dim,
                         "router_temperature": args.router_temperature, "escape_tokens": args.escape_tokens})
    model.load_official_decoder(torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    data = make_data(protocol=args.protocol, data_path=args.data_path,
                     transform=model.encoder.backbone.get_transform((448, 448)), batch_size=args.batch_size,
                     eval_batch_size=args.eval_batch_size, workers=args.workers, seed=args.seed,
                     split_seed=args.split_seed, val_fraction=args.val_fraction, augment=not args.no_augment)
    config_names = ("protocol", "batch_size", "eval_batch_size", "workers", "router_lr", "decoder_lr",
                    "weight_decay", "budget_weight", "seed", "split_seed", "val_fraction", "no_augment")
    context = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "train_config": {name: getattr(args, name) for name in config_names},
        "synthetic_tests": unit_checks,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": "fp32", "tf32": False},
        "external_weights": {"backbone": checkpoint_ref(ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"),
                             "official_decoder": checkpoint_ref(args.decoder_checkpoint)},
        "sources": {name: sha256(ROOT / name) for name in SOURCE_NAMES},
    }
    result = run_pipeline(model, data, device, run_dir=args.run_dir, report_dir=args.report_dir, context=context,
                          router_lr=args.router_lr, decoder_lr=args.decoder_lr, weight_decay=args.weight_decay,
                          budget_weight=args.budget_weight, seed=args.seed, log_every=args.log_every, resume=args.resume)
    artifacts = [checkpoint_ref(path) for path in sorted(args.run_dir.rglob("*.pt"))]
    result["artifacts"] = artifacts
    report_path = args.report_dir / "summary.json"
    write_json(report_path, result)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("a") as stream:
        stream.write(f"\n## T3 · {context['created_utc']} · {args.protocol}\n\n")
        stream.write(f"- code commit: `{context['git_commit']}`; precision: `fp32`\n")
        stream.write(f"- report: `{report_path.resolve()}`; SHA256: `{sha256(report_path)}`; engineering passed: `{result['passed']}`\n")
        for role, item in context["external_weights"].items():
            stream.write(f"- input {role}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
        for item in artifacts:
            stream.write(f"- checkpoint: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    print(f"Report: {report_path}; engineering passed={result['passed']}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
