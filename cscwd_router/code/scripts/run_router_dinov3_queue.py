"""Run B then L on one assigned GPU; GF and VAT queues are independent."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.dinov3_initializers import BACKBONE_SHA, INITIALIZERS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), required=True)
    parser.add_argument("--sizes", choices=("b", "l"), nargs="+", default=["b", "l"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--support-epochs", type=int, default=3)
    parser.add_argument("--decoder-epochs", type=int, default=5)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--batch-b", type=int, default=16)
    parser.add_argument("--batch-l", type=int, default=4)
    parser.add_argument("--eval-batch-b", type=int, default=16)
    parser.add_argument("--eval-batch-l", type=int, default=4)
    parser.add_argument("--accumulation-b", type=int, default=2)
    parser.add_argument("--accumulation-l", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--inout-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--router-hidden-dim", type=int, default=256)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--escape-tokens", type=int, default=8)
    parser.add_argument("--train-frame-rate", type=int, default=6)
    parser.add_argument("--test-frame-rate", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def job_command(args, size, *, python=sys.executable):
    init = INITIALIZERS[args.dataset, size]
    command = [python, "-u", str(ROOT / "cscwd_router/code/scripts/train_router_dinov3.py"),
               "--dataset", args.dataset, "--size", size, "--data-path", args.data_path,
               "--init-checkpoint", str(ROOT / init["path"]), "--init-sha256", init["sha256"],
               "--init-kind", init["kind"], "--backbone-sha256", BACKBONE_SHA[size],
               "--run-dir", str(Path(args.output_root) / args.dataset / f"vit{size}"),
               "--report-dir", str(Path(args.report_root) / args.dataset / f"vit{size}"),
               "--provenance", args.provenance,
               "--batch-size", str(getattr(args, f"batch_{size}")),
               "--eval-batch-size", str(getattr(args, f"eval_batch_{size}")),
               "--grad-accumulation", str(getattr(args, f"accumulation_{size}")), "--device", "cuda"]
    for key in ("support_epochs", "decoder_epochs", "keep_ratio", "workers", "router_lr", "decoder_lr",
                "inout_lr", "weight_decay", "clip_norm", "budget_weight", "router_hidden_dim",
                "router_temperature", "escape_tokens", "train_frame_rate", "test_frame_rate", "seed", "log_every"):
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if args.resume:
        command.append("--resume")
    return command


def main(argv=None):
    args = parse_args(argv)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("set CUDA_VISIBLE_DEVICES to one physical GPU for this queue")
    for size in args.sizes:
        command = job_command(args, size)
        print(json.dumps({"event": "job_start", "dataset": args.dataset, "size": size,
                          "physical_gpu": gpu, "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"QUEUE_COMPLETE dataset={args.dataset} sizes={','.join(args.sizes)}", flush=True)


if __name__ == "__main__":
    main()
