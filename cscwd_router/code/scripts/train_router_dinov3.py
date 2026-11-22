"""Train one existing-initialization DINOv3 B/L Router on official train/test."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.dinov3_run import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), required=True)
    parser.add_argument("--size", choices=("b", "l"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--init-sha256", required=True)
    parser.add_argument("--init-kind", choices=("support", "decoder"), required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--support-epochs", type=int, default=3)
    parser.add_argument("--decoder-epochs", type=int, default=5)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
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
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.support_epochs, args.decoder_epochs, args.batch_size, args.eval_batch_size,
           args.grad_accumulation, args.train_frame_rate, args.test_frame_rate) < 1:
        parser.error("epochs, batch sizes, accumulation and frame rates must be positive")
    if not 0 < args.keep_ratio <= 1:
        parser.error("--keep-ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
