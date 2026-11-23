"""Fine-tune a complete GF Router/decoder on VAT; keep the backbone frozen."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gf-summary", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--router-warmup-epochs", type=int, default=1)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--inout-lr", type=float, default=1e-3)
    parser.add_argument("--router-loss-weight", type=float, default=1.0)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--train-frame-rate", type=int, default=6)
    parser.add_argument("--test-frame-rate", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-closed-gf", action="store_true",
                        help="accept a user-closed GF run only after checking its committed history and best weights")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.grad_accumulation,
           args.train_frame_rate, args.test_frame_rate, args.log_every) < 1:
        parser.error("epochs, batch sizes, accumulation, frame rates and log interval must be positive")
    if not 0 <= args.router_warmup_epochs <= args.epochs:
        parser.error("--router-warmup-epochs must be between zero and --epochs")
    return args


if __name__ == "__main__":
    from cscwd_router.code.stag.dinov3_transfer import run
    run(parse_args())
