"""Continue completed VAT B/L Router decoders at low LR and fixed K50."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("all", "b", "l"), default="all")
    parser.add_argument("--parent-report-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--additional-epochs", type=int, default=8)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--inout-lr", type=float, default=1e-4)
    parser.add_argument("--batch-b", type=int, default=16)
    parser.add_argument("--batch-l", type=int, default=4)
    parser.add_argument("--eval-batch-b", type=int, default=16)
    parser.add_argument("--eval-batch-l", type=int, default=4)
    parser.add_argument("--accumulation-b", type=int, default=2)
    parser.add_argument("--accumulation-l", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.additional_epochs < 1:
        parser.error("--additional-epochs must be positive")
    return args


def child_command(args, size):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--size", size]
    for key, value in vars(args).items():
        if key not in ("size", "resume"):
            command.extend(["--" + key.replace("_", "-"), str(value)])
    if args.resume:
        command.append("--resume")
    return command


def main(argv=None):
    args = parse_args(argv)
    if args.size == "all":
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if args.device == "cuda" and (not gpu or "," in gpu):
            raise ValueError("set CUDA_VISIBLE_DEVICES to one physical GPU")
        for size in ("b", "l"):
            subprocess.run(child_command(args, size), cwd=ROOT, check=True)
        print("QUEUE_COMPLETE vat_continuation sizes=b,l", flush=True)
        return
    # The queue parent imports no torch and does not retain a CUDA context.
    from cscwd_router.code.stag.dinov3_continue import run
    args.batch_size = getattr(args, f"batch_{args.size}")
    args.eval_batch_size = getattr(args, f"eval_batch_{args.size}")
    args.grad_accumulation = getattr(args, f"accumulation_{args.size}")
    run(args)


if __name__ == "__main__":
    main()
