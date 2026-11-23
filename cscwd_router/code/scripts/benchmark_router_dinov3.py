"""One fresh-process efficiency cell: official GazeLLE versus DINOv3 K100/K50."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-summary", "official-checkpoint", "data-path", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--throughput-iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-closed-gf", action="store_true")
    args = parser.parse_args(argv)
    if min(args.warmup, args.latency_iters, args.throughput_iters, args.repeats) < 1:
        parser.error("warmup, iteration counts and repeats must be positive")
    return args


if __name__ == "__main__":
    from cscwd_router.code.stag.dinov3_efficiency import benchmark
    benchmark(parse_args())
