"""Run multiple conditional branches sequentially using a pinned training checkout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-code-root", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=("scene_only", "scene_geometry", "scene_head"), required=True)
    args, remaining = parser.parse_known_args(argv)
    if "--condition" in remaining:
        parser.error("use --conditions for this queue")
    return args, remaining


def commands(args, remaining):
    entry = args.training_code_root / "cscwd_router/code/scripts/run_final_ablation.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    return [[sys.executable, "-u", str(entry), "--condition", condition, *remaining] for condition in args.conditions]


def run(args, remaining):
    for command in commands(args, remaining):
        print("START " + json.dumps(command), flush=True)
        subprocess.run(command, cwd=args.training_code_root, check=True)
    print("COMPLETE conditional queue " + ",".join(args.conditions), flush=True)


if __name__ == "__main__":
    run(*parse_args())
