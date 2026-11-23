"""Run selected models sequentially on one GPU; each timing cell is a new process."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.scripts.benchmark_router_dinov3 import parse_args as cell_args
from cscwd_router.code.stag import dinov3_efficiency as efficiency
from cscwd_router.code.stag.dinov3_run import atomic_json
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_sources import load_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    for key in ("data-path", "official-checkpoint-dir", "report-root", "provenance"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", choices=(1, 4, 8), default=[1, 8])
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--throughput-iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-closed-gf", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def cell_command(args, summary_path, summary, batch, order, output):
    suffix = "_inout" if summary["dataset"] == "vat" else ""
    official = args.official_checkpoint_dir / f"gazelle_dinov2_vit{summary['size']}14{suffix}.pt"
    command = [sys.executable, "-u", str(ROOT / "cscwd_router/code/scripts/benchmark_router_dinov3.py"),
               "--source-summary", str(summary_path), "--official-checkpoint", str(official),
               "--data-path", str(args.data_path), "--batch-size", str(batch), "--order", order,
               "--output", str(output)]
    for name in ("precision", "warmup", "latency_iters", "throughput_iters", "repeats", "seed"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.allow_closed_gf:
        command.append("--allow-closed-gf")
    return command


def run(args):
    for summary_path in args.summaries:
        summary, _, _ = load_source(summary_path, allow_closed_gf=args.allow_closed_gf)
        folder = args.report_root / summary["dataset"] / f"vit{summary['size']}"
        cells, refs = [], []
        for batch in args.batches:
            for order in efficiency.ORDERS:
                output = folder / f"b{batch}_{order}.json"
                command = cell_command(args, summary_path, summary, batch, order, output)
                request = efficiency.make_request(cell_args(command[3:]))
                if args.check_only:
                    print(json.dumps({"checked": str(summary_path), "batch": batch, "order": order,
                                      "source_closed_by_user": summary.get("closed_by_user", False)}), flush=True)
                    continue
                if output.exists():
                    cell = json.loads(output.read_text())
                    if not args.resume or cell["request"] != request:
                        raise ValueError(f"existing benchmark differs or needs --resume: {output}")
                else:
                    subprocess.run(command, cwd=ROOT, check=True)
                    cell = json.loads(output.read_text())
                    if cell["request"] != request:
                        raise ValueError("benchmark inputs changed while the cell was running")
                cells.append(cell)
                refs.append({"path": str(output.resolve()), "sha256": sha256(output)})
        if args.check_only:
            continue
        result = {**efficiency.summarize(cells), "cells": refs}
        path = folder / "summary.json"
        atomic_json(path, result)
        path.with_suffix(".md").write_text(efficiency.render_summary(result))
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        with args.provenance.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(f"\n## DINOv3 efficiency · {summary['dataset']} / vit{summary['size']}\n\n"
                         f"- report: `{path.resolve()}`; SHA256: `{sha256(path)}`\n")
            for item in refs:
                handle.write(f"- cell: `{item['path']}`; SHA256: `{item['sha256']}`\n")
        print(f"COMPLETE {path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
