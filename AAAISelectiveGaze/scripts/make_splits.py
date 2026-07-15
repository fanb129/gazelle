"""CLI for deterministic, leakage-free SelectiveGaze development splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from AAAISelectiveGaze.data.split_builder import SplitRatios, write_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split GazeFollow by image or VAT by complete video sequence into "
            "base/probe, risk-training, and calibration partitions."
        )
    )
    parser.add_argument("--dataset", required=True, choices=("gazefollow", "vat"))
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--group-key",
        default=None,
        help=(
            "Grouping field. Defaults to path for GazeFollow and sequence_id "
            "for VAT; legacy nested VAT JSON may use sequence path instead."
        ),
    )
    parser.add_argument("--base-train-ratio", type=float, default=0.85)
    parser.add_argument("--base-val-ratio", type=float, default=0.05)
    parser.add_argument("--risk-ratio", type=float, default=0.05)
    parser.add_argument("--calibration-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3106)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ratios = SplitRatios(
        base_train=args.base_train_ratio,
        base_val=args.base_val_ratio,
        risk_train=args.risk_ratio,
        risk_calibration=args.calibration_ratio,
    )
    manifest = write_splits(
        args.input_json,
        args.output_dir,
        dataset=args.dataset,
        ratios=ratios,
        group_key=args.group_key,
        seed=args.seed,
    )
    summary = {
        name: {
            "records": info["num_records"],
            "groups": info["num_groups"],
            "samples": info["num_samples"],
        }
        for name, info in manifest["splits"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"Wrote leakage-free splits to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
