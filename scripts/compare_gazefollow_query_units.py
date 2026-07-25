"""Compare legacy per-person and true image-grouped GazeFollow evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

try:
    from eval_coverage_router import main as evaluate_checkpoint
except ModuleNotFoundError:
    from scripts.eval_coverage_router import main as evaluate_checkpoint


GAZE_METRICS = ("auc", "avg_l2", "min_l2")
COVERAGE_METRICS = ("routing_hard_coverage", "routing_gt_point_coverage")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint both per person and per image, then "
            "verify K100 equivalence or report the K<100 union effect."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--keep_ratio_override", type=float, default=None)
    parser.add_argument("--person_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--equivalence_tolerance", type=float, default=1e-5)
    parser.add_argument("--expected_person_count", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _evaluation_argv(
    args: argparse.Namespace,
    *,
    unit: str,
    batch_size: int,
) -> list[str]:
    values = [
        "--checkpoint",
        args.checkpoint,
        "--dataset",
        "gazefollow",
        "--data_path",
        args.data_path,
        "--gazefollow_eval_unit",
        unit,
        "--batch_size",
        str(batch_size),
        "--n_workers",
        str(args.n_workers),
        "--device",
        args.device,
    ]
    if args.keep_ratio_override is not None:
        values.extend(
            ["--keep_ratio_override", str(args.keep_ratio_override)]
        )
    if args.amp:
        values.append("--amp")
    return values


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    if args.person_batch_size <= 0 or args.image_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.equivalence_tolerance < 0:
        raise ValueError("--equivalence_tolerance must be non-negative")
    if args.expected_person_count is not None and args.expected_person_count <= 0:
        raise ValueError("--expected_person_count must be positive")

    person_result = evaluate_checkpoint(
        _evaluation_argv(
            args, unit="person", batch_size=args.person_batch_size
        )
    )
    image_result = evaluate_checkpoint(
        _evaluation_argv(
            args, unit="image", batch_size=args.image_batch_size
        )
    )
    person_metrics = person_result["metrics"]
    image_metrics = image_result["metrics"]
    metric_deltas = {
        metric: image_metrics[metric] - person_metrics[metric]
        for metric in GAZE_METRICS
    }

    effective_keep_ratio = float(image_result["model_config"]["keep_ratio"])
    equivalence_expected = math.isclose(
        effective_keep_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12
    )
    checks = {
        "same_person_count": (
            image_metrics["sample_count"] == person_metrics["sample_count"]
        ),
        "image_grouping_reduces_backbone_inputs": (
            image_metrics["image_count"] < person_metrics["image_count"]
        ),
    }
    if args.expected_person_count is not None:
        checks["expected_person_count"] = (
            image_metrics["sample_count"] == args.expected_person_count
        )
    if equivalence_expected:
        for metric, delta in metric_deltas.items():
            checks[f"{metric}_equivalent"] = (
                abs(delta) <= args.equivalence_tolerance
            )
        for metric in COVERAGE_METRICS:
            value = image_metrics.get(metric)
            checks[f"{metric}_is_one"] = (
                value is not None
                and abs(float(value) - 1.0) <= args.equivalence_tolerance
            )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "effective_keep_ratio": effective_keep_ratio,
        "equivalence_expected": equivalence_expected,
        "equivalence_tolerance": args.equivalence_tolerance,
        "checks": checks,
        "passed": all(checks.values()),
        "gaze_metric_delta_image_minus_person": metric_deltas,
        "person_evaluation": person_result,
        "image_grouped_evaluation": image_result,
        "interpretation": (
            "K100 must preserve per-person predictions; a failure indicates "
            "a flattening/indexing bug."
            if equivalence_expected
            else (
                "At K<100, deltas quantify the real shared fixed-budget "
                "multi-person union effect and are not expected to be zero."
            )
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved comparison to {output_path}")
    return result


if __name__ == "__main__":
    main()
