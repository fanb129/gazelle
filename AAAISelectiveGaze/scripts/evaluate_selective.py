"""Evaluate selective localization metrics without training a neural risk head."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from AAAISelectiveGaze.data.prediction_cache import load_prediction_cache
from AAAISelectiveGaze.metrics.selective import selective_metrics
from AAAISelectiveGaze.scripts._pilot_utils import (
    DISAGREEMENT_FEATURES,
    FINAL_FEATURES,
    feature_matrix,
    json_safe,
    records_to_rows,
    synthetic_records,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--risk-checkpoint", type=Path)
    parser.add_argument("--risk-column", default=None)
    parser.add_argument("--localization-only", action="store_true")
    parser.add_argument("--failure-l2-threshold", type=float, default=0.15)
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def _save_curve_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    for name, (coverage, risk) in curves.items():
        axis.plot(coverage, risk, label=name)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Retained mean normalized L2")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.risk_checkpoint is not None:
        raise SystemExit(
            "Neural risk-head loading is intentionally deferred beyond phase one. "
            "Use --risk-column or the implemented confidence/disagreement baselines."
        )
    if args.synthetic_smoke == (args.predictions is not None):
        raise ValueError("choose exactly one of --synthetic-smoke or --predictions")
    records = (
        synthetic_records(seed=args.seed)
        if args.synthetic_smoke
        else load_prediction_cache(args.predictions)
    )
    rows = records_to_rows(records, args.failure_l2_threshold)
    errors = np.asarray([row["localization_l2"] for row in rows], dtype=float)
    rng = np.random.default_rng(args.seed)

    final_risk = _zscore(feature_matrix(rows, FINAL_FEATURES)).mean(axis=1)
    enhanced_risk = np.column_stack(
        (
            _zscore(feature_matrix(rows, FINAL_FEATURES)),
            _zscore(feature_matrix(rows, DISAGREEMENT_FEATURES)),
        )
    ).mean(axis=1)
    risks = {
        "random": rng.random(len(rows)),
        "final_confidence": final_risk,
        "final_plus_disagreement": enhanced_risk,
    }
    if args.risk_column:
        by_id = {str(record["sample_id"]): record for record in records}
        try:
            risks[args.risk_column] = np.asarray(
                [float(by_id[row["sample_id"]][args.risk_column]) for row in rows]
            )
        except KeyError as exc:
            raise ValueError(f"risk column {args.risk_column!r} is missing") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    curve_rows = []
    curves = {}
    for name, scores in risks.items():
        result = selective_metrics(
            scores,
            errors,
            failure_threshold=args.failure_l2_threshold,
            coverages=args.coverages,
        )
        coverage = result.pop("coverage_curve")
        risk = result.pop("risk_curve")
        metrics[name] = result
        curves[name] = (coverage, risk)
        curve_rows.extend(
            {"method": name, "coverage": float(c), "risk": float(r)}
            for c, r in zip(coverage, risk)
        )

    import pandas as pd

    output_rows = []
    for index, row in enumerate(rows):
        item = dict(row)
        for name, scores in risks.items():
            item[f"risk_{name}"] = float(scores[index])
        output_rows.append(item)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "per_sample.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "coverage_risk.csv", index=False)
    _save_curve_plot(curves, args.output_dir / "coverage_risk.png")
    summary = {
        "status": "ok",
        "synthetic_smoke": args.synthetic_smoke,
        "dataset": args.dataset or records[0]["dataset"],
        "num_in_frame_samples": len(rows),
        "failure_l2_threshold": args.failure_l2_threshold,
        "failure_rate": float(np.mean(errors > args.failure_l2_threshold)),
        "metrics": json_safe(metrics),
        "note": "No neural risk head was loaded or trained in phase one.",
    }
    write_json(args.output_dir / "metrics.json", summary)
    print(f"Saved selective metrics for {len(rows)} samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
