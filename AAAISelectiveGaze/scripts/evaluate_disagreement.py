"""Run the source-trained, three-domain hierarchy-disagreement pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from AAAISelectiveGaze.data.prediction_cache import load_prediction_cache
from AAAISelectiveGaze.metrics.selective import failure_auroc, selective_metrics
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
    parser.add_argument("--id-predictions", type=Path)
    parser.add_argument("--shift-predictions", type=Path)
    parser.add_argument("--ood-predictions", type=Path)
    parser.add_argument("--failure-l2-threshold", type=float, default=0.15)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _model(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def _fit_source_models(source_rows, seed: int):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels = np.asarray([row["failure"] for row in source_rows], dtype=int)
    x_final = feature_matrix(source_rows, FINAL_FEATURES)
    x_enhanced = feature_matrix(source_rows, FINAL_FEATURES + DISAGREEMENT_FEATURES)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        raise ValueError("ID pilot data must contain at least two successes and two failures")
    folds = min(5, int(counts.min()))
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    final_oof = cross_val_predict(
        _model(seed), x_final, labels, cv=split, method="predict_proba"
    )[:, 1]
    enhanced_oof = cross_val_predict(
        _model(seed), x_enhanced, labels, cv=split, method="predict_proba"
    )[:, 1]
    final_model = _model(seed).fit(x_final, labels)
    enhanced_model = _model(seed).fit(x_enhanced, labels)
    return final_model, enhanced_model, final_oof, enhanced_oof, folds


def _bootstrap_delta(labels, baseline, enhanced, iterations: int, seed: int):
    if iterations <= 0:
        return {"mean": float(failure_auroc(labels, enhanced) - failure_auroc(labels, baseline)), "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        value = failure_auroc(labels[indices], enhanced[indices]) - failure_auroc(
            labels[indices], baseline[indices]
        )
        if np.isfinite(value):
            deltas.append(value)
    if not deltas:
        return {"mean": None, "ci95": [None, None]}
    return {
        "mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }


def _top_bottom_ratio(rows) -> dict[str, float | None]:
    order = np.argsort([row["pairwise_js_mean"] for row in rows])
    count = max(1, int(np.ceil(0.2 * len(rows))))
    labels = np.asarray([row["failure"] for row in rows], dtype=float)
    bottom = float(labels[order[:count]].mean())
    top = float(labels[order[-count:]].mean())
    return {
        "top_20_failure_rate": top,
        "bottom_20_failure_rate": bottom,
        # A non-zero numerator over a zero baseline is an infinite improvement,
        # not an undefined/failed Go condition. JSON serialization records it as
        # null while the boolean audit below still treats it correctly.
        "ratio": (float("inf") if top > 0 else 1.0) if bottom == 0 else top / bottom,
    }


def _probe_correlations(rows) -> dict[str, Any]:
    columns = sorted(
        (key for key in rows[0] if key.startswith("probe_l2_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=float)
    correlation = np.corrcoef(matrix, rowvar=False)
    off_diagonal = correlation[np.triu_indices(len(columns), 1)]
    return {
        "layers": [column.rsplit("_", 1)[-1] for column in columns],
        "matrix": correlation,
        "all_pairwise_above_0.95": bool(np.all(off_diagonal > 0.95)),
    }


def _save_plot(curves, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4.2), squeeze=False)
    for axis, (domain, methods) in zip(axes[0], curves.items()):
        for method, (coverage, risk) in methods.items():
            axis.plot(coverage, risk, label=method)
        axis.set_title(domain)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Retained mean L2")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_paths = (args.id_predictions, args.shift_predictions, args.ood_predictions)
    if args.synthetic_smoke:
        if any(path is not None for path in prediction_paths):
            raise ValueError("--synthetic-smoke cannot be combined with prediction paths")
        record_sets = {
            "id": synthetic_records(seed=args.seed, dataset="synthetic_id"),
            "shift": synthetic_records(seed=args.seed + 1, dataset="synthetic_shift"),
            "ood": synthetic_records(seed=args.seed + 2, dataset="synthetic_ood"),
        }
    else:
        if any(path is None for path in prediction_paths):
            raise ValueError("all three prediction paths are required")
        record_sets = {
            "id": load_prediction_cache(args.id_predictions),
            "shift": load_prediction_cache(args.shift_predictions),
            "ood": load_prediction_cache(args.ood_predictions),
        }
    domain_rows = {
        domain: records_to_rows(records, args.failure_l2_threshold)
        for domain, records in record_sets.items()
    }
    final_model, enhanced_model, final_oof, enhanced_oof, folds = _fit_source_models(
        domain_rows["id"], args.seed
    )

    all_rows = []
    summaries = {}
    curves = {}
    curve_rows = []
    for domain_index, (domain, rows) in enumerate(domain_rows.items()):
        labels = np.asarray([row["failure"] for row in rows], dtype=int)
        errors = np.asarray([row["localization_l2"] for row in rows], dtype=float)
        if domain == "id":
            final_scores, enhanced_scores = final_oof, enhanced_oof
        else:
            final_scores = final_model.predict_proba(feature_matrix(rows, FINAL_FEATURES))[:, 1]
            enhanced_scores = enhanced_model.predict_proba(
                feature_matrix(rows, FINAL_FEATURES + DISAGREEMENT_FEATURES)
            )[:, 1]
        domain_metrics = {}
        domain_curves = {}
        for method, scores in (
            ("final_confidence", final_scores),
            ("final_plus_disagreement", enhanced_scores),
        ):
            metrics = selective_metrics(
                scores,
                errors,
                failure_labels=labels,
                coverages=(0.5, 0.7, 0.8, 0.9, 1.0),
            )
            coverage = metrics.pop("coverage_curve")
            risk = metrics.pop("risk_curve")
            domain_metrics[method] = metrics
            domain_curves[method] = (coverage, risk)
            curve_rows.extend(
                {"domain": domain, "method": method, "coverage": c, "risk": r}
                for c, r in zip(coverage, risk)
            )
        delta = float(
            domain_metrics["final_plus_disagreement"]["failure_auroc"]
            - domain_metrics["final_confidence"]["failure_auroc"]
        )
        summaries[domain] = {
            "num_samples": len(rows),
            "failure_rate": float(labels.mean()),
            "metrics": domain_metrics,
            "incremental_failure_auroc": delta,
            "bootstrap_incremental_auroc": _bootstrap_delta(
                labels, final_scores, enhanced_scores, args.bootstrap_iters, args.seed + domain_index
            ),
            "disagreement_extremes": _top_bottom_ratio(rows),
            "probe_error_correlation": _probe_correlations(rows),
        }
        curves[domain] = domain_curves
        for index, row in enumerate(rows):
            item = dict(row)
            item.update(
                {
                    "domain": domain,
                    "risk_final_confidence": float(final_scores[index]),
                    "risk_final_plus_disagreement": float(enhanced_scores[index]),
                }
            )
            all_rows.append(item)

    ratios = [value["disagreement_extremes"]["ratio"] for value in summaries.values()]
    condition_1 = all(ratio >= 1.5 for ratio in ratios)
    gains = [value["incremental_failure_auroc"] for value in summaries.values()]
    condition_2 = sum(gain >= 0.03 for gain in gains) >= 2
    cis = [value["bootstrap_incremental_auroc"]["ci95"] for value in summaries.values()]
    condition_3 = sum(
        gain > 0.01 and (ci[0] is None or ci[0] > -0.01)
        for gain, ci in zip(gains, cis)
    ) >= 2
    condition_4 = not all(
        value["probe_error_correlation"]["all_pairwise_above_0.95"]
        for value in summaries.values()
    )
    go_audit = {
        "condition_1_top_bottom_failure_ratio_at_least_1.5_all_domains": condition_1,
        "condition_2_incremental_auroc_at_least_0.03_two_domains": condition_2,
        "condition_3_controlled_gain_positive_two_domains": condition_3,
        "condition_4_probe_errors_not_all_redundant": condition_4,
        "condition_5_qualitative_gaze_conflict": "manual_review_required",
        "decision": "provisional_go" if condition_1 and condition_2 and condition_3 else "no_go",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(all_rows).to_csv(args.output_dir / "per_sample.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "coverage_risk.csv", index=False)
    _save_plot(curves, args.output_dir / "coverage_risk.png")
    ranked = sorted(all_rows, key=lambda row: row["pairwise_js_mean"])
    case_count = min(20, max(1, len(ranked) // 10))
    write_json(
        args.output_dir / "high_low_disagreement_cases.json",
        {"low": ranked[:case_count], "high": ranked[-case_count:]},
    )
    summary = {
        "status": "ok",
        "synthetic_smoke": args.synthetic_smoke,
        "failure_l2_threshold": args.failure_l2_threshold,
        "source_id_cross_validation_folds": folds,
        "domains": summaries,
        "go_no_go": go_audit,
        "protocol": "ID out-of-fold; shift/OOD scored by logistic models fitted only on ID",
    }
    write_json(args.output_dir / "summary.json", json_safe(summary))
    print(f"Pilot decision: {go_audit['decision']}; outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
