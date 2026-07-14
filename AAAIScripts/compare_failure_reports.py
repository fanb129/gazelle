#!/usr/bin/env python3
"""Paired comparison of two ``failure_taxonomy.py`` record files.

Positive ``improvement`` always means the candidate is better.  Confidence
intervals resample VAT sequence directories (or GazeFollow frames) rather than
treating every temporally adjacent person annotation as independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.common import file_manifest


METRIC_DIRECTIONS = {
    "auc": 1.0,
    "l2": -1.0,
    "avg_l2": -1.0,
    "min_l2": -1.0,
    "target_confusion": -1.0,
    "association_margin": 1.0,
}
KEY_FIELDS = ("dataset", "path", "person_index")
MATCH_FIELDS = (
    "people_count", "crowd_bin", "head_size_bin", "inout",
    "target_x", "target_y", "target_distance_bin", "target_separation_bin",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def number(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def sequence_id(dataset: str, path: str) -> str:
    path_obj = Path(path)
    if dataset == "vat":
        return str(path_obj.parent)
    return path


def build_pairs(reference: list[dict[str, str]], candidate: list[dict[str, str]]) -> list[dict]:
    reference_map = {key(row): row for row in reference}
    candidate_map = {key(row): row for row in candidate}
    if len(reference_map) != len(reference) or len(candidate_map) != len(candidate):
        raise ValueError("duplicate dataset/path/person_index key in an input record file")
    missing_candidate = sorted(set(reference_map) - set(candidate_map))
    missing_reference = sorted(set(candidate_map) - set(reference_map))
    if missing_candidate or missing_reference:
        raise ValueError(
            f"paired inputs differ: missing_candidate={len(missing_candidate)}, "
            f"missing_reference={len(missing_reference)}"
        )

    output = []
    for row_key in sorted(reference_map):
        base, cand = reference_map[row_key], candidate_map[row_key]
        mismatches = [field for field in MATCH_FIELDS if base.get(field, "") != cand.get(field, "")]
        if mismatches:
            raise ValueError(f"annotation/bin mismatch for {row_key}: {mismatches}")
        row = {field: base.get(field, "") for field in (*KEY_FIELDS, *MATCH_FIELDS)}
        row["sequence_id"] = sequence_id(base["dataset"], base["path"])
        row["inout_label"] = int(base["inout"]) if base.get("inout", "") != "" else None
        row["reference_inout_score"] = number(base.get("inout_score"))
        row["candidate_inout_score"] = number(cand.get("inout_score"))
        for metric, direction in METRIC_DIRECTIONS.items():
            base_value, cand_value = number(base.get(metric)), number(cand.get(metric))
            row[f"reference_{metric}"] = base_value
            row[f"candidate_{metric}"] = cand_value
            row[f"improvement_{metric}"] = (
                direction * (cand_value - base_value)
                if base_value is not None and cand_value is not None else None
            )
        output.append(row)
    return output


def bootstrap_ci(rows: list[dict], field: str, iterations: int, seed: int) -> tuple[float | None, float | None]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            clusters[row["sequence_id"]].append(float(value))
    cluster_names = sorted(clusters)
    if not cluster_names:
        return None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        observations = [value for cluster in sampled for value in clusters[str(cluster)]]
        values.append(float(np.mean(observations)))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize(rows: list[dict], iterations: int, seed: int) -> list[dict]:
    groups: list[tuple[str, str, list[dict]]] = [("overall", "all", rows)]
    for dimension, field in (
        ("crowd", "crowd_bin"),
        ("head_size", "head_size_bin"),
        ("inout", "inout"),
        ("target_distance", "target_distance_bin"),
        ("target_separation", "target_separation_bin"),
    ):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(field, "missing"))].append(row)
        groups.extend((dimension, name, bucket) for name, bucket in sorted(buckets.items()))

    output = []
    for dimension, bin_name, group_rows in groups:
        for metric in METRIC_DIRECTIONS:
            field = f"improvement_{metric}"
            values = [float(row[field]) for row in group_rows if row.get(field) is not None]
            lower, upper = bootstrap_ci(group_rows, field, iterations, seed)
            output.append({
                "dimension": dimension,
                "bin": bin_name,
                "metric": metric,
                "positive_means": "candidate_better",
                "n": len(values),
                "cluster_n": len({row["sequence_id"] for row in group_rows if row.get(field) is not None}),
                "mean_improvement": float(np.mean(values)) if values else None,
                "ci95_low": lower,
                "ci95_high": upper,
            })
    ap_rows = [
        row
        for row in rows
        if row.get("inout_label") is not None
        and row.get("reference_inout_score") is not None
        and row.get("candidate_inout_score") is not None
    ]
    if ap_rows:
        from sklearn.metrics import average_precision_score

        labels = np.asarray([row["inout_label"] for row in ap_rows], dtype=np.int64)
        reference_scores = np.asarray(
            [row["reference_inout_score"] for row in ap_rows], dtype=np.float64
        )
        candidate_scores = np.asarray(
            [row["candidate_inout_score"] for row in ap_rows], dtype=np.float64
        )
        point_delta = float(
            average_precision_score(labels, candidate_scores)
            - average_precision_score(labels, reference_scores)
        )
        cluster_names = sorted({row["sequence_id"] for row in ap_rows})
        indices_by_cluster = {
            name: np.asarray(
                [index for index, row in enumerate(ap_rows) if row["sequence_id"] == name],
                dtype=np.int64,
            )
            for name in cluster_names
        }
        rng = np.random.default_rng(seed)
        bootstrap_deltas = []
        for _ in range(iterations):
            sampled = rng.choice(cluster_names, size=len(cluster_names), replace=True)
            indices = np.concatenate([indices_by_cluster[str(name)] for name in sampled])
            bootstrap_deltas.append(
                float(
                    average_precision_score(labels[indices], candidate_scores[indices])
                    - average_precision_score(labels[indices], reference_scores[indices])
                )
            )
        output.append({
            "dimension": "overall",
            "bin": "all",
            "metric": "inout_ap",
            "positive_means": "candidate_better",
            "n": len(ap_rows),
            "cluster_n": len(cluster_names),
            "mean_improvement": point_delta,
            "ci95_low": float(np.percentile(bootstrap_deltas, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_deltas, 97.5)),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    common = {
        "dataset": "vat", "path": "images/a/b/0001.jpg", "person_index": "0",
        "people_count": "4", "crowd_bin": "4", "head_size_bin": "small", "inout": "1",
        "target_x": "0.2", "target_y": "0.3", "target_distance_bin": "far",
        "target_separation_bin": "close",
    }
    reference = [{**common, "auc": "0.8", "l2": "0.2", "target_confusion": "1", "association_margin": "-0.1"}]
    candidate = [{**common, "auc": "0.9", "l2": "0.1", "target_confusion": "0", "association_margin": "0.2"}]
    pairs = build_pairs(reference, candidate)
    assert np.isclose(pairs[0]["improvement_auc"], 0.1)
    assert np.isclose(pairs[0]["improvement_l2"], 0.1)
    assert np.isclose(pairs[0]["improvement_target_confusion"], 1.0)
    assert np.isclose(pairs[0]["improvement_association_margin"], 0.3)
    print("Self-test passed: paired alignment and metric directions.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-records", type=Path)
    parser.add_argument("--candidate-records", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("AAAIScripts/results/paired_comparison"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and (args.reference_records is None or args.candidate_records is None):
        parser.error("--reference-records and --candidate-records are required")
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    rows = build_pairs(read_csv(args.reference_records), read_csv(args.candidate_records))
    summary = summarize(rows, args.bootstrap_iterations, args.seed)
    transient_fields = {"inout_label", "reference_inout_score", "candidate_inout_score"}
    paired_output = [
        {key: value for key, value in row.items() if key not in transient_fields}
        for row in rows
    ]
    # AP is a nonlinear dataset-level metric, so keep its bootstrap result in
    # the JSON report rather than mixing it into the per-observation CSV schema.
    csv_summary = [row for row in summary if row["metric"] != "inout_ap"]
    write_csv(args.output_prefix.with_suffix(".paired.csv"), paired_output)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), csv_summary)
    report = {
        "status": "measured",
        "reference": file_manifest(args.reference_records),
        "candidate": file_manifest(args.candidate_records),
        "pair_count": len(rows),
        "bootstrap": {
            "unit": "VAT sequence directory; GazeFollow frame",
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "metric_directions": {**METRIC_DIRECTIONS, "inout_ap": 1.0},
        "summary": summary,
    }
    report_path = args.output_prefix.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} paired records and {len(summary)} summary rows.")


if __name__ == "__main__":
    main()
