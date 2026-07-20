#!/usr/bin/env python3
"""Paired, sequence-clustered comparison of a COTB run and its control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_key(row: dict) -> tuple[str, int, int]:
    return str(row["path"]), int(row["query_i"]), int(row["query_j"])


def paired_rows(control: list[dict], candidate: list[dict]) -> list[dict]:
    control_by_key = {pair_key(row): row for row in control}
    candidate_by_key = {pair_key(row): row for row in candidate}
    common = sorted(set(control_by_key) & set(candidate_by_key))
    rows = []
    for key in common:
        first, second = control_by_key[key], candidate_by_key[key]
        rows.append(
            {
                "sequence_id": first["sequence_id"],
                "path": key[0],
                "query_i": key[1],
                "query_j": key[2],
                "people_count": int(first["people_count"]),
                "mean_head_size": float(first["mean_head_size"]),
                "target_separation": float(first["target_separation"]),
                "control_swap_error": float(first["swap_error"]),
                "candidate_swap_error": float(second["swap_error"]),
                "delta_swap_error": float(second["swap_error"]) - float(first["swap_error"]),
                "control_diag_margin": float(first["diag_margin"]),
                "candidate_diag_margin": float(second["diag_margin"]),
                "delta_diag_margin": float(second["diag_margin"]) - float(first["diag_margin"]),
            }
        )
    return rows


def clustered_mean(rows: list[dict], field: str, iterations: int, seed: int) -> dict:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    clusters = sorted({row["sequence_id"] for row in rows})
    result = {
        "value": float(values.mean()) if len(values) else None,
        "pair_count": len(values),
        "sequence_count": len(clusters),
        "ci95_low": None,
        "ci95_high": None,
    }
    if len(clusters) < 2 or iterations <= 0:
        return result
    by_cluster = {
        cluster: np.asarray([float(row[field]) for row in rows if row["sequence_id"] == cluster])
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples.append(float(np.concatenate([by_cluster[clusters[index]] for index in selected]).mean()))
    result["ci95_low"], result["ci95_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return result


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "overall": rows,
        "far": [row for row in rows if row["target_separation"] >= 0.30],
        "crowd_ge4": [row for row in rows if row["people_count"] >= 4],
        "crowd_ge5": [row for row in rows if row["people_count"] >= 5],
        "far_crowd_ge4": [
            row for row in rows if row["target_separation"] >= 0.30 and row["people_count"] >= 4
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--control-pairs", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--primary-subset", choices=("overall", "far", "crowd_ge4", "crowd_ge5", "far_crowd_ge4"), default="far")
    parser.add_argument("--minimum-pairs", type=int, default=1000)
    parser.add_argument("--minimum-sequences", type=int, default=30)
    parser.add_argument("--minimum-swap-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-l2-regression", type=float, default=0.003)
    parser.add_argument("--maximum-ap-regression", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    control = read_csv(args.control_pairs)
    candidate = read_csv(args.candidate_pairs)
    rows = paired_rows(control, candidate)
    results = {}
    for offset, (name, selected) in enumerate(subsets(rows).items()):
        results[name] = {
            "control_swap_error": clustered_mean(selected, "control_swap_error", args.bootstrap_iterations, args.seed + offset),
            "candidate_swap_error": clustered_mean(selected, "candidate_swap_error", args.bootstrap_iterations, args.seed + 100 + offset),
            "delta_swap_error": clustered_mean(selected, "delta_swap_error", args.bootstrap_iterations, args.seed + 200 + offset),
            "delta_diag_margin": clustered_mean(selected, "delta_diag_margin", args.bootstrap_iterations, args.seed + 300 + offset),
        }

    l2_delta = None
    ap_delta = None
    if args.control_summary and args.candidate_summary:
        control_summary = json.loads(args.control_summary.read_text(encoding="utf-8"))
        candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
        l2_delta = float(candidate_summary["standard"]["l2"]) - float(control_summary["standard"]["l2"])
        ap_delta = float(candidate_summary["standard"]["inout_ap"]) - float(control_summary["standard"]["inout_ap"])
    primary = results[args.primary_subset]
    delta = primary["delta_swap_error"]
    baseline_error = primary["control_swap_error"]["value"]
    gates = {
        "support": delta["pair_count"] >= args.minimum_pairs and delta["sequence_count"] >= args.minimum_sequences,
        "baseline_failure_exists": baseline_error is not None and baseline_error >= 0.10,
        "swap_improves_by_two_points": delta["value"] is not None and delta["value"] <= -args.minimum_swap_improvement,
        "swap_ci_excludes_zero": delta["ci95_high"] is not None and delta["ci95_high"] < 0.0,
        "l2_preserved": l2_delta is not None and l2_delta <= args.maximum_l2_regression,
        "inout_ap_preserved": ap_delta is not None and ap_delta >= -args.maximum_ap_regression,
    }
    verdict = "GO" if all(gates.values()) else "NO_GO_OR_INCOMPLETE"
    payload = {
        "status": "paired_sequence_cluster_bootstrap",
        "primary_subset": args.primary_subset,
        "matched_pair_count": len(rows),
        "unmatched_control_pairs": len(control) - len(rows),
        "unmatched_candidate_pairs": len(candidate) - len(rows),
        "l2_delta_candidate_minus_control": l2_delta,
        "inout_ap_delta_candidate_minus_control": ap_delta,
        "results": results,
        "decision": {"verdict": verdict, "gates": gates},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
