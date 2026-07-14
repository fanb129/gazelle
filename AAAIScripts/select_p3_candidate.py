#!/usr/bin/env python3
"""Select the P3A winner from validation-only R0/R1/R2/R3 histories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIAlchemyModels.factory import CANDIDATE_CHOICES, resolve_candidate
from AAAIScripts.common import file_manifest


RUN_DIRECTORIES = {
    "r0": "r0_continued_baseline",
    "r1": "r1_residual_refinement",
    "r2": "r2_cross_layer_attention",
    "r3": "r3_residual_coord005",
}


def enrich_rows(rows: list[dict]) -> list[dict]:
    reference = next(row for row in rows if row["candidate"] == "r0")
    for row in rows:
        row["delta_auc"] = row["auc"] - reference["auc"]
        row["delta_l2"] = reference["l2"] - row["l2"]
        row["delta_inout_ap"] = row["inout_ap"] - reference["inout_ap"]
        major_improvement = (
            row["delta_l2"] >= 0.002
            or row["delta_auc"] >= 0.001
            or row["delta_inout_ap"] >= 0.005
        )
        no_collapse = (
            row["delta_auc"] >= -0.002
            and row["delta_l2"] >= -0.003
            and row["delta_inout_ap"] >= -0.010
        )
        row["major_improvement"] = major_improvement
        row["no_collapse"] = no_collapse
        row["qualified"] = row["candidate"] == "r0" or (major_improvement and no_collapse)
    return rows


def dominates(left: dict, right: dict) -> bool:
    not_worse = (
        left["auc"] >= right["auc"]
        and left["l2"] <= right["l2"]
        and left["inout_ap"] >= right["inout_ap"]
    )
    strictly_better = (
        left["auc"] > right["auc"]
        or left["l2"] < right["l2"]
        or left["inout_ap"] > right["inout_ap"]
    )
    return not_worse and strictly_better


def choose_winner(rows: list[dict]) -> tuple[dict, list[str]]:
    rows = enrich_rows(rows)
    qualified = [row for row in rows if row["qualified"]]
    pareto = [
        row for row in qualified
        if not any(dominates(other, row) for other in qualified if other is not row)
    ]
    winner = sorted(pareto, key=lambda row: (row["l2"], -row["auc"], -row["inout_ap"]))[0]
    return winner, [row["candidate"] for row in pareto]


def load_rows(screen_dir: Path) -> list[dict]:
    rows = []
    for candidate in CANDIDATE_CHOICES:
        run_dir = screen_dir / RUN_DIRECTORIES[candidate]
        manifest_path = run_dir / "run_manifest.json"
        history_path = run_dir / "history.json"
        manifest = json.loads(manifest_path.read_text())
        history = json.loads(history_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(f"incomplete run: {manifest_path}")
        if manifest.get("test_used_for_selection") is not False:
            raise RuntimeError(f"test leakage flag is not false: {manifest_path}")
        protocol = manifest.get("validation_protocol", {})
        if protocol.get("source") != "train_preprocessed.json" or protocol.get("unit") != "VAT sequence directory":
            raise RuntimeError(f"P3A requires sequence-level VAT train validation: {manifest_path}")
        if int(protocol.get("seed", -1)) != 3106 or float(protocol.get("validation_fraction", -1)) != 0.1:
            raise RuntimeError(f"P3A split must be 90/10 with seed 3106: {manifest_path}")
        best_epoch = manifest.get("best_epoch")
        matching = [row for row in history if row["epoch"] == best_epoch]
        if len(matching) != 1:
            raise RuntimeError(f"best epoch missing or duplicated in {history_path}")
        metrics = matching[0]["validation"]
        if any(metrics.get(key) is None for key in ("auc", "l2", "inout_ap")):
            raise RuntimeError(f"best epoch lacks required VAT metrics: {history_path}")
        rows.append({
            "candidate": candidate,
            "label": resolve_candidate(candidate).label,
            "best_epoch": best_epoch,
            "auc": float(metrics["auc"]),
            "l2": float(metrics["l2"]),
            "inout_ap": float(metrics["inout_ap"]),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": file_manifest(manifest_path)["sha256"],
        })
    return enrich_rows(rows)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    rows = load_rows(args.screen_dir)
    winner, pareto = choose_winner(rows)
    leaderboard_path = args.screen_dir / "leaderboard.csv"
    fields = [
        "candidate", "label", "best_epoch", "auc", "l2", "inout_ap",
        "delta_auc", "delta_l2", "delta_inout_ap", "major_improvement",
        "no_collapse", "qualified", "manifest", "manifest_sha256",
    ]
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "selected_on_vat_train_derived_validation",
        "test_used_for_selection": False,
        "validation_protocol": {
            "source": "train_preprocessed.json",
            "unit": "VAT sequence directory",
            "train_fraction": 0.9,
            "validation_fraction": 0.1,
            "seed": 3106,
        },
        "selection_rules": {
            "improvement": {"l2": 0.002, "auc": 0.001, "inout_ap": 0.005},
            "maximum_degradation": {"auc": 0.002, "l2": 0.003, "inout_ap": 0.010},
            "tie_break": ["pareto_dominance", "l2", "auc", "inout_ap"],
        },
        "pareto_candidates": pareto,
        "winner": winner,
        "leaderboard": rows,
    }
    (args.screen_dir / "winner.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"Selection self-check passed; winner={winner['candidate']} pareto={pareto}")


def run_self_test() -> None:
    rows = [
        {"candidate": "r0", "auc": 0.9400, "l2": 0.1000, "inout_ap": 0.8900},
        {"candidate": "r1", "auc": 0.9412, "l2": 0.0975, "inout_ap": 0.8940},
        {"candidate": "r2", "auc": 0.9370, "l2": 0.0960, "inout_ap": 0.8950},
        {"candidate": "r3", "auc": 0.9405, "l2": 0.0995, "inout_ap": 0.8910},
    ]
    winner, pareto = choose_winner(rows)
    assert winner["candidate"] == "r1"
    assert pareto == ["r1"]
    assert not next(row for row in rows if row["candidate"] == "r2")["no_collapse"]
    print("Self-test passed: thresholds, collapse guard, Pareto dominance, and tie-break order.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--screen-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.self_test and args.screen_dir is None:
        parser.error("--screen-dir is required")
    return args


if __name__ == "__main__":
    main()
