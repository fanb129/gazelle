#!/usr/bin/env python3
"""Plan, optionally execute, and summarize matched SASA/GGSF controls.

Every suite changes one factor while holding the other fixed.  Dry-run is the
default; use --execute only on the experiment server after inspecting plan.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def variants(suite, fixed_fusion, fixed_spatial):
    if suite == "hierarchy":
        return [
            ("single_l2", "selected_layers", "2", fixed_spatial),
            ("single_l5", "selected_layers", "5", fixed_spatial),
            ("single_l8", "selected_layers", "8", fixed_spatial),
            ("single_l11", "selected_layers", "11", fixed_spatial),
            ("all_raw", "raw_concat", "all", fixed_spatial),
            ("all_equal", "equal_weight", "all", fixed_spatial),
            ("all_sasa", "sasa", "all", fixed_spatial),
        ]
    return [
        ("spatial_none", fixed_fusion, "all", "none"),
        ("fixed_gaussian", fixed_fusion, "all", "fixed_gaussian"),
        ("coordconv", fixed_fusion, "all", "coordconv"),
        ("ggsf", fixed_fusion, "all", "ggsf"),
    ]


def build_commands(args, name, fusion, selected_layers, spatial, seed):
    run_name = f"{args.suite}_{name}_seed{seed}"
    run_dir = args.output_dir / "runs" / run_name
    epoch = (8 if args.dataset == "vat" else 15) if args.epochs is None else args.epochs
    train_script = REPO_ROOT / "scripts" / ("train_vat.py" if args.dataset == "vat" else "train_gazefollow.py")
    eval_script = REPO_ROOT / "scripts" / ("eval_vat.py" if args.dataset == "vat" else "eval_gazefollow.py")
    model = args.model or ("gazelle_dinov3_vitb16_inout" if args.dataset == "vat" else "gazelle_dinov3_vitb16")
    checkpoint = run_dir / f"epoch_{epoch - 1}.pt"
    metrics = args.output_dir / "metrics" / f"{run_name}.json"
    common = ["--model", model, "--spatial_prior", spatial, "--fusion", fusion, "--selected_layers", selected_layers]
    train = [args.python, str(train_script), "--data_path", str(args.data_path), "--exp_name", run_name,
             "--run_dir", str(run_dir), "--max_epochs", str(epoch), "--batch_size", str(args.batch_size),
             "--seed", str(seed), "--wandb_mode", args.wandb_mode, *common]
    if args.init_checkpoint:
        train += ["--init_ckpt", str(args.init_checkpoint)]
    evaluation = [args.python, str(eval_script), "--data_path", str(args.data_path), "--variant_ckpt", str(checkpoint),
                  "--metrics_output", str(metrics), "--batch_size", str(args.eval_batch_size), *common]
    if args.dataset == "vat":
        if not args.json_path:
            raise SystemExit("VAT requires --json-path for an explicit, reproducible evaluation split")
        evaluation += ["--json_path", str(args.json_path)]
    return {"name": run_name, "factor_changed": args.suite, "fusion": fusion, "spatial_prior": spatial,
            "selected_layers": selected_layers, "seed": seed, "checkpoint": str(checkpoint), "metrics": str(metrics),
            "train": train, "eval": evaluation, "status": "planned"}


def plan(args):
    runs = [build_commands(args, *variant, seed) for seed in args.seeds for variant in variants(args.suite, args.fixed_fusion, args.fixed_spatial)]
    payload = {"schema_version": 1, "suite": args.suite, "dataset": args.dataset,
               "matched_factor": "spatial_prior" if args.suite == "hierarchy" else "fusion",
               "estimated_hours_per_run": 7 if args.dataset == "vat" else 36,
               "note": "Estimated time is scheduling metadata supplied by the author, not a measured result.", "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "plan.csv").open("w", newline="") as handle:
        fields = ["name", "factor_changed", "fusion", "spatial_prior", "selected_layers", "seed", "checkpoint", "metrics", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: run[key] for key in fields} for run in runs)
    (args.output_dir / "commands.txt").write_text("\n\n".join(f"# {run['name']}\n{shlex.join(run['train'])}\n{shlex.join(run['eval'])}" for run in runs) + "\n")
    if args.execute:
        for run in runs:
            Path(run["metrics"]).parent.mkdir(parents=True, exist_ok=True)
            if args.resume and Path(run["metrics"]).exists():
                continue
            if not (args.resume and Path(run["checkpoint"]).exists()):
                subprocess.run(run["train"], cwd=REPO_ROOT, check=True)
            subprocess.run(run["eval"], cwd=REPO_ROOT, check=True)
    print(f"Wrote {len(runs)} matched-control runs to {args.output_dir}" + (" and executed them" if args.execute else " (dry-run)"))


def summarize(args):
    plan_payload = json.loads((args.output_dir / "plan.json").read_text())
    rows = []
    for run in plan_payload["runs"]:
        path = Path(run["metrics"])
        metrics = json.loads(path.read_text()) if path.exists() else {}
        row = {key: run[key] for key in ("name", "fusion", "spatial_prior", "selected_layers", "seed")}
        row.update({"sample_count": metrics.get("sample_count"), "auc": metrics.get("auc"),
                    "l2": metrics.get("l2", metrics.get("min_l2")), "inout_ap": metrics.get("inout_ap"),
                    "status": "complete" if path.exists() else "missing", "metrics_path": str(path)})
        rows.append(row)
    fields = list(rows[0]) if rows else ["status"]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")
    print(f"Summarized {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} completed runs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "summarize"))
    parser.add_argument("--suite", choices=("hierarchy", "geometry"), required=True)
    parser.add_argument("--dataset", choices=("vat", "gazefollow"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--fixed-spatial", choices=("none", "fixed_gaussian", "coordconv", "ggsf"), default="none")
    parser.add_argument("--fixed-fusion", choices=("raw_concat", "equal_weight", "sasa", "fpn"), default="raw_concat")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3106])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--python", default="/home/fb/anaconda3/envs/py310/bin/python")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan(args) if args.action == "plan" else summarize(args)


if __name__ == "__main__":
    main()
