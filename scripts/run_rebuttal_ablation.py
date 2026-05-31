import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazelle.ablation_variants import (
    FUSION_CHOICES,
    SPATIAL_PRIOR_CHOICES,
    build_run_metadata,
    parse_selected_layers,
    resolve_variant_config,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build and run P1 rebuttal ablation commands."
    )
    parser.add_argument("--print_plan_only", action="store_true", help="Validate arguments and write a run plan without training.")
    parser.add_argument("--runner_smoke_only", action="store_true", help="Write commands, manifest, and placeholder metrics without launching training/eval.")
    parser.add_argument("--group", choices=("spatial_prior", "fusion", "reliability"), required=True)
    parser.add_argument("--dataset", choices=("vat", "gazefollow"), required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--crowd_json", default=None)
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--input_size", default="512,512")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--selected_layers", default=None)
    parser.add_argument("--selected_layer_sets", nargs="+", default=None)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--output_dir", required=True)
    return parser


def parse_input_size(value):
    try:
        size = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input_size must be formatted as H,W") from exc
    if len(size) != 2 or any(dim <= 0 for dim in size):
        raise argparse.ArgumentTypeError("--input_size must contain two positive integers")
    return size


def _resolve_group_variant(group, variant, selected_layers, seed):
    if group == "spatial_prior":
        if variant not in SPATIAL_PRIOR_CHOICES:
            raise argparse.ArgumentTypeError(f"invalid spatial_prior variant for group spatial_prior: {variant}")
        return resolve_variant_config(spatial_prior=variant, fusion="sasa", selected_layers=selected_layers, seed=seed)

    if group == "fusion":
        if variant not in FUSION_CHOICES:
            raise argparse.ArgumentTypeError(f"invalid fusion variant for group fusion: {variant}")
        return resolve_variant_config(spatial_prior="ggsf", fusion=variant, selected_layers=selected_layers, seed=seed)

    if group == "reliability":
        if variant == "baseline_full":
            return resolve_variant_config(spatial_prior="none", fusion="raw_concat", selected_layers=selected_layers, seed=seed)
        if variant == "gazespot_full":
            return resolve_variant_config(spatial_prior="ggsf", fusion="sasa", selected_layers=selected_layers, seed=seed)
        raise argparse.ArgumentTypeError(f"invalid reliability variant: {variant}")

    raise argparse.ArgumentTypeError(f"unsupported group: {group}")


def build_run_plan(args):
    crowd_json = resolve_crowd_json(args.data_path, args.crowd_json) if args.dataset == "vat" else args.crowd_json
    input_size = parse_input_size(args.input_size)
    selected_layer_sets = args.selected_layer_sets or [args.selected_layers]
    seeds = args.seeds or [args.seed]
    runs = []

    for seed in seeds:
        for variant in args.variants:
            layer_values = selected_layer_sets if variant == "selected_layers" else [args.selected_layers]
            for selected_layers in layer_values:
                if selected_layers is not None:
                    parse_selected_layers(selected_layers)
                config = _resolve_group_variant(args.group, variant, selected_layers, seed)
                metadata = build_run_metadata(
                    dataset=args.dataset,
                    backbone=args.model,
                    input_size=input_size,
                    config=config,
                    checkpoint_path=args.init_ckpt,
                    sample_count=None,
                    group=args.group,
                    variant=variant,
                    data_path=args.data_path,
                    crowd_json=crowd_json,
                )
                run_name = _run_name(args.group, variant, config.selected_layers_label, seed)
                runs.append(
                    {
                        "name": run_name,
                        "metadata": metadata.to_dict(),
                        "legacy_flags": {
                            "use_sasa": config.use_sasa,
                            "use_ggsf": config.use_ggsf,
                        },
                        "commands": build_commands(args, run_name, config, crowd_json),
                        "status": "planned",
                    }
                )

    return {
        "print_plan_only": bool(args.print_plan_only),
        "runner_smoke_only": bool(args.runner_smoke_only),
        "section": "4.training_and_evaluation_runner" if not args.print_plan_only else "1.variant_configuration_and_metadata",
        "group": args.group,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "crowd_json": crowd_json,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "runs": runs,
        "note": "Runner smoke writes manifests and placeholder metrics without training." if args.runner_smoke_only else "Run commands are ready for server training/evaluation.",
    }


def resolve_crowd_json(data_path, requested):
    if requested is None:
        requested = os.path.join(data_path, "test_preprocessed_subsets", "test_crowd_ge4.json")
    requested_path = Path(requested)
    if requested_path.exists():
        return str(requested_path)

    # Only fail locally when the surrounding dataset path is visible enough to list alternatives.
    searchable_dirs = [requested_path.parent, Path(data_path) / "test_preprocessed_subsets"]
    alternatives = []
    for directory in searchable_dirs:
        if directory.exists():
            alternatives.extend(sorted(str(path) for path in directory.glob("test_crowd_*.json")))
    alternatives = sorted(set(alternatives))
    if alternatives:
        raise FileNotFoundError(
            f"crowd_json not found: {requested}. Available alternatives: {', '.join(alternatives)}"
        )
    return requested


def build_commands(args, run_name, config, crowd_json):
    run_dir = os.path.join(args.output_dir, "runs", run_name)
    ckpt_path = os.path.join(run_dir, f"epoch_{max(args.max_epochs - 1, 0)}.pt")
    train_script = "scripts/train_vat.py" if args.dataset == "vat" else "scripts/train_gazefollow.py"
    eval_script = "scripts/eval_vat.py" if args.dataset == "vat" else "scripts/eval_gazefollow.py"

    train = [
        args.python,
        train_script,
        "--model",
        args.model,
        "--data_path",
        args.data_path,
        "--exp_name",
        run_name,
        "--run_dir",
        run_dir,
        "--max_epochs",
        str(args.max_epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(config.seed),
        "--spatial_prior",
        config.spatial_prior,
        "--fusion",
        config.fusion,
        "--selected_layers",
        config.selected_layers_label or ",".join(str(layer) for layer in config.selected_layers),
        "--wandb_mode",
        args.wandb_mode,
    ]
    if args.init_ckpt:
        train.extend(["--init_ckpt", args.init_ckpt])

    eval_cmd = [
        args.python,
        eval_script,
        "--data_path",
        args.data_path,
        "--variant_ckpt",
        ckpt_path,
        "--model",
        args.model,
        "--spatial_prior",
        config.spatial_prior,
        "--fusion",
        config.fusion,
        "--selected_layers",
        config.selected_layers_label or ",".join(str(layer) for layer in config.selected_layers),
        "--metrics_output",
        os.path.join(args.output_dir, f"{run_name}_metrics.json"),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.dataset == "vat":
        eval_cmd.extend(["--json_path", crowd_json])
    return {"train": train, "eval": eval_cmd, "checkpoint_path": ckpt_path}


def _run_name(group, variant, selected_layers_label, seed):
    layer_suffix = f"_{selected_layers_label}" if variant == "selected_layers" and selected_layers_label else ""
    return f"{group}_{variant}{layer_suffix}_seed{seed}"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    plan = build_run_plan(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.print_plan_only:
        plan_path = output_dir / "run_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + os.linesep)
        print(f"Wrote P1 dry-run plan: {plan_path}")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + os.linesep)
    metrics = build_placeholder_metrics(plan, status="smoke_only" if args.runner_smoke_only else "pending")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + os.linesep)
    csv_path = output_dir / "metrics.csv"
    write_metrics_csv(csv_path, metrics["rows"])
    if args.runner_smoke_only:
        print(f"Wrote runner smoke manifest: {manifest_path}")
        print(f"Wrote runner smoke metrics: {metrics_path}")
        print(f"Wrote runner smoke CSV: {csv_path}")
        return

    executed_metrics = execute_plan(plan)
    metrics_path.write_text(json.dumps(executed_metrics, indent=2, sort_keys=True) + os.linesep)
    write_metrics_csv(csv_path, executed_metrics["rows"])
    manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + os.linesep)
    print(f"Wrote execution manifest: {manifest_path}")
    print(f"Wrote execution metrics: {metrics_path}")
    print(f"Wrote execution CSV: {csv_path}")


def build_placeholder_metrics(plan, status):
    rows = []
    for run in plan["runs"]:
        metadata = run["metadata"]
        rows.append(
            {
                "dataset_split": metadata["dataset_split"],
                "variant": metadata["variant"],
                "spatial_prior": metadata["spatial_prior"],
                "fusion": metadata["fusion"],
                "selected_layers": metadata.get("selected_layers_label"),
                "seed": metadata["seed"],
                "checkpoint_path": run["commands"]["checkpoint_path"],
                "sample_count": metadata["sample_count"],
                "auc": "TBD",
                "l2": "TBD",
                "inout_ap": "TBD",
                "status": status,
            }
        )
    return {"section": "4.training_and_evaluation_runner", "rows": rows}


def write_metrics_csv(path, rows):
    fieldnames = [
        "dataset_split",
        "variant",
        "spatial_prior",
        "fusion",
        "seed",
        "status",
        "selected_layers",
        "checkpoint_path",
        "sample_count",
        "auc",
        "l2",
        "inout_ap",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def execute_plan(plan):
    rows = []
    for run in plan["runs"]:
        print(f"Running train command for {run['name']}")
        subprocess.run(run["commands"]["train"], check=True)
        run["status"] = "trained"
        print(f"Running eval command for {run['name']}")
        subprocess.run(run["commands"]["eval"], check=True)
        run["status"] = "evaluated"
        row = _load_metric_row(run)
        rows.append(row)
    return {"section": "4.training_and_evaluation_runner", "rows": rows}


def _load_metric_row(run):
    metadata = run["metadata"]
    metrics_path = None
    eval_command = run["commands"]["eval"]
    if "--metrics_output" in eval_command:
        metrics_path = eval_command[eval_command.index("--metrics_output") + 1]
    metrics = {}
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path) as handle:
            metrics = json.load(handle)
    return {
        "dataset_split": metadata["dataset_split"],
        "variant": metadata["variant"],
        "spatial_prior": metadata["spatial_prior"],
        "fusion": metadata["fusion"],
        "selected_layers": metadata.get("selected_layers_label"),
        "seed": metadata["seed"],
        "checkpoint_path": run["commands"]["checkpoint_path"],
        "sample_count": metrics.get("sample_count", metadata["sample_count"]),
        "auc": metrics.get("auc", "TBD"),
        "l2": metrics.get("l2", metrics.get("min_l2", "TBD")),
        "inout_ap": metrics.get("inout_ap", "TBD"),
        "status": run["status"],
    }


if __name__ == "__main__":
    main()
