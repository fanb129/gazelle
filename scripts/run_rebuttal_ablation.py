import argparse
import json
import os
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
        description="Build P1 rebuttal ablation run plans. Section 1 supports dry-run only."
    )
    parser.add_argument("--print_plan_only", action="store_true", help="Validate arguments and write a run plan without training.")
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
                    crowd_json=args.crowd_json,
                )
                runs.append(
                    {
                        "name": _run_name(args.group, variant, config.selected_layers_label, seed),
                        "metadata": metadata.to_dict(),
                        "legacy_flags": {
                            "use_sasa": config.use_sasa,
                            "use_ggsf": config.use_ggsf,
                        },
                        "status": "planned",
                    }
                )

    return {
        "print_plan_only": bool(args.print_plan_only),
        "section": "1.variant_configuration_and_metadata",
        "group": args.group,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "crowd_json": args.crowd_json,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "runs": runs,
        "note": "Section 1 validates configuration and metadata only; training/evaluation execution is added in later gated sections.",
    }


def _run_name(group, variant, selected_layers_label, seed):
    layer_suffix = f"_{selected_layers_label}" if variant == "selected_layers" and selected_layers_label else ""
    return f"{group}_{variant}{layer_suffix}_seed{seed}"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.print_plan_only:
        parser.error("Section 1 implements dry-run planning only; pass --print_plan_only.")

    plan = build_run_plan(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + os.linesep)
    print(f"Wrote P1 dry-run plan: {plan_path}")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
