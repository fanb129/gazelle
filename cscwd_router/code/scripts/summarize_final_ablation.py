"""Assemble the conditional-scoring table from verified completed branches on CPU."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag import final_io as io
from cscwd_router.code.stag.conditional import CONDITIONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("report-root", "full-gf-summary", "full-vat-summary", "output", "provenance"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args(argv)


def run(args):
    full = {"gazefollow": io.verified_source(args.full_gf_summary), "vat": io.verified_source(args.full_vat_summary)}
    refs, rows = {}, []
    for condition in ("scene_only", "scene_geometry", "scene_head", "full"):
        paths = {dataset: (args.full_gf_summary if dataset == "gazefollow" else args.full_vat_summary)
                 if condition == "full" else args.report_root / condition / dataset / "vitb/summary.json"
                 for dataset in ("gazefollow", "vat")}
        row = {"condition": condition, "head_appearance": CONDITIONS[condition][0], "geometry": CONDITIONS[condition][1]}
        if not all(path.exists() for path in paths.values()):
            if not args.allow_incomplete:
                raise FileNotFoundError(f"conditional branch is not complete: {condition}")
            row["completed"] = False
            rows.append(row)
            continue
        loaded = {dataset: io.verified_source(path) for dataset, path in paths.items()}
        for dataset, (source, manifest, _) in loaded.items():
            reference, reference_manifest, _ = full[dataset]
            if source["model_config"].get("routing_condition", "full") != condition:
                raise ValueError("table row and saved routing condition differ")
            config = {k: v for k, v in source["model_config"].items() if k != "routing_condition"}
            if config != reference["model_config"] or source["size"] != "b":
                raise ValueError("ablation representations or routing budgets differ")
            if source["selection_policy"] != "test_selected" or manifest["recipe"]["seed"] != 0:
                raise ValueError("ablation selection or seed differs")
            if manifest["data"]["test_annotation"] != reference_manifest["data"]["test_annotation"]:
                raise ValueError("ablation test annotations differ")
            if (source["metrics"]["persons"], source["metrics"]["inframe_persons"]) != (
                    reference["metrics"]["persons"], reference["metrics"]["inframe_persons"]):
                raise ValueError("ablation test queries differ")
            if dataset == "gazefollow":
                if source["training_epochs"] != {"support": 3, "sparse_decoder": 5}:
                    raise ValueError("GF ablation update budget differs")
                if manifest["initialization"]["sha256"] != reference_manifest["initialization"]["sha256"]:
                    raise ValueError("GF decoder initializations differ")
                recipe_keys = ("support_epochs", "decoder_epochs", "batch_size", "grad_accumulation", "router_lr", "decoder_lr",
                    "inout_lr", "weight_decay", "clip_norm", "budget_weight", "router_hidden_dim", "router_temperature", "escape_tokens", "seed")
            else:
                if source["epochs"] != 8 or source["router_warmup_epochs"] != 1:
                    raise ValueError("VAT transfer update budget differs")
                if manifest["gf_checkpoint"] != loaded["gazefollow"][0]["artifacts"]["best_test_checkpoint"]:
                    raise ValueError("VAT ablation did not inherit its own GF branch")
                if not source["initialization"]["shared_tensors_identical"]:
                    raise ValueError("GF-to-VAT shared tensors were not preserved")
                recipe_keys = ("epochs", "router_warmup_epochs", "batch_size", "grad_accumulation", "router_lr", "decoder_lr",
                    "inout_lr", "router_loss_weight", "budget_weight", "clip_norm", "train_frame_rate", "test_frame_rate", "seed")
            if any(manifest["recipe"][key] != reference_manifest["recipe"][key] for key in recipe_keys):
                raise ValueError("ablation optimization recipes differ from the full reference")
            refs[f"{condition}_{dataset}"] = io.reference(paths[dataset])
            row[dataset] = source["metrics"]
            row.setdefault("selection", {})[dataset] = source["selection"]
        row["completed"] = True
        rows.append(row)
    result = {"request": {"protocol": "conditional_scoring_table_v1", "reports": refs,
        "sources": io.sources("cscwd_router/code/scripts/summarize_final_ablation.py")},
        "all_branches_complete": all(row["completed"] for row in rows), "rows": rows, "artifacts": {}}
    saved = io.finish(args.output, result, args.provenance, "Final conditional-scoring table")
    lines = ["# Conditional scoring ablation (ViT-B, K50)", "", "| Scene | Head appearance | Geometry | GF AUC | GF Avg L2 | GF GT coverage | VAT AUC | VAT L2 | VAT AP | VAT GT coverage |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        flags = ["✓", "✓" if row["head_appearance"] else "×", "✓" if row["geometry"] else "×"]
        if row["completed"]:
            gf, vat = row["gazefollow"], row["vat"]
            values = [gf["auc"], gf["avg_l2"], gf["routing_gt_point_coverage"], vat["auc"], vat["l2"], vat["inout_ap"], vat["routing_gt_point_coverage"]]
            cells = [f"{value:.6f}" for value in values]
        else:
            cells = ["—"] * 7
        lines.append("| " + " | ".join(flags + cells) + " |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"COMPLETE {args.output}", flush=True)
    return saved


if __name__ == "__main__":
    run(parse_args())
