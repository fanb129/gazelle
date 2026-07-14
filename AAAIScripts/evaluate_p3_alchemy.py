#!/usr/bin/env python3
"""Run one full post-training test evaluation for a P3 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIAlchemyModels.factory import CANDIDATE_CHOICES, resolve_candidate
from AAAIScripts.common import file_manifest


def overall_metrics(records, dataset: str) -> dict:
    def mean(field):
        values = [float(row[field]) for row in records if row.get(field) is not None]
        return float(np.mean(values)) if values else None

    metrics = {"auc": mean("auc")}
    if dataset == "vat":
        from sklearn.metrics import average_precision_score

        labels = [int(row["inout"]) for row in records]
        scores = [float(row["inout_score"]) for row in records]
        metrics.update({
            "l2": mean("l2"),
            "inout_ap": float(average_precision_score(labels, scores))
            if len(set(labels)) > 1 else None,
            "person_records": len(records),
            "inframe_records": sum(labels),
        })
    else:
        metrics.update({
            "avg_l2": mean("avg_l2"),
            "min_l2": mean("min_l2"),
            "person_records": len(records),
        })
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return

    from AAAIAlchemyModels.factory import build_p3_alchemy_model
    from AAAIScripts import failure_taxonomy as taxonomy

    model, transform, candidate = build_p3_alchemy_model(args.candidate, dataset=args.dataset)
    load_report = model.load_alchemy_checkpoint(args.checkpoint)
    if load_report["missing"] or load_report["unexpected"] or load_report["incompatible_shapes"]:
        raise RuntimeError(f"evaluation checkpoint is incomplete or incompatible: {load_report}")

    json_path = args.data_path / "test_preprocessed.json"
    taxonomy_args = SimpleNamespace(
        dataset=args.dataset,
        data_path=str(args.data_path),
        json_path=str(json_path),
        checkpoint=str(args.checkpoint),
        model_label=args.model_label or candidate.label,
        device=args.device,
        max_frames=args.max_frames,
        shared_target_radius=args.shared_target_radius,
        collapse_cosine=args.collapse_cosine,
    )
    records, frames = taxonomy.evaluate(
        taxonomy_args, loaded_model=(model, transform, load_report)
    )
    summary = taxonomy.summarize(records, frames)
    prefix = args.output_prefix
    taxonomy.write_csv(prefix.with_suffix(".records.csv"), records, taxonomy.PERSON_FIELDS)
    taxonomy.write_csv(prefix.with_suffix(".frames.csv"), frames, taxonomy.FRAME_FIELDS)
    taxonomy.write_csv(
        prefix.with_suffix(".summary.csv"), summary,
        sorted({key for row in summary for key in row}),
    )
    report = {
        "status": "measured",
        "phase": args.phase,
        "selection_role": "single_post_training_full_test_evaluation",
        "test_used_for_training_or_epoch_selection": False,
        "dataset": args.dataset,
        "candidate": candidate.to_dict(),
        "seed": args.seed,
        "checkpoint": file_manifest(args.checkpoint),
        "checkpoint_load": load_report,
        "annotation": file_manifest(json_path),
        "num_person_records": len(records),
        "num_frame_records": len(frames),
        "metrics": overall_metrics(records, args.dataset),
        "summary": summary,
    }
    report_path = prefix.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"P3 evaluation complete: {json.dumps(report['metrics'], sort_keys=True)}")


def run_self_test() -> None:
    vat = [
        {"auc": 0.8, "l2": 0.2, "inout": 1, "inout_score": 0.9},
        {"auc": None, "l2": None, "inout": 0, "inout_score": 0.1},
    ]
    metrics = overall_metrics(vat, "vat")
    assert metrics["auc"] == 0.8 and metrics["l2"] == 0.2
    assert metrics["inout_ap"] == 1.0 and metrics["person_records"] == 2
    assert resolve_candidate("r3").coordinate_loss_weight == 0.05
    print("Self-test passed: full-test metric aggregation and candidate mapping.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("p3b", "p3c", "p3d", "smoke"), default="p3b")
    parser.add_argument("--candidate", choices=CANDIDATE_CHOICES)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--model-label")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--collapse-cosine", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    required = ("candidate", "dataset", "data_path", "checkpoint", "output_prefix")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("normal evaluation requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    return args


if __name__ == "__main__":
    main()
