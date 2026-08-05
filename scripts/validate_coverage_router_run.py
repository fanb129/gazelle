#!/usr/bin/env python3
"""Fail fast when a formal coverage-router run violates its fixed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("gazefollow", "vat"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--router_stage", required=True)
    parser.add_argument("--keep_ratio", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--router_trainable", required=True, type=int)
    parser.add_argument("--decoder_trainable", required=True, type=int)
    parser.add_argument("--inout_trainable", required=True, type=int)
    parser.add_argument("--evaluation_split", required=True)
    parser.add_argument("--split_seed", required=True, type=int)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--assignment_fingerprint")
    parser.add_argument("--source_annotation_sha256")
    parser.add_argument("--train_sample_count", type=int)
    parser.add_argument("--eval_sample_count", type=int)
    parser.add_argument("--train_group_count", type=int)
    parser.add_argument("--validation_group_count", type=int)
    parser.add_argument("--selection_metric", required=True)
    parser.add_argument("--selection_mode", required=True, choices=("min", "max"))
    parser.add_argument("--lr_router", type=float)
    parser.add_argument("--lr_decoder", type=float)
    parser.add_argument("--lr_inout", type=float)
    parser.add_argument("--router_warmup_epochs", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--frame_sample_every", type=int)
    parser.add_argument("--eval_frame_sample_every", type=int)
    parser.add_argument(
        "--reinitialize_router_on_init",
        required=True,
        choices=("true", "false"),
    )
    return parser.parse_args()


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    required_files = (
        "run_manifest.json",
        "data_split.json",
        "history.jsonl",
        "summary.json",
        "best_val_selection.pt",
        "last.resume.pt",
    )
    for filename in required_files:
        path = run_dir / filename
        if not path.is_file():
            raise RuntimeError(f"missing required run artifact: {path}")

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    train = manifest["train_config"]
    split = manifest["data_split"]
    counts = manifest["trainable_parameters"]
    require_equal("dataset", train["dataset"], args.dataset)
    require_equal("seed", int(train["seed"]), args.seed)
    require_equal("router_stage", train["router_stage"], args.router_stage)
    require_equal("keep_ratio", float(train["keep_ratio"]), args.keep_ratio)
    require_equal("max_epochs", int(train["max_epochs"]), args.epochs)
    require_equal("amp", bool(train["amp"]), False)
    require_equal("batch_size", int(train["batch_size"]), args.batch_size)
    require_equal(
        "eval_batch_size", int(train["eval_batch_size"]), args.eval_batch_size
    )
    require_equal(
        "reinitialize_router_on_init",
        bool(train["reinitialize_router_on_init"]),
        args.reinitialize_router_on_init == "true",
    )
    require_equal("router trainable", int(counts["router"]), args.router_trainable)
    require_equal(
        "decoder trainable", int(counts["decoder"]), args.decoder_trainable
    )
    require_equal("inout trainable", int(counts["inout"]), args.inout_trainable)
    require_equal("evaluation_split", split["evaluation_split"], args.evaluation_split)
    require_equal("selection_is_formal", bool(split["selection_is_formal"]), True)
    require_equal("split seed", int(split["seed"]), args.split_seed)
    init_path = Path(args.init_checkpoint).resolve()
    init_record = manifest["initialization_checkpoint"]
    require_equal("initialization checkpoint path", init_record["path"], str(init_path))
    require_equal(
        "initialization checkpoint sha256",
        init_record["sha256"],
        sha256_file(init_path),
    )
    selection = summary["selection"]
    require_equal("selection checkpoint", selection["checkpoint"], "best_val_selection.pt")
    require_equal("selection metric", selection["metric"], args.selection_metric)
    require_equal("selection mode", selection["mode"], args.selection_mode)
    optional_train_expectations = {
        "lr_router": args.lr_router,
        "lr_decoder": args.lr_decoder,
        "lr_inout": args.lr_inout,
        "router_warmup_epochs": args.router_warmup_epochs,
        "grad_accum_steps": args.grad_accum_steps,
        "frame_sample_every": args.frame_sample_every,
        "eval_frame_sample_every": args.eval_frame_sample_every,
    }
    for key, expected in optional_train_expectations.items():
        if expected is not None:
            require_equal(key, train[key], expected)
    if args.assignment_fingerprint is not None:
        require_equal(
            "assignment_fingerprint",
            split["assignment_fingerprint"],
            args.assignment_fingerprint,
        )
    optional_manifest_expectations = {
        "train_sample_count": args.train_sample_count,
        "eval_sample_count": args.eval_sample_count,
    }
    for key, expected in optional_manifest_expectations.items():
        if expected is not None:
            require_equal(key, int(manifest[key]), expected)
    optional_split_expectations = {
        "source_annotation_sha256": args.source_annotation_sha256,
        "train_group_count": args.train_group_count,
        "validation_group_count": args.validation_group_count,
    }
    for key, expected in optional_split_expectations.items():
        if expected is not None:
            require_equal(key, split[key], expected)

    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require_equal("history length", len(history), args.epochs)
    require_equal(
        "history epochs",
        [int(row["epoch"]) for row in history],
        list(range(args.epochs)),
    )
    print(f"FORMAL_RUN_VALIDATION passed: {run_dir}")


if __name__ == "__main__":
    main()
