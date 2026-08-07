#!/usr/bin/env python3
"""Validate a fixed-epoch, full-train coverage-router refit artifact.

This validator is intentionally separate from ``validate_coverage_router_run``.
The latter validates validation-selected runs, while a formal refit must never
construct an evaluation loader or write a best-selection checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional


CHECKPOINT_FORMAT_VERSION = 1
FULL_TRAIN_STRATEGY = "gazefollow_full_train_no_eval_v1"
FULL_TRAIN_SELECTION_POLICY = "fixed_final_epoch_no_validation"
DEFAULT_SOURCE_ANNOTATION_SHA256 = (
    "44f1b1e76da9e7acc2b44bd5e1d97b957bea756bd9e1c1c116457b7daa99e0e9"
)
DEFAULT_TRAIN_SAMPLE_COUNT = 113458
DEFAULT_TRAIN_RECORD_COUNT = 117727
DEFAULT_TRAIN_GROUP_COUNT = 117727


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a formal full-train/no-eval coverage-router run."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", default="gazefollow", choices=("gazefollow",))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--router_stage", required=True)
    parser.add_argument("--keep_ratio", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--router_trainable", required=True, type=int)
    parser.add_argument("--decoder_trainable", required=True, type=int)
    parser.add_argument("--inout_trainable", required=True, type=int)
    parser.add_argument("--backbone_trainable", type=int, default=0)
    parser.add_argument(
        "--init_checkpoint",
        default="none",
        help="Expected initialization checkpoint, or 'none' for the dense base.",
    )
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--lr_router", type=float)
    parser.add_argument("--lr_decoder", type=float)
    parser.add_argument("--router_warmup_epochs", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--reinitialize_router_on_init",
        choices=("true", "false"),
        default="false",
    )
    parser.add_argument(
        "--source_annotation_sha256",
        default=DEFAULT_SOURCE_ANNOTATION_SHA256,
    )
    parser.add_argument(
        "--train_sample_count", type=int, default=DEFAULT_TRAIN_SAMPLE_COUNT
    )
    parser.add_argument(
        "--train_record_count", type=int, default=DEFAULT_TRAIN_RECORD_COUNT
    )
    parser.add_argument(
        "--train_group_count", type=int, default=DEFAULT_TRAIN_GROUP_COUNT
    )
    parser.add_argument("--final_checkpoint", default="final.pt")
    parser.add_argument("--checkpoint_role", default="fixed_epoch_final")
    return parser.parse_args(argv)


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require_close(label: str, actual, expected, *, tolerance: float = 1e-12) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{label}: expected numeric {expected!r}, got {actual!r}"
        ) from error
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{label}: expected {expected_value!r}, got {actual_value!r}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required to inspect the structured refit checkpoint"
        ) from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not a structured dictionary: {path}")
    return payload


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _read_history(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"history line {line_number} is not valid JSON: {path}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"history line {line_number} must be an object")
        rows.append(row)
    return rows


def _require_finite_tree(label: str, value) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{label} is not finite: {value!r}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_tree(f"{label}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(f"{label}[{index}]", child)
        return
    raise RuntimeError(f"{label} has unsupported value type: {type(value).__name__}")


def _expected_keep_ratio(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 1:
        return target
    progress = min(1.0, float(epoch) / float(warmup_epochs - 1))
    return 1.0 + progress * (target - 1.0)


def _resolve_init_checkpoint(raw_value: Optional[str]) -> Optional[Path]:
    if raw_value is None or raw_value.strip().lower() in ("", "none", "null"):
        return None
    return Path(raw_value).resolve()


def _validate_initialization(
    manifest: dict,
    expected_path: Optional[Path],
) -> None:
    train = manifest["train_config"]
    record = manifest.get("initialization_checkpoint")
    require_equal("manifest train resume", train.get("resume"), None)
    if expected_path is None:
        require_equal("train init_ckpt", train.get("init_ckpt"), None)
        require_equal("initialization checkpoint", record, None)
        return

    if not expected_path.is_file():
        raise RuntimeError(
            f"expected initialization checkpoint is missing: {expected_path}"
        )
    expected_sha256 = sha256_file(expected_path)
    train_init = train.get("init_ckpt")
    if train_init is None:
        raise RuntimeError("refit manifest train_config.init_ckpt is missing")
    require_equal(
        "train init_ckpt",
        str(Path(train_init).resolve()),
        str(expected_path),
    )
    if not isinstance(record, dict):
        raise RuntimeError("manifest initialization_checkpoint must be an object")
    require_equal(
        "initialization checkpoint path", record.get("path"), str(expected_path)
    )
    require_equal(
        "initialization checkpoint sha256",
        record.get("sha256"),
        expected_sha256,
    )


def _validate_split(split: dict, args: argparse.Namespace) -> None:
    expectations = {
        "strategy": FULL_TRAIN_STRATEGY,
        "evaluation_split": None,
        "evaluation_mode": "none",
        "selection_policy": FULL_TRAIN_SELECTION_POLICY,
        "selection_is_formal": False,
        "provenance_requires_match": True,
        "official_test_accessed": False,
        "validation_fraction": 0.0,
        "group_key": "normalized_image_path",
        "source_annotation_sha256": args.source_annotation_sha256,
        "train_record_count": args.train_record_count,
        "train_head_sample_count": args.train_sample_count,
        "train_group_count": args.train_group_count,
        "validation_record_count": 0,
        "validation_head_sample_count": 0,
    }
    for key, expected in expectations.items():
        require_equal(f"data_split.{key}", split.get(key), expected)
    for key in (
        "source_group_fingerprint",
        "train_group_fingerprint",
        "assignment_fingerprint",
    ):
        value = split.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"data_split.{key} must be a SHA256 fingerprint")


def _validate_manifest(
    run_dir: Path,
    manifest: dict,
    split: dict,
    args: argparse.Namespace,
    expected_init: Optional[Path],
) -> None:
    require_equal("manifest format_version", manifest.get("format_version"), 1)
    require_equal("manifest dataset", manifest.get("dataset"), args.dataset)
    require_equal("manifest git_commit", manifest.get("git_commit"), args.git_commit)
    require_equal(
        "manifest train_sample_count",
        manifest.get("train_sample_count"),
        args.train_sample_count,
    )
    require_equal("manifest eval_sample_count", manifest.get("eval_sample_count"), None)
    require_equal("manifest eval_image_count", manifest.get("eval_image_count"), None)
    require_equal("manifest data_split", manifest.get("data_split"), split)

    train = manifest.get("train_config")
    model = manifest.get("model_config")
    counts = manifest.get("trainable_parameters")
    if (
        not isinstance(train, dict)
        or not isinstance(model, dict)
        or not isinstance(counts, dict)
    ):
        raise RuntimeError("manifest is missing train/model/parameter metadata")

    train_expectations = {
        "dataset": args.dataset,
        "seed": args.seed,
        "router_stage": args.router_stage,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "amp": False,
        "formal_full_train_no_eval": True,
        "gazefollow_val_fraction": 0.0,
        "vat_val_fraction": 0.0,
        "eval_batch_size": None,
        "max_eval_batches": None,
        "max_train_batches": None,
        "save_every": 0,
        "allow_init_without_split_provenance": False,
        "reinitialize_router_on_init": (args.reinitialize_router_on_init == "true"),
        "router_warmup_epochs": args.router_warmup_epochs,
        "grad_accum_steps": args.grad_accum_steps,
        "lr_backbone": 0.0,
        "lr_inout": 0.0,
        "weight_decay": 0.0,
    }
    for key, expected in train_expectations.items():
        require_equal(f"train_config.{key}", train.get(key), expected)
    require_close("train_config.keep_ratio", train.get("keep_ratio"), args.keep_ratio)
    if args.lr_router is not None:
        require_close("train_config.lr_router", train.get("lr_router"), args.lr_router)
    if args.lr_decoder is not None:
        require_close(
            "train_config.lr_decoder", train.get("lr_decoder"), args.lr_decoder
        )
    require_equal(
        "train_config.run_dir", str(Path(train["run_dir"]).resolve()), str(run_dir)
    )

    model_expectations = {
        "model": "gazelle_dinov3_vitb16",
        "router_stage": args.router_stage,
        "route_after_block": 5,
        "router_hidden_dim": 256,
        "router_temperature": 1.0,
        "escape_tokens": 8,
        "spatial_prior": "none",
        "fusion": "raw_concat",
    }
    for key, expected in model_expectations.items():
        require_equal(f"model_config.{key}", model.get(key), expected)
    require_close("model_config.keep_ratio", model.get("keep_ratio"), args.keep_ratio)

    count_expectations = {
        "router": args.router_trainable,
        "decoder": args.decoder_trainable,
        "inout": args.inout_trainable,
        "backbone": args.backbone_trainable,
    }
    for key, expected in count_expectations.items():
        require_equal(f"trainable_parameters.{key}", counts.get(key), expected)
    _validate_initialization(manifest, expected_init)


def _validate_history(
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    require_equal("history length", len(history), args.epochs)
    require_equal(
        "history epochs",
        [row.get("epoch") for row in history],
        list(range(args.epochs)),
    )
    expected_keys = {"epoch", "active_train_keep_ratio", "train"}
    for epoch, row in enumerate(history):
        require_equal(f"history epoch {epoch} keys", set(row), expected_keys)
        train_metrics = row.get("train")
        if not isinstance(train_metrics, dict) or not train_metrics:
            raise RuntimeError(f"history epoch {epoch} train metrics must be non-empty")
        _require_finite_tree(f"history[{epoch}].train", train_metrics)
        expected_keep = _expected_keep_ratio(
            args.keep_ratio,
            epoch,
            args.router_warmup_epochs,
        )
        require_close(
            f"history epoch {epoch} active_train_keep_ratio",
            row.get("active_train_keep_ratio"),
            expected_keep,
        )
        if "router_actual_keep_ratio" in train_metrics:
            require_close(
                f"history epoch {epoch} router_actual_keep_ratio",
                train_metrics["router_actual_keep_ratio"],
                expected_keep,
                tolerance=1e-9,
            )


def _validate_summary(
    summary: dict,
    args: argparse.Namespace,
) -> None:
    require_equal(
        "summary checkpoint_policy",
        summary.get("checkpoint_policy"),
        FULL_TRAIN_SELECTION_POLICY,
    )
    require_equal("summary selection", summary.get("selection"), None)
    require_equal("summary best_metrics", summary.get("best_metrics"), None)
    require_equal("summary evaluation_split", summary.get("evaluation_split"), None)
    require_equal(
        "summary evaluation_performed", summary.get("evaluation_performed"), False
    )
    require_equal("summary last_epoch", summary.get("last_epoch"), args.epochs - 1)
    require_equal(
        "summary final_checkpoint",
        summary.get("final_checkpoint"),
        args.final_checkpoint,
    )
    require_equal("summary completed", summary.get("completed"), True)


def _validate_checkpoint_payload(
    label: str,
    payload: dict,
    *,
    expected_role: str,
    expected_epoch: int,
    manifest: dict,
    split: dict,
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} checkpoint must be a structured dictionary")
    require_equal(
        f"{label} format_version",
        payload.get("format_version"),
        CHECKPOINT_FORMAT_VERSION,
    )
    require_equal(
        f"{label} checkpoint_role", payload.get("checkpoint_role"), expected_role
    )
    require_equal(f"{label} epoch", payload.get("epoch"), expected_epoch)
    require_equal(f"{label} git_commit", payload.get("git_commit"), args.git_commit)
    require_equal(f"{label} data_split", payload.get("data_split"), split)
    require_equal(f"{label} selection", payload.get("selection"), None)
    require_equal(f"{label} best_metrics", payload.get("best_metrics"), None)
    require_equal(
        f"{label} state_scope",
        payload.get("state_scope"),
        "all_non_backbone_and_all_trainable_backbone_parameters",
    )
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"{label} model_state must be a non-empty mapping")
    for state_name in (
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
    ):
        if not isinstance(payload.get(state_name), dict):
            raise RuntimeError(f"{label} {state_name} must be a mapping")
    require_equal(
        f"{label} initialization_checkpoint",
        payload.get("initialization_checkpoint"),
        manifest.get("initialization_checkpoint"),
    )
    require_equal(f"{label} metrics", payload.get("metrics"), history[expected_epoch])

    model = payload.get("model_config", {})
    train = payload.get("train_config", {})
    require_equal(
        f"{label} model router_stage", model.get("router_stage"), args.router_stage
    )
    require_close(f"{label} model keep_ratio", model.get("keep_ratio"), args.keep_ratio)
    require_equal(f"{label} train seed", train.get("seed"), args.seed)
    require_equal(
        f"{label} train formal mode", train.get("formal_full_train_no_eval"), True
    )
    require_equal(f"{label} train max_epochs", train.get("max_epochs"), args.epochs)


def _validate_absent_eval_and_best_artifacts(run_dir: Path) -> None:
    forbidden = []
    for pattern in ("best*.pt", "eval*.json", "evaluation*.json"):
        forbidden.extend(sorted(run_dir.glob(pattern)))
    if forbidden:
        rendered = ", ".join(str(path) for path in forbidden)
        raise RuntimeError(
            f"formal refit contains forbidden eval/best artifacts: {rendered}"
        )


def validate_refit_run(
    args: argparse.Namespace,
    *,
    checkpoint_loader: Callable[[Path], dict] = load_checkpoint,
) -> dict:
    run_dir = Path(args.run_dir).resolve()
    required_files = (
        "run_manifest.json",
        "data_split.json",
        "history.jsonl",
        "summary.json",
        args.final_checkpoint,
        "last.resume.pt",
    )
    for filename in required_files:
        path = run_dir / filename
        if not path.is_file():
            raise RuntimeError(f"missing required refit artifact: {path}")
    _validate_absent_eval_and_best_artifacts(run_dir)

    manifest = _read_json(run_dir / "run_manifest.json")
    split = _read_json(run_dir / "data_split.json")
    summary = _read_json(run_dir / "summary.json")
    history = _read_history(run_dir / "history.jsonl")
    expected_init = _resolve_init_checkpoint(args.init_checkpoint)

    _validate_split(split, args)
    _validate_manifest(run_dir, manifest, split, args, expected_init)
    _validate_history(history, args)
    _validate_summary(summary, args)

    final_path = run_dir / args.final_checkpoint
    final_payload = checkpoint_loader(final_path)
    _validate_checkpoint_payload(
        "final",
        final_payload,
        expected_role=args.checkpoint_role,
        expected_epoch=args.epochs - 1,
        manifest=manifest,
        split=split,
        history=history,
        args=args,
    )
    resume_payload = checkpoint_loader(run_dir / "last.resume.pt")
    _validate_checkpoint_payload(
        "last.resume",
        resume_payload,
        expected_role="resume_checkpoint",
        expected_epoch=args.epochs - 1,
        manifest=manifest,
        split=split,
        history=history,
        args=args,
    )

    result = {
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "final_epoch": args.epochs - 1,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "git_commit": args.git_commit,
    }
    print(
        "FORMAL_REFIT_VALIDATION passed: "
        f"{run_dir} final_sha256={result['final_checkpoint_sha256']}"
    )
    return result


def main(argv: Optional[Iterable[str]] = None) -> None:
    validate_refit_run(parse_args(argv))


if __name__ == "__main__":
    main()
