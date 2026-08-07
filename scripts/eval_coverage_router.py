"""Evaluate a structured coverage-router checkpoint on GazeFollow or VAT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Optional

import torch

from gazelle.data_splits import grouped_holdout_split
from gazelle.dataloader import (
    GazeDataset,
    GazeFollowImageDataset,
    collate_fn,
    collate_gazefollow_images,
    load_data_gazefollow,
)
from gazelle.routing.model import ROUTER_STAGES, get_coverage_router_model

try:  # Works both as ``python scripts/eval_...py`` and as a module import.
    from train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        DEFAULT_DATA_PATHS,
        _gazefollow_group_key,
        _torch_load,
        evaluate_model,
        load_model_state,
    )
except ModuleNotFoundError:
    from scripts.train_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        DEFAULT_DATA_PATHS,
        _gazefollow_group_key,
        _torch_load,
        evaluate_model,
        load_model_state,
    )


EVALUATION_RESULT_FORMAT_VERSION = 1
FORMAL_FULL_TRAIN_STRATEGIES = {
    "gazefollow": "gazefollow_full_train_no_eval_v1",
}
OFFICIAL_EVALUATION_STRATEGIES = {
    "gazefollow": "gazefollow_official_test",
    "vat": "vat_official_test",
}
EVALUATION_CONFIG_FIELDS = (
    "checkpoint",
    "dataset",
    "data_path",
    "router_stage_override",
    "keep_ratio_override",
    "batch_size",
    "n_workers",
    "gazefollow_eval_unit",
    "gazefollow_eval_split",
    "gazefollow_val_fraction",
    "gazefollow_split_seed",
    "gazefollow_head_count_subset",
    "frame_sample_every",
    "max_eval_batches",
    "device",
    "amp",
    "output",
    "require_full_train_no_eval",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_git_metadata() -> dict:
    """Return the exact source revision used by this evaluator."""

    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_full_train_no_eval_checkpoint(
    checkpoint: dict,
    *,
    dataset: str,
) -> None:
    """Reject anything except a fixed-final checkpoint from a no-eval refit."""

    errors = []
    if dataset not in FORMAL_FULL_TRAIN_STRATEGIES:
        errors.append(
            f"no full-train/no-eval fixed-final contract exists for {dataset!r}"
        )
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        errors.append("unsupported or missing checkpoint format_version")
    if checkpoint.get("checkpoint_role") != "fixed_epoch_final":
        errors.append("checkpoint_role must be 'fixed_epoch_final'")
    if checkpoint.get("selection") is not None:
        errors.append("selection must be null (no validation-based selection)")
    if checkpoint.get("best_metrics") is not None:
        errors.append("best_metrics must be null (no validation metrics)")
    if checkpoint.get("state_scope") != (
        "all_non_backbone_and_all_trainable_backbone_parameters"
    ):
        errors.append("state_scope is missing or unsupported")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        errors.append("model_state must be a non-empty mapping")

    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, dict):
        errors.append("train_config must be a mapping")
        train_config = {}
    if train_config.get("dataset") != dataset:
        errors.append(
            "train_config.dataset does not match the requested evaluation dataset"
        )
    if train_config.get("formal_full_train_no_eval") is not True:
        errors.append("train_config.formal_full_train_no_eval must be true")
    for field in ("gazefollow_val_fraction", "vat_val_fraction"):
        if field not in train_config or train_config.get(field) != 0.0:
            errors.append(f"train_config.{field} must be zero")
    for field in ("max_train_batches", "max_eval_batches", "eval_batch_size"):
        if field not in train_config or train_config.get(field) is not None:
            errors.append(f"train_config.{field} must be null")
    if "save_every" not in train_config or train_config.get("save_every") != 0:
        errors.append("train_config.save_every must be zero")

    max_epochs = train_config.get("max_epochs")
    epoch = checkpoint.get("epoch")
    if not _positive_int(max_epochs):
        errors.append("train_config.max_epochs must be a positive integer")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        errors.append("checkpoint epoch must be an integer")
    elif _positive_int(max_epochs) and epoch != max_epochs - 1:
        errors.append(
            f"checkpoint epoch {epoch} is not the fixed final epoch {max_epochs - 1}"
        )

    data_split = checkpoint.get("data_split")
    if not isinstance(data_split, dict):
        errors.append("data_split must be a mapping")
        data_split = {}
    expected_split = {
        "strategy": FORMAL_FULL_TRAIN_STRATEGIES.get(dataset),
        "evaluation_split": None,
        "evaluation_mode": "none",
        "selection_policy": "fixed_final_epoch_no_validation",
        "selection_is_formal": False,
        "provenance_requires_match": True,
        "official_test_accessed": False,
        "validation_fraction": 0.0,
    }
    for field, expected in expected_split.items():
        if field not in data_split or data_split.get(field) != expected:
            errors.append(f"data_split.{field} must equal {expected!r}")
    for field in ("validation_record_count", "validation_head_sample_count"):
        if field not in data_split or data_split.get(field) != 0:
            errors.append(f"data_split.{field} must be zero")
    if not _valid_sha256(data_split.get("source_annotation_sha256")):
        errors.append("data_split.source_annotation_sha256 must be a SHA-256 digest")
    if not isinstance(data_split.get("source_annotation_file"), str) or not (
        data_split["source_annotation_file"].strip()
    ):
        errors.append("data_split.source_annotation_file must be recorded")
    for field in (
        "train_record_count",
        "train_head_sample_count",
        "train_group_count",
    ):
        if not _positive_int(data_split.get(field)):
            errors.append(f"data_split.{field} must be a positive integer")
    if data_split.get("group_key") != "normalized_image_path":
        errors.append("data_split.group_key must equal 'normalized_image_path'")
    for field in (
        "source_group_fingerprint",
        "train_group_fingerprint",
        "assignment_fingerprint",
    ):
        if not _valid_sha256(data_split.get(field)):
            errors.append(f"data_split.{field} must be a SHA-256 digest")

    epoch_metrics = checkpoint.get("metrics")
    if not isinstance(epoch_metrics, dict):
        errors.append("checkpoint metrics must be a mapping")
    else:
        if "eval" in epoch_metrics:
            errors.append("checkpoint metrics unexpectedly contain an eval section")
        if epoch_metrics.get("epoch") != epoch:
            errors.append("checkpoint metrics epoch does not match checkpoint epoch")

    git_commit = checkpoint.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit.strip():
        errors.append("checkpoint git_commit must be recorded")

    if errors:
        rendered = "\n- ".join(errors)
        raise ValueError(
            "checkpoint is not a valid full-train/no-eval fixed-final artifact:\n"
            f"- {rendered}"
        )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a coverage-aware gaze router."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("gazefollow", "vat"), default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--router_stage_override",
        choices=ROUTER_STAGES,
        default=None,
        help=(
            "Evaluation-only override for diagnostic controls. For example, "
            "evaluate a support_pilot checkpoint through backbone_sparse "
            "without taking an optimizer step."
        ),
    )
    parser.add_argument(
        "--keep_ratio_override",
        type=float,
        default=None,
        help=(
            "Evaluation-only sparse keep-ratio override. This does not alter "
            "the checkpoint on disk."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--gazefollow_eval_unit",
        choices=("person", "image"),
        default="person",
        help=(
            "person preserves the legacy one-head-per-image evaluation; image "
            "queries every annotated head together and exercises the shared "
            "per-image router union."
        ),
    )
    parser.add_argument(
        "--gazefollow_eval_split",
        choices=("official_test", "train_holdout"),
        default="official_test",
        help=(
            "Evaluate the official test split or a deterministic image-level "
            "holdout from train_preprocessed.json."
        ),
    )
    parser.add_argument(
        "--gazefollow_val_fraction",
        "--gf_val_fraction",
        dest="gazefollow_val_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--gazefollow_split_seed",
        "--gf_split_seed",
        dest="gazefollow_split_seed",
        type=int,
        default=3106,
    )
    parser.add_argument(
        "--gazefollow_head_count_subset",
        choices=("all", "single", "multi", "two", "three_plus"),
        default="all",
        help="Optionally evaluate only images in one in-frame query-count stratum.",
    )
    parser.add_argument(
        "--frame_sample_every",
        type=int,
        default=1,
        help="VAT evaluation sampling rate. The default 1 evaluates every test frame.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require_full_train_no_eval",
        action="store_true",
        help=(
            "Formal official-test guard: require a full-training-set checkpoint "
            "saved at its predeclared final epoch without any validation or test "
            "evaluation during training. Diagnostic overrides and partial test "
            "evaluation are rejected."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.n_workers < 0:
        raise ValueError("--n_workers must be non-negative")
    if args.frame_sample_every <= 0:
        raise ValueError("--frame_sample_every must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max_eval_batches must be positive")
    if (
        args.gazefollow_eval_split == "train_holdout"
        and not 0.0 < args.gazefollow_val_fraction < 1.0
    ):
        raise ValueError("--gazefollow_val_fraction must be in (0, 1)")
    if (
        args.keep_ratio_override is not None
        and not 0.0 < args.keep_ratio_override <= 1.0
    ):
        raise ValueError("--keep_ratio_override must be in (0, 1]")


def validate_formal_evaluation_request(
    args: argparse.Namespace,
    checkpoint: dict,
    *,
    runtime_git: dict,
) -> None:
    if not args.require_full_train_no_eval:
        return

    validate_full_train_no_eval_checkpoint(checkpoint, dataset=args.dataset)
    if args.router_stage_override is not None or args.keep_ratio_override is not None:
        raise ValueError(
            "--require_full_train_no_eval forbids evaluation-only model overrides"
        )
    if args.max_eval_batches is not None:
        raise ValueError(
            "--require_full_train_no_eval requires the complete official test set"
        )
    if args.dataset == "gazefollow":
        if args.gazefollow_eval_split != "official_test":
            raise ValueError(
                "--require_full_train_no_eval requires --gazefollow_eval_split "
                "official_test"
            )
        if args.gazefollow_head_count_subset != "all":
            raise ValueError(
                "--require_full_train_no_eval requires "
                "--gazefollow_head_count_subset all"
            )
    elif args.frame_sample_every != 1:
        raise ValueError(
            "--require_full_train_no_eval requires --frame_sample_every 1 for VAT"
        )
    if not args.output:
        raise ValueError(
            "--require_full_train_no_eval requires --output so the formal result "
            "and its provenance are preserved"
        )
    if Path(args.output).exists():
        raise FileExistsError(
            "formal evaluation refuses to overwrite an existing result: "
            f"{Path(args.output).resolve()}"
        )
    if runtime_git.get("commit") is None:
        raise ValueError(
            "--require_full_train_no_eval requires an evaluator Git commit"
        )
    if checkpoint.get("git_commit") != runtime_git.get("commit"):
        raise ValueError(
            "--require_full_train_no_eval requires checkpoint.git_commit to "
            "match the evaluator runtime Git commit"
        )
    if runtime_git.get("dirty") is not False:
        raise ValueError(
            "--require_full_train_no_eval requires a clean evaluator Git worktree"
        )


def make_evaluation_config(args: argparse.Namespace) -> dict:
    """Serialize every evaluator CLI decision after resolving defaults."""

    config = {field: getattr(args, field) for field in EVALUATION_CONFIG_FIELDS}
    config["checkpoint"] = str(Path(args.checkpoint).resolve())
    config["data_path"] = str(Path(args.data_path).resolve())
    config["output"] = (
        str(Path(args.output).resolve()) if args.output is not None else None
    )
    return config


def write_evaluation_result(
    output_path: Path,
    rendered: str,
    *,
    refuse_overwrite: bool,
) -> None:
    """Atomically publish a complete result JSON from the same directory."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and output_path.exists():
        raise FileExistsError(
            f"formal evaluation refuses to overwrite an existing result: {output_path}"
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Repeat the guard immediately before publication.  os.replace keeps
        # the visible destination all-or-nothing if the process dies while
        # publishing the completed temporary file.
        if refuse_overwrite and output_path.exists():
            raise FileExistsError(
                "formal evaluation refuses to overwrite an existing result: "
                f"{output_path}"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resolve_dataset(args: argparse.Namespace, checkpoint: dict) -> str:
    checkpoint_dataset = checkpoint.get("train_config", {}).get("dataset")
    if checkpoint_dataset not in ("gazefollow", "vat"):
        raise ValueError("checkpoint train_config does not contain a valid dataset")
    if args.dataset is not None and args.dataset != checkpoint_dataset:
        raise ValueError(
            f"--dataset={args.dataset} conflicts with checkpoint dataset={checkpoint_dataset}"
        )
    return checkpoint_dataset


def build_model(model_config: dict):
    required = {
        "model",
        "router_stage",
        "route_after_block",
        "keep_ratio",
        "router_hidden_dim",
        "router_temperature",
        "escape_tokens",
        "spatial_prior",
        "fusion",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise ValueError(f"checkpoint model_config is missing: {missing}")
    return get_coverage_router_model(
        model_config["model"],
        router_stage=model_config["router_stage"],
        route_after_block=int(model_config["route_after_block"]),
        keep_ratio=float(model_config["keep_ratio"]),
        router_hidden_dim=int(model_config["router_hidden_dim"]),
        router_temperature=float(model_config["router_temperature"]),
        escape_tokens=int(model_config["escape_tokens"]),
        spatial_prior=model_config["spatial_prior"],
        fusion=model_config["fusion"],
    )


def apply_eval_overrides(model_config: dict, args: argparse.Namespace) -> dict:
    """Return the effective model config without mutating checkpoint metadata."""
    effective = dict(model_config)
    if args.router_stage_override is not None:
        effective["router_stage"] = args.router_stage_override
    if args.keep_ratio_override is not None:
        effective["keep_ratio"] = float(args.keep_ratio_override)
    return effective


def _matches_head_count_subset(count: int, subset: str) -> bool:
    if subset == "all":
        return count > 0
    if subset == "single":
        return count == 1
    if subset == "multi":
        return count >= 2
    if subset == "two":
        return count == 2
    if subset == "three_plus":
        return count >= 3
    raise ValueError(f"unknown GazeFollow head-count subset: {subset}")


def _filter_gazefollow_indices(records, indices, subset: str):
    selected = []
    person_count = 0
    histogram = {}
    for index in indices:
        count = sum(
            int(head.get("inout", 1) == 1) for head in records[index].get("heads", ())
        )
        histogram[count] = histogram.get(count, 0) + 1
        if _matches_head_count_subset(count, subset):
            selected.append(index)
            person_count += count
    return (
        tuple(selected),
        person_count,
        {str(count): frequency for count, frequency in sorted(histogram.items())},
    )


def make_eval_loader(args: argparse.Namespace, transform):
    if args.dataset == "gazefollow":
        if args.gazefollow_eval_split == "train_holdout":
            annotation_split = "train"
            annotation_path = Path(args.data_path) / "train_preprocessed.json"
            records = load_data_gazefollow(annotation_path)
            group_keys = [
                _gazefollow_group_key(record, index)
                for index, record in enumerate(records)
            ]
            holdout = grouped_holdout_split(
                group_keys,
                validation_fraction=args.gazefollow_val_fraction,
                seed=args.gazefollow_split_seed,
            )
            candidate_indices = holdout.validation_indices
            protocol = {
                **holdout.metadata(),
                "evaluation_split": "gazefollow_train_holdout",
            }
        else:
            annotation_split = "test"
            annotation_path = Path(args.data_path) / "test_preprocessed.json"
            records = load_data_gazefollow(annotation_path)
            candidate_indices = tuple(range(len(records)))
            protocol = {
                "strategy": "gazefollow_official_test",
                "evaluation_split": "gazefollow_official_test",
            }
        image_indices, expected_person_count, source_histogram = (
            _filter_gazefollow_indices(
                records,
                candidate_indices,
                args.gazefollow_head_count_subset,
            )
        )
        protocol.update(
            {
                "annotation_file": str(annotation_path.resolve()),
                "annotation_sha256": sha256_file(annotation_path),
                "query_unit": args.gazefollow_eval_unit,
                "head_count_subset": args.gazefollow_head_count_subset,
                "candidate_record_count": len(candidate_indices),
                "selected_image_count": len(image_indices),
                "selected_person_count": expected_person_count,
                "candidate_inframe_head_count_histogram": source_histogram,
            }
        )
        if not image_indices:
            raise ValueError(
                "the requested GazeFollow evaluation subset contains no images"
            )
        if args.gazefollow_eval_unit == "image":
            dataset = GazeFollowImageDataset(
                args.data_path,
                annotation_split,
                transform,
                image_indices=image_indices,
                records=records,
            )
            batch_collate = collate_gazefollow_images
        else:
            dataset = GazeDataset(
                "gazefollow",
                args.data_path,
                annotation_split,
                transform,
                image_indices=image_indices,
                augment=False,
                return_heatmap=False,
                records=records,
            )
            batch_collate = collate_fn
        actual_person_count = (
            dataset.person_count
            if isinstance(dataset, GazeFollowImageDataset)
            else len(dataset)
        )
        if actual_person_count != expected_person_count:
            raise RuntimeError(
                "GazeFollow evaluation subset person-count mismatch: "
                f"expected {expected_person_count}, got {actual_person_count}"
            )
    else:
        dataset = GazeDataset(
            "videoattentiontarget",
            args.data_path,
            "test",
            transform,
            in_frame_only=False,
            sample_rate=args.frame_sample_every,
        )
        batch_collate = collate_fn
        annotation_path = Path(args.data_path) / "test_preprocessed.json"
        protocol = {
            "strategy": "vat_official_test",
            "evaluation_split": "vat_official_test",
            "annotation_file": str(annotation_path.resolve()),
            "annotation_sha256": sha256_file(annotation_path),
            "frame_sample_every": args.frame_sample_every,
            "selected_frame_count": len(dataset.data),
            "selected_person_count": len(dataset),
        }
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_collate,
        num_workers=args.n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.n_workers > 0,
    )
    return dataset, loader, protocol


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    runtime_git = runtime_git_metadata()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("--checkpoint must be a structured coverage-router checkpoint")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or missing checkpoint format_version")

    args.dataset = resolve_dataset(args, checkpoint)
    if args.dataset != "gazefollow" and args.gazefollow_eval_unit != "person":
        raise ValueError("--gazefollow_eval_unit=image is only valid for GazeFollow")
    if args.dataset != "gazefollow" and args.gazefollow_eval_split != "official_test":
        raise ValueError("--gazefollow_eval_split is only valid for GazeFollow")
    args.data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    args.data_path = str(Path(args.data_path).resolve())
    validate_formal_evaluation_request(
        args,
        checkpoint,
        runtime_git=runtime_git,
    )
    evaluation_config = make_evaluation_config(args)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("checkpoint has no model_config")
    model_config = apply_eval_overrides(checkpoint_model_config, args)
    args.keep_ratio = float(model_config["keep_ratio"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model, transform = build_model(model_config)
    # Evaluation reconstructs frozen DINO weights from the official local
    # checkpoint and then overlays every stored task/suffix parameter.
    model.set_trainable_backbone_suffix(False)
    load_model_state(model, checkpoint, initialization=False)
    model.to(device).eval()
    dataset, loader, evaluation_protocol = make_eval_loader(args, transform)
    metrics = evaluate_model(
        model,
        loader,
        args,
        device,
        max_batches=args.max_eval_batches,
    )
    official_strategy = OFFICIAL_EVALUATION_STRATEGIES[args.dataset]
    is_official_evaluation = evaluation_protocol.get("strategy") == official_strategy
    if args.require_full_train_no_eval and not is_official_evaluation:
        raise RuntimeError(
            "formal evaluation did not construct the expected official-test protocol"
        )
    result = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "evaluation_result_format_version": EVALUATION_RESULT_FORMAT_VERSION,
        "formal_official_evaluation": bool(args.require_full_train_no_eval),
        "checkpoint_validation": (
            "full_train_no_eval_fixed_final"
            if args.require_full_train_no_eval
            else "not_requested"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "checkpoint_data_split": checkpoint.get("data_split"),
        "checkpoint_selection": checkpoint.get("selection"),
        "checkpoint_best_metrics": checkpoint.get("best_metrics"),
        "runtime_git_commit": runtime_git["commit"],
        "runtime_git_dirty": runtime_git["dirty"],
        "dataset": args.dataset,
        "data_path": args.data_path,
        "gazefollow_eval_unit": (
            args.gazefollow_eval_unit if args.dataset == "gazefollow" else None
        ),
        "evaluation_protocol": evaluation_protocol,
        "official_annotation_sha256": (
            evaluation_protocol.get("annotation_sha256")
            if is_official_evaluation
            else None
        ),
        "evaluation_config": evaluation_config,
        "dataset_sample_count": metrics["sample_count"],
        "dataset_image_count": metrics["image_count"],
        "dataset_loader_item_count": len(dataset),
        "vat_frame_sample_every": (
            args.frame_sample_every if args.dataset == "vat" else None
        ),
        "checkpoint_model_config": checkpoint_model_config,
        "model_config": model_config,
        "evaluation_overrides": {
            "router_stage": args.router_stage_override,
            "keep_ratio": args.keep_ratio_override,
        },
        "metrics": metrics,
        "note": (
            "support_pilot runs dense DINOv3 and is not an efficiency result"
            if model_config["router_stage"] == "support_pilot"
            else "backbone_sparse applies routing inside the DINOv3 encoder"
        ),
    }
    rendered = json.dumps(result, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        write_evaluation_result(
            output_path,
            rendered,
            refuse_overwrite=args.require_full_train_no_eval,
        )
        print(f"Saved metrics to {output_path}")
    return result


if __name__ == "__main__":
    main()
