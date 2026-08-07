"""Validate a formal coverage-router official-test result artifact.

The validator deliberately reopens the checkpoint and official annotation.  A
result JSON therefore cannot pass merely by copying plausible-looking metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

try:  # Works both as ``python scripts/validate_...py`` and module import.
    from eval_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        EVALUATION_CONFIG_FIELDS,
        EVALUATION_RESULT_FORMAT_VERSION,
        OFFICIAL_EVALUATION_STRATEGIES,
        _torch_load,
        sha256_file,
        validate_full_train_no_eval_checkpoint,
    )
except ModuleNotFoundError:
    from scripts.eval_coverage_router import (
        CHECKPOINT_FORMAT_VERSION,
        EVALUATION_CONFIG_FIELDS,
        EVALUATION_RESULT_FORMAT_VERSION,
        OFFICIAL_EVALUATION_STRATEGIES,
        _torch_load,
        sha256_file,
        validate_full_train_no_eval_checkpoint,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a formal full-train/no-eval coverage-router official-test "
            "result against its checkpoint and annotation files."
        )
    )
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Expected checkpoint path; defaults to result['checkpoint'].",
    )
    parser.add_argument(
        "--official_annotation",
        default=None,
        help=(
            "Expected official annotation path; defaults to the path stored in "
            "evaluation_protocol."
        ),
    )
    parser.add_argument("--expect_dataset", choices=("gazefollow", "vat"), default=None)
    parser.add_argument(
        "--expect_gazefollow_eval_unit",
        choices=("person", "image"),
        default=None,
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--expect_amp", dest="expect_amp", action="store_true")
    precision.add_argument("--expect_fp32", dest="expect_amp", action="store_false")
    parser.set_defaults(expect_amp=None)
    parser.add_argument("--expect_runtime_git_commit", default=None)
    parser.add_argument("--expect_official_annotation_sha256", default=None)
    return parser.parse_args(argv)


def _resolved(path) -> str:
    return str(Path(path).expanduser().resolve())


def _git_oid(value) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", value)
    )


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_metric(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_numbers(value, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            errors.append(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{path}[{index}]", errors)


def _load_annotation(path: Path, errors: list[str]):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"cannot read official annotation {path}: {error}")
        return None
    if not isinstance(value, list):
        errors.append("official annotation root must be a JSON list")
        return None
    return value


def _validate_gazefollow_counts(
    records: list,
    result: dict,
    protocol: dict,
    config: dict,
    metrics: dict,
    errors: list[str],
) -> None:
    counts = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("heads"), list):
            errors.append(f"GazeFollow annotation record {index} has no heads list")
            return
        counts.append(sum(int(head.get("inout", 1) == 1) for head in record["heads"]))
    selected_image_count = sum(count > 0 for count in counts)
    selected_person_count = sum(counts)
    histogram = {str(count): counts.count(count) for count in sorted(set(counts))}
    expected = {
        "candidate_record_count": len(records),
        "selected_image_count": selected_image_count,
        "selected_person_count": selected_person_count,
        "candidate_inframe_head_count_histogram": histogram,
        "head_count_subset": "all",
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            errors.append(
                f"evaluation_protocol.{field} does not match the official annotation"
            )

    query_unit = config.get("gazefollow_eval_unit")
    if query_unit not in ("person", "image"):
        errors.append("evaluation_config.gazefollow_eval_unit is invalid")
        return
    if protocol.get("query_unit") != query_unit:
        errors.append("evaluation protocol/config query units differ")
    if result.get("gazefollow_eval_unit") != query_unit:
        errors.append("top-level/config GazeFollow query units differ")
    if metrics.get("query_unit") != query_unit:
        errors.append("metrics/config query units differ")

    expected_loader_items = (
        selected_person_count if query_unit == "person" else selected_image_count
    )
    if result.get("dataset_loader_item_count") != expected_loader_items:
        errors.append("dataset_loader_item_count does not cover the official test set")
    if metrics.get("sample_count") != selected_person_count:
        errors.append("metrics.sample_count does not cover every official-test person")
    if metrics.get("image_count") != expected_loader_items:
        errors.append("metrics.image_count is inconsistent with the query unit")
    if metrics.get("inframe_count") != selected_person_count:
        errors.append("metrics.inframe_count is inconsistent with GazeFollow")


def _validate_vat_counts(
    sequences: list,
    result: dict,
    protocol: dict,
    config: dict,
    metrics: dict,
    errors: list[str],
) -> None:
    frames = []
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, dict) or not isinstance(
            sequence.get("frames"), list
        ):
            errors.append(f"VAT annotation sequence {index} has no frames list")
            return
        frames.extend(sequence["frames"])
    person_count = 0
    inframe_count = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("heads"), list):
            errors.append(f"VAT annotation frame {index} has no heads list")
            return
        person_count += len(frame["heads"])
        inframe_count += sum(int(head.get("inout", 1) == 1) for head in frame["heads"])

    if protocol.get("frame_sample_every") != 1:
        errors.append("formal VAT evaluation must use every test frame")
    if config.get("frame_sample_every") != 1:
        errors.append("evaluation_config.frame_sample_every must be 1 for VAT")
    if protocol.get("selected_frame_count") != len(frames):
        errors.append("evaluation_protocol.selected_frame_count is incorrect")
    if protocol.get("selected_person_count") != person_count:
        errors.append("evaluation_protocol.selected_person_count is incorrect")
    for field in ("dataset_loader_item_count", "dataset_sample_count"):
        if result.get(field) != person_count:
            errors.append(f"{field} does not cover every VAT official-test person")
    if result.get("dataset_image_count") != person_count:
        errors.append("dataset_image_count is inconsistent with VAT person queries")
    if metrics.get("sample_count") != person_count:
        errors.append("metrics.sample_count is inconsistent with VAT annotation")
    if metrics.get("image_count") != person_count:
        errors.append("metrics.image_count is inconsistent with VAT annotation")
    if metrics.get("inframe_count") != inframe_count:
        errors.append("metrics.inframe_count is inconsistent with VAT annotation")
    if metrics.get("query_unit") != "person":
        errors.append("formal VAT evaluation query unit must be person")


def validate_result(
    result_path: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    official_annotation_path: Optional[Path] = None,
    expect_dataset: Optional[str] = None,
    expect_gazefollow_eval_unit: Optional[str] = None,
    expect_amp: Optional[bool] = None,
    expect_runtime_git_commit: Optional[str] = None,
    expect_official_annotation_sha256: Optional[str] = None,
) -> dict:
    errors: list[str] = []
    result_path = result_path.expanduser().resolve()
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read result JSON {result_path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("result JSON root must be an object")

    if result.get("evaluation_result_format_version") != (
        EVALUATION_RESULT_FORMAT_VERSION
    ):
        errors.append("unsupported or missing evaluation_result_format_version")
    if result.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        errors.append("unsupported or missing checkpoint format_version record")
    if result.get("formal_official_evaluation") is not True:
        errors.append("formal_official_evaluation must be true")
    if result.get("checkpoint_validation") != "full_train_no_eval_fixed_final":
        errors.append("checkpoint_validation is not the strict formal policy")

    dataset = result.get("dataset")
    if dataset not in OFFICIAL_EVALUATION_STRATEGIES:
        errors.append("result dataset must be gazefollow or vat")
    if expect_dataset is not None and dataset != expect_dataset:
        errors.append(
            f"result dataset {dataset!r} does not match expected {expect_dataset!r}"
        )

    config = result.get("evaluation_config")
    if not isinstance(config, dict):
        errors.append("evaluation_config must be a mapping")
        config = {}
    missing_config_fields = sorted(set(EVALUATION_CONFIG_FIELDS) - set(config))
    if missing_config_fields:
        errors.append(f"evaluation_config is missing fields: {missing_config_fields}")
    if config.get("require_full_train_no_eval") is not True:
        errors.append("evaluation_config.require_full_train_no_eval must be true")
    if config.get("dataset") != dataset:
        errors.append("evaluation_config.dataset differs from result dataset")
    if config.get("data_path") != result.get("data_path"):
        errors.append("evaluation_config.data_path differs from result data_path")
    if config.get("max_eval_batches") is not None:
        errors.append("formal result cannot use max_eval_batches")
    if config.get("router_stage_override") is not None:
        errors.append("formal result cannot use router_stage_override")
    if config.get("keep_ratio_override") is not None:
        errors.append("formal result cannot use keep_ratio_override")
    if not _positive_int(config.get("batch_size")):
        errors.append("evaluation_config.batch_size must be positive")
    n_workers = config.get("n_workers")
    if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 0:
        errors.append("evaluation_config.n_workers must be non-negative")
    if not isinstance(config.get("amp"), bool):
        errors.append("evaluation_config.amp must be boolean")
    if expect_amp is not None and config.get("amp") is not expect_amp:
        errors.append(f"evaluation precision does not match expected amp={expect_amp}")
    if config.get("output") is None or _resolved(config["output"]) != str(result_path):
        errors.append("evaluation_config.output does not identify this result file")

    runtime_commit = result.get("runtime_git_commit")
    if not _git_oid(runtime_commit):
        errors.append("runtime_git_commit must be a full Git object id")
    if result.get("runtime_git_dirty") is not False:
        errors.append("formal evaluation runtime Git worktree was not clean")
    if result.get("checkpoint_git_commit") != runtime_commit:
        errors.append(
            "checkpoint_git_commit differs from the evaluator runtime Git commit"
        )
    if (
        expect_runtime_git_commit is not None
        and runtime_commit != expect_runtime_git_commit
    ):
        errors.append("runtime_git_commit does not match the expected revision")

    recorded_checkpoint = result.get("checkpoint")
    if not isinstance(recorded_checkpoint, str):
        errors.append("result checkpoint path is missing")
        recorded_checkpoint = ""
    checkpoint_path = (
        checkpoint_path.expanduser().resolve()
        if checkpoint_path is not None
        else Path(recorded_checkpoint).expanduser().resolve()
    )
    if recorded_checkpoint and _resolved(recorded_checkpoint) != str(checkpoint_path):
        errors.append("result checkpoint path differs from --checkpoint")
    if config.get("checkpoint") is None or _resolved(config["checkpoint"]) != str(
        checkpoint_path
    ):
        errors.append("evaluation_config.checkpoint differs from checkpoint path")
    try:
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    except OSError as error:
        errors.append(f"cannot hash checkpoint {checkpoint_path}: {error}")
        actual_checkpoint_sha256 = None
    if result.get("checkpoint_sha256") != actual_checkpoint_sha256:
        errors.append("checkpoint_sha256 does not match the checkpoint file")

    checkpoint = None
    try:
        checkpoint = _torch_load(str(checkpoint_path))
    except Exception as error:  # torch emits several format-specific exceptions.
        errors.append(f"cannot load checkpoint {checkpoint_path}: {error}")
    if isinstance(checkpoint, dict) and dataset in OFFICIAL_EVALUATION_STRATEGIES:
        try:
            validate_full_train_no_eval_checkpoint(checkpoint, dataset=dataset)
        except ValueError as error:
            errors.append(str(error))
        checkpoint_fields = {
            "checkpoint_role": "checkpoint_role",
            "checkpoint_epoch": "epoch",
            "checkpoint_git_commit": "git_commit",
            "checkpoint_data_split": "data_split",
            "checkpoint_selection": "selection",
            "checkpoint_best_metrics": "best_metrics",
            "checkpoint_model_config": "model_config",
        }
        for result_field, checkpoint_field in checkpoint_fields.items():
            if result.get(result_field) != checkpoint.get(checkpoint_field):
                errors.append(
                    f"{result_field} differs from checkpoint.{checkpoint_field}"
                )
    elif checkpoint is not None:
        errors.append("checkpoint root must be a mapping")

    if result.get("model_config") != result.get("checkpoint_model_config"):
        errors.append("formal result model_config differs from checkpoint_model_config")
    if result.get("evaluation_overrides") != {
        "router_stage": None,
        "keep_ratio": None,
    }:
        errors.append("formal result contains evaluation overrides")

    protocol = result.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        errors.append("evaluation_protocol must be a mapping")
        protocol = {}
    if dataset in OFFICIAL_EVALUATION_STRATEGIES:
        expected_strategy = OFFICIAL_EVALUATION_STRATEGIES[dataset]
        if protocol.get("strategy") != expected_strategy:
            errors.append("evaluation_protocol.strategy is not the official test")
        if protocol.get("evaluation_split") != expected_strategy:
            errors.append("evaluation_protocol.evaluation_split is not official")

    recorded_annotation = protocol.get("annotation_file")
    if not isinstance(recorded_annotation, str):
        errors.append("evaluation_protocol.annotation_file is missing")
        recorded_annotation = ""
    annotation_path = (
        official_annotation_path.expanduser().resolve()
        if official_annotation_path is not None
        else Path(recorded_annotation).expanduser().resolve()
    )
    if recorded_annotation and _resolved(recorded_annotation) != str(annotation_path):
        errors.append(
            "evaluation protocol annotation path differs from --official_annotation"
        )
    data_path = config.get("data_path")
    if isinstance(data_path, str):
        expected_annotation_path = (
            Path(data_path).expanduser().resolve() / "test_preprocessed.json"
        )
        if annotation_path != expected_annotation_path:
            errors.append("official annotation is not data_path/test_preprocessed.json")
    try:
        annotation_sha256 = sha256_file(annotation_path)
    except OSError as error:
        errors.append(f"cannot hash official annotation {annotation_path}: {error}")
        annotation_sha256 = None
    if protocol.get("annotation_sha256") != annotation_sha256:
        errors.append("evaluation protocol annotation SHA-256 does not match file")
    if result.get("official_annotation_sha256") != annotation_sha256:
        errors.append("official_annotation_sha256 does not match file")
    if (
        expect_official_annotation_sha256 is not None
        and annotation_sha256 != expect_official_annotation_sha256.lower()
    ):
        errors.append("official annotation SHA-256 does not match the expected digest")

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be a mapping")
        metrics = {}
    if metrics.get("dataset") != dataset:
        errors.append("metrics.dataset differs from result dataset")
    for result_field, metric_field in (
        ("dataset_sample_count", "sample_count"),
        ("dataset_image_count", "image_count"),
    ):
        if result.get(result_field) != metrics.get(metric_field):
            errors.append(f"{result_field} differs from metrics.{metric_field}")
    _finite_numbers(metrics, "metrics", errors)

    annotations = _load_annotation(annotation_path, errors)
    if annotations is not None and dataset == "gazefollow":
        if config.get("gazefollow_eval_split") != "official_test":
            errors.append("formal GazeFollow result must use official_test")
        if config.get("gazefollow_head_count_subset") != "all":
            errors.append("formal GazeFollow result must use head-count subset all")
        _validate_gazefollow_counts(
            annotations, result, protocol, config, metrics, errors
        )
        if (
            expect_gazefollow_eval_unit is not None
            and config.get("gazefollow_eval_unit") != expect_gazefollow_eval_unit
        ):
            errors.append("GazeFollow eval unit does not match the expected unit")
        for metric in (
            "auc",
            "avg_l2",
            "min_l2",
            "routing_soft_coverage",
            "routing_hard_coverage",
            "routing_gt_point_coverage",
            "routing_mean_support",
            "routing_actual_keep_ratio",
        ):
            if not _finite_metric(metrics.get(metric)):
                errors.append(f"metrics.{metric} is missing or non-finite")
    elif annotations is not None and dataset == "vat":
        _validate_vat_counts(annotations, result, protocol, config, metrics, errors)
        for metric in (
            "auc",
            "l2",
            "inout_ap",
            "routing_soft_coverage",
            "routing_hard_coverage",
            "routing_gt_point_coverage",
            "routing_mean_support",
            "routing_actual_keep_ratio",
        ):
            if not _finite_metric(metrics.get(metric)):
                errors.append(f"metrics.{metric} is missing or non-finite")

    if errors:
        rendered = "\n- ".join(errors)
        raise ValueError(f"formal evaluation validation failed:\n- {rendered}")

    return {
        "pass": True,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "dataset": dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "official_annotation": str(annotation_path),
        "official_annotation_sha256": annotation_sha256,
        "runtime_git_commit": runtime_commit,
        "evaluation_config": config,
        "metrics": metrics,
    }


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    report = validate_result(
        Path(args.result),
        checkpoint_path=(Path(args.checkpoint) if args.checkpoint else None),
        official_annotation_path=(
            Path(args.official_annotation) if args.official_annotation else None
        ),
        expect_dataset=args.expect_dataset,
        expect_gazefollow_eval_unit=args.expect_gazefollow_eval_unit,
        expect_amp=args.expect_amp,
        expect_runtime_git_commit=args.expect_runtime_git_commit,
        expect_official_annotation_sha256=(args.expect_official_annotation_sha256),
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
