from argparse import Namespace
import copy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from scripts.eval_coverage_router import (
    EVALUATION_RESULT_FORMAT_VERSION,
    sha256_file,
    validate_formal_evaluation_request,
    validate_full_train_no_eval_checkpoint,
    write_evaluation_result,
)
from scripts.validate_coverage_router_eval import validate_result


def _formal_checkpoint(**overrides):
    split = {
        "strategy": "gazefollow_full_train_no_eval_v1",
        "evaluation_split": None,
        "evaluation_mode": "none",
        "selection_policy": "fixed_final_epoch_no_validation",
        "selection_is_formal": False,
        "provenance_requires_match": True,
        "official_test_accessed": False,
        "validation_fraction": 0.0,
        "group_key": "normalized_image_path",
        "source_annotation_file": "/data/train_preprocessed.json",
        "source_annotation_sha256": "1" * 64,
        "train_record_count": 3,
        "train_head_sample_count": 3,
        "validation_record_count": 0,
        "validation_head_sample_count": 0,
        "source_group_fingerprint": "2" * 64,
        "train_group_fingerprint": "3" * 64,
        "assignment_fingerprint": "4" * 64,
        "train_group_count": 3,
    }
    checkpoint = {
        "format_version": 1,
        "checkpoint_role": "fixed_epoch_final",
        "epoch": 2,
        "model_config": {"router_stage": "backbone_sparse", "keep_ratio": 0.5},
        "train_config": {
            "dataset": "gazefollow",
            "formal_full_train_no_eval": True,
            "max_epochs": 3,
            "gazefollow_val_fraction": 0.0,
            "vat_val_fraction": 0.0,
            "max_train_batches": None,
            "max_eval_batches": None,
            "eval_batch_size": None,
            "save_every": 0,
        },
        "state_scope": "all_non_backbone_and_all_trainable_backbone_parameters",
        "model_state": {"decoder.weight": torch.ones(1)},
        "metrics": {"epoch": 2, "train": {"total": 0.1}},
        "best_metrics": None,
        "data_split": split,
        "selection": None,
        "git_commit": "a" * 40,
    }
    checkpoint.update(overrides)
    return checkpoint


def _formal_args(tmp_path, **overrides):
    values = {
        "require_full_train_no_eval": True,
        "dataset": "gazefollow",
        "router_stage_override": None,
        "keep_ratio_override": None,
        "max_eval_batches": None,
        "gazefollow_eval_split": "official_test",
        "gazefollow_head_count_subset": "all",
        "frame_sample_every": 1,
        "output": str(tmp_path / "formal.json"),
    }
    values.update(overrides)
    return Namespace(**values)


def test_strict_checkpoint_accepts_only_fixed_final_no_eval_artifact():
    validate_full_train_no_eval_checkpoint(_formal_checkpoint(), dataset="gazefollow")

    invalid = _formal_checkpoint(checkpoint_role="training_checkpoint")
    with pytest.raises(ValueError, match="fixed_epoch_final"):
        validate_full_train_no_eval_checkpoint(invalid, dataset="gazefollow")

    invalid = _formal_checkpoint(selection={"epoch": 1})
    with pytest.raises(ValueError, match="selection must be null"):
        validate_full_train_no_eval_checkpoint(invalid, dataset="gazefollow")

    invalid = _formal_checkpoint(epoch=1)
    with pytest.raises(ValueError, match="fixed final epoch"):
        validate_full_train_no_eval_checkpoint(invalid, dataset="gazefollow")


def test_formal_request_accepts_clean_matching_fixed_checkpoint(tmp_path):
    validate_formal_evaluation_request(
        _formal_args(tmp_path),
        _formal_checkpoint(),
        runtime_git={"commit": "a" * 40, "dirty": False},
    )


@pytest.mark.parametrize(
    ("args_override", "runtime_git", "prepare_output", "message"),
    [
        (
            {"router_stage_override": "support_pilot"},
            {"commit": "a" * 40, "dirty": False},
            False,
            "overrides",
        ),
        (
            {"max_eval_batches": 1},
            {"commit": "a" * 40, "dirty": False},
            False,
            "complete official",
        ),
        ({}, {"commit": "b" * 40, "dirty": False}, False, "match the evaluator"),
        ({}, {"commit": "a" * 40, "dirty": True}, False, "clean evaluator"),
        ({}, {"commit": "a" * 40, "dirty": False}, True, "overwrite"),
    ],
)
def test_formal_request_rejects_diagnostics_partial_dirty_or_overwrite(
    tmp_path,
    args_override,
    runtime_git,
    prepare_output,
    message,
):
    args = _formal_args(tmp_path, **args_override)
    if prepare_output:
        (tmp_path / "formal.json").write_text("existing", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError), match=message):
        validate_formal_evaluation_request(
            args,
            _formal_checkpoint(),
            runtime_git=runtime_git,
        )


def test_atomic_formal_writer_refuses_to_replace_existing_result(tmp_path):
    output = tmp_path / "result.json"
    write_evaluation_result(output, '{"pass": true}', refuse_overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"pass": True}

    with pytest.raises(FileExistsError, match="overwrite"):
        write_evaluation_result(output, '{"pass": false}', refuse_overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"pass": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_validator_reopens_checkpoint_annotation_and_accepts_consistent_artifact(
    tmp_path,
):
    annotation_path = tmp_path / "test_preprocessed.json"
    records = [
        {"path": "a.jpg", "heads": [{"inout": 1}]},
        {"path": "b.jpg", "heads": [{"inout": 1}, {"inout": 1}]},
        {"path": "c.jpg", "heads": [{"inout": 0}]},
    ]
    annotation_path.write_text(json.dumps(records), encoding="utf-8")
    annotation_sha256 = hashlib.sha256(annotation_path.read_bytes()).hexdigest()

    checkpoint = _formal_checkpoint()
    checkpoint_path = tmp_path / "final.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    result_path = tmp_path / "official.json"
    config = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": "gazefollow",
        "data_path": str(tmp_path.resolve()),
        "router_stage_override": None,
        "keep_ratio_override": None,
        "batch_size": 1,
        "n_workers": 0,
        "gazefollow_eval_unit": "person",
        "gazefollow_eval_split": "official_test",
        "gazefollow_val_fraction": 0.1,
        "gazefollow_split_seed": 3106,
        "gazefollow_head_count_subset": "all",
        "frame_sample_every": 1,
        "max_eval_batches": None,
        "device": "cuda:1",
        "amp": False,
        "output": str(result_path.resolve()),
        "require_full_train_no_eval": True,
    }
    metrics = {
        "dataset": "gazefollow",
        "query_unit": "person",
        "image_count": 3,
        "sample_count": 3,
        "inframe_count": 3,
        "auc": 0.9,
        "avg_l2": 0.1,
        "min_l2": 0.08,
        "routing_soft_coverage": 0.95,
        "routing_hard_coverage": 0.9,
        "routing_gt_point_coverage": 0.9,
        "routing_mean_support": 0.25,
        "routing_actual_keep_ratio": 0.5,
    }
    result = {
        "format_version": 1,
        "evaluation_result_format_version": EVALUATION_RESULT_FORMAT_VERSION,
        "formal_official_evaluation": True,
        "checkpoint_validation": "full_train_no_eval_fixed_final",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": checkpoint["checkpoint_role"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_git_commit": checkpoint["git_commit"],
        "checkpoint_data_split": copy.deepcopy(checkpoint["data_split"]),
        "checkpoint_selection": None,
        "checkpoint_best_metrics": None,
        "runtime_git_commit": "a" * 40,
        "runtime_git_dirty": False,
        "dataset": "gazefollow",
        "data_path": str(tmp_path.resolve()),
        "gazefollow_eval_unit": "person",
        "evaluation_protocol": {
            "strategy": "gazefollow_official_test",
            "evaluation_split": "gazefollow_official_test",
            "annotation_file": str(annotation_path.resolve()),
            "annotation_sha256": annotation_sha256,
            "query_unit": "person",
            "head_count_subset": "all",
            "candidate_record_count": 3,
            "selected_image_count": 2,
            "selected_person_count": 3,
            "candidate_inframe_head_count_histogram": {"0": 1, "1": 1, "2": 1},
        },
        "official_annotation_sha256": annotation_sha256,
        "evaluation_config": config,
        "dataset_sample_count": 3,
        "dataset_image_count": 3,
        "dataset_loader_item_count": 3,
        "checkpoint_model_config": copy.deepcopy(checkpoint["model_config"]),
        "model_config": copy.deepcopy(checkpoint["model_config"]),
        "evaluation_overrides": {"router_stage": None, "keep_ratio": None},
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    report = validate_result(
        result_path,
        checkpoint_path=checkpoint_path,
        official_annotation_path=annotation_path,
        expect_dataset="gazefollow",
        expect_gazefollow_eval_unit="person",
        expect_amp=False,
        expect_runtime_git_commit="a" * 40,
        expect_official_annotation_sha256=annotation_sha256,
    )

    assert report["pass"] is True
    assert report["checkpoint_sha256"] == checkpoint_sha256
    assert report["official_annotation_sha256"] == annotation_sha256
