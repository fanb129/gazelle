import copy
import hashlib
import json

import pytest

from scripts.validate_coverage_router_refit import (
    DEFAULT_SOURCE_ANNOTATION_SHA256,
    FULL_TRAIN_SELECTION_POLICY,
    FULL_TRAIN_STRATEGY,
    parse_args,
    validate_refit_run,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_history(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_refit_artifacts(
    tmp_path,
    *,
    with_init=True,
    router_stage="backbone_sparse",
    keep_ratio=0.5,
    epochs=3,
    warmup_epochs=3,
):
    run_dir = tmp_path / "refit"
    run_dir.mkdir()
    git_commit = "a" * 40
    init_path = tmp_path / "init.pt"
    init_path.write_bytes(b"immutable initialization checkpoint")
    init_record = None
    if with_init:
        init_record = {
            "path": str(init_path.resolve()),
            "sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        }

    split = {
        "strategy": FULL_TRAIN_STRATEGY,
        "evaluation_split": None,
        "evaluation_mode": "none",
        "selection_policy": FULL_TRAIN_SELECTION_POLICY,
        "selection_is_formal": False,
        "provenance_requires_match": True,
        "official_test_accessed": False,
        "validation_fraction": 0.0,
        "group_key": "normalized_image_path",
        "source_annotation_file": "/data/train_preprocessed.json",
        "source_annotation_sha256": DEFAULT_SOURCE_ANNOTATION_SHA256,
        "source_group_fingerprint": "1" * 64,
        "train_group_fingerprint": "2" * 64,
        "assignment_fingerprint": "3" * 64,
        "train_record_count": 117727,
        "train_head_sample_count": 113458,
        "train_group_count": 117727,
        "validation_record_count": 0,
        "validation_head_sample_count": 0,
    }
    train_config = {
        "dataset": "gazefollow",
        "seed": 3106,
        "router_stage": router_stage,
        "keep_ratio": keep_ratio,
        "max_epochs": epochs,
        "batch_size": 16,
        "amp": False,
        "formal_full_train_no_eval": True,
        "gazefollow_val_fraction": 0.0,
        "vat_val_fraction": 0.0,
        "eval_batch_size": None,
        "max_eval_batches": None,
        "max_train_batches": None,
        "save_every": 0,
        "resume": None,
        "init_ckpt": str(init_path.resolve()) if with_init else None,
        "allow_init_without_split_provenance": False,
        "reinitialize_router_on_init": False,
        "router_warmup_epochs": warmup_epochs,
        "grad_accum_steps": 2,
        "lr_router": 0.0,
        "lr_decoder": 1e-4,
        "lr_backbone": 0.0,
        "lr_inout": 0.0,
        "weight_decay": 0.0,
        "run_dir": str(run_dir.resolve()),
    }
    model_config = {
        "model": "gazelle_dinov3_vitb16",
        "router_stage": router_stage,
        "route_after_block": 5,
        "keep_ratio": keep_ratio,
        "router_hidden_dim": 256,
        "router_temperature": 1.0,
        "escape_tokens": 8,
        "spatial_prior": "none",
        "fusion": "raw_concat",
    }
    manifest = {
        "format_version": 1,
        "dataset": "gazefollow",
        "train_sample_count": 113458,
        "eval_sample_count": None,
        "eval_image_count": None,
        "data_split": copy.deepcopy(split),
        "model_config": copy.deepcopy(model_config),
        "train_config": copy.deepcopy(train_config),
        "initialization_checkpoint": copy.deepcopy(init_record),
        "trainable_parameters": {
            "router": 0,
            "decoder": 3416576,
            "inout": 0,
            "backbone": 0,
        },
        "git_commit": git_commit,
    }

    def expected_keep(epoch):
        if warmup_epochs <= 1:
            return keep_ratio
        progress = min(1.0, epoch / (warmup_epochs - 1))
        return 1.0 + progress * (keep_ratio - 1.0)

    history = [
        {
            "epoch": epoch,
            "active_train_keep_ratio": expected_keep(epoch),
            "train": {
                "total": 1.0 / (epoch + 1),
                "heatmap": 0.1,
                "router_actual_keep_ratio": expected_keep(epoch),
            },
        }
        for epoch in range(epochs)
    ]
    summary = {
        "evaluation_split": None,
        "checkpoint_policy": FULL_TRAIN_SELECTION_POLICY,
        "selection": None,
        "best_metrics": None,
        "evaluation_performed": False,
        "last_epoch": epochs - 1,
        "final_checkpoint": "final.pt",
        "completed": True,
    }
    common_payload = {
        "format_version": 1,
        "epoch": epochs - 1,
        "state_scope": "all_non_backbone_and_all_trainable_backbone_parameters",
        "model_state": {"decoder.weight": "synthetic-tensor"},
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "rng_state": {},
        "model_config": copy.deepcopy(model_config),
        "train_config": copy.deepcopy(train_config),
        "metrics": copy.deepcopy(history[-1]),
        "best_metrics": None,
        "data_split": copy.deepcopy(split),
        "selection": None,
        "initialization_checkpoint": copy.deepcopy(init_record),
        "git_commit": git_commit,
    }
    payloads = {
        "final.pt": {
            **copy.deepcopy(common_payload),
            "checkpoint_role": "fixed_epoch_final",
        },
        "last.resume.pt": {
            **copy.deepcopy(common_payload),
            "checkpoint_role": "resume_checkpoint",
        },
    }

    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "data_split.json", split)
    _write_history(run_dir / "history.jsonl", history)
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "final.pt").write_bytes(b"final checkpoint bytes")
    (run_dir / "last.resume.pt").write_bytes(b"resume checkpoint bytes")

    argv = [
        "--run_dir",
        str(run_dir),
        "--seed",
        "3106",
        "--router_stage",
        router_stage,
        "--keep_ratio",
        str(keep_ratio),
        "--epochs",
        str(epochs),
        "--batch_size",
        "16",
        "--router_trainable",
        "0",
        "--decoder_trainable",
        "3416576",
        "--inout_trainable",
        "0",
        "--git_commit",
        git_commit,
        "--lr_router",
        "0",
        "--lr_decoder",
        "1e-4",
        "--router_warmup_epochs",
        str(warmup_epochs),
        "--grad_accum_steps",
        "2",
        "--init_checkpoint",
        str(init_path) if with_init else "none",
    ]
    return {
        "run_dir": run_dir,
        "args": parse_args(argv),
        "manifest": manifest,
        "split": split,
        "history": history,
        "summary": summary,
        "payloads": payloads,
    }


def _validate(artifacts):
    return validate_refit_run(
        artifacts["args"],
        checkpoint_loader=lambda path: copy.deepcopy(artifacts["payloads"][path.name]),
    )


def test_valid_full_train_refit_passes_without_eval_or_best_artifacts(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)

    result = _validate(artifacts)

    assert result["epochs"] == 3
    assert result["final_epoch"] == 2
    assert len(result["final_checkpoint_sha256"]) == 64


def test_dense_refit_accepts_no_initialization_checkpoint(tmp_path):
    artifacts = _build_refit_artifacts(
        tmp_path,
        with_init=False,
        router_stage="support_pilot",
        keep_ratio=1.0,
        epochs=2,
        warmup_epochs=0,
    )

    _validate(artifacts)


def test_refit_accepts_safe_same_directory_resume_and_preserves_init_record(
    tmp_path,
):
    artifacts = _build_refit_artifacts(tmp_path)
    resume_path = artifacts["run_dir"] / "last.resume.pt"
    for payload in artifacts["payloads"].values():
        payload["train_config"]["init_ckpt"] = None
        payload["train_config"]["resume"] = str(resume_path.resolve())

    _validate(artifacts)


def test_refit_rejects_any_eval_history(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["history"][0]["eval"] = {"auc": 0.99}
    _write_history(
        artifacts["run_dir"] / "history.jsonl",
        artifacts["history"],
    )

    with pytest.raises(RuntimeError, match="history epoch 0 keys"):
        _validate(artifacts)


def test_refit_rejects_best_or_eval_artifacts(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    (artifacts["run_dir"] / "best_val_selection.pt").write_bytes(b"forbidden")

    with pytest.raises(RuntimeError, match="forbidden eval/best artifacts"):
        _validate(artifacts)


def test_refit_rejects_initialization_hash_mismatch(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["manifest"]["initialization_checkpoint"]["sha256"] = "0" * 64
    _write_json(
        artifacts["run_dir"] / "run_manifest.json",
        artifacts["manifest"],
    )

    with pytest.raises(RuntimeError, match="initialization checkpoint sha256"):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_role", "best_validation", "checkpoint_role"),
        ("epoch", 1, "final epoch"),
        ("git_commit", "b" * 40, "final git_commit"),
    ],
)
def test_refit_rejects_wrong_final_checkpoint_identity(
    tmp_path,
    field,
    value,
    message,
):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["payloads"]["final.pt"][field] = value

    with pytest.raises(RuntimeError, match=message):
        _validate(artifacts)


def test_refit_rejects_final_checkpoint_without_model_state(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["payloads"]["final.pt"].pop("model_state")

    with pytest.raises(RuntimeError, match="model_state"):
        _validate(artifacts)


def test_refit_rejects_validation_selection_in_summary(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["summary"]["selection"] = {"epoch": 2}
    _write_json(
        artifacts["run_dir"] / "summary.json",
        artifacts["summary"],
    )

    with pytest.raises(RuntimeError, match="summary selection"):
        _validate(artifacts)


def test_refit_rejects_wrong_sparse_curriculum(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["history"][1]["active_train_keep_ratio"] = 0.5
    _write_history(
        artifacts["run_dir"] / "history.jsonl",
        artifacts["history"],
    )

    with pytest.raises(RuntimeError, match="active_train_keep_ratio"):
        _validate(artifacts)


def test_refit_rejects_manifest_and_data_split_disagreement(tmp_path):
    artifacts = _build_refit_artifacts(tmp_path)
    artifacts["split"]["assignment_fingerprint"] = "4" * 64
    _write_json(
        artifacts["run_dir"] / "data_split.json",
        artifacts["split"],
    )

    with pytest.raises(RuntimeError, match="manifest data_split"):
        _validate(artifacts)
