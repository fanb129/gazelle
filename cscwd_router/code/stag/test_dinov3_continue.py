"""CPU continuation checks with generated inputs; no real model results."""

import json
from pathlib import Path

import pytest
import torch

from cscwd_router.code.scripts.continue_router_dinov3 import child_command, parse_args
from cscwd_router.code.stag import dinov3_continue as continuation
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.test_dinov3_train_test import model_factory, setup_run


def setup_continuation(tmp_path, monkeypatch, size="b"):
    parent_args, _ = setup_run(tmp_path, monkeypatch, "vat", size)
    parent_args.report_dir = str(tmp_path / "parent-reports" / f"vit{size}")
    parent_args.run_dir = str(tmp_path / "parent-weights" / f"vit{size}")
    parent = original.run(parent_args, model_factory=model_factory)
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(continuation, "source_hashes", lambda: {"synthetic_test": "fixed"})
    monkeypatch.setattr(continuation, "make_datasets", original.make_datasets)
    monkeypatch.setattr(continuation, "make_full_test", original.make_full_test)
    args = parse_args(["--size", size, "--parent-report-root", str(tmp_path / "parent-reports"),
                       "--data-path", str(tmp_path), "--output-root", str(tmp_path / "continued"),
                       "--report-root", str(tmp_path / "continued-reports"),
                       "--provenance", str(tmp_path / "provenance.md"), "--additional-epochs", "2",
                       "--decoder-lr", "0.001", "--device", "cpu", "--workers", "0", "--log-every", "100"])
    args.batch_size, args.eval_batch_size, args.grad_accumulation = 2, 2, 2
    return args, parent


@pytest.mark.parametrize("size", ["b", "l"])
def test_complete_constant_k50_continuation_and_verified_resume(tmp_path, monkeypatch, size):
    args, parent = setup_continuation(tmp_path, monkeypatch, size)
    parent_files = {item["path"]: sha256(item["path"]) for item in parent["artifacts"].values()}
    result = continuation.run(args, model_factory=model_factory)
    history = json.loads((Path(args.report_root) / f"vit{size}" / "history.json").read_text())
    assert result["additional_epochs"] == 2 and result["parent_decoder_epochs"] == 2
    assert [row["epoch"] for row in history] == [2, 3]
    assert [row["train"]["train_keep_ratio"] for row in history] == [0.5, 0.5]
    assert history[0]["learning_rates"] == [0.001, 0.0001]
    old = torch.load(parent["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)
    new = torch.load(result["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)
    assert all(torch.equal(tensor, new["model_state"][name])
               for name, tensor in old["model_state"].items() if name.startswith("router."))
    assert any(not torch.equal(tensor, new["model_state"][name])
               for name, tensor in old["model_state"].items() if not name.startswith("router."))
    assert all(sha256(name) == digest for name, digest in parent_files.items())
    assert result["selection"]["metrics"]["l2"] <= parent["selection"]["metrics"]["l2"]
    args.resume = True
    assert continuation.run(args, model_factory=model_factory) == result
    args.decoder_lr *= 2
    with pytest.raises(ValueError, match="configuration"):
        continuation.run(args, model_factory=model_factory)


def test_parent_best_is_retained_when_added_epochs_do_not_improve(tmp_path, monkeypatch):
    args, parent = setup_continuation(tmp_path, monkeypatch)
    real_evaluate = continuation.evaluate
    def worse(*a, **kwargs):
        metrics = real_evaluate(*a, **kwargs)
        if kwargs["keep_ratio"] == 0.5 and "rows_path" not in kwargs:
            metrics["l2"] = 10.0
        return metrics
    monkeypatch.setattr(continuation, "evaluate", worse)
    result = continuation.run(args, model_factory=model_factory)
    assert result["selection"]["origin"] == "parent"
    original_best = torch.load(parent["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)
    best = torch.load(result["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)
    assert all(torch.equal(value, best["model_state"][key]) for key, value in original_best["model_state"].items())


def test_atomic_continuation_recovers_without_repeating_an_epoch(tmp_path, monkeypatch):
    args, _ = setup_continuation(tmp_path, monkeypatch)
    real_json = original.atomic_json
    def interrupted(path, value):
        if Path(path).name == "history.json":
            raise RuntimeError("synthetic interruption after continuation epoch commit")
        real_json(path, value)
    monkeypatch.setattr(original, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        continuation.run(args, model_factory=model_factory)
    monkeypatch.setattr(original, "atomic_json", real_json)
    args.resume = True
    resumed = continuation.run(args, model_factory=model_factory)
    resumed_state = torch.load(resumed["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)["model_state"]
    args.resume = False
    args.output_root, args.report_root = str(tmp_path / "fresh"), str(tmp_path / "fresh-reports")
    fresh = continuation.run(args, model_factory=model_factory)
    fresh_state = torch.load(fresh["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)["model_state"]
    assert resumed["metrics"] == fresh["metrics"]
    assert all(torch.equal(tensor, fresh_state[key]) for key, tensor in resumed_state.items())


def test_probe_mismatch_stops_before_training(tmp_path, monkeypatch):
    args, _ = setup_continuation(tmp_path, monkeypatch)
    real_evaluate = continuation.evaluate
    def unequal(model, *a, **kwargs):
        metrics = real_evaluate(model, *a, **kwargs)
        metrics["l2"] = 0 if model.router_stage == "support_pilot" else 1
        return metrics
    monkeypatch.setattr(continuation, "evaluate", unequal)
    with pytest.raises(ValueError, match="native dense and routed K100"):
        continuation.run(args, model_factory=model_factory)
    assert not list(Path(args.output_root).rglob("last.pt"))


def test_continuation_cannot_overwrite_parent_outputs(tmp_path, monkeypatch):
    args, _ = setup_continuation(tmp_path, monkeypatch)
    args.output_root = str(tmp_path / "parent-weights")
    args.resume = True
    with pytest.raises(ValueError, match="new output/report directories"):
        continuation.run(args, model_factory=model_factory)


def test_queue_preserves_all_low_lr_flags_and_seed_zero():
    args = parse_args(["--parent-report-root", "/parent", "--data-path", "/data",
                       "--output-root", "/weights", "--report-root", "/reports", "--provenance", "/p.md", "--resume"])
    for size in ("b", "l"):
        child = parse_args(child_command(args, size)[3:])
        assert child.size == size and child.seed == 0 and child.resume
        assert child.additional_epochs == 8 and child.decoder_lr == 1e-5 and child.inout_lr == 1e-4
