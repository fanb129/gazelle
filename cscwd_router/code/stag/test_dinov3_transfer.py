"""CPU engineering checks for GF-to-VAT transfer; generated data only."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gazelle.routing import model as native
from cscwd_router.code.data_prep.train_test import make_loader
from cscwd_router.code.scripts.transfer_gf_router_to_vat import parse_args
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag import dinov3_transfer as transfer
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.test_dinov3_train_test import TinyData, TinyModel, model_factory, setup_run


@pytest.mark.parametrize("size", ["b", "l"])
def test_native_transfer_inherits_all_shared_tensors_and_only_adds_five_inout_tensors(monkeypatch, size):
    class ShapeBackbone(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.channels = 768 if "vitb" in name else 1024
        def _validate_route(self, route):
            pass
        def get_dimension(self):
            return self.channels
        def get_out_size(self, shape):
            return shape[0] // 16, shape[1] // 16
        def get_transform(self, shape):
            return None
        def set_trainable_suffix(self, *args, **kwargs):
            pass
    monkeypatch.setattr(native, "RoutedDinoV3Backbone", ShapeBackbone)
    gf, _ = native.get_coverage_router_model(f"gazelle_dinov3_vit{size}16", keep_ratio=.5)
    vat, _ = native.get_coverage_router_model(f"gazelle_dinov3_vit{size}16_inout", keep_ratio=.5)
    before = original.task_state(vat)
    source = original.task_state(gf)
    report = transfer.transfer_state(vat, source)
    after = original.task_state(vat)
    assert report["shared_tensors_identical"] and len(report["new_inout_keys"]) == 5
    assert all(torch.equal(value, after[name]) for name, value in source.items())
    assert all(torch.equal(before[name], after[name]) for name in report["new_inout_keys"])


def test_transfer_rejects_missing_router_and_old_vat_state():
    gf = TinyModel("gazelle_dinov3_vitb16")
    vat = TinyModel("gazelle_dinov3_vitb16_inout")
    state = original.task_state(gf)
    state.pop("router.score_head.2.weight")
    with pytest.raises(ValueError, match="transfer keys differ"):
        transfer.transfer_state(vat, state)
    with pytest.raises(ValueError, match="transfer keys differ"):
        transfer.transfer_state(vat, original.task_state(vat))


def training_args(**updates):
    return SimpleNamespace(**{"router_lr": 1e-3, "decoder_lr": 1e-3, "inout_lr": 1e-3,
        "router_loss_weight": 1.0, "budget_weight": .05, "grad_accumulation": 2,
        "clip_norm": 1.0, "log_every": 100, **updates})


@pytest.mark.parametrize("router_active", [False, True])
def test_training_phases_change_the_intended_modules_only(router_active):
    torch.manual_seed(2)
    model = TinyModel("gazelle_dinov3_vitb16_inout")
    before = {name: value.clone() for name, value in model.state_dict().items()}
    args = training_args()
    trained = transfer.train_epoch(model, make_loader(TinyData("vat", training=True), 1, 0),
        transfer.optimizer_for(model, args), "cpu", args, router_active=router_active)
    after = model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert not any(name.startswith("backbone.") for name in changed)
    assert any(name.startswith("linear.") for name in changed)
    assert any(name.startswith("inout_head.") for name in changed)
    assert any(name.startswith("router.") for name in changed) == router_active
    assert trained["trainable_parameters"]["backbone"] == 0 and trained["optimizer_steps"] == 2
    assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())


def test_router_learning_comes_from_coverage_not_a_false_hard_topk_gradient():
    model = TinyModel("gazelle_dinov3_vitb16_inout")
    before = {name: value.clone() for name, value in model.router.state_dict().items()}
    args = training_args(router_loss_weight=0)
    transfer.train_epoch(model, make_loader(TinyData("vat", training=True), 2, 0),
        transfer.optimizer_for(model, args), "cpu", args, router_active=True)
    assert all(torch.equal(value, model.router.state_dict()[name]) for name, value in before.items())


def test_all_out_of_frame_batches_have_finite_joint_loss():
    model = TinyModel("gazelle_dinov3_vitb16_inout")
    args = training_args()
    result = transfer.train_epoch(model, make_loader(TinyData("vat", training=True, out_only=True), 1, 0),
        transfer.optimizer_for(model, args), "cpu", args, router_active=True)
    assert result["persons"] == 3 and result["optimizer_steps"] == 2
    assert torch.isfinite(torch.tensor(result["loss"]))


def setup_transfer(tmp_path, monkeypatch, size="b"):
    parent_args, _ = setup_run(tmp_path, monkeypatch, "gazefollow", size)
    parent_args.run_dir = str(tmp_path / "gf-weights" / f"vit{size}")
    parent_args.report_dir = str(tmp_path / "gf-reports" / f"vit{size}")
    gf = original.run(parent_args, model_factory=model_factory)
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    monkeypatch.setattr(transfer, "source_hashes", lambda: {"synthetic_test": "fixed"})
    monkeypatch.setattr(transfer, "make_datasets", lambda *a, **kw: (
        TinyData("vat", training=True), TinyData("vat"), {"validation_fraction": 0, "test_frame_sample_every": 6}))
    monkeypatch.setattr(transfer, "make_full_test", lambda *a: TinyData("vat"))
    args = parse_args(["--gf-summary", str(Path(parent_args.report_dir) / "summary.json"),
        "--data-path", str(tmp_path), "--run-dir", str(tmp_path / "vat-weights"),
        "--report-dir", str(tmp_path / "vat-reports"), "--provenance", str(tmp_path / "provenance.md"),
        "--epochs", "3", "--workers", "0", "--batch-size", "2", "--eval-batch-size", "2",
        "--log-every", "100", "--device", "cpu"])
    return args, gf


@pytest.mark.parametrize("size", ["b", "l"])
def test_complete_transfer_and_verified_resume(tmp_path, monkeypatch, size):
    args, gf = setup_transfer(tmp_path, monkeypatch, size)
    source_hash = sha256(gf["artifacts"]["best_test_checkpoint"]["path"])
    result = transfer.run(args, model_factory=model_factory)
    assert result["epochs"] == 3 and result["model_config"]["keep_ratio"] == .5
    assert result["initialization"]["shared_tensors_identical"]
    initial = torch.load(result["artifacts"]["initial_checkpoint"]["path"], weights_only=False)
    first = torch.load(result["artifacts"]["epoch_0"]["path"], weights_only=False)
    last = torch.load(result["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)
    router_names = [name for name in initial["model_state"] if name.startswith("router.")]
    assert all(torch.equal(initial["model_state"][name], first["model_state"][name]) for name in router_names)
    assert any(not torch.equal(first["model_state"][name], last["model_state"][name]) for name in router_names)
    assert all(row["train"]["train_keep_ratio"] == .5 for row in last["history"])
    assert [row["train"]["router_active"] for row in last["history"]] == [False, True, True]
    expected = min(last["history"], key=lambda row: (row["test"]["l2"], -row["test"]["auc"], row["epoch"]))
    assert result["selection"]["epoch"] == expected["epoch"]
    assert source_hash == sha256(gf["artifacts"]["best_test_checkpoint"]["path"])
    args.resume = True
    assert transfer.run(args, model_factory=model_factory) == result
    args.router_lr *= 2
    with pytest.raises(ValueError, match="configuration"):
        transfer.run(args, model_factory=model_factory)


def test_atomic_epoch_resume_matches_uninterrupted_transfer(tmp_path, monkeypatch):
    args, _ = setup_transfer(tmp_path, monkeypatch)
    real_json = original.atomic_json
    def interrupted(path, value):
        if Path(path).name == "history.json":
            raise RuntimeError("synthetic transfer interruption after epoch commit")
        real_json(path, value)
    monkeypatch.setattr(original, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="synthetic transfer interruption"):
        transfer.run(args, model_factory=model_factory)
    monkeypatch.setattr(original, "atomic_json", real_json)
    args.resume = True
    resumed = transfer.run(args, model_factory=model_factory)
    resumed_state = torch.load(resumed["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)["model_state"]
    args.resume = False
    args.run_dir, args.report_dir = str(tmp_path / "fresh-weights"), str(tmp_path / "fresh-reports")
    fresh = transfer.run(args, model_factory=model_factory)
    fresh_state = torch.load(fresh["artifacts"]["decoder_resume_checkpoint"]["path"], weights_only=False)["model_state"]
    assert resumed["metrics"] == fresh["metrics"]
    assert all(torch.equal(value, fresh_state[name]) for name, value in resumed_state.items())


def test_changed_gf_checkpoint_is_rejected(tmp_path, monkeypatch):
    args, gf = setup_transfer(tmp_path, monkeypatch)
    with Path(gf["artifacts"]["best_test_checkpoint"]["path"]).open("ab") as handle:
        handle.write(b"synthetic corruption")
    with pytest.raises(ValueError, match="source checkpoint"):
        transfer.run(args, model_factory=model_factory)


def test_default_recipe_uses_seed0_low_lr_and_fixed_router_warmup():
    args = parse_args(["--gf-summary", "/gf/summary.json", "--data-path", "/data", "--run-dir", "/weights",
                       "--report-dir", "/reports", "--provenance", "/p.md"])
    assert args.seed == 0 and args.epochs == 8 and args.router_warmup_epochs == 1
    assert (args.router_lr, args.decoder_lr, args.inout_lr) == (1e-4, 1e-5, 1e-3)


def test_user_closed_gf_l_transfers_without_claiming_complete_source(tmp_path, monkeypatch):
    import json
    args, gf = setup_transfer(tmp_path, monkeypatch, "l")
    gf.update(completed=False, closed_by_user=True, checkpoint_export_complete=True,
              format="cscwd_router_dinov3_interrupted_v1")
    path = Path(args.gf_summary).with_name("interrupted_summary.json")
    path.write_text(json.dumps(gf))
    args.gf_summary = str(path)
    with pytest.raises(ValueError, match="allow-closed-gf"):
        transfer.run(args, model_factory=model_factory)
    args.allow_closed_gf = True
    result = transfer.run(args, model_factory=model_factory)
    assert result["completed"] and result["size"] == "l"
    assert not result["gf_source_status"]["completed"] and result["gf_source_status"]["closed_by_user"]
    assert json.loads(path.read_text()) == gf
