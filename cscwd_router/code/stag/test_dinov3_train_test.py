"""CPU engineering tests only; generated inputs are not experiment evidence."""

from argparse import Namespace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gazelle.routing.router import CoverageAwareSpatialRouter
from gazelle.routing import model as native_model
from gazelle.utils import get_heatmap
from scripts.train_coverage_router import evaluate_model
from cscwd_router.code.data_prep.train_test import make_datasets, make_loader
from cscwd_router.code.scripts.run_router_dinov3_queue import job_command, parse_args as queue_args
from cscwd_router.code.scripts.train_router_dinov3 import parse_args
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag import dinov3_run as runner
from cscwd_router.code.stag.dinov3_engine import configure_stage, evaluate, support_forward, train_epoch
from cscwd_router.code.stag.dinov3_initializers import INITIALIZERS


class TinyBackbone(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.projection = nn.Conv2d(3, channels, 1)
        self.prefix_calls = 0
        self.suffix_calls = 0

    def forward_prefix(self, images, route):
        self.prefix_calls += 1
        features = self.projection(F.adaptive_avg_pool2d(images, (4, 4)))
        return SimpleNamespace(dense_features={route: features})


class TinyModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model_name = name
        self.route_after_block = 11 if "vitl" in name else 5
        channels = 12 if "vitl" in name else 8
        self.backbone = TinyBackbone(channels)
        self.router = CoverageAwareSpatialRouter(channels, hidden_dim=8, keep_ratio=0.5, escape_tokens=0)
        self.linear = nn.Conv2d(channels, 1, 1)
        if name.endswith("inout"):
            self.inout_head = nn.Linear(channels, 1)
        self.router_stage = "backbone_sparse"

    def forward(self, inputs):
        state = self.backbone.forward_prefix(inputs["images"], self.route_after_block)
        features = state.dense_features[self.route_after_block]
        routing = self.router(features, inputs["bboxes"])
        self.backbone.suffix_calls += 1
        features = features * (1 + routing.image_hard_masks[:, None].float())
        maps = F.interpolate(self.linear(features), (64, 64), mode="bilinear", align_corners=False).sigmoid()
        scores = self.inout_head(features.mean((2, 3))).sigmoid().flatten() if hasattr(self, "inout_head") else None
        return {"heatmap": [value for value in maps],
                "inout": [value.reshape(1) for value in scores] if scores is not None else None,
                "routing": routing}

    def get_model_config(self):
        return {"model": self.model_name, "route_after_block": self.route_after_block,
                "router_stage": self.router_stage, "keep_ratio": self.router.keep_ratio,
                "fusion": "raw_concat", "spatial_prior": "none"}


class TinyData(torch.utils.data.Dataset):
    def __init__(self, dataset, *, training=False, out_only=False):
        self.dataset = dataset
        self.training = training
        self.out_only = out_only
        self.data = [{"path": f"frame_{i}.jpg"} for i in range(3)]
        self.data_idxs = [(i, 0) for i in range(3)]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        flag = int(not self.out_only and (self.dataset == "gazefollow" or index != 1))
        x, y = ([0.7, 0.8], [0.6, 0.7]) if self.dataset == "gazefollow" else ([0.7], [0.6])
        if not flag:
            x, y = [-1.0], [-1.0]
        image = torch.randn(3, 14, 14, generator=torch.Generator().manual_seed(index + 10))
        item = (image, [0.1, 0.1, 0.3, 0.3], x, y, torch.tensor(flag), 14, 14)
        return (*item, get_heatmap(x[0], y[0], 64, 64) if flag else torch.zeros(64, 64)) if self.training else item


def model_factory(name, **_kwargs):
    return TinyModel(name), None


def test_selection_uses_min_l2_and_one_complete_tuple():
    old = {"metrics": {"min_l2": 0.05, "avg_l2": 0.1, "auc": 0.95}}
    candidate = {"min_l2": 0.04, "avg_l2": 0.2, "auc": 0.90}
    assert runner.selection_better(candidate, old, "min_l2")
    assert not runner.selection_better(old["metrics"], old, "min_l2")


@pytest.mark.parametrize("dataset", ["gazefollow", "vat"])
@pytest.mark.parametrize("size", ["b", "l"])
def test_queue_uses_size_compatible_independent_initialization(dataset, size):
    args = queue_args(["--dataset", dataset, "--data-path", "/data", "--output-root", "/weights",
                       "--report-root", "/reports", "--provenance", "/provenance.md"])
    command = job_command(args, size, python="/python")
    child = parse_args(command[3:])
    assert child.dataset == dataset and child.size == size
    assert args.seed == child.seed == 0
    assert command[0] == "/python"
    assert command[command.index("--init-kind") + 1] == INITIALIZERS[dataset, size]["kind"]
    assert "val_fraction" not in " ".join(command)
    if dataset == "vat":
        assert "testselected" not in command[command.index("--init-checkpoint") + 1]
    if size == "l":
        assert command[command.index("--batch-size") + 1] == "4"
        assert command[command.index("--grad-accumulation") + 1] == "8"


@pytest.mark.parametrize("size,inout", [("b", False), ("b", True), ("l", False), ("l", True)])
def test_native_factory_b_l_route_depth_and_four_level_projection(monkeypatch, size, inout):
    class ShapeBackbone(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.channels = 768 if "vitb" in name else 1024
            self.depth = 12 if "vitb" in name else 24
        def _validate_route(self, route):
            assert route < self.depth - 1
        def get_dimension(self):
            return self.channels
        def get_out_size(self, shape):
            return shape[0] // 16, shape[1] // 16
        def get_transform(self, shape):
            return None
        def set_trainable_suffix(self, route, *, trainable):
            assert not trainable
    monkeypatch.setattr(native_model, "RoutedDinoV3Backbone", ShapeBackbone)
    name = f"gazelle_dinov3_vit{size}16" + ("_inout" if inout else "")
    model, _ = native_model.get_coverage_router_model(name)
    assert model.route_after_block == (5 if size == "b" else 11)
    assert model.linear.in_channels == 4 * (768 if size == "b" else 1024)
    assert not model.use_sasa and not model.use_ggsf


def test_official_views_use_complete_train_and_test(tmp_path):
    for split in ("train", "test"):
        record = {"path": "image.png", "heads": [{"bbox_norm": [0.1, 0.1, 0.4, 0.4],
                  "gazex_norm": [0.7], "gazey_norm": [0.6], "inout": 1}]}
        (tmp_path / f"{split}_preprocessed.json").write_text(json.dumps([record, record]))
    train, test, metadata = make_datasets("gazefollow", tmp_path, None)
    assert len(train) == len(test) == 2
    assert train.aug and not test.aug
    assert metadata["validation_fraction"] == 0
    assert metadata["selection_policy"] == "test_selected"
    assert train.split == "train" and test.split == "test"


@pytest.mark.parametrize("name", ["gazelle_dinov3_vitb16", "gazelle_dinov3_vitl16"])
def test_support_only_uses_prefix_and_has_frozen_backbone(name):
    torch.manual_seed(4)
    model = TinyModel(name)
    original = runner.task_state(model)
    backbone = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    configure_stage(model, "support")
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=0.01)
    train_epoch(model, make_loader(TinyData("gazefollow", training=True), 2, 0), optimizer, "cpu",
                stage="support", accumulation=1, clip_norm=1, keep_ratio=0.25,
                budget_weight=0.05, log_every=100)
    assert model.backbone.suffix_calls == 0 and model.backbone.prefix_calls == 2
    assert all(torch.equal(v, model.backbone.state_dict()[k]) for k, v in backbone.items())
    assert torch.equal(original["linear.weight"], model.linear.weight)
    assert not torch.equal(original["router.score_head.2.weight"], model.router.score_head[2].weight)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_decoder_handles_out_of_frame_and_partial_accumulation():
    model = TinyModel("gazelle_dinov3_vitb16_inout")
    configure_stage(model, "sparse_decoder")
    old_router = {k: v.clone() for k, v in model.router.state_dict().items()}
    old_inout = model.inout_head.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    metrics = train_epoch(model, make_loader(TinyData("vat", training=True, out_only=True), 1, 0),
                           optimizer, "cpu", stage="sparse_decoder", accumulation=2, clip_norm=1,
                           keep_ratio=0.5, budget_weight=0.05, log_every=100)
    assert metrics["optimizer_steps"] == 2
    assert not torch.equal(old_inout, model.inout_head.weight)
    assert all(torch.equal(v, model.router.state_dict()[k]) for k, v in old_router.items())
    assert all(p.grad is None for p in model.backbone.parameters())


@pytest.mark.parametrize("dataset", ["gazefollow", "vat"])
def test_metrics_match_existing_evaluator_and_export_rows(tmp_path, dataset):
    model = TinyModel("gazelle_dinov3_vitb16" + ("_inout" if dataset == "vat" else ""))
    loader = make_loader(TinyData(dataset), 2, 0)
    reference = evaluate_model(model, loader, Namespace(dataset=dataset, amp=False, keep_ratio=0.5), torch.device("cpu"))
    path = tmp_path / "rows.jsonl.gz"
    actual = evaluate(model, loader, "cpu", dataset=dataset, keep_ratio=0.5, rows_path=path)
    for key in ("auc", "avg_l2", "min_l2") if dataset == "gazefollow" else ("auc", "l2", "inout_ap"):
        assert actual[key] == pytest.approx(reference[key], abs=1e-12)
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 3 and rows[0]["path"] == "frame_0.jpg"


def test_initialization_rejects_wrong_scale_and_missing_decoder():
    small = TinyModel("gazelle_dinov3_vitb16")
    large = TinyModel("gazelle_dinov3_vitl16")
    with pytest.raises(RuntimeError, match="size mismatch"):
        runner.initialize(large, runner.task_state(small), "support")
    with pytest.raises(ValueError, match="initialization keys differ"):
        runner.initialize(small, {}, "decoder")
    state = {k: v for k, v in runner.task_state(small).items() if not k.startswith("router.")}
    assert runner.initialize(small, state, "decoder")["router_reinitialized"]
    inherited = runner.task_state(small)
    before = small.router.scene_projection.weight.detach().clone()
    inherited["router.scene_projection.weight"].fill_(10)
    runner.initialize(small, inherited, "decoder")
    assert torch.equal(before, small.router.scene_projection.weight)


def setup_run(tmp_path, monkeypatch, dataset="vat", size="b"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"synthetic_test": "fixed"})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "synthetic-test\n")
    torch.manual_seed(9)
    name = f"gazelle_dinov3_vit{size}16" + ("_inout" if dataset == "vat" else "")
    initial = TinyModel(name)
    init = tmp_path / "initial.pt"
    torch.save(runner.task_state(initial), init)
    backbone = tmp_path / "checkpoints" / f"dinov3_vit{size}16_pretrain.pth"
    backbone.parent.mkdir(exist_ok=True)
    backbone.write_bytes(b"synthetic-backbone-reference")
    args = parse_args(["--dataset", dataset, "--size", size, "--data-path", str(tmp_path),
                       "--init-checkpoint", str(init), "--init-sha256", sha256(init),
                       "--init-kind", "decoder", "--backbone-sha256", sha256(backbone),
                       "--run-dir", str(tmp_path / "run"), "--report-dir", str(tmp_path / "reports"),
                       "--provenance", str(tmp_path / "provenance.md"), "--device", "cpu",
                       "--workers", "0", "--batch-size", "2", "--eval-batch-size", "2",
                       "--support-epochs", "1", "--decoder-epochs", "2", "--log-every", "100"])
    monkeypatch.setattr(runner, "make_datasets", lambda *a, **kw: (
        TinyData(dataset, training=True), TinyData(dataset),
        {"train_split": "train", "evaluation_split": "test", "validation_fraction": 0,
         "test_frame_sample_every": 1 if dataset == "gazefollow" else 6}))
    calls = []
    def full_test(*args):
        calls.append("rate1")
        return TinyData(dataset)
    monkeypatch.setattr(runner, "make_full_test", full_test)
    return args, calls


@pytest.mark.parametrize("dataset,size", [("gazefollow", "b"), ("gazefollow", "l"), ("vat", "b"), ("vat", "l")])
def test_complete_job_and_verified_resume(tmp_path, monkeypatch, dataset, size):
    args, calls = setup_run(tmp_path, monkeypatch, dataset, size)
    summary = runner.run(args, model_factory=model_factory)
    assert summary["completed"] and summary["training_epochs"] == {"support": 1, "sparse_decoder": 2}
    assert summary["model_config"]["route_after_block"] == (11 if size == "l" else 5)
    assert bool(calls) == (dataset == "vat")
    args.resume = True
    assert runner.run(args, model_factory=model_factory) == summary
    args.seed = 1
    with pytest.raises(ValueError, match="configuration"):
        runner.run(args, model_factory=model_factory)


def test_resume_recovers_epoch_committed_before_report_write(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch)
    real_json = runner.atomic_json
    crashed = []
    def interrupted(path, value):
        if Path(path).name == "sparse_decoder_history.json" and not crashed:
            crashed.append(True)
            raise RuntimeError("synthetic interruption after epoch commit")
        real_json(path, value)
    monkeypatch.setattr(runner, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(args, model_factory=model_factory)
    args.resume = True
    monkeypatch.setattr(runner, "atomic_json", real_json)
    resumed = runner.run(args, model_factory=model_factory)
    resumed_state = torch.load(resumed["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)["model_state"]
    args.resume = False
    args.run_dir, args.report_dir = str(tmp_path / "fresh"), str(tmp_path / "fresh-reports")
    fresh = runner.run(args, model_factory=model_factory)
    fresh_state = torch.load(fresh["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)["model_state"]
    assert resumed["metrics"] == fresh["metrics"]
    assert all(torch.equal(value, fresh_state[key]) for key, value in resumed_state.items())
