"""CPU checks for cue removal, native equivalence and GF-to-VAT inheritance."""

import json
from pathlib import Path

import pytest
import torch

from gazelle.routing.router import CoverageAwareSpatialRouter
from gazelle.routing import model as native
from cscwd_router.code.stag import conditional, dinov3_run as gf_run, dinov3_transfer as vat_run
from cscwd_router.code.scripts import run_final_ablation as queue, train_conditioned_router as entry
from cscwd_router.code.stag.test_dinov3_train_test import TinyData, TinyModel, setup_run
from cscwd_router.code.stag.test_dinov3_efficiency import TinyRoutedBackbone


@pytest.mark.parametrize("inout", [False, True])
def test_full_native_model_is_bitwise_unchanged(monkeypatch, inout):
    torch.set_num_threads(1)
    monkeypatch.setattr(native, "RoutedDinoV3Backbone", TinyRoutedBackbone)
    monkeypatch.setattr(conditional, "RoutedDinoV3Backbone", TinyRoutedBackbone)
    name = "gazelle_dinov3_vitb16" + ("_inout" if inout else "")
    torch.manual_seed(0)
    baseline, _ = native.get_coverage_router_model(name, keep_ratio=.5)
    torch.manual_seed(0)
    candidate, _ = conditional.get_conditional_model(name, condition="full")
    assert baseline.get_model_config() == candidate.get_model_config()
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in baseline.state_dict().items())
    inputs = {"images": torch.randn(1, 3, 32, 32), "bboxes": [[[.1, .1, .2, .3], [.6, .2, .8, .4]]]}
    with torch.inference_mode():
        a, b = baseline.eval()(inputs), candidate.eval()(inputs)
    assert all(torch.equal(x, y) for x, y in zip(a["heatmap"], b["heatmap"]))
    assert torch.equal(a["routing"].image_keep_indices, b["routing"].image_keep_indices)
    if inout:
        assert all(torch.equal(x, y) for x, y in zip(a["inout"], b["inout"]))


@pytest.mark.parametrize("condition", ["scene_only", "scene_geometry", "scene_head"])
def test_removed_cues_cannot_leak_through_head_projection(condition):
    torch.manual_seed(0)
    router = CoverageAwareSpatialRouter(8, hidden_dim=8, keep_ratio=.5)
    keys = set(router.state_dict())
    conditional.configure_inputs(router, condition)
    assert keys == set(router.state_dict())
    value = torch.randn(2, 12, requires_grad=True)
    projected = router.head_projection(value)
    projected.square().sum().backward()
    if condition == "scene_only":
        assert torch.count_nonzero(projected) == 0
    elif condition == "scene_geometry":
        assert torch.count_nonzero(value.grad[:, :8]) == 0
        assert torch.count_nonzero(value.grad[:, 8:]) > 0
    else:
        assert torch.count_nonzero(value.grad[:, :8]) > 0
        assert torch.count_nonzero(value.grad[:, 8:]) == 0
    if condition != "scene_geometry":
        assert torch.count_nonzero(router.geometry_projection(torch.randn(2, 6, 4, 4))) == 0


def test_scene_only_scores_ignore_the_query_but_keep_native_head_priority():
    torch.manual_seed(4)
    router = conditional.configure_inputs(CoverageAwareSpatialRouter(8, hidden_dim=8, keep_ratio=.5, escape_tokens=2), "scene_only")
    torch.nn.init.normal_(router.score_head[-1].weight)
    features = torch.randn(1, 8, 4, 4)
    first = router(features, [[[0, 0, .25, .25]]])
    second = router(features, [[[.75, .75, 1, 1]]])
    assert torch.equal(first.support_logits, second.support_logits)
    assert first.image_hard_masks[0, 0, 0] and second.image_hard_masks[0, -1, -1]
    assert first.image_keep_indices.shape == second.image_keep_indices.shape == (1, 8)


class TinyConditional(TinyModel):
    def __init__(self, name, condition):
        super().__init__(name)
        self.condition = condition
        conditional.configure_inputs(self.router, condition)

    def get_model_config(self):
        return {**super().get_model_config(), "routing_condition": self.condition}


@pytest.mark.parametrize("condition", ["scene_only", "scene_geometry", "scene_head"])
def test_training_resume_and_vat_inherit_the_matching_branch(tmp_path, monkeypatch, condition):
    args, _ = setup_run(tmp_path, monkeypatch, "gazefollow", "b")
    args.condition = condition
    factory = lambda name, **kw: (TinyConditional(name, condition), None)
    gf = gf_run.run(args, model_factory=factory, extra_sources={"conditional": "test-version"})
    assert gf["model_config"]["routing_condition"] == condition
    assert gf["training_epochs"] == {"support": 1, "sparse_decoder": 2}
    args.resume = True
    assert gf_run.run(args, model_factory=factory, extra_sources={"conditional": "test-version"}) == gf
    with pytest.raises(ValueError, match="configuration|manifest"):
        gf_run.run(args, model_factory=factory, extra_sources={"conditional": "changed-version"})
    monkeypatch.setattr(vat_run, "ROOT", tmp_path)
    monkeypatch.setattr(vat_run, "source_hashes", lambda: {"synthetic": "fixed"})
    monkeypatch.setattr(vat_run, "make_datasets", lambda *a, **kw: (
        TinyData("vat", training=True), TinyData("vat"), {"validation_fraction": 0, "test_frame_sample_every": 6}))
    monkeypatch.setattr(vat_run, "make_full_test", lambda *a: TinyData("vat"))
    from cscwd_router.code.scripts.transfer_gf_router_to_vat import parse_args
    transferred = parse_args(["--gf-summary", str(Path(args.report_dir) / "summary.json"),
        "--data-path", str(tmp_path), "--run-dir", str(tmp_path / "vat-run"),
        "--report-dir", str(tmp_path / "vat-report"), "--provenance", str(tmp_path / "provenance.md"),
        "--epochs", "2", "--workers", "0", "--batch-size", "2", "--eval-batch-size", "2", "--device", "cpu"])
    transferred.condition = condition
    result = vat_run.run(transferred, model_factory=factory, extra_sources={"conditional": "test-version"})
    assert result["initialization"]["shared_tensors_identical"]
    assert result["model_config"]["routing_condition"] == condition
    initial = torch.load(result["artifacts"]["initial_checkpoint"]["path"], weights_only=False)
    source = torch.load(gf["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)
    assert all(torch.equal(v, initial["model_state"][k]) for k, v in source["model_state"].items())
    assert all(row["train"]["trainable_parameters"]["backbone"] == 0 for row in json.loads((tmp_path / "vat-report/history.json").read_text()))


@pytest.mark.parametrize("condition", ["scene_only", "scene_geometry", "scene_head"])
def test_queue_preserves_the_full_recipe_and_branch_identity(condition):
    args = queue.parse_args(["--condition", condition, "--gf-data-path", "/gf", "--vat-data-path", "/vat",
        "--initialization-root", "/old", "--artifact-root", "/weights", "--report-root", "/reports", "--provenance", "/provenance", "--resume"])
    gf, vat = queue.commands(args, python="/python")
    phase, a = entry.parse_args(gf[3:]);phase2, b = entry.parse_args(vat[3:])
    assert (phase, phase2) == ("gf", "vat")
    assert a.seed == b.seed == 0 and a.condition == b.condition == condition
    assert a.support_epochs == 3 and a.decoder_epochs == 5 and b.epochs == 8
    assert b.gf_summary == str(Path(a.report_dir) / "summary.json")
    assert a.resume and b.resume and a.batch_size * a.grad_accumulation == 32


def test_wrong_gf_branch_is_rejected_before_training(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"model_config": {"routing_condition": "scene_head"}}))
    from argparse import Namespace
    with pytest.raises(ValueError, match="own conditional GF"):
        entry.run("vat", Namespace(gf_summary=str(summary), condition="scene_only"))
