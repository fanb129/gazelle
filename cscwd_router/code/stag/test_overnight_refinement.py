"""Check the changed scientific controls and the classifier-only boundary."""

import torch
from torch import nn

from gazelle.routing.router import CoverageAwareSpatialRouter
from cscwd_router.code.stag.conditional import configure_inputs
from cscwd_router.code.scripts.train_vat_ap import configure_head, better_ap


def router():
    torch.manual_seed(0)
    module = CoverageAwareSpatialRouter(8, hidden_dim=8, keep_ratio=.5, escape_tokens=0)
    nn.init.normal_(module.score_head[-1].weight)
    return module


def test_unconditional_route_ignores_box_values_and_query_count():
    model = configure_inputs(router(), "unconditional")
    scene = torch.randn(1, 8, 4, 4)
    a = model(scene, [[[0., 0., .25, .25]]])
    b = model(scene, [[[.5, .5, 1., 1.], [.2, .3, .6, .8], [.1, .1, .2, .2]]])
    assert torch.equal(a.image_keep_indices, b.image_keep_indices)
    assert all(torch.equal(a.support_probs[0], value) for value in b.support_probs)
    assert not b.person_head_masks.any()
    assert b.image_hard_masks.sum() == 8
    b.support_logits.square().mean().backward()
    assert model.scene_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.head_projection.parameters())
    assert all(p.grad is None for p in model.geometry_projection.parameters())


def test_head_priority_is_actually_removed_without_changing_learned_scene_scores():
    base = router()
    strict = configure_inputs(base, "unconditional")
    scene_only = configure_inputs(base, "scene_only")
    scene = torch.randn(1, 8, 4, 4)
    scores = strict(scene, [[[0., 0., .25, .25]]])
    index = scores.image_union_probs.flatten().argmin().item()
    row, col = divmod(index, 4)
    boxes = [[(col / 4, row / 4, (col + 1) / 4, (row + 1) / 4)]]
    old, new = scene_only(scene, boxes), strict(scene, boxes)
    assert torch.allclose(old.support_logits, new.support_logits)
    assert old.image_hard_masks.flatten()[index]
    assert not new.image_hard_masks.flatten()[index]


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
        self.inout_token = nn.Parameter(torch.zeros(4))
        self.inout_head = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())


def test_ap_update_freezes_token_and_shared_path():
    torch.manual_seed(0)
    model = Toy()
    configure_head(model)
    assert not model.training and model.inout_head.training
    x = torch.randn(8, 4)
    before = model.shared(x).detach().clone()
    token = model.inout_token.detach().clone()
    initial = model.inout_head[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.inout_head.parameters(), lr=.01, weight_decay=0)
    prediction = model.inout_head(model.shared(x) + model.inout_token)
    nn.functional.binary_cross_entropy(prediction, torch.ones_like(prediction)).backward()
    optimizer.step()
    assert torch.equal(before, model.shared(x))
    assert torch.equal(token, model.inout_token)
    assert not torch.equal(initial, model.inout_head[0].weight)
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("inout_head."))


def test_ap_selection_maximizes_ap_and_keeps_earlier_ties():
    incumbent = {"metrics": {"inout_ap": .88, "l2": .10}}
    assert better_ap({"inout_ap": .89, "l2": .11}, incumbent)
    assert not better_ap({"inout_ap": .88, "l2": .09}, incumbent)


def test_ap_runner_saves_one_complete_checkpoint_and_resumes(tmp_path, monkeypatch):
    from cscwd_router.code.scripts import train_vat_ap as ap
    from cscwd_router.code.stag.test_dinov3_train_test import TinyModel, TinyData
    from cscwd_router.code.stag.dinov3_run import task_state
    from cscwd_router.code.stag.dinov3_engine import evaluate
    from cscwd_router.code.data_prep.train_test import make_loader
    torch.set_num_threads(1)
    model = TinyModel("gazelle_dinov3_vitb16_inout")
    metrics = evaluate(model, make_loader(TinyData("vat"), 2, 0), "cpu", dataset="vat", keep_ratio=.5)
    source_file = tmp_path / "source.json"
    source_file.write_text('{}')
    source = {"dataset": "vat", "size": "b", "selection": {"metrics": metrics},
              "metrics": metrics, "artifacts": {"backbone": {"path": "backbone", "sha256": "unused"}}}
    payload = {"model_state": task_state(model)}
    monkeypatch.setattr(ap.io, "verified_source", lambda *_: (source, {}, payload))
    monkeypatch.setattr(ap.io, "build_router", lambda *_: (model, None))
    monkeypatch.setattr(ap.io, "sources", lambda *_: {"test": "fixed"})
    monkeypatch.setattr(ap, "make_datasets", lambda *_args, **_kw: (TinyData("vat", training=True), TinyData("vat"), {}))
    monkeypatch.setattr(ap, "make_full_test", lambda *_: TinyData("vat"))
    args = ap.parse_args(["--source-summary", str(source_file), "--data-path", str(tmp_path),
        "--run-dir", str(tmp_path / "weights"), "--report-dir", str(tmp_path / "reports"),
        "--provenance", str(tmp_path / "provenance.md"), "--epochs", "2", "--workers", "0",
        "--batch-size", "2", "--eval-batch-size", "2", "--device", "cpu"])
    result = ap.run(args)
    saved = torch.load(result["artifacts"]["best_test_checkpoint"]["path"], weights_only=False)
    assert saved["selection"] == result["selection"]
    assert set(saved["model_state"]) == set(payload["model_state"])
    assert all(key.startswith("inout_head.") for key in result["changed_tensors"])
    args.resume = True
    assert ap.run(args) == result
