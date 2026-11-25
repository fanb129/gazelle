"""CPU synthetic visual-output tests, never paper evidence."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from cscwd_router.code.stag import final_visuals as visuals, final_io as io
from cscwd_router.code.stag.test_final_evaluation import MultiTiny, transform, make_records
from cscwd_router.code.data_prep.final_queries import cohort
from cscwd_router.code.stag.final_multipeople import IndependentQueries
from cscwd_router.code.scripts.render_final_visualizations import render


def test_lower_and_higher_error_selection_is_explicit_and_deterministic():
    predictions = [{"errors": [{"l2": value}]} for value in (.8, .2, .9, .1, .3)]
    selected = visuals.choose_error_examples(predictions, 4)
    assert selected == [(3, "lower_error"), (1, "lower_error"), (0, "higher_error"), (2, "higher_error")]
    with pytest.raises(ValueError, match="candidate pool"):
        visuals.choose_error_examples(predictions, 6)


def test_person_variation_measures_different_supports_without_gt_or_predictions():
    equal = {"support_probs": [[[.25, .25], [.25, .25]]] * 3}
    different = {"support_probs": [[[.97, .01], [.01, .01]], [[.01, .97], [.01, .01]], [[.01, .01], [.97, .01]]]}
    assert visuals.person_variation(equal) == 0
    assert visuals.person_variation(different) > .9


def test_multi_visual_eligibility_uses_the_fixed_selected_heads(tmp_path):
    make_records(tmp_path)
    records, selection = cohort(tmp_path)
    assert len(visuals.multi_candidates(records, selection, 0, 16, 0)) == 2
    chosen = visuals.multi_candidates(records, selection, 0, 16, 3)
    assert len(chosen) == 1 and records[chosen[0]]["path"] == "frame_0.jpg"
    assert visuals.multi_candidates(records, selection, 0, 16, 2) == visuals.multi_candidates(records, selection, 0, 16, 0)


def test_prediction_point_is_saved_before_array_rounding(tmp_path):
    make_records(tmp_path)
    record = json.loads((tmp_path / "test_preprocessed.json").read_text())[0]["frames"][0]
    class Peak(nn.Module):
        def forward(self, inputs):
            heatmap = torch.zeros(1, 64, 64)
            heatmap[0, 50, 40], heatmap[0, 51, 41] = .00000041, .00000049
            return {"heatmap": [heatmap], "inout": [torch.tensor([.8])]}
    saved = visuals.predict(Peak(), transform, tmp_path, record, [0], "vat", "cpu")
    assert np.max(saved["heatmaps"]) == 0
    assert saved["prediction_xy"][0] == [41 / 64, 51 / 64]
    assert saved["errors"][0]["l2"] > 0


class VisualTiny(MultiTiny):
    def __init__(self, gf=False):
        super().__init__("gazelle_dinov3_vitb16_inout")
        self.gf = gf

    def forward(self, inputs):
        output = super().forward(inputs)
        if self.gf:
            output["inout"] = None
        return output


def exported_fixture(tmp_path, monkeypatch):
    project = tmp_path / "server" / "cscwd_router"
    data_path = tmp_path / "data"
    data_path.mkdir()
    records = make_records(data_path)
    # Four candidate heads from real records in this synthetic dataset.
    candidates = [(0, 0), (0, 1), (0, 2), (1, 0)]
    multi_records, selection = cohort(data_path)
    source_gf = {"size": "b", "dataset": "gazefollow", "selection": {"epoch": 0}}
    source_vat = {"size": "b", "dataset": "vat", "selection": {"epoch": 0}}
    sources = {"gf": (source_gf, {}, {}), "vat": (source_vat, {}, {}), "multi": (source_vat, {}, {})}
    request = {"protocol": "synthetic_CPU_visualization_fixture", "seed": 0, "multi_selection_policy": "synthetic_pool", "multi_min_inframe": 2}
    args = SimpleNamespace(report_root=project / "results" / "visuals", figure_root=project / "latex" / "figures" / "visuals",
        gf_data_path=data_path, vat_data_path=data_path, device="cpu", examples_per_dataset=4, multi_examples=1,
        official_checkpoint_dir=tmp_path, provenance=project / "results" / "provenance.md", resume=True, check_only=False)
    monkeypatch.setattr(visuals, "prepare", lambda _a: (sources, {"gf": records, "vat": records, "multi": multi_records},
        {"gf": candidates, "vat": candidates}, [1], selection, request))
    calls = []
    def builder(source, payload, **kwargs):
        calls.append(source["dataset"])
        return VisualTiny(source["dataset"] == "gazefollow").eval().requires_grad_(False), transform
    monkeypatch.setattr(visuals.io, "build_router", builder)
    def variant(source, payload, official, which):
        model, trans = builder(source, payload)
        return (IndependentQueries(model).eval() if which == "independent" else model), trans
    monkeypatch.setattr(visuals.multi, "build_variant", variant)
    monkeypatch.setattr(visuals.efficiency, "build_variant", variant)
    torch.set_num_threads(1)
    result = visuals.export(args)
    return args, result, calls


def test_export_has_true_maps_masks_gt_and_resume_without_forward(tmp_path, monkeypatch):
    args, result, calls = exported_fixture(tmp_path, monkeypatch)
    assert len(result["cases"]) == 9
    case = json.loads((args.report_root / result["cases"][-1]["path"]).read_text())
    assert case["kind"] == "multi" and len(case["heads"]) == 3
    assert np.asarray(case["predictions"]["union"]["kept_masks"]).shape == (1, 4, 4)
    assert np.asarray(case["predictions"]["independent"]["kept_masks"]).shape == (3, 4, 4)
    np.testing.assert_allclose(case["predictions"]["union"]["union_probs"][0],
        np.max(case["predictions"]["union"]["support_probs"], axis=0), atol=1e-8)
    count = len(calls)
    assert visuals.export(args) == result and len(calls) == count
    assert all("error_bucket" in json.loads((args.report_root / ref["path"]).read_text()) for ref in result["cases"][:8])


def test_cpu_rendering_is_portable_after_server_paths_are_removed(tmp_path, monkeypatch):
    args, result, _ = exported_fixture(tmp_path, monkeypatch)
    local = tmp_path / "local" / "cscwd_router"
    shutil.copytree(tmp_path / "server" / "cscwd_router", local)
    shutil.rmtree(tmp_path / "server")
    shutil.rmtree(tmp_path / "data")
    # Renderer intentionally needs only exported JSON and relative RGB assets.
    report = render(local / "results/visuals/manifest.json", tmp_path / "offline_figures")
    assert not report["model_forward_executed"] and len(report["figures"]) == 8
    for ref in report["figures"]:
        assert Path(ref["path"]).stat().st_size > 1000
        if ref["path"].endswith(".png"):
            with Image.open(ref["path"]) as image:
                assert image.width > 1000 and image.height > 1000


def test_render_rejects_a_swapped_rgb_asset(tmp_path, monkeypatch):
    args, result, _ = exported_fixture(tmp_path, monkeypatch)
    case = json.loads((args.report_root / result["cases"][0]["path"]).read_text())
    (args.report_root / case["image_relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="RGB SHA256"):
        render(args.report_root / "manifest.json", tmp_path / "figures")
