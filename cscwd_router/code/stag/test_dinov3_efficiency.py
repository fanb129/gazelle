"""CPU controls only; synthetic tensors and timing fixtures are not results."""

import copy
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gazelle.routing import model as native
from gazelle.routing.backbone import RoutedDinoV3Backbone
from cscwd_router.code.stag import dinov3_efficiency as efficiency
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_sources import load_source
from cscwd_router.code.stag.test_dinov3_train_test import setup_run, model_factory
from cscwd_router.code.scripts.run_router_efficiency import cell_command, parse_args
from cscwd_router.code.scripts.benchmark_router_dinov3 import parse_args as cell_args


def closed_source(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch, "gazefollow", "l")
    summary = original.run(args, model_factory=model_factory)
    summary.update(completed=False, closed_by_user=True, checkpoint_export_complete=True,
                   format="cscwd_router_dinov3_interrupted_v1")
    path = tmp_path / "reports/interrupted_summary.json"
    path.write_text(json.dumps(summary))
    return path, summary


def test_closed_source_needs_explicit_consent_and_preserves_incomplete_status(tmp_path, monkeypatch):
    path, summary = closed_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="allow-closed-gf"):
        load_source(path)
    loaded, _, _ = load_source(path, allow_closed_gf=True)
    assert loaded == summary and not loaded["completed"]
    assert json.loads(path.read_text()) == summary


@pytest.mark.parametrize("damage", ["history", "best_state", "checkpoint_sha"])
def test_closed_source_rejects_inconsistent_evidence(tmp_path, monkeypatch, damage):
    path, summary = closed_source(tmp_path, monkeypatch)
    if damage == "history":
        history = path.with_name("sparse_decoder_history.json")
        history.write_text(json.dumps(json.loads(history.read_text())[:-1]))
    elif damage == "best_state":
        item = summary["artifacts"]["best_test_checkpoint"]
        payload = torch.load(item["path"], weights_only=False)
        next(iter(payload["model_state"].values())).add_(1)
        torch.save(payload, item["path"])
        item["sha256"] = sha256(item["path"])
        path.write_text(json.dumps(summary))
    else:
        with open(summary["artifacts"]["best_test_checkpoint"]["path"], "ab") as handle:
            handle.write(b"synthetic corruption")
    with pytest.raises(ValueError):
        load_source(path, allow_closed_gf=True)


class TinyBlock(nn.Module):
    def forward(self, x, rope=None):
        return x + 0.01


class TinyViT(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.n_storage_tokens = 4
        self.embed_dim = 24
        self.patch_size = 16
        self.chunked_blocks = False
        self.rope_embed = None
        self.projection = nn.Conv2d(3, 24, 1)
        self.norm = nn.LayerNorm(24)
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(depth)])

    def prepare_tokens_with_masks(self, images):
        patches = self.projection(F.adaptive_avg_pool2d(images, (32, 32))).flatten(2).transpose(1, 2)
        special = patches.new_zeros(images.shape[0], 5, 24)
        return torch.cat((special, patches), 1), (32, 32)


class TinyRoutedBackbone(RoutedDinoV3Backbone):
    def __init__(self, name):
        nn.Module.__init__(self)
        large = "vitl" in name
        self.model = TinyViT(24 if large else 12)
        self.out_indices = [5, 11, 17, 23] if large else [2, 5, 8, 11]


@pytest.mark.parametrize("size", ["b", "l"])
@pytest.mark.parametrize("variant", ["k100", "k50"])
def test_token_trace_runs_native_routing_prefix_suffix_and_multilayer_decoder(monkeypatch, size, variant):
    torch.set_num_threads(1)
    monkeypatch.setattr(native, "RoutedDinoV3Backbone", TinyRoutedBackbone)
    model, _ = native.get_coverage_router_model(f"gazelle_dinov3_vit{size}16", keep_ratio=.5 if variant == "k50" else 1.)
    model.eval().requires_grad_(False)
    inputs = {"images": torch.randn(1, 3, 512, 512), "bboxes": [[[.1, .1, .3, .3]]]}
    trace = efficiency.trace_forward(model, inputs, variant, "fp32")
    assert len(trace["block_inputs"]) == (24 if size == "l" else 12)
    assert trace["block_inputs"][0]["tokens"] == 1029
    assert trace["block_inputs"][-1]["tokens"] == (517 if variant == "k50" else 1029)
    assert trace["out_indices"] == model.backbone.out_indices
    assert all(not block._forward_pre_hooks for block in model.backbone.model.blocks)


@pytest.mark.parametrize("dataset", ["gazefollow", "vat"])
def test_samples_are_real_single_queries_stable_across_batch_sizes(tmp_path, dataset):
    records = []
    for index in range(4):
        name = f"frame{index}.png"
        Image.new("RGB", (20, 20), (index, 0, 0)).save(tmp_path / name)
        records.append({"path": name, "heads": [{"inout": 0, "bbox_norm": [.5, .5, .8, .8]},
                                                {"inout": 1, "bbox_norm": [.1, .1, .3, .3]}]})
    annotation = records if dataset == "gazefollow" else [
        {"path": "clip0", "frames": records[:2]}, {"path": "clip1", "frames": records[2:]}]
    (tmp_path / "train_preprocessed.json").write_text(json.dumps(annotation))
    one = efficiency.select_samples(tmp_path, 1, 0, dataset=dataset)
    many = efficiency.select_samples(tmp_path, 4, 0, dataset=dataset)
    assert one["samples"] == many["samples"][:1]
    assert all(row["head_index"] == 1 and row["head_bbox"] == [.1, .1, .3, .3] for row in many["samples"])
    transform = lambda image: torch.ones(3, 448, 448)
    inputs = efficiency.prepare_inputs(tmp_path, one["samples"], transform, "cpu")
    assert inputs["images"].shape == (1, 3, 448, 448) and len(inputs["bboxes"][0]) == 1


def timing_cells():
    cells = []
    for batch in (1, 8):
        for order in efficiency.ORDERS:
            cells.append({"request": {"dataset": "gazefollow", "size": "b", "batch_size": batch,
                                      "order": order, "raw_input": {"fixture": batch}, "source": "same"},
                          "environment": {"gpu_uuid": "synthetic-test"}, "source_metrics": {},
                          "source_selection": {}, "source_closed_by_user": False,
                          "results": {name: {"timing": {"latency_ms_median": value,
                               "throughput_images_per_second_median": batch * 1000 / value,
                               "peak_allocated_mib": 1}} for name, value in (("official", 12), ("k100", 16), ("k50", 10))}})
    return cells


def test_summary_uses_geometric_order_ratios_and_does_not_invent_a_gate():
    result = efficiency.summarize(timing_cells())
    assert result["efficiency"]["8"]["k50_speedup_vs_official"] == pytest.approx(1.2)
    assert result["efficiency"]["8"]["k50_speedup_vs_k100"] == pytest.approx(1.6)
    assert "gate" not in result
    assert "K100" in efficiency.render_summary(result)


@pytest.mark.parametrize("damage", ["missing", "gpu", "queries", "checkpoint"])
def test_summary_rejects_incomplete_or_incomparable_cells(damage):
    cells = copy.deepcopy(timing_cells())
    if damage == "missing": cells.pop()
    if damage == "gpu": cells[0]["environment"]["gpu_uuid"] = "other"
    if damage == "queries": cells[0]["request"]["raw_input"] = {"fixture": "other"}
    if damage == "checkpoint": cells[0]["request"]["source"] = "other"
    with pytest.raises(ValueError): efficiency.summarize(cells)


def test_queue_passes_every_timing_option_and_uses_official_inout_for_vat():
    args = parse_args(["--summaries", "/source.json", "--data-path", "/data", "--official-checkpoint-dir", "/ckpt",
                       "--report-root", "/reports", "--provenance", "/prov.md", "--allow-closed-gf"])
    command = cell_command(args, args.summaries[0], {"dataset": "vat", "size": "l"}, 8, "reverse", "/cell.json")
    child = cell_args(command[3:])
    assert child.official_checkpoint.name == "gazelle_dinov2_vitl14_inout.pt"
    assert child.batch_size == 8 and child.order == "reverse" and child.allow_closed_gf
    assert (child.warmup, child.latency_iters, child.throughput_iters, child.repeats, child.seed) == (50, 200, 500, 5, 0)


@pytest.mark.parametrize("dataset", ["gazefollow", "vat"])
def test_queue_fresh_cells_verified_resume_and_changed_settings(tmp_path, monkeypatch, dataset):
    from cscwd_router.code.scripts import run_router_efficiency as queue
    args, _ = setup_run(tmp_path, monkeypatch, dataset, "b")
    original.run(args, model_factory=model_factory)
    monkeypatch.setattr(efficiency, "ROOT", tmp_path)
    monkeypatch.setattr(efficiency, "source_hashes", lambda: {"fixture": "fixed"})
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "checkpoints/dinov2_vitb14_pretrain.pth").write_bytes(b"synthetic backbone")
    suffix = "_inout" if dataset == "vat" else ""
    (tmp_path / f"checkpoints/gazelle_dinov2_vitb14{suffix}.pt").write_bytes(b"synthetic decoder")
    rows = []
    for index in range(8):
        name = f"sample{index}.png"
        Image.new("RGB", (8, 8)).save(tmp_path / name)
        rows.append({"path": name, "heads": [{"inout": 1, "bbox_norm": [.1, .1, .3, .3]}]})
    annotation = rows if dataset == "gazefollow" else [{"path": "clip", "frames": rows}]
    (tmp_path / "train_preprocessed.json").write_text(json.dumps(annotation))
    calls = []
    def fake_cell(command, **kwargs):
        child = cell_args(command[3:])
        cell = timing_cells()[0]
        cell["request"] = efficiency.make_request(child)
        child.output.parent.mkdir(parents=True, exist_ok=True)
        child.output.write_text(json.dumps(cell))
        calls.append(command)
    monkeypatch.setattr(queue.subprocess, "run", fake_cell)
    settings = parse_args(["--summaries", str(tmp_path / "reports/summary.json"), "--data-path", str(tmp_path),
        "--official-checkpoint-dir", str(tmp_path / "checkpoints"), "--report-root", str(tmp_path / "timing"),
        "--provenance", str(tmp_path / "timing_provenance.md")])
    queue.run(settings)
    assert len(calls) == 4 and (tmp_path / f"timing/{dataset}/vitb/summary.json").exists()
    settings.resume = True
    queue.run(settings)
    assert len(calls) == 4
    settings.warmup += 1
    with pytest.raises(ValueError, match="existing benchmark differs"):
        queue.run(settings)
