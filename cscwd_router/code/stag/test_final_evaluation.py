"""CPU checks for nested GT queries, real sparse execution and K100 immutability."""

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from cscwd_router.code.data_prep.final_queries import cohort
from cscwd_router.code.stag import final_multipeople as multi, final_io as io, conditional
from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.dinov3_engine import evaluate as legacy_evaluate
from cscwd_router.code.stag.dinov3_sources import load_source
from cscwd_router.code.stag.test_dinov3_train_test import TinyModel, TinyData, setup_run, model_factory
from cscwd_router.code.stag.test_dinov3_efficiency import TinyRoutedBackbone
from cscwd_router.code.scripts import run_final_k100 as k100, run_final_multi as queue


def transform(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def make_records(tmp_path):
    records = []
    for index, count in enumerate((4, 3, 2)):
        path = f"frame_{index}.jpg"
        Image.new("RGB", (32, 24), (100 + index * 20, 70, 90)).save(tmp_path / path)
        heads = [{"bbox_norm": [.1 * h, .1, .1 * h + .1, .3], "gazex_norm": [.7], "gazey_norm": [.6],
                  "inout": int(index == 0 or h != 1)} for h in range(count)]
        for head in heads:
            if not head["inout"]:
                head.update(gazex_norm=[-1.], gazey_norm=[-1.])
        records.append({"path": path, "heads": heads})
    for split in ("train", "test"):
        (tmp_path / f"{split}_preprocessed.json").write_text(json.dumps([{"frames": records}]))
    return records


def test_same_real_frames_and_nested_gt_heads_without_gaze_selection(tmp_path):
    make_records(tmp_path)
    records, manifest = cohort(tmp_path)
    assert len(records) == manifest["frames"] == 2
    assert manifest == cohort(tmp_path)[1]
    loaders = [multi.loader_for(tmp_path, records, manifest, n, transform, 2, 0) for n in (1, 2, 3)]
    for index in range(2):
        groups = [loader.dataset.head_indices[index] for loader in loaders]
        assert groups[0] == groups[1][:1] == groups[2][:1] and groups[1] == groups[2][:2]
    for n, loader in enumerate(loaders, 1):
        batch = next(iter(loader))
        assert batch[0].shape[0] == 2 and batch[4].numel() == 2 * n
        assert len(loader.dataset.data_idxs) == 2 * n


class MultiTiny(TinyModel):
    def forward(self, inputs):
        features = self.backbone.forward_prefix(inputs["images"], self.route_after_block).dense_features[self.route_after_block]
        routing = self.router(features, inputs["bboxes"])
        features = features * (1 + routing.image_hard_masks[:, None].float())
        maps = F.interpolate(self.linear(features), (64, 64), mode="bilinear", align_corners=False).sigmoid()
        scores = self.inout_head(features.mean((2, 3))).sigmoid().flatten()
        return {"heatmap": [maps[i].expand(len(boxes), -1, -1) for i, boxes in enumerate(inputs["bboxes"])],
                "inout": [scores[i].expand(len(boxes)) for i, boxes in enumerate(inputs["bboxes"])], "routing": routing}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("independent", [False, True])
def test_grouped_metrics_query_ids_outside_labels_and_saved_predictions(tmp_path, n, independent):
    torch.set_num_threads(1)
    make_records(tmp_path)
    records, manifest = cohort(tmp_path)
    loader = multi.loader_for(tmp_path, records, manifest, n, transform, 2, 0)
    base = MultiTiny("gazelle_dinov3_vitb16_inout").eval()
    model = multi.IndependentQueries(base) if independent else base
    path = tmp_path / "predictions.gz"
    metrics = multi.evaluate(model, loader, "cpu", path)
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    assert metrics["persons"] == len(rows) == 2 * n
    assert metrics["inframe_persons"] == sum(row["in_frame"] for row in rows)
    for index, row in enumerate(rows):
        frame, head = loader.dataset.data_idxs[index]
        assert row["path"] == records[frame]["path"] and row["head_index"] == head
        assert row["in_frame"] == records[frame]["heads"][head]["inout"]
    if not independent:
        old = legacy_evaluate(model, loader, "cpu", dataset="vat", keep_ratio=.5)
        for key in ("auc", "l2", "inout_ap", "persons", "inframe_persons", "routing_gt_point_coverage"):
            assert old[key] == metrics[key]


@pytest.mark.parametrize("n", [1, 3])
def test_native_sparse_suffix_is_shared_once_or_batched_independently(monkeypatch, n):
    torch.set_num_threads(1)
    monkeypatch.setattr(conditional, "RoutedDinoV3Backbone", TinyRoutedBackbone)
    model, _ = conditional.get_conditional_model("gazelle_dinov3_vitb16_inout")
    model.eval()
    inputs = {"images": torch.randn(1, 3, 32, 32),
              "bboxes": [[[.1 * i, .1, .1 * i + .1, .3] for i in range(n)]]}
    union = multi.trace(model, inputs, "union")
    wrapped = multi.IndependentQueries(model).eval()
    independent = multi.trace(wrapped, inputs, "independent")
    assert union["encoded_image_copies"] == 1 and independent["encoded_image_copies"] == n
    assert union["block_inputs"][-1]["tokens"] == independent["block_inputs"][-1]["tokens"] == 517
    if n == 1:
        with torch.inference_mode():
            assert torch.equal(model(inputs)["heatmap"][0], wrapped(inputs)["heatmap"][0])


def test_official_predictions_do_not_claim_router_coverage(tmp_path):
    make_records(tmp_path)
    records, selection = cohort(tmp_path)
    model = MultiTiny("gazelle_dinov3_vitb16_inout")
    real = model.forward
    def forward(inputs):
        out = real(inputs)
        out.pop("routing")
        return out
    model.forward = forward
    metrics = multi.evaluate(model, multi.loader_for(tmp_path, records, selection, 3, transform, 2, 0), "cpu", tmp_path / "rows.gz")
    assert metrics["routing_gt_point_coverage"] is None and metrics["persons"] == 6


def test_result_resume_rejects_changed_inputs_and_damaged_artifacts(tmp_path):
    artifact = tmp_path / "rows.gz"
    artifact.write_bytes(b"synthetic")
    path, provenance = tmp_path / "summary.json", tmp_path / "provenance.md"
    io.finish(path, {"request": {"n": 3}, "artifacts": {"rows": io.reference(artifact)}}, provenance, "CPU test")
    assert io.existing(path, {"n": 3}, True)["completed"]
    with pytest.raises(ValueError):
        io.existing(path, {"n": 2}, True)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        io.existing(path, {"n": 3}, True)


def test_k100_does_not_retrain_reselect_or_change_the_original_checkpoint(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch, "gazefollow", "b")
    annotation = tmp_path / "test_preprocessed.json"
    annotation.write_text("[]")
    monkeypatch.setattr(original, "make_datasets", lambda *a, **kw: (TinyData("gazefollow", training=True), TinyData("gazefollow"),
        {"test_frame_sample_every": 1, "test_annotation": io.reference(annotation)}))
    source = original.run(args, model_factory=model_factory)
    summary_path = Path(args.report_dir) / "summary.json"
    source_hash, checkpoint_hash = io.reference(summary_path), io.reference(source["artifacts"]["best_test_checkpoint"]["path"])
    monkeypatch.setattr(io, "verified_source", lambda p, *_: load_source(p))
    monkeypatch.setattr(io, "sources", lambda *names: {"synthetic": "fixed"})
    monkeypatch.setattr(k100, "make_full_test", lambda *a: TinyData("gazefollow"))
    monkeypatch.setattr(k100.efficiency, "trace_forward", lambda model, *a: {"keep_ratio": model.router.keep_ratio})
    calls = []
    def builder(summary, payload, keep_ratio):
        model = TinyModel(summary["model_config"]["model"])
        original.restore_task(model, payload["model_state"])
        model.router.keep_ratio = keep_ratio
        calls.append(keep_ratio)
        return model.eval().requires_grad_(False), None
    request = k100.parse_args(["--summaries", str(summary_path), "--gf-data-path", str(tmp_path), "--vat-data-path", str(tmp_path),
        "--artifact-root", str(tmp_path / "control-weights"), "--report-root", str(tmp_path / "control-reports"),
        "--provenance", str(tmp_path / "provenance.md"), "--batch-size", "2", "--workers", "0", "--device", "cpu", "--resume"])
    result = k100.run(request, model_builder=builder)[0]
    assert calls == [1.] and result["source_selection"] == source["selection"]
    assert result["K100"]["keep_ratio"] == 1 and result["K50_saved"] == source["metrics"]
    assert io.reference(summary_path) == source_hash and io.reference(source["artifacts"]["best_test_checkpoint"]["path"]) == checkpoint_hash
    assert k100.run(request, model_builder=builder)[0] == result and calls == [1.]


def test_timing_summary_reports_images_and_queries_separately():
    cells = []
    for n in (1, 2, 3):
        for order in ("forward", "reverse"):
            cells.append({"request": {"n": n, "order": order, "sample": {"path": "same.jpg"}},
                "results": {v: {"timing": {"latency_ms_median": 10., "throughput_images_per_second_median": 100.,
                                          "peak_allocated_mib": 500.}} for v in multi.VARIANTS}})
    summary = multi.aggregate_timing(cells)
    assert summary["3"]["union"]["queries_per_second"] == pytest.approx(300)
    assert summary["3"]["union"]["images_per_second"] == pytest.approx(100)
    with pytest.raises(ValueError, match="both orders"):
        multi.aggregate_timing(cells[:-1])


def test_multi_dispatcher_completes_fresh_cells_and_validates_resume(tmp_path, monkeypatch):
    common = {"protocol": "synthetic", "seed": 0}
    sample = {"path": "same.jpg", "head_indices": [0, 1, 2], "head_bboxes": [[.1, .1, .2, .2]] * 3}
    args = queue.parse_args(["--source-summary", "/source", "--data-path", "/data", "--official-checkpoint", "/official",
        "--artifact-root", str(tmp_path / "large"), "--report-root", str(tmp_path / "reports"),
        "--provenance", str(tmp_path / "provenance.md"), "--device", "cpu", "--resume"])
    args.report_root.mkdir()
    (args.report_root / "cohort.json").write_text("{}")
    monkeypatch.setattr(queue, "prepare", lambda _a: (None, None, None, {"frames": 2}, common, sample))
    launches = []
    def subprocess_cell(command, **kwargs):
        child = queue.parse_args(command[3:])
        launches.append((child.cell_role, child.n))
        request = queue.request_for(child, common, sample, child.cell_role, child.n, child.variant, child.order)
        path = queue.output_path(child, child.cell_role, child.n, child.variant, child.order)
        if io.existing(path, request, child.resume):
            return
        result = {"request": request, "artifacts": {}}
        if child.cell_role == "accuracy":
            result["metrics"] = {"auc": .9, "l2": .1, "inout_ap": .8, "routing_gt_point_coverage": .95}
        else:
            result.update(gpu_uuid="synthetic-GPU", results={v: {"timing": {
                "latency_ms_median": 10., "throughput_images_per_second_median": 100., "peak_allocated_mib": 500.}}
                for v in multi.VARIANTS})
        io.finish(path, result, child.provenance, "synthetic cell")
    monkeypatch.setattr(queue.subprocess, "run", subprocess_cell)
    report = queue.run(args)
    assert len(launches) == 15 and report["quality_query_counts"]["3"] == 6
    assert len(report["quality"]) == len(report["timing"]) == 3
    assert (args.report_root / "summary.md").exists()
    again = queue.run(args)
    assert again == report
    args.repeats = 6
    with pytest.raises(ValueError, match="existing result differs"):
        queue.run(args)
