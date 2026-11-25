"""CPU-only graph, query accounting and timing aggregation controls."""

import copy

import pytest
import torch

from cscwd_router.code.stag import conditional, dinov3_run, self_multi_timing as timing
from cscwd_router.code.stag.test_dinov3_efficiency import TinyRoutedBackbone
from cscwd_router.code.scripts.run_self_multi_timing import cell_request, command_for, parse_args


@pytest.mark.parametrize("variant,copies,kept", [("k100", 1, 1024), ("independent", 3, 512), ("union", 1, 512)])
def test_native_prefix_suffix_and_decoder_execute_expected_shared_graph(monkeypatch, variant, copies, kept):
    torch.set_num_threads(1)
    monkeypatch.setattr(conditional, "RoutedDinoV3Backbone", TinyRoutedBackbone)
    model, _ = conditional.get_conditional_model("gazelle_dinov3_vitb16_inout")
    source = {"model_config": model.get_model_config()}
    payload = {"model_state": dinov3_run.task_state(model)}
    model, _ = timing.build_variant(source, payload, variant)
    inputs = {"images": torch.randn(1, 3, 512, 512), "bboxes": [[[.1, .1, .3, .3], [.4, .2, .6, .4], [.7, .5, .9, .7]]]}
    observed = timing.trace(model, inputs, variant)
    assert observed["encoded_image_copies"] == copies
    assert observed["queried_people"] == 3
    assert observed["kept_patches_per_copy"] == kept
    assert observed["block_inputs"][-1] == {"block": 11, "batch": copies, "tokens": kept + 5}


def fixture_cells():
    cells = []
    for n in (1, 2, 3):
        for order in ("forward", "reverse"):
            cells.append({"gpu_uuid": "synthetic-test-gpu", "request": {"n": n, "order": order, "sample": [1, 2, 3][:n]},
                "results": {variant: {"timing": {"latency_ms_median": 12., "throughput_images_per_second_median": 100.,
                    "peak_allocated_mib": 16.}} for variant in timing.VARIANTS}})
    return cells


def test_queries_per_second_are_not_reported_as_images_per_second():
    result = timing.aggregate(fixture_cells())
    assert result["3"]["variants"]["union"]["queries_per_second"] == 300.
    assert result["3"]["variants"]["union"]["images_per_second"] == 100.
    assert result["3"]["union_speedup_vs_k100"] == 1.


@pytest.mark.parametrize("damage", ["gpu", "sample", "order", "missing"])
def test_aggregation_rejects_mismatched_cells(damage):
    cells = copy.deepcopy(fixture_cells())
    if damage == "gpu":
        cells[0]["gpu_uuid"] = "another-synthetic-gpu"
    elif damage == "sample":
        cells[0]["request"]["sample"] = [9]
    elif damage == "order":
        cells[0]["request"]["order"] = "reverse"
    else:
        cells.pop()
    with pytest.raises(ValueError):
        timing.aggregate(cells)


def test_nested_requests_and_child_command_preserve_settings(tmp_path):
    args = parse_args(["--source-summary", str(tmp_path / "source.json"), "--data-path", str(tmp_path),
        "--report-root", str(tmp_path / "reports"), "--provenance", str(tmp_path / "provenance.md"), "--resume"])
    common = {"sample": {"head_indices": [7, 4, 2], "head_bboxes": [[1], [2], [3]]}}
    assert cell_request(common, 2, "forward")["sample"]["head_indices"] == [7, 4]
    child = parse_args(command_for(args, 3, "reverse")[3:])
    assert child.n == 3 and child.order == "reverse" and child.resume
    assert child.seed == 0 and child.repeats == 5
