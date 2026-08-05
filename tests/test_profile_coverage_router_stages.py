from argparse import Namespace

import pytest

pytest.importorskip("torch")

from scripts.profile_coverage_router_stages import (
    CudaStageRecorder,
    aggregate_stage_values,
    compare_k50_k100,
    profiled_forward,
    summarize_stage_repeats,
    validate_args,
    validate_raw_stage_names,
)


def _args(**overrides):
    values = {
        "variants": ["k100", "k50"],
        "batch_size": 1,
        "num_people": 1,
        "image_size": 512,
        "warmup_iters": 0,
        "e2e_iters": 2,
        "profile_iters": 2,
        "repeats": 1,
        "output": "profile.json",
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_args_rejects_duplicate_variants():
    with pytest.raises(ValueError, match="duplicates"):
        validate_args(_args(variants=["k50", "k50"]))


def test_validate_args_requires_the_matched_k100_k50_pair():
    with pytest.raises(ValueError, match="both k100 and k50"):
        validate_args(_args(variants=["dense", "k50"]))


def test_validate_args_requires_a_json_output():
    with pytest.raises(ValueError, match="must end in .json"):
        validate_args(_args(output="profile.md"))


def test_aggregate_stage_values_sums_block_and_map_intervals():
    aggregated = aggregate_stage_values(
        {
            "token_prepare_rope": 1.0,
            "prefix.block_0": 2.0,
            "prefix.block_1": 3.0,
            "prefix.map_0": 0.5,
            "suffix.block_2": 4.0,
            "scatter_norm_2": 0.75,
            "decoder": 5.0,
        }
    )

    assert aggregated["prefix_blocks"] == pytest.approx(5.0)
    assert aggregated["prefix_maps"] == pytest.approx(0.5)
    assert aggregated["suffix_blocks"] == pytest.approx(4.0)
    assert aggregated["scatter_norm"] == pytest.approx(0.75)
    assert "prefix.block_0" not in aggregated


def test_missing_raw_stage_fails_instead_of_becoming_zero():
    class Blocks:
        def __len__(self):
            return 4

    class Backbone:
        out_indices = (0, 1, 2, 3)

        def __init__(self):
            self.model = type("Dino", (), {"blocks": Blocks()})()

    model = type(
        "Routed",
        (),
        {"route_after_block": 1, "backbone": Backbone()},
    )()

    with pytest.raises(RuntimeError, match="incomplete stage instrumentation"):
        validate_raw_stage_names({"router": 1.0}, model, "k50")


def test_profiled_forward_calls_the_separate_routed_diagnostic_path():
    class Model:
        def __init__(self):
            self.call = None

        def forward_profiled(self, model_input, recorder):
            self.call = (model_input, recorder)
            return "profiled-output"

    model = Model()
    model_input = {"images": "image", "bboxes": "boxes"}
    recorder = object()

    output = profiled_forward(model, model_input, "k50", recorder)

    assert output == "profiled-output"
    assert model.call == (model_input, recorder)


def test_cuda_stage_context_does_not_synchronize_each_stage(monkeypatch):
    events = []

    class Event:
        def __init__(self, **_):
            self.synchronize_calls = 0
            events.append(self)

        def record(self, _stream):
            return None

        def synchronize(self):
            self.synchronize_calls += 1

    monkeypatch.setattr("torch.cuda.Event", Event)
    monkeypatch.setattr("torch.cuda.current_stream", lambda _device: object())
    recorder = CudaStageRecorder("cuda")

    with recorder.stage("router"):
        pass

    assert len(events) == 2
    assert sum(event.synchronize_calls for event in events) == 0


def test_stage_summary_reports_repeat_medians_and_unattributed_time():
    repeats = [
        [
            {"dense_backbone": 4.0, "decoder": 3.0, "instrumented_full_cuda": 8.0},
            {"dense_backbone": 6.0, "decoder": 3.0, "instrumented_full_cuda": 10.0},
        ],
        [
            {"dense_backbone": 8.0, "decoder": 4.0, "instrumented_full_cuda": 13.0},
            {"dense_backbone": 10.0, "decoder": 4.0, "instrumented_full_cuda": 15.0},
        ],
    ]

    summary = summarize_stage_repeats(repeats, ("dense_backbone", "decoder"))

    assert summary["dense_backbone"]["repeat_medians_ms"] == [5.0, 9.0]
    assert summary["phase_sum"]["repeat_medians_ms"] == [8.0, 13.0]
    assert summary["unattributed"]["repeat_medians_ms"] == [1.0, 1.0]
    assert summary["dense_backbone"]["share_of_instrumented_full"] == pytest.approx(7.0 / 11.5)


def test_k50_k100_comparison_uses_matched_stage_medians():
    stage_names = (
        "token_prepare_rope",
        "prefix_blocks",
        "prefix_maps",
        "route_map",
        "router",
        "sparse_prepare",
        "suffix_blocks",
        "scatter_norm",
        "decoder",
        "phase_sum",
        "unattributed",
        "instrumented_full_cuda",
    )

    def item(name, e2e, stage_value):
        return {
            "variant": name,
            "e2e_uninstrumented": {"median_ms": e2e},
            "profiled": {
                stage: {"median_ms": stage_value}
                for stage in stage_names
            },
        }

    comparison = compare_k50_k100(
        [item("k100", 20.0, 2.0), item("k50", 19.0, 1.5)]
    )

    assert comparison["e2e_uninstrumented"]["k50_minus_k100_ms"] == -1.0
    assert comparison["stages"]["suffix_blocks"]["k50_minus_k100_fraction"] == pytest.approx(-0.25)
