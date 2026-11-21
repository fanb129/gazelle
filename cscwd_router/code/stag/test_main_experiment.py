"""T4 CPU controls; timing/accuracy values below are unit-test fixtures only."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from cscwd_router.code.scripts.benchmark_router_dinov2 import build_variant, percentile, prepare_inputs, trace_forward
from cscwd_router.code.scripts.run_main_experiment import benchmark_command, cached_evaluation_matches, evaluation_command, parse_args, training_command, training_is_finalized
from cscwd_router.code.stag.checkpoint import clone_state
from cscwd_router.code.stag.main_results import BATCHES, ORDERS, render_comparison, summarize_results
from cscwd_router.code.stag.model_dinov2 import build_model
from cscwd_router.code.stag.test_training_dinov2 import tiny_backbone


def fixture_results():
    accuracy = {}
    for stage, auc in (("dense_official", 0.95), ("sparse_decoder", 0.949), ("matched_dense", 0.951)):
        accuracy[stage] = {"stage": stage, "split": "official_test", "data": {"annotation_sha256": "unit-test"},
                           "training_signature": "fixture", "checkpoint": {"sha256": stage}, "precision": "fp32", "batch_size": 8,
                           "metrics": {"auc": auc, "avg_l2": 0.1, "min_l2": 0.05, "images": 8, "persons": 8}}
    cells = []
    for batch in BATCHES:
        for order in ORDERS:
            dense_time = 12.0 if order == "forward" else 18.0
            cell = {"batch_size": batch, "order": order, "precision": "fp16",
                    "settings": {"autocast_cache_enabled": False, "inference_mode": True,
                                 "transfers_and_preprocessing_timed": False, "image_size": 448, "people_per_image": 1},
                    "environment": {"gpu_uuid": "unit-test"}, "sources": {"fixture": "same"},
                    "checkpoints": {stage: {"sha256": stage} for stage in ("dense_official", "sparse_decoder")},
                    "input": {"fingerprint": f"fixture-b{batch}"}, "results": {}}
            for variant, factor in (("dense", 1.0), ("k100", 1.1), ("k50", 0.8)):
                cell["results"][variant] = {"timing": {"latency_ms_median": dense_time * factor,
                                                       "throughput_images_per_second_median": batch * 1000 / (dense_time * factor),
                                                       "peak_allocated_mib": 1000.0}}
            cells.append(cell)
    return accuracy, cells


class MainResultsTests(unittest.TestCase):
    def test_order_ratios_are_aggregated_geometrically(self):
        accuracy, cells = fixture_results()
        result = summarize_results(accuracy, cells)
        self.assertAlmostEqual(result["efficiency"]["8"]["k50_speedup_vs_official_dense"], 1.25)
        self.assertAlmostEqual(result["efficiency"]["8"]["k50_speedup_vs_k100"], 1.375)
        self.assertTrue(result["gate"]["passed"])
        self.assertIn("matched_dense", render_comparison(result))

    def test_accuracy_and_efficiency_fail_independently(self):
        accuracy, cells = fixture_results()
        accuracy["sparse_decoder"]["metrics"]["auc"] = 0.94
        result = summarize_results(accuracy, cells)
        self.assertFalse(result["gate"]["accuracy_pass"])
        self.assertTrue(result["gate"]["efficiency_pass"])
        accuracy, cells = fixture_results()
        for cell in cells:
            cell["results"]["k50"]["timing"]["latency_ms_median"] = cell["results"]["dense"]["timing"]["latency_ms_median"]
        result = summarize_results(accuracy, cells)
        self.assertTrue(result["gate"]["accuracy_pass"])
        self.assertFalse(result["gate"]["efficiency_pass"])

    def test_inclusive_gate_boundaries(self):
        accuracy, cells = fixture_results()
        accuracy["sparse_decoder"]["metrics"]["auc"] = accuracy["dense_official"]["metrics"]["auc"] - 0.002
        for cell in cells:
            cell["results"]["k50"]["timing"]["latency_ms_median"] = cell["results"]["dense"]["timing"]["latency_ms_median"] / 1.10
        self.assertTrue(summarize_results(accuracy, cells)["gate"]["passed"])

    def test_incomplete_or_mixed_evidence_is_rejected(self):
        accuracy, cells = fixture_results()
        with self.assertRaisesRegex(ValueError, "both forward and reverse"):
            summarize_results(accuracy, cells[:-1])
        wrong = copy.deepcopy(cells)
        wrong[0]["input"]["fingerprint"] = "different-images"
        with self.assertRaisesRegex(ValueError, "identical RGB"):
            summarize_results(accuracy, wrong)
        wrong = copy.deepcopy(cells)
        wrong[0]["checkpoints"]["sparse_decoder"]["sha256"] = "different-checkpoint"
        with self.assertRaisesRegex(ValueError, "same final checkpoints"):
            summarize_results(accuracy, wrong)

    def test_percentile_interpolates_sorted_samples(self):
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 0.95), 3.85)


class MainCommandTests(unittest.TestCase):
    def setUp(self):
        self.args = parse_args(["--t3-summary", "/tmp/t3.json", "--data-path", "/tmp/data", "--decoder-checkpoint", "/tmp/official.pt",
                                "--run-dir", "/tmp/run", "--report-dir", "/tmp/reports", "--provenance", "/tmp/provenance.md"])

    def test_commands_pin_full_train_and_fixed_official_evaluation(self):
        command = training_command(self.args)
        self.assertEqual(command[command.index("--protocol") + 1], "full_train")
        self.assertEqual(command[command.index("--val-fraction") + 1], "0")
        self.assertNotIn("--no-augment", command)
        self.assertNotIn("--resume", command)
        self.assertIn("--resume", training_command(self.args, resume=True))
        command = evaluation_command(self.args, "sparse_decoder", Path("/tmp/result.json"))
        self.assertEqual(command[command.index("--split") + 1], "official_test")
        self.assertIn("/tmp/run/sparse_decoder/final.pt", command)
        for batch in BATCHES:
            for order in ORDERS:
                command = benchmark_command(self.args, batch, order, Path("/tmp/cell.json"))
                self.assertEqual(command[command.index("--precision") + 1], "fp16")
                self.assertEqual(command[command.index("--order") + 1], order)

    def test_cached_test_requires_the_same_checkpoint_annotation_and_batch(self):
        result = {"stage": "sparse_decoder", "split": "official_test", "checkpoint": {"sha256": "a"},
                  "training_signature": "b", "data": {"annotation_sha256": "c"}, "batch_size": 8}
        request = {"stage": "sparse_decoder", "checkpoint_sha": "a", "training_signature": "b", "annotation_sha": "c", "batch_size": 8}
        self.assertTrue(cached_evaluation_matches(result, **request))
        self.assertFalse(cached_evaluation_matches(result, **{**request, "checkpoint_sha": "changed"}))
        self.assertFalse(cached_evaluation_matches(result, **{**request, "batch_size": 4}))

    def test_training_resume_finalizes_artifact_hashes_before_starting_test(self):
        self.assertFalse(training_is_finalized({"completed": True, "passed": True}))
        self.assertTrue(training_is_finalized({"completed": True, "passed": True, "artifacts": [{"sha256": "fixture"}]}))


class BenchmarkInputAndGraphTests(unittest.TestCase):
    def test_inputs_use_rgb_and_true_single_head_train_boxes_without_gaze_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(8):
                Image.new("RGBA", (20, 30), color=(index, 10, 20, 255)).save(root / f"{index}.png")
                records.append({"path": f"{index}.png", "heads": [{"inout": 1, "bbox_norm": [0.1, 0.2, 0.3, 0.4]}]})
            (root / "train_preprocessed.json").write_text(json.dumps(records))

            def transform(image):
                self.assertEqual(image.mode, "RGB")
                return torch.zeros(3, 448, 448)

            small, small_meta = prepare_inputs(root, transform, 1, 3106)
            large, large_meta = prepare_inputs(root, transform, 8, 3106)
            self.assertEqual(tuple(large["images"].shape), (8, 3, 448, 448))
            self.assertEqual(small_meta["samples"], large_meta["samples"][:1])
            self.assertEqual(large["bboxes"], [[[0.1, 0.2, 0.3, 0.4]]] * 8)
            self.assertTrue(torch.equal(small["images"][0], large["images"][0]))
            _, repeated = prepare_inputs(root, transform, 8, 3106)
            self.assertEqual(large_meta["fingerprint"], repeated["fingerprint"])

    def test_dense_k100_and_k50_have_the_expected_actual_token_graph(self):
        torch.manual_seed(3106)
        torch.set_num_threads(1)
        base = build_model({"image_size": 448, "decoder_dim": 32, "decoder_layers": 1,
                            "router_hidden_dim": 16, "escape_tokens": 0}, backbone=tiny_backbone())
        payload = {"model_config": base.config, "decoder": clone_state(base.decoder), "router": clone_state(base.encoder.router),
                   "stage": "sparse_decoder", "keep_ratio": 0.5}
        inputs = {"images": torch.randn(1, 3, 448, 448), "bboxes": [[[0.1, 0.1, 0.25, 0.3]]]}
        traces = {}
        outputs = {}
        for variant in ("dense", "k100", "k50"):
            model = build_variant(payload, variant, backbone=base.encoder.backbone)
            self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
            traces[variant] = trace_forward(model, inputs, variant, "fp32")
            with torch.inference_mode():
                outputs[variant] = torch.cat(model(inputs)["heatmap"])
        self.assertEqual([traces[name]["kept_patches"] for name in ("dense", "k100", "k50")], [1024, 1024, 512])
        self.assertEqual([row["tokens"] for row in traces["k50"]["block_inputs"]], [1025] * 6 + [513] * 6)
        self.assertTrue(torch.equal(outputs["dense"], outputs["k100"]))


if __name__ == "__main__":
    unittest.main()
