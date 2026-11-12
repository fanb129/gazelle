"""Synthetic full-cache T5 checks; all results live in temporary directories."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from eval_temporal import run_evaluation
from run_ema_ablation import run_ablation
from run_stag import run_comparison
from stag.cache import file_sha256, load_cache, write_json, write_prediction_cache
from stag.metrics import TemporalConfig


SCRIPT = Path(__file__).with_name("run_ema_ablation.py")
METHODS = ["single_frame", "stag", "ema_bias_matched", "ema_alpha_0p25", "ema_alpha_0p5", "ema_alpha_0p75"]


class EMAAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        labels_by_person = ([1] * 6 + [0] * 3 + [1] * 3, [1] * 5)
        tracks, peaks, labels = [], [], []
        for person_number, person_labels in enumerate(labels_by_person):
            person, frames = str(person_number), []
            for position, label in enumerate(person_labels):
                point = (16, 16) if position < 3 else (32, 32) if position < 9 else (40, 10)
                if person_number:
                    point = (8, 8)
                peaks.append((point[0], point[1] + int(position == 1)))
                labels.append(label)
                frame_index = position + 1 + int(not person_number and position >= 6)
                frames.append({
                    "path": f"images/video_a/clip_a/{frame_index:08d}.jpg",
                    "frame_index": frame_index, "person_id": person,
                    "bbox_norm": [0.1, 0.1, 0.3, 0.3], "inout": label,
                    "gazex_norm": [point[1] / 64] if label else [-1.0],
                    "gazey_norm": [point[0] / 64] if label else [-1.0],
                })
            tracks.append({"person_id": person, "track_id": f"test/video_a/clip_a/{person}",
                           "source_csv": f"annotations/test/video_a/clip_a/{person}.txt", "frames": frames})
        sequences = [{"path": "images/video_a/clip_a", "width": 640, "height": 480, "tracks": tracks}]
        heatmaps = np.zeros((len(peaks), 64, 64), dtype=np.float32)
        for index, (row, column) in enumerate(peaks):
            heatmaps[index, row, column] = 0.9
        inout = np.array([0.9 if label else 0.1 for label in labels], dtype=np.float32)
        checkpoint, track_json = self.root / "epoch_7.pt", self.root / "test_tracks.json"
        checkpoint.write_bytes(b"synthetic checkpoint provenance only")
        write_json(track_json, sequences)
        metadata = {"model": "synthetic_gazespot",
                    "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
                    "track_json": {"path": str(track_json), "sha256": file_sha256(track_json)},
                    "protocol": {"modality": "RGB", "boxes": "current_frame_ground_truth"}}
        self.baseline_cache, self.baseline_report = self.root / "t2_cache", self.root / "t2.json"
        cache = write_prediction_cache(sequences, [(heatmaps, inout)], self.baseline_cache, metadata)
        write_json(self.baseline_report, {"task": "T2", "passed": True, "cache": cache,
                                         **{key: metadata[key] for key in ("model", "checkpoint", "track_json")}})
        self.stag_cache, self.stag_report = self.root / "t3_cache", self.root / "t3.json"
        result = run_comparison(self.baseline_cache, self.baseline_report, self.stag_cache,
                                [-2, 20, 2, 10])
        write_json(self.stag_report, {"task": "T3", **result})
        self.temporal_report = self.root / "t4.json"
        self.t4 = run_evaluation(self.baseline_cache, self.baseline_report, self.stag_cache,
                                 self.stag_report, self.root / "t4_details", TemporalConfig())
        write_json(self.temporal_report, {"task": "T4", **self.t4})
        self.output_dir, self.report = self.root / "t5_cache", self.root / "t5.json"
        self.summary, self.provenance = self.root / "t5.md", self.root / "RESULT_PROVENANCE.md"
        self.provenance.write_text("# Synthetic T5 provenance\n")

    def command(self):
        return [sys.executable, str(SCRIPT), "--single-frame-cache", str(self.baseline_cache),
                "--single-frame-report", str(self.baseline_report), "--stag-cache", str(self.stag_cache),
                "--stag-report", str(self.stag_report), "--temporal-report", str(self.temporal_report),
                "--output-dir", str(self.output_dir), "--ema-alphas", "0.25", "0.5", "0.75",
                "--report", str(self.report), "--summary", str(self.summary),
                "--provenance", str(self.provenance)]

    def source_hashes(self):
        return {str(path.relative_to(self.root)): file_sha256(path)
                for folder in (self.baseline_cache, self.stag_cache) for path in folder.iterdir()}

    def test_cli_keeps_all_six_methods_and_writes_json_markdown_provenance(self):
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("T5 ablation completed: 6 methods", completed.stdout)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["task"], "T5")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(list(report["methods"]), METHODS)
        self.assertEqual([variant["name"] for variant in report["variants"]], METHODS[2:])
        self.assertEqual([variant["alpha"] for variant in report["variants"]][1:], [0.25, 0.5, 0.75])
        self.assertAlmostEqual(report["variants"][0]["alpha"], 1 / (1 + np.exp(2)))
        self.assertEqual(report["protocol"]["temporal"], self.t4["temporal_protocol"])
        for name in ("single_frame", "stag"):
            self.assertEqual(report["methods"][name]["standard"], self.t4["standard_metrics"][name])
            self.assertEqual(report["methods"][name]["temporal"], self.t4["temporal_metrics"][name])
        for name in METHODS:
            self.assertIn(f"| {name} |", self.summary.read_text())
            self.assertIn(f"- {name}：", self.provenance.read_text())
        self.assertEqual(report["details"]["sha256"], file_sha256(report["details"]["path"]))
        self.assertIn(str(self.output_dir), self.provenance.read_text())

    def test_ema_caches_share_records_scores_resets_and_direct_paired_events(self):
        original_hashes = self.source_hashes()
        result = run_ablation(self.baseline_cache, self.baseline_report, self.stag_cache,
                              self.stag_report, self.temporal_report, self.output_dir)
        source_index, source_arrays = load_cache(self.baseline_cache)
        details = json.loads(Path(result["details"]["path"]).read_text())["methods"]
        for variant in result["variants"]:
            name, alpha = variant["name"], variant["alpha"]
            cache = result["methods"][name]["cache"]
            index, arrays = load_cache(cache["path"])
            self.assertEqual(index["task"], "T5")
            self.assertEqual(index["tracks"], source_index["tracks"])
            self.assertEqual(cache["index_sha256"], file_sha256(Path(cache["path"]) / "index.json"))
            np.testing.assert_array_equal(arrays["inout"], source_arrays["inout"])
            self.assertEqual(result["methods"][name]["standard"]["inout_ap"], result["methods"]["single_frame"]["standard"]["inout_ap"])
            for artifact in {**cache["files"], **cache["diagnostics"]}.values():
                self.assertEqual(artifact["sha256"], file_sha256(Path(cache["path"]) / artifact["path"]))
            reset = np.load(Path(cache["path"]) / "reset.npy", allow_pickle=False)
            coefficients = np.load(Path(cache["path"]) / "alpha.npy", allow_pickle=False)
            np.testing.assert_array_equal(np.flatnonzero(reset), [0, 6, 12])
            np.testing.assert_array_equal(coefficients[reset], 1)
            np.testing.assert_array_equal(coefficients[~reset], np.float32(alpha))
            np.testing.assert_array_equal(arrays["heatmap"][reset], source_arrays["heatmap"][reset])
            for reference in ("single_frame", "stag"):
                base_events, candidate_events = details[reference]["events"], details[name]["events"]
                response = result["paired_response"][f"{name}_minus_{reference}"]["all"]
                pairs = list(zip(base_events, candidate_events))
                self.assertEqual([(event["type"], event["event_global_index"], event["window_frames"])
                                  for event in base_events],
                                 [(event["type"], event["event_global_index"], event["window_frames"])
                                  for event in candidate_events])
                differences = [candidate["response_delay_frames"] - base["response_delay_frames"]
                               for base, candidate in pairs if base["status"] == candidate["status"] == "responded"]
                self.assertEqual(response["both_responded_events"], len(differences))
                self.assertEqual(response["reference_only_responded_events"],
                                 sum(base["status"] == "responded" and candidate["status"] != "responded"
                                     for base, candidate in pairs))
                self.assertEqual(response["candidate_only_responded_events"],
                                 sum(candidate["status"] == "responded" and base["status"] != "responded"
                                     for base, candidate in pairs))
                self.assertEqual(response["common_responded_delay_delta_frames"]["mean"],
                                 float(np.mean(differences)) if differences else None)
        self.assertEqual(self.source_hashes(), original_hashes)

    def test_source_report_and_npy_tampering_fail_before_success_outputs(self):
        original_t4 = self.temporal_report.read_text()
        corrupt = json.loads(original_t4)
        corrupt["sources"]["stag"]["report_sha256"] = "incorrect_source_hash"
        write_json(self.temporal_report, corrupt)
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("T5 source verification failed", completed.stderr)
        self.assertFalse(self.output_dir.exists())
        self.temporal_report.write_text(original_t4)
        heatmap = np.load(self.baseline_cache / "heatmap.npy", mmap_mode="r+", allow_pickle=False)
        heatmap[0, 0, 0] = 0.25
        heatmap.flush()
        del heatmap
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Cache SHA256 mismatch", completed.stderr)
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.report.exists())
        self.assertFalse(self.summary.exists())
        self.assertEqual(self.provenance.read_text(), "# Synthetic T5 provenance\n")

    def test_full_offline_ablation_imports_no_torch(self):
        source = (
            "import sys; sys.path.insert(0, sys.argv[1]); from run_ema_ablation import run_ablation; "
            "result = run_ablation(*sys.argv[2:8]); "
            "assert result['passed']; assert len(result['methods']) == 6; "
            "assert 'torch' not in sys.modules; print('NO_TORCH')"
        )
        completed = subprocess.run([sys.executable, "-c", source, str(SCRIPT.parent.resolve()),
                                    str(self.baseline_cache), str(self.baseline_report), str(self.stag_cache),
                                    str(self.stag_report), str(self.temporal_report), str(self.output_dir)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.rstrip().endswith("NO_TORCH"))


if __name__ == "__main__":
    unittest.main()
