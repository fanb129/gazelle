"""Synthetic T2-to-T3 integration tests; all artifacts stay in temporary dirs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from run_stag import run_comparison
from stag.cache import file_sha256, load_cache, write_json, write_prediction_cache


SCRIPT = Path(__file__).with_name("run_stag.py")


class STAGRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequences = []
        offset = 0
        peaks = [(20, 20), (20, 21), (30, 30), (8, 8), (8, 9)]
        labels = [1, 1, 0, 1, 0]
        for person, indices in (("007", [1, 2, 5]), ("020", [1, 2])):
            frames = []
            for frame_index in indices:
                row, column = peaks[offset]
                frames.append({
                    "path": f"images/video_a/clip_a/{frame_index:08d}.jpg",
                    "frame_index": frame_index, "person_id": person,
                    "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                    "gazex_norm": [column / 64] if labels[offset] else [-1.0],
                    "gazey_norm": [row / 64] if labels[offset] else [-1.0],
                    "inout": labels[offset],
                })
                offset += 1
            track = {"person_id": person, "track_id": f"test/video_a/clip_a/{person}",
                     "source_csv": f"annotations/test/video_a/clip_a/{person}.txt", "frames": frames}
            if not self.sequences:
                self.sequences.append({"path": "images/video_a/clip_a", "width": 640,
                                       "height": 480, "tracks": []})
            self.sequences[0]["tracks"].append(track)

        self.heatmaps = np.zeros((5, 64, 64), dtype=np.float32)
        for index, (row, column) in enumerate(peaks):
            self.heatmaps[index, row, column] = 0.9
        self.inout = np.array([0.9, 0.8, 0.2, 0.7, 0.1], dtype=np.float32)
        checkpoint = self.root / "synthetic_epoch_7.pt"
        checkpoint.write_bytes(b"synthetic checkpoint provenance fixture")
        track_json = self.root / "test_preprocessed_tracks.json"
        write_json(track_json, self.sequences)
        metadata = {
            "model": "synthetic_gazespot",
            "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "track_json": {"path": str(track_json), "sha256": file_sha256(track_json)},
            "protocol": {"modality": "RGB", "boxes": "current_frame_ground_truth"},
        }
        self.cache_dir = self.root / "t2_cache"
        cache = write_prediction_cache(
            self.sequences,
            [(self.heatmaps[:2], self.inout[:2]), (self.heatmaps[2:], self.inout[2:])],
            self.cache_dir, metadata,
        )
        self.cache_report = self.root / "t2_report.json"
        write_json(self.cache_report, {"task": "T2", "passed": True, "cache": cache,
                                      **{key: metadata[key] for key in ("model", "checkpoint", "track_json")}})
        self.source_hashes = {path.name: file_sha256(path) for path in self.cache_dir.iterdir()}
        self.output_dir = self.root / "t3_cache"
        self.report = self.root / "t3_report.json"
        self.provenance = self.root / "RESULT_PROVENANCE.md"
        self.provenance.write_text("# Synthetic provenance fixture\n")
        self.weights = [-2.0, 20.0, 2.0, 10.0]

    def command(self):
        return [sys.executable, str(SCRIPT), "--cache-dir", str(self.cache_dir),
                "--cache-report", str(self.cache_report), "--output-dir", str(self.output_dir),
                "--weights", *map(str, self.weights), "--report", str(self.report),
                "--provenance", str(self.provenance)]

    def test_cli_reports_paired_metrics_coverage_and_source_hashes(self):
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("T3 run completed", completed.stdout)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["task"], "T3")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual((report["cache"]["tracks"], report["cache"]["person_frames"],
                          report["cache"]["unique_image_frames"]), (2, 5, 3))
        self.assertEqual(report["source"]["report_sha256"], file_sha256(self.cache_report))
        self.assertEqual(report["source"]["index_sha256"], self.source_hashes["index.json"])
        self.assertEqual(report["cache"]["index_sha256"], file_sha256(self.output_dir / "index.json"))
        for method in ("single_frame", "stag"):
            metrics = report["metrics"][method]
            self.assertEqual((metrics["person_frames"], metrics["localization_frames"]), (5, 3))
            self.assertTrue(all(np.isfinite(metrics[key]) for key in ("auc", "l2", "inout_ap")))
        self.assertEqual(report["metrics"]["single_frame"]["l2"], 0)
        self.assertEqual(report["metrics"]["single_frame"]["inout_ap"], 1)
        self.assertEqual(report["metrics"]["stag"]["inout_ap"], 1)
        self.assertEqual(report["metrics"]["delta_stag_minus_single_frame"]["inout_ap"], 0)
        self.assertIn(str(self.output_dir), self.provenance.read_text())
        self.assertIn("single_frame", self.provenance.read_text())

    def test_output_diagnostics_resets_and_original_cache_unchanged(self):
        report = run_comparison(self.cache_dir, self.cache_report, self.output_dir, self.weights)
        index, arrays = load_cache(self.output_dir)
        source_index, _ = load_cache(self.cache_dir)
        self.assertEqual(index["task"], "T3")
        self.assertEqual(index["tracks"], source_index["tracks"])
        np.testing.assert_array_equal(arrays["inout"], self.inout)
        for name, artifact in report["cache"]["diagnostics"].items():
            path = self.output_dir / artifact["path"]
            self.assertEqual(file_sha256(path), artifact["sha256"])
            self.assertEqual(index["diagnostics"][name], artifact)
        reset = np.load(self.output_dir / "reset.npy", allow_pickle=False)
        alpha = np.load(self.output_dir / "alpha.npy", allow_pickle=False)
        signals = np.load(self.output_dir / "signals.npy", allow_pickle=False)
        np.testing.assert_array_equal(reset, [True, False, True, True, False])
        np.testing.assert_array_equal(alpha[reset], [1, 1, 1])
        np.testing.assert_array_equal(signals[reset], np.zeros((3, 3), dtype=np.float32))
        np.testing.assert_array_equal(arrays["heatmap"][reset], self.heatmaps[reset])
        self.assertEqual({path.name: file_sha256(path) for path in self.cache_dir.iterdir()},
                         self.source_hashes)

    def test_offline_comparison_imports_no_torch(self):
        source = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from run_stag import run_comparison; "
            "report = run_comparison(sys.argv[2], sys.argv[3], sys.argv[4], [-2,20,2,10]); "
            "assert report['passed']; assert 'torch' not in sys.modules; print('NO_TORCH')"
        )
        completed = subprocess.run([sys.executable, "-c", source, str(SCRIPT.parent.resolve()),
                                    str(self.cache_dir), str(self.cache_report), str(self.output_dir)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.rstrip().endswith("NO_TORCH"))

    def test_tampered_source_is_rejected_before_output_or_success_report(self):
        heatmaps = np.load(self.cache_dir / "heatmap.npy", mmap_mode="r+", allow_pickle=False)
        heatmaps[0, 0, 0] = 0.25
        heatmaps.flush()
        del heatmaps
        original_provenance = self.provenance.read_text()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("T2 cache SHA256 mismatch: heatmap.npy", completed.stderr)
        self.assertFalse(self.report.exists())
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.provenance.read_text(), original_provenance)


if __name__ == "__main__":
    unittest.main()
