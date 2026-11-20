"""Synthetic T2-to-T6 integration checks; no VAT data, DINO, or GPU required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from eval_temporal import run_evaluation
from run_ema_ablation import run_ablation
from run_stag import run_comparison
from stag.cache import file_sha256, load_cache, write_json, write_prediction_cache
from stag.case_selection import CATEGORY_ORDER
from stag.metrics import TemporalConfig
from visualize_temporal import heads_by_image, preview_cases


SCRIPT = Path(__file__).with_name("visualize_temporal.py")


class TemporalVisualizationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data_path = self.root / "dataset"
        sequence_path = "images/video_a/clip_a"
        image_dir = self.data_path / sequence_path
        image_dir.mkdir(parents=True)
        for position in range(6):
            image = Image.new("RGB", (160, 96), (30 + position * 12, 55, 75))
            draw = ImageDraw.Draw(image)
            for person in range(5):
                draw.rectangle((person * 28 + 4, 12, person * 28 + 20, 32), outline=(220, 220, 220))
            image.save(image_dir / f"{position + 1:08d}.png")

        tracks, predictions = [], []
        for person in range(5):
            old = (16 + person * 4, 16)
            new = (old[0], 26) if person < 2 else (36, 36) if person == 2 else old
            frames = []
            for position in range(6):
                target = old if position < 3 else new
                predicted = (target[0], target[1] + 2) if person == 3 and position == 1 else target
                predictions.append(predicted)
                frames.append({
                    "path": f"{sequence_path}/{position + 1:08d}.png",
                    "frame_index": position + 1, "person_id": str(person),
                    "bbox_norm": [person * 0.175 + 0.025, 0.125, person * 0.175 + 0.125, 0.35],
                    "gazex_norm": [target[1] / 64], "gazey_norm": [target[0] / 64], "inout": 1,
                })
            tracks.append({"person_id": str(person), "track_id": f"test/video_a/clip_a/{person}",
                           "source_csv": f"annotations/test/video_a/clip_a/{person}.txt", "frames": frames})
        sequences = [{"path": sequence_path, "width": 160, "height": 96, "tracks": tracks}]
        heatmaps = np.zeros((len(predictions), 64, 64), dtype=np.float32)
        for position, (row, column) in enumerate(predictions):
            heatmaps[position, row, column] = 0.9
        inout = np.full(len(predictions), 0.9, dtype=np.float32)
        checkpoint, track_json = self.root / "epoch_7.pt", self.root / "test_tracks.json"
        checkpoint.write_bytes(b"synthetic checkpoint metadata only")
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
        t3 = run_comparison(self.baseline_cache, self.baseline_report, self.stag_cache, [-2, 20, 2, 10])
        write_json(self.stag_report, {"task": "T3", **t3})
        self.temporal_report = self.root / "t4.json"
        t4 = run_evaluation(self.baseline_cache, self.baseline_report, self.stag_cache,
                            self.stag_report, self.root / "t4_details", TemporalConfig())
        write_json(self.temporal_report, {"task": "T4", **t4})
        self.t5_report = self.root / "t5.json"
        self.t5 = run_ablation(self.baseline_cache, self.baseline_report, self.stag_cache,
                               self.stag_report, self.temporal_report, self.root / "t5_cache")
        write_json(self.t5_report, {"task": "T5", **self.t5})
        self.output_dir, self.preview_dir = self.root / "t6_full", self.root / "t6_previews"
        self.report, self.summary = self.root / "t6.json", self.root / "t6.md"
        self.provenance = self.root / "RESULT_PROVENANCE.md"
        self.provenance.write_text("# Synthetic T6 provenance\n")

    def command(self):
        return [sys.executable, str(SCRIPT), "--data-path", str(self.data_path),
                "--ablation-report", str(self.t5_report), "--output-dir", str(self.output_dir),
                "--preview-dir", str(self.preview_dir), "--cases-per-group", "1", "--preview-cases", "1",
                "--context-before", "1", "--context-after", "3", "--min-heads", "4",
                "--snapshots", "2", "--dpi", "40", "--report", str(self.report),
                "--summary", str(self.summary), "--provenance", str(self.provenance)]

    def cache_hashes(self):
        return {str(path.relative_to(self.root)): file_sha256(path)
                for method in self.t5["methods"].values() for path in Path(method["cache"]["path"]).iterdir()}

    def test_cli_outputs_verified_figures_manifest_previews_and_keeps_predictions(self):
        original_hashes = self.cache_hashes()
        wrapper = ("import runpy,sys; from pathlib import Path; script=sys.argv.pop(1); sys.argv[0]=script; "
                   "sys.path.insert(0,str(Path(script).resolve().parent)); "
                   "runpy.run_path(script,run_name='__main__'); "
                   "assert 'torch' not in sys.modules; print('NO_TORCH')")
        completed = subprocess.run([sys.executable, "-c", wrapper, *self.command()[1:]],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("T6 visualization completed", completed.stdout)
        self.assertTrue(completed.stdout.rstrip().endswith("NO_TORCH"))
        report = json.loads(self.report.read_text())
        self.assertEqual(report["task"], "T6")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["source"]["report_sha256"], file_sha256(self.t5_report))
        expected_failures = self.t5["paired_response"]["stag_minus_single_frame"]["gaze_shift"]["reference_only_responded_events"]
        self.assertEqual(expected_failures, 2)
        self.assertEqual(report["candidate_counts"]["single_frame_only"], {"candidates": 2, "selected": 2})
        self.assertEqual(len(report["single_frame_only_failures"]), expected_failures)
        manifest_path = Path(report["manifest"]["path"])
        self.assertEqual(report["manifest"]["sha256"], file_sha256(manifest_path))
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(report["cases"]), len(manifest["selected"]))
        preview_cases_written = []
        for item in report["cases"]:
            case = item["case"]
            self.assertIn(case["anchor_global_index"], item["frame_global_indices"])
            if any(artifact["kind"].startswith("preview_") for artifact in item["files"]):
                preview_cases_written.append(case["case_id"])
            for artifact in item["files"]:
                path = Path(artifact["path"])
                self.assertEqual(artifact["sha256"], file_sha256(path))
                self.assertEqual(artifact["size_bytes"], path.stat().st_size)
                self.assertGreater(path.stat().st_size, 0)
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        self.assertGreater(min(image.size), 32)
                        image.verify()
        self.assertEqual(len(preview_cases_written), 1)
        self.assertEqual(preview_cases_written,
                         list(preview_cases(manifest["selected"], 1)))
        for path, digest in report["source_images_sha256"].items():
            self.assertEqual(digest, file_sha256(path))
        self.assertEqual(self.cache_hashes(), original_hashes)
        self.assertIn("single_frame_only", self.summary.read_text())
        self.assertIn("t6_previews/", self.summary.read_text())
        self.assertTrue((self.preview_dir / "index.md").exists())
        self.assertIn(str(manifest_path), self.provenance.read_text())

    def test_tampered_details_are_rejected_before_success_outputs(self):
        path = Path(self.t5["details"]["path"])
        path.write_text(path.read_text() + "\n")
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("T5 per-track/event details SHA256 mismatch", completed.stderr)
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.preview_dir.exists())
        self.assertFalse(self.report.exists())
        self.assertFalse(self.summary.exists())
        self.assertEqual(self.provenance.read_text(), "# Synthetic T6 provenance\n")

    def test_preview_quota_prioritizes_categories_and_heads_use_true_track_ids(self):
        selected = [{"category": category, "case_id": f"{category}_{position}"}
                    for category in CATEGORY_ORDER for position in range(2)]
        expected = {f"{category}_0" for category in CATEGORY_ORDER}
        self.assertEqual(preview_cases(selected, len(CATEGORY_ORDER)), expected)
        self.assertEqual(len(preview_cases(selected, 1)), 1)
        self.assertEqual(preview_cases(selected, 0), set())
        index, _ = load_cache(self.baseline_cache)
        heads = heads_by_image(index)
        self.assertTrue(all(len(people) == 5 for people in heads.values()))
        self.assertTrue(all(len({person["track_id"] for person in people}) == 5 for people in heads.values()))


if __name__ == "__main__":
    unittest.main()
