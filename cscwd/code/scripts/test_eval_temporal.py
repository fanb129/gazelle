"""Synthetic T2/T3-to-T4 integration checks; no real VAT files or Torch needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from eval_temporal import run_evaluation
from run_stag import run_comparison
from stag.cache import file_sha256, write_json, write_prediction_cache
from stag.metrics import TemporalConfig


SCRIPT = Path(__file__).with_name("eval_temporal.py")


class TemporalEvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pair = self.make_pair("dynamic")
        self.details_dir = self.root / "t4_details"
        self.report = self.root / "t4_report.json"
        self.provenance = self.root / "RESULT_PROVENANCE.md"
        self.provenance.write_text("# Synthetic T4 provenance fixture\n")

    def make_pair(self, name, static=False):
        root = self.root / name
        root.mkdir()
        labels_by_person = ([1] * 6, [1] * 4) if static else (
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1])
        tracks, peaks, labels = [], [], []
        for person_number, person_labels in enumerate(labels_by_person):
            person = f"{person_number:03d}"
            frames = []
            for position, label in enumerate(person_labels):
                target = ((16, 16) if static or position < 3 else
                          (32, 32) if position < 9 else (40, 10))
                if person_number:
                    target = (8, 8)
                # Slight prediction motion in the first stable segment.
                predicted = (target[0], target[1] + int(position == 1 and not static))
                peaks.append(predicted)
                labels.append(label)
                row, column = target
                frames.append({
                    "path": f"images/video_a/clip_a/{position + 1:08d}.jpg",
                    "frame_index": position + 1, "person_id": person,
                    "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                    "gazex_norm": [column / 64] if label else [-1.0],
                    "gazey_norm": [row / 64] if label else [-1.0], "inout": label,
                })
            tracks.append({"person_id": person, "track_id": f"test/video_a/clip_a/{person}",
                           "source_csv": f"annotations/test/video_a/clip_a/{person}.txt", "frames": frames})
        sequences = [{"path": "images/video_a/clip_a", "width": 640, "height": 480, "tracks": tracks}]
        heatmaps = np.zeros((len(peaks), 64, 64), dtype=np.float32)
        for position, (row, column) in enumerate(peaks):
            heatmaps[position, row, column] = 0.9
        inout = np.asarray([0.9 if label else 0.1 for label in labels], dtype=np.float32)
        checkpoint = root / "synthetic_epoch_7.pt"
        checkpoint.write_bytes(b"synthetic checkpoint metadata only")
        track_json = root / "test_preprocessed_tracks.json"
        write_json(track_json, sequences)
        metadata = {
            "model": "synthetic_gazespot", "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "track_json": {"path": str(track_json), "sha256": file_sha256(track_json)},
            "protocol": {"modality": "RGB", "boxes": "current_frame_ground_truth"},
        }
        baseline_cache = root / "baseline_cache"
        baseline_report = root / "t2_report.json"
        cache = write_prediction_cache(sequences, [(heatmaps, inout)], baseline_cache, metadata)
        write_json(baseline_report, {"task": "T2", "passed": True, "cache": cache,
                                     **{key: metadata[key] for key in ("model", "checkpoint", "track_json")}})
        stag_cache = root / "stag_cache"
        stag_report = root / "t3_report.json"
        result = run_comparison(baseline_cache, baseline_report, stag_cache, [-2, 20, 2, 10])
        write_json(stag_report, {"task": "T3", **result})
        return {"baseline_cache": baseline_cache, "baseline_report": baseline_report,
                "stag_cache": stag_cache, "stag_report": stag_report, "t3": result,
                "person_frames": len(peaks), "unique_frames": len(labels_by_person[0])}

    def command(self):
        return [sys.executable, str(SCRIPT),
                "--single-frame-cache", str(self.pair["baseline_cache"]),
                "--single-frame-report", str(self.pair["baseline_report"]),
                "--stag-cache", str(self.pair["stag_cache"]), "--stag-report", str(self.pair["stag_report"]),
                "--details-dir", str(self.details_dir), "--shift-threshold", "0.15",
                "--response-radius", "0.10", "--stable-threshold", "0.02", "--hold-frames", "3",
                "--max-response-frames", "30", "--report", str(self.report),
                "--provenance", str(self.provenance)]

    def source_hashes(self):
        return {f"{kind}/{path.name}": file_sha256(path)
                for kind in ("baseline_cache", "stag_cache") for path in self.pair[kind].iterdir()}

    def test_cli_preserves_caches_and_reports_paired_metrics_details_and_provenance(self):
        originals = self.source_hashes()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("T4 evaluation completed", completed.stdout)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["task"], "T4")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["coverage"], {"tracks": 2, "person_frames": 17, "unique_image_frames": 12})
        self.assertEqual(report["standard_metrics"],
                         {name: self.pair["t3"]["metrics"][name] for name in ("single_frame", "stag")})
        self.assertEqual(report["sources"]["single_frame"]["report_sha256"], file_sha256(self.pair["baseline_report"]))
        self.assertEqual(report["sources"]["stag"]["report_sha256"], file_sha256(self.pair["stag_report"]))
        details_path = Path(report["details"]["path"])
        self.assertEqual(report["details"]["sha256"], file_sha256(details_path))
        details = json.loads(details_path.read_text())["methods"]
        event_keys = ("track_id", "type", "event_global_index", "event_frame_index", "window_frames")
        self.assertEqual([{key: event[key] for key in event_keys} for event in details["single_frame"]["events"]],
                         [{key: event[key] for key in event_keys} for event in details["stag"]["events"]])
        self.assertEqual(report["temporal_metrics"]["single_frame"]["jitter"]["ground_truth"],
                         report["temporal_metrics"]["stag"]["jitter"]["ground_truth"])
        self.assertEqual(report["temporal_metrics"]["single_frame"]["response"]["leave_frame"],
                         report["temporal_metrics"]["stag"]["response"]["leave_frame"])
        paired = report["paired_response"]["all"]
        self.assertEqual(paired["events"], 3)
        self.assertEqual(sum(paired[key] for key in ("both_responded_events",
                                                    "single_frame_only_responded_events",
                                                    "stag_only_responded_events",
                                                    "both_censored_events",
                                                    "insufficient_followup_events")), paired["events"])
        common_deltas = [stag["response_delay_frames"] - base["response_delay_frames"]
                         for base, stag in zip(details["single_frame"]["events"], details["stag"]["events"])
                         if base["status"] == stag["status"] == "responded"]
        self.assertEqual(paired["both_responded_events"], len(common_deltas))
        self.assertEqual(paired["common_responded_delay_delta_frames"]["mean"], float(np.mean(common_deltas)))
        self.assertEqual(self.source_hashes(), originals)
        self.assertIn(str(details_path), self.provenance.read_text())
        self.assertIn("准确且稳定占比", self.provenance.read_text())

    def test_no_events_roundtrip_as_json_null(self):
        pair = self.make_pair("static", static=True)
        report = run_evaluation(pair["baseline_cache"], pair["baseline_report"],
                                pair["stag_cache"], pair["stag_report"], self.details_dir, TemporalConfig())
        write_json(self.report, report)
        text = self.report.read_text()
        restored = json.loads(text)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertIn("null", text)
        for method in ("single_frame", "stag"):
            response = restored["temporal_metrics"][method]["response"]["all"]
            self.assertEqual(response["events"], 0)
            self.assertIsNone(response["response_rate"])
            self.assertIsNone(response["responded_delay_frames"]["mean"])
        self.assertIsNone(restored["paired_response"]["all"]["common_responded_delay_delta_frames"]["mean"])

    def test_tampered_source_fails_before_success_outputs(self):
        heatmaps = np.load(self.pair["baseline_cache"] / "heatmap.npy", mmap_mode="r+", allow_pickle=False)
        heatmaps[0, 0, 0] = 0.25
        heatmaps.flush()
        del heatmaps
        original_provenance = self.provenance.read_text()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Cache SHA256 mismatch", completed.stderr)
        self.assertFalse(self.report.exists())
        self.assertFalse(self.details_dir.exists())
        self.assertEqual(self.provenance.read_text(), original_provenance)

    def test_offline_evaluation_imports_no_torch(self):
        source = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from eval_temporal import run_evaluation; from stag.metrics import TemporalConfig; "
            "result = run_evaluation(*sys.argv[2:7], TemporalConfig()); "
            "assert result['passed']; assert 'torch' not in sys.modules; print('NO_TORCH')"
        )
        completed = subprocess.run([sys.executable, "-c", source, str(SCRIPT.parent.resolve()),
                                    str(self.pair["baseline_cache"]), str(self.pair["baseline_report"]),
                                    str(self.pair["stag_cache"]), str(self.pair["stag_report"]), str(self.details_dir)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.rstrip().endswith("NO_TORCH"))


if __name__ == "__main__":
    unittest.main()
