"""R1 preparation integration checks with synthetic frozen predictions and RGB."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from prepare_development import run_preparation
from stag.cache import file_sha256, load_cache, write_json, write_prediction_cache
from stag.metrics import TemporalConfig, evaluate_predictions, evaluate_temporal, temporal_protocol
from stag.state_metrics import evaluate_state_temporal, state_temporal_protocol


SCRIPT = Path(__file__).with_name("prepare_development.py")


class DevelopmentPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = self.make_fixture(self.root / "fixture")
        self.output_dir = self.root / "preparation"
        self.preview_dir = self.root / "previews"
        self.report_path = self.root / "r1_report.json"
        self.summary_path = self.root / "r1_report.md"
        self.provenance_path = self.root / "RESULT_PROVENANCE.md"
        self.provenance_path.write_text("# Synthetic R1 provenance\n")

    def make_fixture(self, root, all_outside=False):
        root.mkdir()
        data_path = root / "data"
        sequences, peaks, scores = [], [], []
        for video_number in range(8):
            video, clip = f"video_{video_number}", "clip_a"
            sequence_path = f"images/{video}/{clip}"
            image_dir = data_path / sequence_path
            image_dir.mkdir(parents=True)
            for position in range(12):
                Image.new("RGB", (160, 120), (40 + video_number * 10, 70, 100)).save(
                    image_dir / f"{position + 1:08d}.jpg")
            tracks = []
            for person_number in range(2):
                person = f"p{person_number:02d}"
                frames = []
                for position in range(12):
                    inside = int(not all_outside and (position < 6 or position >= 9))
                    target = (16, 16) if position < 3 else (44, 44) if position < 9 else (16, 32)
                    prediction = (54, 6) if position == 1 else target
                    row, column = target
                    frames.append({
                        "path": f"{sequence_path}/{position + 1:08d}.jpg",
                        "frame_index": position + 1, "person_id": person,
                        "bbox_norm": [0.1 + person_number * 0.4, 0.1,
                                      0.3 + person_number * 0.4, 0.4],
                        "gazex_norm": [column / 64] if inside else [-1.0],
                        "gazey_norm": [row / 64] if inside else [-1.0],
                        "inout": inside,
                    })
                    peaks.append(prediction)
                    scores.append(0.8 if position == 7 else 0.2 if position == 1 else 0.9 if inside else 0.1)
                tracks.append({"person_id": person, "track_id": f"train/{video}/{clip}/{person}",
                               "source_csv": f"annotations/train/{video}/{clip}/{person}.txt",
                               "frames": frames})
            sequences.append({"path": sequence_path, "width": 160, "height": 120, "tracks": tracks})

        track_json = root / "train_preprocessed_tracks.json"
        write_json(track_json, sequences)
        checkpoint, backbone = root / "synthetic_epoch_7.pt", root / "synthetic_backbone.pth"
        checkpoint.write_bytes(b"synthetic checkpoint identity, not a model")
        backbone.write_bytes(b"synthetic backbone identity, not a model")
        metadata = {
            "model": "gazelle_dinov3_vitb16_inout",
            "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "backbone_checkpoint": {"path": str(backbone), "sha256": file_sha256(backbone)},
            "track_json": {"path": str(track_json), "sha256": file_sha256(track_json)},
            "protocol": {"modality": "RGB", "boxes": "T1_current_frame_raw_ground_truth",
                         "spatial_prior": "ggsf", "fusion": "sasa", "aux": False,
                         "precision": "float32; no_autocast; tf32_disabled",
                         "temporal_context": "none", "labels_used_in_forward": False,
                         "all_parameters_frozen": True, "training": False},
        }
        heatmaps = np.zeros((len(peaks), 64, 64), dtype=np.float32)
        for index, (row, column) in enumerate(peaks):
            heatmaps[index, row, column] = 0.9
        inout = np.asarray(scores, dtype=np.float32)
        cache_dir = root / "prediction_cache"
        cache = write_prediction_cache(sequences, [(heatmaps, inout)], cache_dir, metadata)
        cache_report, recovery_report, reference_report = (
            root / "cache_report.json", root / "recovery_report.json", root / "reference_report.json")
        run = {"task": "T2", "passed": True, "cache": cache, **metadata}
        write_json(cache_report, run)
        write_json(reference_report, copy.deepcopy(run))
        write_json(recovery_report, {"task": "T1", "passed": True, "splits": {
            "train": {"passed": True,
                      **{key: cache[key] for key in ("tracks", "person_frames", "unique_image_frames")},
                      "output": {"path": str(track_json), "sha256": file_sha256(track_json)}}}})
        return {"data_path": data_path, "cache_dir": cache_dir, "cache_report": cache_report,
                "recovery_report": recovery_report, "reference_report": reference_report,
                "coverage": {key: cache[key] for key in ("tracks", "person_frames", "unique_image_frames")}}

    def prepare(self, **kwargs):
        return run_preparation(
            self.fixture["data_path"], self.fixture["cache_dir"], self.fixture["cache_report"],
            self.fixture["recovery_report"], self.fixture["reference_report"],
            self.output_dir, self.preview_dir, validation_fraction=0.25, seed=97,
            cases_per_category=2, preview_count=4, context_before=2, context_after=3,
            snapshots=3, **kwargs)

    def command(self):
        return [sys.executable, str(SCRIPT),
                "--data-path", str(self.fixture["data_path"]),
                "--cache-dir", str(self.fixture["cache_dir"]),
                "--cache-report", str(self.fixture["cache_report"]),
                "--recovery-report", str(self.fixture["recovery_report"]),
                "--reference-report", str(self.fixture["reference_report"]),
                "--output-dir", str(self.output_dir), "--preview-dir", str(self.preview_dir),
                "--validation-fraction", "0.25", "--seed", "97", "--cases-per-category", "2",
                "--preview-cases", "4", "--context-before", "2", "--context-after", "3",
                "--snapshots", "3", "--report", str(self.report_path),
                "--summary", str(self.summary_path), "--provenance", str(self.provenance_path)]

    def source_hashes(self):
        return {path.name: file_sha256(path) for path in self.fixture["cache_dir"].iterdir()}

    def test_grouped_diagnostics_are_train_only_and_do_not_claim_manual_review(self):
        originals = self.source_hashes()
        result = self.prepare()
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["stage_status"], "awaiting_local_review")
        self.assertEqual(result["coverage"], self.fixture["coverage"])
        manifest = json.loads(Path(result["artifacts"]["split_manifest"]["path"]).read_text())
        train = manifest["partitions"]["train"]
        validation = manifest["partitions"]["validation"]
        self.assertFalse(set(train["group_ids"]) & set(validation["group_ids"]))
        self.assertFalse(set(train["track_ids"]) & set(validation["track_ids"]))
        self.assertEqual(set(result["split"]["partitions"]), {"train", "validation"})
        self.assertTrue(result["diagnosis"]["selected"])
        for case in result["diagnosis"]["selected"]:
            self.assertIn(case["track_id"], train["track_ids"])
            self.assertNotIn(case["track_id"], validation["track_ids"])
            self.assertEqual(case["review_status"], "unreviewed")
            self.assertEqual(case["labels"], [])
            self.assertEqual(case["notes"], "")
        self.assertEqual(self.source_hashes(), originals)
        self.assertFalse(self.report_path.exists())

    def test_partition_metrics_preserve_global_slices_and_existing_temporal_protocol(self):
        result = self.prepare()
        manifest = json.loads(Path(result["artifacts"]["split_manifest"]["path"]).read_text())
        index, arrays = load_cache(self.fixture["cache_dir"])
        for name in ("train", "validation"):
            selected_ids = set(manifest["partitions"][name]["track_ids"])
            tracks = [track for track in index["tracks"] if track["track_id"] in selected_ids]
            self.assertEqual(result["baseline"][name]["standard"],
                             evaluate_predictions(tracks, arrays["heatmap"], arrays["inout"]))
            self.assertEqual(result["baseline"][name]["temporal"],
                             evaluate_temporal(tracks, arrays["argmax_xy"], arrays["inout"], TemporalConfig())["summary"])
            self.assertEqual(result["baseline"][name]["state"],
                             evaluate_state_temporal(tracks, arrays["inout"], TemporalConfig())["summary"])
        self.assertEqual(result["selection_plan"]["joint_temporal_protocol"], temporal_protocol(TemporalConfig()))
        self.assertEqual(result["selection_plan"]["state_temporal_protocol"], state_temporal_protocol(TemporalConfig()))
        json.dumps(result, allow_nan=False)

    def test_cli_publishes_hashed_artifacts_and_pngs_without_changing_sources(self):
        originals = self.source_hashes()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report_text = self.report_path.read_text()
        self.assertNotIn("NaN", report_text)
        self.assertNotIn("Infinity", report_text)
        report = json.loads(report_text)
        self.assertEqual(report["task"], "R1")
        self.assertEqual(report["stage_status"], "awaiting_local_review")
        self.assertTrue(report["passed"])
        for artifact in report["artifacts"].values():
            path = Path(artifact["path"])
            self.assertEqual(artifact["sha256"], file_sha256(path))
            self.assertIn(str(path), self.provenance_path.read_text())
        self.assertTrue(report["diagnosis"]["preview_files"])
        for artifact in report["diagnosis"]["preview_files"]:
            path = Path(artifact["path"])
            self.assertEqual(artifact["sha256"], file_sha256(path))
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
        self.assertTrue(self.summary_path.read_text())
        self.assertEqual(self.source_hashes(), originals)

    def test_tampered_cache_cannot_publish_a_success_report_or_provenance(self):
        values = np.load(self.fixture["cache_dir"] / "inout.npy", mmap_mode="r+", allow_pickle=False)
        values[0] = 0.25
        values.flush()
        del values
        provenance = self.provenance_path.read_text()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SHA256", completed.stderr)
        self.assertFalse(self.report_path.exists())
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.provenance_path.read_text(), provenance)

    def test_reference_checkpoint_identity_must_match_before_artifacts_are_written(self):
        reference = json.loads(self.fixture["reference_report"].read_text())
        reference["checkpoint"]["sha256"] = "0" * 64
        write_json(self.fixture["reference_report"], reference)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output_dir.exists())

    def test_all_outside_development_partitions_remain_json_serializable(self):
        self.fixture = self.make_fixture(self.root / "outside_fixture", all_outside=True)
        result = self.prepare()
        self.assertTrue(result["passed"])
        for name in ("train", "validation"):
            self.assertIsNone(result["baseline"][name]["standard"]["auc"])
            self.assertIsNone(result["baseline"][name]["standard"]["l2"])
            self.assertEqual(result["baseline"][name]["standard"]["localization_frames"], 0)
            self.assertEqual(result["baseline"][name]["state"]["response"]["all"]["events"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
