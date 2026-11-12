"""T1 regression checks with synthetic VAT files; no dataset or GPU required."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from preprocess_vat_tracks import make_track, preprocess_split, read_rows, sha256, summarize


SCRIPT = Path(__file__).with_name("preprocess_vat_tracks.py")


class TrackRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence_path = Path("images/video_a/clip_a")
        image_dir = self.root / self.sequence_path
        image_dir.mkdir(parents=True)
        for index in (1, 2, 4):
            Image.new("RGB", (100, 80)).save(image_dir / f"{index:08d}.jpg")
        for split in ("train", "test"):
            annotation_dir = self.root / "annotations" / split / "video_a/clip_a"
            annotation_dir.mkdir(parents=True)
            # Native .txt files contain the same CSV columns as .csv files.
            rows_by_person = {
                "007.txt": [
                    ["00000001.jpg", -10, -5, 120, 90, 50, 40],
                    ["00000002.jpg", 10, 20, 30, 40, 20, -1],
                    ["00000004.jpg", 60, 40, 80, 60, -1, -1],
                ],
                "020.csv": [
                    ["00000002.jpg", 20, 10, 40, 30, -1, 40],
                    ["00000004.jpg", 25, 10, 45, 30, -1, -1],
                ],
            }
            for filename, rows in rows_by_person.items():
                with (annotation_dir / filename).open("w", newline="") as stream:
                    csv.writer(stream).writerows(rows)
            legacy = [{
                "path": self.sequence_path.as_posix(), "width": 100, "height": 80,
                "frames": [
                    {"path": f"{self.sequence_path}/{index:08d}.jpg", "heads": [{}] * people}
                    for index, people in [(1, 1), (2, 2), (4, 2)]
                ],
            }]
            (self.root / f"{split}_preprocessed.json").write_text(json.dumps(legacy))

    def test_native_identity_order_boxes_and_labels(self):
        sequences, source = preprocess_split(self.root, "train")
        sequence = sequences[0]
        first, second = sequence["tracks"]
        self.assertEqual((sequence["width"], sequence["height"]), (100, 80))
        self.assertEqual((first["person_id"], second["person_id"]), ("007", "020"))
        self.assertEqual(first["track_id"], "train/video_a/clip_a/007")
        self.assertEqual(first["source_csv"], "annotations/train/video_a/clip_a/007.txt")
        self.assertEqual([frame["frame_index"] for frame in first["frames"]], [1, 2, 4])
        self.assertTrue(all(frame["person_id"] == "007" for frame in first["frames"]))
        self.assertEqual(first["frames"][0]["bbox_norm"], [0, 0, 1, 1])
        self.assertEqual(first["frames"][1]["bbox_norm"], [0.1, 0.25, 0.3, 0.5])
        self.assertEqual(first["frames"][0]["gazex_norm"], [0.5])
        self.assertEqual(first["frames"][0]["gazey_norm"], [0.5])
        self.assertEqual(first["frames"][1]["gazey_norm"], [0])
        self.assertEqual(second["frames"][0]["gazex_norm"], [0])
        self.assertEqual([frame["inout"] for frame in first["frames"]], [1, 1, 0])
        self.assertEqual(first["frames"][2]["gazex_norm"], [-0.01])
        self.assertEqual(first["frames"][2]["gazey_norm"], [-0.0125])
        self.assertEqual((source["person_csv_files"], source["csv_rows"]), (2, 5))

    def test_future_rows_do_not_change_past_frames(self):
        annotation = self.root / "annotations/train/video_a/clip_a/007.txt"
        rows = read_rows(annotation)
        full = make_track(rows, annotation, self.root, self.sequence_path, 100, 80, "train")
        prefix = make_track(rows[:2], annotation, self.root, self.sequence_path, 100, 80, "train")
        rows[-1][1:] = [0, 0, 1, 1, 90, 70]
        changed = make_track(rows, annotation, self.root, self.sequence_path, 100, 80, "train")
        self.assertEqual(full["frames"][:2], prefix["frames"])
        self.assertEqual(full["frames"][:2], changed["frames"][:2])

    def test_counts_distinguish_images_from_person_frames(self):
        sequences, source = preprocess_split(self.root, "train")
        result = summarize(sequences, source, self.root / "train_preprocessed.json")
        self.assertTrue(result["passed"])
        self.assertEqual(result["tracks"], 2)
        self.assertEqual(result["unique_image_frames"], 3)
        self.assertEqual(result["person_frames"], 5)
        self.assertEqual(result["in_frame_person_frames"], 3)
        self.assertEqual(result["out_of_frame_person_frames"], 2)
        self.assertEqual(result["frame_index_gaps"], 2)

    def test_equal_totals_with_wrong_frame_coverage_fail(self):
        sequences, source = preprocess_split(self.root, "train")
        legacy_path = self.root / "train_preprocessed.json"
        legacy = json.loads(legacy_path.read_text())
        legacy[0]["frames"][0]["heads"].append({})
        legacy[0]["frames"][1]["heads"].pop()
        legacy_path.write_text(json.dumps(legacy))
        result = summarize(sequences, source, legacy_path)
        self.assertTrue(result["checks"]["person_frames_match_legacy"])
        self.assertFalse(result["checks"]["per_frame_head_counts_match_legacy"])
        self.assertFalse(result["passed"])

    def test_nonchronological_csv_is_rejected(self):
        annotation = self.root / "annotations/train/video_a/clip_a/007.txt"
        rows = read_rows(annotation)
        with self.assertRaisesRegex(ValueError, "frame indices must increase"):
            make_track(list(reversed(rows)), annotation, self.root, self.sequence_path, 100, 80, "train")

    def test_cli_outputs_reports_hashes_and_failure_status(self):
        report_path = self.root / "results/t1_track_recovery.json"
        provenance_path = self.root / "results/RESULT_PROVENANCE.md"
        command = [
            sys.executable, str(SCRIPT), "--data-path", str(self.root),
            "--output-dir", str(self.root / "tracks"), "--legacy-dir", str(self.root),
            "--splits", "train", "test", "--report", str(report_path),
            "--provenance", str(provenance_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(report_path.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["splits"]), {"train", "test"})
        self.assertEqual(report["script_sha256"], sha256(SCRIPT))
        for split in ("train", "test"):
            output = self.root / "tracks" / f"{split}_preprocessed_tracks.json"
            self.assertEqual(report["splits"][split]["output"]["sha256"], sha256(output))
            self.assertEqual(len(json.loads(output.read_text())[0]["tracks"]), 2)
            self.assertIn(str(output), provenance_path.read_text())
        saved_provenance = provenance_path.read_text()
        legacy_path = self.root / "test_preprocessed.json"
        legacy = json.loads(legacy_path.read_text())
        legacy[0]["frames"][0]["heads"].pop()
        legacy_path.write_text(json.dumps(legacy))
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Track count verification failed", completed.stderr)
        self.assertFalse(json.loads(report_path.read_text())["passed"])
        self.assertEqual(provenance_path.read_text(), saved_provenance)


if __name__ == "__main__":
    unittest.main()
