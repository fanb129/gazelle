"""Synthetic RGB rendering checks; figures are observations of cached predictions."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stag.cache import argmax_points, file_sha256
from stag.metrics import TemporalConfig
from stag.plotting import METHODS, _case_records, _timeline_figure, render_case, snapshot_indices


def make_fixture(root):
    root = Path(root)
    frames = []
    heads = {}
    for position, frame_index in enumerate(range(101, 108)):
        path = f"images/video_a/clip_a/{frame_index:08d}.png"
        image_path = root / path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((90, 160, 3), dtype=np.uint8)
        image[..., 0] = np.arange(160, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(90, dtype=np.uint8)[:, None] + 55
        image[..., 2] = 100 + position * 12
        Image.fromarray(image).save(image_path)
        inout = int(position != 6)
        frames.append({"path": path, "frame_index": frame_index, "person_id": "007",
                       "bbox_norm": [0.10, 0.15, 0.25, 0.45], "inout": inout,
                       "gazex_norm": [0.45 + position / 100 if position < 3 else 0.8] if inout else [-1],
                       "gazey_norm": [0.55] if inout else [-1]})
        heads[path] = [{"track_id": "test/video_a/clip_a/007", "bbox_norm": frames[-1]["bbox_norm"]},
                       {"track_id": "test/video_a/clip_a/020", "bbox_norm": [0.65, 0.2, 0.85, 0.5]}]
    track = {"track_id": "test/video_a/clip_a/007", "person_id": "007", "start": 2,
             "stop": 9, "frames": frames, "width": 160, "height": 90}
    arrays = {}
    for method in METHODS:
        maps = np.zeros((9, 64, 64), dtype=np.float32)
        for index in range(2, 9):
            position = index - 2
            column = 28 + position if position < 3 else 51
            if position >= 3 and method in ("stag", "ema_alpha_0p25"):
                column = 28
            if position == 3 and method == "ema_alpha_0p5":
                column = 28
            maps[index, 35, column] = 0.75
        arrays[method] = {"heatmap": maps, "argmax_xy": argmax_points(maps),
                          "inout": np.array([0.8] * 9, dtype=np.float32)}
    diagnostics = {"alpha": np.linspace(0.1, 0.9, 9, dtype=np.float32),
                   "signals": np.column_stack((np.linspace(0, 0.1, 9), np.zeros(9), np.linspace(0.1, 0.3, 9))).astype(np.float32),
                   "reset": np.array([False, False, True, False, False, False, False, False, False])}
    case = {"case_id": "single_frame_only/test/video_a/clip_a/007/104", "category": "single_frame_only",
            "track_id": track["track_id"], "anchor_global_index": 5, "anchor_frame_index": 104,
            "context_start": 2, "context_stop": 9,
            "event": {method: {"track_id": track["track_id"], "event_global_index": 5,
                               "event_frame_index": 104, "window_frames": 3,
                               "status": "responded" if method == "single_frame" else "censored",
                               "response_delay_frames": 0 if method == "single_frame" else None}
                      for method in METHODS}}
    return case, track, arrays, diagnostics, heads


class CasePlottingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = make_fixture(self.root)

    def test_snapshot_sampling_preserves_anchor_and_original_global_indices(self):
        selected = snapshot_indices(10, 41, 22, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue({21, 22, 23}.issubset(selected))
        self.assertEqual((selected[0], selected[-1]), (10, 40))
        self.assertEqual(snapshot_indices(10, 11, 10, 5), [10])
        self.assertEqual(snapshot_indices(10, 41, 22, 1), [22])

    def test_render_outputs_hashes_raw_records_and_does_not_change_arrays(self):
        case, track, arrays, diagnostics, heads = self.fixture
        original_arrays = copy.deepcopy(arrays)
        original_diagnostics = copy.deepcopy(diagnostics)
        result = render_case(case, track, arrays, diagnostics, self.root, heads,
                              self.root / "full", TemporalConfig(), preview_dir=self.root / "previews", dpi=90)
        self.assertEqual({artifact["kind"] for artifact in result["files"]},
                         {"data", "full_frames", "full_timeline", "preview_frames", "preview_timeline"})
        for artifact in result["files"]:
            path = Path(artifact["path"])
            self.assertTrue(path.is_absolute())
            self.assertEqual(artifact["sha256"], file_sha256(path))
            if path.suffix == ".png":
                with Image.open(path) as image:
                    self.assertGreater(min(image.size), 500)
                    self.assertGreater(np.asarray(image).std(), 5)
        data = json.loads(Path(result["data_path"]).read_text())
        self.assertEqual([record["frame_index"] for record in data["frames"]], list(range(101, 108)))
        self.assertIn(5, result["frame_global_indices"])
        self.assertIsNone(data["frames"][6]["gt_xy"])
        self.assertTrue(all(value["error"] is None for value in data["frames"][6]["predictions"].values()))
        self.assertEqual(data["heatmap_display"], {"vmin": 0, "vmax": 1, "alpha": 0.38})
        self.assertTrue(all(record["head_count"] == 2 for record in data["frames"]))
        self.assertNotIn("NaN", Path(result["data_path"]).read_text())
        for method in METHODS:
            for key in arrays[method]:
                np.testing.assert_array_equal(arrays[method][key], original_arrays[method][key])
        for key in diagnostics:
            np.testing.assert_array_equal(diagnostics[key], original_diagnostics[key])

    def test_plotting_imports_no_torch(self):
        source = "import sys; sys.path.insert(0,sys.argv[1]); from stag.plotting import render_case; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", source, str(Path(__file__).resolve().parents[1])],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_timeline_labels_outcomes_and_limits_scoring_window(self):
        from matplotlib import pyplot as plt
        case, track, arrays, diagnostics, heads = self.fixture
        records = _case_records(case, track, arrays, diagnostics, heads)
        figure = _timeline_figure(case, records, TemporalConfig())
        self.assertIn("Scoring window: 3 frames", figure._suptitle.get_text())
        self.assertIn("SF: responded +0f", figure._suptitle.get_text())
        self.assertIn("STAG: censored", figure._suptitle.get_text())
        self.assertEqual(len(figure.axes[0].patches), 1)
        window = figure.axes[0].patches[0]
        vertices = window.get_path().transformed(window.get_transform()).transformed(
            figure.axes[0].transData.inverted()).vertices
        self.assertAlmostEqual(vertices[:, 0].min(), 103.5)
        self.assertAlmostEqual(vertices[:, 0].max(), 106.5)
        plt.close(figure)
        case = {**case, "event": None}
        figure = _timeline_figure(case, records, TemporalConfig())
        self.assertNotIn("Scoring window", figure._suptitle.get_text())
        self.assertEqual(len(figure.axes[0].patches), 0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
