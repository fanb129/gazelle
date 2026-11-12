"""Synthetic checks against the repository's original VAT metric functions."""

import ast
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import evaluate_predictions, vat_auc_numpy


class TensorLike(np.ndarray):
    """Supply the original AUC's .cpu() interface without importing PyTorch."""

    def cpu(self):
        return self


def original_vat_functions():
    source = Path(__file__).resolve().parents[3] / "gazelle/utils.py"
    parsed = ast.parse(source.read_text())
    functions = [node for node in parsed.body
                 if isinstance(node, ast.FunctionDef) and node.name in {"vat_auc", "vat_l2"}]
    namespace = {"np": np, "roc_auc_score": roc_auc_score}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["vat_auc"], namespace["vat_l2"]


def frame(gazex, gazey, inout=1):
    return {"gazex_norm": [gazex], "gazey_norm": [gazey], "inout": inout}


class StandardMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_auc, cls.original_l2 = map(staticmethod, original_vat_functions())

    def test_auc_matches_original_at_interior_and_image_boundaries(self):
        rng = np.random.default_rng(7)
        for gazex, gazey in [(0.5, 0.4), (0, 0), (1, 1), (0, 1), (1, 0), (63 / 64, 63 / 64)]:
            with self.subTest(gazex=gazex, gazey=gazey):
                heatmap = rng.random((64, 64), dtype=np.float32)
                expected = self.original_auc(heatmap.view(TensorLike), gazex, gazey)
                self.assertEqual(vat_auc_numpy(heatmap, gazex, gazey), expected)

    def test_evaluator_matches_original_auc_l2_and_full_ap(self):
        rng = np.random.default_rng(23)
        heatmaps = rng.random((5, 64, 64), dtype=np.float32)
        frames = [frame(0.2, 0.7), frame(-1, -1, 0), frame(1, 1),
                  frame(0, 0), frame(-1, -1, 0)]
        tracks = [{"start": 0, "stop": 2, "frames": frames[:2]},
                  {"start": 2, "stop": 5, "frames": frames[2:]}]
        probabilities = np.array([0.9, 0.1, 0.8, 0.7, 0.6], dtype=np.float32)
        result = evaluate_predictions(tracks, heatmaps, probabilities)
        expected_auc, expected_l2 = [], []
        for index, annotation in enumerate(frames):
            if annotation["inout"]:
                x, y = annotation["gazex_norm"][0], annotation["gazey_norm"][0]
                tensor = heatmaps[index].view(TensorLike)
                expected_auc.append(self.original_auc(tensor, x, y))
                expected_l2.append(self.original_l2(tensor, x, y))
        self.assertEqual(result["auc"], float(np.mean(expected_auc)))
        self.assertAlmostEqual(result["l2"], float(np.mean(expected_l2)), places=15)
        self.assertEqual(result["inout_ap"], average_precision_score([1, 0, 1, 1, 0], probabilities))
        self.assertEqual((result["person_frames"], result["localization_frames"]), (5, 3))

    def test_out_of_frame_heatmaps_and_coordinates_do_not_affect_localization(self):
        heatmaps = np.zeros((2, 64, 64), dtype=np.float32)
        heatmaps[0, 32, 32] = 1
        annotations = [frame(0.5, 0.5), frame(-1, -1, 0)]
        tracks = [{"start": 0, "stop": 2, "frames": annotations}]
        result = evaluate_predictions(tracks, heatmaps, np.array([0.8, 0.2]))
        heatmaps[1] = np.random.default_rng(8).random((64, 64))
        annotations[1]["gazex_norm"] = [999]
        annotations[1]["gazey_norm"] = [-999]
        self.assertEqual(evaluate_predictions(tracks, heatmaps, np.array([0.8, 0.2])), result)
        self.assertEqual(result["l2"], 0)

    def test_ap_positive_class_is_in_frame_and_includes_all_frames(self):
        heatmaps = np.zeros((4, 64, 64), dtype=np.float32)
        annotations = [frame(0, 0, label) for label in (1, 0, 1, 0)]
        tracks = [{"start": 0, "stop": 4, "frames": annotations}]
        perfect = evaluate_predictions(tracks, heatmaps, np.array([0.9, 0.2, 0.8, 0.1]))
        reversed_scores = evaluate_predictions(tracks, heatmaps, np.array([0.1, 0.8, 0.2, 0.9]))
        self.assertEqual(perfect["inout_ap"], 1)
        self.assertAlmostEqual(reversed_scores["inout_ap"], (1 / 3 + 2 / 4) / 2)
        self.assertEqual(reversed_scores["localization_frames"], 2)

    def test_longer_tracks_receive_frame_proportional_weight(self):
        heatmaps = np.zeros((4, 64, 64), dtype=np.float32)
        heatmaps[0, 0, 0] = 1
        heatmaps[1:, 32, 32] = 1
        tracks = [{"start": 0, "stop": 1, "frames": [frame(0, 0)]},
                  {"start": 1, "stop": 4, "frames": [frame(0, 0)] * 3}]
        result = evaluate_predictions(tracks, heatmaps, np.ones(4))
        self.assertAlmostEqual(result["l2"], 3 * np.sqrt(0.5) / 4)
        expected_aucs = [vat_auc_numpy(heatmap, 0, 0) for heatmap in heatmaps]
        self.assertEqual(result["auc"], float(np.mean(expected_aucs)))

    def test_tied_heatmaps_use_first_flat_argmax_and_half_auc(self):
        tracks = [{"start": 0, "stop": 1, "frames": [frame(0, 0)]}]
        result = evaluate_predictions(tracks, np.ones((1, 64, 64), dtype=np.float32), np.ones(1))
        self.assertEqual(result["auc"], 0.5)
        self.assertEqual(result["l2"], 0)


if __name__ == "__main__":
    unittest.main()
