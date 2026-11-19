"""Actual D6 five-epoch CLI, fixed controls, online reuse and test-GT isolation."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_utility_selector as training
from test_official_data import make_official_fixture
from stag.cache import file_sha256, load_cache
from stag.official_data import official_training_inputs
from stag.official_protocol import select_checkpoint
from stag.selector_bank import load_selector_bank
from stag.utility_selector import (VERSION, MODEL_ID, BANK_CANDIDATES, ANCHOR_INDEX,
                                   HISTORY_INDEX, UtilitySelector, step_selector_inputs)


class UtilitySelectorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.fixture = make_official_fixture(cls.root / "sources")
        cls.original_sources = {path: file_sha256(path) for path in (cls.root / "sources").rglob("*")
                                if path.is_file()}
        cls.plan = cls.root / "selector_plan.md"
        cls.plan.write_text((training.REPO_ROOT / "cscwd/r4_utility_selector_plan.md").read_text())
        cls.report_path, cls.summary_path, cls.manifest_path, cls.provenance = (
            cls.root / name for name in ("result.json", "result.md", "selection.json", "provenance.md"))
        command = [sys.executable, str(Path(training.__file__))]
        for name in training.INPUT_NAMES:
            command.extend(("--" + name.replace("_", "-"), str(cls.fixture[name])))
        for name, value in training.SELECTOR_BUDGET.items():
            command.extend(("--" + name.replace("_", "-"), str(value)))
        command.extend(("--plan", str(cls.plan), "--output-dir", str(cls.root / "cli_outputs"),
            "--device", "cpu", "--report", str(cls.report_path), "--summary", str(cls.summary_path),
            "--selection-manifest", str(cls.manifest_path), "--provenance", str(cls.provenance)))
        cls.process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if cls.process.returncode:
            raise AssertionError(cls.process.stdout + cls.process.stderr)
        cls.report = json.loads(cls.report_path.read_text())
        cls.banks = {split: load_selector_bank(record) for split, record in cls.report["banks"].items()}

    def inputs(self):
        return official_training_inputs(*(self.fixture[name] for name in training.INPUT_NAMES))

    def checkpoint(self, item):
        return torch.load(item["path"], weights_only=True, map_location="cpu")

    def test_actual_cli_keeps_five_epochs_controls_and_artifact_provenance(self):
        report = self.report
        self.assertIn("R4 utility selector completed", self.process.stdout)
        self.assertEqual(report["task"], "R4_utility_selector_training")
        self.assertEqual(report["version"], VERSION)
        self.assertEqual(report["model_id"], MODEL_ID)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["test_used_for_checkpoint_selection"])
        self.assertFalse(report["independent_held_out_test_estimate"])
        self.assertFalse(report["config"]["has_validation_split"])
        self.assertEqual(list(report["baselines"]), list(BANK_CANDIDATES))
        self.assertEqual([item["epoch"] for item in report["history"]], [1, 2, 3, 4, 5])
        self.assertEqual(set(report["delta_vs_baselines"]), set(BANK_CANDIDATES))
        self.assertEqual(set(report["paired_response"]), set(BANK_CANDIDATES))
        self.assertIn("grid_ema_075", self.summary_path.read_text())
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["selection"], report["selection"])
        self.assertEqual(manifest["selected"], report["selected"])
        self.assertEqual(manifest["report"]["sha256"], file_sha256(self.report_path))
        self.assertIn(file_sha256(self.report_path), self.provenance.read_text())
        self.assertIn(file_sha256(self.manifest_path), self.provenance.read_text())
        for path, digest in report["code_sha256"].items():
            self.assertEqual(digest, file_sha256(training.REPO_ROOT / path))
        self.assertEqual(self.original_sources,
                         {path: file_sha256(path) for path in self.original_sources})

    def test_two_banks_cover_complete_splits_and_grid_ema_is_a_fixed_control(self):
        for split, bank in self.banks.items():
            self.assertEqual(self.report["coverage"][split],
                {"tracks": 2, "person_frames": 24, "unique_image_frames": 12})
            self.assertEqual(bank["index"]["metadata"]["split"], split)
            self.assertTrue(all(track["track_id"].startswith(split + "/")
                                for track in bank["index"]["tracks"]))
            self.assertEqual(bank["index"]["candidate_names"], list(BANK_CANDIDATES))
            self.assertEqual(bank["observations"].shape, (24, 41))
            for name in ("heatmaps", "observations", "inout", "candidate_xy", "history_available"):
                self.assertFalse(bank[name].flags.writeable)
            for track in bank["index"]["tracks"]:
                previous, previous_index = None, None
                for row, frame in enumerate(track["frames"], start=track["start"]):
                    anchor = bank["heatmaps"][row, ANCHOR_INDEX]
                    expected = (.75 * anchor + .25 * previous
                        if previous_index is not None and frame["frame_index"] == previous_index + 1 else anchor)
                    np.testing.assert_array_equal(bank["heatmaps"][row, 3], expected)
                    previous, previous_index = expected, frame["frame_index"]

    def test_all_fixed_controls_and_epoch_predictions_keep_raw_scores_ap_and_state(self):
        reference = self.report["baselines"]["current"]
        rows = [*self.report["baselines"].values(),
                *(item["test"] for item in self.report["history"])]
        for evaluation in rows:
            self.assertEqual(evaluation["standard"]["person_frames"], 24)
            self.assertEqual(evaluation["standard"]["localization_frames"], 18)
            self.assertEqual(evaluation["standard"]["inout_ap"], reference["standard"]["inout_ap"])
            self.assertEqual(evaluation["state"], reference["state"])
            self.assertEqual(evaluation["temporal"]["response"]["leave_frame"],
                             reference["temporal"]["response"]["leave_frame"])
            self.assertTrue(all(evaluation["cache"]["checks"].values()))
            _, arrays = load_cache(evaluation["cache"]["path"])
            np.testing.assert_array_equal(arrays["inout"], self.banks["test"]["inout"])

    def test_weighted_loss_and_supervision_are_recomputed_from_train_only(self):
        bank = self.banks["train"]
        gt, inside = training.training_targets(bank)
        xy = bank["candidate_xy"].astype(np.float64)
        advantage = np.linalg.norm(xy[:, ANCHOR_INDEX] - gt, axis=-1) - np.linalg.norm(
            xy[:, HISTORY_INDEX] - gt, axis=-1)
        supervised = inside & bank["history_available"]
        weights = np.abs(advantage) * supervised
        targets = (advantage > 0).astype(np.float64)
        self.assertEqual(self.report["expected_train_supervised_frames"], int(supervised.sum()))
        torch.manual_seed(0)
        model = UtilitySelector(4, 64)
        for item in self.report["history"]:
            with torch.no_grad():
                logits = model(torch.tensor(np.array(bank["observations"]))).numpy().astype(np.float64)
            expected_sum = float(((np.logaddexp(0., logits) - targets * logits) * weights).sum())
            training_row = item["training"]
            self.assertEqual(training_row["person_frames"], 24)
            self.assertEqual(training_row["supervised_frames"], int(supervised.sum()))
            self.assertEqual(training_row["prefer_history_frames"], int((supervised & (advantage > 0)).sum()))
            self.assertEqual(training_row["tie_frames"], int((supervised & (advantage == 0)).sum()))
            self.assertAlmostEqual(training_row["weight_sum"], float(weights.sum()), places=11)
            self.assertAlmostEqual(training_row["weighted_bce_sum"], expected_sum, places=6)
            self.assertAlmostEqual(training_row["loss"],
                training_row["weighted_bce_sum"] / max(training_row["weight_sum"], 1e-12), places=12)
            model.load_state_dict(self.checkpoint(item["checkpoint"])["model"])

    def test_logits_choices_and_whole_candidate_maps_match_each_saved_epoch(self):
        bank = self.banks["test"]
        for item in self.report["history"]:
            evaluation = item["test"]
            choices = np.load(evaluation["selection"]["choices"]["path"], allow_pickle=False)
            logits = np.load(evaluation["selection"]["logits"]["path"], allow_pickle=False)
            expected = np.where((logits > 0) & bank["history_available"], HISTORY_INDEX, ANCHOR_INDEX)
            np.testing.assert_array_equal(choices, expected)
            self.assertTrue(set(choices.tolist()) <= {ANCHOR_INDEX, HISTORY_INDEX})
            _, predicted = load_cache(evaluation["cache"]["path"])
            positions = np.arange(24)
            np.testing.assert_array_equal(predicted["heatmap"], bank["heatmaps"][positions, choices])
            np.testing.assert_array_equal(predicted["argmax_xy"], bank["candidate_xy"][positions, choices])
            self.assertEqual(evaluation["selection"]["logits"]["sha256"],
                             file_sha256(evaluation["selection"]["logits"]["path"]))

    def test_selected_checkpoint_reload_and_online_current_frame_match_bank_outputs(self):
        report, bank = self.report, self.banks["test"]
        winner = select_checkpoint(report["candidates"])
        self.assertEqual(report["selection"]["epoch"], winner["epoch"])
        selected = report["history"][winner["epoch"] - 1]
        self.assertEqual(report["selection"]["checkpoint"]["sha256"], selected["checkpoint"]["sha256"])
        self.assertEqual(report["last_checkpoint"]["sha256"], report["history"][-1]["checkpoint"]["sha256"])
        for item in report["history"]:
            self.assertEqual(item["checkpoint"]["sha256"], file_sha256(item["checkpoint"]["path"]))
        checkpoint = self.checkpoint(report["selection"]["checkpoint"])
        model = UtilitySelector(checkpoint["feature_channels"], checkpoint["config"]["hidden_dim"])
        model.load_state_dict(checkpoint["model"])
        logits, choices = training.infer_selector(model, bank, "cpu", 256)
        stored = selected["test"]["selection"]
        np.testing.assert_array_equal(logits, np.load(stored["logits"]["path"], allow_pickle=False))
        np.testing.assert_array_equal(choices, np.load(stored["choices"]["path"], allow_pickle=False))
        part = self.inputs()["test"]
        feature_rows = {path: row for row, path in enumerate(part["feature_index"]["image_paths"])}
        for track in part["tracks"]:
            state = None
            for row, frame in enumerate(track["frames"], start=track["start"]):
                with torch.no_grad():
                    maps, observation, available, state = step_selector_inputs(
                        torch.tensor(np.array(part["arrays"]["heatmap"][row])),
                        torch.tensor(np.array(part["features"][feature_rows[frame["path"]]])),
                        torch.tensor(part["arrays"]["inout"][row].item()),
                        torch.tensor(frame["bbox_norm"]), frame["frame_index"], state)
                    online_logit = model(observation[None]).item()
                np.testing.assert_array_equal(observation.numpy(), bank["observations"][row])
                self.assertAlmostEqual(online_logit, float(logits[row]), places=7)
                online_choice = HISTORY_INDEX if online_logit > 0 and available else ANCHOR_INDEX
                self.assertEqual(online_choice, choices[row])
                np.testing.assert_array_equal(maps[BANK_CANDIDATES[online_choice]].numpy(),
                                              bank["heatmaps"][row, choices[row]])

    def test_changed_test_labels_do_not_change_any_of_five_training_models(self):
        inputs = self.inputs()
        changed = {**inputs, "test": {**inputs["test"],
            "tracks": copy.deepcopy(inputs["test"]["tracks"])}}
        changed["test"]["prediction_index"] = {**inputs["test"]["prediction_index"],
                                               "tracks": changed["test"]["tracks"]}
        for track in changed["test"]["tracks"]:
            for frame in track["frames"]:
                frame["inout"] = 1 - frame["inout"]
                frame["gazex_norm"] = [.81] if frame["inout"] else [-1.]
                frame["gazey_norm"] = [.13] if frame["inout"] else [-1.]
        with patch.object(training, "official_training_inputs", return_value=changed) as load:
            report = training.run_selector_training(*(self.fixture[name] for name in training.INPUT_NAMES),
                self.plan, self.root / "changed_test_labels", device="cpu")
        load.assert_called_once()
        for original, altered in zip(self.report["history"], report["history"]):
            first = self.checkpoint(original["checkpoint"])["model"]
            second = self.checkpoint(altered["checkpoint"])["model"]
            self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
        self.assertEqual(report["sources"], self.report["sources"])
        self.assertEqual(self.original_sources,
                         {path: file_sha256(path) for path in self.original_sources})

    def test_fixed_budget_and_plan_are_rejected_before_outputs(self):
        arguments = tuple(self.fixture[name] for name in training.INPUT_NAMES)
        for settings in ({"epochs": 1}, {"seed": 1}, {"batch_size": 32}):
            with self.subTest(settings=settings):
                destination = self.root / ("wrong_" + next(iter(settings)))
                with self.assertRaisesRegex(ValueError, "budget"):
                    training.run_selector_training(*arguments, self.plan, destination,
                                                   device="cpu", **settings)
                self.assertFalse(destination.exists())
        wrong_plan = self.root / "wrong_plan.md"
        wrong_plan.write_text("# Different experiment\n")
        with self.assertRaisesRegex(ValueError, "Plan|version"):
            training.run_selector_training(*arguments, wrong_plan, self.root / "wrong_plan_outputs", device="cpu")


if __name__ == "__main__":
    unittest.main()
