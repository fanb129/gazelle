"""Small-report controls and exact before/after deltas for D3 comparison."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_state_supervised as comparison
from compare_compact_joint import artifact
from stag.cache import write_json
from stag.official_protocol import make_candidate, select_checkpoint


class StateSupervisionComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = self.root / "plan.md"
        self.plan.write_text(comparison.PROTOCOL_VERSION + "\n" + comparison.SUPERVISION_VERSION)
        results = comparison.REPO_ROOT / "cscwd/results"
        self.previous = json.loads((results / "r4_official_comparison.json").read_text())
        self.previous_path = self.root / "previous.json"
        self.current = copy.deepcopy(self.previous)
        self.current["plan"] = artifact(self.plan)
        config = self.current["shared_contract"]["config"]
        config.update(supervision_version=comparison.SUPERVISION_VERSION, **comparison.SUPERVISION_SETTINGS)
        self.run_paths = []
        self.old_paths = {}
        for model_id in comparison.MODEL_IDS:
            old_run = json.loads((results / f"r4_official_{model_id}.json").read_text())
            old_path = self.root / f"old_{model_id}.json"
            write_json(old_path, old_run)
            self.old_paths[model_id] = old_path
            self.previous["source_reports"][model_id] = artifact(old_path)
            run = copy.deepcopy(old_run)
            run["supervision_version"] = comparison.SUPERVISION_VERSION
            run["config"].update(supervision_version=comparison.SUPERVISION_VERSION,
                                  **comparison.SUPERVISION_SETTINGS)
            run["plan"] = artifact(self.plan)
            run["source_reports"]["plan"] = artifact(self.plan)
            evaluation = run["history"][-1]["test"]
            evaluation["standard"]["auc"] += .00005
            evaluation["standard"]["l2"] -= .00002
            evaluation["standard"]["inout_ap"] += .0001
            stability = evaluation["temporal"]["stability"]
            stability["accurate_stable_frames"] += 2
            stability["accurate_stable_ratio"] = stability["accurate_stable_frames"] / stability["eligible_windows"]
            errors = evaluation["state"]["state_error"]
            errors["errors"] -= 3
            errors["error_rate"] = errors["errors"] / errors["person_frames"]
            errors["by_gt_state"]["in_frame"]["errors"] -= 3
            flips = evaluation["state"]["stable_state_flips"]
            flips["flips"] -= 4
            flips["flip_rate"] = flips["flips"] / flips["eligible_pairs"]
            for kind in ("enter_frame", "gaze_shift"):
                response = evaluation["temporal"]["response"][kind]
                response["responded_events"] += 1
                response["response_rate"] = response["responded_events"] / response["assessable_events"]
            run["candidates"] = [make_candidate(model_id, item["epoch"], item["test"])
                                 for item in run["history"]]
            winner = select_checkpoint(run["candidates"])
            self.current["selected"][model_id] = winner
            path = self.root / f"D3_{model_id}.json"
            write_json(path, run)
            self.run_paths.append(path)
            self.current["source_reports"][model_id] = artifact(path)
        write_json(self.previous_path, self.previous)

    def compare(self):
        with patch.object(comparison, "compare_runs", return_value=copy.deepcopy(self.current)) as primary:
            result = comparison.compare_supervised_runs(self.run_paths, self.plan, self.previous_path)
            primary.assert_called_once_with(self.run_paths, self.plan)
            return result

    def rewrite_old(self, model_id, mutate):
        path = self.old_paths[model_id]
        run = json.loads(path.read_text())
        mutate(run)
        write_json(path, run)
        self.previous["source_reports"][model_id] = artifact(path)
        write_json(self.previous_path, self.previous)

    def test_all_new_epochs_retained_and_deltas_compare_selected_original_outputs(self):
        result = self.compare()
        self.assertEqual(result["candidate_rows"], self.current["candidate_rows"])
        self.assertEqual(result["selected"], self.current["selected"])
        self.assertEqual(result["supervision_version"], comparison.SUPERVISION_VERSION)
        self.assertTrue(all(result["checks"].values()))
        prior = result["previous_round"]
        self.assertEqual(prior["comparison_scope"], "original_uncalibrated_outputs; no_D2_state_calibration")
        for model_id in comparison.MODEL_IDS:
            delta = prior["delta_vs_previous_same_model"][model_id]
            self.assertAlmostEqual(delta["metrics"]["auc"], .00005)
            self.assertAlmostEqual(delta["metrics"]["l2"], -.00002)
            self.assertAlmostEqual(delta["metrics"]["inout_ap"], .0001)
            self.assertEqual(delta["counts"]["accurate_stable_windows"], 2)
            self.assertEqual(delta["counts"]["state_errors"], -3)
            self.assertEqual(delta["counts"]["stable_state_flips"], -4)
            self.assertEqual(delta["counts"]["joint_enter_frame_responded"], 1)
            self.assertEqual(delta["counts"]["joint_gaze_shift_responded"], 1)
            self.assertEqual(prior["current_selected"][model_id]["epoch"], 5)
        self.assertEqual(prior["comparison_report"], artifact(self.previous_path))

    def test_previous_large_artifacts_are_not_needed_for_small_report_comparison(self):
        # Fixture checkpoints and caches retain their unavailable server paths.
        old = json.loads(self.old_paths["history_validity"].read_text())
        self.assertFalse(Path(old["selection"]["checkpoint"]["path"]).exists())
        self.assertFalse(Path(old["history"][-1]["test"]["cache"]["path"]).exists())
        result = self.compare()
        self.assertIn("previous_NPY_PT_and_event_details_not_reopened",
                      result["previous_round"]["previous_artifact_verification"])

    def test_changed_supervision_settings_or_version_are_rejected_before_primary_comparison(self):
        for field,value in (("lambda_time", .2), ("response_frames", 2), ("supervision_version", "old")):
            with self.subTest(field=field):
                path = self.run_paths[0]
                original = path.read_text()
                run = json.loads(original)
                run["config"][field] = value
                write_json(path, run)
                with patch.object(comparison, "compare_runs") as primary:
                    with self.assertRaisesRegex(ValueError, "fixed state-supervision"):
                        comparison.compare_supervised_runs(self.run_paths, self.plan, self.previous_path)
                    primary.assert_not_called()
                path.write_text(original)

    def test_previous_frozen_sources_budget_or_baseline_must_match(self):
        for key in ("sources", "feature_contract", "shared_initialization_sha256"):
            with self.subTest(key=key):
                old = copy.deepcopy(self.previous)
                old["shared_contract"][key] = "different"
                write_json(self.previous_path, old)
                with self.assertRaisesRegex(ValueError, "Previous and D3 rounds differ"):
                    self.compare()
        write_json(self.previous_path, self.previous)
        self.current["baseline_test"]["standard"]["auc"] += .01
        with self.assertRaisesRegex(ValueError, "Previous and D3 rounds differ"):
            self.compare()

    def test_previous_report_tampering_is_detected(self):
        self.old_paths["ordinary_gru"].write_text("{}")
        with self.assertRaisesRegex(ValueError, "Artifact SHA256"):
            self.compare()

    def test_previous_selection_and_architecture_are_checked_without_loading_weights(self):
        self.previous["selected"]["history_validity"]["epoch"] = 1
        write_json(self.previous_path, self.previous)
        with self.assertRaisesRegex(ValueError, "Selected epoch"):
            self.compare()
        self.previous["selected"]["history_validity"]["epoch"] = 5
        write_json(self.previous_path, self.previous)
        self.rewrite_old("history_validity", lambda run: run.update(parameter_count=1))
        with self.assertRaisesRegex(ValueError, "sources or architecture"):
            self.compare()

    def test_summary_and_selection_manifest_keep_complete_epoch_selection_and_prior_source(self):
        result = self.compare()
        summary = comparison.summary_markdown(result)
        self.assertIn("R4-D3 state-supervision comparison", summary)
        self.assertEqual(summary.count("| D3 |"), 2)
        self.assertEqual(summary.count("| previous |"), 2)
        self.assertIn("no D2 calibration", summary)
        manifest = comparison.selection_manifest(result)
        self.assertEqual(manifest["supervision_version"], comparison.SUPERVISION_VERSION)
        self.assertEqual(manifest["previous_comparison"], artifact(self.previous_path))
        self.assertEqual(manifest["per_model_selection"], result["per_model_selection"])
        self.assertEqual(manifest["output_state_calibration"], "none")
        self.assertTrue(manifest["test_used_for_checkpoint_selection"])


if __name__ == "__main__":
    unittest.main()
