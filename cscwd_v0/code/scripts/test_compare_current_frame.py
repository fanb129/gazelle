"""D4 small-report attribution, matched epochs and fixed control identity."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_current_frame as comparison
from compare_compact_joint import artifact
from stag.cache import write_json
from stag.official_protocol import make_candidate, select_checkpoint


class CurrentFrameComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = self.root / "plan.md"
        self.plan.write_text(comparison.PROTOCOL_VERSION + "\n" + comparison.CONTROL_VERSION)
        results = comparison.REPO_ROOT / "cscwd/results"
        self.reference = json.loads((results / "r4_official_comparison.json").read_text())
        self.reference_path = self.root / "reference.json"
        self.reference_runs = {}
        self.old_paths = {}
        for model_id in ("ordinary_gru", "history_validity"):
            run = json.loads((results / f"r4_official_{model_id}.json").read_text())
            self.reference_runs[model_id] = run
            path = self.root / f"old_{model_id}.json"
            write_json(path, run)
            self.old_paths[model_id] = path
            self.reference["source_reports"][model_id] = artifact(path)
        write_json(self.reference_path, self.reference)
        self.run = copy.deepcopy(self.reference_runs["ordinary_gru"])
        self.run.update(model_id=comparison.MODEL_ID, control_version=comparison.CONTROL_VERSION)
        self.run["config"].update(model_id=comparison.MODEL_ID, control_version=comparison.CONTROL_VERSION,
            history=comparison.HISTORY_POLICY, output_calibration="none", temporal_loss="none")
        self.run["plan"] = artifact(self.plan)
        self.run["source_reports"]["plan"] = artifact(self.plan)
        for item in self.run["history"]:
            item["test"]["diagnostics"] = None
        # Controlled fixture values differ from each matched temporal epoch;
        # they are small-report scaffolding, not new measured VAT results.
        last = self.run["history"][-1]["test"]
        last["standard"]["auc"] += .00005
        last["standard"]["inout_ap"] += .0001
        last["temporal"]["stability"]["accurate_stable_frames"] += 2
        last["temporal"]["stability"]["accurate_stable_ratio"] = (
            last["temporal"]["stability"]["accurate_stable_frames"] /
            last["temporal"]["stability"]["eligible_windows"])
        self.run["candidates"] = [make_candidate(comparison.MODEL_ID, item["epoch"], item["test"])
                                  for item in self.run["history"]]
        self.winner = select_checkpoint(self.run["candidates"])
        self.run["selection"].update(candidate_id=self.winner["candidate_id"], epoch=self.winner["epoch"])
        self.run_path = self.root / "current.json"
        write_json(self.run_path, self.run)
        self.details = {"joint": {"events": []}, "state": {"events": []}}

    def compare(self):
        with patch.object(comparison, "_verify_current_run", return_value=(
                self.run["candidates"], self.winner, self.details, self.details)) as verify:
            result = comparison.compare_current_frame(self.run_path, self.plan, self.reference_path)
            verify.assert_called_once()
            return result

    def rewrite_reference_run(self, model_id, mutate):
        run = json.loads(self.old_paths[model_id].read_text())
        mutate(run)
        write_json(self.old_paths[model_id], run)
        self.reference["source_reports"][model_id] = artifact(self.old_paths[model_id])
        write_json(self.reference_path, self.reference)

    def test_all_five_current_epochs_and_matched_temporal_rows_are_retained(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["control_version"], comparison.CONTROL_VERSION)
        self.assertEqual(len(result["candidate_rows"]), 6)
        self.assertEqual(result["selected"], self.winner)
        self.assertEqual(result["selection"], self.run["selection"])
        self.assertEqual(result["current_selected"]["epoch"], 5)
        self.assertEqual(result["output_state_calibration"], "none")
        self.assertEqual(result["temporal_supervision"], "none")
        prior = result["reference_round"]
        self.assertEqual(set(prior["same_epoch"]), {"ordinary_gru", "history_validity"})
        self.assertEqual([row["epoch"] for row in prior["same_epoch"]["ordinary_gru"]], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(prior["same_epoch"]["ordinary_gru"][-1][
            "delta_current_frame_minus_reference"]["metrics"]["inout_ap"], .0001)
        self.assertEqual(prior["same_epoch"]["ordinary_gru"][-1][
            "delta_current_frame_minus_reference"]["counts"]["accurate_stable_windows"], 2)
        self.assertEqual(prior["same_epoch"]["ordinary_gru"][0][
            "delta_current_frame_minus_reference"]["metrics"]["inout_ap"], 0)
        self.assertIn("current_frame_vs_single_frame", result["paired_response"])

    def test_control_identity_and_disabled_temporal_losses_are_enforced(self):
        for key, value in (("control_version", "different"), ("history", "own_history"),
                           ("output_calibration", "D2"), ("temporal_loss", "D3")):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.run)
                changed["config"][key] = value
                write_json(self.run_path, changed)
                with patch.object(comparison, "_verify_current_run") as verify:
                    with self.assertRaisesRegex(ValueError, "fixed D4"):
                        comparison.compare_current_frame(self.run_path, self.plan, self.reference_path)
                    verify.assert_not_called()

    def test_original_budget_and_plan_are_enforced_before_artifact_work(self):
        changed = copy.deepcopy(self.run)
        changed["config"]["epochs"] = 1
        write_json(self.run_path, changed)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            self.compare()
        write_json(self.run_path, self.run)
        self.plan.write_text(comparison.PROTOCOL_VERSION)
        with self.assertRaisesRegex(ValueError, "D4 control plan"):
            self.compare()

    def test_reference_sources_initializer_and_baseline_must_match(self):
        for key in ("sources", "shared_initialization_sha256", "feature_contract"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.reference)
                changed["shared_contract"][key] = "different"
                write_json(self.reference_path, changed)
                with self.assertRaisesRegex(ValueError, "sources, initialization"):
                    self.compare()
        write_json(self.reference_path, self.reference)
        changed = copy.deepcopy(self.run)
        changed["baseline_test"]["standard"]["auc"] += .01
        write_json(self.run_path, changed)
        with self.assertRaisesRegex(ValueError, "sources, initialization"):
            self.compare()

    def test_reference_reports_are_hashed_without_requiring_previous_large_artifacts(self):
        old = self.reference_runs["ordinary_gru"]
        self.assertFalse(Path(old["selection"]["checkpoint"]["path"]).exists())
        self.assertFalse(Path(old["history"][-1]["test"]["cache"]["path"]).exists())
        self.compare()
        self.old_paths["ordinary_gru"].write_text("{}")
        with self.assertRaisesRegex(ValueError, "Artifact SHA256"):
            self.compare()

    def test_changed_reference_selection_or_stored_structure_is_rejected(self):
        self.reference["selected"]["ordinary_gru"]["epoch"] = 1
        write_json(self.reference_path, self.reference)
        with self.assertRaisesRegex(ValueError, "Selected epoch"):
            self.compare()
        self.reference["selected"]["ordinary_gru"]["epoch"] = 5
        write_json(self.reference_path, self.reference)
        self.rewrite_reference_run("ordinary_gru", lambda run: run.update(parameter_count=1))
        with self.assertRaisesRegex(ValueError, "stored parameter count"):
            self.compare()

    def test_summary_and_manifest_preserve_complete_row_and_effective_usage_scope(self):
        result = self.compare()
        summary = comparison.summary_markdown(result)
        self.assertIn("R4-D4 current-frame attribution", summary)
        self.assertIn("no D2 calibration or D3 temporal loss", summary)
        self.assertIn("effective parameter usage differs", summary)
        self.assertEqual(summary.count("| current_frame |"), 6)
        manifest = comparison.selection_manifest(result)
        self.assertEqual(manifest["control_version"], comparison.CONTROL_VERSION)
        self.assertEqual(manifest["selection"], self.run["selection"])
        self.assertEqual(manifest["selected"], self.winner)
        self.assertTrue(manifest["test_used_for_checkpoint_selection"])
        self.assertFalse(manifest["independent_test_estimate"])


if __name__ == "__main__":
    unittest.main()
