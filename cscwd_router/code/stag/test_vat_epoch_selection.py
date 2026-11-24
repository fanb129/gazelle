"""CPU-only checks for multi-metric VAT epoch comparison."""

import json
from pathlib import Path
import unittest

from cscwd_router.code.stag.vat_epoch_selection import compare_epochs


class VatEpochSelectionTests(unittest.TestCase):
    def test_three_metric_dominance_excludes_inferior_checkpoint(self):
        result = compare_epochs([
            {"epoch_one_based": 1, "metrics": {"auc": .9, "l2": .1, "inout_ap": .8}},
            {"epoch_one_based": 2, "metrics": {"auc": .91, "l2": .09, "inout_ap": .81}}])
        self.assertEqual(result["pareto_epochs"], [2])
        self.assertEqual(result["recommended_epoch_one_based"], 2)

    def test_equal_metrics_keep_both_candidates_and_choose_earlier_epoch(self):
        metrics = {"auc": .9, "l2": .1, "inout_ap": .8}
        result = compare_epochs([{"epoch_one_based": epoch, "metrics": metrics} for epoch in (3, 1)])
        self.assertEqual(result["pareto_epochs"], [3, 1])
        self.assertEqual(result["recommended_epoch_one_based"], 1)

    def test_ap_localization_tradeoff_retains_both_candidates(self):
        result = compare_epochs([
            {"epoch_one_based": 1, "metrics": {"auc": .91, "l2": .09, "inout_ap": .8}},
            {"epoch_one_based": 2, "metrics": {"auc": .9, "l2": .1, "inout_ap": .9}}])
        self.assertEqual(result["pareto_epochs"], [1, 2])

    def test_real_vatl_history_recommends_epoch_three_and_retains_ap_best(self):
        root = Path(__file__).resolve().parents[3]
        history = json.loads((root / "cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/history.json").read_text())
        result = compare_epochs([{"epoch_one_based": row["epoch"] + 1, "metrics": row["test"]} for row in history])
        self.assertEqual(result["pareto_epochs"], [1, 2, 3, 5, 8])
        self.assertEqual(result["recommended_epoch_one_based"], 3)
        self.assertEqual(result["rows"][3]["dominated_by"], [3])
        self.assertGreater(result["rows"][4]["metrics"]["inout_ap"], result["rows"][2]["metrics"]["inout_ap"])


if __name__ == "__main__":
    unittest.main()
