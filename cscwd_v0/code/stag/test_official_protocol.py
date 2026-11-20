"""Whole-epoch official test ranking follows the declared AUC/L2/AP order."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.official_protocol import (PROTOCOL_VERSION, TRAINING_BUDGET,
                                    make_candidate, select_checkpoint, validate_budget)
from test_compact_protocol import validation


class OfficialProtocolTests(unittest.TestCase):
    def candidate(self, model="history_validity", epoch=1, **values):
        return make_candidate(model, epoch, validation(**values))

    def test_new_protocol_preserves_the_common_training_budget(self):
        self.assertEqual(PROTOCOL_VERSION, "r4_official_train_test_v1")
        self.assertEqual(TRAINING_BUDGET["epochs"], 5)
        self.assertIsNone(validate_budget(dict(TRAINING_BUDGET, model_id="ordinary_gru")))
        with self.assertRaisesRegex(ValueError, "epochs"):
            validate_budget(dict(TRAINING_BUDGET, epochs=1))

    def test_auc_is_primary_even_if_other_metrics_or_old_precision_gate_disagree(self):
        one = self.candidate(epoch=1, auc=.95, ap=.96, l2=.08, stability=.9)
        two = self.candidate(epoch=4, auc=.951, ap=.8, l2=.1, stability=.4)
        self.assertIs(select_checkpoint([one, two]), two)
        self.assertNotIn("precision_eligible", two)
        self.assertEqual(len(two["metrics"]), 13)

    def test_tied_auc_uses_l2_then_ap(self):
        baseline = self.candidate(epoch=1)
        better_l2 = self.candidate(epoch=3, l2=.079, ap=.8)
        better_ap = self.candidate(epoch=5, ap=.97)
        self.assertIs(select_checkpoint([baseline, better_l2, better_ap]), better_l2)
        self.assertIs(select_checkpoint([baseline, better_ap]), better_ap)

    def test_complete_ties_use_earlier_epoch_then_model_id(self):
        early = self.candidate(model="z", epoch=1)
        late = self.candidate(model="a", epoch=2)
        lexical = self.candidate(model="a", epoch=1)
        self.assertIs(select_checkpoint([early, late]), early)
        self.assertIs(select_checkpoint([early, lexical]), lexical)
        self.assertIsNone(select_checkpoint([]))

    def test_selected_metrics_remain_one_complete_epoch_without_columnwise_maxima(self):
        best_auc = self.candidate(epoch=2, auc=.97, l2=.09, ap=.94, gaze=None)
        best_l2 = self.candidate(epoch=3, auc=.96, l2=.06, ap=.98)
        chosen = select_checkpoint([best_auc, best_l2])
        self.assertIs(chosen, best_auc)
        self.assertEqual(chosen["metrics"]["l2"], .09)
        self.assertEqual(chosen["metrics"]["inout_ap"], .94)
        self.assertIsNone(chosen["metrics"]["joint_gaze_shift_response_rate"])


if __name__ == "__main__":
    unittest.main()
