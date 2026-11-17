"""Meaningful selection and tradeoff checks for the compact R4 experiment."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stag.compact_protocol import (
    COMPACT_BUDGET, PLAN_VERSION, make_candidate, pareto_front,
    select_checkpoint, validate_budget,
)


def validation(auc=0.95, l2=0.08, ap=0.96, stability=0.6, gaze=0.7,
               state_error=0.1, enter=0.9):
    return {
        "standard": {"auc": auc, "l2": l2, "inout_ap": ap},
        "temporal": {
            "jitter": {"predicted": {"mean": 0.01}},
            "stability": {"accurate_stable_ratio": stability},
            "response": {"enter_frame": {"response_rate": enter},
                         "leave_frame": {"response_rate": 0.6},
                         "gaze_shift": {"response_rate": gaze}},
        },
        "state": {
            "state_error": {"error_rate": state_error},
            "stable_state_flips": {"flip_rate": 0.02},
            "response": {"all": {"response_rate": 0.8},
                         "enter_frame": {"response_rate": enter},
                         "leave_frame": {"response_rate": 0.6}},
        },
    }


class CompactProtocolTests(unittest.TestCase):
    def setUp(self):
        self.baseline = validation()

    def candidate(self, model="history_validity", epoch=1, **values):
        return make_candidate(model, epoch, validation(**values), self.baseline)

    def test_budget_is_fixed_and_independent_of_the_R1_full_search(self):
        self.assertEqual(PLAN_VERSION, "r4_compact_v2")
        self.assertIsNone(validate_budget(dict(COMPACT_BUDGET, model_id="ordinary_gru")))
        for name, value in (("epochs", 1), ("seed", 1), ("hidden_dim", 64),
                            ("learning_rate", 0.0003), ("temperature", 0.2)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_budget(dict(COMPACT_BUDGET, **{name: value}))
        missing = dict(COMPACT_BUDGET)
        del missing["lambda_state"]
        with self.assertRaisesRegex(ValueError, "lambda_state"):
            validate_budget(missing)

    def test_precision_boundaries_are_inclusive_but_not_unlimited(self):
        reference = self.baseline["standard"]
        exact = self.candidate(auc=reference["auc"] - 0.002,
                               l2=reference["l2"] + 0.002,
                               ap=reference["inout_ap"] - 0.002)
        self.assertTrue(exact["precision_eligible"])
        for key, value in (("auc", 0.9479), ("l2", 0.0821), ("ap", 0.9579)):
            with self.subTest(key=key):
                self.assertFalse(self.candidate(**{key: value})["precision_eligible"])

    def test_response_loss_is_reported_without_the_old_per_class_veto(self):
        candidate = self.candidate(enter=0.4)
        self.assertTrue(candidate["precision_eligible"])
        self.assertEqual(candidate["metrics"]["state_enter_frame_response_rate"], 0.4)
        self.assertEqual(candidate["metrics"]["joint_enter_frame_response_rate"], 0.4)
        self.assertIs(select_checkpoint([candidate]), candidate)

    def test_higher_stability_cannot_override_failed_precision(self):
        bad = self.candidate(epoch=2, stability=0.99, auc=0.9)
        acceptable = self.candidate(epoch=1, stability=0.61)
        self.assertIs(select_checkpoint([bad, acceptable]), acceptable)
        self.assertIsNone(select_checkpoint([bad]))

    def test_lexicographic_ranking_uses_each_declared_priority(self):
        for values in ({"stability": 0.61}, {"gaze": 0.71}, {"ap": 0.961},
                       {"l2": 0.079}, {"auc": 0.951}):
            with self.subTest(values=values):
                better = self.candidate(epoch=3, **values)
                earlier = self.candidate(epoch=1)
                self.assertIs(select_checkpoint([earlier, better]), better)
        stability = self.candidate(epoch=4, stability=0.61, gaze=0.1, ap=0.959)
        response = self.candidate(epoch=1, stability=0.6, gaze=0.99, ap=0.99)
        self.assertIs(select_checkpoint([response, stability]), stability)
        response = self.candidate(epoch=4, gaze=0.71, ap=0.959)
        ap = self.candidate(epoch=1, gaze=0.7, ap=0.99)
        self.assertIs(select_checkpoint([ap, response]), response)

    def test_ties_use_earlier_epoch_then_stable_model_order(self):
        early = self.candidate(model="z_model", epoch=1)
        late = self.candidate(model="a_model", epoch=2)
        self.assertIs(select_checkpoint([late, early]), early)
        lexical = self.candidate(model="a_model", epoch=1)
        self.assertIs(select_checkpoint([early, lexical]), lexical)

    def test_undefined_responses_are_not_fabricated_as_zero(self):
        undefined = self.candidate(epoch=1, gaze=None)
        zero = self.candidate(epoch=2, gaze=0.)
        self.assertIsNone(undefined["metrics"]["joint_gaze_shift_response_rate"])
        self.assertIs(select_checkpoint([undefined, zero]), zero)
        self.assertEqual(pareto_front([undefined, zero]),
                         sorted([undefined["candidate_id"], zero["candidate_id"]]))
        absent = self.candidate(auc=None, l2=None, ap=None, stability=None, gaze=None)
        self.assertFalse(absent["precision_eligible"])
        self.assertIsNone(select_checkpoint([absent]))

    def test_pareto_front_keeps_real_tradeoffs_and_single_frame_reference(self):
        baseline = make_candidate("single_frame", 0, self.baseline, self.baseline)
        stable = self.candidate(stability=0.7, state_error=0.12)
        worse = self.candidate(model="ordinary_gru", auc=0.949, l2=0.081,
                               ap=0.959, stability=0.59, gaze=0.69, state_error=0.11)
        self.assertEqual(pareto_front([baseline, stable, worse]),
                         sorted([baseline["candidate_id"], stable["candidate_id"]]))
        stronger = self.candidate(epoch=2, stability=0.7, state_error=0.09)
        self.assertEqual(pareto_front([baseline, stable, worse, stronger]),
                         [stronger["candidate_id"]])

    def test_pareto_includes_all_epochs_without_precision_filter_or_ranking(self):
        baseline = make_candidate("single_frame", 0, self.baseline, self.baseline)
        tradeoff = self.candidate(auc=0.94, stability=0.8)
        self.assertFalse(tradeoff["precision_eligible"])
        self.assertEqual(pareto_front([baseline, tradeoff]),
                         sorted([baseline["candidate_id"], tradeoff["candidate_id"]]))
        identical = copy.deepcopy(tradeoff)
        identical["candidate_id"] = "other:epoch_001"
        self.assertEqual(pareto_front([tradeoff, identical]),
                         sorted([tradeoff["candidate_id"], identical["candidate_id"]]))

    def test_pareto_skips_shared_undefined_axes_but_not_comparability_gaps(self):
        one = self.candidate(epoch=1, gaze=None, stability=0.6)
        two = self.candidate(epoch=2, gaze=None, stability=0.7)
        self.assertEqual(pareto_front([one, two]), [two["candidate_id"]])
        three = self.candidate(epoch=3, gaze=0.8, stability=0.9)
        self.assertEqual(pareto_front([one, two, three]),
                         sorted([two["candidate_id"], three["candidate_id"]]))

    def test_existing_R2_smoke_is_not_rewritten_by_the_new_protocol(self):
        path = Path(__file__).resolve().parents[2] / "results/r2_joint_smoke.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        candidate = make_candidate("ordinary_gru", 1,
                                   report["history"][0]["validation"], report["baseline_validation"])
        self.assertTrue(candidate["precision_eligible"])
        self.assertEqual(candidate["metrics"]["state_enter_frame_response_rate"], 175 / 189)
        self.assertEqual(candidate["metrics"]["accurate_stable_ratio"], 12563 / 21129)
        self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
