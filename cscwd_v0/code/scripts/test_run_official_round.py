"""Check official pipeline ordering, GPU inheritance and failure propagation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).with_name("run_official_round.sh")


class OfficialRoundTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "repository with spaces"
        self.results = self.repository / "results with spaces"
        self.results.mkdir(parents=True)
        self.train_cache = self.root / "train features"
        self.test_cache = self.root / "test features"
        self.experiments = self.repository / "experiment outputs"
        self.calls = self.root / "calls.jsonl"
        self.python = self.root / "fake python"
        self.python.write_text("#!" + sys.executable + '\n' + r'''
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
script = Path(args[1]).name
stage = {"cache_official_test_features.py": "cache",
         "compare_official_joint.py": "comparison"}.get(script)
if stage is None:
    stage = args[args.index("--model-id") + 1]
with open(os.environ["OFFICIAL_CALLS"], "a") as stream:
    stream.write(json.dumps({"stage": stage, "args": args,
                             "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
if stage == os.environ.get("OFFICIAL_FAIL_STAGE"):
    sys.exit(23)
Path(args[args.index("--report") + 1]).write_text("{}\n")
''')
        self.python.chmod(0o755)

    def run_round(self, fail=None, cached=False):
        if cached:
            (self.results / "r4_official_test_features.json").write_text("{}\n")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1",
                       "OFFICIAL_CALLS": str(self.calls)}
        if fail:
            environment["OFFICIAL_FAIL_STAGE"] = fail
        process = subprocess.run(["bash", str(WRAPPER), str(self.python),
            str(self.repository), str(self.train_cache), str(self.test_cache),
            str(self.experiments), str(self.results)], capture_output=True,
            text=True, env=environment)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return process, calls

    def test_complete_round_runs_cache_train_models_compare_on_gpu_one(self):
        process, calls = self.run_round()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([call["stage"] for call in calls],
                         ["cache", "ordinary_gru", "history_validity", "comparison"])
        self.assertTrue(all(call["gpu"] == "1" for call in calls))
        self.assertIn("R4 official train/test round completed.", process.stdout)
        for call in calls[1:3]:
            args = call["args"]
            for key, value in {"--train-feature-cache": self.train_cache,
                               "--test-feature-cache": self.test_cache,
                               "--epochs": 5, "--seed": 0, "--device": "cuda:0"}.items():
                self.assertEqual(args[args.index(key)+1], str(value))
            self.assertTrue(args[args.index("--train-prediction-report")+1].endswith("r1_train_single_frame_cache.json"))
            self.assertTrue(args[args.index("--test-prediction-report")+1].endswith("t2_single_frame_cache.json"))
        self.assertTrue((self.results / "r4_official_comparison.json").exists())

    def test_existing_feature_report_skips_cache_stage(self):
        process, calls = self.run_round(cached=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([call["stage"] for call in calls],
                         ["ordinary_gru", "history_validity", "comparison"])

    def check_failure(self, stage, expected):
        process, calls = self.run_round(fail=stage)
        self.assertEqual(process.returncode, 23)
        self.assertEqual([call["stage"] for call in calls], expected)
        self.assertNotIn("R4 official train/test round completed.", process.stdout)

    def test_cache_failure_stops_training(self):
        self.check_failure("cache", ["cache"])

    def test_first_model_failure_stops_second_model(self):
        self.check_failure("ordinary_gru", ["cache", "ordinary_gru"])

    def test_second_model_failure_stops_comparison(self):
        self.check_failure("history_validity", ["cache", "ordinary_gru", "history_validity"])

    def test_comparison_failure_is_not_reported_as_success(self):
        self.check_failure("comparison", ["cache", "ordinary_gru", "history_validity", "comparison"])


if __name__ == "__main__":
    unittest.main()
