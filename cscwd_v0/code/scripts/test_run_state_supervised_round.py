"""D3 stage ordering, quoted paths, fixed settings and failure propagation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).with_name('run_state_supervised_round.sh')


class StateSupervisedRoundTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / 'repository with spaces'
        self.results = self.repository / 'results with spaces'
        self.results.mkdir(parents=True)
        self.train_cache, self.test_cache = (self.root / name for name in ('train features', 'test features'))
        self.experiments = self.repository / 'experiment outputs'
        self.calls = self.root / 'calls.jsonl'
        self.python = self.root / 'fake python'
        self.python.write_text('#!' + sys.executable + '\n' + r'''
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
script = Path(args[1]).name
stage = 'comparison' if script == 'compare_state_supervised.py' else args[args.index('--model-id') + 1]
with open(os.environ['D3_CALLS'], 'a') as stream:
    stream.write(json.dumps({'stage': stage, 'script': script, 'args': args,
                             'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
if stage == os.environ.get('D3_FAIL_STAGE'):
    sys.exit(23)
Path(args[args.index('--report') + 1]).write_text('{}\n')
''')
        self.python.chmod(0o755)

    def run_round(self, fail=None):
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'D3_CALLS': str(self.calls)}
        if fail:
            environment['D3_FAIL_STAGE'] = fail
        process = subprocess.run(['bash', str(WRAPPER), str(self.python), str(self.repository),
            str(self.train_cache), str(self.test_cache), str(self.experiments), str(self.results)],
            capture_output=True, text=True, env=environment)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return process, calls

    def test_two_fresh_models_then_comparison_reuse_features_and_inherit_gpu_one(self):
        process, calls = self.run_round()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([call['stage'] for call in calls], ['ordinary_gru', 'history_validity', 'comparison'])
        self.assertEqual([call['script'] for call in calls],
            ['train_state_supervised.py', 'train_state_supervised.py', 'compare_state_supervised.py'])
        self.assertTrue(all(call['gpu'] == '1' for call in calls))
        self.assertIn('R4 state-supervised round completed.', process.stdout)
        for call in calls[:2]:
            args = call['args']
            for key, value in {'--train-feature-cache': self.train_cache, '--test-feature-cache': self.test_cache,
                '--epochs': 5, '--seed': 0, '--lambda-state': .1, '--lambda-time': .1,
                '--response-frames': 3, '--device': 'cuda:0'}.items():
                self.assertEqual(args[args.index(key)+1], str(value))
            self.assertTrue(args[args.index('--plan')+1].endswith('cscwd/r4_state_supervision_plan.md'))
            self.assertNotIn('--resume', args)
        args = calls[-1]['args']
        self.assertEqual(args[args.index('--previous-comparison')+1], str(self.results / 'r4_official_comparison.json'))
        self.assertTrue((self.results / 'r4_state_supervised_comparison.json').exists())

    def check_failure(self, stage, expected):
        process, calls = self.run_round(stage)
        self.assertEqual(process.returncode, 23)
        self.assertEqual([call['stage'] for call in calls], expected)
        self.assertNotIn('R4 state-supervised round completed.', process.stdout)

    def test_first_model_failure_stops_the_round(self):
        self.check_failure('ordinary_gru', ['ordinary_gru'])

    def test_second_model_failure_stops_comparison(self):
        self.check_failure('history_validity', ['ordinary_gru', 'history_validity'])

    def test_comparison_failure_is_not_reported_as_success(self):
        self.check_failure('comparison', ['ordinary_gru', 'history_validity', 'comparison'])


if __name__ == '__main__':
    unittest.main()
