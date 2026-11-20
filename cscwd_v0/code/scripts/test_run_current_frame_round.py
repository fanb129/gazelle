"""D4 ordering, fixed budget, quoted paths, GPU inheritance and failure propagation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).with_name('run_current_frame_round.sh')


class CurrentFrameRoundTests(unittest.TestCase):
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
stage = 'comparison' if script == 'compare_current_frame.py' else 'training'
with open(os.environ['D4_CALLS'], 'a') as stream:
    stream.write(json.dumps({'stage': stage, 'script': script, 'args': args,
                             'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
if stage == os.environ.get('D4_FAIL_STAGE'):
    sys.exit(23)
Path(args[args.index('--report') + 1]).write_text('{}\n')
''')
        self.python.chmod(0o755)

    def run_round(self, fail=None):
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'D4_CALLS': str(self.calls)}
        if fail:
            environment['D4_FAIL_STAGE'] = fail
        process = subprocess.run(['bash', str(WRAPPER), str(self.python), str(self.repository),
            str(self.train_cache), str(self.test_cache), str(self.experiments), str(self.results)],
            capture_output=True, text=True, env=environment)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return process, calls

    def test_one_fresh_model_then_comparison_reuses_features_and_inherits_gpu_one(self):
        process, calls = self.run_round()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([call['stage'] for call in calls], ['training', 'comparison'])
        self.assertEqual([call['script'] for call in calls], ['train_current_frame.py', 'compare_current_frame.py'])
        self.assertTrue(all(call['gpu'] == '1' for call in calls))
        self.assertIn('R4 current-frame round completed.', process.stdout)
        args = calls[0]['args']
        for key, value in {'--train-feature-cache': self.train_cache, '--test-feature-cache': self.test_cache,
            '--epochs': 5, '--seed': 0, '--lambda-state': .1, '--device': 'cuda:0',
            '--output-dir': self.experiments / 'current_frame'}.items():
            self.assertEqual(args[args.index(key)+1], str(value))
        self.assertTrue(args[args.index('--plan')+1].endswith('cscwd/r4_current_frame_plan.md'))
        self.assertNotIn('--model-id', args)
        self.assertNotIn('--resume', args)
        self.assertNotIn('--lambda-time', args)
        comparison = calls[1]['args']
        self.assertEqual(comparison[comparison.index('--reference-comparison')+1],
                         str(self.results / 'r4_official_comparison.json'))
        self.assertTrue((self.results / 'r4_current_frame.json').exists())
        self.assertTrue((self.results / 'r4_current_frame_comparison.json').exists())

    def test_training_failure_stops_comparison(self):
        process, calls = self.run_round('training')
        self.assertEqual(process.returncode, 23)
        self.assertEqual([call['stage'] for call in calls], ['training'])
        self.assertNotIn('R4 current-frame round completed.', process.stdout)

    def test_comparison_failure_is_not_reported_as_success(self):
        process, calls = self.run_round('comparison')
        self.assertEqual(process.returncode, 23)
        self.assertEqual([call['stage'] for call in calls], ['training', 'comparison'])
        self.assertNotIn('R4 current-frame round completed.', process.stdout)


if __name__ == '__main__':
    unittest.main()
