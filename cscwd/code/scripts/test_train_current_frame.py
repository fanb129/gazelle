"""Actual synthetic D4 training, history-free checkpoint reuse and comparison CLI."""

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

import test_compare_official_joint as fixtures
from compare_official_joint import compare_runs
import train_current_frame as training
from stag.cache import file_sha256, load_cache, write_json
from stag.current_frame_model import CONTROL_VERSION, MODEL_ID, CurrentFrameRefiner
from stag.official_data import official_training_inputs
from stag.official_protocol import PROTOCOL_VERSION, TRAINING_BUDGET, select_checkpoint


SCRIPT = Path(__file__).with_name('train_current_frame.py')


class CurrentFrameTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixtures.OfficialComparisonTests.setUpClass()
        cls.addClassCleanup(fixtures.OfficialComparisonTests.doClassCleanups)
        cls.fixture = fixtures.OfficialComparisonTests
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.plan = cls.root / 'plan.md'
        cls.plan.write_text(f'# Synthetic D4 plan\n\n{PROTOCOL_VERSION}\n{CONTROL_VERSION}\n')
        cls.reference = cls.root / 'reference_comparison.json'
        write_json(cls.reference, compare_runs(cls.fixture.reports, cls.fixture.plan))
        source_paths = [path for path in (cls.fixture.root / 'sources').rglob('*') if path.is_file()]
        cls.original_sources = {str(path): file_sha256(path) for path in source_paths}
        cls.report_path, cls.summary = cls.root / 'current_frame.json', cls.root / 'current_frame.md'
        cls.provenance = cls.root / 'provenance.md'
        command = [sys.executable, str(SCRIPT)]
        for field in training.INPUT_NAMES:
            command.extend(('--' + field.replace('_', '-'), str(cls.fixture.fixture[field])))
        for field, value in TRAINING_BUDGET.items():
            command.extend(('--' + field.replace('_', '-'), str(value)))
        command.extend(('--plan', str(cls.plan), '--output-dir', str(cls.root / 'current_frame'),
            '--device', 'cpu', '--report', str(cls.report_path), '--summary', str(cls.summary),
            '--provenance', str(cls.provenance)))
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        if process.returncode:
            raise AssertionError(process.stdout + process.stderr)
        cls.report = json.loads(cls.report_path.read_text())

    def inputs(self):
        return official_training_inputs(*(self.fixture.fixture[name] for name in training.INPUT_NAMES))

    def test_fresh_control_keeps_all_epochs_full_test_and_original_loss(self):
        report = self.report
        original = json.loads(self.fixture.reports[0].read_text())
        self.assertTrue(report['passed'])
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(report['model_id'], MODEL_ID)
        self.assertEqual(report['control_version'], CONTROL_VERSION)
        self.assertEqual(report['config']['control_version'], CONTROL_VERSION)
        self.assertEqual(report['shared_initialization_sha256'], original['shared_initialization_sha256'])
        self.assertEqual(report['parameter_count'], original['parameter_count'])
        self.assertFalse(report['config']['has_validation_split'])
        self.assertTrue(report['test_used_for_checkpoint_selection'])
        self.assertFalse(report['independent_held_out_test_estimate'])
        self.assertEqual([item['epoch'] for item in report['history']], [1, 2, 3, 4, 5])
        for item in report['history']:
            train, test = item['training'], item['test']
            self.assertEqual(train['valid_frames'], 24)
            self.assertEqual(train['inframe_frames'], 18)
            self.assertNotIn('stable_loss', train)
            self.assertNotIn('transition_loss', train)
            self.assertAlmostEqual(train['total_loss'], train['heatmap_loss'] + .1 * train['state_loss'])
            self.assertEqual(test['standard']['person_frames'], 24)
            self.assertEqual(test['evaluation_split'], 'test')
            self.assertIsNone(test['diagnostics'])
            self.assertEqual(item['checkpoint']['sha256'], file_sha256(item['checkpoint']['path']))
        winner = select_checkpoint(report['candidates'])
        self.assertEqual(report['selection']['epoch'], winner['epoch'])
        self.assertEqual(report['selection']['checkpoint']['sha256'],
                         report['history'][winner['epoch']-1]['checkpoint']['sha256'])
        self.assertEqual(report['last_checkpoint']['sha256'], report['history'][-1]['checkpoint']['sha256'])
        self.assertEqual(self.original_sources, {path: file_sha256(path) for path in self.original_sources})

    def test_checkpoint_reload_matches_saved_test_and_never_returns_history(self):
        report = self.report
        item = report['history'][-1]
        checkpoint = torch.load(report['last_checkpoint']['path'], weights_only=True, map_location='cpu')
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['control_version'], CONTROL_VERSION)
        model = CurrentFrameRefiner(4, 128)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        test = self.inputs()['test']
        _, saved = load_cache(item['test']['cache']['path'])
        track = test['tracks'][0]
        frame = track['frames'][0]
        feature_position = test['feature_index']['image_paths'].index(frame['path'])
        inputs = (
            torch.tensor(test['features'][feature_position:feature_position+1]),
            torch.tensor(np.array(test['arrays']['heatmap'][track['start']:track['start']+1])),
            torch.tensor(np.array(test['arrays']['inout'][track['start']:track['start']+1])),
            torch.tensor([frame['bbox_norm']]))
        contaminated = {'hidden': torch.ones(1, 128), 'heatmap': torch.ones(1, 64, 64),
                        'inout': torch.ones(1)}
        with torch.no_grad():
            heatmap, score, state = model.step(*inputs, state=contaminated)
        self.assertIsNone(state)
        np.testing.assert_allclose(heatmap[0].numpy(), saved['heatmap'][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(score[0].numpy(), saved['inout'][0], atol=1e-6, rtol=1e-5)

    def test_test_labels_do_not_change_any_of_five_trained_checkpoints(self):
        loaded = self.inputs()
        changed = {**loaded, 'test': {**loaded['test'], 'tracks': copy.deepcopy(loaded['test']['tracks'])}}
        for track in changed['test']['tracks']:
            for frame in track['frames']:
                frame['inout'] = 1 - frame['inout']
                frame['gazex_norm'] = [.75] if frame['inout'] else [-1.0]
                frame['gazey_norm'] = [.25] if frame['inout'] else [-1.0]
        with patch.object(training, 'official_training_inputs', return_value=changed):
            report = training.run_current_frame_training(
                *(self.fixture.fixture[name] for name in training.INPUT_NAMES), self.plan,
                self.root / 'changed_test_labels', device='cpu')
        for first, second in zip(self.report['history'], report['history']):
            a = torch.load(first['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            b = torch.load(second['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))

    def test_fixed_budget_and_plan_are_enforced_before_creating_run(self):
        inputs = tuple(self.fixture.fixture[name] for name in training.INPUT_NAMES)
        for settings in ({'epochs': 1}, {'lambda_state': .2}, {'seed': 1}):
            with self.subTest(settings=settings):
                destination = self.root / ('rejected_' + next(iter(settings)))
                with self.assertRaisesRegex(ValueError, 'budget'):
                    training.run_current_frame_training(*inputs, self.plan, destination,
                        device='cpu', **settings)
                self.assertFalse(destination.exists())
        wrong_plan = self.root / 'wrong_plan.md'
        wrong_plan.write_text(PROTOCOL_VERSION)
        with self.assertRaisesRegex(ValueError, 'protocol|control|version'):
            training.run_current_frame_training(*inputs, wrong_plan, self.root / 'wrong_plan', device='cpu')

    def test_actual_comparison_cli_retains_control_rows_and_reference_selection(self):
        report, summary, manifest = (self.root / name for name in ('comparison.json', 'comparison.md', 'selection.json'))
        command = [sys.executable, str(SCRIPT.with_name('compare_current_frame.py')),
            '--run-report', str(self.report_path), '--plan', str(self.plan),
            '--reference-comparison', str(self.reference), '--report', str(report), '--summary', str(summary),
            '--selection-manifest', str(manifest), '--provenance', str(self.provenance)]
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(report.read_text())
        selected = json.loads(manifest.read_text())
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['control_version'], CONTROL_VERSION)
        self.assertEqual(selected['control_version'], CONTROL_VERSION)
        self.assertEqual(selected['comparison_report']['sha256'], file_sha256(report))
        self.assertIn(file_sha256(report), self.provenance.read_text())
        self.assertIn('current_frame', summary.read_text())
        for path, digest in result['code_sha256'].items():
            self.assertEqual(digest, file_sha256(training.REPO_ROOT / path))

    def test_training_cli_reports_source_hashes_and_original_loss_summary(self):
        self.assertIn(file_sha256(self.report_path), self.provenance.read_text())
        self.assertIn('not an independent held-out', self.summary.read_text())
        self.assertEqual(self.report['runtime']['device_name'], 'CPU')
        for source, digest in self.report['code_sha256'].items():
            self.assertEqual(digest, file_sha256(training.REPO_ROOT / source))


if __name__ == '__main__':
    unittest.main()
