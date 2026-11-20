"""Actual synthetic D3 training, checkpoint reuse and comparison CLI checks."""

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
from train_compact_joint import build_model
import train_state_supervised as training
from stag.cache import file_sha256, load_cache, write_json
from stag.official_data import official_training_inputs
from stag.official_protocol import PROTOCOL_VERSION, TRAINING_BUDGET, select_checkpoint
from stag.state_supervision import SUPERVISION_VERSION


SCRIPT = Path(__file__).with_name('train_state_supervised.py')


class StateSupervisedTrainingTests(unittest.TestCase):
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
        cls.plan.write_text(f'# Synthetic D3 plan\n\n{PROTOCOL_VERSION}\n{SUPERVISION_VERSION}\n')
        cls.previous = cls.root / 'previous_comparison.json'
        write_json(cls.previous, compare_runs(cls.fixture.reports, cls.fixture.plan))
        source_paths = [path for path in (cls.fixture.root / 'sources').rglob('*') if path.is_file()]
        cls.original_sources = {str(path): file_sha256(path) for path in source_paths}
        cls.reports = {}
        cls.report_paths = []
        cls.summary_paths = []
        cls.provenance = cls.root / 'provenance.md'
        for model_id in ('ordinary_gru', 'history_validity'):
            report, summary = (cls.root / f'{model_id}{suffix}' for suffix in ('.json', '.md'))
            command = [sys.executable, str(SCRIPT), '--model-id', model_id]
            for field in training.INPUT_NAMES:
                command.extend(('--' + field.replace('_', '-'), str(cls.fixture.fixture[field])))
            for field, value in TRAINING_BUDGET.items():
                command.extend(('--' + field.replace('_', '-'), str(value)))
            command.extend(('--lambda-time', '0.1', '--response-frames', '3',
                '--plan', str(cls.plan), '--output-dir', str(cls.root / model_id), '--device', 'cpu',
                '--report', str(report), '--summary', str(summary), '--provenance', str(cls.provenance)))
            process = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
            cls.reports[model_id] = json.loads(report.read_text())
            cls.report_paths.append(report)
            cls.summary_paths.append(summary)

    def inputs(self):
        return official_training_inputs(*(self.fixture.fixture[name] for name in training.INPUT_NAMES))

    def test_both_fresh_models_keep_all_epochs_and_complete_supervision_counts(self):
        initializers = {report['shared_initialization_sha256'] for report in self.reports.values()}
        self.assertEqual(len(initializers), 1)
        previous = json.loads(self.fixture.reports[0].read_text())
        self.assertEqual(next(iter(initializers)), previous['shared_initialization_sha256'])
        for report in self.reports.values():
            self.assertTrue(report['passed'])
            self.assertTrue(all(report['checks'].values()))
            self.assertEqual(report['supervision_version'], SUPERVISION_VERSION)
            self.assertEqual(report['config']['supervision_version'], SUPERVISION_VERSION)
            self.assertEqual(report['config']['lambda_time'], .1)
            self.assertEqual(report['config']['response_frames'], 3)
            self.assertFalse(report['config']['has_validation_split'])
            self.assertTrue(report['test_used_for_checkpoint_selection'])
            self.assertFalse(report['independent_held_out_test_estimate'])
            self.assertEqual([item['epoch'] for item in report['history']], [1, 2, 3, 4, 5])
            for item in report['history']:
                train, test = item['training'], item['test']
                self.assertEqual(train['valid_frames'], 24)
                self.assertEqual(train['stable_pairs'], 18)
                self.assertEqual(train['transition_frames'], 12)
                self.assertTrue(np.isfinite(train['stable_loss']))
                self.assertTrue(np.isfinite(train['transition_loss']))
                self.assertAlmostEqual(train['total_loss'], train['heatmap_loss'] + .1 *
                    (train['state_loss'] + .1 * (train['stable_loss'] + train['transition_loss'])))
                self.assertEqual(test['standard']['person_frames'], 24)
                self.assertEqual(test['evaluation_split'], 'test')
                self.assertEqual(item['checkpoint']['sha256'], file_sha256(item['checkpoint']['path']))
                if report['model_id'] == 'history_validity':
                    self.assertTrue(test['diagnostics']['passed'])
                else:
                    self.assertIsNone(test['diagnostics'])
            winner = select_checkpoint(report['candidates'])
            self.assertEqual(report['selection']['epoch'], winner['epoch'])
            selected = report['history'][winner['epoch']-1]
            self.assertEqual(report['selection']['checkpoint']['sha256'], selected['checkpoint']['sha256'])
            self.assertEqual(report['last_checkpoint']['sha256'], report['history'][-1]['checkpoint']['sha256'])
        self.assertEqual(self.original_sources, {path: file_sha256(path) for path in self.original_sources})

    def test_saved_checkpoint_reloads_the_unmodified_causal_test_forward(self):
        report = self.reports['history_validity']
        item = report['history'][-1]
        checkpoint = torch.load(report['last_checkpoint']['path'], weights_only=True, map_location='cpu')
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['config']['supervision_version'], SUPERVISION_VERSION)
        model = build_model('history_validity', 4, 128, .1)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        test = self.inputs()['test']
        _, saved = load_cache(item['test']['cache']['path'])
        track = test['tracks'][0]
        frame = track['frames'][0]
        feature_position = test['feature_index']['image_paths'].index(frame['path'])
        with torch.no_grad():
            heatmap, score, _ = model.step(
                torch.tensor(test['features'][feature_position:feature_position+1]),
                torch.tensor(np.array(test['arrays']['heatmap'][track['start']:track['start']+1])),
                torch.tensor(np.array(test['arrays']['inout'][track['start']:track['start']+1])),
                torch.tensor([frame['bbox_norm']]))
        np.testing.assert_allclose(heatmap[0].numpy(), saved['heatmap'][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(score[0].numpy(), saved['inout'][0], atol=1e-6, rtol=1e-5)

    def test_test_labels_do_not_change_any_trained_epoch_weights(self):
        loaded = self.inputs()
        changed = {**loaded, 'test': {**loaded['test'], 'tracks': copy.deepcopy(loaded['test']['tracks'])}}
        for track in changed['test']['tracks']:
            for frame in track['frames']:
                frame['inout'] = 1 - frame['inout']
                frame['gazex_norm'] = [.75] if frame['inout'] else [-1.0]
                frame['gazey_norm'] = [.25] if frame['inout'] else [-1.0]
        with patch.object(training, 'official_training_inputs', return_value=changed):
            report = training.run_state_supervised_training(
                *(self.fixture.fixture[name] for name in training.INPUT_NAMES), self.plan,
                self.root / 'changed_test_labels', model_id='ordinary_gru', device='cpu')
        for first, second in zip(self.reports['ordinary_gru']['history'], report['history']):
            a = torch.load(first['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            b = torch.load(second['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))

    def test_fixed_supervision_budget_and_plan_reject_changes_before_training(self):
        inputs = tuple(self.fixture.fixture[name] for name in training.INPUT_NAMES)
        for settings in ({'epochs': 1}, {'lambda_time': .2}, {'response_frames': 4}):
            with self.subTest(settings=settings):
                destination = self.root / ('rejected_' + next(iter(settings)))
                with self.assertRaisesRegex(ValueError, 'budget|supervision|setting'):
                    training.run_state_supervised_training(*inputs, self.plan, destination,
                        model_id='ordinary_gru', device='cpu', **settings)
                self.assertFalse(destination.exists())
        wrong_plan = self.root / 'wrong_plan.md'
        wrong_plan.write_text(PROTOCOL_VERSION)
        with self.assertRaisesRegex(ValueError, 'protocol|supervision|version'):
            training.run_state_supervised_training(*inputs, wrong_plan, self.root / 'wrong_plan',
                model_id='ordinary_gru', device='cpu')

    def test_actual_comparison_cli_records_previous_outputs_and_all_test_selected_rows(self):
        report, summary, manifest = (self.root / name for name in ('comparison.json', 'comparison.md', 'selection.json'))
        command = [sys.executable, str(SCRIPT.with_name('compare_state_supervised.py')),
            '--run-report', *(str(path) for path in self.report_paths), '--plan', str(self.plan),
            '--previous-comparison', str(self.previous), '--report', str(report), '--summary', str(summary),
            '--selection-manifest', str(manifest), '--provenance', str(self.provenance)]
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(report.read_text())
        selected = json.loads(manifest.read_text())
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['supervision_version'], SUPERVISION_VERSION)
        self.assertEqual(len(result['candidate_rows']), 11)
        self.assertTrue(result['test_used_for_checkpoint_selection'])
        self.assertFalse(result['independent_test_estimate'])
        self.assertEqual(selected['supervision_version'], SUPERVISION_VERSION)
        self.assertEqual(selected['comparison_report']['sha256'], file_sha256(report))
        self.assertIn(file_sha256(report), self.provenance.read_text())
        self.assertIn('ordinary_gru', summary.read_text())
        self.assertIn('history_validity', summary.read_text())
        for path, digest in result['code_sha256'].items():
            self.assertEqual(digest, file_sha256(training.REPO_ROOT / path))

    def test_training_cli_publishes_reproducible_source_hashes_and_summaries(self):
        for path, summary, report in zip(self.report_paths, self.summary_paths, self.reports.values()):
            self.assertIn(file_sha256(path), self.provenance.read_text())
            self.assertIn('not an independent held-out', summary.read_text())
            self.assertEqual(report['runtime']['device_name'], 'CPU')
            for source, digest in report['code_sha256'].items():
                self.assertEqual(digest, file_sha256(training.REPO_ROOT / source))


if __name__ == '__main__':
    unittest.main()
