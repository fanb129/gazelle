"""Synthetic end-to-end train-only calibration of fixed official checkpoints."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_compare_official_joint as fixtures
import calibrate_official_state as calibration
from compare_official_joint import compare_runs
from compare_compact_joint import artifact
from stag.cache import file_sha256, load_cache, write_json
from stag.official_protocol import make_candidate, select_checkpoint
from stag.state_calibration import CALIBRATION_VERSION


class OfficialStateCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfficialComparisonTests.setUpClass()
        cls.addClassCleanup(fixtures.OfficialComparisonTests.doClassCleanups)
        cls.fixture = fixtures.OfficialComparisonTests
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        # Prescribe the fixed-epoch input contract independently of synthetic
        # training's accidental metric ranking. These AUCs are fixture metadata,
        # not measured results; D2 retains AUC and tests the score-only path.
        for path in cls.fixture.reports:
            run = json.loads(path.read_text())
            for item in run['history']:
                item['test']['standard']['auc'] = 0.8 + 0.01 * item['epoch']
            run['candidates'] = [make_candidate(run['model_id'], item['epoch'], item['test'])
                                 for item in run['history']]
            winner = select_checkpoint(run['candidates'])
            shutil.copyfile(run['history'][-1]['checkpoint']['path'], run['selection']['checkpoint']['path'])
            run['selection'] = {'status': 'selected', 'epoch': 5, 'candidate_id': winner['candidate_id'],
                                'checkpoint': artifact(run['selection']['checkpoint']['path'])}
            write_json(path, run)
        cls.comparison = cls.root / 'comparison.json'
        write_json(cls.comparison, compare_runs(cls.fixture.reports, cls.fixture.plan))
        cls.plan = cls.root / 'calibration.md'
        cls.plan.write_text(CALIBRATION_VERSION)

    def run_calibration(self, comparison=None):
        sources = self.fixture.fixture
        return calibration.run_calibration(comparison or self.comparison,
            sources['train_prediction_report'], sources['test_prediction_report'],
            sources['train_feature_cache'], sources['train_feature_report'], self.plan,
            self.root / self._testMethodName, 'cpu')

    def test_complete_flow_fits_only_train_and_preserves_sources_and_test_endpoints(self):
        sources = self.fixture.fixture
        train_run = json.loads(sources['train_prediction_report'].read_text())
        train_index, train_arrays = load_cache(train_run['cache']['path'])
        labels = np.asarray([frame['inout'] for track in train_index['tracks'] for frame in track['frames']])
        paths = [path for path in self.fixture.root.rglob('*') if path.is_file()]
        original = {path: file_sha256(path) for path in paths}
        with patch.object(calibration, 'fit_state_bias', wraps=calibration.fit_state_bias) as fitting, \
                patch.object(calibration, 'cache_frozen_train_scores', wraps=calibration.cache_frozen_train_scores) as replay:
            report = self.run_calibration()
        self.assertTrue(report['passed'])
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(set(report['models']), {'single_frame', 'ordinary_gru', 'history_validity'})
        self.assertEqual(fitting.call_count, 3)
        for call in fitting.call_args_list:
            np.testing.assert_array_equal(call.args[1], labels)
        np.testing.assert_array_equal(fitting.call_args_list[0].args[0], train_arrays['inout'])
        self.assertEqual(replay.call_count, 2)
        for call in replay.call_args_list:
            self.assertTrue(all(track['track_id'].startswith('train/') for track in call.args[1]['tracks']))
        self.assertFalse(report['test_used_for_calibration'])
        self.assertTrue(report['test_used_for_checkpoint_selection'])
        self.assertFalse(report['independent_held_out_test_estimate'])
        for model_id, result in report['models'].items():
            self.assertEqual(result['before']['standard'], result['after']['standard'])
            self.assertTrue(all(result['train_ranking']['checks'].values()))
            self.assertTrue(all(result['test_ranking']['checks'].values()))
            calibrated = np.load(result['calibrated_test_scores']['path'], allow_pickle=False)
            self.assertEqual(calibrated.dtype, np.float64)
            self.assertEqual(result['output_changes']['all']['coordinates_changed'], 0)
            self.assertEqual(result['epoch'], None if model_id == 'single_frame' else 5)
            for name in ('details', 'calibration', 'calibrated_test_scores'):
                record = result[name]
                self.assertEqual(record['sha256'], file_sha256(record['path']))
            if model_id != 'single_frame':
                cached = result['train_scores']
                self.assertTrue(all(cached['checks'].values()))
                index = json.loads((Path(cached['path']) / 'index.json').read_text())
                self.assertEqual(index['metadata']['prediction_index']['sha256'], train_run['cache']['index_sha256'])
                self.assertEqual(cached['person_frames'], len(labels))
        self.assertEqual(original, {path: file_sha256(path) for path in paths})

    def test_rank_failure_prevents_success_even_when_reported_ap_is_unchanged(self):
        original = calibration.ranking_checks
        def broken_order(*args):
            result = original(*args)
            result['checks']['strict_order_preserved'] = False
            return result
        with patch.object(calibration, 'ranking_checks', side_effect=broken_order):
            with self.assertRaisesRegex(ValueError, 'invariant check failed'):
                self.run_calibration()

    def test_changed_fixed_epoch_or_source_is_rejected_before_rollout(self):
        comparison = json.loads(self.comparison.read_text())
        comparison['selected']['ordinary_gru']['epoch'] = 4
        altered = self.root / 'changed_epoch.json'
        write_json(altered, comparison)
        with patch.object(calibration, 'cache_frozen_train_scores') as replay:
            with self.assertRaisesRegex(ValueError, 'fixed, previously selected epoch 5'):
                self.run_calibration(altered)
            replay.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'checkpoint sources'):
            calibration.fixed_runs(json.loads(self.comparison.read_text()), self.plan,
                self.fixture.fixture['test_prediction_report'], self.fixture.fixture['train_feature_report'])

    def test_cli_publishes_all_three_models_and_artifact_provenance(self):
        report, summary, provenance = (self.root / name for name in ('result.json', 'result.md', 'provenance.md'))
        sources = self.fixture.fixture
        command = [sys.executable, str(Path(calibration.__file__)),
            '--comparison-report', str(self.comparison), '--plan', str(self.plan),
            '--output-dir', str(self.root / 'cli_outputs'), '--device', 'cpu',
            '--report', str(report), '--summary', str(summary), '--provenance', str(provenance)]
        for name in ('train_prediction_report', 'test_prediction_report', 'train_feature_cache', 'train_feature_report'):
            command.extend(('--' + name.replace('_', '-'), str(sources[name])))
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('R4 state calibration completed', process.stdout)
        result = json.loads(report.read_text())
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['runtime']['device'], 'cpu')
        self.assertIn('float64_calibration', result['runtime']['precision'])
        for path, digest in result['code_sha256'].items():
            self.assertEqual(digest, file_sha256(calibration.REPO_ROOT / path))
        self.assertEqual(summary.read_text().count('| before |'), 3)
        self.assertEqual(summary.read_text().count('| after |'), 3)
        self.assertIn(file_sha256(report), provenance.read_text())
        for item in result['models'].values():
            self.assertIn(item['details']['sha256'], provenance.read_text())
            self.assertIn(item['calibrated_test_scores']['sha256'], provenance.read_text())


if __name__ == '__main__':
    unittest.main()
