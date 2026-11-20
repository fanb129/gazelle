"""Selected official caches flow through a NumPy-only diagnostic CLI."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import test_compare_official_joint as fixtures
from compare_official_joint import compare_runs
from diagnose_official_cache import DIAGNOSIS_VERSION, align_prediction_cache, run_diagnosis
from stag.cache import file_sha256, load_cache, write_json


class OfficialCacheDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfficialComparisonTests.setUpClass()
        cls.addClassCleanup(fixtures.OfficialComparisonTests.doClassCleanups)
        cls.fixture = fixtures.OfficialComparisonTests
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.comparison = cls.root / 'comparison.json'
        write_json(cls.comparison, compare_runs(cls.fixture.reports, cls.fixture.plan))
        cls.plan = cls.root / 'diagnosis.md'
        cls.plan.write_text(DIAGNOSIS_VERSION)
        cls.raw_report = cls.fixture.fixture['test_prediction_report']

    def diagnose(self, comparison=None):
        return run_diagnosis(comparison or self.comparison, self.raw_report,
                             self.plan, self.root / self._testMethodName)

    def test_full_cross_diagnosis_reproduces_both_endpoints_and_preserves_all_sources(self):
        paths = [self.raw_report, *self.fixture.reports]
        runs = [json.loads(p.read_text()) for p in self.fixture.reports]
        for run in runs:
            selected = run['history'][run['selection']['epoch']-1]['test']
            paths.extend(Path(selected['cache']['path']).iterdir())
        paths = [p for p in paths if p.is_file()]
        before = {str(p): file_sha256(p) for p in paths}
        report = self.diagnose()
        self.assertTrue(report['passed'])
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(set(report['models']), {'ordinary_gru', 'history_validity'})
        for result in report['models'].values():
            self.assertEqual(set(result['variants']), {'raw_xy_raw_score', 'refined_xy_raw_score',
                             'raw_xy_refined_score', 'refined_xy_refined_score'})
            self.assertTrue(Path(result['details']['path']).is_file())
            self.assertNotIn('auc', result['variants']['raw_xy_raw_score']['standard'])
        self.assertIn('history_controls', report['models']['history_validity'])
        self.assertEqual(before, {str(p): file_sha256(p) for p in paths})

    def test_prediction_alignment_supports_different_packed_track_order(self):
        source_report = json.loads(self.raw_report.read_text())
        source, arrays = load_cache(source_report['cache']['path'])
        index = copy.deepcopy(source)
        xy, scores, start = [], [], 0
        index['tracks'] = list(reversed(index['tracks']))
        for track in index['tracks']:
            first, stop = track['start'], track['stop']
            xy.extend(arrays['argmax_xy'][first:stop])
            scores.extend(arrays['inout'][first:stop])
            track.update(source_global_start=first, source_global_stop=stop,
                         start=start, stop=start + stop-first)
            start = track['stop']
        aligned_xy, aligned_scores = align_prediction_cache(source, index,
            {'argmax_xy': np.asarray(xy), 'inout': np.asarray(scores)})
        np.testing.assert_array_equal(aligned_xy, arrays['argmax_xy'])
        np.testing.assert_array_equal(aligned_scores, arrays['inout'])
        index['tracks'][0]['frames'][0]['frame_index'] += 1
        with self.assertRaisesRegex(ValueError, 'frame identities'):
            align_prediction_cache(source, index, {'argmax_xy': np.asarray(xy), 'inout': np.asarray(scores)})

    def test_changed_report_identity_or_selected_epoch_is_rejected(self):
        report = json.loads(self.comparison.read_text())
        report['source_reports']['ordinary_gru']['sha256'] = 'changed'
        altered = self.root / 'changed_source.json'
        write_json(altered, report)
        with self.assertRaisesRegex(ValueError, 'Artifact SHA256'):
            self.diagnose(altered)
        report = json.loads(self.comparison.read_text())
        report['selected']['ordinary_gru']['epoch'] = 999
        altered = self.root / 'changed_selection.json'
        write_json(altered, report)
        with self.assertRaisesRegex(ValueError, 'Selected epoch'):
            run_diagnosis(altered, self.raw_report, self.plan, self.root / 'changed_epoch')

    def test_cli_runs_with_torch_import_disabled_and_records_report_artifacts(self):
        directory = self.root / 'import_guard'
        directory.mkdir()
        (directory / 'sitecustomize.py').write_text('''import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("Torch import is disabled for cache diagnosis")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
''')
        report, summary, provenance = (self.root / name for name in ('result.json', 'result.md', 'provenance.md'))
        command = [sys.executable, str(Path(__file__).with_name('diagnose_official_cache.py')),
            '--comparison-report', str(self.comparison), '--single-frame-report', str(self.raw_report),
            '--plan', str(self.plan), '--output-dir', str(self.root / 'cli_details'),
            '--event-radius', '3', '--report', str(report), '--summary', str(summary),
            '--provenance', str(provenance)]
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': str(directory), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('Official cache diagnosis completed', process.stdout)
        result = json.loads(report.read_text())
        self.assertTrue(result['passed'])
        self.assertEqual(result['runtime']['device'], 'CPU')
        self.assertIn(file_sha256(report), provenance.read_text())
        self.assertIn('准确稳定', summary.read_text())
        for path, digest in result['code_sha256'].items():
            self.assertEqual(digest, file_sha256(Path(__file__).resolve().parents[3] / path))


if __name__ == '__main__':
    unittest.main()
