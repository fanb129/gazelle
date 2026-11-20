"""Synthetic official split training, per-epoch test and whole-epoch selection."""

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

from test_official_data import INPUT_NAMES, make_official_fixture
from train_compact_joint import build_model
from train_official_joint import run_official_training
from stag.cache import file_sha256, load_cache
from stag.metrics import evaluate_predictions
from stag.official_data import official_training_inputs
from stag.official_evaluation import evaluate_official_model
from stag.official_protocol import PROTOCOL_VERSION, TRAINING_BUDGET, select_checkpoint


SCRIPT = Path(__file__).with_name("train_official_joint.py")


class OfficialTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = make_official_fixture(self.root / "fixture")
        self.plan = self.root / "plan.md"
        self.plan.write_text(f"# Synthetic train/test plan\n\nVersion: {PROTOCOL_VERSION}\n")

    def inputs(self):
        return official_training_inputs(*(self.fixture[name] for name in INPUT_NAMES))

    def train(self, model_id="ordinary_gru", name="run", **kwargs):
        return run_official_training(*(self.fixture[field] for field in INPUT_NAMES),
            self.plan, self.root / name, model_id=model_id, device="cpu", **kwargs)

    def test_full_official_train_and_test_every_epoch_both_models_and_sources_unchanged(self):
        paths = [path for path in (self.root / "fixture").rglob('*') if path.is_file()]
        before = {str(path): file_sha256(path) for path in paths}
        reports = [self.train(model_id, model_id) for model_id in ("ordinary_gru", "history_validity")]
        self.assertEqual(reports[0]["shared_initialization_sha256"], reports[1]["shared_initialization_sha256"])
        self.assertLess(reports[0]["parameter_count"], reports[1]["parameter_count"])
        loaded = self.inputs()
        expected = evaluate_predictions(loaded['test']['tracks'], loaded['test']['arrays']['heatmap'],
                                        loaded['test']['arrays']['inout'])
        for report in reports:
            self.assertTrue(report['passed'])
            self.assertTrue(all(report['checks'].values()))
            self.assertEqual(report['coverage']['train']['person_frames'], 24)
            self.assertEqual(report['coverage']['test']['person_frames'], 24)
            self.assertEqual(report['baseline_test']['standard'], expected)
            self.assertTrue(report['test_used_for_checkpoint_selection'])
            self.assertFalse(report['independent_held_out_test_estimate'])
            self.assertFalse(report['config']['has_validation_split'])
            self.assertEqual([h['epoch'] for h in report['history']], [1, 2, 3, 4, 5])
            self.assertNotIn('validation', report)
            for h in report['history']:
                self.assertEqual(h['training']['valid_frames'], 24)
                self.assertEqual(h['test']['standard']['person_frames'], 24)
                self.assertNotIn('validation', h)
                self.assertEqual(h['test']['evaluation_split'], 'test')
                index, arrays = load_cache(h['test']['cache']['path'])
                self.assertEqual(index['task'], 'R4_official_test_evaluation')
                self.assertEqual(index['evaluation_split'], 'test')
                self.assertTrue(index['checks']['all_test_frames_written_once'])
                self.assertTrue(all(t['track_id'].startswith('test/') for t in index['tracks']))
                self.assertEqual(index['metadata']['evaluation_split'], 'test')
                self.assertTrue(index['metadata']['test_used_for_checkpoint_selection'])
                self.assertEqual(file_sha256(Path(h['test']['cache']['path']) / 'index.json'),
                                 h['test']['cache']['index_sha256'])
                self.assertTrue(np.isfinite(arrays['heatmap']).all())
                self.assertEqual(Path(h['test']['details']['path']).name, 'test_details.json')
                self.assertEqual(h['checkpoint']['sha256'], file_sha256(h['checkpoint']['path']))
                if report['model_id'] == 'ordinary_gru':
                    self.assertIsNone(h['test']['diagnostics'])
                else:
                    self.assertTrue(h['test']['diagnostics']['passed'])
            self.assertEqual(report['selection']['epoch'], select_checkpoint(report['candidates'])['epoch'])
        self.assertEqual(before, {str(path): file_sha256(path) for path in paths})

    def test_auc_selects_one_whole_epoch_without_precision_or_stability_gate(self):
        aucs = iter((.71, .9, .8, .75, .7))
        stability = iter((.95, .1, .8, .9, .9))
        def controlled(*args, **kwargs):
            result = evaluate_official_model(*args, **kwargs)
            result['standard'].update(auc=next(aucs), l2=.4, inout_ap=.5)
            result['temporal']['stability']['accurate_stable_ratio'] = next(stability)
            return result
        with patch('train_official_joint.evaluate_official_model', side_effect=controlled):
            report = self.train()
        self.assertEqual(report['selection']['epoch'], 2)
        self.assertNotIn('precision_eligible', report['candidates'][1])
        self.assertEqual(report['candidates'][1]['metrics']['accurate_stable_ratio'], .1)
        selected = torch.load(report['selection']['checkpoint']['path'], weights_only=True, map_location='cpu')
        last = torch.load(report['last_checkpoint']['path'], weights_only=True, map_location='cpu')
        self.assertEqual(selected['epoch'], 2)
        self.assertEqual(last['epoch'], 5)
        self.assertEqual(report['selection']['checkpoint']['sha256'], report['history'][1]['checkpoint']['sha256'])

    def test_official_test_forward_does_not_change_parameters_gradients_or_source_arrays(self):
        loaded = self.inputs()
        test = loaded['test']
        model = build_model('history_validity', 4, 128, .1)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        gradients = {key: value.grad.clone() for key, value in model.named_parameters()}
        evaluate_official_model(model, test['tracks'], test['arrays'], test['feature_index'],
            test['features'], self.root / 'evaluation', 24, 'cpu', 16, 8,
            {'protocol_version': PROTOCOL_VERSION})
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        for key, value in model.named_parameters():
            self.assertTrue(torch.equal(gradients[key], value.grad))

    def test_changed_test_labels_do_not_change_training_parameters(self):
        first = self.train(name='first')
        loaded = self.inputs()
        changed = {**loaded, 'test': {**loaded['test'], 'tracks': copy.deepcopy(loaded['test']['tracks'])}}
        for track in changed['test']['tracks']:
            for frame in track['frames']:
                if frame['inout']:
                    frame['gazex_norm'], frame['gazey_norm'] = [.75], [.25]
        with patch('train_official_joint.official_training_inputs', return_value=changed):
            second = self.train(name='changed_test_labels')
        for first_epoch, second_epoch in zip(first['history'], second['history']):
            a = torch.load(first_epoch['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            b = torch.load(second_epoch['checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))

    def test_budget_or_plan_mismatch_does_not_start_training(self):
        with self.assertRaisesRegex(ValueError, 'budget mismatch'):
            self.train(epochs=1)
        self.assertFalse((self.root / 'run').exists())
        self.plan.write_text('# Different plan\n')
        with self.assertRaisesRegex(ValueError, 'protocol version'):
            self.train()
        self.assertFalse((self.root / 'run').exists())

    def test_cli_keeps_all_epoch_test_rows_and_checkpoint_reload_matches_saved_test_prediction(self):
        report_path, summary, provenance = (self.root / name for name in ('report.json', 'summary.md', 'provenance.md'))
        provenance.write_text('# Synthetic official train/test provenance\n')
        command = [sys.executable, str(SCRIPT), '--model-id', 'history_validity']
        for field in INPUT_NAMES:
            command.extend(('--' + field.replace('_', '-'), str(self.fixture[field])))
        for field, value in TRAINING_BUDGET.items():
            command.extend(('--' + field.replace('_', '-'), str(value)))
        command.extend(('--plan', str(self.plan), '--output-dir', str(self.root / 'cli'),
            '--device', 'cpu', '--report', str(report_path), '--summary', str(summary),
            '--provenance', str(provenance)))
        process = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('R4 official training completed', process.stdout)
        report = json.loads(report_path.read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(report['runtime']['device_name'], 'CPU')
        self.assertIn('not an independent held-out test estimate', summary.read_text())
        self.assertIn(file_sha256(report_path), provenance.read_text())
        for name, digest in report['code_sha256'].items():
            self.assertEqual(digest, file_sha256(Path(__file__).resolve().parents[3] / name))
        checkpoint = torch.load(report['last_checkpoint']['path'], weights_only=True, map_location='cpu')
        model = build_model('history_validity', 4, 128, .1)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        loaded = self.inputs()['test']
        prediction_index, arrays = load_cache(report['history'][-1]['test']['cache']['path'])
        track = loaded['tracks'][0]
        frame = track['frames'][0]
        feature_position = loaded['feature_index']['image_paths'].index(frame['path'])
        with torch.no_grad():
            heatmap, score, _ = model.step(
                torch.tensor(loaded['features'][feature_position:feature_position+1]),
                torch.tensor(np.array(loaded['arrays']['heatmap'][track['start']:track['start']+1])),
                torch.tensor(np.array(loaded['arrays']['inout'][track['start']:track['start']+1])),
                torch.tensor([frame['bbox_norm']]))
        self.assertEqual(prediction_index['person_frames'], 24)
        np.testing.assert_allclose(heatmap[0].numpy(), arrays['heatmap'][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(score[0].numpy(), arrays['inout'][0], atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
