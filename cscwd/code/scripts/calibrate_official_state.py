"""Fit train-only state biases for fixed R4 checkpoints and score saved test outputs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'cscwd/code'))

from compare_compact_joint import artifact, paired_details, verify_artifact
from compare_official_joint import verify_selected_test
from diagnose_official_cache import align_prediction_cache, endpoint_matches
from train_compact_joint import build_model
from stag.cache import file_sha256, load_cache, write_json
from stag.cache_diagnosis import _prediction_changes
from stag.calibration_rollout import cache_frozen_train_scores
from stag.metrics import TemporalConfig, evaluate_temporal, temporal_protocol
from stag.official_data import _load_split, load_official_predictions
from stag.official_protocol import PROTOCOL_VERSION, select_checkpoint
from stag.state_calibration import (CALIBRATION_VERSION, apply_state_bias,
                                    fit_state_bias, ranking_checks)
from stag.state_metrics import evaluate_state_temporal


def score_outputs(tracks, xy, scores, retained_auc):
    """Recompute score/coordinate metrics; retain AUC from the identical heatmaps."""
    frames = [frame for track in tracks for frame in track['frames']]
    labels = np.asarray([frame['inout'] for frame in frames])
    gt = np.asarray([[frame['gazex_norm'][0], frame['gazey_norm'][0]] for frame in frames])
    inside = labels == 1
    temporal = evaluate_temporal(tracks, xy, scores, TemporalConfig())
    state = evaluate_state_temporal(tracks, scores, TemporalConfig())
    result = {'standard': {
        'auc': retained_auc,
        'l2': float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[inside] - gt[inside], axis=1).mean()),
        'inout_ap': float(average_precision_score(labels, scores)),
        'person_frames': len(frames), 'localization_frames': int(inside.sum())},
        'temporal': temporal['summary'], 'state': state['summary']}
    details = {'joint': {key: temporal[key] for key in ('tracks', 'events')},
               'state': {key: state[key] for key in ('tracks', 'events')}}
    return result, details


def fixed_runs(comparison, train_prediction_report, test_prediction_report, train_feature_report):
    """Pin the already-selected epoch, source identities and forward implementation."""
    if (comparison['task'] != 'R4_official_comparison' or not comparison['passed']
            or not all(comparison['checks'].values())
            or comparison['protocol_version'] != PROTOCOL_VERSION):
        raise ValueError('Calibration requires the completed official comparison report')
    runs = {}
    for model_id in ('ordinary_gru', 'history_validity'):
        run = json.loads(verify_artifact(comparison['source_reports'][model_id]).read_text())
        winner = select_checkpoint(run['candidates'])
        if (not run['passed'] or not all(run['checks'].values())
                or run['model_id'] != model_id or run['protocol_version'] != PROTOCOL_VERSION
                or winner != comparison['selected'][model_id]
                or run['selection'] != comparison['per_model_selection'][model_id]
                or winner['epoch'] != 5 or run['selection']['epoch'] != 5):
            raise ValueError('D2 requires the fixed, previously selected epoch 5')
        for name, path in (('train_prediction', train_prediction_report),
                           ('test_prediction', test_prediction_report), ('train_feature', train_feature_report)):
            if run['sources'][name + '_report_sha256'] != file_sha256(path):
                raise ValueError('Calibration inputs differ from the fixed checkpoint sources')
        for name, digest in run['code_sha256'].items():
            if file_sha256(REPO_ROOT / name) != digest:
                raise ValueError(f'Frozen forward source code SHA256 mismatch: {name}')
        item = next(item for item in run['history'] if item['epoch'] == 5)
        verify_artifact(run['selection']['checkpoint'])
        if run['selection']['checkpoint']['sha256'] != item['checkpoint']['sha256']:
            raise ValueError('Selected checkpoint differs from epoch 5')
        runs[model_id] = (run, item)
    return runs


def frozen_model(run, feature_contract, device):
    checkpoint = torch.load(run['selection']['checkpoint']['path'], map_location='cpu', weights_only=True)
    if (checkpoint['epoch'] != 5 or checkpoint['model_id'] != run['model_id']
            or checkpoint['config'] != run['config'] or checkpoint['sources'] != run['sources']
            or checkpoint['feature_contract'] != feature_contract
            or checkpoint['plan_sha256'] != run['plan']['sha256']
            or checkpoint['protocol_version'] != PROTOCOL_VERSION):
        raise ValueError('Frozen checkpoint metadata differs from its training report')
    config = run['config']
    model = build_model(run['model_id'], feature_contract['channels'],
                        config['hidden_dim'], config['temperature'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device)


def run_calibration(comparison_report, train_prediction_report, test_prediction_report,
                    train_feature_cache, train_feature_report, plan, output_dir, device='cuda:0'):
    if CALIBRATION_VERSION not in Path(plan).read_text():
        raise ValueError('Plan does not contain the state calibration protocol version')
    comparison = json.loads(Path(comparison_report).read_text())
    runs = fixed_runs(comparison, train_prediction_report, test_prediction_report, train_feature_report)
    train, train_run, train_features = _load_split(
        'train', train_feature_cache, train_feature_report, train_prediction_report)
    test_run, test_index, raw_test = load_official_predictions(test_prediction_report, 'test')
    if any(train_run[key] != test_run[key] for key in ('model', 'checkpoint', 'backbone_checkpoint', 'protocol')):
        raise ValueError('Train/test base model identities differ')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    train_labels = np.asarray([frame['inout'] for track in train['tracks'] for frame in track['frames']])
    test_tracks = test_index['tracks']
    test_labels = np.asarray([frame['inout'] for track in test_tracks for frame in track['frames']])
    models, details_by_model, checks = {}, {}, {}
    for model_id in ('single_frame', 'ordinary_gru', 'history_validity'):
        directory = output_dir / model_id
        directory.mkdir()
        if model_id == 'single_frame':
            train_scores = train['arrays']['inout']
            train_cache = {'origin': 'existing_single_frame_train_cache',
                           'source_report': artifact(train_prediction_report),
                           'index': artifact(Path(train_run['cache']['path']) / 'index.json'),
                           'inout': artifact(Path(train_run['cache']['path']) / 'inout.npy')}
            xy, test_scores = raw_test['argmax_xy'], raw_test['inout']
            expected = comparison['baseline_test']
            checkpoint = None
            test_cache = {'source_report': artifact(test_prediction_report), 'cache': test_run['cache']}
        else:
            run, item = runs[model_id]
            verify_selected_test(run, item, test_run['cache'])
            index, arrays = load_cache(item['test']['cache']['path'])
            xy, test_scores = align_prediction_cache(test_index, index, arrays)
            expected, checkpoint = item['test'], run['selection']['checkpoint']
            test_cache = {'source_report': comparison['source_reports'][model_id], 'cache': item['test']['cache']}
            print(f'Frozen train rollout: {model_id}, epoch 5', flush=True)
            model = frozen_model(run, train_features['contract'], device)
            config = run['config']
            train_cache = cache_frozen_train_scores(model, train, directory / 'train_scores', device,
                config['clip_frames'], config['batch_size'], metadata={
                    'calibration_version': CALIBRATION_VERSION, 'model_id': model_id,
                    'epoch': 5, 'checkpoint': checkpoint,
                    'prediction_report': artifact(train_prediction_report),
                    'prediction_index': artifact(Path(train_run['cache']['path']) / 'index.json'),
                    'feature_report': artifact(train_feature_report), 'feature_contract': train_features['contract'],
                    'clip_frames': config['clip_frames'], 'batch_size': config['batch_size']})
            train_scores = np.load(Path(train_cache['path']) / 'inout.npy', mmap_mode='r', allow_pickle=False)
            del model
        print(f'Fitting train-only scalar state bias: {model_id}', flush=True)
        fit = fit_state_bias(train_scores, train_labels)
        calibrated_train = apply_state_bias(train_scores, fit['bias'])
        calibrated_test = apply_state_bias(test_scores, fit['bias'])
        train_rank = ranking_checks(train_scores, calibrated_train, train_labels)
        test_rank = ranking_checks(test_scores, calibrated_test, test_labels)
        before, before_details = score_outputs(test_tracks, xy, test_scores, expected['standard']['auc'])
        after, after_details = score_outputs(test_tracks, xy, calibrated_test, expected['standard']['auc'])
        # Calibrated scores are output artifacts, never inputs to a recurrent step.
        score_path = directory / 'test_inout_calibrated.npy'
        np.save(score_path, calibrated_test, allow_pickle=False)
        calibration_path = directory / 'calibration.json'
        write_json(calibration_path, {'version': CALIBRATION_VERSION, 'model_id': model_id,
            'epoch': None if model_id == 'single_frame' else 5, 'fit_split': 'train',
            'fit': fit, 'train_scores': train_cache, 'application': 'final_output_only; no_history_feedback'})
        details_path = directory / 'test_details.json'
        write_json(details_path, {'before': before_details, 'after': after_details})
        model_checks = {
            **{'train:' + key: value for key, value in train_rank['checks'].items()},
            **{'test:' + key: value for key, value in test_rank['checks'].items()},
            'train_variable_nll_did_not_increase': (
                fit['train_after']['variable_nll'] <= fit['train_before']['variable_nll'] + 1e-12),
            'original_test_endpoint_reproduced': endpoint_matches(before, expected),
            'test_auc_retained_from_unchanged_heatmaps': after['standard']['auc'] == before['standard']['auc'],
            'test_l2_unchanged': after['standard']['l2'] == before['standard']['l2'],
            'test_ap_unchanged': after['standard']['inout_ap'] == before['standard']['inout_ap'],
        }
        models[model_id] = {'epoch': None if model_id == 'single_frame' else 5,
            'checkpoint': checkpoint, 'fit': fit, 'calibration': artifact(calibration_path),
            'train_scores': train_cache, 'test_source': test_cache,
            'train_ranking': train_rank, 'test_ranking': test_rank,
            'before': before, 'after': after,
            'calibrated_test_scores': {**artifact(score_path), 'dtype': 'float64',
                                       'row_order': 'original_official_test_prediction_index'},
            'details': artifact(details_path),
            'paired_before_after': paired_details(model_id + '_before', model_id + '_after',
                                                  before_details, after_details),
            'output_changes': _prediction_changes(np.arange(len(test_labels)), test_labels,
                                                  xy, test_scores, xy, calibrated_test),
            'checks': model_checks}
        details_by_model[model_id] = after_details
        checks.update({model_id + ':' + key: value for key, value in model_checks.items()})
    if not all(checks.values()):
        raise ValueError(f'State calibration endpoint or invariant check failed: {checks}')
    return {'task': 'R4_state_calibration', 'calibration_version': CALIBRATION_VERSION,
        'protocol_version': PROTOCOL_VERSION, 'passed': True, 'stage_status': 'awaiting_result_review',
        'source_reports': {name: artifact(path) for name, path in (
            ('comparison', comparison_report), ('train_prediction', train_prediction_report),
            ('test_prediction', test_prediction_report), ('train_feature', train_feature_report), ('plan', plan))},
        'coverage': {split: {key: run['cache'][key] for key in ('tracks', 'person_frames', 'unique_image_frames')}
                     for split, run in (('train', train_run), ('test', test_run))},
        'models': models, 'checks': checks, 'temporal_protocol': temporal_protocol(TemporalConfig()),
        'paired_calibrated': {model_id + '_vs_single_frame': paired_details('single_frame_after', model_id + '_after',
            details_by_model['single_frame'], details_by_model[model_id])
            for model_id in ('ordinary_gru', 'history_validity')},
        'fit_split': 'train', 'test_used_for_calibration': False,
        'test_used_for_checkpoint_selection': True, 'independent_held_out_test_estimate': False,
        'auc_accounting': 'retained_source_AUC; heatmaps_identical_and_not_recomputed',
        'history': 'frozen_original_rollout; uncalibrated_own_predictions; no_calibrated_score_feedback',
        'interpretation': 'global_state_operating_point_control; not_new_temporal_training_or_history_mechanism_ablation'}


def summary_markdown(report):
    lines = ['# R4-D2 train拟合的单调状态校准', '',
        '- 固定原epoch 5；全局偏置只在train拟合，校准仅作用最终输出，不反馈历史。',
        '- test阈值仍为score > 0.5；原test参与过epoch选择，本轮不重新选轮。',
        '- AUC沿用同一热图的原报告；L2与AP重算并核对不变。', '',
        '| 模型 | 输出 | bias | AUC | L2 | AP | 准确稳定 | 状态错误率 | gaze_shift响应率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def value(number):
        return 'null' if number is None else f'{number:.6f}'
    for model_id, item in report['models'].items():
        for key in ('before', 'after'):
            result = item[key]
            numbers = [0.0 if key == 'before' else item['fit']['bias'],
                *(result['standard'][name] for name in ('auc', 'l2', 'inout_ap')),
                result['temporal']['stability']['accurate_stable_ratio'], result['state']['state_error']['error_rate'],
                result['temporal']['response']['gaze_shift']['response_rate']]
            lines.append(f'| {model_id} | {key} | ' + ' | '.join(map(value, numbers)) + ' |')
    lines.extend(('', '完整JSON保留画内/画外错帧、翻转、joint/state进出响应、事件配对与train目标值。',
        '单调校准只改变运行点；不把其收益当作新的AP排序或历史机制收益。',
        '下一步审核三种方法校准后的折中，再决定一次状态输出或训练目标调整。'))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-report', 'train-prediction-report', 'test-prediction-report',
                 'train-feature-cache', 'train-feature-report', 'plan', 'output-dir', 'report', 'summary', 'provenance'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_calibration(args.comparison_report, args.train_prediction_report, args.test_prediction_report,
        args.train_feature_cache, args.train_feature_report, args.plan, args.output_dir, args.device)
    paths = [Path(__file__), *(Path(__file__).with_name(name + '.py') for name in
        ('compare_compact_joint', 'compare_official_joint', 'diagnose_official_cache', 'eval_temporal',
         'train_compact_joint', 'train_temporal_joint', 'prepare_development')),
        *(REPO_ROOT / f'cscwd/code/stag/{name}.py' for name in
        ('state_calibration', 'calibration_rollout', 'official_data', 'feature_cache', 'cache', 'cache_diagnosis',
         'metrics', 'state_metrics', 'temporal_data', 'temporal_model', 'history_model', 'history_diagnostics',
         'history_diagnosis', 'official_protocol', 'compact_protocol'))]
    report.update({'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, *sys.argv]),
        'code_sha256': {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
        'elapsed_seconds': time.perf_counter() - started,
        'runtime': {'device': args.device, 'torch': torch.__version__, 'numpy': np.__version__,
                    'scikit_learn': sklearn.__version__, 'precision': 'float32_frozen_rollout; float64_calibration'}})
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report))
    with args.provenance.open('a') as stream:
        stream.write(f'\n## R4-D2 train-only monotone state calibration\n\n'
            f'- UTC: `{report["created_at_utc"]}`; commit: `{report["git_commit"]}`.\n'
            f'- Command: `{report["command"]}`.\n')
        for name, path in (('Report', args.report), ('Summary', args.summary)):
            stream.write(f'- {name}: `{path}`; SHA256: `{file_sha256(path)}`.\n')
        for model_id, item in report['models'].items():
            for name in ('calibration', 'calibrated_test_scores', 'details'):
                record = item[name]
                stream.write(f'- {model_id} {name}: `{record["path"]}`; SHA256: `{record["sha256"]}`.\n')
            if model_id != 'single_frame':
                cache = item['train_scores']
                stream.write(f'- {model_id} frozen train scores: `{cache["path"]}`; '
                             f'index SHA256: `{cache["index_sha256"]}`; '
                             f'inout SHA256: `{cache["files"]["inout"]["sha256"]}`.\n')
    print(f'R4 state calibration completed; report: {args.report}', flush=True)


if __name__ == '__main__':
    main()
