"""Decompose saved test predictions and summarize history controls without inference."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'cscwd/code'))

from compare_compact_joint import artifact, verify_artifact
from compare_official_joint import verify_selected_test
from stag.cache import load_cache, file_sha256, write_json
from stag.cache_diagnosis import diagnose_prediction_pair
from stag.history_diagnosis import summarize_history_controls
from stag.metrics import TemporalConfig, temporal_protocol
from stag.official_data import load_official_predictions
from stag.official_protocol import PROTOCOL_VERSION, select_checkpoint

DIAGNOSIS_VERSION = 'r4_cache_diagnosis_v1'


def align_prediction_cache(source_index, index, arrays):
    """Map selected-cache rows to source track rows without moving large heatmaps."""
    source = {track['track_id']: track for track in source_index['tracks']}
    if (len(source) != len(index['tracks'])
            or {t['track_id'] for t in index['tracks']} != set(source)):
        raise ValueError('Selected prediction tracks differ from the source test cache')
    count = source_index['person_frames']
    xy = np.empty((count, 2), dtype=np.float32)
    scores = np.empty(count, dtype=np.float32)
    seen = np.zeros(count, dtype=bool)
    for track in index['tracks']:
        original = source[track['track_id']]
        if (track['frames'] != original['frames']
                or track['source_global_start'] != original['start']
                or track['source_global_stop'] != original['stop']):
            raise ValueError('Selected test frame identities or source slices differ')
        destination = slice(original['start'], original['stop'])
        selection = slice(track['start'], track['stop'])
        if seen[destination].any():
            raise ValueError('Selected cache contains duplicate source tracks')
        xy[destination] = arrays['argmax_xy'][selection]
        scores[destination] = arrays['inout'][selection]
        seen[destination] = True
    if not seen.all():
        raise ValueError('Selected cache does not cover every source test frame')
    return xy, scores


def metrics_match(actual, expected):
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(metrics_match(actual[k], v) for k, v in expected.items())
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def endpoint_matches(variant, expected):
    return (metrics_match(variant['temporal'], expected['temporal'])
            and metrics_match(variant['state'], expected['state'])
            and all(metrics_match(variant['standard'][key], expected['standard'][key])
                    for key in ('l2', 'inout_ap', 'person_frames', 'localization_frames')))


def run_diagnosis(comparison_report, single_frame_report, plan, output_dir, event_radius=3):
    if DIAGNOSIS_VERSION not in Path(plan).read_text():
        raise ValueError('Plan does not contain the cache diagnosis protocol version')
    comparison = json.loads(Path(comparison_report).read_text())
    if (comparison['task'] != 'R4_official_comparison' or not comparison['passed']
            or not all(comparison['checks'].values())
            or comparison['protocol_version'] != PROTOCOL_VERSION):
        raise ValueError('Diagnosis requires the completed official comparison report')
    base_run, base_index, raw = load_official_predictions(single_frame_report, 'test')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = TemporalConfig()
    models, checks, inputs = {}, {}, {}
    for model_id in ('ordinary_gru', 'history_validity'):
        run_path = verify_artifact(comparison['source_reports'][model_id])
        run = json.loads(run_path.read_text())
        winner = select_checkpoint(run['candidates'])
        if (winner != comparison['selected'][model_id]
                or run['selection'] != comparison['per_model_selection'][model_id]
                or winner['epoch'] != run['selection']['epoch']
                or run['sources']['test_prediction_report_sha256'] != file_sha256(single_frame_report)):
            raise ValueError('Selected epoch or source test predictions differ from the comparison')
        item = next(h for h in run['history'] if h['epoch'] == winner['epoch'])
        verify_artifact(run['selection']['checkpoint'])
        if run['selection']['checkpoint']['sha256'] != item['checkpoint']['sha256']:
            raise ValueError('Selected checkpoint SHA differs from its epoch record')
        verify_selected_test(run, item, base_run['cache'])
        index, arrays = load_cache(item['test']['cache']['path'])
        xy, scores = align_prediction_cache(base_index, index, arrays)
        print(f'Decomposing selected test cache: {model_id}, epoch {winner["epoch"]}', flush=True)
        result = diagnose_prediction_pair(base_index['tracks'], raw['argmax_xy'], raw['inout'], xy, scores, config)
        variants = result['variants']
        checks[model_id + ':raw_endpoint_reproduced'] = endpoint_matches(
            variants['raw_xy_raw_score'], comparison['baseline_test'])
        checks[model_id + ':selected_endpoint_reproduced'] = endpoint_matches(
            variants['refined_xy_refined_score'], item['test'])
        details_path = output_dir / f'{model_id}_cross_details.json'
        write_json(details_path, result.pop('details'))
        result.update({'epoch': winner['epoch'], 'selected_checkpoint': run['selection']['checkpoint'],
                       'details': artifact(details_path)})
        if model_id == 'history_validity':
            record = item['test']['diagnostics']
            directory = Path(record['path'])
            diagnostic_index = json.loads((directory / 'index.json').read_text())
            diagnostic_arrays = {name: np.load(directory / value['path'], mmap_mode='r', allow_pickle=False)
                                 for name, value in record['files'].items()}
            print('Summarizing saved history controls and correspondence', flush=True)
            result['history_controls'] = summarize_history_controls(
                base_index['tracks'], raw['argmax_xy'], raw['inout'], xy, scores,
                diagnostic_index, diagnostic_arrays, config, event_radius)
        models[model_id] = result
        inputs[model_id] = {'run_report': artifact(run_path), 'prediction_cache': item['test']['cache'],
                            'diagnostics': item['test']['diagnostics']}
    if not all(checks.values()):
        raise ValueError(f'Cached endpoint metrics do not reproduce their source reports: {checks}')
    return {'task': 'R4_official_cache_diagnosis', 'diagnosis_version': DIAGNOSIS_VERSION,
            'protocol_version': PROTOCOL_VERSION, 'passed': True,
            'source_reports': {'comparison': artifact(comparison_report),
                               'single_frame': artifact(single_frame_report), 'plan': artifact(plan)},
            'coverage': {k: base_run['cache'][k] for k in ('tracks', 'person_frames', 'unique_image_frames')},
            'temporal_protocol': temporal_protocol(config), 'event_radius': event_radius,
            'models': models, 'inputs': inputs, 'checks': checks,
            'execution': 'CPU_only; saved_predictions_and_controls; no_inference_or_training',
            'interpretation': 'output_cross_combination_diagnosis; not_retrained_ablations_or_causal_mechanism_proof',
            'next_decision': 'Separate coordinate/state contributions and inspect history controls before choosing a training adjustment.'}


def summary_markdown(report):
    lines = ['# 官方 test 缓存贡献与历史控制诊断', '',
             f'- 版本：`{report["diagnosis_version"]}`；仅使用已保存预测，不训练或重新前向。',
             '- 四种交叉组合用于输出诊断；不作为重新训练的单输出消融。',
             '- L2/AP由坐标和分数计分；本任务不重新计算热图AUC。', '',
             '| 模型 | 坐标 | 状态分数 | L2 | AP | 准确稳定 | 状态错误 | gaze_shift响应 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    def value(x):
        return 'null' if x is None else f'{x:.6f}'
    for model_id, result in report['models'].items():
        for variant in result['variants'].values():
            numbers = [variant['standard']['l2'], variant['standard']['inout_ap'],
                       variant['temporal']['stability']['accurate_stable_ratio'],
                       variant['state']['state_error']['error_rate'],
                       variant['temporal']['response']['gaze_shift']['response_rate']]
            lines.append(f'| {model_id} | {variant["coordinate_source"]} | {variant["state_source"]} | '
                         + ' | '.join(map(value, numbers)) + ' |')
        lines.extend(('', f'## {model_id} 的窗口分解', '',
                      '```json', json.dumps(result['attribution'], ensure_ascii=False, indent=2), '```', ''))
    lines.extend(('## 历史控制量', '',
                  '| 离线分组 | 帧数 | 历史可用帧 | 有历史时 validity均值 | 有历史时低/高validity占比 | refresh均值 |',
                  '|---|---:|---:|---:|---|---:|'))
    groups = report['models']['history_validity']['history_controls']['groups']
    for name, group in groups.items():
        available = group['validity_on_available_history']
        fractions = available['gate_fractions']
        low_high = value(fractions['low_fraction']) + ' / ' + value(fractions['high_fraction'])
        lines.append(f'| {name} | {group["frames"]} | {group["history_available_frames"]} | '
                     + value(available['distribution']['mean']) + ' | ' + low_high + ' | '
                     + value(group['controls']['refresh']['mean']) + ' |')
    lines.extend(('', '低/高validity阈值固定为0.1/0.9，分母只含历史可用帧；refresh均值按该组全部帧汇总。',
                  '完整分布、对应距离和修正幅度关联见JSON的 `models.history_validity.history_controls`。',
                  '事件邻域使用标签仅作离线分组，可重叠；控制量相关性不当作历史机制的因果证据。',
                  '', '下一步依据窗口分解与门控统计决定一次针对性调整；所有源缓存保持只读。'))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-report', 'single-frame-report', 'plan', 'output-dir', 'report', 'summary', 'provenance'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--event-radius', type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_diagnosis(args.comparison_report, args.single_frame_report, args.plan,
                           args.output_dir, args.event_radius)
    paths = [Path(__file__), *(Path(__file__).with_name(name + '.py') for name in
             ('compare_official_joint', 'compare_compact_joint', 'eval_temporal')),
             *(REPO_ROOT / f'cscwd/code/stag/{name}.py' for name in
             ('cache_diagnosis', 'history_diagnosis', 'history_diagnostics', 'cache', 'metrics', 'state_metrics',
              'official_data', 'feature_cache', 'official_protocol', 'compact_protocol'))]
    report.update({'created_at_utc': datetime.now(timezone.utc).isoformat(),
                   'git_commit': subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                   'command': shlex.join([sys.executable, *sys.argv]),
                   'code_sha256': {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths},
                   'elapsed_seconds': time.perf_counter() - started,
                   'runtime': {'device': 'CPU', 'numpy': np.__version__, 'scikit_learn': sklearn.__version__}})
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(report))
    with args.provenance.open('a') as stream:
        stream.write(f'\n## Official test cache diagnosis\n\n'
                     f'- UTC: `{report["created_at_utc"]}`; commit: `{report["git_commit"]}`.\n'
                     f'- Command: `{report["command"]}`.\n'
                     f'- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n'
                     f'- Summary: `{args.summary}`; SHA256: `{file_sha256(args.summary)}`.\n')
        for model_id, result in report['models'].items():
            stream.write(f'- {model_id} cross details: `{result["details"]["path"]}`; SHA256: `{result["details"]["sha256"]}`.\n')
    print(f'Official cache diagnosis completed; report: {args.report}', flush=True)


if __name__ == '__main__':
    main()
