"""Training-only opening ablations, paired held-out tests and washout audit."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_alignment import predict_rows, sha
from weapon_lamp_detect.evaluate_opening import opening_decisions, source_protocol
from weapon_lamp_detect.evaluate_poc import prediction, summarize
from weapon_lamp_detect.match_data import atomic_json, now, read_json
from weapon_lamp_detect.opening_matcher import CONFIGURATIONS, OpeningMatcher, quality
from weapon_lamp_detect.region_matcher import OBSRegionMatcher


def hashes():
    return {name: sha(Path(__file__).parent/name) for name in
            ('opening_matcher.py', 'evaluate_improved_opening.py', 'evaluate_opening.py',
             'region_matcher.py', 'evaluate_alignment.py', 'build_opening_dataset.py')}


def decisions(rows, count, policy="mean"):
    if policy not in ("mean", "quality", "median", "best_quality"):
        raise ValueError("invalid aggregation policy")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method'], row['match_id'], row['side'], row['slot_index'], row['scope']].append(row)
    result = []
    for group in grouped.values():
        bounded = [r for r in group if r['opening_frame_ordinal'] < count]
        usable = [r for r in bounded if r['all_candidates'] and not r.get('reserved_template_frame')]
        weights = np.ones(len(usable), dtype=float)
        if policy == 'quality':
            weights = np.array([r['image_quality']['weight'] for r in usable])
        elif policy == 'best_quality' and usable:
            weights[:] = 0
            weights[int(np.argmax([r['image_quality']['weight'] for r in usable]))] = 1
        modified = []
        all_names = {c['weapon'] for r in usable for c in r['all_candidates']}
        median = {w: float(np.median([next((c['score'] for c in r['all_candidates'] if c['weapon'] == w), 0.) for r in usable]))
                  for w in all_names} if policy == 'median' else {}
        for row in group:
            candidates = row['all_candidates']
            if row in usable:
                i = usable.index(row)
                factor = len(usable)*weights[i]/weights.sum()
                candidates = [{**c, 'score': median[c['weapon']] if policy == 'median' else c['score']*factor}
                              for c in candidates]
            modified.append({**row, 'all_candidates': candidates})
        decision = opening_decisions(modified, count)[0]
        decision['method'] = decision['method'].replace(f'_{count}frame_', f'_{policy}_{count}frame_')
        decision['aggregation_policy'] = policy
        decision['frame_quality'] = [r.get('image_quality') for r in sorted(bounded, key=lambda r: r['opening_frame_ordinal'])]
        decision['aggregation_weights'] = {r['sample_id']: float(weights[i]/weights.sum()) for i, r in enumerate(usable)}
        # Frames with near-zero quality are retained in the window/denominator.
        decision['decision_note'] = f'Opening-only {policy}; first global {count} frames, no later rescue; automatic state is not a gate.'
        result.append(decision)
    return result


def attach_quality(rows, dataset, metadata):
    cache = {}
    for row in rows:
        if row['sample_id'] not in cache:
            cache[row['sample_id']] = quality(sample_crop(dataset, row, metadata))
        row['image_quality'] = cache[row['sample_id']]
    return rows


def freeze(source, dataset, output, workers=8):
    source, dataset, output = map(lambda p: Path(p).resolve(), (source, dataset, output))
    if output.exists():
        raise ValueError('新しいoutputを指定してください')
    templates, manifest, train, test, reserved, _ = source_protocol(source, 'independent')
    metadata, rows = load_dataset(dataset)
    verify_sources(metadata)
    if metadata.get('opening_window_seconds') != 5 or metadata.get('sampling_interval') != 1:
        raise ValueError('5秒/1秒間隔の開始datasetを指定してください')
    checks = [r for r in rows if r['match_id'] in train and r['weapon_label'] in manifest['weapons']
              and (r['video_id'], r['frame_index']) not in reserved]
    assert checks and not ({r['match_id'] for r in checks} & test)
    matchers = {'obs_improved_baseline_foreground': OBSRegionMatcher(templates, True)}
    matchers.update({f'obs_improved_{name}_foreground': OpeningMatcher(templates, name) for name in CONFIGURATIONS})
    raw = attach_quality(predict_rows(checks, dataset, metadata, manifest, matchers, workers), dataset, metadata)
    summaries = {}
    for method in matchers:
        for policy in ('mean', 'quality', 'median', 'best_quality'):
            for count in (3, 5):
                decision = decisions([r for r in raw if r['method'] == method], count, policy)
                summaries[decision[0]['method']] = summarize(decision)
    # Five seconds is the maximum startup latency; choose without held-out labels.
    options = [name for name in summaries if '_5frame_' in name]
    def rank(name):
        summary = summaries[name]
        return float(np.mean([v['exact_top1']['accuracy'] for v in summary['per_weapon'].values()])), summary['metrics']['exact_top1']['accuracy']
    selected = max(options, key=rank)
    baseline = summaries['obs_improved_baseline_mean_5frame_foreground']
    output.mkdir(parents=True)
    (output/'training_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in raw), encoding='utf-8')
    frozen = {'created_at': now(), 'source_experiment': str(source), 'dataset': str(dataset),
              'dataset_sha256': dataset_digest(dataset), 'algorithm_sha256': hashes(),
              'protocol_sha256': sha(source/'protocol.json'), 'training_templates_sha256': sha(templates/'templates.json'),
              'training_sample_ids': [r['sample_id'] for r in checks], 'selected_method': selected,
              'selection': 'training-only 5-frame per-weapon macro, then micro, stable configuration order; no held-out tuning',
              'training_results': summaries, 'baseline_training': baseline,
              'configuration': CONFIGURATIONS,
              'caveat': 'Previously observed recordings; exploratory held-out matches, not an untouched new recording.'}
    atomic_json(output/'configuration_frozen.json', frozen)
    for name in options:
        print('training', name, rank(name), summaries[name]['metrics']['exact_top1'], flush=True)
    print('selected:', selected, flush=True)
    return frozen


def run(output, kind='independent', workers=8):
    output = Path(output).resolve()
    frozen = read_json(output/'configuration_frozen.json')
    if hashes() != frozen['algorithm_sha256'] or dataset_digest(frozen['dataset']) != frozen['dataset_sha256']:
        raise ValueError('固定後に実装/datasetが変更されています')
    source = Path(frozen['source_experiment'])
    if sha(source/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('分割protocolが変更されています')
    independent_templates, *_ = source_protocol(source, 'independent')
    if sha(independent_templates/'templates.json') != frozen['training_templates_sha256']:
        raise ValueError('training templateが変更されています')
    directory = output/kind
    if directory.exists():
        raise ValueError('評価outputを上書きしません')
    templates, manifest, train, test, reserved, source_match_weapon = source_protocol(source, kind)
    dataset = Path(frozen['dataset'])
    metadata, rows = load_dataset(dataset)
    verify_sources(metadata)
    selected = [r for r in rows if r['match_id'] in test and r['weapon_label'] in manifest['weapons']]
    if {r['sample_id'] for r in selected} & set(frozen['training_sample_ids']):
        raise ValueError('training/evaluation overlap')
    scope = {r['sample_id']: ('within_match_other_timestamp' if (r['match_id'], r['weapon_label']) in source_match_weapon else 'cross_match') for r in selected}
    check_manifest = copy.deepcopy(manifest)
    check_manifest['split']['evaluation_scope'] = scope
    winner = frozen['selected_method'].removeprefix('obs_improved_').removesuffix('_5frame_foreground')
    configuration, policy = winner.rsplit('_', 1) if not winner.endswith('best_quality') else (winner.removesuffix('_best_quality'), 'best_quality')
    matchers = {'obs_improved_baseline_foreground': OBSRegionMatcher(templates, True)}
    if configuration != 'baseline':
        matchers[f'obs_improved_{configuration}_foreground'] = OpeningMatcher(templates, configuration)
    usable = [r for r in selected if (r['video_id'], r['frame_index']) not in reserved]
    raw = predict_rows(usable, dataset, metadata, check_manifest, matchers, workers)
    for row in selected:
        if (row['video_id'], row['frame_index']) in reserved:
            for method in matchers:
                raw.append(prediction({**row, 'reserved_template_frame': True}, [], method, scope[row['sample_id']]))
    attach_quality(raw, dataset, metadata)
    result = []
    for method in matchers:
        for aggregation in dict.fromkeys(('mean', 'quality', policy)):
            for count in (1, 3, 5):
                result.extend(decisions([r for r in raw if r['method'] == method], count, aggregation))
    default = frozen['selected_method']
    methods = sorted({r['method'] for r in result})
    report = {'created_at': now(), 'dataset': str(dataset), 'candidate_universe': manifest['weapons'],
              'configuration': frozen, 'evaluation_kind': kind, 'viewer_default_method': default,
              'methods': {name: {'single_frame': summarize([r for r in result if r['method'] == name])} for name in methods},
              'raw_frame_methods': {name: summarize([r for r in raw if r['method'] == name]) for name in matchers},
              'provenance': {'templates_manifest': manifest, 'evaluation_template_overlap_frames': 0,
                             'reserved_unavailable_frame_count': len(selected)-len(usable)},
              'preprocessors': {name: configuration if name.startswith(f'obs_improved_{configuration}_') and configuration != 'baseline' else 'legacy' for name in methods},
              'runtime_policy': 'first 5 seconds only; infer once, hold for the match; no ongoing HUD accuracy requirement',
              'automatic_state_note': 'Not a gate: bows are sometimes automatically misclassified down.',
              'scope_note': frozen['caveat']}
    directory.mkdir()
    for name, contents in [('frame_predictions.jsonl', raw), ('predictions.jsonl', result)]:
        (directory/name).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in contents), encoding='utf-8')
    atomic_json(directory/'report.json', report)
    held = defaultdict(dict)
    for row in result:
        if row['method'] == default:
            held[row['match_id']][f"{row['side']}{row['slot_index']}"] = {k: row[k] for k in ('predicted', 'confidence', 'margin', 'top_k', 'decision_timestamp', 'aggregation_weights')}
    atomic_json(directory/'match_predictions.json', {'source': 'model predictions, NOT ground truth', 'hold_for_match': True, 'method': default, 'matches': dict(held)})
    analysis = {}
    for video in sorted({r['video_id'] for r in result}):
        analysis[video] = {}
        for name in methods:
            summary = summarize([r for r in result if r['video_id'] == video and r['method'] == name and r['scope'] == 'cross_match'])
            analysis[video][name] = summary
            print(kind, video, name, summary['metrics']['exact_top1'], flush=True)
    atomic_json(directory/'analysis.json', analysis)
    return report


def main():
    p = argparse.ArgumentParser(description='GO白飛び・背景除去の開始5秒限定比較（設定選択はtrainingのみ）')
    commands = p.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze'); f.add_argument('source', type=Path); f.add_argument('dataset', type=Path); f.add_argument('--output', type=Path, required=True); f.add_argument('--workers', type=int, default=8)
    r = commands.add_parser('run'); r.add_argument('output', type=Path); r.add_argument('--kind', choices=['independent', 'full_training'], default='independent'); r.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.command == 'freeze':
        freeze(a.source, a.dataset, a.output, a.workers)
    else:
        run(a.output, a.kind, a.workers)


if __name__ == '__main__':
    main()
