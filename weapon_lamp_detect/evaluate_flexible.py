"""Second ablation: keep original geometry and avoid side-restricted exemplars."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.detail_matcher import DETAIL_CONFIG, DetailMatcher
from weapon_lamp_detect.evaluate_alignment import paired, predict_rows, sha
from weapon_lamp_detect.evaluate_poc import aggregate, summarize, validate_split
from weapon_lamp_detect.flexible_matcher import CONFIGURATIONS, FlexibleMatcher
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json
from weapon_lamp_detect.region_matcher import OBSRegionMatcher


def algorithms():
    return {name:sha(Path(__file__).parent/name) for name in
            ('evaluate_flexible.py','flexible_matcher.py','detail_matcher.py','region_matcher.py','evaluate_alignment.py')}


def freeze(experiment,output,workers=4):
    experiment,output=Path(experiment).resolve(),Path(output).resolve()
    if (output/'configuration_frozen.json').exists():
        raise ValueError('固定済み設定は上書きしません')
    protocol=read_json(experiment/'protocol.json')
    directory=experiment/'augmented_templates'
    manifest=read_json(directory/'templates.json')
    dataset=Path(manifest['dataset'])
    metadata,rows=load_dataset(dataset)
    evaluation=validate_split(manifest,rows,dataset)
    train=set(protocol['old_train_match_ids']+protocol['new_train_match_ids'])
    used={i for e in manifest['templates'] for i in e['sample_ids']}
    groups=defaultdict(list)
    for row in rows:
        if row['match_id'] in train and row['state']=='alive' and row['sample_id'] not in used and row['weapon_label'] in manifest['weapons']:
            groups[row['match_id'],row['side'],row['slot_index']].append(row)
    checks=[r for group in groups.values() for r in sorted(group,key=lambda r:r['frame_index'])[:3]]
    if not checks or {r['sample_id'] for r in checks}&evaluation:
        raise ValueError('training checkが空またはevaluationと重複')
    matchers={'baseline':OBSRegionMatcher(directory,True),
              **{name:FlexibleMatcher(directory,name) for name in CONFIGURATIONS},
              'detail_fixed':DetailMatcher(directory)}
    results=predict_rows(checks,dataset,metadata,manifest,matchers,workers)
    summaries={name:summarize([r for r in results if r['method']==name]) for name in matchers}
    def rank(name):
        s=summaries[name]
        macro=float(np.mean([r['exact_top1']['accuracy'] for r in s['per_weapon'].values()]))
        return macro,s['metrics']['exact_top1']['accuracy']
    selected=max(CONFIGURATIONS,key=rank)
    tied=[name for name in CONFIGURATIONS if rank(name)==rank(selected)]
    recommended=max(matchers,key=rank)
    result={'created_at':now(),'experiment':str(experiment),'dataset':str(dataset),
            'dataset_sha256':dataset_digest(dataset),'source_protocol_sha256':sha(experiment/'protocol.json'),
            'algorithm_sha256':algorithms(),'selected_configuration':selected,'tied_configurations':tied,'training_recommendation':recommended,
            'configurations':CONFIGURATIONS,'detail_configuration':DETAIL_CONFIG,
            'training_check_sample_ids':[r['sample_id'] for r in checks],'training_only_results':summaries,
            'training_match_ids':sorted(train),'evaluation_labels_used_for_configuration_selection':False,
            'selection':'macro top-1 then micro on training only; compare selected flexible variant plus fixed-crop detail; no test-based parameter selection',
            'training_scope_note':'Same training match, different crop/timestamp from template source. Other players in a template frame may appear in tuning checks; these are not independent evaluation results.',
            'iteration_note':'Exploratory follow-up after an unsuccessful ink-envelope registration experiment; previously observed evaluation sets are not a fresh final test.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'training_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    atomic_json(output/'configuration_frozen.json',result)
    print('frozen',selected,'recommended on training',recommended,{n:(rank(n),s['metrics']['exact_top1']) for n,s in summaries.items()},flush=True)
    return result


def run(output,kind='independent',workers=4):
    output=Path(output).resolve()
    frozen=read_json(output/'configuration_frozen.json')
    if frozen['algorithm_sha256']!=algorithms() or frozen['dataset_sha256']!=dataset_digest(frozen['dataset']):
        raise ValueError('固定後にコード/datasetが変更されています')
    experiment=Path(frozen['experiment'])
    if frozen['source_protocol_sha256']!=sha(experiment/'protocol.json'):
        raise ValueError('固定後に分割が変更されています')
    if kind=='full_training':
        experiment=experiment/'full_additional_training'
    elif kind!='independent':
        raise ValueError('unknown kind')
    directory=output/kind
    if (directory/'report.json').exists():
        raise ValueError('既存reportは上書きしません')
    templates=experiment/'augmented_templates'
    manifest=read_json(templates/'templates.json')
    dataset=Path(manifest['dataset'])
    metadata,rows=load_dataset(dataset)
    verify_sources(metadata)
    selected=validate_split(manifest,rows,dataset)
    if selected&set(frozen['training_check_sample_ids']):
        raise ValueError('training checkと評価が重複')
    eval_rows=[r for r in rows if r['sample_id'] in selected]
    matchers={'obs_flexible_foreground':FlexibleMatcher(templates,frozen['selected_configuration'],directory/'processed_templates/obs_flexible_foreground'),
              'obs_detail_fixed_foreground':DetailMatcher(templates,False,directory/'processed_templates/obs_detail_fixed_foreground')}
    for configuration in frozen['tied_configurations']:
        if configuration!=frozen['selected_configuration']:
            name=f'obs_{configuration}_foreground'
            matchers[name]=FlexibleMatcher(templates,configuration,directory/'processed_templates'/name)
    results=predict_rows(eval_rows,dataset,metadata,manifest,matchers,workers)
    baseline_report=read_json(experiment/'augmented_evaluation/report.json')
    baseline=[json.loads(l) for l in (experiment/'augmented_evaluation/predictions.jsonl').read_text().splitlines()]
    if baseline_report['candidate_universe']!=manifest['weapons']:
        raise ValueError('候補集合が変更されています')
    changes={name:paired(baseline,[r for r in results if r['method']==name]) for name in matchers}
    for row in baseline:
        row['method']='obs_baseline_foreground'
    results=baseline+results
    methods=sorted({r['method'] for r in results})
    summaries={name:{'single_frame':summarize([r for r in results if r['method']==name]),
                     'multi_frame':aggregate([r for r in results if r['method']==name])} for name in methods}
    primary={name:summarize([r for r in results if r['method']==name and r['state']=='alive' and r['scope']=='cross_match']) for name in methods}
    report={'created_at':now(),'dataset':str(dataset),'dataset_sha256':dataset_digest(dataset),
            'candidate_universe':manifest['weapons'],'methods':summaries,'paired_alive_cross_match':primary,
            'configuration':frozen,'evaluation_kind':kind,'paired_changes_all_states':changes,
            'provenance':{name:{'templates_manifest':manifest,'preprocessor':'detail' if 'detail' in name else 'legacy_foreground'} for name in matchers},
            'baseline_report':str(experiment/'augmented_evaluation/report.json'),
            'note':'Original crop pixels and geometry, sample IDs, labels, state, scope and source frames are unchanged. Bounded scale/shift matching is label-blind. No pair-specific rules.'}
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in results),encoding='utf-8')
    atomic_json(directory/'report.json',report)
    summarize_run(directory,report,results)
    return report


def summarize_run(directory,report,rows):
    names=sorted(report['methods']);videos=sorted({r['video_id'] for r in rows})
    catalog=Catalog();display=lambda w:catalog.entries.get(w,{}).get('display_name',w)
    def pct(value):
        return '—' if value is None else f'{value*100:.2f}%'
    analysis={};lines=['# 固定crop＋サイズ探索・side制限の比較','',f"条件: {report['evaluation_kind']}。候補58武器固定、alive/cross-match。",
                      '','| video | 方式 | support | single top-1 | top-3 | top-5 | 3frame top-1 / slot support |',
                      '|---|---|---:|---:|---:|---:|---:|']
    for video in videos:
        selected=[r for r in rows if r['video_id']==video and r['state']=='alive' and r['scope']=='cross_match']
        analysis[video]={}
        for name in names:
            subset=[r for r in selected if r['method']==name]
            single=summarize(subset);multi=aggregate(subset,strategies=('mean_score',))['mean_score']
            analysis[video][name]={'single_frame':single,'multi_frame':multi}
            m=single['metrics'];a=multi['3']['common_cohort']['metrics']
            lines.append(f"| {video} | {name} | {m['support']} | {pct(m['exact_top1']['accuracy'])} | {pct(m['exact_top3']['accuracy'])} | {pct(m['exact_top5']['accuracy'])} | {pct(a['exact_top1']['accuracy'])} / {a['support']} |")
            print(video,name,m['exact_top1'],a['exact_top1'],flush=True)
        ordered=['obs_baseline_foreground']+[name for name in names if name!='obs_baseline_foreground']
        lines+=['',f'## 全武器: {video}','','| 武器 | support | '+' | '.join(ordered)+' |','|---|---:|'+''.join('---:|' for _ in ordered)]
        for weapon,b in analysis[video]['obs_baseline_foreground']['single_frame']['per_weapon'].items():
            vals=[analysis[video][n]['single_frame']['per_weapon'][weapon]['exact_top1']['accuracy'] for n in ordered]
            lines.append(f"| {display(weapon)} | {b['support']} | {' | '.join(pct(v) for v in vals)} |")
    lines+=['','設定はtraining側で固定。既に観察した評価集合での探索的比較。正解ラベル・crop・state・候補・分割の変更、失敗標本の除外は行わない。',
            '1/3/5/10frameは同じ10frame以上の共通cohortで比較。武器ごとに悪化していないかreport.json/analysis.jsonを確認する。']
    atomic_json(directory/'analysis.json',analysis)
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='元cropを保持したscale/side制限のtraining固定・同一標本比較')
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('experiment',type=Path);f.add_argument('--output',type=Path,required=True);f.add_argument('--workers',type=int,default=4)
    r=commands.add_parser('run');r.add_argument('output',type=Path);r.add_argument('--kind',choices=['independent','full_training'],default='independent');r.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='freeze':
        freeze(a.experiment,a.output,a.workers)
    else:
        run(a.output,a.kind,a.workers)


if __name__=='__main__':
    main()
