"""Decide once at opening and hold: evaluate match-slot decisions, not late frames."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_alignment import predict_rows, sha
from weapon_lamp_detect.evaluate_poc import prediction, summarize, validate_split
from weapon_lamp_detect.flexible_matcher import CONFIGURATIONS, FlexibleMatcher
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json
from weapon_lamp_detect.region_matcher import OBSRegionMatcher


def algorithms():
    return {name:sha(Path(__file__).parent/name) for name in
            ('build_opening_dataset.py','evaluate_opening.py','flexible_matcher.py','region_matcher.py','evaluate_alignment.py')}


def opening_decisions(rows,count):
    """First global N frames, never first N alive frames at arbitrary timestamps."""
    if count<1:
        raise ValueError('count >= 1')
    groups=defaultdict(list)
    for row in rows:
        groups[row['method'],row['match_id'],row['side'],row['slot_index'],row['scope']].append(row)
    results=[]
    for group in groups.values():
        group.sort(key=lambda r:r['opening_frame_ordinal'])
        if len({r['opening_frame_ordinal'] for r in group})!=len(group):
            raise ValueError('opening frame ordinal重複')
        if len({r['expected'] for r in group})!=1:
            raise ValueError('同一slotの正解が変わっています')
        subset=[r for r in group if r['opening_frame_ordinal']<count]
        usable=[r for r in subset if r['all_candidates'] and not r.get('reserved_template_frame')]
        scores,classes=defaultdict(float),{}
        for row in usable:
            for candidate in row['all_candidates']:
                scores[candidate['weapon']]+=candidate['score']/len(usable)
                classes[candidate['weapon']]=candidate['weapon_class']
        ranked=[{'weapon':name,'weapon_class':classes[name],'score':score,'confidence':0.}
                for name,score in sorted(scores.items(),key=lambda p:(-p[1],p[0]))]
        margin=ranked[0]['score']-ranked[1]['score'] if len(ranked)>1 else 0.
        for candidate in ranked:
            candidate['confidence']=float(np.clip(.58*candidate['score']+2.2*margin,0,1))
        method=group[0]['method'].removesuffix('_foreground')+f'_{count}frame_foreground'
        result=prediction(group[0],ranked,method,group[0]['scope'])
        result.update({'decision_frame_count':count,'usable_frame_count':len(usable),
                       'insufficient_frames':len(subset)<count or len(usable)<count,
                       'decision_status':'inferred' if ranked else 'unreadable',
                       'decision_timestamp':max((r['timestamp'] for r in subset),default=group[0]['timestamp']),
                       'frame_indices':[r['frame_index'] for r in subset],
                       'decision_frames':[{k:r.get(k) for k in ('crop','context','rect','frame_index','timestamp','state','remaining_seconds','reserved_template_frame')}
                                          for r in subset],
                       'decision_note':'Mean scores of the first N opening frames; automatic alive/down/unknown are recorded but do not remove weapon shapes mistaken for X. No later-frame rescue.'})
        results.append(result)
    return results


def source_protocol(experiment,kind):
    experiment=Path(experiment)
    protocol=read_json(experiment/'protocol.json')
    if kind=='full_training':
        experiment=experiment/'full_additional_training'
    elif kind!='independent':
        raise ValueError('unknown kind')
    templates=experiment/'augmented_templates'
    manifest=read_json(templates/'templates.json')
    _,source_rows=load_dataset(manifest['dataset'])
    validate_split(manifest,source_rows,manifest['dataset'])
    by_id={r['sample_id']:r for r in source_rows}
    source_ids={i for entry in manifest['templates'] for i in entry['sample_ids']}
    source_frames={(by_id[i]['video_id'],by_id[i]['frame_index']) for i in manifest['split']['template_sample_ids']}
    source_match_weapon={(by_id[i]['match_id'],by_id[i]['weapon_label']) for i in source_ids}
    train=set(manifest['split']['train_match_ids'])
    all_matches={r['match_id'] for r in source_rows}
    return templates,manifest,train,all_matches-train,source_frames,source_match_weapon


def models(templates,names):
    result={}
    for name in names:
        method=f'obs_opening_{name}_foreground'
        result[method]=OBSRegionMatcher(templates,True) if name=='baseline' else FlexibleMatcher(templates,name)
    return result


def freeze(experiment,dataset,output,workers=4):
    experiment,dataset,output=map(lambda p:Path(p).resolve(),(experiment,dataset,output))
    if (output/'configuration_frozen.json').exists():
        raise ValueError('固定済み設定を上書きしません')
    templates,manifest,train,test,reserved,_=source_protocol(experiment,'independent')
    metadata,rows=load_dataset(dataset);verify_sources(metadata)
    if not metadata.get('opening_window_seconds'):
        raise ValueError('開始時点専用datasetが必要です')
    checks=[r for r in rows if r['match_id'] in train and r['weapon_label'] in manifest['weapons'] and (r['video_id'],r['frame_index']) not in reserved]
    if not checks or {r['match_id'] for r in checks}&test:
        raise ValueError('training checkが空またはtestと重複')
    candidates=['baseline',*CONFIGURATIONS]
    results=predict_rows(checks,dataset,metadata,manifest,models(templates,candidates),workers)
    training={name:summarize(opening_decisions([r for r in results if r['method']==f'obs_opening_{name}_foreground'],3)) for name in candidates}
    def rank(name):
        s=training[name]
        return float(np.mean([v['exact_top1']['accuracy'] for v in s['per_weapon'].values()])),s['metrics']['exact_top1']['accuracy']
    selected=max(candidates,key=rank)
    tied=[name for name in candidates if rank(name)==rank(selected)]
    frozen={'created_at':now(),'experiment':str(experiment),'dataset':str(dataset),'dataset_sha256':dataset_digest(dataset),
            'source_protocol_sha256':sha(experiment/'protocol.json'),'algorithm_sha256':algorithms(),
            'opening_window_seconds':metadata['opening_window_seconds'],'sampling_interval':metadata['sampling_interval'],
            'selected_configuration':selected,'tied_configurations':tied,'training_only_results':training,
            'training_check_sample_ids':[r['sample_id'] for r in checks],
            'selection':'opening-only 3-frame match-slot macro top-1 then micro; stable candidate order for ties. No late frames / test-driven configuration selection.',
            'evaluation_labels_used_for_configuration_selection':False,
            'scope_note':'Previously inspected recordings/matches. Opening images are bounded and sparse; this is not an untouched new-recording final test.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'training_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    atomic_json(output/'configuration_frozen.json',frozen)
    print('opening frozen',selected,'ties',tied,{n:(rank(n),v['metrics']['exact_top1']) for n,v in training.items()},flush=True)
    return frozen


def run(output,kind='independent',workers=4):
    output=Path(output).resolve();frozen=read_json(output/'configuration_frozen.json')
    if frozen['algorithm_sha256']!=algorithms() or frozen['dataset_sha256']!=dataset_digest(frozen['dataset']):
        raise ValueError('固定後に実装/datasetが変わりました')
    if frozen['source_protocol_sha256']!=sha(Path(frozen['experiment'])/'protocol.json'):
        raise ValueError('分割が変更されています')
    directory=output/kind
    if (directory/'report.json').exists():
        raise ValueError('既存reportは上書きしません')
    templates,manifest,train,test,reserved,source_match_weapon=source_protocol(frozen['experiment'],kind)
    dataset=Path(frozen['dataset']);metadata,rows=load_dataset(dataset)
    selected=[r for r in rows if r['match_id'] in test and r['weapon_label'] in manifest['weapons']]
    if {r['sample_id'] for r in selected}&set(frozen['training_check_sample_ids']):
        raise ValueError('training checkと評価が重複')
    check_manifest=copy.deepcopy(manifest)
    check_manifest['split']['evaluation_scope']={r['sample_id']:('within_match_other_timestamp' if (r['match_id'],r['weapon_label']) in source_match_weapon else 'cross_match') for r in selected}
    usable=[r for r in selected if (r['video_id'],r['frame_index']) not in reserved]
    omitted=[r for r in selected if (r['video_id'],r['frame_index']) in reserved]
    names=list(dict.fromkeys(['baseline',frozen['selected_configuration'],*frozen['tied_configurations']]))
    matchers=models(templates,names)
    predictions=predict_rows(usable,dataset,metadata,check_manifest,matchers,workers)
    for row in omitted:
        for method in matchers:
            predictions.append(prediction({**row,'reserved_template_frame':True},[],method,check_manifest['split']['evaluation_scope'][row['sample_id']]))
    decisions=[d for count in (1,3,5) for d in opening_decisions(predictions,count)]
    methods=sorted({r['method'] for r in decisions})
    default=f"obs_opening_{frozen['selected_configuration']}_3frame_foreground"
    report={'created_at':now(),'dataset':str(dataset),'dataset_sha256':dataset_digest(dataset),
            'candidate_universe':manifest['weapons'],'configuration':frozen,'evaluation_kind':kind,
            'evaluation_unit':'one match-slot decision, first global 1/3/5 frames in first 5 seconds of stable HUD',
            'viewer_default_method':default,'methods':{name:{'single_frame':summarize([r for r in decisions if r['method']==name])} for name in methods},
            'raw_frame_methods':{name:summarize([r for r in predictions if r['method']==name]) for name in matchers},
            'provenance':{'templates_manifest':manifest,'template_frames_reserved':len(reserved),
                          'evaluation_template_overlap_frames':len({r['frame_index'] for r in omitted}),
                          'opening_detection':metadata['opening_detection']},
            'automatic_state_note':'Recorded only, not used to drop opening slots. Bows/other shapes may be incorrectly marked down; all 8 human-labeled slots remain eligible.',
            'reserved_frames_note':'Template frames are never inferred/evaluated against themselves. Omitted frames stay as unavailable, never replaced by later gameplay.',
            'runtime_policy':'Infer opening once and hold the weapon result throughout the match; no ongoing HUD inference.',
            'scope_note':'Cross-match and within-match-other-timestamp metrics remain separate. No new-recording generalization claim.'}
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'frame_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in predictions),encoding='utf-8')
    (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in decisions),encoding='utf-8')
    atomic_json(directory/'report.json',report)
    held=defaultdict(dict)
    for row in decisions:
        if row['method']==default:
            held[row['match_id']][f"{row['side']}{row['slot_index']}"]={k:row[k] for k in ('predicted','confidence','margin','top_k','decision_timestamp','usable_frame_count')}
    atomic_json(directory/'match_predictions.json',{'source':'model prediction, NOT ground truth','method':default,'hold_for_match':True,'matches':dict(held)})
    summarize_run(directory,report,decisions)
    return report


def summarize_run(directory,report,rows):
    catalog=Catalog();display=lambda w:catalog.entries.get(w,{}).get('display_name',w)
    def pct(v):
        return '—' if v is None else f'{v*100:.2f}%'
    lines=['# 試合開始時点だけの武器識別','',f"条件: {report['evaluation_kind']} / 候補58武器固定 / HUD開始から5秒以内、1秒間隔。",
           '', '進行中の精度は評価対象外。先頭1/3/5枚で一度確定し試合中は保持する。alive/down/unknownの自動判定でslotを除外しない。',
           '', '| video | 方式 | match-slot support | top-1 | top-3 | top-5 |', '|---|---|---:|---:|---:|---:|']
    analysis={}
    for video in sorted({r['video_id'] for r in rows}):
        analysis[video]={}
        for name in sorted(report['methods']):
            subset=[r for r in rows if r['video_id']==video and r['method']==name and r['scope']=='cross_match']
            summary=summarize(subset);analysis[video][name]=summary
            m=summary['metrics']
            lines.append(f"| {video} | {name} | {m['support']} | {pct(m['exact_top1']['accuracy'])} | {pct(m['exact_top3']['accuracy'])} | {pct(m['exact_top5']['accuracy'])} |")
            print(video,name,m['exact_top1'],flush=True)
        lines+=['',f'## 武器別・開始3枚: {video}','','| 武器 | support | 方式 | top-1 |','|---|---:|---|---:|']
        for name,s in analysis[video].items():
            if '_3frame_' in name:
                for weapon,m in s['per_weapon'].items():
                    lines.append(f"| {display(weapon)} | {m['support']} | {name} | {pct(m['exact_top1']['accuracy'])} |")
    lines+=['','- 分割・候補集合・template sourceは前回のものを維持。開始時点に評価対象を変更したので、前の707cropや46slotの数字とは母集団が違う。',
            '- 同一match別timestampの結果はreport.jsonで別集計。templateそのもののsource frameは評価しない。',
            '- 自動stateは人力のalive/down正解ではない。開始時点の全slot判定を主指標とする。',
            '- 設定選択はtraining試合の開始画像のみ。過去に見た録画なので新規未観察動画の最終試験ではない。']
    atomic_json(directory/'analysis.json',analysis)
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='試合開始の先頭1/3/5枚だけで武器確定精度を評価')
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('experiment',type=Path);f.add_argument('dataset',type=Path);f.add_argument('--output',type=Path,required=True);f.add_argument('--workers',type=int,default=4)
    r=commands.add_parser('run');r.add_argument('output',type=Path);r.add_argument('--kind',choices=['independent','full_training'],default='independent');r.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='freeze':
        freeze(a.experiment,a.dataset,a.output,a.workers)
    else:
        run(a.output,a.kind,a.workers)


if __name__=='__main__':
    main()
