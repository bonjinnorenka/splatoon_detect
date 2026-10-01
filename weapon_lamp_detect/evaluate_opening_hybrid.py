"""Opening-only hybrid templates, tuned with whole-training-match exclusion."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_alignment import predict_rows, sha
from weapon_lamp_detect.evaluate_improved_opening import attach_quality, decisions
from weapon_lamp_detect.evaluate_opening import source_protocol
from weapon_lamp_detect.evaluate_poc import prediction, summarize
from weapon_lamp_detect.hybrid_opening_matcher import CONFIGURATIONS, HybridOpeningMatcher, fuse, hybrid_bank
from weapon_lamp_detect.match_data import atomic_json, now, read_json
from weapon_lamp_detect.region_matcher import OBSRegionMatcher


def hashes():
    return {name:sha(Path(__file__).parent/name) for name in ('hybrid_opening_matcher.py','evaluate_opening_hybrid.py',
            'opening_template_bank.py','opening_matcher.py','evaluate_improved_opening.py',
            'region_matcher.py','evaluate_opening.py','evaluate_alignment.py')}


def fold(job):
    bank,dataset,match_id=job;cv2.setNumThreads(1)
    metadata,rows=load_dataset(dataset)
    names={e['weapon'] for e in bank['templates'] if all(f['match_id']!=match_id for f in e['source_frames'])}
    checks=[r for r in rows if r['match_id']==match_id and r['weapon_label'] in names and r['opening_frame_ordinal'] in (3,4)]
    if not checks:return []
    matcher=HybridOpeningMatcher(bank,excluded_matches=(match_id,))
    raw=[]
    for row in checks:
        crop=sample_crop(dataset,row,metadata);components=matcher.components(crop,row['side'])
        sample={**row,'training_original_ordinal':row['opening_frame_ordinal'],'opening_frame_ordinal':row['opening_frame_ordinal']-3}
        for name in CONFIGURATIONS:
            candidates=[c.to_dict() for c in fuse(components,name,len(bank['weapons']))]
            raw.append(prediction(sample,candidates,f'obs_hybrid_{name}_foreground','training_leave_match_out'))
    return attach_quality(raw,dataset,metadata)


def freeze(source,dataset,output,workers=8):
    source,dataset,output=map(lambda p:Path(p).resolve(),(source,dataset,output))
    if output.exists():raise ValueError('新しいoutputを指定してください')
    metadata,_=load_dataset(dataset);verify_sources(metadata)
    if metadata.get('opening_window_seconds')!=5 or metadata.get('sampling_interval')!=1:
        raise ValueError('5秒/1秒間隔の開始datasetが必要です')
    bank=hybrid_bank(source,dataset,training_only=True)
    raw=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(fold,[(bank,str(dataset),m) for m in bank['train_match_ids']])):
            raw.extend(result);print('hybrid fold',i+1,'/',len(bank['train_match_ids']),'rows',len(result),flush=True)
    if not raw:raise ValueError('training checksがありません')
    summaries={}
    for name in CONFIGURATIONS:
        for policy in ('quality','mean','best_quality'):
            summaries[f'{name}:{policy}']=summarize(decisions([r for r in raw if r['method']==f'obs_hybrid_{name}_foreground'],2,policy))
    def rank(key):
        s=summaries[key]
        return float(np.mean([v['exact_top1']['accuracy'] for v in s['per_weapon'].values()])),s['metrics']['exact_top1']['accuracy']
    selected=max(summaries,key=rank)
    template_hashes={kind:sha(source_protocol(source,kind)[0]/'templates.json') for kind in ('independent','full_training')}
    frozen={'created_at':now(),'source_experiment':str(source),'dataset':str(dataset),
            'dataset_sha256':dataset_digest(dataset),'protocol_sha256':sha(source/'protocol.json'),
            'source_template_manifest_sha256':template_hashes,'algorithm_sha256':hashes(),
            'selected_configuration':selected.split(':')[0],'selected_aggregation':selected.split(':')[1],
            'training_bank':bank,'settings':CONFIGURATIONS,'training_results':summaries,
            'training_sample_ids':sorted({r['sample_id'] for r in raw}),
            'selection':'LOMO macro then micro on training-only +3/+4-second images; opening +0/+1 and original sources of other training matches only; quality preferred on exact tie',
            'caveat':'Previously observed recordings and iterated experiments; no untouched-new-recording claim. LOMO tuning excludes unavailable weapons; test denominator never excludes them.'}
    output.mkdir()
    (output/'training_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in raw),encoding='utf-8')
    atomic_json(output/'configuration_frozen.json',frozen)
    for key,s in summaries.items():print('hybrid training',key,rank(key),s['metrics']['exact_top1'],flush=True)
    print('selected',selected,flush=True);return frozen


def run(output,kind='independent',workers=8):
    output=Path(output).resolve();frozen=read_json(output/'configuration_frozen.json')
    source=Path(frozen['source_experiment']);dataset=Path(frozen['dataset'])
    if hashes()!=frozen['algorithm_sha256'] or dataset_digest(dataset)!=frozen['dataset_sha256'] or sha(source/'protocol.json')!=frozen['protocol_sha256']:
        raise ValueError('固定後に実装/dataset/protocolが変更されています')
    for mode,expected in frozen['source_template_manifest_sha256'].items():
        if sha(source_protocol(source,mode)[0]/'templates.json')!=expected:raise ValueError('template manifest変更')
    directory=output/kind
    if directory.exists():raise ValueError('既存outputを上書きしません')
    templates,original,train,test,reserved,source_pairs=source_protocol(source,kind)
    bank=hybrid_bank(source,dataset,kind)
    reserved|={tuple(k) for k in bank['template_frame_keys']}
    all_pairs={(f['match_id'],e['weapon']) for e in bank['templates'] for f in e['source_frames']}
    metadata,rows=load_dataset(dataset);verify_sources(metadata)
    selected=[r for r in rows if r['match_id'] in test and r['weapon_label'] in bank['weapons']]
    if {r['sample_id'] for r in selected}&set(frozen['training_sample_ids']):raise ValueError('training/test overlap')
    scope={r['sample_id']:('within_match_other_timestamp' if (r['match_id'],r['weapon_label']) in source_pairs|all_pairs else 'cross_match') for r in selected}
    manifest=copy.deepcopy(original);manifest['split']['evaluation_scope']=scope
    configuration=frozen['selected_configuration'];policy=frozen['selected_aggregation']
    method=f'obs_hybrid_{configuration}_foreground'
    matchers={'obs_hybrid_baseline_foreground':OBSRegionMatcher(templates,True),method:HybridOpeningMatcher(bank,configuration)}
    usable=[r for r in selected if (r['video_id'],r['frame_index']) not in reserved]
    raw=predict_rows(usable,dataset,metadata,manifest,matchers,workers)
    for row in selected:
        if (row['video_id'],row['frame_index']) in reserved:
            for name in matchers:raw.append(prediction({**row,'reserved_template_frame':True},[],name,scope[row['sample_id']]))
    attach_quality(raw,dataset,metadata)
    result=[]
    for name in matchers:
        for aggregation in dict.fromkeys(('mean','quality',policy)):
            for count in (1,3,5):result.extend(decisions([r for r in raw if r['method']==name],count,aggregation))
    default=f'obs_hybrid_{configuration}_{policy}_5frame_foreground';methods=sorted({r['method'] for r in result})
    report={'created_at':now(),'dataset':str(dataset),'candidate_universe':bank['weapons'],'configuration':frozen,
            'evaluation_kind':kind,'viewer_default_method':default,'preprocessors':{name:'legacy' for name in methods},
            'secondary_preprocessors':{name:'local_core' for name in methods if not name.startswith('obs_hybrid_baseline_')},
            'methods':{name:{'single_frame':summarize([r for r in result if r['method']==name])} for name in methods},
            'raw_frame_methods':{name:summarize([r for r in raw if r['method']==name]) for name in matchers},
            'provenance':{'hybrid_templates_manifest':bank,'legacy_templates_manifest':original,'actual_inferred_template_frame_overlap':0,
                          'reserved_unavailable_samples':len(selected)-len(usable)},
            'runtime_policy':'infer first5seconds once and hold; no later frames','scope_note':frozen['caveat']}
    directory.mkdir()
    for name,contents in [('frame_predictions.jsonl',raw),('predictions.jsonl',result)]:
        (directory/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in contents),encoding='utf-8')
    atomic_json(directory/'report.json',report);atomic_json(directory/'templates.json',bank)
    held=defaultdict(dict)
    for r in result:
        if r['method']==default:held[r['match_id']][f"{r['side']}{r['slot_index']}"]={k:r[k] for k in ('predicted','confidence','margin','top_k','decision_timestamp','aggregation_weights')}
    atomic_json(directory/'match_predictions.json',{'source':'model prediction, NOT ground truth','hold_for_match':True,'method':default,'matches':dict(held)})
    analysis={}
    for video in sorted({r['video_id'] for r in result}):
        analysis[video]={}
        for name in methods:
            s=summarize([r for r in result if r['video_id']==video and r['method']==name and r['scope']=='cross_match'])
            analysis[video][name]=s;print(kind,video,name,s['metrics']['exact_top1'],flush=True)
    atomic_json(directory/'analysis.json',analysis);return report


def main():
    p=argparse.ArgumentParser(description='開始/従来見本＋局所コントラスト照合、設定選択はLOMO trainingのみ')
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('source',type=Path);f.add_argument('dataset',type=Path);f.add_argument('--output',type=Path,required=True);f.add_argument('--workers',type=int,default=8)
    r=commands.add_parser('run');r.add_argument('output',type=Path);r.add_argument('--kind',choices=['independent','full_training'],default='independent');r.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.source,a.dataset,a.output,a.workers)
    else:run(a.output,a.kind,a.workers)


if __name__=='__main__':main()
