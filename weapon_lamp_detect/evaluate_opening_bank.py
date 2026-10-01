"""Choose opening template processing with leave-one-training-match-out checks."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
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
from weapon_lamp_detect.match_data import atomic_json, now, read_json
from weapon_lamp_detect.opening_template_bank import OpeningBankMatcher, make_bank
from weapon_lamp_detect.region_matcher import OBSRegionMatcher


SETTINGS={
    'legacy':{'configuration':'legacy','exemplars':False,'shared':False},
    'legacy_shared':{'configuration':'legacy','exemplars':False,'shared':True},
    'legacy_exemplars':{'configuration':'legacy','exemplars':True,'shared':False},
    'raw_core':{'configuration':'raw_core','exemplars':False,'shared':False},
    'raw_exemplars':{'configuration':'raw_core','exemplars':True,'shared':False},
    'local_exemplars':{'configuration':'local_core','exemplars':True,'shared':False},
}


def hashes():
    return {name:sha(Path(__file__).parent/name) for name in ('opening_template_bank.py','evaluate_opening_bank.py',
            'opening_matcher.py','evaluate_improved_opening.py','region_matcher.py','evaluate_opening.py','evaluate_alignment.py')}


def training_fold(job):
    bank,dataset,match_id=job
    cv2.setNumThreads(1)
    metadata,rows=load_dataset(dataset)
    names={e['weapon'] for e in bank['templates'] if all(f['match_id']!=match_id for f in e['source_frames'])}
    selected=[r for r in rows if r['match_id']==match_id and r['weapon_label'] in names and r['opening_frame_ordinal'] in (3,4)]
    if not selected:
        return []
    matchers={name:OpeningBankMatcher(bank,excluded_matches=(match_id,),**config) for name,config in SETTINGS.items()}
    result=[]
    for row in selected:
        crop=sample_crop(dataset,row,metadata)
        for name,matcher in matchers.items():
            candidates=[c.to_dict() for c in matcher.predict_crop(crop,len(bank['weapons']),row['side'])]
            # Remap only for tuning: these are the two held-out training images
            # at +3/+4sec, not runtime samples 0/1. Store original ordinal.
            r={**row,'training_original_ordinal':row['opening_frame_ordinal'],
               'opening_frame_ordinal':row['opening_frame_ordinal']-3}
            result.append(prediction(r,candidates,f'obs_bank_{name}_foreground','training_leave_match_out'))
    return attach_quality(result,dataset,metadata)


def freeze(source,dataset,output,workers=8):
    source,dataset,output=map(lambda p:Path(p).resolve(),(source,dataset,output))
    if output.exists():
        raise ValueError('新しいoutputを指定してください')
    metadata,rows=load_dataset(dataset);verify_sources(metadata)
    if metadata.get('opening_window_seconds')!=5 or metadata.get('sampling_interval')!=1:
        raise ValueError('5秒/1秒間隔の開始datasetが必要です')
    bank=make_bank(source,dataset,training_only=True)
    jobs=[(bank,str(dataset),match) for match in bank['train_match_ids']]
    raw=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(training_fold,jobs)):
            raw.extend(result)
            print('opening bank fold',i+1,'/',len(jobs),'rows',len(result),flush=True)
    if not raw:
        raise ValueError('cross-match training validationがありません')
    summaries={}
    for name in SETTINGS:
        for policy in ('mean','quality','best_quality'):
            result=decisions([r for r in raw if r['method']==f'obs_bank_{name}_foreground'],2,policy)
            summaries[f'{name}:{policy}']=summarize(result)
    def rank(key):
        s=summaries[key]
        return float(np.mean([v['exact_top1']['accuracy'] for v in s['per_weapon'].values()])),s['metrics']['exact_top1']['accuracy']
    selected=max(summaries,key=rank)
    frozen={'created_at':now(),'source_experiment':str(source),'dataset':str(dataset),
            'dataset_sha256':dataset_digest(dataset),'algorithm_sha256':hashes(),
            'protocol_sha256':sha(source/'protocol.json'),'selected_configuration':selected.split(':')[0],
            'selected_aggregation':selected.split(':')[1],'training_results':summaries,'settings':SETTINGS,
            'training_bank':bank,'training_sample_ids':sorted({r['sample_id'] for r in raw}),
            'selection':'leave-one-training-match-out macro then micro; templates ordinals0/1 of other training matches; checks ordinals3/4 of excluded match; no test matches or rare test fallback templates in tuning',
            'caveat':'Previously inspected recordings. LOMO training checks exclude weapons missing in other training matches; test slots are never excluded for poor quality/missing templates.'}
    output.mkdir()
    (output/'training_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in raw),encoding='utf-8')
    atomic_json(output/'configuration_frozen.json',frozen)
    for key,s in summaries.items():
        print('training bank',key,rank(key),s['metrics']['exact_top1'],flush=True)
    print('selected',selected,flush=True)
    return frozen


def run(output,kind='independent',workers=8):
    output=Path(output).resolve();frozen=read_json(output/'configuration_frozen.json')
    dataset=Path(frozen['dataset']);source=Path(frozen['source_experiment'])
    if hashes()!=frozen['algorithm_sha256'] or dataset_digest(dataset)!=frozen['dataset_sha256'] or sha(source/'protocol.json')!=frozen['protocol_sha256']:
        raise ValueError('固定後に実装/dataset/protocolが変更されています')
    directory=output/kind
    if directory.exists():
        raise ValueError('既存outputを上書きしません')
    templates,original,train,test,original_reserved,source_match_weapon=source_protocol(source,kind)
    bank=make_bank(source,dataset,kind)
    metadata,rows=load_dataset(dataset);verify_sources(metadata)
    selected=[r for r in rows if r['match_id'] in test and r['weapon_label'] in bank['weapons']]
    if {r['sample_id'] for r in selected}&set(frozen['training_sample_ids']):
        raise ValueError('training check and test overlap')
    reserved=original_reserved|{tuple(k) for k in bank['template_frame_keys']}
    fallback_pairs={tuple(k) for k in bank['fallback_match_weapon']}
    scope={r['sample_id']:('within_match_other_timestamp' if (r['match_id'],r['weapon_label']) in source_match_weapon|fallback_pairs else 'cross_match') for r in selected}
    manifest=copy.deepcopy(original);manifest['split']['evaluation_scope']=scope
    configuration=frozen['selected_configuration'];policy=frozen['selected_aggregation']
    # Freeze before tests; include legacy opening bank to isolate source-time
    # improvement from preprocessing changes without choosing on test labels.
    configs=list(dict.fromkeys(('legacy',configuration)))
    matchers={'obs_bank_baseline_foreground':OBSRegionMatcher(templates,True)}
    matchers.update({f'obs_bank_{name}_foreground':OpeningBankMatcher(bank,**SETTINGS[name]) for name in configs})
    usable=[r for r in selected if (r['video_id'],r['frame_index']) not in reserved]
    raw=predict_rows(usable,dataset,metadata,manifest,matchers,workers)
    for row in selected:
        if (row['video_id'],row['frame_index']) in reserved:
            for method in matchers:
                raw.append(prediction({**row,'reserved_template_frame':True},[],method,scope[row['sample_id']]))
    attach_quality(raw,dataset,metadata)
    result=[]
    for method in matchers:
        for aggregation in dict.fromkeys(('mean','quality',policy)):
            for count in (1,3,5):
                result.extend(decisions([r for r in raw if r['method']==method],count,aggregation))
    default=f'obs_bank_{configuration}_{policy}_5frame_foreground'
    methods=sorted({r['method'] for r in result})
    preprocessors={name:('legacy' if name.startswith(('obs_bank_baseline_','obs_bank_legacy_')) else SETTINGS[configuration]['configuration']) for name in methods}
    report={'created_at':now(),'dataset':str(dataset),'candidate_universe':bank['weapons'],'configuration':frozen,
            'evaluation_kind':kind,'viewer_default_method':default,'preprocessors':preprocessors,
            'methods':{name:{'single_frame':summarize([r for r in result if r['method']==name])} for name in methods},
            'provenance':{'opening_templates_manifest':bank,'legacy_templates_manifest':original,
                          'actual_inferred_template_frame_overlap':0,'reserved_unavailable_samples':len(selected)-len(usable)},
            'scope_note':frozen['caveat'],'runtime_policy':'first5seconds, infer once and hold, no later-frame rescue'}
    directory.mkdir()
    for name,contents in [('frame_predictions.jsonl',raw),('predictions.jsonl',result)]:
        (directory/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in contents),encoding='utf-8')
    atomic_json(directory/'report.json',report);atomic_json(directory/'templates.json',bank)
    analysis={}
    for video in sorted({r['video_id'] for r in result}):
        analysis[video]={}
        for name in methods:
            s=summarize([r for r in result if r['video_id']==video and r['method']==name and r['scope']=='cross_match'])
            analysis[video][name]=s
            print(kind,video,name,s['metrics']['exact_top1'],flush=True)
    atomic_json(directory/'analysis.json',analysis)
    return report


def main():
    p=argparse.ArgumentParser(description='開始HUD見本＋training試合を丸ごと除外した設定選択')
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('source',type=Path);f.add_argument('dataset',type=Path);f.add_argument('--output',type=Path,required=True);f.add_argument('--workers',type=int,default=8)
    r=commands.add_parser('run');r.add_argument('output',type=Path);r.add_argument('--kind',choices=['independent','full_training'],default='independent');r.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.source,a.dataset,a.output,a.workers)
    else:run(a.output,a.kind,a.workers)


if __name__=='__main__':main()
