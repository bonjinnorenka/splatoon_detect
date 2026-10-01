"""All labeled matches: hold out an entire match before using any of its crops.

This is supplementary full-data cross-validation with frozen hyperparameters,
not the independent 13/26 protocol. It never substitutes for those results.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_alignment import sha
from weapon_lamp_detect.evaluate_improved_opening import attach_quality, decisions
from weapon_lamp_detect.evaluate_opening import source_protocol
from weapon_lamp_detect.evaluate_opening_hybrid import hashes
from weapon_lamp_detect.evaluate_poc import prediction, summarize
from weapon_lamp_detect.hybrid_opening_matcher import HybridOpeningMatcher, hybrid_bank
from weapon_lamp_detect.match_data import atomic_json, now, read_json
from weapon_lamp_detect.opening_template_bank import OpeningBankMatcher


def fold(job):
    bank,historical,dataset,match,configuration=job;cv2.setNumThreads(1)
    metadata,rows=load_dataset(dataset)
    selected=[r for r in rows if r['match_id']==match and r['weapon_label'] in bank['weapons']]
    matchers={'obs_cv_baseline_foreground':OpeningBankMatcher(historical,excluded_matches=(match,)),
              'obs_cv_historical_shared_foreground':OpeningBankMatcher(historical,shared=True,excluded_matches=(match,)),
              f'obs_cv_{configuration}_foreground':HybridOpeningMatcher(bank,configuration,excluded_matches=(match,))}
    remaining=[e for e in bank['templates'] if all(f['match_id']!=match for f in e['source_frames'])]
    source_keys={(f['video_id'],f['frame_index']) for e in remaining for f in e['source_frames']}
    if source_keys&{(r['video_id'],r['frame_index']) for r in selected}:
        raise ValueError('fold source frame overlaps evaluation')
    present={e['weapon'] for e in remaining}
    raw=[]
    for row in selected:
        crop=sample_crop(dataset,row,metadata)
        for name,matcher in matchers.items():
            candidates=[c.to_dict() for c in matcher.predict_crop(crop,len(bank['weapons']),row['side'])]
            raw.append(prediction({**row,'template_missing_in_fold':row['weapon_label'] not in present},candidates,name,'cross_match'))
    return attach_quality(raw,dataset,metadata),{'match_id':match,'template_source_matches':sorted({f['match_id'] for e in remaining for f in e['source_frames']}),
           'inferred_template_frame_overlap':0,'missing_expected_weapons':sorted({r['weapon_label'] for r in selected}-present),
           'evaluated_slots':len(selected)//5}


def run(experiment,output,workers=8):
    experiment,output=Path(experiment).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('既存outputを上書きしません')
    frozen=read_json(experiment/'configuration_frozen.json');source=Path(frozen['source_experiment']);dataset=Path(frozen['dataset'])
    if hashes()!=frozen['algorithm_sha256'] or dataset_digest(dataset)!=frozen['dataset_sha256'] or sha(source/'protocol.json')!=frozen['protocol_sha256']:
        raise ValueError('固定後に実装/dataset/protocolが変更されています')
    for kind,expected in frozen['source_template_manifest_sha256'].items():
        if sha(source_protocol(source,kind)[0]/'templates.json')!=expected:raise ValueError('template manifest変更')
    bank=hybrid_bank(source,dataset,'full_training')
    _,original,*_=source_protocol(source,'full_training')
    historical={**bank,'templates':[{**e,'dataset':original['dataset']} for e in original['templates']]}
    metadata,rows=load_dataset(dataset);verify_sources(metadata)
    matches=sorted({r['match_id'] for r in rows if r['weapon_label'] in bank['weapons']})
    raw=[];folds=[]
    configuration=frozen['selected_configuration'];policy=frozen['selected_aggregation']
    jobs=[(bank,historical,str(dataset),match,configuration) for match in matches]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,(result,record) in enumerate(pool.map(fold,jobs)):
            raw.extend(result);folds.append(record);print('full-data CV',i+1,'/',len(jobs),'slots',record['evaluated_slots'],flush=True)
    result=[]
    for name in sorted({r['method'] for r in raw}):
        for aggregation in dict.fromkeys(('mean','quality',policy)):
            for count in (1,3,5):result.extend(decisions([r for r in raw if r['method']==name],count,aggregation))
    default=f'obs_cv_{configuration}_{policy}_5frame_foreground';methods=sorted({r['method'] for r in result})
    report={'created_at':now(),'dataset':str(dataset),'candidate_universe':bank['weapons'],'configuration':frozen,
            'evaluation_kind':'leave_one_match_out_full_data','viewer_default_method':default,
            'methods':{name:{'single_frame':summarize([r for r in result if r['method']==name])} for name in methods},
            'raw_frame_methods':{name:summarize([r for r in raw if r['method']==name]) for name in sorted({r['method'] for r in raw})},
            'preprocessors':{name:'legacy' for name in methods},
            'secondary_preprocessors':{name:'local_core' for name in methods if name.startswith(f'obs_cv_{configuration}_')},
            'provenance':{'folds':folds,'source_bank':bank,'cv_algorithm_sha256':sha(Path(__file__))},
            'scope_note':'Supplementary whole-match cross-validation on two previously inspected recordings, including tuning matches. Hyperparameters frozen before these scores; not independent/new-recording accuracy. Missing expected templates remain incorrect in denominator.',
            'runtime_policy':'opening first5seconds once, hold thereafter'}
    output.mkdir()
    for name,contents in [('frame_predictions.jsonl',raw),('predictions.jsonl',result)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in contents),encoding='utf-8')
    atomic_json(output/'report.json',report)
    analysis={}
    for video in sorted({r['video_id'] for r in result}):
        analysis[video]={}
        for name in methods:
            s=summarize([r for r in result if r['video_id']==video and r['method']==name])
            analysis[video][name]=s;print('CV',video,name,s['metrics']['exact_top1'],flush=True)
    atomic_json(output/'analysis.json',analysis)
    held=defaultdict(dict)
    for r in result:
        if r['method']==default:held[r['match_id']][f"{r['side']}{r['slot_index']}"]={k:r[k] for k in ('predicted','confidence','margin','top_k','decision_timestamp')}
    atomic_json(output/'match_predictions.json',{'source':'out-of-match model predictions, NOT GT','hold_for_match':True,'method':default,'matches':dict(held)})
    return report


def main():
    p=argparse.ArgumentParser(description='設定固定済みモデルを、全53試合を1試合ずつ丸ごと除外して評価')
    p.add_argument('experiment',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.experiment,a.output,a.workers)


if __name__=='__main__':main()
