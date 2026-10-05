"""Held-out match evaluation, reusing existing metrics/aggregation/error viewer."""
from __future__ import annotations

import argparse
import csv
import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from weapon_cnn.data import digest, sha, verify, write_rows
from weapon_cnn.model import load_model
from weapon_cnn.train import arrays, check_protocol, probabilities
from weapon_lamp_detect.evaluate_improved_opening import decisions
from weapon_lamp_detect.evaluate_poc import prediction, summarize
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json


def confidence_fix(row):
    # The reused template aggregator computes a heuristic confidence. For CNN
    # outputs replace it with the actual averaged probability (uncalibrated).
    row['method']=row['method'].removesuffix('_foreground')
    for key in ('all_candidates','top_k'):
        for c in row[key]:c['confidence']=c['score']
    row['confidence']=row['best_score']
    return row


def cluster_interval(rows,seed=20261004,draws=2000):
    groups=defaultdict(list)
    for r in rows:groups[r['match_id']].append(r['correct'])
    totals=np.array([[sum(v),len(v)] for v in groups.values()])
    rng=np.random.default_rng(seed)
    if not len(totals):return {'lower':None,'upper':None}
    sampled=totals[rng.integers(0,len(totals),(draws,len(totals)))].sum(1)
    values=sampled[:,0]/sampled[:,1]
    return {'lower':float(np.quantile(values,.025)),'upper':float(np.quantile(values,.975)),
            'unit':'whole match bootstrap','matches':len(totals),'draws':draws,'seed':seed}


def whole_match_accuracy(rows):
    """Only count matches with exactly one prediction for each of eight slots."""
    groups=defaultdict(dict)
    expected={(side,slot) for side in ('left','right') for slot in range(4)}
    for row in rows:
        slots=groups[row['match_id']];key=(row['side'],row['slot_index'])
        if key in slots:raise ValueError('whole-match指標に同一slotの重複があります')
        slots[key]=row['correct']
    complete=[slots for slots in groups.values() if set(slots)==expected]
    correct=sum(all(slots.values()) for slots in complete)
    return {'correct':correct,'support':len(complete),
            'accuracy':correct/len(complete) if complete else None,
            'excluded_incomplete_matches':len(groups)-len(complete),
            'definition':'All eight slots correct together; incomplete matches excluded.'}


def assessment(report, catalog):
    def pct(v):return 'N/A' if v is None else f'{v*100:.2f}%'
    lines=['# CPU CNN：開始HUDの武器識別・実測評価','',
           f"学習{report['split_counts']['train']}試合 / validation {report['split_counts']['validation']}試合 / test {report['split_counts']['test']}試合。",
           f"candidate {len(report['candidate_universe'])}武器。testは{report['test_weapons']}武器、{report['test_match_slots']} match-slot、{report['test_crops']}crop。",
           '','同一matchの全slot・全frameを同じsplitに固定。checkpoint選択はvalidationだけ。testは最終評価のみ。',
           '単一frameは先頭frameのmatch-slot accuracy。raw全frameのaccuracyとは分けて記載する。',
           '','| 集約 | top-1 | top-3 | top-5 | class | macro top-1 | support |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in report['methods'].items():
        m=s['metrics'];macro=float(np.mean([v['exact_top1']['accuracy'] for v in s['per_weapon'].values()]))
        lines.append(f"| {name} | {pct(m['exact_top1']['accuracy'])} | {pct(m['exact_top3']['accuracy'])} | {pct(m['exact_top5']['accuracy'])} | {pct(m['class_top1']['accuracy'])} | {pct(macro)} | {m['support']} |")
    primary=report['methods'][report['primary_method']];ci=report['primary_match_bootstrap_95ci']
    complete=report['primary_whole_match_accuracy']
    lines+=['',f"主指標のmatch bootstrap 95%区間: {pct(ci['lower'])}–{pct(ci['upper'])}。",
            f"8武器すべて正解だった試合: {complete['correct']}/{complete['support']} = {pct(complete['accuracy'])}（不完全な{complete['excluded_incomplete_matches']}試合は除外）。slot単位の正解率とは異なる。",
            '', '## 混同（主指標）','']
    for p in primary['confusion_pairs'][:20]:
        expected=catalog.entries[p['expected']]['display_name']
        predicted=catalog.entries.get(p['predicted'],{}).get('display_name',p['predicted'])
        lines.append(f"- {expected} → {predicted}: {p['count']}件")
    lines+=['','## 評価範囲・制限','',
            '- 1試合だけの武器はtrain専用。別試合での精度は未測定。',
            '- 未収集武器はモデルのcandidateに含めない。未知武器の検出は未評価。',
            '- cross-match評価であり、別録画日・別session・別動画への汎化を保証しない。過去に確認したOBS/ライブ記録の固定分割で、完全未観測の新録画ではない。',
            '- 自動alive/down/unknownで主評価を除外しない。shapeをdownと誤認する場合がある。state別frame指標はraw reportに保存する。',
            '- softmax / marginは記録するが、confidenceの校正・Unknown閾値は未実施。',
            '- 過去の58武器template評価とは候補数・splitが違うため、精度の直接比較はできない。',
            '', 'train-only singleton: '+', '.join(catalog.entries[w]['display_name'] for w in report['singleton_training_only']),
            '', '未学習catalog武器: '+', '.join(catalog.entries[w]['display_name'] for w in report['untrained_catalog_weapons']),
            '', f"checkpoint epoch: {report['checkpoint_epoch']} / validation選択。",
            f"8 cropのCPU推論: median {report['cpu_latency_ms']['batch8_median']:.2f}ms（crop入力・前処理済みtensor、warm実行）。",
            f"torch {report['torch_version']} / threads {report['threads']}。"]
    return '\n'.join(lines)+'\n'


def run(dataset,checkpoint_path,protocol_path,output,threads=8,batch_size=128):
    dataset,checkpoint_path,protocol_path,output=map(lambda p:Path(p).resolve(),(dataset,checkpoint_path,protocol_path,output))
    if output.exists():raise ValueError('評価結果を上書きしません。新しいoutputを指定してください')
    if threads<1 or batch_size<1:raise ValueError('threads/batch_size >= 1')
    metadata,rows=verify(dataset);protocol=read_json(protocol_path);parts=check_protocol(dataset,protocol,rows)
    torch.set_num_threads(threads);model,checkpoint=load_model(checkpoint_path)
    if checkpoint['dataset_sha256']!=digest(dataset) or checkpoint['protocol_sha256']!=sha(protocol_path):
        raise ValueError('checkpointとdataset/protocolが一致しません')
    if checkpoint['classes']!=protocol['classes']:raise ValueError('class順が異なります')
    if checkpoint['algorithm_sha256']['model.py']!=sha(Path(__file__).parent/'model.py'):
        raise ValueError('学習後にmodel前処理が変わっています')
    selected=[r for r in rows if r['match_id'] in parts['test']]
    x,labels=arrays(dataset,selected,protocol['classes']);probs=probabilities(model,x,batch_size)
    catalog=Catalog(catalog_path=dataset/'catalog.json');raw=[]
    for row,p in zip(selected,probs):
        candidates=[{'weapon':protocol['classes'][i],
                     'weapon_class':catalog.entries[protocol['classes'][i]]['weapon_class'],
                     'score':float(p[i]),'confidence':float(p[i])} for i in np.argsort(-p)]
        raw.append(prediction(row,candidates,'cnn_raw_frame','cross_match'))
    result=list(raw);methods={'cnn_raw_frame':summarize(raw)}
    for policy in ('mean','quality'):
        for count in (1,3,5):
            aggregated=[confidence_fix(r) for r in decisions(raw,count,policy)]
            result.extend(aggregated);methods[aggregated[0]['method']]=summarize(aggregated)
    primary='cnn_raw_frame_quality_5frame'
    with torch.inference_mode():
        batch=x[:8].float()/255
        for _ in range(10):model(batch)
        timings=[]
        for _ in range(40):
            tick=time.perf_counter();model(batch);timings.append((time.perf_counter()-tick)*1000)
    report={'created_at':now(),'dataset':str(dataset),'dataset_sha256':digest(dataset),
            'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),
            'protocol_sha256':sha(protocol_path),'checkpoint_epoch':checkpoint['epoch'],
            'candidate_universe':protocol['classes'],'methods':methods,'primary_method':primary,
            'viewer_default_method':primary,'split_counts':{k:len(v) for k,v in parts.items()},
            'test_crops':len(raw),'test_match_slots':len(aggregated),'test_weapons':len({r['expected'] for r in raw}),
            'singleton_training_only':protocol['singleton_training_only'],
            'test_unrepresented':protocol['test_unrepresented'],'per_weapon_split_support':protocol['per_weapon_matches'],
            'untrained_catalog_weapons':checkpoint['untrained_catalog_weapons'],
            'primary_match_bootstrap_95ci':cluster_interval([r for r in result if r['method']==primary]),
            'primary_whole_match_accuracy':whole_match_accuracy([r for r in result if r['method']==primary]),
            'confidence_note':'Uncalibrated softmax / mean probabilities. No Unknown threshold fixed.',
            'scope_note':protocol['scope'],'torch_version':str(torch.__version__),'threads':threads,
            'cpu_latency_ms':{'batch8_median':float(np.median(timings)),'batch8_p95':float(np.percentile(timings,95)),
                              'excludes':'file decoding / capture / crop extraction / OpenCV canonical ROI and resize; model RGB+local tensor preprocessing included'},
            'test_used_for_selection':False}
    output.mkdir(parents=True);atomic_json(output/'report.json',report)
    write_rows(output/'predictions.jsonl',result)
    (output/'assessment.md').write_text(assessment(report,catalog),encoding='utf-8')
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['武器名','weapon','test_match_slots','correct','top1','top3','top5','train_matches','validation_matches','test_matches'])
    for weapon,s in methods[primary]['per_weapon'].items():
        support=protocol['per_weapon_matches'][weapon]
        writer.writerow([catalog.entries[weapon]['display_name'],weapon,s['support'],s['exact_top1']['correct'],
                         s['exact_top1']['accuracy'],s['exact_top3']['accuracy'],s['exact_top5']['accuracy'],
                         support['train'],support['validation'],support['test']])
    (output/'per_weapon.csv').write_text(stream.getvalue(),encoding='utf-8-sig')
    print('HELD-OUT TEST',methods[primary]['metrics'],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description='固定CNNの別試合test精度・1/3/5frame・誤判定を評価')
    p.add_argument('dataset',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args();run(a.dataset,a.checkpoint,a.protocol,a.output,a.threads,a.batch_size)


if __name__=='__main__':main()
