"""Paired opening decision audit, quality effects and reproducible crop contact sheets."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.evaluate_alignment import paired
from weapon_lamp_detect.evaluate_poc import summarize
from weapon_lamp_detect.match_data import atomic_json, read_json, write_image
from weapon_lamp_detect.opening_matcher import features, processed_region
from weapon_lamp_detect.region_matcher import weapon_region


def analyze(directory):
    directory=Path(directory)
    report=read_json(directory/'report.json')
    rows=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
    default=report['viewer_default_method']
    prefix=default.split('_quality_')[0] if '_quality_' in default else default.split('_mean_')[0]
    metadata,_=load_dataset(report['dataset'])
    def pct(m):
        v=m['accuracy'];return '—' if v is None else f'{100*v:.2f}% ({m["correct"]}/{m["support"]})'
    analysis={};lines=['# 開始時点の改善：同じ試合slotでの比較','',
           f'条件: {report["evaluation_kind"]} / 候補 {len(report["candidate_universe"])} / {default}',
           '',report.get('scope_note',''),'',
           '| video | 開始枚数 | 方式 | top-1 | top-3 | top-5 |','|---|---:|---|---:|---:|---:|']
    examples=[]
    for video in sorted({r['video_id'] for r in rows}):
        base_prefix=next(r['method'].split('_mean_')[0] for r in rows if '_baseline_mean_' in r['method'])
        analysis[video]={}
        for count in (1,3,5):
            base=[r for r in rows if r['video_id']==video and r['scope']=='cross_match' and r['method']==f'{base_prefix}_mean_{count}frame_foreground']
            new=[r for r in rows if r['video_id']==video and r['scope']=='cross_match' and r['method']==f'{prefix}_quality_{count}frame_foreground']
            if not new:
                new=[r for r in rows if r['video_id']==video and r['scope']=='cross_match' and r['method']==f'{prefix}_mean_{count}frame_foreground']
            changes=paired(base,new)
            analysis[video][str(count)]={'baseline':summarize(base),'improved':summarize(new),'paired':changes}
            for label,group in [('baseline',base),('improved',new)]:
                m=summarize(group)['metrics']
                lines.append(f'| {video} | {count} | {label} | {pct(m["exact_top1"])} | {pct(m["exact_top3"])} | {pct(m["exact_top5"])} |')
            if count==5:
                old={r['sample_id']:r for r in base}
                for r in new:
                    if r['expected'] in ('Sploosh-o-matic','Splash-o-matic') or (old[r['sample_id']]['correct'] and not r['correct']):
                        examples.append((old[r['sample_id']],r))
        lines+=['',f'## 武器別・開始5枚：{video}','','| 武器 | support | baseline | improved |','|---|---:|---:|---:|']
        b,n=analysis[video]['5']['baseline'],analysis[video]['5']['improved']
        for weapon,s in n['per_weapon'].items():
            lines.append(f'| {weapon} | {s["support"]} | {pct(b["per_weapon"][weapon]["exact_top1"])} | {pct(s["exact_top1"])} |')
        lines+=['','### 改善後の混同ペア','']
        lines.extend(f'- {p["expected"]} → {p["predicted"]}: {p["count"]}' for p in n['confusion_pairs'])
        mean={r['sample_id']:r for r in rows if r['video_id']==video and r['scope']=='cross_match' and r['method']==f'{prefix}_mean_3frame_foreground'}
        quality=[r for r in rows if r['video_id']==video and r['scope']=='cross_match' and r['method']==f'{prefix}_quality_3frame_foreground']
        if mean and quality:
            changed=[r for r in quality if r['predicted']!=mean[r['sample_id']]['predicted']]
            analysis[video]['quality_effect_3frames']={'changed_decisions':len(changed),
                'improved':sum(r['correct'] and not mean[r['sample_id']]['correct'] for r in changed),
                'regressed':sum(not r['correct'] and mean[r['sample_id']]['correct'] for r in changed),
                'case_ids':[r['sample_id'] for r in changed]}
            for r in changed:examples.append((mean[r['sample_id']],r))
    # Deliberately selected cause-review examples, not a random prevalence sample.
    seen=set();records=[]
    for before,row in examples:
        key=row['method']+':'+row['sample_id']
        if key in seen:continue
        seen.add(key)
        frames=row.get('decision_frames',[])
        width=max(1,len(frames))*210
        canvas=np.full((390,width,3),35,np.uint8)
        title=f"expected {row['expected']} | before {before['predicted']} -> {row['predicted']}"
        cv2.putText(canvas,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1)
        for i,f in enumerate(frames):
            crop=sample_crop(report['dataset'],{**row,**f},metadata)
            gray,_,_=features(processed_region(crop,'local_core'),True)
            display=cv2.cvtColor(np.clip(128+2*gray,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            for j,image in enumerate((crop,weapon_region(crop,True),display)):
                image=cv2.resize(image,(180,100),interpolation=cv2.INTER_NEAREST)
                canvas[50+j*110:150+j*110,10+i*210:190+i*210]=image
            q=row.get('frame_quality',[{}]*len(frames))[i] or {}
            cv2.putText(canvas,f"{f['timestamp']:.2f}s white {q.get('white_fraction',0):.2f}",(10+i*210,43),cv2.FONT_HERSHEY_SIMPLEX,.38,(245,245,245),1)
        relative=f'audit/case_{len(records):03d}.png';write_image(directory/relative,canvas)
        records.append({'review_id':key,'expected':row['expected'],'before':before['predicted'],'predicted':row['predicted'],
                        'margin':row['margin'],'image':relative,'note':'Rows: raw crop, legacy foreground, signed local contrast (zero=gray). Selected examples, not random.'})
    atomic_json(directory/'paired_analysis.json',analysis);atomic_json(directory/'audit_cases.json',records)
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('paired audit',directory,'images',len(records),flush=True)
    return analysis


def main():
    p=argparse.ArgumentParser(description='開始評価の同一slot比較・武器別accuracy・GO重み付け差分・crop標本')
    p.add_argument('directory',type=Path);a=p.parse_args();analyze(a.directory)


if __name__=='__main__':main()
