"""Paired metrics and visual crop audit, without modifying labels or evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import cv2_read, load_dataset
from weapon_lamp_detect.detail_matcher import detail_region
from weapon_lamp_detect.evaluate_poc import aggregate, metrics
from weapon_lamp_detect.match_data import Catalog, atomic_json, read_json, write_image

PAIR=("Sploosh-o-matic","Splash-o-matic")


def geometry_category(row):
    quality=row.get("crop_registration",{})
    if not quality.get("accepted"):
        return "unreliable_envelope_fallback"
    if abs(quality.get("scale",1)-1)>.08 or max(abs(quality.get("shift_x",0)),abs(quality.get("shift_y",0)))>6:
        return "large_geometry_change"
    return "small_or_no_geometry_change"


def cohort(rows):
    common=aggregate(rows,strategies=("mean_score",))["mean_score"]
    return {"single":metrics(rows),
            "multi":{n:r["common_cohort"]["metrics"] for n,r in common.items()},
            "per_weapon_single":{w:metrics([r for r in rows if r["expected"]==w]) for w in sorted({r["expected"] for r in rows})},
            "pair_confusions":dict(Counter(f"{r['expected']} -> {r['predicted']}" for r in rows
                                           if r["expected"] in PAIR and not r["correct"]))}


def thumbnail(image,width,height):
    canvas=np.full((height,width,3),30,np.uint8)
    h,w=image.shape[:2];scale=min(width/w,height/h)
    sized=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
    y,x=(height-sized.shape[0])//2,(width-sized.shape[1])//2
    canvas[y:y+sized.shape[0],x:x+sized.shape[1]]=sized
    return canvas


def audit_sheet(dataset, rows, output):
    """Selected examples are a diagnostic sample, not a prevalence estimate."""
    groups=defaultdict(list)
    for row in rows:
        if row["state"]!="alive" or row["scope"]!="cross_match":
            continue
        if row["expected"] in PAIR and not row["baseline_correct"]:
            groups[geometry_category(row)].append(row)
    chosen=[];seen=set()
    for category,cases in groups.items():
        for row in sorted(cases,key=lambda r:-(r.get("margin") or 0)):
            key=(row["match_id"],row["side"],row["slot_index"])
            if key in seen:
                continue
            seen.add(key);chosen.append(row)
            if sum(geometry_category(r)==category for r in chosen)>=3:
                break
    # Also inspect large changes outside the pair: crop repair must be generic.
    for row in sorted(rows,key=lambda r:-abs(r.get("crop_registration",{}).get("scale",1)-1)):
        if row["state"]=="alive" and row["scope"]=="cross_match" and row["expected"] not in PAIR and geometry_category(row)=="large_geometry_change":
            key=(row["match_id"],row["side"],row["slot_index"])
            if key not in seen:
                seen.add(key);chosen.append(row)
            if len([r for r in chosen if r["expected"] not in PAIR])>=3:
                break
    records=[]
    for page in range(0,len(chosen),4):
        items=chosen[page:page+4]
        sheet=np.full((len(items)*255,1280,3),25,np.uint8)
        for j,row in enumerate(items):
            y=j*255
            text=f"{page+j+1}. {row['expected']} -> {row['predicted']} | baseline {row['baseline_predicted']} | {geometry_category(row)}"
            cv2.putText(sheet,text,(12,y+22),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(sheet,f"{row['sample_id']} | scale {row.get('crop_registration',{}).get('scale',1):.3f}",(12,y+43),cv2.FONT_HERSHEY_SIMPLEX,.40,(200,200,200),1,cv2.LINE_AA)
            current=cv2_read(dataset/row['crop'])
            original=cv2_read(dataset/('original_'+row['crop']))
            context=cv2_read(dataset/row['context'])
            for rect,color in ((row['original_rect'],(0,255,255)),(row['rect'],(255,255,0))):
                x1,y1,x2,y2=rect;cv2.rectangle(context,(x1,y1),(x2,y2),color,2)
            context=context[:165,480:1440]
            for x,image,width,label in ((5,original,150,'original'),(165,current,150,'registered'),
                                         (325,detail_region(current),160,'detail input'),(495,context,780,'HUD yellow=original cyan=registered')):
                cv2.putText(sheet,label,(x,y+65),cv2.FONT_HERSHEY_SIMPLEX,.40,(225,225,225),1,cv2.LINE_AA)
                sheet[y+75:y+245,x:x+width]=thumbnail(image,width,170)
            records.append({k:row[k] for k in ('sample_id','expected','predicted','baseline_predicted','rect','original_rect','crop_registration')})
        write_image(output/f'audit_{page//4+1}.jpg',sheet)
    atomic_json(output/'audit_examples.json',{'selection':'high-margin pair failures from distinct match-slots, plus largest non-pair geometry changes; not random','examples':records})


def analyze(directory):
    directory=Path(directory).resolve()
    report=read_json(directory/'report.json')
    dataset=Path(report['dataset'])
    rows=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
    methods=sorted({r['method'] for r in rows})
    primary=[r for r in rows if r['state']=='alive' and r['scope']=='cross_match']
    baseline={r['sample_id']:r for r in rows if r['method']=='obs_baseline_foreground'}
    registered={r['sample_id']:r for r in rows if r['method']=='obs_registered_foreground'}
    videos=sorted({r['video_id'] for r in primary})
    results={video:{name:cohort([r for r in primary if r['method']==name and r['video_id']==video]) for name in methods} for video in videos}
    changes={};geometry={}
    for name in methods:
        subset=[r for r in primary if r['method']==name]
        c=Counter()
        for row in subset:
            before=baseline[row['sample_id']]['correct'];after=row['correct']
            c['both_correct' if before and after else 'both_wrong' if not before and not after else 'improved' if after else 'regressed']+=1
        changes[name]=dict(c)
        geometry[name]={category:metrics([r for r in subset if geometry_category(registered[r['sample_id']])==category])
                        for category in ('unreliable_envelope_fallback','large_geometry_change','small_or_no_geometry_change')}
    pair_audit={}
    for name in methods:
        pair_audit[name]={category:metrics([r for r in primary if r['method']==name and r['expected'] in PAIR and geometry_category(registered[r['sample_id']])==category])
                          for category in ('unreliable_envelope_fallback','large_geometry_change','small_or_no_geometry_change')}
    metadata,dataset_rows=load_dataset(dataset)
    pair_review=[]
    for row in primary:
        if row['method']=='obs_detail_foreground':
            pair_review.append({**row,'baseline_correct':baseline[row['sample_id']]['correct'],
                                'baseline_predicted':baseline[row['sample_id']]['predicted']})
    audit_sheet(dataset,pair_review,directory)
    artifact={'by_video':results,'paired_primary_changes':changes,'geometry_subgroups':geometry,
              'pair_geometry_subgroups':pair_audit,'registration_counts':dict(Counter(r['crop_registration']['reason'] for r in dataset_rows)),
              'caveat':'Geometry categories are automatic suspected changes, NOT human-confirmed crop correctness. No failed crop or weapon was removed from primary metrics.'}
    atomic_json(directory/'analysis.json',artifact)
    catalog=Catalog();display=lambda w:catalog.entries.get(w,{}).get('display_name',w)
    def pct(x):
        return '—' if x is None else f'{x*100:.2f}%'
    lines=['# crop / 照合改善の同一標本比較','',f"評価: {report['evaluation_kind']}。alive / cross-match。候補58武器を固定。",
           '', '全標本を保持。補正失敗は元cropへfallback。state・正解・timestamp・分割は変更していない。',
           '', '| video | 方式 | crop support | single top-1 | top-3 | top-5 | 3frame top-1 / slot support |',
           '|---|---|---:|---:|---:|---:|---:|']
    for video in videos:
        for name in methods:
            c=results[video][name];m=c['single'];multi=c['multi']['3']
            lines.append(f"| {video} | {name} | {m['support']} | {pct(m['exact_top1']['accuracy'])} | {pct(m['exact_top3']['accuracy'])} | {pct(m['exact_top5']['accuracy'])} | {pct(multi['exact_top1']['accuracy'])} / {multi['support']} |")
    lines+=['','## 全武器の改善・悪化（single）','','| 武器 | video | support | baseline | registered | detail |','|---|---|---:|---:|---:|---:|']
    for video in videos:
        for weapon,b in results[video]['obs_baseline_foreground']['per_weapon_single'].items():
            v=[results[video][name]['per_weapon_single'][weapon]['exact_top1']['accuracy'] for name in ('obs_baseline_foreground','obs_registered_foreground','obs_detail_foreground')]
            lines.append(f"| {display(weapon)} | {video} | {b['support']} | {' | '.join(pct(a) for a in v)} |")
    lines+=['','## 注意','','- 3frameは10frame以上ある同じmatch-slotの共通cohortで比較。','- 既に観察した評価集合での探索的比較。新規の未観察動画による最終検証ではない。',
            '- 自動のgeometry区分を「crop正常／異常」の人力正解とは扱わない。画像と枠をViewerで確認する。',
            '- 混同ペア・全武器のsupport・marginはreport.json。1/3/5/10frame・geometry区分はanalysis.json。',
            '- audit_*.jpgは原因確認用に選んだ標本であり、異常の発生率の推定には使わない。']
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for video in videos:
        print(video,{n:(results[video][n]['single']['exact_top1'],results[video][n]['multi']['3']['exact_top1']) for n in methods})
    return artifact


def main():
    p=argparse.ArgumentParser(description='同一sampleの全武器比較・crop原因確認標本を生成')
    p.add_argument('report_dir',type=Path)
    analyze(p.parse_args().report_dir)


if __name__=='__main__':
    main()
