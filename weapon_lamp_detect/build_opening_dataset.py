"""Only the first stable gameplay HUD, not arbitrary later alive frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, verify_sources
from weapon_lamp_detect.match_data import atomic_json, calibrated_crop, calibrated_slot, now, squid_detector, write_image


def opening_timer(seconds):
    # Ranked 5:00 / Turf 3:00. Never reinterpret an arbitrary late timer as start.
    return isinstance(seconds,int) and (295<=seconds<=300 or 175<=seconds<=180)


def prepare(dataset,output,window=5.,interval=1.,workers=4):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('新しいoutputを指定してください')
    if not all(math.isfinite(v) and v>0 for v in (window,interval)) or workers<1:
        raise ValueError('window/interval > 0, workers >= 1')
    metadata,_=load_dataset(dataset);verify_sources(metadata)
    matches=[m for m in metadata['matches'] if m.get('confirmed') and not m.get('rejected')]
    output.mkdir(parents=True);cv2.setNumThreads(1)
    def job(match):
        detector=squid_detector(match['ally_side'])
        video=metadata['videos'][match['video_id']]
        cap=cv2.VideoCapture(video['path'])
        def read(timestamp):
            index=round(timestamp*video['fps'])
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:
                raise ValueError(f"frameを読めません: {match['match_id']}/{index}")
            return frame,index,index/video['fps']
        start=match['start_timestamp']
        configured=float(match.get('hud_offset_seconds',20))
        # Search only around the intro -> HUD transition, with a hard upper bound.
        first,last=start+max(8.,configured-10),min(start+configured+8,match['end_timestamp'])
        consecutive=[];base=None;scan=[]
        try:
            t=first
            while t<last:
                frame,index,actual=read(t)
                reading=detector.read_frame(frame,actual,index)
                timer=detector.timer_ocr.read_frame(frame,timestamp=actual,frame_index=index) if detector.timer_ocr else None
                slots=[calibrated_slot(frame,reading,side,i,match.get('geometry')) for side in ('left','right') for i in range(4)]
                alive=sum(s.state=='alive' for s in slots)
                seconds=getattr(timer,'seconds',None)
                valid=reading.hud_state=='match' and getattr(timer,'kind',None)=='time' and opening_timer(seconds) and alive>=6
                scan.append({'timestamp':actual,'frame_index':index,'remaining_seconds':seconds,'alive_slots':alive,'valid':valid})
                consecutive=(consecutive+[actual]) if valid else []
                if len(consecutive)>=2:
                    # Allow the HUD's initial animation to settle, still at opening.
                    base=consecutive[0]+1.
                    break
                t+=.5
            fallback=base is None
            if fallback:
                base=start+configured
            rows=[];timers=[];seen=set();sample=0
            while sample*interval<window and base+sample*interval<match['end_timestamp']:
                frame,index,actual=read(base+sample*interval);sample+=1
                if index in seen:
                    continue
                seen.add(index)
                reading=detector.read_frame(frame,actual,index)
                timer=detector.timer_ocr.read_frame(frame,timestamp=actual,frame_index=index) if detector.timer_ocr else None
                seconds=getattr(timer,'seconds',None)
                timers.append({'timestamp':actual,'frame_index':index,'remaining_seconds':seconds,'hud_state':reading.hud_state})
                context=f"frames/{match['match_id']}/{index:09d}.jpg";write_image(output/context,frame)
                for side in ('left','right'):
                    for slot_index in range(4):
                        key=f'{side}{slot_index}';label=match['slots'][key]
                        if label['status']!='labeled':
                            continue
                        slot=calibrated_slot(frame,reading,side,slot_index,match.get('geometry'))
                        crop,rect=calibrated_crop(frame,side,slot_index,match.get('geometry'))
                        # Non-match/fallback is recorded, not silently removed.
                        state=slot.state if reading.hud_state=='match' else 'unknown'
                        row={'sample_id':f"{match['match_id']}_{index:09d}_{key}",'source_video':video['path'],
                             'video_id':match['video_id'],'match_id':match['match_id'],'timestamp':actual,'frame_index':index,
                             'side':side,'slot_index':slot_index,'state':state,'state_source':'automatic squid_lamp_detect',
                             'weapon_label':label['weapon'],'weapon_class':label.get('weapon_class'),'label_status':'labeled',
                             'label_source':'human_confirmed_match','match_revision':match['revision'],'rect':rect,
                             'opening_timestamp':base,'opening_frame_ordinal':sample-1,'seconds_after_opening':actual-base,
                             'remaining_seconds':seconds,'opening_detection_fallback':fallback,
                             'crop':f"crops/{match['match_id']}/{index:09d}_{key}.png",'context':context,
                             'special_score':slot.special_score,
                             'blur_variance':float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())}
                        write_image(output/row['crop'],crop);rows.append(row)
            print('opening:',match['match_id'],round(base-start,2),'intro seconds,',len(rows),'crops',flush=True)
            return rows,{'match_id':match['match_id'],'opening_timestamp':base,'opening_offset_from_intro':base-start,
                         'fallback':fallback,'scan':scan,'frames':timers}
        finally:
            cap.release()
    with ThreadPoolExecutor(workers) as pool:
        results=list(pool.map(job,matches))
    from weapon_lamp_detect.match_data import Catalog
    catalog=Catalog()
    rows=[r for group,_ in results for r in group]
    for r in rows:
        r['weapon_class']=catalog.entries[r['weapon_label']]['weapon_class']
    (output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    # Seed extraction settings/counts are not the new opening export's settings.
    # Keep their provenance separately and count the frames actually exported.
    seed_keys=('sampling_interval','start_offset','end_offset','effective_start_offsets',
               'max_samples','max_frames_per_match','samples_per_match','excluded')
    excluded=Counter()
    for match,(_,record) in zip(matches,results):
        for label in match['slots'].values():
            if label['status']!='labeled':
                excluded['label_'+label['status']]+=len(record['frames'])
    metadata={**metadata,'created_at':now(),'source_dataset':str(dataset),'samples':len(rows),
              'source_dataset_config':{k:metadata[k] for k in seed_keys if k in metadata},
              'samples_per_match':dict(Counter(r['match_id'] for r in rows)),
              'max_frames_per_match':max((len(record['frames']) for _,record in results),default=0),
              'start_offset':None,'end_offset':0.,'max_samples':0,'excluded':dict(excluded),
              'effective_start_offsets':{record['match_id']:record['opening_offset_from_intro'] for _,record in results},
              'source_dataset_sha256':hashlib.sha256((dataset/'samples.jsonl').read_bytes()).hexdigest(),
              'opening_window_seconds':window,'sampling_interval':interval,'crop_export':True,
              'opening_detection':dict((record['match_id'],record) for _,record in results),
              'opening_definition':'Two consecutive gameplay HUD + 5:00/3:00 opening timer + at least 6 alive slots, then 1 second settle; fallback uses configured intro HUD offset and stays in denominator.',
              'window_note':'Hard bounded opening window. Never seek later alive frames to fill missing/down slots.',
              'opening_algorithm_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(output/'dataset.json',metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description='試合開始HUDの最初の5秒だけを元OBSから収集（進行中frameで補わない）')
    p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=float,default=5.);p.add_argument('--sample-interval',type=float,default=1.);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();prepare(a.dataset,a.output,a.window,a.sample_interval,a.workers)


if __name__=='__main__':
    main()
