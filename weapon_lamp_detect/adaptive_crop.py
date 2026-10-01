"""Label-blind lamp registration; original annotations and sample IDs stay intact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.match_data import atomic_json, now, write_image


CONFIG = {"hue_tolerance": 12, "minimum_saturation": 85, "minimum_value": 70,
          "search_expansion": .20, "scale_limits": [.70, 1.35],
          "max_shift_fraction": .28, "normalization": "uniform lamp-envelope scale"}


def locate_lamp(frame, rect):
    """Find the colored lamp envelope near a human-specified slot, never a weapon.

    Work only in the upper HUD band. A closed team-color component must reach
    the triangle-tip zone and the body zone; badges below the lamp are excluded.
    Uncertain detections are reported, not silently used to delete samples.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, rect)
    rw, rh = x2-x1, y2-y1
    if rw < 8 or rh < 8:
        return {"valid": False, "reason": "small_crop"}
    margin = round(rw*CONFIG["search_expansion"])
    a, b = max(0, x1-margin), min(w, x2+margin)
    c, d = max(0, y1-round(rh*.15)), min(h, y1+round(rh*.91))
    image = frame[c:d, a:b]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Estimate ink only above the weapon/badge band, reducing background votes.
    tip = hsv[:max(1, round(rh*.40))]
    eligible = (tip[:,:,1] >= CONFIG["minimum_saturation"]) & (tip[:,:,2] >= CONFIG["minimum_value"])
    hues = tip[:,:,0][eligible]
    if len(hues) < rw*rh*.025:
        return {"valid": False, "reason": "no_ink_tip"}
    hist = np.bincount(hues, minlength=180).astype(np.float32)
    hist = sum(np.roll(hist, i) for i in range(-4,5))
    hue = int(hist.argmax())
    delta = np.abs(hsv[:,:,0].astype(np.int16)-hue)
    delta = np.minimum(delta,180-delta)
    mask = ((delta <= CONFIG["hue_tolerance"]) & (hsv[:,:,1] >= CONFIG["minimum_saturation"])
            & (hsv[:,:,2] >= CONFIG["minimum_value"])).astype(np.uint8)*255
    kernel = max(3, round(rh*.055)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel,kernel),np.uint8))
    # Neighboring lamps can join along their lower edge after closing. Split at
    # color-projection valleys around the slot rather than following that bridge.
    center_local = (x1+x2)/2-a
    profile = (mask>0).sum(axis=0)
    def valley(lo,hi):
        indices=np.arange(max(0,round(lo)),min(len(profile),round(hi)))
        return int(indices[np.argmin(profile[indices])])
    left=valley(center_local-rw*.60,center_local-rw*.28)
    right=valley(center_local+rw*.28,center_local+rw*.60)
    mask[:,:left+1]=0
    mask[:,right:]=0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    choices = []
    for i in range(1,n):
        x,y,cw,ch,area = map(int,stats[i])
        center = a+x+cw/2
        if not (.36*rw <= cw <= 1.20*rw and .48*rh <= ch <= 1.10*rh):
            continue
        if y > rh*.38 or y+ch < rh*.63 or abs(center-(x1+x2)/2) > rw*.32:
            continue
        if area < rw*rh*.13 or area/(cw*ch) < .30:
            continue
        top_band=(labels[y:y+max(2,round(ch*.10)),x:x+cw]==i)
        if y<=1 or top_band.mean()>.55:
            continue  # Clipped/full-width scene ink is not a squid triangle tip.
        # A component touching a horizontal search edge can be scene ink.
        if x == 0 or x+cw >= b-a:
            continue
        choices.append((area-abs(center-(x1+x2)/2)*rh, [a+x,c+y,a+x+cw,c+y+ch], area))
    if not choices:
        return {"valid": False, "reason": "unreliable_envelope", "hue": hue}
    _, bbox, area = max(choices)
    return {"valid": True, "bbox": bbox, "hue": hue, "ink_area": area,
            "reason": "colored_lamp_envelope"}


def registered_rect(rect, reference, current, frame_shape):
    if not reference.get("valid") or not current.get("valid"):
        return list(rect), {"accepted": False, "reason": "missing_envelope"}
    ref, cur = reference["bbox"], current["bbox"]
    sx = (cur[2]-cur[0])/(ref[2]-ref[0])
    sy = (cur[3]-cur[1])/(ref[3]-ref[1])
    scale = float(np.sqrt(sx*sy))
    low, high = CONFIG["scale_limits"]
    rx, ry = (ref[0]+ref[2])/2, (ref[1]+ref[3])/2
    cx, cy = (cur[0]+cur[2])/2, (cur[1]+cur[3])/2
    dx,dy = cx-rx,cy-ry
    rw,rh = rect[2]-rect[0],rect[3]-rect[1]
    details = {"scale": scale, "shift_x": dx, "shift_y": dy,
               "reference_bbox": ref, "current_bbox": cur}
    if not low <= scale <= high or max(sx/sy,sy/sx) > 1.35:
        return list(rect), {**details,"accepted":False,"reason":"unstable_scale"}
    if abs(dx)>rw*CONFIG["max_shift_fraction"] or abs(dy)>rh*CONFIG["max_shift_fraction"]:
        return list(rect), {**details,"accepted":False,"reason":"unstable_position"}
    proposed = [round(cx+(rect[0]-rx)*scale),round(cy+(rect[1]-ry)*scale),
                round(cx+(rect[2]-rx)*scale),round(cy+(rect[3]-ry)*scale)]
    h,w = frame_shape[:2]
    if proposed[0]<0 or proposed[1]<0 or proposed[2]>w or proposed[3]>h:
        return list(rect), {**details,"accepted":False,"reason":"outside_frame"}
    return proposed, {**details,"accepted":True,"reason":"registered"}


def prepare(dataset, output, workers=4):
    dataset,output = Path(dataset).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError("新しいoutputを指定してください（元datasetは上書きしません）")
    if workers<1:
        raise ValueError("workers >= 1 が必要です")
    metadata,rows = load_dataset(dataset)
    verify_sources(metadata)
    output.mkdir(parents=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["match_id"]].append(row)
    cv2.setNumThreads(1)

    def match_job(item):
        match_id, samples = item
        samples = sorted(samples,key=lambda r:(r["frame_index"],r["side"],r["slot_index"]))
        frames = defaultdict(list)
        for row in samples:
            frames[row["frame_index"]].append(row)
        cap = cv2.VideoCapture(metadata["videos"][samples[0]["video_id"]]["path"])
        reference,updated = {},[]
        try:
            for index, group in frames.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,frame = cap.read()
                if not ok:
                    raise ValueError(f"frameを読めません: {match_id}/{index}")
                # Never use JPEG context pixels for matching: recrop OBS source.
                context = f"frames/{match_id}/{index:09d}.jpg"
                write_image(output/context,frame)
                for original in group:
                    row = copy.deepcopy(original)
                    key = (row["side"],row["slot_index"])
                    detection = locate_lamp(frame,row["rect"])
                    if key not in reference and row["state"]=="alive" and detection["valid"]:
                        reference[key] = detection
                    rect,quality = registered_rect(row["rect"],reference.get(key,{}),detection,frame.shape)
                    # Fallbacks stay in the main denominator, including down.
                    row["original_rect"],row["rect"] = row["rect"],rect
                    row["crop_registration"] = {**quality,"detection":detection,
                                                "quality_source":"automatic, label-blind"}
                    row["crop"] = f"crops/{match_id}/{index:09d}_{row['side']}{row['slot_index']}.png"
                    row["context"] = context
                    x1,y1,x2,y2 = rect
                    write_image(output/row["crop"],frame[y1:y2,x1:x2])
                    updated.append(row)
        finally:
            cap.release()
        print(f"adaptive crop: {match_id} / {len(updated)} samples",flush=True)
        return updated

    with ThreadPoolExecutor(workers) as pool:
        changed = [row for group in pool.map(match_job,grouped.items()) for row in group]
    by_id = {r["sample_id"]:r for r in changed}
    ordered = [by_id[r["sample_id"]] for r in rows]
    (output/"samples.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in ordered),encoding="utf-8")
    metadata = {**metadata,"created_at":now(),"source_dataset":str(dataset),
                "source_dataset_sha256":dataset_digest(dataset),"crop_export":True,
                "crop_profile":"label_blind_registered_v1","registration_configuration":CONFIG,
                "registration_algorithm_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "registration_note":"First reliable alive envelope per slot anchors original human geometry; no weapon label or prediction is consulted; failed detections fall back without excluding samples."}
    atomic_json(output/"dataset.json",metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description="既存sampleを保持し、元OBSから位置/サイズ補正cropを別datasetへ出力")
    p.add_argument("dataset",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    prepare(a.dataset,a.output,a.workers)


if __name__=="__main__":
    main()
