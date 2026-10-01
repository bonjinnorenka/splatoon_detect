"""Freeze region hypotheses after checking only template-side matches."""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset,sample_crop,verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_poc import validate_split
from weapon_lamp_detect.match_data import TEMPLATES,atomic_json,now,read_json,write_image
from weapon_lamp_detect.region_matcher import OfficialRegionMatcher,OBSRegionMatcher,REGION,SIZES,SEARCH_SHIFT,weapon_region
from weapon_lamp_detect.weapon_lamp import WeaponIconMatcher


def check_training(dataset,templates,output,workers=4,template_dir=TEMPLATES):
    dataset,templates,output=map(lambda p:Path(p).resolve(),(dataset,templates,output))
    if (output/"configuration_frozen.json").exists() or (output/"report.json").exists():
        raise ValueError("固定済み設定を上書きしません。新しいoutputを指定してください")
    if workers<1:
        raise ValueError("workers ≥ 1 が必要です")
    metadata,rows=load_dataset(dataset)
    verify_sources(metadata)
    manifest=read_json(templates/"templates.json")
    evaluation_ids=validate_split(manifest,rows,dataset)
    used={i for e in manifest["templates"] for i in e["sample_ids"]}
    train=set(manifest["split"]["train_match_ids"])
    groups=defaultdict(list)
    for r in rows:
        if r["match_id"] in train and r["state"]=="alive" and r["sample_id"] not in used:
            groups[r["match_id"],r["side"],r["slot_index"]].append(r)
    chosen=[r for group in groups.values() for r in group[:2]]
    if not chosen:
        raise ValueError("template専用matchに、templateとして直接使用していない確認cropがありません")
    if {r["sample_id"] for r in chosen}&evaluation_ids:
        raise ValueError("調整用cropと評価cropが重複しています")
    previous=cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        base=WeaponIconMatcher.from_dir(Path(template_dir))
        official=[t for t in base.templates if t.name in manifest["weapons"]]
        matchers={"official_region":OfficialRegionMatcher(official),
                  "official_foreground":OfficialRegionMatcher(official,True),
                  "obs_region":OBSRegionMatcher(templates),"obs_foreground":OBSRegionMatcher(templates,True)}
        def predict(row):
            crop=sample_crop(dataset,row,metadata)
            result={}
            for name,matcher in matchers.items():
                ranked=matcher.predict_crop(crop,len(manifest["weapons"]),**({"side":row["side"]} if name.startswith("obs") else {}))
                result[name]=int(bool(ranked) and ranked[0].weapon==row["weapon_label"])
            return result
        with ThreadPoolExecutor(workers) as pool:
            results=list(pool.map(predict,chosen))
    finally:
        cv2.setNumThreads(previous)
    counts={name:{"correct":sum(r[name] for r in results),"support":len(chosen)} for name in matchers}
    result={"created_at":now(),"dataset_sha256":dataset_digest(dataset),"region":REGION,"sizes":SIZES,
            "search_shift":SEARCH_SHIFT,"training_only_checks":counts,"training_match_ids":sorted(train),
            "training_check_sample_ids":[r["sample_id"] for r in chosen],"evaluation_labels_used_for_tuning":False,
            "algorithm_sha256":hashlib.sha256((Path(__file__).parent/"region_matcher.py").read_bytes()).hexdigest(),
            "note":"Fixed hypothesis tested on training-only crops. No parameter search after held-out scores."}
    atomic_json(output/"configuration_frozen.json",result)
    for row in chosen[:8]:
        image,mask=weapon_region(sample_crop(dataset,row,metadata),True,True)
        write_image(output/"training_debug"/(row["sample_id"]+".png"),image)
        write_image(output/"training_debug"/(row["sample_id"]+"_mask.png"),mask)
    return result


def main():
    p=argparse.ArgumentParser(description="評価集合を使わず、template専用試合で領域/背景除去設定を確認・固定")
    p.add_argument("dataset",type=Path)
    p.add_argument("--obs-templates",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--template-dir",type=Path,default=TEMPLATES)
    a=p.parse_args()
    result=check_training(a.dataset,a.obs_templates,a.output,a.workers,a.template_dir)
    print(result["training_only_checks"])


if __name__=="__main__":
    main()
