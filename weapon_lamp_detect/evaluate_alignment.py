"""Paired crop/detail ablations; freeze on training data before reading test labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.adaptive_crop import CONFIG as CROP_CONFIG
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.detail_matcher import DETAIL_CONFIG, DetailMatcher
from weapon_lamp_detect.evaluate_poc import aggregate, prediction, summarize, validate_split
from weapon_lamp_detect.match_data import Catalog, atomic_json, canonical_crop, now, read_json, write_image
from weapon_lamp_detect.region_matcher import OBSRegionMatcher

_WORKER=None


def _initialize_worker(dataset,metadata,manifest,matchers):
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER=(dataset,metadata,manifest,matchers)


def _predict(row):
    dataset,metadata,manifest,matchers=_WORKER
    count=max(5,len(manifest["weapons"]))
    crop=sample_crop(dataset,row,metadata)
    result=[]
    for name,matcher in matchers.items():
        candidates=matcher.predict_crop(crop,count,side=row["side"])
        result.append(prediction(row,[c.to_dict() for c in candidates],name,
                                 manifest["split"]["evaluation_scope"].get(row["sample_id"],"training_check")))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithms():
    return {name:sha(Path(__file__).parent/name) for name in
            ("adaptive_crop.py","detail_matcher.py","region_matcher.py","evaluate_alignment.py")}


def rebase_manifest(source,dataset,output):
    """Keep all source sample IDs/splits, changing only their crop preprocessing."""
    source,dataset,output=map(Path,(source,dataset,output))
    original=read_json(source/"templates.json")
    metadata,rows=load_dataset(dataset)
    if metadata.get("source_dataset_sha256")!=dataset_digest(original["dataset"]):
        raise ValueError("補正datasetの元datasetがtemplateのdatasetと違います")
    by_id={r["sample_id"]:r for r in rows}
    m=copy.deepcopy(original)
    m["dataset"]=str(dataset.resolve())
    m["dataset_sha256"]=dataset_digest(dataset)
    m["original_templates_manifest_sha256"]=sha(source/"templates.json")
    m["registration_configuration"]=CROP_CONFIG
    validate_split(m,rows,dataset)
    if output.exists():
        existing=read_json(output/"templates.json")
        if existing!=m:
            raise ValueError("既存補正manifestと入力が違います")
        return output
    output.mkdir(parents=True)
    import numpy as np
    for entry in m["templates"]:
        images=[sample_crop(dataset,by_id[i],metadata) for i in entry["sample_ids"]]
        write_image(output/entry["path"],np.median(np.stack(images),axis=0).astype(np.uint8))
    atomic_json(output/"templates.json",m)
    return output


def predict_rows(rows,dataset,metadata,manifest,matchers,workers):
    if workers<1:
        raise ValueError("workers >= 1 が必要です")
    cv2.setNumThreads(1)
    output=[]
    import multiprocessing
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_worker,initargs=(dataset,metadata,manifest,matchers)) as pool:
        for i,results in enumerate(pool.map(_predict,rows,chunksize=8)):
            output.extend(results)
            if i%100==0:
                print(f"alignment evaluation: {i+1}/{len(rows)}",flush=True)
    return output


def freeze(experiment,dataset,output,workers=4):
    experiment,dataset,output=map(lambda p:Path(p).resolve(),(experiment,dataset,output))
    if (output/"configuration_frozen.json").exists():
        raise ValueError("固定済み設定は上書きしません")
    protocol=read_json(experiment/"protocol.json")
    source=experiment/"augmented_templates"
    templates=rebase_manifest(source,dataset,output/"training_templates")
    manifest=read_json(templates/"templates.json")
    metadata,rows=load_dataset(dataset)
    train=set(protocol["old_train_match_ids"]+protocol["new_train_match_ids"])
    evaluation=set(manifest["split"]["evaluation_sample_ids"])
    used_frames={(r["video_id"],r["frame_index"]) for r in rows
                 if r["sample_id"] in set(manifest["split"]["template_sample_ids"])}
    groups=defaultdict(list)
    for row in rows:
        if (row["match_id"] in train and row["state"]=="alive" and row["weapon_label"] in manifest["weapons"]
                and (row["video_id"],row["frame_index"]) not in used_frames):
            groups[row["match_id"],row["side"],row["slot_index"]].append(row)
    chosen=[r for g in groups.values() for r in sorted(g,key=lambda r:r["frame_index"])[:2]]
    if not chosen or {r["sample_id"] for r in chosen}&evaluation:
        raise ValueError("training checkが空またはevaluationと重複しています")
    matchers={"registered":OBSRegionMatcher(templates,True),
              "detail_median":DetailMatcher(templates),"detail_exemplars":DetailMatcher(templates,True)}
    results=predict_rows(chosen,dataset,metadata,manifest,matchers,workers)
    summaries={name:summarize([r for r in results if r["method"]==name]) for name in matchers}
    # Macro accuracy prevents repeated N-ZAP crops alone selecting a setting.
    def rank(name):
        import numpy as np
        summary=summaries[name]
        macro=float(np.mean([v["exact_top1"]["accuracy"] for v in summary["per_weapon"].values()]))
        return macro,summary["metrics"]["exact_top1"]["accuracy"],name=="detail_median"
    selected=max(("detail_median","detail_exemplars"),key=rank)
    result={"created_at":now(),"experiment":str(experiment),"dataset":str(dataset),
            "dataset_sha256":dataset_digest(dataset),"source_protocol_sha256":sha(experiment/"protocol.json"),
            "algorithm_sha256":algorithms(),"crop_configuration":CROP_CONFIG,"detail_configuration":DETAIL_CONFIG,
            "selected_detail":selected,"training_check_sample_ids":[r["sample_id"] for r in chosen],
            "training_match_ids":sorted(train),"selection":"highest training-only per-weapon macro top-1, then micro; no test-driven parameter changes",
            "training_only_results":summaries,"evaluation_labels_used_for_configuration_selection":False,
            "caveat":"Evaluation sets were inspected in earlier PoC iterations; this remains exploratory, not a fresh untouched final test."}
    output.mkdir(parents=True,exist_ok=True)
    (output/"training_predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results),encoding="utf-8")
    atomic_json(output/"configuration_frozen.json",result)
    print("frozen detail:",selected,{n:s["metrics"]["exact_top1"] for n,s in summaries.items()},flush=True)
    return result


def assert_frozen(output):
    frozen=read_json(Path(output)/"configuration_frozen.json")
    if frozen["algorithm_sha256"]!=algorithms():
        raise ValueError("固定後に実装が変わりました。新しい実験としてtraining確認をやり直してください")
    if frozen["dataset_sha256"]!=dataset_digest(frozen["dataset"]):
        raise ValueError("固定後にdatasetが変わりました")
    if frozen["source_protocol_sha256"]!=sha(Path(frozen["experiment"])/"protocol.json"):
        raise ValueError("固定後にsplit protocolが変わりました")
    return frozen


def paired(before,after):
    a,b={r["sample_id"]:r for r in before},{r["sample_id"]:r for r in after}
    if len(a)!=len(before) or len(b)!=len(after) or set(a)!=set(b):
        raise ValueError("比較するsample ID集合が違います")
    keys=("expected","weapon_label","state","scope","match_id","timestamp","frame_index","video_id","side","slot_index","match_revision")
    counts=Counter()
    for i,r in b.items():
        if any(a[i].get(k)!=r.get(k) for k in keys):
            raise ValueError("正解/state/scope/source情報が変更されています")
        if a[i]["rect"]!=r.get("original_rect",r["rect"]):
            raise ValueError("補正前rectのprovenanceが一致しません")
        x,y=a[i]["correct"],r["correct"]
        counts["both_correct" if x and y else "both_wrong" if not x and not y else "improved" if y else "regressed"]+=1
    return dict(counts)


def run(output,kind="independent",workers=4):
    output=Path(output).resolve()
    frozen=assert_frozen(output)
    experiment=Path(frozen["experiment"])
    if kind=="full_training":
        experiment=experiment/"full_additional_training"
    elif kind!="independent":
        raise ValueError("unknown evaluation kind")
    directory=output/kind
    if (directory/"report.json").exists():
        raise ValueError("既存reportは上書きしません")
    dataset=Path(frozen["dataset"])
    templates=rebase_manifest(experiment/"augmented_templates",dataset,directory/"templates")
    manifest=read_json(templates/"templates.json")
    metadata,rows=load_dataset(dataset)
    verify_sources(metadata)
    selected=validate_split(manifest,rows,dataset)
    checks=set(frozen["training_check_sample_ids"])
    if selected&checks:
        raise ValueError("training checkと評価が重複しています")
    eval_rows=[r for r in rows if r["sample_id"] in selected]
    detail_name="obs_detail_foreground"
    matchers={"obs_registered_foreground":OBSRegionMatcher(templates,True,directory/"processed_templates"/"obs_registered_foreground"),
              detail_name:DetailMatcher(templates,frozen["selected_detail"]=="detail_exemplars",directory/"processed_templates"/detail_name)}
    results=predict_rows(eval_rows,dataset,metadata,manifest,matchers,workers)
    baseline_report=read_json(experiment/"augmented_evaluation/report.json")
    baseline=[json.loads(l) for l in (experiment/"augmented_evaluation/predictions.jsonl").read_text().splitlines()]
    if baseline_report["candidate_universe"]!=manifest["weapons"]:
        raise ValueError("候補武器集合が変更されています")
    paired_changes={name:paired(baseline,[r for r in results if r["method"]==name]) for name in matchers}
    # Keep original crops available alongside recrops in one viewer dataset.
    for row in baseline:
        old=Path(baseline_report["dataset"])/row["crop"]
        relative="original_"+row["crop"]
        target=dataset/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:
                os.link(old,target)
            except OSError:
                import shutil
                shutil.copyfile(old,target)
        row["crop"]=relative
        row["method"]="obs_baseline_foreground"
    results=baseline+results
    methods=sorted({r["method"] for r in results})
    all_summaries={name:{"single_frame":summarize([r for r in results if r["method"]==name]),
                         "multi_frame":aggregate([r for r in results if r["method"]==name])} for name in methods}
    primary={name:summarize([r for r in results if r["method"]==name and r["state"]=="alive" and r["scope"]=="cross_match"]) for name in methods}
    report={"created_at":now(),"dataset":str(dataset),"dataset_sha256":dataset_digest(dataset),
            "candidate_universe":manifest["weapons"],"methods":all_summaries,"paired_alive_cross_match":primary,
            "configuration":frozen,"evaluation_kind":kind,"paired_changes_all_states":paired_changes,
            "provenance":{name:{"templates_manifest":manifest,"preprocessor":"detail" if name==detail_name else "legacy_foreground"}
                          for name in matchers},"baseline_report":str(experiment/"augmented_evaluation/report.json"),
            "missing_templates":{name:manifest.get("missing_obs_templates",[]) for name in methods},
            "note":"Same sample IDs, labels, state, scope, frame indices and candidate weapons. Only crop geometry / matcher changes. Unreliable crop detection remains in denominator via original-crop fallback."}
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in results),encoding="utf-8")
    atomic_json(directory/"report.json",report)
    for name in methods:
        print(kind,name,primary[name]["metrics"]["exact_top1"],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description="trainingだけで補正/照合設定を固定し、同じsampleで比較")
    commands=p.add_subparsers(dest="command",required=True)
    f=commands.add_parser("freeze")
    f.add_argument("experiment",type=Path)
    f.add_argument("dataset",type=Path)
    f.add_argument("--output",type=Path,required=True)
    f.add_argument("--workers",type=int,default=4)
    r=commands.add_parser("run")
    r.add_argument("output",type=Path)
    r.add_argument("--kind",choices=["independent","full_training"],default="independent")
    r.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.command=="freeze":
        freeze(a.experiment,a.dataset,a.output,a.workers)
    else:
        run(a.output,a.kind,a.workers)


if __name__=="__main__":
    main()
