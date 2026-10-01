"""Evaluate additional annotations without reshuffling the previous hold-out.

Keep ROI/scoring frozen. Reserve 1/3 of additional matches by hash for new
templates, append train-only median exemplars, and compare on identical crops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_poc import aggregate, evaluate, summarize, validate_split
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json, validate_match, write_image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_added_matches(matches, seed=7):
    matches = sorted(set(matches), key=lambda m: hashlib.sha256(f"{seed}:{m}".encode()).hexdigest())
    if len(matches) < 2:
        raise ValueError("追加データは最低2試合必要です")
    train = sorted(matches[:max(1, len(matches)//3)])
    test = sorted(set(matches)-set(train))
    return train, test


def prepare(session, baseline_dataset, baseline_templates, output, seed=7, baseline_configuration=None):
    session, baseline_dataset, baseline_templates, output = map(
        lambda p: Path(p).resolve(), (session, baseline_dataset, baseline_templates, output))
    if output.exists():
        raise ValueError("新しいoutput directoryを指定してください")
    catalog = Catalog()
    videos = read_json(session/"videos.json")
    paths = sorted((session/"matches").glob("*.json"))
    matches = [read_json(p) for p in paths]
    matches = [m for m in matches if m.get("confirmed") and not m.get("rejected")]
    matches.sort(key=lambda m: (m["video_id"], m["start_timestamp"]))
    previous_end = {}
    for m in matches:
        validate_match(m, videos[m["video_id"]], catalog)
        if m["start_timestamp"] < previous_end.get(m["video_id"], -1):
            raise ValueError("確定試合の区間が重複しています")
        previous_end[m["video_id"]] = m["end_timestamp"]
    old_metadata, old_rows = load_dataset(baseline_dataset)
    manifest = read_json(baseline_templates/"templates.json")
    validate_split(manifest, old_rows, baseline_dataset)
    current = {m["match_id"]:m for m in matches}
    # Reuse exported old crops only when labels, geometry and source are identical.
    fields = ("video_id", "start_timestamp", "end_timestamp", "reference_timestamp",
              "reference_frame_index", "slots", "geometry", "ally_side", "revision", "hud_offset_seconds")
    for old in old_metadata["matches"]:
        if old["match_id"] not in current or any(current[old["match_id"]].get(k)!=old.get(k) for k in fields):
            raise ValueError(f"前回GTが変更されています。再抽出・比較計画の更新が必要: {old['match_id']}")
    old_ids = {m["match_id"] for m in old_metadata["matches"]}
    added = [m for m in matches if m["match_id"] not in old_ids]
    train, test = freeze_added_matches([m["match_id"] for m in added], seed)
    verify_sources({"videos":{v:videos[v] for v in {m["video_id"] for m in matches}}})
    config_path = Path(baseline_configuration) if baseline_configuration else baseline_templates.parent/"poc_obs_20261001_regions"/"configuration_frozen.json"
    frozen = read_json(config_path)
    if frozen.get("algorithm_sha256") not in (None, sha(Path(__file__).parent/"region_matcher.py")):
        raise ValueError("前回固定した照合実装が変更されています")
    output.mkdir(parents=True)
    snapshot = output/"snapshot_session"
    atomic_json(snapshot/"videos.json", {v:videos[v] for v in {m["video_id"] for m in added}})
    for m in added:
        atomic_json(snapshot/"matches"/(m["match_id"]+".json"), m)
    protocol = {"schema_version":1, "created_at":now(), "source_session":str(session),
                "baseline_dataset":str(baseline_dataset), "baseline_templates":str(baseline_templates),
                "baseline_dataset_sha256":dataset_digest(baseline_dataset),
                "baseline_manifest_sha256":sha(baseline_templates/"templates.json"),
                "algorithm_sha256":sha(Path(__file__).parent/"region_matcher.py"),
                "original_configuration":frozen, "original_configuration_path":str(config_path.resolve()), "seed":seed,
                "new_train_match_ids":train, "new_test_match_ids":test,
                "old_train_match_ids":manifest["split"]["train_match_ids"],
                "old_evaluation_sample_ids":manifest["split"]["evaluation_sample_ids"],
                "baseline_weapons":manifest["weapons"],
                "expanded_weapons":sorted({s["weapon"] for m in matches for s in m["slots"].values() if s["status"]=="labeled"}),
                "matches":matches, "videos":videos,
                "original_gt_sha256":{str(p):sha(p) for p in paths},
                "sampling":{"interval":5,"max_frames_per_match":20,"end_offset":5},
                "template_growth":"retain original templates; append one median per new training match/weapon/side, earliest 3 distinct alive source frames",
                "evaluation_labels_used_for_tuning":False}
    atomic_json(output/"protocol.json", protocol)
    return {"added_matches":len(added),"new_train_matches":len(train),"new_test_matches":len(test),
            "baseline_weapons":len(manifest["weapons"]),"expanded_weapons":len(protocol["expanded_weapons"])}


def link_export(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):
            raise ValueError("export先に異なる画像があります")
        return
    try:
        os.link(source,target)
    except OSError:
        shutil.copy2(source,target)  # Tiny crop/context only; never copy videos.


def merge_datasets(experiment, protocol):
    combined = experiment/"dataset"
    if (combined/"dataset.json").exists():
        metadata, rows = load_dataset(combined)
        if metadata.get("experiment_protocol_sha256")!=sha(experiment/"protocol.json"):
            raise ValueError("既存datasetのprotocolが異なります")
        if metadata["source_dataset_sha256"]!=[dataset_digest(p) for p in metadata["source_datasets"]]:
            raise ValueError("統合元datasetが変更されています")
        return combined, metadata, rows
    old = Path(protocol["baseline_dataset"])
    added = Path(protocol.get("new_dataset",experiment/"new_dataset"))
    old_meta, old_rows = load_dataset(old)
    added_meta, added_rows = load_dataset(added)
    if dataset_digest(old)!=protocol["baseline_dataset_sha256"]:
        raise ValueError("前回datasetが変更されました")
    if {m["match_id"] for m in added_meta["matches"]}!=set(protocol["new_train_match_ids"]+protocol["new_test_match_ids"]):
        raise ValueError("追加datasetと固定分割の試合集合が異なります")
    expected_matches={m["match_id"]:m for m in protocol["matches"]}
    if any(m!=expected_matches[m["match_id"]] for m in added_meta["matches"]):
        raise ValueError("追加datasetの正解snapshotが固定時と異なります")
    expected = protocol["sampling"]
    for meta in (old_meta,added_meta):
        if (meta["sampling_interval"]!=expected["interval"] or meta["max_frames_per_match"]!=expected["max_frames_per_match"]
                or meta["end_offset"]!=expected["end_offset"] or meta["start_offset"] is not None
                or meta["max_samples"]!=0 or set(meta["states"])!={"alive","down","unknown"}):
            raise ValueError("前回と異なるsampling条件です")
    rows = sorted(old_rows+added_rows,key=lambda r:r["sample_id"])
    if len({r["sample_id"] for r in rows})!=len(rows):
        raise ValueError("重複sample IDがあります")
    combined.mkdir(parents=True,exist_ok=True)
    for source, samples in ((old,old_rows),(added,added_rows)):
        exported=set()
        for row in samples:
            for key in ("crop","context"):
                rel=row.get(key)
                if not rel:
                    raise ValueError("--export-cropsで抽出してください")
                path=(source/rel).resolve()
                if source.resolve() not in path.parents:
                    raise ValueError("dataset外のexport pathです")
                if rel not in exported:
                    link_export(path,combined/rel)
                    exported.add(rel)
    metadata={**old_meta,"created_at":now(),"session_dir":protocol["source_session"],
              "matches":old_meta["matches"]+added_meta["matches"],
              "videos":{**old_meta["videos"],**added_meta["videos"]},
              "source_datasets":[str(old),str(added)],"source_dataset_sha256":[dataset_digest(old),dataset_digest(added)],
              "experiment_protocol_sha256":sha(experiment/"protocol.json"),"total_samples":len(rows),
              "samples":len(rows),"samples_per_match":dict(Counter(r["match_id"] for r in rows)),
              "effective_start_offsets":{**old_meta["effective_start_offsets"],**added_meta["effective_start_offsets"]},
              "excluded":dict(Counter(old_meta["excluded"])+Counter(added_meta["excluded"])),
              "composition_note":"immutable original crops plus separately extracted new video; images hardlinked where possible"}
    (combined/"samples.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    atomic_json(combined/"dataset.json",metadata)
    return combined,metadata,rows


def growth_manifest(experiment, protocol, dataset, metadata, rows, kind):
    output=experiment/(kind+"_templates")
    if (output/"templates.json").exists():
        existing=read_json(output/"templates.json")
        if existing.get("experiment_protocol_sha256")!=sha(experiment/"protocol.json"):
            raise ValueError("templateのprotocolが異なります")
        validate_split(existing,rows,dataset)
        return output
    original=read_json(Path(protocol["baseline_templates"])/"templates.json")
    if sha(Path(protocol["baseline_templates"])/"templates.json")!=protocol["baseline_manifest_sha256"]:
        raise ValueError("元template manifestが変更されました")
    names=protocol["expanded_weapons"] if kind=="expanded" else protocol["baseline_weapons"]
    names_set=set(names)
    entries=[dict(e) for e in original["templates"]]
    output.mkdir(parents=True,exist_ok=True)
    for e in entries:
        link_export(Path(protocol["baseline_templates"])/e["path"],output/e["path"])
    template_ids=set(original["split"]["template_sample_ids"])
    if kind!="baseline":
        train=set(protocol["new_train_match_ids"])
        groups=defaultdict(list)
        for r in rows:
            if r["match_id"] in train and r["state"]=="alive" and r["weapon_label"] in names_set:
                groups[r["match_id"],r["weapon_label"],r["side"]].append(r)
        for key, group in sorted(groups.items()):
            unique={}
            for r in sorted(group,key=lambda r:(r["frame_index"],r["sample_id"])):
                unique.setdefault((r["video_id"],r["frame_index"]),r)
            chosen=list(unique.values())[:3]
            images=[sample_crop(dataset,r,metadata) for r in chosen]
            median=np.median(np.stack(images),axis=0).astype(np.uint8)
            path=f"added_median_{len(entries):03d}.png"
            write_image(output/path,median)
            entries.append({"weapon":key[1],"weapon_class":chosen[0]["weapon_class"],"side":key[2],"path":path,
                            "method":"unaligned pixel median of normalized HUD crops; one exemplar per training match/weapon/side",
                            "sample_ids":[r["sample_id"] for r in chosen],
                            "source_frames":[{k:r[k] for k in ("video_id","match_id","timestamp","frame_index","side","slot_index")} for r in chosen]})
            template_ids.update(r["sample_id"] for r in chosen)
    by_id={r["sample_id"]:r for r in rows}
    old_eval=set(protocol["old_evaluation_sample_ids"])
    new_test=set(protocol["new_test_match_ids"])
    evaluation=[r for r in rows if r["weapon_label"] in names_set and
                (r["sample_id"] in old_eval or r["match_id"] in new_test)]
    train_by_weapon=defaultdict(set)
    for i in template_ids:
        train_by_weapon[by_id[i]["weapon_label"]].add(by_id[i]["match_id"])
    split={"template_sample_ids":sorted(template_ids),"evaluation_sample_ids":[r["sample_id"] for r in evaluation],
           "train_match_ids":sorted(set(protocol["old_train_match_ids"]+protocol["new_train_match_ids"])),
           "evaluation_scope":{r["sample_id"]:("within_match_other_timestamp" if r["match_id"] in train_by_weapon[r["weapon_label"]] else "cross_match") for r in evaluation},
           "within_match_template_sample_ids":original["split"]["within_match_template_sample_ids"],"seed":protocol["seed"]}
    manifest={**original,"created_at":now(),"dataset":str(dataset),"dataset_sha256":dataset_digest(dataset),
              "weapons":names,"max_candidate_weapons":len(names),"templates":entries,"split":split,
              "missing_obs_templates":sorted(names_set-{e["weapon"] for e in entries}),
              "experiment_protocol_sha256":sha(experiment/"protocol.json"),"experiment_kind":kind,
              "note":"New held-out matches are never used for template construction, including rare weapons."}
    validate_split(manifest,rows,dataset)
    atomic_json(output/"templates.json",manifest)
    return output


def frozen_for_run(protocol, dataset):
    return {**protocol["original_configuration"],"created_at":now(),"dataset_sha256":dataset_digest(dataset),
            "algorithm_sha256":protocol["algorithm_sha256"],
            "note":"Original ROI/color/scoring copied unchanged. Only additional training-match exemplars are appended; no parameter tuning."}


def load_predictions(directory):
    return [json.loads(line) for line in (directory/"predictions.jsonl").open(encoding="utf-8")]


def paired(before, after):
    left={r["sample_id"]:r for r in before}
    right={r["sample_id"]:r for r in after}
    if set(left)!=set(right):
        raise ValueError("比較するsample集合が異なります")
    for i,r in left.items():
        if any(r.get(k)!=right[i].get(k) for k in ("source_video","frame_index","expected","state","scope","rect","match_revision")):
            raise ValueError("比較するsampleの正解/状態/由来が異なります")
    return {"support":len(left),"wrong_to_correct":sum(not r["correct"] and right[i]["correct"] for i,r in left.items()),
            "correct_to_wrong":sum(r["correct"] and not right[i]["correct"] for i,r in left.items()),
            "both_correct":sum(r["correct"] and right[i]["correct"] for i,r in left.items()),
            "both_wrong":sum(not r["correct"] and not right[i]["correct"] for i,r in left.items())}


def summarize_experiment(experiment, protocol, dataset):
    comparison=experiment/"comparison"
    if (comparison/"report.json").exists():
        raise ValueError("既存comparisonを上書きしません")
    runs={kind:read_json(experiment/(kind+"_evaluation")/"report.json") for kind in ("baseline","augmented","expanded")}
    predictions={kind:load_predictions(experiment/(kind+"_evaluation")) for kind in runs}
    old_ids=set(protocol["old_evaluation_sample_ids"])
    test_ids=set(protocol["new_test_match_ids"])
    subsets={"old_fixed_holdout":lambda r:r["sample_id"] in old_ids,
             "new_fixed_holdout":lambda r:r["match_id"] in test_ids}
    primary=lambda r:r["state"]=="alive" and r["scope"]=="cross_match"
    sections={}
    for label, select in subsets.items():
        sections[label]={}
        for kind,rows in predictions.items():
            chosen=[r for r in rows if select(r) and primary(r)]
            sections[label][kind]={"single_frame":summarize(chosen),"multi_frame":aggregate(chosen)}
        sections[label]["paired_58_weapons"]=paired([r for r in predictions["baseline"] if select(r) and primary(r)],
                                                     [r for r in predictions["augmented"] if select(r) and primary(r)])
    # Also test the effect of adding competing weapon classes on the same known crops.
    known=set(protocol["baseline_weapons"])
    extended_on_known=[r for r in predictions["expanded"] if r["expected"] in known]
    sections["expanded_candidates_on_known_58"]={}
    for label,select in subsets.items():
        sections["expanded_candidates_on_known_58"][label]={"single_frame":summarize([r for r in extended_on_known if select(r) and primary(r)]),
                                                         "multi_frame":aggregate([r for r in extended_on_known if select(r) and primary(r)])}
    old_metadata,old_rows=load_dataset(protocol["baseline_dataset"])
    actual_old_primary={r["sample_id"] for r in predictions["baseline"] if r["sample_id"] in old_ids and primary(r)}
    actual_old_rows={r["sample_id"] for r in old_rows if r["sample_id"] in actual_old_primary}
    if actual_old_rows!=actual_old_primary:
        raise ValueError("前回の固定評価sampleが維持されていません")
    old_manifest=read_json(Path(protocol["baseline_templates"])/"templates.json")
    expected_old_primary={r["sample_id"] for r in old_rows if r["sample_id"] in old_ids and r["state"]=="alive"
                          and old_manifest["split"]["evaluation_scope"][r["sample_id"]]=="cross_match"}
    if actual_old_primary!=expected_old_primary:
        raise ValueError("前回alive/cross-matchの全sampleが保持されていません")
    all_rows=[]; methods={}; provenance={}; missing={}; candidates={}
    aliases={"baseline":"baseline_obs_foreground","augmented":"obs_foreground","expanded":"expanded_obs_foreground"}
    for kind,run in runs.items():
        name=aliases[kind]
        methods[name]=run["methods"]["obs_foreground"]
        provenance[name]=run["provenance"]["obs_foreground"]
        missing[name]=run["missing_templates"]["obs_foreground"]
        candidates[name]=run["candidate_universe"]
        all_rows.extend({**r,"method":name} for r in predictions[kind])
    report={"schema_version":1,"created_at":now(),"dataset":str(dataset),"dataset_sha256":dataset_digest(dataset),
            "candidate_universe":protocol["expanded_weapons"],"candidate_universe_by_method":candidates,
            "closed_set":True,"source":"human-labeled OBS recordings, fixed old holdout and additional match split",
            "methods":methods,"provenance":provenance,"missing_templates":missing,
            "protocol":protocol,"paired_comparisons":sections,
            "state_note":"Automatic alive/down/unknown; includes imperfect HUD gating, unchanged from baseline.",
            "comparison_note":"Only baseline vs augmented have identical 58-weapon candidates and samples; expanded 105-weapon evaluation is separate.",
            "evaluation_note":("Additional-video holdout is independent of template construction; old holdout was previously observed."
                               if protocol["new_test_match_ids"] else "All additional-video matches are template-only. Only previously observed old holdout is evaluated; no independent accuracy on the new video."),
            "aggregation_note":"Earliest N alive frames; common_cohort needs >=10 frames; no new parameter tuning."}
    comparison.mkdir(parents=True,exist_ok=True)
    (comparison/"predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in all_rows),encoding="utf-8")
    atomic_json(comparison/"report.json",report)
    design=("追加動画は試合単位で1/3をtemplate用、残りを評価用に固定。" if protocol["new_test_match_ids"]
            else "追加動画の全試合をtemplate専用にし、前回の固定holdoutだけで評価。追加動画での精度は測定しない。")
    lines=["# 追加ラベルによるテンプレート増強評価","", "照合アルゴリズム・ROI・背景除去・scoreは変更していない。"+design, "",
           "| 評価集合 | 候補数 | テンプレート | alive/cross support | top-1 | top-3 | top-5 |","|---|---:|---|---:|---:|---:|---:|"]
    for label in subsets:
        for kind in runs:
            m=sections[label][kind]["single_frame"]["metrics"]
            values=["—" if m[f"exact_top{k}"]["accuracy"] is None else f'{m[f"exact_top{k}"]["accuracy"]:.2%}' for k in (1,3,5)]
            lines.append(f'| {label} | {len(runs[kind]["candidate_universe"])} | {kind} | {m["support"]} | '+" | ".join(values)+" |")
    lines += ["", "## 同じcohortでの複数frame平均","", "| 評価集合 | テンプレート | 1 frame | 3 frames | 5 frames | 10 frames |","|---|---|---:|---:|---:|---:|"]
    for label in subsets:
        for kind in runs:
            values=[]
            for n in (1,3,5,10):
                m=sections[label][kind]["multi_frame"]["mean_score"][str(n)]["common_cohort"]["metrics"]["exact_top1"]
                values.append("—" if m["accuracy"] is None else f'{m["accuracy"]:.2%} ({m["correct"]}/{m["support"]})')
            lines.append(f'| {label} | {kind} | '+" | ".join(values)+" |")
    lines += ["", "## 注意", "", f'- 追加試合: template {len(protocol["new_train_match_ids"])} / held-out {len(protocol["new_test_match_ids"])}。全登録武器 {len(protocol["expanded_weapons"])}種。',
              "- 58種の比較では新規武器の正解cropは含めない。105種の評価ではtemplate欠損の武器も失敗としてsupportに残す。",
              ("- old_fixed_holdoutは過去の改善実験で見た集合。独立の検証はnew_fixed_holdout（ただし録画は1本なので試合間・撮影条件の相関は残る）。" if protocol["new_test_match_ids"]
               else "- old_fixed_holdoutは過去に見た評価集合。全量投入runに新動画の独立holdoutはなく、new_fixed_holdout欄は未評価。"),
              "- 少数frameの同一match-slotは独立した試合数として扱わない。within-matchは主評価に混ぜない。",
              "- 前回と同じHUD gate・samplingなので、mapや下部表示等の影響は残る。今回のデータ増量とアルゴリズム変更は混同しない。",
              "- 詳細な武器別指標・confusion・margin・正誤遷移はreport.json。誤判定画像はcomparisonをview_errors.pyで開く。"]
    (comparison/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return sections


def run(experiment, workers=8):
    experiment=Path(experiment).resolve()
    protocol=read_json(experiment/"protocol.json")
    if sha(Path(__file__).parent/"region_matcher.py")!=protocol["algorithm_sha256"]:
        raise ValueError("分割固定後に照合実装が変更されました")
    dataset,metadata,rows=merge_datasets(experiment,protocol)
    for kind in ("baseline","augmented","expanded"):
        templates=growth_manifest(experiment,protocol,dataset,metadata,rows,kind)
        output=experiment/(kind+"_evaluation")
        if (output/"report.json").exists():
            previous=read_json(output/"report.json")
            if previous["dataset_sha256"]!=dataset_digest(dataset) or previous["provenance"]["obs_foreground"]["templates_manifest"]!=read_json(templates/"templates.json"):
                raise ValueError("既存runとdataset/templateが異なります")
            print(f"resume {kind}: existing completed report",flush=True)
            continue
        atomic_json(output/"configuration_frozen.json",frozen_for_run(protocol,dataset))
        print(f"RUN {kind}: templates={templates}",flush=True)
        evaluate(dataset,output,"obs_foreground",templates,max_weapons=len(protocol["expanded_weapons"]),workers=workers,executor="process")
    results=summarize_experiment(experiment,protocol,dataset)
    print(json.dumps({label:{kind:results[label][kind]["single_frame"]["metrics"] for kind in ("baseline","augmented","expanded")} for label in ("old_fixed_holdout","new_fixed_holdout")},ensure_ascii=False,indent=2))


def full_training(parent, workers=8):
    """Use all additional labels, but never evaluate their source matches."""
    parent=Path(parent).resolve()
    output=parent/"full_additional_training"
    original=read_json(parent/"protocol.json")
    if not (output/"protocol.json").exists():
        protocol={**original,"created_at":now(),"new_dataset":str(parent/"new_dataset"),
                  "new_train_match_ids":sorted(original["new_train_match_ids"]+original["new_test_match_ids"]),
                  "new_test_match_ids":[],"parent_protocol_sha256":sha(parent/"protocol.json"),
                  "experiment_note":"All additional matches become templates. Only the old fixed holdout is evaluated; no accuracy claim on the new recording."}
        atomic_json(output/"protocol.json",protocol)
    else:
        protocol=read_json(output/"protocol.json")
        if protocol.get("parent_protocol_sha256")!=sha(parent/"protocol.json"):
            raise ValueError("元の固定protocolが変更されています")
    if set(protocol["new_train_match_ids"])!=set(original["new_train_match_ids"]+original["new_test_match_ids"]) or protocol["new_test_match_ids"]:
        raise ValueError("全追加試合をtemplate専用にした分割ではありません")
    run(output,workers)


def main():
    p=argparse.ArgumentParser(description="追加OBSラベルのtemplate増強を固定holdoutで測定")
    commands=p.add_subparsers(dest="command",required=True)
    a=commands.add_parser("prepare")
    a.add_argument("--session-dir",type=Path,required=True)
    a.add_argument("--baseline-dataset",type=Path,required=True)
    a.add_argument("--baseline-templates",type=Path,required=True)
    a.add_argument("--baseline-configuration",type=Path,help="前回固定したconfiguration_frozen.json")
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--seed",type=int,default=7)
    b=commands.add_parser("run")
    b.add_argument("experiment",type=Path)
    b.add_argument("--workers",type=int,default=8)
    c=commands.add_parser("full-training",help="追加全試合をtemplateに使い、前回holdoutだけで測定")
    c.add_argument("experiment",type=Path)
    c.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    if args.command=="prepare":
        print(prepare(args.session_dir,args.baseline_dataset,args.baseline_templates,args.output,args.seed,args.baseline_configuration))
    elif args.command=="run":
        run(args.experiment,args.workers)
    else:
        full_training(args.experiment,args.workers)


if __name__=="__main__":
    main()
