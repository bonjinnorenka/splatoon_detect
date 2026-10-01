"""Reuse baseline predictions and compare only exactly paired evaluation crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.match_data import Catalog,atomic_json,now,read_json
from weapon_lamp_detect.evaluate_poc import summarize


def load_predictions(directory):
    return [json.loads(line) for line in (directory/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def paired_rows(baseline, alternative):
    keys=("weapon_label","expected","state","scope","match_id","timestamp","frame_index","video_id",
          "source_video","side","slot_index","rect","match_revision")
    by_method={}
    for row in baseline+alternative:
        group=by_method.setdefault(row["method"],{})
        if row["sample_id"] in group:
            raise ValueError("method内でsampleが重複しています")
        group[row["sample_id"]]=row
    if not by_method:
        raise ValueError("予測が空です")
    reference=next(iter(by_method.values()))
    for group in by_method.values():
        if set(group)!=set(reference):
            raise ValueError("評価sample集合が違うため改善差を比較できません")
        for sample_id,row in group.items():
            if any(row.get(key)!=reference[sample_id].get(key) for key in keys):
                raise ValueError("正解/state/scope/source情報が一致しません")
    return by_method


def pct(value):
    return "—" if value is None else f"{value:.2%}"


def compare_runs(baseline,alternative,output,refresh_summary=False):
    baseline,alternative,output=map(lambda p:Path(p).resolve(),(baseline,alternative,output))
    if (output/"report.json").exists():
        existing=read_json(output/"report.json")
        if not refresh_summary or existing.get("source_reports")!=[str(baseline),str(alternative)]:
            raise ValueError("既存reportは上書きしません（同じ入力のsummary再生成だけ --refresh-summary で許可）")
    a,b=read_json(baseline/"report.json"),read_json(alternative/"report.json")
    if a["dataset_sha256"]!=b["dataset_sha256"] or a["candidate_universe"]!=b["candidate_universe"]:
        raise ValueError("datasetまたは候補武器集合が違います")
    if set(a["methods"]) & set(b["methods"]):
        raise ValueError("方式名が重複しています")
    old,new=load_predictions(baseline),load_predictions(alternative)
    if refresh_summary and (output/"predictions.jsonl").exists():
        expected=hashlib.sha256()
        for row in old+new:
            expected.update((json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n").encode())
        actual=hashlib.sha256((output/"predictions.jsonl").read_bytes()).hexdigest()
        if actual!=expected.hexdigest():
            raise ValueError("元runの予測または保存済み比較予測が変わっています。summaryのみの再生成はできません")
    groups=paired_rows(old,new)
    parent=a["provenance"]["obs"]["templates_manifest"]
    for name,provenance in b["provenance"].items():
        if name.startswith("obs") and provenance["templates_manifest"]!=parent:
            raise ValueError("OBS templateの元sampleまたはsplitが変わっています")
    report={**b,"created_at":now(),"methods":{**a["methods"],**b["methods"]},
            "provenance":{**a["provenance"],**b["provenance"]},
            "missing_templates":{**a["missing_templates"],**b["missing_templates"]},
            "paired_comparison":True,"source_reports":[str(baseline),str(alternative)],
            "evaluation_note":"Exploratory improvement on the same previously observed held-out set; a new recording remains untested."}
    for name in b["provenance"]:
        if name.startswith("obs"):
            paths=sorted((alternative/"processed_templates"/name).glob("*.png"))
            report["provenance"][name]["processed_template_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    catalog=Catalog()
    display=lambda name:catalog.entries.get(name,{}).get("display_name",name)
    primary={name:summarize([r for r in rows.values() if r["state"]=="alive" and r["scope"]=="cross_match"])
             for name,rows in groups.items()}
    report["paired_alive_cross_match"]=primary
    report["delta_percentage_points"]={}
    report["background_only_delta_percentage_points"]={}
    for family in ("official","obs"):
        region=primary[family+"_region"]["metrics"]
        foreground=primary[family+"_foreground"]["metrics"]
        report["background_only_delta_percentage_points"][family]={f"top{k}":100*(foreground[f"exact_top{k}"]["accuracy"]-region[f"exact_top{k}"]["accuracy"])
                                                                       for k in (1,3,5) if region[f"exact_top{k}"]["accuracy"] is not None}
    report["paired_changes"]={}
    for name in b["methods"]:
        base="obs" if name.startswith("obs") else "official"
        counts={"improved":0,"regressed":0,"both_correct":0,"both_incorrect":0}
        for sample_id,row in groups[name].items():
            if row["state"]!="alive" or row["scope"]!="cross_match":
                continue
            before=groups[base][sample_id]["correct"]
            after=row["correct"]
            key="both_correct" if before and after else "both_incorrect" if not before and not after else "improved" if after else "regressed"
            counts[key]+=1
        report["paired_changes"][name]=counts
        report["delta_percentage_points"][name]={}
        for k in (1,3,5):
            before=primary[base]["metrics"][f"exact_top{k}"]["accuracy"]
            after=primary[name]["metrics"][f"exact_top{k}"]["accuracy"]
            report["delta_percentage_points"][name][f"top{k}"]=100*(after-before) if before is not None and after is not None else None
    lines=["# 背景除去・照合領域変更の実OBS比較","",
           "前回と同じ候補58種・同じ1,512評価crop・同じsplit。主評価はalive/cross_match 707crop。",
           "設定はテンプレート専用4試合の63cropで動作確認後に固定。CNN・GPUは使用していない。",
           "過去に見た評価集合での改善実験であり、新しい動画への汎化を保証しない。","",
           "| 方式 | support | top-1 | top-3 | top-5 | 元方式からのtop-1差 |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,s in primary.items():
        m=s["metrics"];delta=report["delta_percentage_points"].get(name,{}).get("top1")
        values=[pct(m[f"exact_top{k}"]["accuracy"]) for k in (1,3,5)]
        lines.append(f"| {name} | {m['support']} | "+" | ".join(values)+f" | {'—' if delta is None else f'{delta:+.2f}pt'} |")
    lines += ["","## 各変更の意味","",
              "- official：前回の公式画像方式。obs：前回の全lamp pixel median方式。",
              "- official_region：固定武器ROI、54/70px template、masked zero-mean相関、gray/Lab/edgeのスコア再構成。単純なcropだけの差ではない。",
              "- official_foreground：official_regionと同じ設定にインク色の除去を追加。",
              "- obs_region：同じ元sampleから固定ROIのmedianを再生成。raw gray/Lab/edgeで±4pxの位置補正。cropだけでなく位置補正も含む。",
              "- obs_foreground：obs_regionと同じ設定に、median作成前と推論時のインク色除去を追加。",
              "- インク色は既存squid detectorの色推定を再利用し、同系色・高彩度のpixelを中立灰色128に置換する。色が武器自体と重なる場合は情報を落とすリスクがある。","",
              f"背景除去だけの追加top-1差（region→foreground）：OBS {report['background_only_delta_percentage_points']['obs'].get('top1',0):+.2f}pt、公式画像 {report['background_only_delta_percentage_points']['official'].get('top1',0):+.2f}pt。","",
              "## 同じcropの正誤遷移","",
              "改善＝元方式の誤り→変更方式の正解。悪化＝元方式の正解→変更方式の誤り。",
              "| 方式 | 改善 | 悪化 | 両方正解 | 両方誤り |","|---|---:|---:|---:|---:|"]
    for name,counts in report["paired_changes"].items():
        lines.append(f"| {name} | {counts['improved']} | {counts['regressed']} | {counts['both_correct']} | {counts['both_incorrect']} |")
    lines += ["",
              "## 同一cohort：alive/cross_match、score平均top-1","",
              "同じ46match-slotで比較。707cropのframe指標とは分母が異なる。",
              "| 方式 | 1 frame | 3 frames | 5 frames | 10 frames |","|---|---:|---:|---:|---:|"]
    for name,result in report["methods"].items():
        values=[pct(result["multi_frame"]["mean_score"][str(n)]["common_cohort"]["by_state_and_scope"]["alive"]["cross_match"]["exact_top1"]["accuracy"]) for n in (1,3,5,10)]
        lines.append(f"| {name} | "+" | ".join(values)+" |")
    lines += ["","## 武器別：alive/cross_match","",
              "| 武器 | support | official | official_region | official_foreground | obs | obs_region | obs_foreground |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    weapons=sorted(primary["official"]["per_weapon"],key=display)
    order=("official","official_region","official_foreground","obs","obs_region","obs_foreground")
    for weapon in weapons:
        support=primary["official"]["per_weapon"][weapon]["support"]
        values=[pct(primary[name]["per_weapon"][weapon]["exact_top1"]["accuracy"]) for name in order]
        lines.append(f"| {display(weapon)} | {support} | "+" | ".join(values)+" |")
    lines += ["","## 主要混同とmargin","",
              "各方式のalive/cross_matchの混同件数は同じ707cropを分母とする。"]
    for name,s in primary.items():
        lines += ["",f"### {name}",""]
        for pair in s["confusion_pairs"][:5]:
            lines.append(f"- {display(pair['expected'])} → {display(pair['predicted'])}: {pair['count']} crop")
        correct=s["margins"]["correct"]["quantiles"].get("median")
        incorrect=s["margins"]["incorrect"]["quantiles"].get("median")
        lines.append(f"- margin中央値：正解 {correct} / 誤り {incorrect}。confidenceは校正確率ではなく、Unknown閾値は固定していない。")
    lines += ["","## down / unknown / 同一試合別timestamp","",
              "| 方式 | down support/top-1 | unknown support/top-1 | 同一試合別timestamp alive support/top-1 |",
              "|---|---:|---:|---:|"]
    for name,result in report["methods"].items():
        s=result["single_frame"]
        values=[s["by_state"]["down"],s["by_state"]["unknown"],s["by_state_and_scope"]["alive"]["within_match_other_timestamp"]]
        lines.append(f"| {name} | "+" | ".join(f"{m['support']} / {pct(m['exact_top1']['accuracy'])}" for m in values)+" |")
    lines += ["","stateは自動推定。公式画像欠損2種は失敗supportに残す。held-outにない12種は引き続き未測定。",
              "前処理画像・context・top-k・score・marginはViewerで確認できる。正解ラベルは変更していない。",
              "結果に応じた人間の判断をassessment.mdに記録し、新しい試合・動画を使った独立評価は次段階とする。"]
    output.mkdir(parents=True,exist_ok=True)
    if not refresh_summary or not (output/"predictions.jsonl").exists():
        (output/"predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in old+new),encoding="utf-8")
    atomic_json(output/"report.json",report)
    (output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description="同じ評価sampleの前回方式と改善方式を再計算なしで比較")
    p.add_argument("baseline",type=Path)
    p.add_argument("alternative",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--refresh-summary",action="store_true",help="同じ2入力の比較summaryのみ再生成。予測・レビューは変更しない")
    a=p.parse_args()
    report=compare_runs(a.baseline,a.alternative,a.output,a.refresh_summary)
    print(json.dumps(report["delta_percentage_points"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
