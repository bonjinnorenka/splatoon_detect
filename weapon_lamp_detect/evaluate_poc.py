"""Real OBS evaluation, with state/split-aware metrics and temporal aggregation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.match_data import Catalog, TEMPLATES, atomic_json, now, read_json
from weapon_lamp_detect.obs_matcher import OBSMatcher
from weapon_lamp_detect.weapon_lamp import WeaponIconMatcher


_worker_context = None


def _predict_row(row, dataset, metadata, manifest, matchers, names, cache):
    crop = sample_crop(dataset, row, metadata, cache)
    scope = manifest["split"]["evaluation_scope"][row["sample_id"]] if manifest else "official_only"
    results = []
    for name, matcher in matchers.items():
        candidates = matcher.predict_crop(crop, top_k=max(5, len(names)),
                                          **({"side": row["side"]} if name.startswith("obs") else {}))
        data = [{**c.to_dict(), "score": float(c.score), "confidence": float(c.confidence)} for c in candidates]
        results.append(prediction(row, data, name, scope))
    return results


def _init_process_worker(dataset, metadata, manifest, matchers, names):
    import cv2
    global _worker_context
    cv2.setNumThreads(1)
    _worker_context = (dataset, metadata, manifest, matchers, names, {})


def _process_predict_row(row):
    return _predict_row(row, *_worker_context)


def prediction(row, candidates, method, scope):
    names = [c["weapon"] for c in candidates]
    best = candidates[0] if candidates else {}
    second = candidates[1].get("score") if len(candidates) > 1 else None
    score = best.get("score")
    return {**row, "method": method, "scope": scope, "expected": row["weapon_label"],
            "predicted": best.get("weapon"), "predicted_class": best.get("weapon_class"),
            "correct": bool(names and names[0] == row["weapon_label"]), "best_score": score,
            "second_best_score": second, "margin": score - second if second is not None else None,
            "confidence": best.get("confidence"), "top_k": candidates[:5], "all_candidates": candidates}


def metrics(rows):
    n = len(rows)
    def measure(ok):
        correct = sum(ok(r) for r in rows)
        return {"correct": correct, "support": n, "accuracy": correct / n if n else None}
    return {**{f"exact_top{k}": measure(lambda r, k=k: r["expected"] in [c["weapon"] for c in r["top_k"][:k]]) for k in (1, 3, 5)},
            "class_top1": measure(lambda r: r["weapon_class"] == r["predicted_class"]),
            "support": n, "matches": len({r["match_id"] for r in rows}),
            "match_slots": len({(r["match_id"], r["side"], r["slot_index"]) for r in rows})}


def distribution(values):
    values = [v for v in values if v is not None]
    edges = [-1.0, 0, .005, .01, .02, .05, .1, .2, .5, 1.000001]
    counts, _ = np.histogram(values, bins=edges)
    return {"support": len(values), "mean": float(np.mean(values)) if values else None,
            "quantiles": dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max"],
                                  np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist())) if values else {},
            "histogram_edges": edges, "histogram_counts": counts.tolist()}


def summarize(rows):
    weapons = sorted({r["expected"] for r in rows})
    labels = sorted({r["expected"] for r in rows} | {r["predicted"] or "<no prediction>" for r in rows})
    confusion = Counter((r["expected"], r["predicted"] or "<no prediction>") for r in rows)
    pairs = [{"expected": a, "predicted": b, "count": count,
              "order_related": "order" in (a + b).lower()}
             for (a, b), count in confusion.most_common() if a != b]
    return {"metrics": metrics(rows),
            "by_state": {s: metrics([r for r in rows if r["state"] == s]) for s in ("alive", "down", "unknown")},
            "by_scope": {s: metrics([r for r in rows if r["scope"] == s]) for s in ("cross_match", "within_match_other_timestamp", "official_only")},
            "by_state_and_scope": {s: {scope: metrics([r for r in rows if r["state"] == s and r["scope"] == scope])
                                        for scope in ("cross_match", "within_match_other_timestamp", "official_only")}
                                   for s in ("alive", "down", "unknown")},
            "per_weapon": {w: {**metrics([r for r in rows if r["expected"] == w]),
                                "by_state": {s: metrics([r for r in rows if r["expected"] == w and r["state"] == s]) for s in ("alive", "down", "unknown")}}
                           for w in weapons},
            "confusion_matrix": {"labels": labels, "rows": "expected", "columns": "predicted",
                                 "counts": [[confusion[a, b] for b in labels] for a in labels]},
            "confusion_pairs": pairs,
            "margins": {"correct": distribution([r["margin"] for r in rows if r["correct"]]),
                        "incorrect": distribution([r["margin"] for r in rows if not r["correct"]])},
            "scores": {"correct": distribution([r["best_score"] for r in rows if r["correct"]]),
                       "incorrect": distribution([r["best_score"] for r in rows if not r["correct"]])}}


def aggregate(rows, sizes=(1, 3, 5, 10), strategies=("mean_score", "majority", "confidence_vote")):
    groups = defaultdict(list)
    for row in rows:
        groups[row["match_id"], row["side"], row["slot_index"], row["state"], row["scope"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (r["frame_index"], r["sample_id"]))
        if len({r["frame_index"] for r in group}) != len(group):
            raise ValueError("重複フレームは集約しません")
        if len({r["expected"] for r in group}) != 1:
            raise ValueError("同一match slotに複数の正解武器があります")
    result = {}
    for strategy in strategies:
        result[strategy] = {}
        for count in sizes:
            predictions, common = [], []
            for group in groups.values():
                if len(group) < count:
                    continue
                subset = group[:count]
                scores, classes = defaultdict(float), {}
                for row in subset:
                    for c in row["all_candidates"]:
                        classes[c["weapon"]] = c["weapon_class"]
                        if strategy == "mean_score":
                            scores[c["weapon"]] += c["score"] / count
                        elif c["weapon"] == row["predicted"]:
                            scores[c["weapon"]] += (1 if strategy == "majority" else row["confidence"] or 0) / count
                        else:
                            scores[c["weapon"]] += 0
                ranked = [{"weapon": name, "weapon_class": classes[name], "score": score, "confidence": score}
                          for name, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                          if strategy == "mean_score" or score > 0]
                row = prediction(subset[0], ranked, subset[0]["method"], subset[0]["scope"])
                row["frame_indices"] = [r["frame_index"] for r in subset]
                predictions.append(row)
                if len(group) >= max(sizes):
                    common.append(row)
            result[strategy][str(count)] = {**summarize(predictions), "frames": count,
                                            "eligible_groups": len(predictions), "insufficient_groups": len(groups) - len(predictions),
                                            "common_cohort": summarize(common)}
    return result


def validate_split(manifest, rows, dataset):
    if manifest["dataset_sha256"] != dataset_digest(dataset):
        raise ValueError("datasetがtemplate作成時から変わりました。templateを作り直してください")
    by_id = {r["sample_id"]: r for r in rows}
    split = manifest["split"]
    train = set(split["template_sample_ids"])
    evaluation = set(split["evaluation_sample_ids"])
    if not train or not evaluation or train & evaluation or (train | evaluation) - set(by_id):
        raise ValueError("template/evaluation splitが不正です")
    used = {sample_id for entry in manifest["templates"] for sample_id in entry["sample_ids"]}
    if used - train:
        raise ValueError("OBS templateにsplit外のsampleが使用されています")
    train_frames = {(by_id[i]["video_id"], by_id[i]["frame_index"]) for i in train}
    eval_frames = {(by_id[i]["video_id"], by_id[i]["frame_index"]) for i in evaluation}
    if train_frames & eval_frames:
        raise ValueError("同一source frameがtemplateとevaluationに含まれています")
    for i in evaluation:
        row = by_id[i]
        overlap = any(by_id[j]["match_id"] == row["match_id"] and by_id[j]["weapon_label"] == row["weapon_label"] for j in train)
        expected_scope = "within_match_other_timestamp" if overlap else "cross_match"
        if split["evaluation_scope"].get(i) != expected_scope:
            raise ValueError("cross-match / within-match scopeがprovenanceと矛盾しています")
    return evaluation


def evaluate(dataset, output, method="official", obs_templates=None, template_dir=TEMPLATES, weapons=None, max_weapons=20, workers=1, executor="thread"):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if (output / "report.json").exists():
        raise ValueError("reportの上書きはしません。新しいoutputを指定してください")
    if workers < 1:
        raise ValueError("workers ≥ 1 が必要です")
    if executor not in {"thread", "process"}:
        raise ValueError("executor must be thread or process")
    metadata, rows = load_dataset(dataset)
    verify_sources(metadata)
    catalog = Catalog(template_dir)
    manifest = read_json(Path(obs_templates) / "templates.json") if obs_templates else None
    if method != "official" and not manifest:
        raise ValueError("obs/compareには --obs-templates が必要です")
    names = sorted(set(catalog.resolve(w) for w in weapons)) if weapons else (manifest["weapons"] if manifest else sorted({r["weapon_label"] for r in rows}))
    if not names or max_weapons < 1 or len(names) > max_weapons:
        raise ValueError(f"PoCの候補集合は1〜{max_weapons}武器です。--weapons または --max-weapons を指定してください")
    if set(names) - {r["weapon_label"] for r in rows}:
        raise ValueError("候補集合に人力収集していない武器があります")
    if manifest and set(names) != set(manifest["weapons"]):
        raise ValueError("A/B比較の候補武器集合はtemplate manifestと一致させてください")
    selected = validate_split(manifest, rows, dataset) if manifest else {r["sample_id"] for r in rows if r["weapon_label"] in names}
    eval_rows = [r for r in rows if r["sample_id"] in selected]
    if not eval_rows:
        raise ValueError("評価sampleがありません")
    region_mode = method in {"regions", "obs_region", "obs_foreground"}
    methods = (["official_region", "official_foreground", "obs_region", "obs_foreground"] if region_mode
               else ["official", "obs"] if method == "compare" else [method])
    if method in {"obs_region", "obs_foreground"}:
        methods = [method]
    matchers, missing, provenance = {}, {}, {}
    if any(name.startswith("official") for name in methods):
        matcher = WeaponIconMatcher.from_dir(Path(template_dir))
        matcher.templates = [t for t in matcher.templates if t.name in names]
        matcher.variants = matcher._build_variants()
        source = {"template_dir": str(Path(template_dir).resolve()), "variant_sizes": list(matcher.variant_sizes),
                                  "variant_angles": list(matcher.variant_angles),
                                  "template_sha256": {t.name: hashlib.sha256(Path(t.path).read_bytes()).hexdigest() for t in matcher.templates}}
        if region_mode:
            from weapon_lamp_detect.region_matcher import OfficialRegionMatcher, REGION, SIZES
            for method_name, remove_ink in (("official_region", False), ("official_foreground", True)):
                matchers[method_name] = OfficialRegionMatcher(matcher.templates, remove_ink)
                missing[method_name] = sorted(set(names) - {t.name for t in matcher.templates})
                provenance[method_name] = {**source,"region":list(REGION),"variant_sizes":list(SIZES),
                                           "remove_ink":remove_ink,"gray_method":"masked zero-mean correlation",
                                           "score_weights":{"gray":.55,"lab":.25,"edge":.20}}
        else:
            matchers["official"] = matcher
            missing["official"] = sorted(set(names) - {t.name for t in matcher.templates})
            provenance["official"] = source
    if "obs" in methods:
        matchers["obs"] = OBSMatcher(obs_templates)
        missing["obs"] = sorted(set(names) - {entry["weapon"] for entry in manifest["templates"]})
        provenance["obs"] = {"templates_manifest": manifest, "directory": str(Path(obs_templates).resolve()),
                             "template_sha256": {e["path"]: hashlib.sha256((Path(obs_templates) / e["path"]).read_bytes()).hexdigest() for e in manifest["templates"]}}
    if region_mode:
        from weapon_lamp_detect.region_matcher import OBSRegionMatcher, REGION, SEARCH_SHIFT
        for method_name, remove_ink in (("obs_region", False), ("obs_foreground", True)):
            if method_name not in methods:
                continue
            matchers[method_name] = OBSRegionMatcher(obs_templates,remove_ink,output/"processed_templates"/method_name)
            missing[method_name] = sorted(set(names)-{entry["weapon"] for entry in manifest["templates"]})
            provenance[method_name] = {"templates_manifest":manifest,"region":list(REGION),"search_shift":SEARCH_SHIFT,
                                       "remove_ink":remove_ink,"source_samples":"same exact samples as baseline OBS median",
                                       "preprocess_before_median":True,"score_weights":{"gray":.65,"lab":.15,"edge":.20},
                                       "template_sha256":{e["path"]:hashlib.sha256((Path(obs_templates)/e["path"]).read_bytes()).hexdigest()
                                                          for e in manifest["templates"]}}
        frozen = output / "configuration_frozen.json"
        if frozen.exists():
            configuration = read_json(frozen)
            if configuration.get("dataset_sha256") != dataset_digest(dataset):
                raise ValueError("固定設定のdatasetが異なります")
            if set(configuration.get("training_check_sample_ids", [])) & selected:
                raise ValueError("調整用sampleに評価データが含まれています")
            from weapon_lamp_detect.region_matcher import SIZES
            if (configuration.get("region") != list(REGION) or configuration.get("sizes") != list(SIZES)
                    or configuration.get("search_shift") != SEARCH_SHIFT):
                raise ValueError("固定設定と実装のROI/scale/shiftが異なります")
            current_hash = hashlib.sha256((Path(__file__).parent/"region_matcher.py").read_bytes()).hexdigest()
            if configuration.get("algorithm_sha256",current_hash) != current_hash:
                raise ValueError("固定後に背景除去・照合実装が変わっています。新しい設定確認が必要です")
            for source in provenance.values():
                source["training_configuration"] = configuration
        for source in provenance.values():
            source["algorithm_sha256"] = hashlib.sha256((Path(__file__).parent/"region_matcher.py").read_bytes()).hexdigest()
            source["ink_removal"] = {"dominant_sat_min":100,"dominant_value_min":70,"confidence_min":.55,
                                      "hue_distance_max":15,"sat_min_exclusive":75,"value_min_exclusive":65,"neutral_bgr":[128,128,128]}
    # Workers read immutable matchers. Separate per-thread frame caches preserve
    # on-demand regeneration without mixing source frames across concurrent jobs.
    import threading
    import cv2
    local = threading.local()
    previous_threads = cv2.getNumThreads()
    if workers > 1:
        cv2.setNumThreads(1)
    def predict_row(row):
        if not hasattr(local, "cache"):
            local.cache = {}
        return _predict_row(row, dataset, metadata, manifest, matchers, names, local.cache)
    predictions = []
    try:
        if executor == "process":
            import multiprocessing
            pool_context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=_init_process_worker,
                                               initargs=(dataset,metadata,manifest,matchers,names))
            fn = _process_predict_row
        else:
            pool_context = ThreadPoolExecutor(max_workers=workers)
            fn = predict_row
        with pool_context as pool:
            for index, results in enumerate(pool.map(fn, eval_rows)):
                predictions.extend(results)
                if index % 25 == 0:
                    print(f"evaluation: {index + 1}/{len(eval_rows)}", flush=True)
    finally:
        if workers > 1:
            cv2.setNumThreads(previous_threads)
    report = {"schema_version": 1, "created_at": now(), "dataset": str(dataset), "dataset_sha256": dataset_digest(dataset),
              "source": "human-labeled OBS recording", "candidate_universe": names, "closed_set": True,
              "max_candidate_weapons": max_weapons,
              "cpu_workers": workers,
              "cpu_executor": executor,
              "missing_templates": missing, "provenance": provenance,
              "normalization": ("aspect-preserving 134x108 canvas, then fixed weapon ROI; preprocessing details in provenance"
                                if region_mode else "both methods use aspect-preserving 134x108 letterboxed HUD crops"),
              "confidence_note": "heuristic score, not a calibrated probability; no unknown threshold fixed",
              "state_note": "alive/down/unknown are automatic squid_lamp_detect estimates; audit them in viewer",
              "primary_metric": "alive / cross_match top-1; within-match results are separate",
              "aggregation_note": "earliest N distinct timestamps per match-slot-state-scope; short groups excluded; common_cohort requires ≥10 frames",
              "methods": {name: {"single_frame": summarize([r for r in predictions if r["method"] == name]),
                                 "multi_frame": aggregate([r for r in predictions if r["method"] == name])} for name in methods},
              "decision": "Human review required: compare alive cross-match accuracy, weapon support, confusion pairs and common-cohort aggregation before deciding on CNN."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in predictions), encoding="utf-8")
    atomic_json(output / "report.json", report)
    lines = ["# OBSテンプレート評価", "", f"候補武器数: {len(names)} / 評価crop数: {len(eval_rows)}", "",
             "正解は人力試合ラベル。stateは自動推定。候補集合を固定したclosed-set評価。", "",
             "| 方式 | state | support | top-1 | top-3 | top-5 |", "|---|---|---:|---:|---:|---:|"]
    for name in methods:
        for state in ("alive", "down", "unknown"):
            m = report["methods"][name]["single_frame"]["by_state"][state]
            values = ["—" if m[f"exact_top{k}"]["accuracy"] is None else f'{m[f"exact_top{k}"]["accuracy"]:.3%}' for k in (1, 3, 5)]
            lines.append(f'| {name} | {state} | {m["support"]} | ' + " | ".join(values) + " |")
    lines += ["", "cross-match / 同一match別timestampはreport.jsonのby_scope・by_state_and_scopeで必ず別確認。",
              "各武器のsupport、混同ペア、margin分布、共通cohortの1/3/5/10 frame結果はreport.jsonに収録。",
              "", "## 判定のためのレビュー", "",
              "1. aliveの別試合評価に十分な武器別supportがあるか確認。未収集武器・template欠損を除外して成功と扱わない。",
              "2. Viewerで派生武器・Order系・X・ブラー・HUDずれ・色・スペシャル重畳を人力確認。",
              "3. 同じ10frame以上のcohortで1/3/5/10の改善を見る。frame数で母集団が変わる全体値だけを比較しない。",
              "4. 少数の安定frameで必要精度を満たすならtemplateを継続。残る系統的な混同や解像度への弱さが実測で確認されたらCNNを検討。",
              "", "本reportはCNN導入を自動決定しない。元動画・試合数・対象武器の範囲を超えた一般化は未検証。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description="実OBSデータの公式画像/OBS median方式を評価")
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["official", "obs", "compare", "regions", "obs_region", "obs_foreground"], default="official")
    p.add_argument("--obs-templates", type=Path)
    p.add_argument("--template-dir", type=Path, default=TEMPLATES)
    p.add_argument("--weapons", nargs="+")
    p.add_argument("--max-weapons", type=int, default=20, help="候補数の安全上限。人力収集済み範囲を明示的に拡張する場合のみ変更")
    p.add_argument("--workers", type=int, default=1, help="CPU worker数。例: 4（GPU不要）")
    p.add_argument("--executor", choices=["thread", "process"], default="thread", help="CPU並列方式。processはWindows/WSLでspawn起動")
    a = p.parse_args()
    report = evaluate(a.dataset, a.output, a.method, a.obs_templates, a.template_dir, a.weapons, a.max_weapons, a.workers, a.executor)
    for name, results in report["methods"].items():
        print(name, json.dumps(results["single_frame"]["by_state"], ensure_ascii=False))


if __name__ == "__main__":
    main()
