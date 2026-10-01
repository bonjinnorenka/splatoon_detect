"""Simple median OBS templates and a persisted, auditable split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, write_image


def dataset_digest(directory):
    return hashlib.sha256((Path(directory) / "samples.jsonl").read_bytes()).hexdigest()


def make_split(rows, weapons, allow_within=False, seed=7):
    relevant = [r for r in rows if r["weapon_label"] in weapons]
    matches = sorted({r["match_id"] for r in relevant},
                     key=lambda m: hashlib.sha256(f"{seed}:{m}".encode()).hexdigest())
    train_matches = set(matches[:max(1, len(matches) // 3)]) if len(matches) > 1 else set()
    template_rows = [r for r in relevant if r["match_id"] in train_matches and r["state"] == "alive"]
    template_names = {r["weapon_label"] for r in template_rows}
    fallback = []
    if allow_within:
        for weapon in weapons:
            if weapon in template_names:
                continue
            choices = sorted([r for r in relevant if r["weapon_label"] == weapon and r["state"] == "alive"],
                             key=lambda r: (r["match_id"], r["frame_index"], r["side"], r["slot_index"]))
            if choices:
                chosen = choices[0]
                template_rows.append(chosen)
                fallback.append(chosen)
    # All slots of a reserved source frame are held out, not just its labeled crop.
    reserved_frames = {(r["video_id"], r["frame_index"]) for r in template_rows}
    evaluation = [r for r in relevant if r["match_id"] not in train_matches
                  and (r["video_id"], r["frame_index"]) not in reserved_frames]
    if not template_rows:
        raise ValueError("試合を分けたOBS template用alive cropがありません。別試合を追加するか --allow-within-match を明示してください")
    if not evaluation:
        raise ValueError("templateと分離された評価cropがありません。別timestamp / 別試合を追加してください")
    template_matches_by_weapon = defaultdict(set)
    for r in template_rows:
        template_matches_by_weapon[r["weapon_label"]].add(r["match_id"])
    scopes = {}
    for r in evaluation:
        scopes[r["sample_id"]] = ("within_match_other_timestamp" if r["match_id"] in template_matches_by_weapon[r["weapon_label"]]
                                  else "cross_match")
    return {"template_sample_ids": [r["sample_id"] for r in template_rows],
            "evaluation_sample_ids": [r["sample_id"] for r in evaluation],
            "train_match_ids": sorted(train_matches), "evaluation_scope": scopes,
            "within_match_template_sample_ids": [r["sample_id"] for r in fallback], "seed": seed}


def build_templates(dataset, output, weapons=None, max_per_weapon=3, allow_within=False, seed=7, max_weapons=20):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if (output / "templates.json").exists():
        raise ValueError("templatesの上書きはしません。新しいoutputを指定してください")
    metadata, rows = load_dataset(dataset)
    verify_sources(metadata)
    present = sorted({r["weapon_label"] for r in rows})
    catalog = Catalog()
    weapons = sorted(set(catalog.resolve(w) for w in weapons)) if weapons else present
    if max_weapons < 1 or len(weapons) > max_weapons:
        raise ValueError(f"候補数の上限は{max_weapons}武器です。--weapons または --max-weapons を指定してください")
    if not weapons or max_per_weapon < 1 or set(weapons) - set(present):
        raise ValueError("weaponsが空/未収集、またはmax-per-weaponが不正です")
    split = make_split(rows, weapons, allow_within, seed)
    selected = set(split["template_sample_ids"])
    groups = defaultdict(list)
    for row in rows:
        if row["sample_id"] in selected:
            groups[row["weapon_label"], row["side"]].append(row)
    entries, cache = [], {}
    output.mkdir(parents=True, exist_ok=True)
    for (weapon, side), group in sorted(groups.items()):
        # Distinct timestamps; pick at most one crop per source frame for composition.
        unique = {(r["video_id"], r["frame_index"]): r for r in sorted(group, key=lambda r: r["sample_id"])}
        chosen = list(unique.values())[:max_per_weapon]
        images = [sample_crop(dataset, row, metadata, cache) for row in chosen]
        median = np.median(np.stack(images), axis=0).astype(np.uint8)
        path = f"median_{len(entries):03d}.png"
        write_image(output / path, median)
        entries.append({"weapon": weapon, "weapon_class": group[0]["weapon_class"], "side": side,
                        "path": path, "method": "unaligned pixel median of normalized HUD crops",
                        "sample_ids": [r["sample_id"] for r in chosen],
                        "source_frames": [{k: r[k] for k in ("video_id", "match_id", "timestamp", "frame_index", "side", "slot_index")} for r in chosen]})
    result = {"schema_version": 1, "created_at": now(), "dataset": str(dataset),
              "dataset_sha256": dataset_digest(dataset), "weapons": weapons, "split": split,
              "templates": entries, "missing_obs_templates": sorted(set(weapons) - {e["weapon"] for e in entries}),
              "allow_within_match": allow_within, "max_frames_per_weapon_side": max_per_weapon}
    result["max_candidate_weapons"] = max_weapons
    atomic_json(output / "templates.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description="OBS median templateと評価分割を生成")
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weapons", nargs="+", help="日本語またはcanonical英語名。省略時dataset全武器（最大20）")
    p.add_argument("--max-per-weapon", type=int, default=3)
    p.add_argument("--allow-within-match", action="store_true", help="同一試合の別timestampを利用（cross-matchとは別集計）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-weapons", type=int, default=20, help="候補数の安全上限。人力収集済み範囲を明示的に拡張する場合のみ変更")
    a = p.parse_args()
    result = build_templates(a.dataset, a.output, a.weapons, a.max_per_weapon, a.allow_within_match, a.seed, a.max_weapons)
    print(json.dumps({"weapons": result["weapons"], "templates": len(result["templates"]),
                      "evaluation_samples": len(result["split"]["evaluation_sample_ids"]),
                      "missing_obs_templates": result["missing_obs_templates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
