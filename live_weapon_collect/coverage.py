"""Collection coverage counts human-confirmed matches, never inferred labels/crops."""
from __future__ import annotations

import argparse
import csv
import io
import json
import threading
from collections import defaultdict
from pathlib import Path

from weapon_lamp_detect.match_data import Catalog, SLOTS, local_path, now, read_json

DEFAULT_EXISTING = Path(__file__).resolve().parent.parent / "weapon_lamp_detect" / "data" / "obs_session"


class CoverageIndex:
    def __init__(self, session_dir, catalog, source_dirs=None):
        self.catalog = catalog
        # None uses the existing video annotations. [] explicitly disables them.
        extras = [DEFAULT_EXISTING] if source_dirs is None else source_dirs
        self.directories = list(dict.fromkeys(local_path(p) for p in [session_dir, *extras]))
        self.cache = {}
        self.lock = threading.RLock()

    def read(self, path):
        stat = path.stat()
        # Atomic replacements can have the same size/mtime on coarse filesystems.
        # Include file identity and ctime so rapid edits/undo invalidate the cache.
        key = (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        cached = self.cache.get(path)
        if cached is None or cached[0] != key:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError("JSON objectではありません")
            self.cache[path] = (key, value)
        return self.cache[path][1]

    def snapshot(self, target=3):
        if type(target) is not int or target not in {2, 3}:
            raise ValueError("収集目安は2または3試合を指定してください")
        with self.lock:
            return self._snapshot(target)

    def _snapshot(self, target):
        warnings, sources, candidates = [], [], defaultdict(list)
        seen_paths = set()
        for directory in self.directories:
            source = {"directory": str(directory), "exists": directory.is_dir(), "annotation_files": 0}
            sources.append(source)
            if not source["exists"]:
                warnings.append(f"集計元がありません: {directory}")
                continue
            base = directory if directory.name == "matches" else directory / "matches"
            paths = sorted([*base.glob("*.json"), *base.glob("*/annotation.json")])
            source["annotation_files"] = len(paths)
            for path in paths:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    a = self.read(path)
                    if type(a.get("confirmed")) is not bool or type(a.get("rejected")) is not bool:
                        raise ValueError("confirmed / rejectedフラグが不正です")
                    match_id, revision = a["match_id"], a.get("revision", 0)
                    if not isinstance(match_id, str) or not match_id or type(revision) is not int or revision < 0:
                        raise ValueError("match id / revisionが不正です")
                    capture_path = path.parent / "capture.json"
                    kind, incomplete, usable = "video", False, True
                    if path.name == "annotation.json":
                        c = self.read(capture_path)
                        if c["match_id"] != match_id:
                            raise ValueError("capture / annotationのmatch idが一致しません")
                        kind = "live"
                        incomplete = c["status"] == "interrupted"
                        usable = bool(c["frames"]) and c["status"] in {"complete", "interrupted"}
                    labels = set()
                    eligible = a.get("confirmed") is True and not a.get("rejected") and usable
                    if eligible:
                        if not isinstance(a.get("slots"), dict) or set(a["slots"]) != set(SLOTS):
                            raise ValueError("8スロットがありません")
                        for key, slot in a["slots"].items():
                            if not isinstance(slot, dict):
                                raise ValueError(f"{key}: slot objectが不正です")
                            if slot.get("status") == "labeled" and slot.get("reviewed") is True:
                                try:
                                    labels.add(self.catalog.resolve_label(slot.get("weapon")))
                                except ValueError as exc:
                                    warnings.append(f"{path} / {key}: {exc}（集計から除外）")
                    # Include unconfirmed newer revisions to supersede stale copies.
                    candidates[match_id].append({"path": str(path), "revision": revision, "kind": kind,
                                                 "incomplete": incomplete, "eligible": eligible, "labels": labels})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    warnings.append(f"{path}: {exc}（集計から除外）")
        counts = {w["name"]: {"all": set(), "video": set(), "live": set(), "incomplete": set()}
                  for w in self.catalog.to_list()}
        confirmed_matches = set()
        duplicate_copies = 0
        for match_id, copies in candidates.items():
            duplicate_copies += len(copies) - 1
            latest = max(c["revision"] for c in copies)
            copies = [c for c in copies if c["revision"] == latest]
            signatures = {(c["eligible"], frozenset(c["labels"])) for c in copies}
            if len(signatures) > 1:
                warnings.append(f"同じmatch idの同revisionに異なるラベルがあります: {match_id}（試合全体を集計から除外）")
                continue
            chosen = copies[0]
            if not chosen["eligible"]:
                continue
            confirmed_matches.add(match_id)
            for name in chosen["labels"]:
                group = counts[name]
                group["all"].add(match_id)
                group[chosen["kind"]].add(match_id)
                if chosen["incomplete"]:
                    group["incomplete"].add(match_id)
        weapons = []
        for weapon in self.catalog.to_list():
            group = counts[weapon["name"]]
            count = len(group["all"])
            weapons.append({"name": weapon["name"], "display_name": weapon["display_name"],
                            "category": weapon["category"], "weapon_class": weapon["weapon_class"],
                            "match_count": count, "video_match_count": len(group["video"]),
                            "live_match_count": len(group["live"]), "incomplete_match_count": len(group["incomplete"]),
                            "remaining_matches": max(0, target-count),
                            "status": "missing" if count == 0 else "insufficient" if count < target else "at_target",
                            "match_ids": sorted(group["all"])})
        weapons.sort(key=lambda w: (w["match_count"], w["category"], w["display_name"]))
        return {"created_at": now(), "target": target, "catalog_path": str(self.catalog.catalog_path),
                "sources": sources, "warnings": warnings, "weapons": weapons,
                "summary": {"weapons_total": len(weapons), "weapons_with_data": sum(w["match_count"] > 0 for w in weapons),
                            "weapons_missing": sum(w["match_count"] == 0 for w in weapons),
                            "weapons_below_target": sum(w["match_count"] < target for w in weapons),
                            "weapons_at_target": sum(w["match_count"] >= target for w in weapons),
                            "unique_confirmed_matches": len(confirmed_matches), "duplicate_copies_ignored": duplicate_copies},
                "counting_note": "人力確定済み・対象外でない試合のreviewed/labeled武器のみ。同一武器が同一matchに複数スロット/フレームあっても1試合。同一match idのコピーは最新revisionだけ。認識精度・学習済み・テンプレート採用数ではない。元動画の所在と画質はこの一覧では検証しない。"}


def csv_text(result, missing_only=False):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["武器名", "英語名", "カテゴリ", "合計試合", "動画ラベル", "ライブ収集", "不足試合", "目安試合", "中断試合を含む"])
    for w in result["weapons"]:
        if not missing_only or w["remaining_matches"]:
            writer.writerow([w["display_name"], w["name"], w["category"], w["match_count"], w["video_match_count"],
                             w["live_match_count"], w["remaining_matches"], result["target"], w["incomplete_match_count"]])
    return stream.getvalue()


def main():
    p = argparse.ArgumentParser(description="wepons.jsonの全武器について人力確認済み試合数と収集不足を集計")
    p.add_argument("--session-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "session")
    p.add_argument("--coverage-dir", action="append", type=Path, help="追加集計元のsession directory。複数指定可。省略時は既存obs_session")
    p.add_argument("--no-existing", action="store_true", help="追加集計元を使わず現在のlive sessionだけ集計")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--target", type=int, choices=[2, 3], default=3)
    p.add_argument("--format", choices=["text", "json", "csv"], default="text")
    p.add_argument("--missing-only", action="store_true", help="目安に不足している武器だけ表示")
    a = p.parse_args()
    result = CoverageIndex(a.session_dir, Catalog(catalog_path=a.catalog), [] if a.no_existing else a.coverage_dir).snapshot(a.target)
    if a.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.format == "csv":
        print(csv_text(result, a.missing_only), end="")
    else:
        print(f'収集目安 {a.target}試合 / {result["summary"]["unique_confirmed_matches"]}確定試合 / '+
              f'未収集 {result["summary"]["weapons_missing"]}武器 / 不足 {result["summary"]["weapons_below_target"]}武器')
        print("武器名\t合計試合\t動画\tライブ\tあと何試合")
        for w in result["weapons"]:
            if not a.missing_only or w["remaining_matches"]:
                print(f'{w["display_name"]}\t{w["match_count"]}\t{w["video_match_count"]}\t{w["live_match_count"]}\t{w["remaining_matches"]}')
        for warning in result["warnings"]:
            print("注意: " + warning)


if __name__ == "__main__":
    main()
