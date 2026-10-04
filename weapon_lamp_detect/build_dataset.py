"""Expand human-confirmed matches to sparse, reproducible crop references."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.match_data import (Catalog, DEFAULT_HUD_OFFSET, HUD_CROP_PROFILE, MatchStore, atomic_json, calibrated_crop, calibrated_slot, canonical_crop, frame_at,
                                         now, read_json, squid_detector, validate_match, video_metadata, write_image)
from weapon_lamp_detect.weapon_lamp import iter_video_frames


def load_dataset(directory):
    directory = Path(directory).resolve()
    metadata = read_json(directory / "dataset.json")
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample ids")
    return metadata, rows


def sample_crop(directory, row, metadata, frame_cache=None):
    directory = Path(directory)
    if row.get("crop"):
        image = cv2_read(directory / row["crop"])
    else:
        video = metadata["videos"][row["video_id"]]
        key = (row["video_id"], row["frame_index"])
        frame = (frame_cache or {}).get(key)
        if frame is None:
            frame, _, _ = frame_at(video, frame_index=row["frame_index"])
            if frame_cache is not None:
                frame_cache.clear()
                frame_cache[key] = frame
        x1, y1, x2, y2 = row["rect"]
        image = frame[y1:y2, x1:x2]
    return canonical_crop(image)


def cv2_read(path):
    import cv2
    import numpy as np
    # imdecode + pathlib supports Unicode filenames on Windows.
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を開けません: {path}")
    return image


def verify_sources(metadata):
    # Live collection has persisted opening PNGs rather than an invented video.
    # Keep the existing video check unchanged and verify every source PNG hash.
    if metadata.get("source_kind") == "image_sequences":
        import hashlib
        sources = metadata.get("image_sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("保存PNGのsource manifestがありません")
        for source in sources.values():
            path = Path(source["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"保存PNGが作成時と異なります: {path}")
    for video_id, video in metadata["videos"].items():
        actual = video_metadata(video["path"])
        if actual["video_id"] != video_id:
            raise ValueError(f'動画が作成時と異なります: {video["path"]}')


def build_dataset(session_dir, output, interval=5, start_offset=None, end_offset=5,
                  states=("alive",), max_samples=0, max_frames_per_match=20, export=False):
    if not all(math.isfinite(n) for n in (interval, end_offset)) or interval <= 0 or end_offset < 0 \
            or (start_offset is not None and (not math.isfinite(start_offset) or start_offset < 0)):
        raise ValueError("interval > 0 / offsets ≥ 0 が必要です")
    if max_samples < 0 or max_frames_per_match < 0:
        raise ValueError("sample limits ≥ 0 が必要です")
    output = Path(output).resolve()
    if (output / "dataset.json").exists() or (output / "samples.jsonl").exists():
        raise ValueError("datasetの上書きはしません。新しいoutput directoryを指定してください")
    store = MatchStore(session_dir, Catalog())
    matches = [m for m in store.matches() if m.get("confirmed") and not m.get("rejected")]
    if not matches:
        raise ValueError("人力確定済み試合がありません。UIで8スロットと区間を確認・保存してください")
    previous = {}
    for match in matches:
        video_id = match["video_id"]
        validate_match(match, store.videos[video_id], store.catalog)
        if match["start_timestamp"] < previous.get(video_id, -1):
            raise ValueError("確定済み試合の区間が重複しています。UIで終了を修正してください")
        previous[video_id] = match["end_timestamp"]
    videos = {m["video_id"]: store.videos[m["video_id"]] for m in matches}
    verify_sources({"videos": videos})
    detector = {side: squid_detector(side) for side in ("left", "right")}
    output.mkdir(parents=True, exist_ok=True)
    rows, excluded, counts, effective_offsets = [], Counter(), Counter(), {}
    for m in matches:
        offset = float(m.get("hud_offset_seconds", DEFAULT_HUD_OFFSET)) if start_offset is None else start_offset
        effective_offsets[m["match_id"]] = offset
        first = m["start_timestamp"] + offset
        last = m["end_timestamp"] - end_offset
        if last <= first:
            excluded["empty_interval"] += 1
            continue
        frame_count, seen = 0, set()
        source_path = videos[m["video_id"]]["path"]
        for _, index, frame in iter_video_frames(Path(source_path), sample_interval=interval, start=first, end=last):
            t = index / videos[m["video_id"]]["fps"]
            if index in seen or t >= last:
                continue
            seen.add(index)
            reading = detector[m["ally_side"]].read_frame(frame, t, index)
            if reading.hud_state != "match":
                excluded["non_match_frame"] += 1
                continue
            frame_count += 1
            for key, label in m["slots"].items():
                if label["status"] != "labeled":
                    excluded["label_" + label["status"]] += 1
                    continue
                side, slot_index = key[:-1], int(key[-1])
                slot = calibrated_slot(frame, reading, side, slot_index, m.get("geometry"))
                if slot.state not in states:
                    excluded["state_" + slot.state] += 1
                    continue
                sample_id = f'{m["match_id"]}_{index:09d}_{key}'
                crop, rect = calibrated_crop(frame, side, slot_index, m.get("geometry"))
                import cv2
                row = {"sample_id": sample_id, "source_video": source_path, "video_id": m["video_id"],
                       "match_id": m["match_id"], "timestamp": t, "frame_index": index, "side": side,
                       "slot_index": slot_index, "state": slot.state, "state_source": "squid_lamp_detect (automatic)",
                       "weapon_label": label["weapon"], "weapon_class": store.catalog.entries[label["weapon"]]["weapon_class"],
                       "label_status": "labeled", "label_source": "human_confirmed_match", "match_revision": m["revision"],
                       "rect": rect, "special_score": slot.special_score,
                       "blur_variance": float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())}
                if export:
                    row["crop"] = f'crops/{m["match_id"]}/{index:09d}_{key}.png'
                    row["context"] = f'frames/{m["match_id"]}/{index:09d}.jpg'
                    write_image(output / row["crop"], crop)
                    if not (output / row["context"]).exists():
                        write_image(output / row["context"], frame)
                rows.append(row)
                counts[m["match_id"]] += 1
                if max_samples and len(rows) >= max_samples:
                    break
            if (max_frames_per_match and frame_count >= max_frames_per_match) or (max_samples and len(rows) >= max_samples):
                break
        if max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError(f"評価cropがありません。区間とstate指定を確認してください: {dict(excluded)}")
    (output / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    metadata = {"schema_version": 1, "created_at": now(), "session_dir": str(store.directory), "videos": videos,
                "matches": matches, "sampling_interval": interval, "start_offset": start_offset,
                "end_offset": end_offset, "states": list(states), "state_source": "automatic, not human ground truth",
                "effective_start_offsets": effective_offsets,
                "max_samples": max_samples, "max_frames_per_match": max_frames_per_match,
                "crop_export": export, "normalization": "aspect-preserving 134x108 letterbox for both A and B",
                "crop_profile": HUD_CROP_PROFILE,
                "samples": len(rows), "samples_per_match": dict(counts), "excluded": dict(excluded)}
    atomic_json(output / "dataset.json", metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description="人力試合ラベルから疎なcrop datasetを展開")
    p.add_argument("session_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-interval", type=float, default=5)
    p.add_argument("--start-offset", type=float,
                   help="開始から抽出までの秒数。省略時は各試合のHUD待ち（初期値20秒）")
    p.add_argument("--end-offset", type=float, default=5)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--alive-only", action="store_true", help="default")
    group.add_argument("--include-down", action="store_true", help="alive/down/unknownを保存")
    p.add_argument("--states", nargs="+", choices=["alive", "down", "unknown"])
    p.add_argument("--max-samples", type=int, default=0, help="crop総数、0=無制限")
    p.add_argument("--max-frames-per-match", type=int, default=20, help="0=無制限")
    p.add_argument("--export-crops", action="store_true")
    a = p.parse_args()
    states = a.states or (["alive", "down", "unknown"] if a.include_down else ["alive"])
    result = build_dataset(a.session_dir, a.output, a.sample_interval, a.start_offset, a.end_offset,
                           states, a.max_samples, a.max_frames_per_match, a.export_crops)
    print(json.dumps({k: result[k] for k in ("samples", "samples_per_match", "excluded")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
