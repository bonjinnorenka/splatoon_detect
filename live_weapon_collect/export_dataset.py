"""Export confirmed live matches to the existing crop evaluation schema."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2

from live_weapon_collect.store import LiveStore
from live_weapon_collect.io_utils import atomic_json
from weapon_lamp_detect.build_dataset import cv2_read
from weapon_lamp_detect.match_data import (
    Catalog, SLOTS, calibrated_crop, calibrated_slot, local_path, now, squid_detector, write_image,
)


def export_dataset(session_dir, output, catalog=None, states=("alive", "down", "unknown"), include_interrupted=False):
    output = local_path(output)
    if output.exists():
        raise ValueError("上書きしません。存在しないoutput directoryを指定してください")
    if not states or set(states) - {"alive", "down", "unknown"}:
        raise ValueError("statesが不正です")
    store = LiveStore(session_dir, catalog or Catalog())
    selected = []
    excluded = Counter()
    for item in store.list():
        value = store.get(item["match_id"])
        c, a = value["capture"], value["annotation"]
        if not a["confirmed"] or a["rejected"]:
            excluded["not_human_confirmed"] += 1
            continue
        if c["status"] != "complete" and not include_interrupted:
            excluded["incomplete_capture"] += 1
            continue
        store.validate(a)
        selected.append(value)
    if not selected:
        raise ValueError("人力確定済みの収集完了試合がありません")
    output.mkdir(parents=True)
    detectors = {side: squid_detector(side) for side in ("left", "right")}
    rows, sources, matches, counts = [], {}, [], Counter()
    for value in selected:
        c, a = value["capture"], value["annotation"]
        match_id = c["match_id"]
        reference = c["frames"][a["reference_ordinal"]]
        matches.append({**a, "capture": c, "video_id": c["session_id"], "source_kind": c["source_kind"],
                        "start_timestamp": c["intro_timestamp"] if c["intro_timestamp"] is not None else c["opening_timestamp"],
                        "reference_timestamp": reference["timestamp"], "reference_frame_index": reference["frame_index"]})
        for ordinal, record in enumerate(c["frames"]):
            _, _, path = store.frame(match_id, ordinal)
            frame = cv2_read(path)
            reading = detectors[a["ally_side"]].read_frame(frame, record["timestamp"], record["frame_index"])
            context = f"frames/{match_id}/{ordinal:02d}.png"
            (output / context).parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            (output / context).write_bytes(data)
            sources[context] = {"path": str(output / context), "sha256": record["sha256"],
                                "source_path": str(path), "session_id": c["session_id"],
                                "match_id": match_id, "frame_index": record["frame_index"]}
            for key in SLOTS:
                label = a["slots"][key]
                if label["status"] != "labeled":
                    excluded["label_" + label["status"]] += 1
                    continue
                side, index = key[:-1], int(key[-1])
                slot = calibrated_slot(frame, reading, side, index, a["geometry"])
                state = slot.state if reading.hud_state == "match" else "unknown"
                if state not in states:
                    excluded["state_" + state] += 1
                    continue
                crop, rect = calibrated_crop(frame, side, index, a["geometry"])
                name = f"crops/{match_id}/{ordinal:02d}_{key}.png"
                write_image(output / name, crop)
                rows.append({"sample_id": f"{match_id}_{record['frame_index']:09d}_{key}",
                             "source_video": c["source"], "source_kind": c["source_kind"],
                             "source_image": str(path), "source_image_sha256": record["sha256"],
                             "video_id": c["session_id"], "video_id_definition": "live acquisition session UUID, not a video file hash",
                             "match_id": match_id, "timestamp": record["timestamp"], "frame_index": record["frame_index"],
                             "timing": c["device"]["timing"], "frame_index_definition": c["device"]["frame_index_definition"],
                             "side": side, "slot_index": index, "state": state,
                             "state_source": "automatic squid_lamp_detect, not human ground truth",
                             "weapon_label": label["weapon"], "weapon_class": store.catalog.entries[label["weapon"]]["weapon_class"],
                             "label_status": "labeled", "label_source": "human_confirmed_match", "match_revision": a["revision"],
                             "rect": rect, "crop": name, "context": context, "special_score": slot.special_score,
                             "opening_timestamp": c["opening_timestamp"], "opening_frame_ordinal": ordinal,
                             "seconds_after_opening": record["seconds_after_opening"], "remaining_seconds": record["remaining_seconds"],
                             "opening_detection_fallback": c["trigger"] != "intro_hud_timer",
                             "blur_variance": float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())})
                counts[match_id] += 1
    if not rows:
        raise ValueError("対象cropがありません。slot status / state指定を確認してください（出力は診断用に残しています）")
    atomic_json(output / "dataset.json", {
        "schema_version": 1, "source_kind": "image_sequences", "created_at": now(),
        "session_dir": str(store.directory), "videos": {}, "image_sources": sources, "matches": matches,
        "states": list(states), "crop_export": True, "samples": len(rows), "samples_per_match": dict(counts),
        "excluded": dict(excluded), "include_interrupted": include_interrupted,
        "normalization": "aspect-preserving 134x108 letterbox for both A and B",
        "source_note": "Persisted lossless opening PNGs, not fabricated video files. No match end or subsequent gameplay is required.",
        "catalog_path": str(store.catalog.catalog_path),
        "catalog_sha256": hashlib.sha256(store.catalog.catalog_path.read_bytes()).hexdigest(),
    })
    (output / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="人力確定済み開始PNGから既存評価CLI用datasetを生成")
    p.add_argument("session_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog", type=Path)
    p.add_argument("--states", nargs="+", choices=["alive", "down", "unknown"], default=["alive", "down", "unknown"])
    p.add_argument("--include-interrupted", action="store_true", help="人間が確認・確定した不足frame試合も含める")
    a = p.parse_args()
    count = export_dataset(a.session_dir, a.output, Catalog(catalog_path=a.catalog), a.states, a.include_interrupted)
    print(f"{count} crops: {a.output}")


if __name__ == "__main__":
    main()
