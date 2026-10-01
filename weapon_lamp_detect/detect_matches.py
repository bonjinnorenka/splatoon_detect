"""Reuse start_detect's HOG extractor and checked-in SVM JSON; no retraining."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from start_detect.hog_svm_predictor import HOGConfig, extract_hog_features, sigmoid
from weapon_lamp_detect.match_data import Catalog, DEFAULT_HUD_OFFSET, HERE, MatchStore, atomic_json, blank_match, now, read_json

MODEL = HERE.parent / "start_detect" / "hog_svm_model.json"


class StartPredictor:
    def __init__(self, model=MODEL):
        self.data = read_json(model)
        self.config = HOGConfig(**self.data["hog_config"])
        self.weights = np.asarray(self.data["weights"])
        self.means = np.asarray(self.data["scaler"]["means"])
        self.stds = np.asarray(self.data["scaler"]["stds"])

    def predict(self, frame):
        h, w = frame.shape[:2]
        # start_detect/cropsize.md: (760,260) + (400,400) at 1920x1080.
        crop = frame[round(h * 260 / 1080):round(h * 660 / 1080),
                     round(w * 760 / 1920):round(w * 1160 / 1920)]
        features = extract_hog_features(crop, self.config)
        score = float(np.dot((features - self.means) / self.stds, self.weights) + self.data["bias"])
        shifted = score - self.data["decision_threshold"]
        calibration = self.data["calibration"]
        return {"is_start": shifted >= 0, "score": score,
                "probability": sigmoid(calibration["coef"] * shifted + calibration["intercept"])}


def scan_video(video, interval=0.5, start=0.0, end=None, cooldown=30.0, progress=None):
    if interval <= 0 or cooldown < 0:
        raise ValueError("interval > 0 / cooldown ≥ 0 が必要です")
    start = max(0.0, start)
    end = min(video["duration"], video["duration"] if end is None else end)
    if not start < end:
        raise ValueError("scan start < end が必要です")
    predictor = StartPredictor()
    cap = cv2.VideoCapture(video["path"])
    hits, group = [], []
    step = max(1, round(interval * video["fps"]))
    try:
        # Seek sampled frames rather than decode all 60fps frames into Python.
        for index in range(round(start * video["fps"]), min(video["frame_count"], round(end * video["fps"])), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"開始候補検出中の動画読み込み失敗: {index}")
            result = predictor.predict(frame)
            t = index / video["fps"]
            if result["is_start"]:
                hit = {"timestamp": t, "frame_index": index, **result}
                if group and t - group[-1]["timestamp"] > max(interval * 2, cooldown):
                    hits.append(max(group, key=lambda x: x["score"]))
                    group = []
                group.append(hit)
            if progress:
                progress({"timestamp": t, "end": end, "fraction": (t - start) / (end - start)})
        if group:
            hits.append(max(group, key=lambda x: x["score"]))
    finally:
        cap.release()
    return {"schema_version": 1, "video": video, "detector": "start_detect/hog_svm_model.json",
            "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "interval": interval,
            "scan_start": start, "scan_end": end, "cooldown": cooldown, "created_at": now(), "candidates": hits}


def register_candidates(store, video, result, hud_offset=DEFAULT_HUD_OFFSET):
    atomic_json(store.directory / "candidates" / (video["video_id"] + ".json"), result)
    existing = {m["match_id"] for m in store.matches()}
    for i, candidate in enumerate(result["candidates"]):
        end = result["candidates"][i + 1]["timestamp"] if i + 1 < len(result["candidates"]) else video["duration"]
        match = blank_match(video, candidate["timestamp"], end, "hog_svm", candidate, hud_offset)
        if match["match_id"] not in existing:
            store.save(match)


def main():
    p = argparse.ArgumentParser(description="既存HOG+SVMで試合開始候補を検出（人力確認前の下書き）")
    p.add_argument("videos", nargs="+")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--scan-interval", type=float, default=0.5)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument("--cooldown", type=float, default=30)
    p.add_argument("--hud-offset", type=float, default=DEFAULT_HUD_OFFSET)
    args = p.parse_args()
    store = MatchStore(args.session_dir, Catalog())
    store.add_videos(args.videos)
    store.upgrade_unedited_matches(args.hud_offset)
    for video in store.videos.values():
        last = [-1]
        def progress(info):
            percent = int(info["fraction"] * 100)
            if percent // 5 != last[0]:
                print(f'{Path(video["path"]).name}: {percent}% ({info["timestamp"]:.1f}s)', flush=True)
                last[0] = percent // 5
        result = scan_video(video, args.scan_interval, args.start, args.end, args.cooldown, progress)
        register_candidates(store, video, result, args.hud_offset)
        print(f'{video["video_id"]}: {len(result["candidates"])} candidates')


if __name__ == "__main__":
    main()
