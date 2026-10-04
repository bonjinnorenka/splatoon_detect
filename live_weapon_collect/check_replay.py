"""Bounded, faster offline worker smoke test using the real detectors, no sockets."""
from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from live_weapon_collect.capture import CaptureConfig, capture_worker
from live_weapon_collect.store import LiveStore
from weapon_lamp_detect.match_data import Catalog, local_path


def main():
    p = argparse.ArgumentParser(description="実動画の短区間で開始収集を検証（HTTP待受・カメラなし、元動画は複製しない）")
    p.add_argument("video")
    p.add_argument("--session-dir", type=Path, required=True, help="検証専用の新しいdirectoryを指定")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=40)
    a = p.parse_args()
    if a.session_dir.exists() or a.start < 0 or not 0 < a.duration <= 120:
        p.error("新しいsession-dir / start≥0 / 0<duration≤120 が必要です")
    config = CaptureConfig(source=str(local_path(a.video)), replay=True)
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        p.error("動画を開けません")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        cap.release()
        p.error("fpsが不正です")
    class SampledReader:
        # Tests worker logic at sampled source timestamps, not real camera latency.
        thread = SimpleNamespace(start=lambda: None)
        metadata = {"timing": "source frame_index / fps (accelerated offline smoke test)",
                    "frame_index_definition": "decoded source frame index", "reported_fps": fps,
                    "backend": cap.getBackendName(), "test_mode": True}
        error = None
        ended = False
        t = a.start - config.scan_interval
        def get(self):
            self.t += config.scan_interval
            if self.t >= a.start + a.duration:
                self.ended = True
            index = round(self.t*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = cap.read()
            if not ok:
                self.ended = True
                return None
            return frame, index, index/fps
    store = LiveStore(a.session_dir, Catalog())
    try:
        with patch("live_weapon_collect.capture.LatestCapture", return_value=SampledReader()):
            capture_worker(store.directory, config, queue.Queue(), threading.Event())
        for m in store.list():
            c = store.capture(m["match_id"])
            print(c["match_id"], c["status"], [(round(f["timestamp"],3),f["remaining_seconds"]) for f in c["frames"]])
        if not store.list():
            raise SystemExit("この区間では開始HUDを収集できませんでした。開始候補と実動画の位置を確認してください")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
