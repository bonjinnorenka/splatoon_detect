"""Local match labeler, using the existing standard-library HTTP approach."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.detect_matches import register_candidates, scan_video
from weapon_lamp_detect.match_data import (Catalog, DEFAULT_HUD_OFFSET, HERE, MatchStore, TEMPLATES, atomic_json,
                                         blank_match, calibrated_crop, calibrated_slot, canonical_crop,
                                         frame_at, image_bytes, read_json, squid_detector, validate_geometry)
from weapon_lamp_detect.weapon_lamp import WeaponIconMatcher


class MatchServer(ThreadingHTTPServer):
    def __init__(self, address, store, matcher, hud_offset=DEFAULT_HUD_OFFSET):
        super().__init__(address, MatchHandler)
        self.store, self.matcher = store, matcher
        self.hud_offset = hud_offset
        self.lock = threading.RLock()
        self.jobs = {}
        self.detectors = {side: squid_detector(side) for side in ("left", "right")}

    @lru_cache(maxsize=8)
    def get_frame(self, video_id, index):
        video = self.store.videos[video_id]
        return frame_at(video, frame_index=index)

    def state(self):
        resume = self.store.directory / "resume.json"
        return {"videos": list(self.store.videos.values()), "matches": self.store.matches(),
                "weapons": self.store.catalog.to_list(), "catalog_path": str(self.store.catalog.catalog_path),
                "jobs": self.jobs,
                "resume": read_json(resume) if resume.exists() else {}, "session_dir": str(self.store.directory),
                "hud_offset_default": self.hud_offset}

    def scan(self, video_id, interval, start, end):
        with self.lock:
            if self.jobs.get(video_id, {}).get("status") == "running":
                raise ValueError("この動画は検出中です")
            self.jobs[video_id] = {"status": "running", "fraction": 0}
        def work():
            try:
                def progress(info):
                    with self.lock:
                        self.jobs[video_id].update(info)
                video = self.store.videos[video_id]
                result = scan_video(video, interval, start, end, progress=progress)
                with self.lock:
                    register_candidates(self.store, video, result, self.hud_offset)
                    self.jobs[video_id].update(status="done", fraction=1, candidates=len(result["candidates"]))
            except Exception as exc:
                with self.lock:
                    self.jobs[video_id].update(status="error", error=str(exc))
        threading.Thread(target=work, daemon=True).start()


class MatchHandler(BaseHTTPRequestHandler):
    server: MatchServer

    def log_message(self, *args):
        pass

    def send(self, value, status=200, content_type="application/json; charset=utf-8"):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if url.path == "/":
                return self.send((HERE / "match_labeler.html").read_bytes(), content_type="text/html; charset=utf-8")
            if url.path == "/api/state":
                with self.server.lock:
                    return self.send(self.server.state())
            if url.path in {"/api/frame", "/api/candidates"}:
                video_id = q["video_id"][0]
                video = self.server.store.videos[video_id]
                index = int(q.get("frame_index", [round(float(q.get("timestamp", [0])[0]) * video["fps"])])[0])
                frame, index, timestamp = self.server.get_frame(video_id, index)
                ally = q.get("ally_side", ["right"])[0]
                reading = self.server.detectors[ally].read_frame(frame, timestamp, index)
                geometry = json.loads(q.get("geometry", ["{}"])[0])
                validate_geometry(geometry, video)
                slots = []
                for side in ("left", "right"):
                    for i in range(4):
                        crop, rect = calibrated_crop(frame, side, i, geometry)
                        slot = calibrated_slot(frame, reading, side, i, geometry)
                        row = {"slot": f"{side}{i}", "state": slot.state, "rect": rect}
                        if url.path == "/api/candidates":
                            with self.server.lock:
                                row["candidates"] = [c.to_dict() for c in self.server.matcher.predict_crop(canonical_crop(crop), top_k=5)]
                        else:
                            row["image"] = "data:image/png;base64," + base64.b64encode(image_bytes(crop, ".png")).decode()
                        slots.append(row)
                if url.path == "/api/candidates":
                    return self.send({"frame_index": index, "slots": slots})
                def uri(img):
                    return "data:image/jpeg;base64," + base64.b64encode(image_bytes(img)).decode()
                preview = frame.copy()
                for row in slots:
                    x1, y1, x2, y2 = row["rect"]
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,220,255), 2)
                    cv2.putText(preview, row["slot"], (x1,max(14,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,220,255), 1)
                return self.send({"timestamp": timestamp, "frame_index": index, "frame": uri(frame),
                                  "hud": uri(preview[:round(frame.shape[0] * .16)]), "hud_state": reading.hud_state,
                                  "slots": slots})
            return self.send({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            # Local UI writes only, with same-origin requests and a bounded body.
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ValueError("cross-site request refused")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise ValueError("origin mismatch")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("application/json required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1000000:
                raise ValueError("invalid body length")
            data = json.loads(self.rfile.read(length))
            with self.server.lock:
                store = self.server.store
                if self.path == "/api/save":
                    saved = store.save(data)
                    atomic_json(store.directory / "resume.json", {"match_id": saved["match_id"]})
                    return self.send(saved)
                if self.path == "/api/resume":
                    atomic_json(store.directory / "resume.json", data)
                    return self.send({"ok": True})
                if self.path == "/api/videos":
                    store.add_videos([data["path"]])
                    return self.send(self.server.state())
                if self.path == "/api/manual":
                    video = store.videos[data["video_id"]]
                    value = blank_match(video, data["timestamp"], min(video["duration"], float(data["timestamp"]) + 180),
                                        hud_offset=self.server.hud_offset)
                    if store.path(value["match_id"]).exists():
                        raise ValueError("この開始位置の試合は登録済みです")
                    return self.send(store.save(value))
                if self.path == "/api/scan":
                    self.server.scan(data["video_id"], float(data.get("interval", .5)),
                                     float(data.get("start", 0)), data.get("end"))
                    return self.send({"ok": True})
                if self.path == "/api/undo":
                    path = store.path(data["match_id"])
                    current = read_json(path)
                    previous = store.directory / "history" / current["match_id"] / f'{current["revision"] - 1:06d}.json'
                    if not previous.exists():
                        raise ValueError("戻せる履歴がありません")
                    old = read_json(previous)
                    old["revision"] = current["revision"]
                    return self.send(store.save(old))
            return self.send({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send({"error": str(exc)}, 400)


def main():
    p = argparse.ArgumentParser(description="8武器を試合単位で登録するローカルブラウザUI")
    p.add_argument("videos", nargs="*")
    p.add_argument("--session-dir", type=Path, default=HERE / "data" / "matches_obs")
    p.add_argument("--template-dir", type=Path, default=TEMPLATES)
    p.add_argument("--catalog", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--scan", action="store_true", help="登録動画をバックグラウンドで検出")
    p.add_argument("--hud-offset", type=float, default=DEFAULT_HUD_OFFSET, help="開始候補からHUDまでの待ち時間（秒）")
    args = p.parse_args()
    catalog = Catalog(args.template_dir, args.catalog)
    store = MatchStore(args.session_dir, catalog)
    if args.videos:
        store.add_videos(args.videos)
    updated = store.upgrade_unedited_matches(args.hud_offset)
    if updated:
        print(f"未編集の自動初期値を修正: {updated} matches", flush=True)
    matcher = WeaponIconMatcher.from_dir(args.template_dir)
    server = MatchServer(("127.0.0.1", args.port), store, matcher, args.hud_offset)
    if args.scan:
        for video_id in store.videos:
            server.scan(video_id, .5, 0, None)
    print(f"http://127.0.0.1:{args.port}/  session={store.directory}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
