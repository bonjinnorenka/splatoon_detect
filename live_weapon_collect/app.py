"""Run with python -m live_weapon_collect.app (Windows-safe spawn entry point)."""
from __future__ import annotations

import argparse
import base64
import json
import multiprocessing
import queue
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

from live_weapon_collect.capture import CaptureConfig, CollectorLock, capture_worker, recover_interrupted
from live_weapon_collect.coverage import CoverageIndex, csv_text
from live_weapon_collect.hybrid_candidates import DEFAULT_BUNDLE, bundle_info, load as load_hybrid
from live_weapon_collect.io_utils import atomic_json
from live_weapon_collect.store import LiveStore
from weapon_lamp_detect.build_dataset import cv2_read
from weapon_lamp_detect.match_data import (
    Catalog, TEMPLATES, SLOTS, calibrated_crop, canonical_crop,
    image_bytes, local_path, read_json, validate_geometry,
)

HERE = Path(__file__).resolve().parent


class LiveServer(ThreadingHTTPServer):
    def __init__(self, address, store, commands=None, stopped=None, process=None, template_dir=TEMPLATES, errors=None, coverage_dirs=None,
                 candidate_matcher="hybrid", hybrid_dir=DEFAULT_BUNDLE):
        super().__init__(address, LiveHandler)
        # Tiny HUD crops are slower with large OpenCV thread pools. The camera
        # worker is a separate spawned process and isn't affected by this setting.
        cv2.setNumThreads(1)
        self.store, self.commands, self.stopped, self.process = store, commands, stopped, process
        self.template_dir = Path(template_dir)
        self.matcher = None
        self.matcher_lock = threading.Lock()
        self.candidate_matcher, self.hybrid_dir = candidate_matcher, Path(hybrid_dir)
        self.candidate_info = (bundle_info(self.hybrid_dir) if candidate_matcher == "hybrid" else
                               {"engine": "official", "available": bool(list(self.template_dir.glob("*.png"))), "single_frame": True})
        self.errors, self.startup_error = errors, None
        self.coverage = CoverageIndex(store.directory, store.catalog, coverage_dirs)

    def state(self):
        if self.errors is not None:
            try:
                self.startup_error = self.errors.get_nowait()
            except queue.Empty:
                pass
        path = self.store.directory / "collector_status.json"
        status = read_json(path) if path.exists() else {"status": "offline", "phase": "offline"}
        if self.process is None:
            status = {**status, "status": "offline", "phase": "offline"}
        elif not self.process.is_alive() and status.get("status") in {"starting", "running"}:
            status = {**status, "status": "error", "error": "収集プロセスが終了しました。端末ログと接続を確認してください"}
        if self.startup_error:
            status = {"status": "error", "phase": "error", "error": self.startup_error}
        return {"collector": status, "matches": self.store.list(), "weapons": self.store.catalog.to_list(),
                "candidate_model": self.candidate_info,
                "catalog_path": str(self.store.catalog.catalog_path), "session_dir": str(self.store.directory),
                "resume": self.store.resume(), "can_capture": self.process is not None and self.process.is_alive() and not self.stopped.is_set()}

    @lru_cache(maxsize=4)
    def load_frame(self, match_id, ordinal):
        capture, row, path = self.store.frame(match_id, ordinal)
        return cv2_read(path), row

    def render(self, data, candidates=False):
        match_id, ordinal = data["match_id"], int(data.get("ordinal", 0))
        # Verify persisted file on every request, including cached decoded frames.
        capture, _, _ = self.store.frame(match_id, ordinal)
        frame, row = self.load_frame(match_id, ordinal)
        geometry = data.get("geometry") or self.store.get(match_id)["annotation"]["geometry"]
        validate_geometry(geometry, capture["resolution"])
        slots = []
        preview = frame.copy()
        for key in SLOTS:
            crop, rect = calibrated_crop(frame, key[:-1], int(key[-1]), geometry)
            slot = {"slot": key, "rect": rect, "state": row["slots"][key]["state"]}
            if candidates:
                with self.matcher_lock:
                    if self.matcher is None:
                        if self.candidate_matcher == "hybrid":
                            self.matcher = load_hybrid(self.hybrid_dir)
                        else:
                            from weapon_lamp_detect.weapon_lamp import WeaponIconMatcher
                            if not list(self.template_dir.glob("*.png")):
                                raise ValueError("公式テンプレートがありません。候補なしでも人力入力・収集できます")
                            self.matcher = WeaponIconMatcher.from_dir(self.template_dir)
                    # Rank the complete candidate universe before catalog filtering;
                    # never turn predictions or this match's annotation into templates.
                    kwargs = {"side": key[:-1]} if self.candidate_matcher == "hybrid" else {}
                    inferred = self.matcher.predict_crop(canonical_crop(crop), top_k=len(self.store.catalog.entries), **kwargs)
                    slot["candidates"] = [{**c.to_dict(), "display_name": self.store.catalog.entries[c.weapon]["display_name"]}
                                          for c in inferred if c.weapon in self.store.catalog.label_names][:5]
                    scores = slot["candidates"]
                    slot["margin"] = scores[0]["score"] - scores[1]["score"] if len(scores) > 1 else None
            else:
                slot["image"] = "data:image/png;base64," + base64.b64encode(image_bytes(crop, ".png")).decode()
            slots.append(slot)
            x1, y1, x2, y2 = rect
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(preview, key, (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
        def uri(image):
            return "data:image/jpeg;base64," + base64.b64encode(image_bytes(image)).decode()
        return {"ordinal": ordinal, "metadata": row, "slots": slots,
                **({"candidate_model": self.candidate_info} if candidates else {}),
                **({} if candidates else {"frame": uri(frame), "hud": uri(preview[:round(frame.shape[0] * .16)])})}


class LiveHandler(BaseHTTPRequestHandler):
    server: LiveServer

    def log_message(self, *args):
        pass

    def local_request(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("localhostから操作してください")

    def send(self, value, status=200, content_type="application/json; charset=utf-8"):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            self.local_request()
            url = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                return self.send((HERE / "ui.html").read_bytes(), content_type="text/html; charset=utf-8")
            if url.path == "/api/state":
                return self.send(self.server.state())
            if url.path in {"/api/coverage", "/api/coverage.csv"}:
                result = self.server.coverage.snapshot(int(q.get("target", 3)))
                if url.path.endswith(".csv"):
                    return self.send(("\ufeff" + csv_text(result, q.get("missing_only") == "1")).encode("utf-8"),
                                     content_type="text/csv; charset=utf-8")
                return self.send(result)
            if url.path == "/api/match":
                return self.send(self.server.store.get(q["match_id"]))
            if url.path == "/api/preview":
                path = self.server.store.directory / "preview.jpg"
                if not path.exists():
                    return self.send({"error": "入力previewはまだありません"}, 404)
                return self.send(path.read_bytes(), content_type="image/jpeg")
            return self.send({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            self.local_request()
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
            if self.path == "/api/save":
                return self.send(self.server.store.save(data))
            if self.path == "/api/undo":
                return self.send(self.server.store.undo(data["match_id"], data["revision"]))
            if self.path == "/api/resume":
                self.server.store.capture(data["match_id"])
                atomic_json(self.server.store.directory / "resume.json", {"match_id": data["match_id"]})
                return self.send({"ok": True})
            if self.path in {"/api/frame", "/api/candidates"}:
                return self.send(self.server.render(data, self.path == "/api/candidates"))
            if self.path == "/api/manual":
                if not self.server.state()["can_capture"]:
                    raise ValueError("収集プロセスが動いていません")
                try:
                    self.server.commands.put_nowait("manual")
                except queue.Full as exc:
                    raise ValueError("手動要求が溜まっています。少し待ってください") from exc
                return self.send({"ok": True, "notice": "現在の画像から収集を要求しました。開始直後か必ず確認してください"})
            if self.path == "/api/stop":
                if self.server.stopped:
                    self.server.stopped.set()
                return self.send({"ok": True})
            return self.send({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send({"error": str(exc)}, 400)


def main():
    p = argparse.ArgumentParser(description="カメラから開始HUDだけ保存し、試合終了後に8武器を人力入力")
    p.add_argument("--session-dir", type=Path, default=HERE / "data" / "session")
    p.add_argument("--source", default="0", help="カメラ番号 / Linux device path / --replay時の動画path")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2"], default="auto")
    p.add_argument("--replay", action="store_true", help="動画を等速でカメラ相当に入力（動作確認用、元動画を複製しない）")
    p.add_argument("--label-only", action="store_true", help="カメラを開かず保存済み試合の入力だけ行う")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=60)
    p.add_argument("--fourcc", default="", help="camera format例: MJPG / YUY2。実際の設定結果はUIに表示")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--interval", type=float, default=1)
    p.add_argument("--scan-interval", type=float, default=.5)
    p.add_argument("--hud-delay-min", type=float, default=8)
    p.add_argument("--fallback-offset", type=float, default=20, help="HUD/OCR未確認でも開始候補からこの秒数後に要確認画像を保存（例: 15 / 20）")
    p.add_argument("--hud-timeout", type=float, default=35)
    p.add_argument("--cooldown", type=float, default=60)
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--catalog", type=Path, help="既存wepons.json。未指定なら既存catalogの探索規則を使用")
    p.add_argument("--template-dir", type=Path, default=TEMPLATES, help="任意: top-k手動候補の公式PNG")
    p.add_argument("--candidate-matcher", choices=["hybrid", "official"], default="hybrid", help="top-5候補の方式（既定は最新OBS hybrid）")
    p.add_argument("--hybrid-dir", type=Path, default=DEFAULT_BUNDLE, help="Windowsでも読めるhybrid crop bundle")
    p.add_argument("--coverage-dir", action="append", type=Path, help="武器別試合数の追加集計元session directory。複数指定可。省略時は既存obs_session")
    p.add_argument("--no-existing", action="store_true", help="武器別試合数は現在のlive sessionだけ集計")
    a = p.parse_args()
    source = str(local_path(a.source)) if a.replay else a.source
    config = CaptureConfig(source=source, backend=a.backend, replay=a.replay,
                           width=a.width, height=a.height, fps=a.fps, frames=a.frames, interval=a.interval,
                           scan_interval=a.scan_interval, hud_delay_min=a.hud_delay_min, fallback_offset=a.fallback_offset,
                           hud_timeout=a.hud_timeout, cooldown=a.cooldown, fourcc=a.fourcc)
    config.validate()
    store = LiveStore(a.session_dir, Catalog(a.template_dir, a.catalog))
    if a.label_only:
        try:
            recovery_lock = CollectorLock(store.directory)
        except ValueError:
            # Another collector is live; its acquisition records must not be changed.
            pass
        else:
            try:
                recover_interrupted(store.directory)
            finally:
                recovery_lock.close()
    context = multiprocessing.get_context("spawn")
    commands, stopped, errors = context.Queue(maxsize=10), context.Event(), context.Queue(maxsize=1)
    # Bind before starting acquisition so a port conflict never leaves a worker.
    server = LiveServer(("127.0.0.1", a.port), store, commands, stopped, template_dir=a.template_dir, errors=errors,
                        coverage_dirs=[] if a.no_existing else a.coverage_dir,
                        candidate_matcher=a.candidate_matcher, hybrid_dir=a.hybrid_dir)
    process = None
    try:
        if not a.label_only:
            process = context.Process(target=capture_worker, args=(str(store.directory), config, commands, stopped, errors), daemon=True)
            process.start()
            server.process = process
        print(f"http://127.0.0.1:{a.port}/  session={store.directory}", flush=True)
        print("ブラウザを閉じても収集は続きます。終了は端末Ctrl+C（未入力・保存済み画像は残ります）", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        if process:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        server.server_close()
        commands.close()
        errors.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
