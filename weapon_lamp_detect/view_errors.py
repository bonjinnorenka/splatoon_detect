"""Paginated local error viewer; regenerate crop/context from source when needed."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import cv2_read, load_dataset
from weapon_lamp_detect.match_data import Catalog, HERE, atomic_json, frame_at, image_bytes, read_json

TAGS = ("variant", "similar_shape", "order", "x_overlay", "blur", "hud_shift", "crop_scale", "crop_truncated", "badge_overlay", "color", "special_overlay", "state_error", "label_error")


class ErrorServer(ThreadingHTTPServer):
    def __init__(self, address, report_dir):
        super().__init__(address, ErrorHandler)
        self.directory = Path(report_dir).resolve()
        self.report = read_json(self.directory / "report.json")
        catalog = Catalog()
        self.weapon_names = {name: entry["display_name"] for name, entry in catalog.entries.items()}
        self.dataset = Path(self.report["dataset"])
        self.metadata, _ = load_dataset(self.dataset)
        self.rows = [json.loads(line) for line in (self.directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.by_id = {r["method"] + ":" + r["sample_id"]: r for r in self.rows}
        path = self.directory / "error_reviews.json"
        self.reviews = read_json(path) if path.exists() else {}
        self.lock = threading.RLock()

    @lru_cache(maxsize=12)
    def frame(self, video_id, frame_index):
        return frame_at(self.metadata["videos"][video_id], frame_index=frame_index)[0]


class ErrorHandler(BaseHTTPRequestHandler):
    server: ErrorServer

    def log_message(self, *args):
        pass

    def send(self, data, content_type="application/json; charset=utf-8", status=200):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                return self.send((HERE / "error_viewer.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/report":
                return self.send({"report": self.server.report, "tags": TAGS, "weapon_names": self.server.weapon_names,
                                  "videos": self.server.metadata.get("videos",{})})
            if url.path == "/api/errors":
                rows = self.server.rows
                for key in ("method", "expected", "predicted", "state", "scope", "video_id"):
                    if q.get(key):
                        rows = [r for r in rows if r[key] == q[key]]
                if q.get("all") != "1":
                    rows = [r for r in rows if not r["correct"]]
                offset, limit = max(0, int(q.get("offset", 0))), max(1, min(100, int(q.get("limit", 24))))
                rows = sorted(rows, key=lambda r: (-(r["margin"] or 0), r["sample_id"]))
                result = [{k: v for k, v in r.items() if k != "all_candidates"} for r in rows[offset:offset+limit]]
                for r in result:
                    r["review_id"] = r["method"] + ":" + r["sample_id"]
                    r["review"] = self.server.reviews.get(r["review_id"], {})
                return self.send({"total": len(rows), "rows": result})
            if url.path == "/api/image":
                row = self.server.by_id[q["id"]]
                if "decision_frame" in q:
                    index=int(q["decision_frame"])
                    frames=row.get("decision_frames",[])
                    if not 0<=index<len(frames):
                        raise ValueError("invalid decision frame")
                    row={**row,**frames[index]}
                context = q.get("context") == "1"
                rel = row.get("context" if context else "crop")
                if q.get("original") == "1":
                    if context or "original_rect" not in row or not rel:
                        raise ValueError("補正前cropがありません")
                    rel="original_"+rel
                if rel:
                    path = (self.server.dataset / rel).resolve()
                    if self.server.dataset.resolve() not in path.parents:
                        raise ValueError("invalid exported path")
                    frame = cv2_read(path)
                else:
                    with self.server.lock:
                        frame = self.server.frame(row["video_id"], row["frame_index"])
                    if not context:
                        x1, y1, x2, y2 = row["rect"]
                        frame = frame[y1:y2, x1:x2]
                if q.get("processed") == "1":
                    if context or not row["method"].endswith(("_region", "_foreground")):
                        raise ValueError("この方式の前処理画像はありません")
                    preprocessor=self.server.report.get("preprocessors",{}).get(row["method"],"legacy")
                    if q.get("feature")=="local":
                        preprocessor=self.server.report.get("secondary_preprocessors",{}).get(row["method"])
                        if not preprocessor:
                            raise ValueError("この方式の局所照合画像はありません")
                    if preprocessor!="legacy":
                        from weapon_lamp_detect.opening_matcher import CONFIGURATIONS, features, processed_region
                        frame=processed_region(frame,preprocessor)
                        if CONFIGURATIONS[preprocessor]["processing"]=="local":
                            import cv2
                            import numpy as np
                            gray,_,_=features(frame,True)
                            # Signed high-pass luminance: 0 shown as neutral gray.
                            frame=cv2.cvtColor(np.clip(128+2*gray,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                    elif row["method"].startswith("obs_detail"):
                        from weapon_lamp_detect.detail_matcher import detail_region
                        frame=detail_region(frame)
                    else:
                        from weapon_lamp_detect.region_matcher import weapon_region
                        frame = weapon_region(frame, row["method"].endswith("_foreground"))
                if q.get("hud") == "1":
                    if not context:
                        raise ValueError("HUD表示にはcontext=1が必要です")
                    import cv2
                    frame=frame.copy()
                    if row.get("original_rect"):
                        x1,y1,x2,y2=row["original_rect"]
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                    x1,y1,x2,y2=row["rect"]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
                    frame=frame[:max(y2+15,round(frame.shape[0]*.16))]
                return self.send(image_bytes(frame), "image/jpeg")
            self.send({"error": "not found"}, status=404)
        except (KeyError, ValueError, OSError) as exc:
            self.send({"error": str(exc)}, status=400)

    def do_POST(self):
        try:
            if self.path != "/api/review":
                return self.send({"error": "not found"}, status=404)
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ValueError("cross-site request refused")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise ValueError("origin mismatch")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("application/json required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100000:
                raise ValueError("invalid body length")
            data = json.loads(self.rfile.read(length))
            if data["id"] not in self.server.by_id or set(data["tags"]) - set(TAGS):
                raise ValueError("invalid review id / tags")
            with self.server.lock:
                self.server.reviews[data["id"]] = {"tags": data["tags"], "notes": str(data.get("notes", ""))}
                atomic_json(self.server.directory / "error_reviews.json", self.server.reviews)
            self.send({"ok": True})
        except (KeyError, ValueError, OSError) as exc:
            self.send({"error": str(exc)}, status=400)


def main():
    p = argparse.ArgumentParser(description="crop/context・混同・score・marginをローカルで確認")
    p.add_argument("report_dir", type=Path)
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    server = ErrorServer(("127.0.0.1", a.port), a.report_dir)
    print(f"http://127.0.0.1:{a.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
