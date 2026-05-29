from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2

try:
    from .weapon_lamp import WeaponIconMatcher, crop_slot, iter_video_frames
except ImportError:
    from weapon_lamp import WeaponIconMatcher, crop_slot, iter_video_frames


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "sample_data" / "Main Weapons"
DEFAULT_SESSION_PARENT = SCRIPT_DIR / "data" / "labeling_sessions"


INDEX_HTML = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Weapon Lamp Labeler</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111;
      color: #f4f4f4;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      border-bottom: 1px solid #333;
      background: #171717;
    }
    button, input {
      font: inherit;
      border: 1px solid #444;
      background: #222;
      color: #fff;
      border-radius: 6px;
      min-height: 36px;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      background: #2d6cdf;
      border-color: #3f7cf0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(360px, 520px);
      gap: 16px;
      padding: 16px;
      align-items: start;
    }
    .media {
      display: grid;
      gap: 14px;
    }
    .crop {
      width: min(100%, 720px);
      image-rendering: pixelated;
      background: #050505;
      border: 1px solid #333;
    }
    .context {
      width: min(100%, 960px);
      background: #050505;
      border: 1px solid #333;
    }
    .panel {
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .meta {
      color: #c8c8c8;
      line-height: 1.5;
    }
    input {
      width: 100%;
      box-sizing: border-box;
      padding: 0 10px;
    }
    .row, .candidates {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .candidate {
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .muted {
      color: #9a9a9a;
    }
    @media (max-width: 820px) {
      main {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <span id="progress" class="muted"></span>
  </header>
  <main>
    <section class="media">
      <img id="crop" class="crop" alt="">
      <img id="context" class="context" alt="">
    </section>
    <section class="panel">
      <div id="meta" class="meta"></div>
      <input id="weapon" list="weapon-list" autocomplete="off" placeholder="weapon name">
      <datalist id="weapon-list"></datalist>
      <div class="row">
        <button class="primary" id="save">Save</button>
        <button id="unknown">Unknown</button>
        <button id="skip">Skip</button>
      </div>
      <div id="candidates" class="candidates"></div>
      <div id="status" class="muted"></div>
    </section>
  </main>
  <script>
    let samples = [];
    let weapons = [];
    let labels = {};
    let index = 0;

    const $ = (id) => document.getElementById(id);

    function fileUrl(path) {
      return "/files/" + encodeURIComponent(path).replaceAll("%2F", "/");
    }

    function current() {
      return samples[index] || null;
    }

    function setIndex(nextIndex) {
      if (!samples.length) return;
      index = Math.max(0, Math.min(samples.length - 1, nextIndex));
      render();
    }

    function render() {
      const sample = current();
      if (!sample) {
        $("progress").textContent = "0 / 0";
        return;
      }
      const labeled = Object.keys(labels).length;
      $("progress").textContent = `${index + 1} / ${samples.length}  labeled ${labeled}`;
      $("crop").src = fileUrl(sample.crop);
      $("context").src = fileUrl(sample.frame);
      $("meta").textContent = `${sample.video}  ${sample.timestamp.toFixed(2)}s  ${sample.side}${sample.index}  ${sample.state}`;
      const existing = labels[sample.id] || {};
      $("weapon").value = existing.weapon || "";
      $("status").textContent = existing.status ? `current: ${existing.status}` : "";
      $("candidates").replaceChildren();
      (sample.candidates || []).forEach((candidate) => {
        const button = document.createElement("button");
        button.className = "candidate";
        button.textContent = `${candidate.weapon} ${candidate.score.toFixed(3)}`;
        button.addEventListener("click", () => {
          $("weapon").value = candidate.weapon;
          $("weapon").focus();
        });
        $("candidates").appendChild(button);
      });
    }

    async function postLabel(status) {
      const sample = current();
      if (!sample) return;
      const payload = {
        sample_id: sample.id,
        status,
        weapon: status === "labeled" ? $("weapon").value.trim() : "",
        video: sample.video,
        timestamp: sample.timestamp,
        side: sample.side,
        index: sample.index,
        state: sample.state,
      };
      const response = await fetch("/api/label", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        $("status").textContent = await response.text();
        return;
      }
      labels[sample.id] = payload;
      setIndex(index + 1);
    }

    async function boot() {
      [samples, weapons, labels] = await Promise.all([
        fetch("/samples.json").then((r) => r.json()),
        fetch("/weapons.json").then((r) => r.json()),
        fetch("/labels_current.json").then((r) => r.ok ? r.json() : {}),
      ]);
      const datalist = $("weapon-list");
      weapons.forEach((weapon) => {
        const option = document.createElement("option");
        option.value = weapon;
        datalist.appendChild(option);
      });
      const firstOpen = samples.findIndex((sample) => !labels[sample.id]);
      index = firstOpen >= 0 ? firstOpen : 0;
      render();
    }

    $("prev").addEventListener("click", () => setIndex(index - 1));
    $("next").addEventListener("click", () => setIndex(index + 1));
    $("save").addEventListener("click", () => postLabel("labeled"));
    $("unknown").addEventListener("click", () => postLabel("unknown"));
    $("skip").addEventListener("click", () => postLabel("skip"));
    $("weapon").addEventListener("keydown", (event) => {
      if (event.key === "Enter") postLabel("labeled");
    });
    boot();
  </script>
</body>
</html>
"""


def _load_squid_detector(ally_side: str):
    private_dir = SCRIPT_DIR.parent
    if str(private_dir) not in sys.path:
        sys.path.append(str(private_dir))
    try:
        from squid_lamp_detect.squid_lamp import SquidLampDetector

        return SquidLampDetector(ally_side=ally_side)  # type: ignore[arg-type]
    except Exception:
        return None


def _slot_states_from_reading(reading) -> dict[str, list[str]]:
    states: dict[str, list[str]] = {"left": [], "right": []}
    if reading is None:
        return states
    for side in ("left", "right"):
        side_reading = reading.sides.get(side)
        if side_reading is not None:
            states[side] = [slot.state for slot in side_reading.slots]
    return states


def _draw_context(frame, states: dict[str, list[str]], output_path: Path) -> None:
    out = frame.copy()
    for side in ("left", "right"):
        for index in range(4):
            _crop, rect = crop_slot(frame, side, index)
            x1, y1, x2, y2 = rect
            state = states.get(side, ["unknown"] * 4)[index] if index < len(states.get(side, [])) else "unknown"
            color = (0, 220, 0) if state == "alive" else (0, 190, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{side[0]}{index}", (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_h = max(1, int(round(out.shape[0] * 0.22)))
    cv2.imwrite(str(output_path), out[:crop_h])


def _load_current_labels(session_dir: Path) -> dict[str, dict[str, object]]:
    path = session_dir / "labels_current.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_label(session_dir: Path, payload: dict[str, object]) -> None:
    sample_id = str(payload.get("sample_id") or "")
    if not sample_id:
        raise ValueError("missing sample_id")
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    labels = _load_current_labels(session_dir)
    labels[sample_id] = payload
    with (session_dir / "labels.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (session_dir / "labels_current.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_session(
    videos: list[Path],
    session_dir: Path,
    template_dir: Path,
    start: float,
    end: float | None,
    sample_interval: float,
    max_slots: int,
    include_down: bool,
    ally_side: str,
    top_k: int,
    variant_sizes: list[int],
    variant_angles: list[float],
) -> None:
    matcher = WeaponIconMatcher.from_dir(template_dir, variant_sizes=variant_sizes, variant_angles=variant_angles)
    detector = _load_squid_detector(ally_side)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "crops").mkdir(exist_ok=True)
    (session_dir / "frames").mkdir(exist_ok=True)
    samples: list[dict[str, object]] = []
    total_slots = 0

    for video_number, video in enumerate(videos):
        for timestamp, frame_index, frame in iter_video_frames(video, sample_interval=sample_interval, start=start, end=end):
            reading = detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index) if detector is not None else None
            if getattr(reading, "hud_state", "match") == "non_match":
                continue
            states = _slot_states_from_reading(reading)
            frame_id = f"v{video_number:02d}_{int(round(timestamp * 1000)):08d}"
            frame_rel = Path("frames") / f"{frame_id}.jpg"
            frame_written = False
            for side in ("left", "right"):
                for index in range(4):
                    state = "unknown"
                    if index < len(states.get(side, [])):
                        state = states[side][index]
                    if state == "down" and not include_down:
                        continue
                    crop, _rect = crop_slot(frame, side, index)
                    slot_id = f"{frame_id}_{side}{index}"
                    crop_rel = Path("crops") / f"{slot_id}.jpg"
                    cv2.imwrite(str(session_dir / crop_rel), crop)
                    if not frame_written:
                        _draw_context(frame, states, session_dir / frame_rel)
                        frame_written = True
                    candidates = [candidate.to_dict() for candidate in matcher.predict_crop(crop, top_k=max(1, top_k))]
                    samples.append(
                        {
                            "id": slot_id,
                            "video": str(video),
                            "timestamp": round(float(timestamp), 4),
                            "frame_index": int(frame_index),
                            "side": side,
                            "index": index,
                            "state": state,
                            "crop": crop_rel.as_posix(),
                            "frame": frame_rel.as_posix(),
                            "candidates": candidates,
                        }
                    )
                    total_slots += 1
                    if max_slots > 0 and total_slots >= max_slots:
                        break
                if max_slots > 0 and total_slots >= max_slots:
                    break
            if max_slots > 0 and total_slots >= max_slots:
                break
        if max_slots > 0 and total_slots >= max_slots:
            break

    weapons = sorted(template.name for template in matcher.templates)
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "videos": [str(video) for video in videos],
        "template_dir": str(template_dir),
        "sample_interval": sample_interval,
        "start": start,
        "end": end,
        "include_down": include_down,
        "samples": len(samples),
        "variant_sizes": variant_sizes,
        "variant_angles": variant_angles,
    }
    (session_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (session_dir / "weapons.json").write_text(json.dumps(weapons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (session_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not (session_dir / "labels_current.json").exists():
        (session_dir / "labels_current.json").write_text("{}\n", encoding="utf-8")


class LabelRequestHandler(BaseHTTPRequestHandler):
    server: "LabelServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_bytes(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path in {"/samples.json", "/weapons.json", "/labels_current.json"}:
            file_path = self.server.session_dir / self.path.lstrip("/")
            if not file_path.exists():
                self._send_json({} if self.path.endswith("labels_current.json") else [], HTTPStatus.OK)
                return
            self._send_bytes(HTTPStatus.OK, file_path.read_bytes(), "application/json; charset=utf-8")
            return
        if self.path.startswith("/files/"):
            rel = unquote(self.path[len("/files/") :])
            file_path = (self.server.session_dir / rel).resolve()
            root = self.server.session_dir.resolve()
            if root not in file_path.parents or not file_path.exists():
                self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
                return
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._send_bytes(HTTPStatus.OK, file_path.read_bytes(), content_type)
            return
        self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/label":
            self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            _write_label(self.server.session_dir, payload)
        except Exception as exc:
            self._send_bytes(HTTPStatus.BAD_REQUEST, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send_json({"ok": True})


class LabelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], session_dir: Path) -> None:
        super().__init__(server_address, handler_class)
        self.session_dir = session_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and serve a local weapon-lamp labeling session.")
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--max-slots", type=int, default=96)
    parser.add_argument("--include-down", action="store_true")
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variant-sizes", type=int, nargs="+", default=[62, 82])
    parser.add_argument("--variant-angles", type=float, nargs="+", default=[-8.0, 0.0, 8.0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    if args.session_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = DEFAULT_SESSION_PARENT / f"session_{stamp}"
    else:
        session_dir = args.session_dir

    if args.videos:
        build_session(
            videos=args.videos,
            session_dir=session_dir,
            template_dir=args.template_dir,
            start=args.start,
            end=args.end,
            sample_interval=max(0.1, args.sample_interval),
            max_slots=args.max_slots,
            include_down=args.include_down,
            ally_side=args.ally_side,
            top_k=max(1, args.top_k),
            variant_sizes=args.variant_sizes,
            variant_angles=args.variant_angles,
        )

    samples_path = session_dir / "samples.json"
    if not samples_path.exists():
        parser.error("no samples.json found; pass videos or --session-dir for an existing session")

    print(f"session_dir={session_dir}")
    if args.no_serve:
        return
    server = LabelServer((args.host, args.port), LabelRequestHandler, session_dir)
    print(f"url=http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
