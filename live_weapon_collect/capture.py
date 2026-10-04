"""Spawned worker with a continuously draining capture thread and sparse detection."""
from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from live_weapon_collect.io_utils import FileLock, atomic_json
from weapon_lamp_detect.build_opening_dataset import opening_timer
from weapon_lamp_detect.detect_matches import MODEL, StartPredictor
from weapon_lamp_detect.match_data import (
    calibrated_slot, image_bytes, now, read_json, squid_detector,
)


@dataclass
class CaptureConfig:
    source: str = "0"
    backend: str = "auto"
    replay: bool = False
    width: int = 1920
    height: int = 1080
    fps: float = 60.
    scan_interval: float = .5
    frames: int = 5
    interval: float = 1.
    hud_delay_min: float = 8.
    fallback_offset: float = 20.
    hud_timeout: float = 35.
    cooldown: float = 60.
    reset_frames: int = 4
    fourcc: str = ""

    def validate(self):
        if self.backend not in {"auto", "dshow", "msmf", "v4l2"}:
            raise ValueError("backendが不正です")
        if self.backend in {"dshow", "msmf"} and os.name != "nt":
            raise ValueError("dshow/msmfはWindowsで使用してください")
        if self.backend == "v4l2" and os.name == "nt":
            raise ValueError("v4l2はLinux専用です")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.fps, self.scan_interval, self.interval, self.hud_timeout, self.cooldown)):
            raise ValueError("fps / interval / timeout / cooldownは有限の正数が必要です")
        if not math.isfinite(self.hud_delay_min) or not 0 <= self.hud_delay_min < self.hud_timeout:
            raise ValueError("0 ≤ hud-delay-min < hud-timeout が必要です")
        if not math.isfinite(self.fallback_offset) or not self.hud_delay_min <= self.fallback_offset < self.hud_timeout:
            raise ValueError("hud-delay-min ≤ fallback-offset < hud-timeout が必要です")
        if not 1 <= self.frames <= 10 or not 1 <= self.reset_frames <= 30 or self.width < 320 or self.height < 180:
            raise ValueError("frames / reset-frames / resolutionが不正です")
        if self.fourcc and len(self.fourcc) != 4:
            raise ValueError("fourccは4文字です")


class OpeningGate:
    """Prefer verified opening HUD; keep uncertain intro+20s frames for human review."""
    def __init__(self, config):
        self.config = config
        self.phase = "seeking_intro"
        self.intro = None
        self.valid_since = None
        self.settle_until = None
        self.blocked_until = -1.
        self.non_match_reads = 0
        self.last_intro = None
        self.verified = False
        self.intro_armed = True
        self.non_intro_reads = 0

    def observe(self, t, is_intro, opening_hud, hud_state):
        self.non_intro_reads = 0 if is_intro else self.non_intro_reads + 1
        if self.non_intro_reads >= self.config.reset_frames:
            self.intro_armed = True
        fresh_intro = is_intro and self.intro_armed
        if self.phase == "cooldown":
            self.non_match_reads = self.non_match_reads + 1 if hud_state != "match" else 0
            if t >= self.blocked_until and (fresh_intro or self.non_match_reads >= self.config.reset_frames):
                self.phase = "seeking_intro"
            else:
                return None
        if self.phase == "seeking_intro" or (self.phase in {"waiting_hud", "settling"} and fresh_intro):
            if fresh_intro:
                self.intro = t
                self.last_intro = t
                self.valid_since = None
                self.settle_until = None
                self.verified = False
                self.intro_armed = False
                self.phase = "waiting_hud"
            return None
        if self.phase in {"waiting_hud", "settling"}:
            if t - self.intro > self.config.hud_timeout:
                self.finish(t)
                return "timeout"
            if t - self.intro < self.config.hud_delay_min:
                return None
            # Do not discard the opening just because OCR/slot classification fails.
            # The fallback is evidence, NOT a verified opening or a weapon label.
            if (t - self.intro >= self.config.fallback_offset and not
                    (opening_hud and self.phase == "settling" and t >= self.settle_until)):
                self.phase = "collecting"
                self.verified = False
                return "capture_unverified"
            if not opening_hud:
                self.valid_since = None
                self.settle_until = None
                self.phase = "waiting_hud"
                return None
            if self.phase == "settling":
                if t >= self.settle_until:
                    self.phase = "collecting"
                    self.verified = True
                    return "capture"
            elif self.valid_since is None:
                self.valid_since = t
            elif t - self.valid_since >= self.config.scan_interval * .8:
                self.settle_until = self.valid_since + 1.
                self.phase = "settling"
            return None
        return None

    def manual(self, t):
        if self.phase == "collecting":
            raise ValueError("既に収集中です")
        # A manual capture after an old match is not associated with its intro.
        self.last_intro = self.intro if self.phase in {"waiting_hud", "settling"} else None
        self.verified = False
        self.intro_armed = False
        self.phase = "collecting"

    def finish(self, t):
        self.phase = "cooldown"
        # An uncertain/false candidate must not block the real intro for a minute.
        self.blocked_until = t + (self.config.cooldown if self.verified else 2.)
        self.non_match_reads = 0


class LatestCapture:
    """Driver reads never wait for labeling, PNG encoding or template matching."""
    def __init__(self, config, stopped):
        self.config, self.stopped = config, stopped
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.ended = False
        self.metadata = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def get(self):
        with self.lock:
            return self.latest

    def run(self):
        cap = None
        try:
            cfg = self.config
            backend = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                       "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}[cfg.backend]
            source = cfg.source if cfg.replay or not cfg.source.isdecimal() else int(cfg.source)
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                raise ValueError(f"入力を開けません: {cfg.source}（OBSとのデバイス競合も確認してください）")
            if not cfg.replay:
                if cfg.fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
                cap.set(cv2.CAP_PROP_FPS, cfg.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort, driver-dependent.
            reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else cfg.fps
            self.metadata = {"requested": asdict(cfg), "backend": cap.getBackendName(),
                             "reported_fps": reported_fps if math.isfinite(reported_fps) else None,
                             "replay_effective_fps": fps if cfg.replay else None,
                             "reported_width": int(cap.get(3)),
                             "reported_height": int(cap.get(4)), "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
                             "timing": "video frame_index / fps" if cfg.replay else "monotonic elapsed since capture open",
                             "frame_index_definition": "decoded source frame index" if cfg.replay else "successful camera read counter (not device hardware frame number)"}
            started = time.monotonic()
            index, failures = 0, 0
            while not self.stopped.is_set():
                ok, frame = cap.read()
                if not ok:
                    if cfg.replay:
                        self.ended = True
                        return
                    failures += 1
                    if failures >= 30:
                        raise ValueError("camera readが連続失敗しました。接続・OBS競合を確認してください")
                    self.stopped.wait(.05)
                    continue
                failures = 0
                t = index / fps if cfg.replay else time.monotonic() - started
                if cfg.replay and self.stopped.wait(max(0., started + t - time.monotonic())):
                    return
                with self.lock:
                    self.latest = (frame, index, t)
                index += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ended = True
            if cap is not None:
                cap.release()


class CaptureWriter:
    def __init__(self, directory, session, source, device, config, first, trigger, intro):
        frame, index, t = first
        self.directory = Path(directory)
        self.config = config
        match_id = f'{session}_{index:09d}_{uuid.uuid4().hex[:6]}'
        self.path = self.directory / "matches" / match_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = {"schema_version": 1, "match_id": match_id, "session_id": session,
                       "source_kind": "video_replay" if config.replay else "camera",
                       "source": source, "device": device, "created_at": now(), "updated_at": now(),
                       "status": "collecting", "trigger": trigger, "intro_timestamp": intro,
                       "opening_timestamp": t, "opening_frame_index": index,
                       "resolution": {"width": frame.shape[1], "height": frame.shape[0]},
                       "sampling_interval": config.interval, "target_frames": config.frames,
                       "frames": [], "warnings": [],
                       "needs_review": trigger != "intro_hud_timer"}
        atomic_json(self.path / "capture.json", self.record)

    def append(self, item, reading, seconds=None):
        frame, index, t = item
        if self.record["frames"] and index <= self.record["frames"][-1]["frame_index"]:
            return False
        if (frame.shape[1], frame.shape[0]) != (self.record["resolution"]["width"], self.record["resolution"]["height"]):
            raise ValueError("収集中に解像度が変わりました")
        ordinal = len(self.record["frames"])
        data = image_bytes(frame, ".png")
        name = f"frames/{ordinal:02d}.png"
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        slots = {}
        for side in ("left", "right"):
            for i in range(4):
                slot = calibrated_slot(frame, reading, side, i)
                slots[f"{side}{i}"] = slot.to_dict()
        row = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "ordinal": ordinal,
               "timestamp": t, "frame_index": index, "captured_at": now(),
               "seconds_after_opening": t - self.record["opening_timestamp"],
               "hud_state": reading.hud_state, "slots": slots, "remaining_seconds": seconds}
        self.record["frames"].append(row)
        self.record["updated_at"] = now()
        atomic_json(self.path / "capture.json", self.record)
        return True

    def finish(self, status="complete", warning=None):
        self.record.update(status=status, updated_at=now())
        if warning:
            self.record["warnings"].append(warning)
        atomic_json(self.path / "capture.json", self.record)


def recover_interrupted(directory):
    """Only called after acquiring the exclusive collector lock."""
    for path in (Path(directory) / "matches").glob("*/capture.json"):
        record = read_json(path)
        if record["status"] == "collecting":
            record.update(status="interrupted", updated_at=now())
            record["warnings"].append("前回の収集が中断されました。保存済みframeのみ残しています")
            atomic_json(path, record)


class CollectorLock(FileLock):
    """OS file lock; automatically released after process crashes, Windows included."""
    def __init__(self, directory):
        super().__init__(Path(directory) / "collector.lock", "このsession-dirは別の収集アプリが使用中です")


def capture_worker(directory, config, commands, stopped, errors=None):
    """Browser/server is in the parent process. Never accept labels in this worker."""
    directory = Path(directory)
    lock = None
    writer = None
    status = {"status": "starting", "phase": "starting", "error": None, "updated_at": now()}
    last_status, last_preview, last_scan, last_index = -1., -1., -1., -1
    history = deque(maxlen=80)
    def publish():
        status["updated_at"] = now()
        atomic_json(directory / "collector_status.json", status)
    try:
        config.validate()
        lock = CollectorLock(directory)
        recover_interrupted(directory)
        session = uuid.uuid4().hex[:16]
        predictor, detector = StartPredictor(), squid_detector()
        if detector.timer_ocr is None:
            status["last_notice"] = "タイマーOCRが利用できないため、開始候補からの時間で要確認画像を保存します"
        reader = LatestCapture(config, stopped)
        reader.thread.start()
        gate = OpeningGate(config)
        publish()
        while not stopped.wait(.025):
            item = reader.get()
            if reader.error:
                raise ValueError(reader.error)
            if item is None:
                if reader.ended:
                    break
                continue
            frame, index, t = item
            if index == last_index:
                if reader.ended:
                    break
                continue
            last_index = index
            manual = False
            while True:
                try:
                    command = commands.get_nowait()
                    if command == "manual":
                        if writer is None:
                            manual = True
                        else:
                            status["last_notice"] = "収集中のため手動収集要求を無視しました"
                except queue.Empty:
                    break
            scan_due = t - last_scan >= config.scan_interval
            # Never catch up missed samples with a burst of adjacent frames.
            save_due = writer is not None and (not writer.record["frames"] or
                       t >= writer.record["frames"][-1]["timestamp"] + config.interval)
            reading = None
            if scan_due or save_due or manual:
                hud_error = None
                try:
                    reading = detector.read_frame(frame, t, index)
                except Exception as exc:
                    # The reused detector calls OCR internally too. In this worker
                    # it is not shared: retry without OCR rather than lose images.
                    if detector.timer_ocr is None:
                        raise
                    hud_error = str(exc)
                    timer_ocr = detector.timer_ocr
                    detector.timer_ocr = None
                    try:
                        reading = detector.read_frame(frame, t, index)
                    finally:
                        detector.timer_ocr = timer_ocr
                timer = None
                timer_error = None
                if detector.timer_ocr is not None:
                    try:
                        timer = detector.timer_ocr.read_frame(frame, timestamp=t, frame_index=index)
                    except Exception as exc:
                        timer_error = str(exc)
                seconds = getattr(timer, "seconds", None)
                if scan_due:
                    # Track real intro edges even while collecting; synthetic False
                    # readings would re-arm a continuously displayed false intro.
                    start = predictor.predict(frame)
                    alive = sum(calibrated_slot(frame, reading, side, i).state == "alive"
                                for side in ("left", "right") for i in range(4))
                    valid = reading.hud_state == "match" and getattr(timer, "kind", None) == "time" and opening_timer(seconds)
                    diagnostic = {"timestamp": t, "frame_index": index, "is_intro": bool(start["is_start"]),
                                  "intro_score": start.get("score"), "hud_state": reading.hud_state,
                                  "timer_kind": getattr(timer, "kind", None), "remaining_seconds": seconds,
                                  "alive_slots": alive, "opening_hud_valid": valid,
                                  "timer_error": timer_error, "hud_error": hud_error}
                    history.append(diagnostic)
                    action = None if manual else gate.observe(t, start["is_start"], valid, reading.hud_state)
                    last_scan = t
                    status.update(hud_state=reading.hud_state, remaining_seconds=seconds,
                                  alive_slots=alive, opening_hud_valid=valid,
                                  intro_timestamp=gate.intro,
                                  fallback_due=gate.intro + config.fallback_offset if gate.phase in {"waiting_hud", "settling"} else None,
                                  detection=diagnostic)
                    if action == "timeout":
                        status["last_notice"] = "入力処理が遅れて開始の保存窓を超えました。次の開始候補を待ちます（60秒の待機はしません）"
                    if action in {"capture", "capture_unverified"} and not manual:
                        verified = action == "capture"
                        writer = CaptureWriter(directory, session, config.source, reader.metadata, config, item,
                                               "intro_hud_timer" if verified else "intro_offset_unverified", gate.intro)
                        writer.record["detection_history"] = list(history)
                        writer.record["opening_hud_verified"] = verified
                        if not verified:
                            writer.record["warnings"].append(
                                f"開始候補+{t-gate.intro:.1f}秒で保存しました。HUD/タイマー未確認のため開始画像か人間が確認してください"
                                f"（HUD={reading.hud_state}、残り={seconds}秒、alive={alive}）")
                        status["last_notice"] = "開始画像を収集中" if verified else "開始候補から画像を収集中（要確認・自動ラベルなし）"
                        save_due = True
                if manual:
                    gate.manual(t)
                    writer = CaptureWriter(directory, session, config.source, reader.metadata, config, item, "manual_current_frame", gate.last_intro)
                    writer.record["warnings"].append("手動収集です。開始直後HUDか人間が確認してください")
                    status["last_notice"] = "手動で開始画像を収集中（要確認）"
                    save_due = True
                if save_due and writer:
                    elapsed = t - writer.record["opening_timestamp"]
                    # A slow device cannot extend capture into later gameplay.
                    if elapsed >= config.frames * config.interval:
                        writer.finish("interrupted", "開始の収集窓を超えたため不足frameを後から補っていません")
                        writer = None
                        gate.finish(t)
                    else:
                        writer.append(item, reading, seconds)
                        status["last_match_id"] = writer.record["match_id"]
                        status["saved_frames"] = len(writer.record["frames"])
                        if len(writer.record["frames"]) >= config.frames:
                            status["last_notice"] = f'{config.frames}枚を保存しました' + ("（開始画像か要確認）" if writer.record["needs_review"] else "（武器名は人力入力してください）")
                            writer.finish()
                            writer = None
                            gate.finish(t)
            clock = time.monotonic()
            status.update(status="running", phase=gate.phase, timestamp=t, frame_index=index,
                          session_id=session, device=reader.metadata,
                          actual_resolution={"width": frame.shape[1], "height": frame.shape[0]})
            if clock - last_preview >= 1.:
                preview = cv2.resize(frame, (960, round(960 * frame.shape[0] / frame.shape[1])))
                data = image_bytes(preview)
                temporary = directory / "preview.jpg.tmp"
                temporary.write_bytes(data)
                try:
                    os.replace(temporary, directory / "preview.jpg")
                except PermissionError:
                    # Windows can briefly hold the preview open in another thread.
                    # A transient preview failure must not stop acquisition.
                    pass
                last_preview = clock
            if clock - last_status >= 1.:
                publish()
                last_status = clock
            if reader.ended:
                break
        status.update(status="stopped", phase="stopped", last_notice="入力終了または収集停止。保存済み試合は入力できます")
    except Exception as exc:
        status.update(status="error", phase="error", error=str(exc))
    finally:
        stopped.set()
        if writer:
            writer.finish("interrupted", "入力終了・停止・エラーにより収集中断")
        # Never overwrite the active collector's status when failing its lock.
        if lock is not None:
            publish()
            lock.close()
        elif status["status"] == "error" and errors is not None:
            # Report startup failures to the parent without touching another session.
            errors.put(status["error"])
