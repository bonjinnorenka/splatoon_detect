from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


GaugeState = Literal["unknown", "not_ready", "near_ready", "ready", "using"]


@dataclass(frozen=True)
class ROIConfig:
    x_ratio: float = 0.82
    y_ratio: float = 0.02
    w_ratio: float = 0.17
    h_ratio: float = 0.20

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> "ROIConfig":
        return cls(
            x_ratio=float(data.get("x_ratio", cls.x_ratio)),
            y_ratio=float(data.get("y_ratio", cls.y_ratio)),
            w_ratio=float(data.get("w_ratio", cls.w_ratio)),
            h_ratio=float(data.get("h_ratio", cls.h_ratio)),
        )

    def to_rect(self, shape: Sequence[int]) -> tuple[int, int, int, int]:
        height, width = int(shape[0]), int(shape[1])
        left = max(0, min(width - 1, int(round(width * self.x_ratio))))
        top = max(0, min(height - 1, int(round(height * self.y_ratio))))
        right = max(left + 1, min(width, int(round(width * (self.x_ratio + self.w_ratio)))))
        bottom = max(top + 1, min(height, int(round(height * (self.y_ratio + self.h_ratio)))))
        return left, top, right, bottom


@dataclass
class GaugeReading:
    timestamp: float
    frame_index: int
    timestamp_ms: int
    state: GaugeState
    confidence: float
    roi_visible: bool
    smoothed: bool = False
    raw_state: GaugeState | None = None
    raw_confidence: float | None = None
    features: dict[str, object] = field(default_factory=dict)
    roi_rect: tuple[int, int, int, int] | None = None

    def gauge_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "roi_visible": bool(self.roi_visible),
            "timestamp_ms": int(self.timestamp_ms),
        }

    def to_dict(self, include_debug: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "timestamp": round(float(self.timestamp), 4),
            "timestamp_ms": int(self.timestamp_ms),
            "frame_index": int(self.frame_index),
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "roi_visible": bool(self.roi_visible),
            "smoothed": bool(self.smoothed),
        }
        if self.raw_state is not None:
            data["raw_state"] = self.raw_state
        if self.raw_confidence is not None:
            data["raw_confidence"] = round(float(self.raw_confidence), 4)
        if include_debug:
            data["features"] = _round_nested(self.features)
            data["roi_rect"] = list(self.roi_rect) if self.roi_rect is not None else None
        return data

    def to_row(self) -> dict[str, object]:
        return {
            "timestamp": round(float(self.timestamp), 4),
            "timestamp_ms": int(self.timestamp_ms),
            "frame_index": int(self.frame_index),
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "roi_visible": bool(self.roi_visible),
            "raw_state": self.raw_state or "",
            "raw_confidence": "" if self.raw_confidence is None else round(float(self.raw_confidence), 4),
            "ready_score": round(float(self.features.get("ready_score", 0.0)), 4),
            "visible_score": round(float(self.features.get("visible_score", 0.0)), 4),
            "fill_coverage": round(float(self.features.get("fill_coverage", 0.0)), 4),
            "ready_coverage": round(float(self.features.get("ready_coverage", 0.0)), 4),
            "smoothed": bool(self.smoothed),
        }


@dataclass
class GaugeUpdate:
    special_gauge: GaugeReading
    events: list[dict[str, object]]

    def to_dict(self, include_debug: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "special_gauge": self.special_gauge.gauge_dict(),
            "events": [_round_nested(event) for event in self.events],
        }
        if include_debug:
            data["reading"] = self.special_gauge.to_dict(include_debug=True)
        return data

    def to_row(self) -> dict[str, object]:
        row = self.special_gauge.to_row()
        row["events"] = ",".join(str(event.get("type", "")) for event in self.events)
        row["event_confidences"] = ",".join(f"{float(event.get('confidence', 0.0)):.4f}" for event in self.events)
        return row


@dataclass(frozen=True)
class _CircleCandidate:
    cx: int
    cy: int
    radius: int
    score: float
    dark_core: float
    annulus_ui: float


def _round_nested(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, tuple):
        return [_round_nested(item) for item in value]
    return value


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _mask_fraction(mask: np.ndarray, area: np.ndarray) -> float:
    count = int(area.sum())
    if count <= 0:
        return 0.0
    return float(mask[area].mean())


def _circle_geometry(shape: Sequence[int], cx: int, cy: int) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:height, 0:width]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def _angular_coverage(mask: np.ndarray, cx: int, cy: int, area: np.ndarray, bins: int = 32, min_density: float = 0.08) -> float:
    ys, xs = np.where(area)
    if len(xs) == 0:
        return 0.0
    angles = (np.arctan2(ys - cy, xs - cx) + 2.0 * math.pi) % (2.0 * math.pi)
    bucket = np.floor(angles / (2.0 * math.pi) * bins).astype(np.int32)
    covered = 0
    for index in range(bins):
        points = bucket == index
        if not np.any(points):
            continue
        density = float(mask[ys[points], xs[points]].mean())
        if density >= min_density:
            covered += 1
    return covered / bins


def crop_roi(frame: np.ndarray, roi: ROIConfig) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi.to_rect(frame.shape)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _special_icon_score(crop: np.ndarray) -> tuple[float, float]:
    height, width = crop.shape[:2]
    x1 = int(round(width * 0.15))
    x2 = int(round(width * 0.52))
    y1 = 0
    y2 = int(round(height * 0.34))
    icon_crop = crop[y1:y2, x1:x2]
    if icon_crop.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    yellow = (hue >= 18) & (hue <= 47) & (sat >= 70) & (val >= 120)
    yellow_fraction = float(yellow.mean())
    score = min(1.0, yellow_fraction / 0.16)
    return float(score), yellow_fraction


class SpecialGaugeDetector:
    def __init__(
        self,
        roi: ROIConfig | None = None,
        ready_threshold: float = 0.58,
        visible_threshold: float = 0.43,
        enable_near_ready: bool = True,
    ) -> None:
        self.roi = roi or ROIConfig()
        self.ready_threshold = ready_threshold
        self.visible_threshold = visible_threshold
        self.enable_near_ready = enable_near_ready

    def read_frame(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = -1) -> GaugeReading:
        crop, roi_rect = crop_roi(frame, self.roi)
        timestamp_ms = int(round(timestamp * 1000.0))
        state, confidence, roi_visible, features = self._classify_crop(crop)
        if features.get("circle") is not None:
            circle = features["circle"]
            if isinstance(circle, dict):
                x1, y1, _x2, _y2 = roi_rect
                circle["frame_center"] = [int(circle["cx"]) + x1, int(circle["cy"]) + y1]

        return GaugeReading(
            timestamp=timestamp,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            state=state,
            confidence=confidence,
            roi_visible=roi_visible,
            smoothed=False,
            features=features,
            roi_rect=roi_rect,
        )

    def _classify_crop(self, crop: np.ndarray) -> tuple[GaugeState, float, bool, dict[str, object]]:
        if crop.size == 0:
            return "unknown", 0.0, False, {"visible_score": 0.0}

        candidate = self._find_gauge_circle(crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        dist = _circle_geometry(crop.shape, candidate.cx, candidate.cy)

        radius = candidate.radius
        inner = dist <= radius * 0.66
        annulus = (dist >= radius * 0.64) & (dist <= radius * 1.03)
        outer = (dist >= radius * 1.03) & (dist <= radius * 1.18)

        bright_white = (val >= 185) & (sat <= 105)
        pale_yellow = (val >= 175) & (sat <= 170) & (hue >= 16) & (hue <= 46)
        pale_pink = (val >= 178) & (sat <= 165) & (hue >= 140) & (hue <= 178)
        bright_ready = bright_white | pale_yellow | pale_pink

        colored_fill = (val >= 115) & (sat >= 85)
        orange_fill = (val >= 115) & (sat >= 70) & (hue >= 5) & (hue <= 36)
        ui_fill = bright_ready | colored_fill | orange_fill
        icon_score, icon_yellow_fraction = _special_icon_score(crop)

        ready_fraction = _mask_fraction(bright_ready, annulus)
        ready_outer_fraction = _mask_fraction(bright_ready, outer)
        ready_coverage = _angular_coverage(bright_ready, candidate.cx, candidate.cy, annulus, bins=32, min_density=0.085)
        fill_coverage = _angular_coverage(ui_fill, candidate.cx, candidate.cy, annulus, bins=32, min_density=0.10)
        dark_inner = _mask_fraction(val < 75, inner)

        ready_score = _clamp(
            0.46 * min(1.0, ready_coverage / 0.78)
            + 0.32 * min(1.0, ready_fraction / 0.26)
            + 0.22 * min(1.0, ready_outer_fraction / 0.12)
        )
        raw_visible_score = _clamp(
            0.54 * candidate.score
            + 0.22 * min(1.0, dark_inner / 0.48)
            + 0.24 * min(1.0, fill_coverage / 0.42)
        )
        visible_score = _clamp(raw_visible_score * (0.25 + 0.75 * icon_score))

        features: dict[str, object] = {
            "visible_score": visible_score,
            "raw_visible_score": raw_visible_score,
            "icon_score": icon_score,
            "icon_yellow_fraction": icon_yellow_fraction,
            "ready_score": ready_score,
            "ready_fraction": ready_fraction,
            "ready_outer_fraction": ready_outer_fraction,
            "ready_coverage": ready_coverage,
            "fill_coverage": fill_coverage,
            "dark_inner": dark_inner,
            "circle": {
                "cx": candidate.cx,
                "cy": candidate.cy,
                "radius": candidate.radius,
                "score": candidate.score,
                "dark_core": candidate.dark_core,
                "annulus_ui": candidate.annulus_ui,
            },
        }

        roi_visible = visible_score >= self.visible_threshold
        if not roi_visible:
            confidence = _clamp(0.18 + 0.55 * (1.0 - visible_score))
            return "unknown", confidence, False, features

        if ready_score >= self.ready_threshold:
            confidence = _clamp(0.70 + 0.30 * min(1.0, (ready_score - self.ready_threshold) / 0.30))
            return "ready", confidence, True, features

        if self.enable_near_ready and fill_coverage >= 0.82 and ready_score >= 0.36:
            confidence = _clamp(0.66 + 0.22 * min(1.0, (fill_coverage - 0.82) / 0.18))
            return "near_ready", confidence, True, features

        not_ready_separation = 1.0 - min(1.0, ready_score / max(self.ready_threshold, 1e-6))
        confidence = _clamp(0.63 + 0.28 * visible_score + 0.12 * not_ready_separation, high=0.96)
        return "not_ready", confidence, True, features

    def _find_gauge_circle(self, crop: np.ndarray) -> _CircleCandidate:
        height, width = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        dark = val < 88
        bright_or_colored = ((val > 125) & (sat > 75)) | ((val > 175) & (sat < 140))

        expected_cx = int(round(width * 0.66))
        expected_cy = int(round(height * 0.48))
        expected_r = int(round(height * 0.37))
        radii = np.linspace(max(24, height * 0.30), max(25, height * 0.43), 6)
        x_offsets = np.linspace(-width * 0.08, width * 0.08, 7)
        y_offsets = np.linspace(-height * 0.09, height * 0.09, 7)

        best: _CircleCandidate | None = None
        for radius_value in radii:
            radius = int(round(radius_value))
            for dx in x_offsets:
                cx = int(round(expected_cx + dx))
                if cx - radius < 0 or cx + radius >= width:
                    continue
                for dy in y_offsets:
                    cy = int(round(expected_cy + dy))
                    if cy - radius < 0 or cy + radius >= height:
                        continue
                    dist = _circle_geometry(crop.shape, cx, cy)
                    inner = dist <= radius * 0.66
                    annulus = (dist >= radius * 0.62) & (dist <= radius * 1.08)
                    outer = (dist >= radius * 1.08) & (dist <= radius * 1.22)

                    dark_core = _mask_fraction(dark, inner)
                    annulus_ui = _mask_fraction(dark | bright_or_colored, annulus)
                    outer_ui = _mask_fraction(dark | bright_or_colored, outer)
                    position_penalty = math.hypot((cx - expected_cx) / max(1, width * 0.12), (cy - expected_cy) / max(1, height * 0.13))
                    radius_penalty = abs(radius - expected_r) / max(1, height * 0.12)
                    score = (
                        0.52 * min(1.0, dark_core / 0.46)
                        + 0.32 * min(1.0, annulus_ui / 0.48)
                        + 0.16 * min(1.0, outer_ui / 0.22)
                    )
                    score *= max(0.74, 1.0 - 0.08 * position_penalty - 0.06 * radius_penalty)
                    candidate = _CircleCandidate(cx, cy, radius, _clamp(score), dark_core, annulus_ui)
                    if best is None or candidate.score > best.score:
                        best = candidate

        if best is not None:
            return best
        return _CircleCandidate(expected_cx, expected_cy, max(12, expected_r), 0.0, 0.0, 0.0)


class SpecialGaugeTracker:
    def __init__(
        self,
        detector: SpecialGaugeDetector,
        window_size: int = 5,
        ready_votes: int = 4,
        state_votes: int = 3,
        history_ms: int = 7000,
        death_lookback_ms: int = 2000,
        ready_unused_ms: int = 15000,
        using_hold_ms: int = 1800,
        event_min_confidence: float = 0.85,
        death_event_min_confidence: float = 0.85,
    ) -> None:
        self.detector = detector
        self.window_size = max(1, int(window_size))
        self.ready_votes = max(1, min(int(ready_votes), self.window_size))
        self.state_votes = max(1, min(int(state_votes), self.window_size))
        self.history_ms = max(1000, int(history_ms))
        self.death_lookback_ms = max(250, int(death_lookback_ms))
        self.ready_unused_ms = max(1000, int(ready_unused_ms))
        self.using_hold_ms = max(0, int(using_hold_ms))
        self.event_min_confidence = float(event_min_confidence)
        self.death_event_min_confidence = float(death_event_min_confidence)

        self.raw_history: deque[GaugeReading] = deque(maxlen=self.window_size)
        self.state_history: deque[GaugeReading] = deque()
        self.stable_state: GaugeState = "unknown"
        self.ready_started_ms: int | None = None
        self.ready_event_emitted = False
        self.ready_unused_emitted = False
        self.ready_streak_confidence = 0.0
        self.using_until_ms = -1
        self.recent_death_until_ms = -1

    def read_frame(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = -1) -> GaugeUpdate:
        raw = self.detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        return self.update(raw)

    def update(self, raw: GaugeReading) -> GaugeUpdate:
        self.raw_history.append(raw)
        state, confidence, roi_visible = self._smooth_state()
        events: list[dict[str, object]] = []
        previous_state = self.stable_state
        previous_ready_confidence = self.ready_streak_confidence

        event_state = state
        if raw.timestamp_ms <= self.using_until_ms and state in {"not_ready", "near_ready"}:
            event_state = "using"
            confidence = max(confidence, 0.74)

        used_detected = (
            previous_state == "ready"
            and state in {"not_ready", "near_ready"}
            and raw.roi_visible
            and raw.timestamp_ms > self.recent_death_until_ms
        )
        if used_detected:
            ready_confidence = previous_ready_confidence if previous_ready_confidence > 0.0 else confidence
            event_confidence = _clamp(min(ready_confidence, raw.confidence) * 0.96)
            if event_confidence >= self.event_min_confidence:
                events.append({"type": "special_used", "confidence": round(event_confidence, 4)})
            self.using_until_ms = raw.timestamp_ms + self.using_hold_ms
            event_state = "using"
            confidence = max(confidence, event_confidence)
            self.ready_started_ms = None
            self.ready_event_emitted = False
            self.ready_unused_emitted = False
            self.ready_streak_confidence = 0.0

        stable = replace(
            raw,
            state=event_state,
            confidence=confidence,
            roi_visible=roi_visible,
            smoothed=True,
            raw_state=raw.state,
            raw_confidence=raw.confidence,
        )

        if event_state == "ready":
            if previous_state != "ready":
                self.ready_started_ms = raw.timestamp_ms
                self.ready_event_emitted = False
                self.ready_unused_emitted = False
                self.ready_streak_confidence = confidence
            else:
                self.ready_streak_confidence = max(self.ready_streak_confidence, confidence)

            if not self.ready_event_emitted and confidence >= self.event_min_confidence:
                events.append({"type": "special_ready", "confidence": round(confidence, 4)})
                self.ready_event_emitted = True
            if (
                self.ready_started_ms is not None
                and not self.ready_unused_emitted
                and raw.timestamp_ms - self.ready_started_ms >= self.ready_unused_ms
            ):
                duration_sec = (raw.timestamp_ms - self.ready_started_ms) / 1000.0
                event_confidence = _clamp(confidence * 0.98)
                if event_confidence >= self.event_min_confidence:
                    events.append(
                        {
                            "type": "special_ready_unused_too_long",
                            "ready_duration_sec": round(duration_sec, 3),
                            "confidence": round(event_confidence, 4),
                        }
                    )
                    self.ready_unused_emitted = True
        elif event_state != "using":
            self.ready_started_ms = None
            self.ready_event_emitted = False
            self.ready_unused_emitted = False
            self.ready_streak_confidence = 0.0

        self.stable_state = event_state
        self._append_state(stable)
        return GaugeUpdate(stable, events)

    def handle_external_event(self, event: dict[str, object]) -> list[dict[str, object]]:
        if event.get("type") != "player_death":
            return []
        timestamp_ms = int(event.get("timestamp_ms", 0))
        source_confidence = float(event.get("confidence", 1.0))
        self.recent_death_until_ms = max(self.recent_death_until_ms, timestamp_ms + self.death_lookback_ms)

        best_ready = 0.0
        for reading in self.state_history:
            if reading.state != "ready":
                continue
            delta = timestamp_ms - reading.timestamp_ms
            if 0 <= delta <= self.death_lookback_ms:
                best_ready = max(best_ready, float(reading.confidence))

        self.raw_history.clear()
        self.state_history.clear()
        self.stable_state = "unknown"
        self.ready_started_ms = None
        self.ready_event_emitted = False
        self.ready_unused_emitted = False
        self.ready_streak_confidence = 0.0

        if best_ready <= 0.0:
            return []

        confidence = _clamp(source_confidence * (0.08 + 0.92 * best_ready))
        if confidence < self.death_event_min_confidence:
            return []

        return [{"type": "death_with_special_ready", "confidence": round(confidence, 4)}]

    def reset(self) -> None:
        self.raw_history.clear()
        self.state_history.clear()
        self.stable_state = "unknown"
        self.ready_started_ms = None
        self.ready_event_emitted = False
        self.ready_unused_emitted = False
        self.ready_streak_confidence = 0.0
        self.using_until_ms = -1
        self.recent_death_until_ms = -1

    def _smooth_state(self) -> tuple[GaugeState, float, bool]:
        history = list(self.raw_history)
        if len(history) < min(self.window_size, self.ready_votes):
            latest = history[-1]
            confidence = _clamp(latest.confidence * 0.82, high=0.82)
            return latest.state if latest.confidence >= 0.70 else "unknown", confidence, latest.roi_visible

        latest = history[-1]
        visible_votes = sum(1 for reading in history if reading.roi_visible and reading.confidence >= 0.55)
        roi_visible = visible_votes >= self.state_votes
        if not roi_visible:
            confidence = float(np.mean([reading.confidence for reading in history[-self.state_votes :]]))
            return "unknown", _clamp(confidence, high=0.80), False

        votes: dict[GaugeState, float] = {
            "unknown": 0.0,
            "not_ready": 0.0,
            "near_ready": 0.0,
            "ready": 0.0,
            "using": 0.0,
        }
        confidence_sum: dict[GaugeState, float] = {state: 0.0 for state in votes}
        counts: dict[GaugeState, int] = {state: 0 for state in votes}
        for offset, reading in enumerate(reversed(history)):
            recency = 1.0 + (len(history) - offset) * 0.08
            weight = recency * max(0.05, reading.confidence)
            votes[reading.state] += weight
            confidence_sum[reading.state] += float(reading.confidence)
            counts[reading.state] += 1

        ready_count = counts["ready"]
        if ready_count >= self.ready_votes:
            confidence = confidence_sum["ready"] / max(1, ready_count)
            confidence = _clamp(0.82 + 0.18 * min(1.0, (confidence - 0.70) / 0.30))
            return "ready", confidence, True

        active_not_ready_count = counts["not_ready"] + counts["near_ready"]
        if active_not_ready_count >= self.ready_votes:
            if counts["near_ready"] >= self.ready_votes:
                confidence = confidence_sum["near_ready"] / max(1, counts["near_ready"])
                return "near_ready", _clamp(confidence), True
            confidence = (
                confidence_sum["not_ready"] + confidence_sum["near_ready"]
            ) / max(1, active_not_ready_count)
            return "not_ready", _clamp(confidence), True

        dominant = max(votes, key=votes.get)
        confidence = confidence_sum[dominant] / max(1, counts[dominant])
        if dominant == "unknown" or votes[dominant] < 0.48 * sum(votes.values()):
            return latest.state if latest.confidence >= 0.72 else "unknown", _clamp(latest.confidence, high=0.84), latest.roi_visible
        return dominant, _clamp(confidence, high=0.86), True

    def _append_state(self, reading: GaugeReading) -> None:
        self.state_history.append(reading)
        cutoff = reading.timestamp_ms - self.history_ms
        while self.state_history and self.state_history[0].timestamp_ms < cutoff:
            self.state_history.popleft()


def sample_video(
    video_path: Path,
    detector: SpecialGaugeDetector | None = None,
    sample_interval: float = 0.25,
    start: float = 0.0,
    end: float | None = None,
    smooth: bool = True,
    death_events: Iterable[dict[str, object]] | None = None,
) -> list[GaugeUpdate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    detector = detector or SpecialGaugeDetector()
    tracker = SpecialGaugeTracker(detector) if smooth else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0
    stop = duration if end is None else min(end, duration)

    pending_deaths = sorted(death_events or [], key=lambda item: int(item.get("timestamp_ms", 0)))
    death_index = 0
    updates: list[GaugeUpdate] = []

    timestamp = max(0.0, start)
    while timestamp <= stop + 1e-6:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        if tracker is not None:
            pre_events: list[dict[str, object]] = []
            timestamp_ms = int(round(timestamp * 1000.0))
            while death_index < len(pending_deaths) and int(pending_deaths[death_index].get("timestamp_ms", 0)) <= timestamp_ms:
                pre_events.extend(tracker.handle_external_event(pending_deaths[death_index]))
                death_index += 1
            update = tracker.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
            update.events = pre_events + update.events
        else:
            reading = detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
            update = GaugeUpdate(reading, [])
        updates.append(update)
        timestamp += sample_interval

    cap.release()
    return updates


def load_death_events(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("death event JSON must be a list or JSONL")
        return [dict(item) for item in data]

    events: list[dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(dict(json.loads(line)))
    return events


def write_jsonl(path: Path, updates: Iterable[GaugeUpdate], include_debug: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for update in updates:
            file.write(json.dumps(update.to_dict(include_debug=include_debug), ensure_ascii=False) + "\n")


def write_csv(path: Path, updates: Iterable[GaugeUpdate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "timestamp_ms",
        "frame_index",
        "state",
        "confidence",
        "roi_visible",
        "raw_state",
        "raw_confidence",
        "ready_score",
        "visible_score",
        "fill_coverage",
        "ready_coverage",
        "smoothed",
        "events",
        "event_confidences",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for update in updates:
            writer.writerow(update.to_row())


def draw_debug_overlay(frame: np.ndarray, reading: GaugeReading) -> np.ndarray:
    out = frame.copy()
    if reading.roi_rect is not None:
        x1, y1, x2, y2 = reading.roi_rect
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 255), 2)

    circle = reading.features.get("circle")
    if isinstance(circle, dict) and "frame_center" in circle:
        cx, cy = [int(value) for value in circle["frame_center"]]
        radius = int(circle["radius"])
        color = (0, 240, 0) if reading.state == "ready" else (0, 160, 255)
        if reading.state == "unknown":
            color = (0, 220, 220)
        cv2.circle(out, (cx, cy), radius, color, 2)
        cv2.circle(out, (cx, cy), max(2, int(radius * 0.64)), color, 1)
        cv2.circle(out, (cx, cy), max(2, int(radius * 1.18)), color, 1)

    label = f"special {reading.state} {reading.confidence:.2f}"
    cv2.putText(out, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out
