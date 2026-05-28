from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import cv2
import numpy as np


RankedRule = Literal["splat_zones", "tower_control", "rainmaker", "clam_blitz", "rule_unknown"]
Leader = Literal["our", "enemy", "tie", "unknown"]
ProgressDirection = Literal["our_count_decreasing", "enemy_count_decreasing", "stable", "unknown"]
Freshness = Literal["fresh", "stale", "unknown"]
SideName = Literal["left", "right"]
DistanceClass = Literal["near", "mid", "far", "offscreen", "unknown"]


DIGIT_TEMPLATE_SIZE = (42, 28)
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "match_time_ocr" / "templates" / "digits"

# 1920x1080 Splatoon 3 16:9 HUD layout, expressed as relative rectangles.
SIDE_COUNT_ROI_CANDIDATES: dict[SideName, tuple[tuple[str, tuple[float, float, float, float]], ...]] = {
    "left": (
        ("standard_plate", (0.405, 0.125, 0.470, 0.225)),
        ("standard", (0.375, 0.095, 0.500, 0.245)),
        ("standard_tight", (0.400, 0.100, 0.485, 0.235)),
        ("tower", (0.300, 0.145, 0.445, 0.305)),
    ),
    "right": (
        ("standard_plate", (0.535, 0.125, 0.605, 0.225)),
        ("standard", (0.500, 0.095, 0.630, 0.245)),
        ("standard_tight", (0.525, 0.100, 0.610, 0.235)),
        ("tower", (0.555, 0.145, 0.700, 0.305)),
    ),
}
SIDE_PENALTY_ROIS: dict[SideName, tuple[float, float, float, float]] = {
    "left": (0.350, 0.085, 0.462, 0.150),
    "right": (0.538, 0.085, 0.650, 0.150),
}
OBJECTIVE_SEARCH_ROI = (0.070, 0.170, 0.930, 0.875)
COUNT_SEARCH_ROIS: dict[str, tuple[float, float, float, float]] = {
    "standard": (0.345, 0.105, 0.660, 0.250),
    "tower": (0.245, 0.120, 0.755, 0.325),
}
COUNT_SLOT_LAYOUTS: dict[int, tuple[tuple[float, float, float, float], ...]] = {
    1: ((0.34, 0.32, 0.66, 0.98),),
    2: ((0.18, 0.30, 0.50, 0.98), (0.48, 0.30, 0.80, 0.98)),
    3: ((0.04, 0.30, 0.34, 0.98), (0.33, 0.30, 0.63, 0.98), (0.62, 0.30, 0.92, 0.98)),
}


@dataclass(frozen=True)
class TrackerContext:
    rule: RankedRule = "rule_unknown"
    match_started: bool = True
    rule_confidence: float = 1.0
    rule_locked: bool = True
    time_left_sec: int | None = None
    ally_side: SideName = "right"
    squid_lamps: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "TrackerContext":
        if data is None:
            return cls()
        match_data = data.get("match")
        if isinstance(match_data, Mapping):
            rule_value = str(match_data.get("rule", data.get("rule", "rule_unknown")))
            rule_confidence = float(match_data.get("rule_confidence", data.get("rule_confidence", 1.0)) or 0.0)
            rule_locked = bool(match_data.get("rule_locked", data.get("rule_locked", True)))
        else:
            rule_value = str(data.get("rule", "rule_unknown"))
            rule_confidence = float(data.get("rule_confidence", 1.0) or 0.0)
            rule_locked = bool(data.get("rule_locked", True))

        if rule_value not in {"splat_zones", "tower_control", "rainmaker", "clam_blitz"}:
            rule: RankedRule = "rule_unknown"
        else:
            rule = rule_value  # type: ignore[assignment]

        ally_side_value = str(data.get("ally_side", "right"))
        ally_side: SideName = "left" if ally_side_value == "left" else "right"
        time_left = data.get("time_left_sec")
        time_left_sec = int(time_left) if isinstance(time_left, (int, float)) else None
        squid_lamps = data.get("squid_lamps")
        return cls(
            rule=rule,
            match_started=bool(data.get("match_started", True)),
            rule_confidence=rule_confidence,
            rule_locked=rule_locked,
            time_left_sec=time_left_sec,
            ally_side=ally_side,
            squid_lamps=dict(squid_lamps) if isinstance(squid_lamps, Mapping) else {},
        )


@dataclass
class DigitBlob:
    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class NumberReading:
    value: int | None
    confidence: float
    text: str = ""
    raw_scores: list[float] = field(default_factory=list)
    rect: tuple[int, int, int, int] | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "text": self.text,
            "raw_scores": [round(float(score), 4) for score in self.raw_scores],
            "rect": list(self.rect) if self.rect is not None else None,
            "source": self.source,
        }


@dataclass
class CounterConfidence:
    ourCount: float = 0.0
    enemyCount: float = 0.0
    ourPenalty: float = 0.0
    enemyPenalty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "ourCount": round(float(self.ourCount), 4),
            "enemyCount": round(float(self.enemyCount), 4),
            "ourPenalty": round(float(self.ourPenalty), 4),
            "enemyPenalty": round(float(self.enemyPenalty), 4),
        }


@dataclass
class CounterState:
    ourCount: int | None = None
    enemyCount: int | None = None
    ourPenalty: int | None = None
    enemyPenalty: int | None = None
    leader: Leader = "unknown"
    isOvertime: bool | None = None
    isKnockout: bool | None = None
    progressDirection: ProgressDirection = "unknown"
    recentDelta: dict[str, int | None] = field(
        default_factory=lambda: {"ourCountDelta5s": None, "enemyCountDelta5s": None}
    )
    confidence: CounterConfidence = field(default_factory=CounterConfidence)
    freshness: Freshness = "unknown"
    lastSeenMsAgo: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ourCount": self.ourCount,
            "enemyCount": self.enemyCount,
            "ourPenalty": self.ourPenalty,
            "enemyPenalty": self.enemyPenalty,
            "leader": self.leader,
            "isOvertime": self.isOvertime,
            "isKnockout": self.isKnockout,
            "progressDirection": self.progressDirection,
            "recentDelta": dict(self.recentDelta),
            "confidence": self.confidence.to_dict(),
            "freshness": self.freshness,
            "lastSeenMsAgo": self.lastSeenMsAgo,
        }


@dataclass
class MarkerReading:
    visible: bool
    x: float | None
    y: float | None
    distanceClass: DistanceClass
    confidence: float
    rect: tuple[int, int, int, int] | None = None
    kind: str = "objective_marker"

    @classmethod
    def offscreen(cls) -> "MarkerReading":
        return cls(False, None, None, "offscreen", 0.0)

    def screen_position_dict(self) -> dict[str, object]:
        return {
            "visible": bool(self.visible),
            "x": None if self.x is None else round(float(self.x), 4),
            "y": None if self.y is None else round(float(self.y), 4),
            "distanceClass": self.distanceClass,
        }

    def debug_dict(self) -> dict[str, object]:
        return {
            "visible": bool(self.visible),
            "x": self.x,
            "y": self.y,
            "distanceClass": self.distanceClass,
            "confidence": round(float(self.confidence), 4),
            "rect": list(self.rect) if self.rect is not None else None,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ClamGoalStateHint:
    activeScoreSide: SideName
    goalState: dict[str, str]
    confidence: float
    metrics: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "activeScoreSide": self.activeScoreSide,
            "goalState": dict(self.goalState),
            "confidence": round(float(self.confidence), 4),
            "metrics": _round_nested(self.metrics),
        }


@dataclass
class RankedObjectiveReading:
    timestamp: float
    frame_index: int
    timestampMs: int
    rule: RankedRule
    counter: CounterState
    objective: dict[str, object]
    quality: dict[str, object]
    raw: dict[str, object] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "timestampMs": int(self.timestampMs),
            "rule": self.rule,
            "counter": self.counter.to_dict(),
            "objective": _round_nested(self.objective),
            "quality": _round_nested(self.quality),
        }
        if include_debug:
            data["raw"] = _round_nested(self.raw)
            data["timestamp"] = round(float(self.timestamp), 4)
            data["frame_index"] = int(self.frame_index)
        return data


@dataclass
class RankedObjectiveUpdate:
    reading: RankedObjectiveReading
    events: list[dict[str, object]]

    def to_dict(self, include_debug: bool = False) -> dict[str, object]:
        data = self.reading.to_dict(include_debug=include_debug)
        data["events"] = [_round_nested(event) for event in self.events]
        return data

    def to_row(self) -> dict[str, object]:
        counter = self.reading.counter
        objective = self.reading.objective
        screen_position = objective.get("screenPosition")
        if not isinstance(screen_position, Mapping):
            screen_position = objective.get("position") if isinstance(objective.get("position"), Mapping) else {}
        marker_visible = screen_position.get("visible") if isinstance(screen_position, Mapping) else ""
        marker_x = screen_position.get("x") if isinstance(screen_position, Mapping) else ""
        marker_y = screen_position.get("y") if isinstance(screen_position, Mapping) else ""
        return {
            "timestamp": round(float(self.reading.timestamp), 4),
            "timestamp_ms": int(self.reading.timestampMs),
            "frame_index": int(self.reading.frame_index),
            "rule": self.reading.rule,
            "our_count": counter.ourCount,
            "enemy_count": counter.enemyCount,
            "our_penalty": counter.ourPenalty,
            "enemy_penalty": counter.enemyPenalty,
            "leader": counter.leader,
            "progress_direction": counter.progressDirection,
            "our_delta_5s": counter.recentDelta.get("ourCountDelta5s"),
            "enemy_delta_5s": counter.recentDelta.get("enemyCountDelta5s"),
            "counter_freshness": counter.freshness,
            "objective_kind": objective.get("kind", ""),
            "objective_state": _objective_state_text(objective),
            "marker_visible": marker_visible,
            "marker_x": marker_x,
            "marker_y": marker_y,
            "confidence": self.reading.quality.get("overallConfidence", 0.0),
            "events": ",".join(str(event.get("event", event.get("type", ""))) for event in self.events),
        }


class TemplateDigitReader:
    def __init__(self, templates_dir: Path | None = None, min_digit_score: float = 0.62) -> None:
        self.templates_dir = templates_dir or DEFAULT_TEMPLATE_DIR
        self.min_digit_score = float(min_digit_score)
        self.templates = self._load_templates(self.templates_dir)
        missing = [digit for digit in "0123456789" if digit not in self.templates]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(f"digit templates are missing ({joined}); run match_time_ocr/train_templates.py")

    def read_number(
        self,
        crop: np.ndarray,
        value_min: int = 0,
        value_max: int = 100,
        max_digits: int = 3,
    ) -> NumberReading:
        if crop.size == 0:
            return NumberReading(None, 0.0)
        best = NumberReading(None, 0.0)
        for _mask_name, mask in make_counter_text_masks(crop):
            reading = self._read_number_from_mask(mask, value_min=value_min, value_max=value_max, max_digits=max_digits)
            score = reading.confidence + 0.025 * min(3, len(reading.text))
            best_score = best.confidence + 0.025 * min(3, len(best.text))
            if reading.value is not None and score > best_score:
                best = reading
            slot_reading = self._read_number_from_slots(mask, value_min=value_min, value_max=value_max, max_digits=max_digits)
            slot_score = slot_reading.confidence + 0.025 * min(3, len(slot_reading.text))
            best_score = best.confidence + 0.025 * min(3, len(best.text))
            if slot_reading.value is not None and slot_score > best_score:
                best = slot_reading
            best_score = best.confidence + 0.025 * min(3, len(best.text))
            if best.value is not None and best_score >= 0.94:
                break
        return best

    def _read_number_from_mask(
        self,
        mask: np.ndarray,
        value_min: int,
        value_max: int,
        max_digits: int,
    ) -> NumberReading:
        blobs = extract_number_blobs(mask, max_digits=max_digits)
        if not blobs:
            return NumberReading(None, 0.0)

        classified: list[tuple[str, float, DigitBlob]] = []
        for blob in blobs:
            digit, score = self.classify_digit(normalize_glyph(mask, blob))
            if digit and score >= self.min_digit_score:
                classified.append((digit, score, blob))
        if not classified:
            return NumberReading(None, 0.0)

        # A plus sign in penalty regions can survive as a component. Keep the right-most digits
        # when too many glyphs are present.
        classified.sort(key=lambda item: item[2].x)
        classified = classified[-max_digits:]
        text = "".join(digit for digit, _score, _blob in classified)
        if not text:
            return NumberReading(None, 0.0)
        value = int(text)
        if text.startswith("0") and len(text) > 1:
            return NumberReading(None, 0.0, text=text, raw_scores=[score for _digit, score, _blob in classified])
        if value < value_min or value > value_max:
            return NumberReading(None, 0.0, text=text, raw_scores=[score for _digit, score, _blob in classified])

        scores = [score for _digit, score, _blob in classified]
        confidence = float(min(scores))
        if len(classified) == 1 and value >= 10:
            confidence *= 0.72
        if len(classified) > 1:
            gaps = [classified[idx + 1][2].x - (classified[idx][2].x + classified[idx][2].w) for idx in range(len(classified) - 1)]
            if any(gap > mask.shape[1] * 0.22 for gap in gaps):
                confidence *= 0.76
        first = min(blob.x for _digit, _score, blob in classified)
        top = min(blob.y for _digit, _score, blob in classified)
        right = max(blob.x + blob.w for _digit, _score, blob in classified)
        bottom = max(blob.y + blob.h for _digit, _score, blob in classified)
        return NumberReading(value, _clamp(confidence), text=text, raw_scores=scores, rect=(first, top, right, bottom))

    def _read_number_from_slots(
        self,
        mask: np.ndarray,
        value_min: int,
        value_max: int,
        max_digits: int,
    ) -> NumberReading:
        best = NumberReading(None, 0.0)
        for digit_count, slots in COUNT_SLOT_LAYOUTS.items():
            if digit_count > max_digits:
                continue
            glyphs = extract_slot_glyphs(mask, digit_count)
            if len(glyphs) != digit_count:
                continue
            classified = [self.classify_digit(glyph) for glyph in glyphs]
            text = "".join(digit for digit, _score in classified)
            scores = [score for _digit, score in classified]
            if not text or min(scores) < self.min_digit_score:
                continue
            value = int(text)
            if value < value_min or value > value_max:
                continue
            if text.startswith("0") and len(text) > 1:
                continue
            confidence = float(min(scores))
            rect = slot_layout_rect(mask.shape, digit_count)
            reading = NumberReading(value, _clamp(confidence), text=text, raw_scores=scores, rect=rect)
            score = reading.confidence + 0.025 * digit_count
            best_score = best.confidence + 0.025 * len(best.text)
            if score > best_score:
                best = reading
        return best

    def search_numbers(
        self,
        crop: np.ndarray,
        value_min: int = 0,
        value_max: int = 100,
        max_digits: int = 3,
    ) -> list[NumberReading]:
        readings: list[NumberReading] = []
        for _mask_name, mask in make_counter_text_masks(crop):
            blobs = extract_number_blobs(mask, max_digits=32)
            blobs = [blob for blob in blobs if blob.h >= max(18, int(mask.shape[0] * 0.18))]
            blobs.sort(key=lambda blob: blob.x)
            for start in range(len(blobs)):
                for length in range(1, max_digits + 1):
                    group = blobs[start : start + length]
                    if len(group) != length:
                        continue
                    reading = self._read_blob_group(mask, group, value_min, value_max)
                    if reading.value is not None:
                        readings.append(reading)
        return dedupe_number_readings(readings)

    def _read_blob_group(
        self,
        mask: np.ndarray,
        group: Sequence[DigitBlob],
        value_min: int,
        value_max: int,
    ) -> NumberReading:
        if not group:
            return NumberReading(None, 0.0)
        group = sorted(group, key=lambda blob: blob.x)
        heights = [blob.h for blob in group]
        centers_y = [blob.y + blob.h * 0.5 for blob in group]
        if max(heights) - min(heights) > max(14, mask.shape[0] * 0.18):
            return NumberReading(None, 0.0)
        if max(centers_y) - min(centers_y) > max(16, mask.shape[0] * 0.20):
            return NumberReading(None, 0.0)
        gaps = [group[idx + 1].x - (group[idx].x + group[idx].w) for idx in range(len(group) - 1)]
        if gaps and (min(gaps) < -2 or max(gaps) > mask.shape[1] * 0.11):
            return NumberReading(None, 0.0)
        classified: list[tuple[str, float]] = [self.classify_digit(normalize_glyph(mask, blob)) for blob in group]
        text = "".join(digit for digit, _score in classified)
        scores = [score for _digit, score in classified]
        if not text or min(scores) < self.min_digit_score:
            return NumberReading(None, 0.0)
        value = int(text)
        if value < value_min or value > value_max:
            return NumberReading(None, 0.0, text=text, raw_scores=scores)
        if len(text) > 1 and text.startswith("0"):
            return NumberReading(None, 0.0, text=text, raw_scores=scores)
        first = min(blob.x for blob in group)
        top = min(blob.y for blob in group)
        right = max(blob.x + blob.w for blob in group)
        bottom = max(blob.y + blob.h for blob in group)
        confidence = float(min(scores))
        if len(text) == 1 and value not in {0, 1}:
            confidence *= 0.92
        return NumberReading(value, _clamp(confidence), text=text, raw_scores=scores, rect=(first, top, right, bottom), source="search")

    def classify_digit(self, glyph: np.ndarray) -> tuple[str, float]:
        flat = glyph.reshape(-1).astype(np.float32)
        flat_norm = float(np.linalg.norm(flat))
        if flat_norm == 0.0:
            return "", 0.0
        best_digit = ""
        best_score = -1.0
        for digit, template in self.templates.items():
            other = template.reshape(-1).astype(np.float32)
            denom = flat_norm * float(np.linalg.norm(other))
            score = 0.0 if denom == 0.0 else float(np.dot(flat, other) / denom)
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score

    @staticmethod
    def _load_templates(templates_dir: Path) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for digit in "0123456789":
            path = templates_dir / f"{digit}.png"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if image.shape != DIGIT_TEMPLATE_SIZE:
                image = cv2.resize(image, (DIGIT_TEMPLATE_SIZE[1], DIGIT_TEMPLATE_SIZE[0]), interpolation=cv2.INTER_AREA)
            templates[digit] = image.astype(np.float32) / 255.0
        return templates


class MarkerDetector:
    def __init__(self, min_confidence: float = 0.88) -> None:
        self.min_confidence = float(min_confidence)

    def read_frame(self, frame: np.ndarray, rule: RankedRule) -> MarkerReading:
        if rule == "splat_zones" or frame.size == 0:
            return MarkerReading.offscreen()

        search, search_rect = crop_rel(frame, OBJECTIVE_SEARCH_ROI)
        if search.size == 0:
            return MarkerReading.offscreen()

        original_h, original_w = search.shape[:2]
        max_search_width = 480
        if original_w > max_search_width:
            scale = max_search_width / max(1, original_w)
            resized_h = max(1, int(round(original_h * scale)))
            processed = cv2.resize(search, (max_search_width, resized_h), interpolation=cv2.INTER_AREA)
            scale_x = original_w / max(1, processed.shape[1])
            scale_y = original_h / max(1, processed.shape[0])
        else:
            processed = search
            scale_x = 1.0
            scale_y = 1.0

        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        bright_white = (val >= 180) & (sat <= 95)
        bright_ui = (val >= 135) & (sat >= 65)
        yellow_or_orange = (val >= 125) & (sat >= 65) & (hue >= 10) & (hue <= 45)
        mask = np.where(bright_white | bright_ui | yellow_or_orange, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        crop_h, crop_w = processed.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        best: tuple[float, tuple[int, int, int, int], str] | None = None
        for idx in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[idx]]
            if x <= 2 or y <= 2 or x + w >= crop_w - 2 or y + h >= crop_h - 2:
                continue
            if area < max(45, int(crop_w * crop_h * 0.000045)):
                continue
            if area > crop_w * crop_h * 0.030:
                continue
            if w < 7 or h < 7:
                continue
            if w > crop_w * 0.22 or h > crop_h * 0.25:
                continue
            aspect = w / max(1, h)
            if not 0.50 <= aspect <= 2.6:
                continue

            pad = max(3, int(round(max(w, h) * 0.22)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(crop_w, x + w + pad)
            y2 = min(crop_h, y + h + pad)
            local_hsv = hsv[y1:y2, x1:x2]
            local_mask = mask[y1:y2, x1:x2] > 0
            if local_hsv.size == 0:
                continue
            local_val = local_hsv[:, :, 2]
            dark_outline = (local_val < 70) & (~local_mask)
            dark_fraction = float(dark_outline.mean())
            fill_fraction = area / max(1, (x2 - x1) * (y2 - y1))
            compactness = area / max(1, w * h)
            size_score = min(1.0, math.sqrt(area) / max(1.0, frame_h * 0.040))
            center_x = (x + w * 0.5) / max(1, crop_w)
            center_y = (y + h * 0.5) / max(1, crop_h)
            center_bias = 1.0 - min(1.0, math.hypot(center_x - 0.5, center_y - 0.48) / 0.72)
            score = (
                0.32 * min(1.0, fill_fraction / 0.35)
                + 0.24 * min(1.0, compactness / 0.72)
                + 0.20 * min(1.0, dark_fraction / 0.38)
                + 0.14 * size_score
                + 0.10 * center_bias
            )
            original_area = area * scale_x * scale_y
            if rule == "clam_blitz" and original_area < frame_h * frame_w * 0.00008:
                score *= 0.84
            if best is None or score > best[0]:
                best = (float(score), (x, y, w, h), "visible_power_clam" if rule == "clam_blitz" else "objective_marker")

        if best is None or best[0] < self.min_confidence:
            return MarkerReading.offscreen()

        score, (x, y, w, h), marker_kind = best
        sx1, sy1, _sx2, _sy2 = search_rect
        local_left = int(round(x * scale_x))
        local_top = int(round(y * scale_y))
        local_right = int(round((x + w) * scale_x))
        local_bottom = int(round((y + h) * scale_y))
        left = sx1 + local_left
        top = sy1 + local_top
        right = sx1 + local_right
        bottom = sy1 + local_bottom
        center_x = (left + right) * 0.5 / max(1, frame_w)
        center_y = (top + bottom) * 0.5 / max(1, frame_h)
        marker_height = (bottom - top) / max(1, frame_h)
        if marker_height >= 0.095:
            distance_class: DistanceClass = "near"
        elif marker_height >= 0.040:
            distance_class = "mid"
        else:
            distance_class = "far"
        return MarkerReading(
            visible=True,
            x=float(center_x),
            y=float(center_y),
            distanceClass=distance_class,
            confidence=_clamp(score),
            rect=(left, top, right, bottom),
            kind=marker_kind,
        )


class RankedObjectiveDetector:
    def __init__(
        self,
        templates_dir: Path | None = None,
        min_digit_score: float = 0.62,
        use_timer_ocr: bool = True,
        marker_min_confidence: float = 0.88,
    ) -> None:
        self.digit_reader = TemplateDigitReader(templates_dir=templates_dir, min_digit_score=min_digit_score)
        self.marker_detector = MarkerDetector(marker_min_confidence)
        self.timer_ocr = _load_timer_ocr() if use_timer_ocr else None
        self.uses_timer_templates_fallback = (templates_dir or DEFAULT_TEMPLATE_DIR).resolve() == DEFAULT_TEMPLATE_DIR.resolve()

    def read_frame(
        self,
        frame: np.ndarray,
        context: TrackerContext | Mapping[str, object] | None = None,
        timestamp: float = 0.0,
        frame_index: int = -1,
    ) -> RankedObjectiveReading:
        ctx = context if isinstance(context, TrackerContext) else TrackerContext.from_mapping(context)
        timestamp_ms = int(round(timestamp * 1000.0))
        warnings: list[str] = []
        sources: list[str] = ["hud_ocr", "template"]
        if self.uses_timer_templates_fallback:
            warnings.append("using_match_time_digit_templates_for_counter")

        if not ctx.match_started:
            counter = CounterState(isOvertime=False, isKnockout=False, freshness="unknown")
            objective = build_unknown_objective(ctx.rule)
            return RankedObjectiveReading(
                timestamp=timestamp,
                frame_index=frame_index,
                timestampMs=timestamp_ms,
                rule=ctx.rule,
                counter=counter,
                objective=objective,
                quality={"overallConfidence": 0.0, "source": [], "warnings": ["match_not_started"]},
            )

        left_count, left_penalty, left_raw = self._read_side(frame, "left", ctx.rule)
        right_count, right_penalty, right_raw = self._read_side(frame, "right", ctx.rule)
        left_count, right_count = suppress_lone_count_noise(left_count, right_count)
        left_raw["count"] = left_count.to_dict()
        right_raw["count"] = right_count.to_dict()
        our_side = ctx.ally_side
        enemy_side: SideName = "left" if our_side == "right" else "right"
        count_by_side = {"left": left_count, "right": right_count}
        penalty_by_side = {"left": left_penalty, "right": right_penalty}
        our_count = count_by_side[our_side]
        enemy_count = count_by_side[enemy_side]
        our_penalty = penalty_by_side[our_side]
        enemy_penalty = penalty_by_side[enemy_side]

        timer_reading = None
        if self.timer_ocr is not None:
            try:
                timer_reading = self.timer_ocr.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
                if timer_reading is not None:
                    sources.append("timer_ocr")
            except Exception as exc:
                warnings.append(f"timer_ocr_unavailable:{exc.__class__.__name__}")
                self.timer_ocr = None

        is_overtime: bool | None = None
        if timer_reading is not None:
            is_overtime = bool(getattr(timer_reading, "kind", "") == "overtime")
        elif ctx.time_left_sec is not None:
            is_overtime = ctx.time_left_sec <= 0

        counter = CounterState(
            ourCount=our_count.value,
            enemyCount=enemy_count.value,
            ourPenalty=our_penalty.value,
            enemyPenalty=enemy_penalty.value,
            leader=compute_leader(our_count.value, enemy_count.value),
            isOvertime=is_overtime,
            isKnockout=_compute_knockout(our_count.value, enemy_count.value),
            progressDirection="unknown",
            recentDelta={"ourCountDelta5s": None, "enemyCountDelta5s": None},
            confidence=CounterConfidence(
                ourCount=our_count.confidence,
                enemyCount=enemy_count.confidence,
                ourPenalty=our_penalty.confidence,
                enemyPenalty=enemy_penalty.confidence,
            ),
            freshness="fresh" if our_count.value is not None or enemy_count.value is not None else "unknown",
            lastSeenMsAgo=0 if our_count.value is not None or enemy_count.value is not None else None,
        )
        clam_goal_hint: ClamGoalStateHint | None = None
        if ctx.rule == "clam_blitz":
            clam_goal_hint = detect_clam_goal_state_hint(frame, ctx.ally_side, {"left": left_count, "right": right_count})

        marker = self.marker_detector.read_frame(frame, ctx.rule)
        if marker.visible:
            sources.append("screen_marker")
        elif ctx.rule in {"tower_control", "rainmaker", "clam_blitz"}:
            warnings.append("objective_marker_not_visible")

        objective = build_objective(ctx.rule, counter, marker, freshness=counter.freshness, last_seen_ms_ago=0)
        if clam_goal_hint is not None:
            objective = apply_clam_goal_state_hint(objective, clam_goal_hint)
        confidence_values = [
            counter.confidence.ourCount,
            counter.confidence.enemyCount,
            marker.confidence if marker.visible else 0.0,
        ]
        count_confidence = _mean([value for value in confidence_values[:2] if value > 0.0])
        marker_bonus = 0.12 * marker.confidence if marker.visible else 0.0
        overall = _clamp(0.88 * count_confidence + marker_bonus)
        if ctx.rule == "rule_unknown":
            warnings.append("rule_unknown")
            overall *= 0.78

        return RankedObjectiveReading(
            timestamp=timestamp,
            frame_index=frame_index,
            timestampMs=timestamp_ms,
            rule=ctx.rule,
            counter=counter,
            objective=objective,
            quality={"overallConfidence": overall, "source": sources, "warnings": warnings},
            raw={
                "left": left_raw,
                "right": right_raw,
                "marker": marker.debug_dict(),
                "clam_goal_hint": None if clam_goal_hint is None else clam_goal_hint.to_dict(),
                "timer": _timer_debug(timer_reading),
            },
        )

    def _read_side(self, frame: np.ndarray, side: SideName, rule: RankedRule) -> tuple[NumberReading, NumberReading, dict[str, object]]:
        count_reading, count_candidates = self._read_best_count_candidate(frame, side, rule)
        penalty_crop, penalty_rect = crop_rel(frame, SIDE_PENALTY_ROIS[side])
        penalty_reading = self.digit_reader.read_number(penalty_crop, value_min=0, value_max=99, max_digits=2)
        penalty_reading.rect = _offset_rect(penalty_reading.rect, penalty_rect)
        if penalty_reading.value is None:
            penalty_reading.confidence = 0.0
        return count_reading, penalty_reading, {
            "count": count_reading.to_dict(),
            "count_candidates": count_candidates,
            "penalty": penalty_reading.to_dict(),
        }

    def _read_best_count_candidate(self, frame: np.ndarray, side: SideName, rule: RankedRule) -> tuple[NumberReading, list[dict[str, object]]]:
        best: NumberReading | None = None
        candidates: list[dict[str, object]] = []

        primary_candidates: list[tuple[str, tuple[float, float, float, float]]] = []
        fallback_candidates: list[tuple[str, tuple[float, float, float, float]]] = []
        for name, roi in SIDE_COUNT_ROI_CANDIDATES[side]:
            if rule == "tower_control":
                target = primary_candidates if name == "tower" else fallback_candidates
            else:
                target = fallback_candidates if name == "tower" else primary_candidates
            target.append((name, roi))

        for candidate_group in (primary_candidates, fallback_candidates):
            group_best: NumberReading | None = None
            group_best_score = -1.0
            for name, roi in candidate_group:
                crop, rect = crop_rel(frame, roi)
                reading = self.digit_reader.read_number(crop, value_min=0, value_max=100, max_digits=3)
                reading.rect = _offset_rect(reading.rect, rect)
                reading.source = f"roi:{name}"
                score = count_candidate_score(reading)
                candidates.append({"name": name, "roi": list(roi), "reading": reading.to_dict(), "score": round(float(score), 4)})
                if reading.value is not None and score > group_best_score:
                    group_best = reading
                    group_best_score = score
                if name == "standard_plate" and reading.value is not None and score >= 0.95 and reading.confidence >= 0.88:
                    group_best = reading
                    group_best_score = score
                    break
            if group_best is not None:
                best = group_best
                break

        if best is None:
            search_reading = self._read_search_count_candidate(frame, side, rule)
            if search_reading.value is not None:
                score = count_candidate_score(search_reading)
                candidates.append({"name": "search", "roi": None, "reading": search_reading.to_dict(), "score": round(float(score), 4)})
                best = search_reading
        if best is None:
            return NumberReading(None, 0.0), candidates
        return best, candidates

    def _read_search_count_candidate(self, frame: np.ndarray, side: SideName, rule: RankedRule) -> NumberReading:
        mode = "tower" if rule == "tower_control" else "standard"
        search_crop, search_rect = crop_rel(frame, COUNT_SEARCH_ROIS[mode])
        readings = self.digit_reader.search_numbers(search_crop, value_min=0, value_max=100, max_digits=3)
        if not readings:
            return NumberReading(None, 0.0, source="search")
        sx1, sy1, sx2, sy2 = search_rect
        midpoint = (sx2 - sx1) * 0.5
        side_readings: list[tuple[float, NumberReading]] = []
        for reading in readings:
            if reading.rect is None:
                continue
            x1, y1, x2, y2 = reading.rect
            cx = (x1 + x2) * 0.5
            if side == "left" and cx >= midpoint:
                continue
            if side == "right" and cx < midpoint:
                continue
            reading = replace(reading, rect=_offset_rect(reading.rect, search_rect), source=f"search:{mode}")
            side_bias = 1.0 - min(1.0, abs((cx / max(1, sx2 - sx1)) - (0.32 if side == "left" else 0.68)) / 0.40)
            score = count_candidate_score(reading) + 0.08 * side_bias
            side_readings.append((score, reading))
        if not side_readings:
            return NumberReading(None, 0.0, source=f"search:{mode}")
        side_readings.sort(key=lambda item: item[0], reverse=True)
        return side_readings[0][1]


class RankedObjectiveTracker:
    def __init__(
        self,
        detector: RankedObjectiveDetector,
        window_size: int = 7,
        history_ms: int = 16000,
        fresh_ms: int = 1000,
        stale_ms: int = 3000,
        min_count_confidence: float = 0.62,
    ) -> None:
        self.detector = detector
        self.window_size = max(1, int(window_size))
        self.history_ms = max(5000, int(history_ms))
        self.fresh_ms = max(250, int(fresh_ms))
        self.stale_ms = max(self.fresh_ms, int(stale_ms))
        self.min_count_confidence = float(min_count_confidence)
        self.raw_history: deque[RankedObjectiveReading] = deque(maxlen=self.window_size)
        self.stable_history: deque[RankedObjectiveReading] = deque()
        self.last_stable_counter: CounterState | None = None
        self.last_objective_state: dict[str, object] | None = None
        self.last_count_seen_ms: dict[str, int] = {"ourCount": -1, "enemyCount": -1}
        self.last_objective_seen_ms = -1
        self.last_leader: Leader = "unknown"
        self.overtime_emitted = False
        self.knockout_emitted = False
        self.counter_unreadable_since_ms: int | None = None

    def read_frame(
        self,
        frame: np.ndarray,
        context: TrackerContext | Mapping[str, object] | None = None,
        timestamp: float = 0.0,
        frame_index: int = -1,
    ) -> RankedObjectiveUpdate:
        raw = self.detector.read_frame(frame, context=context, timestamp=timestamp, frame_index=frame_index)
        return self.update(raw)

    def update(self, raw: RankedObjectiveReading) -> RankedObjectiveUpdate:
        self.raw_history.append(raw)
        now_ms = raw.timestampMs
        stable_counter = self._smooth_counter(raw)
        marker = _marker_from_objective(raw.objective)
        stable_objective = build_objective(
            raw.rule,
            stable_counter,
            marker,
            freshness=stable_counter.freshness,
            last_seen_ms_ago=stable_counter.lastSeenMsAgo,
        )
        stable_objective = merge_static_clam_goal_state(stable_objective, raw.objective)
        stable_objective = self._smooth_objective(raw, stable_objective)

        warnings = list(raw.quality.get("warnings", [])) if isinstance(raw.quality.get("warnings", []), list) else []
        if stable_counter.freshness != "fresh":
            warnings.append(f"counter_{stable_counter.freshness}")
        sources = list(raw.quality.get("source", [])) if isinstance(raw.quality.get("source", []), list) else []
        if "temporal_smoothing" not in sources:
            sources.append("temporal_smoothing")
        overall = self._overall_confidence(raw, stable_counter, stable_objective)
        reading = replace(
            raw,
            counter=stable_counter,
            objective=stable_objective,
            quality={"overallConfidence": overall, "source": sources, "warnings": sorted(set(str(item) for item in warnings))},
        )
        self._append_stable(reading)
        events = self._events(reading)
        return RankedObjectiveUpdate(reading, events)

    def reset(self) -> None:
        self.raw_history.clear()
        self.stable_history.clear()
        self.last_stable_counter = None
        self.last_objective_state = None
        self.last_count_seen_ms = {"ourCount": -1, "enemyCount": -1}
        self.last_objective_seen_ms = -1
        self.last_leader = "unknown"
        self.overtime_emitted = False
        self.knockout_emitted = False
        self.counter_unreadable_since_ms = None

    def _smooth_counter(self, raw: RankedObjectiveReading) -> CounterState:
        previous = self.last_stable_counter
        now_ms = raw.timestampMs
        our_count, our_conf = self._stable_count("ourCount", previous.ourCount if previous else None, now_ms)
        enemy_count, enemy_conf = self._stable_count("enemyCount", previous.enemyCount if previous else None, now_ms)
        our_penalty, our_penalty_conf = self._stable_penalty("ourPenalty")
        enemy_penalty, enemy_penalty_conf = self._stable_penalty("enemyPenalty")

        seen_times = [value for value in self.last_count_seen_ms.values() if value >= 0]
        last_seen_ms = max(seen_times) if seen_times else -1
        if last_seen_ms < 0:
            freshness: Freshness = "unknown"
            last_seen_ago = None
        else:
            last_seen_ago = max(0, now_ms - last_seen_ms)
            if last_seen_ago <= self.fresh_ms:
                freshness = "fresh"
            elif last_seen_ago <= self.stale_ms:
                freshness = "stale"
            else:
                freshness = "unknown"
                our_count = None
                enemy_count = None

        recent_delta = self._recent_delta(now_ms, our_count, enemy_count)
        progress_direction = compute_progress_direction(recent_delta.get("ourCountDelta5s"), recent_delta.get("enemyCountDelta5s"))
        counter = CounterState(
            ourCount=our_count,
            enemyCount=enemy_count,
            ourPenalty=our_penalty,
            enemyPenalty=enemy_penalty,
            leader=compute_leader(our_count, enemy_count),
            isOvertime=raw.counter.isOvertime if raw.counter.isOvertime is not None else (previous.isOvertime if previous else None),
            isKnockout=_compute_knockout(our_count, enemy_count),
            progressDirection=progress_direction,
            recentDelta=recent_delta,
            confidence=CounterConfidence(
                ourCount=our_conf,
                enemyCount=enemy_conf,
                ourPenalty=our_penalty_conf,
                enemyPenalty=enemy_penalty_conf,
            ),
            freshness=freshness,
            lastSeenMsAgo=last_seen_ago,
        )
        self.last_stable_counter = counter
        return counter

    def _stable_count(self, field_name: Literal["ourCount", "enemyCount"], previous_value: int | None, now_ms: int) -> tuple[int | None, float]:
        conf_name = field_name
        previous_seen_ms = self.last_count_seen_ms.get(field_name, -1)
        candidates: list[tuple[int, float, int]] = []
        for reading in reversed(self.raw_history):
            value = getattr(reading.counter, field_name)
            confidence = getattr(reading.counter.confidence, conf_name)
            if value is None or confidence < self.min_count_confidence:
                continue
            elapsed_ms = max(0, reading.timestampMs - previous_seen_ms) if previous_seen_ms >= 0 else now_ms - reading.timestampMs
            if previous_value is not None and not _plausible_count_transition(previous_value, int(value), elapsed_ms):
                continue
            candidates.append((int(value), float(confidence), reading.timestampMs))

        if not candidates:
            if previous_value is None:
                return None, 0.0
            return previous_value, _decay_confidence(getattr(self.last_stable_counter.confidence, field_name) if self.last_stable_counter else 0.0)

        latest_value, latest_conf, latest_ms = candidates[0]
        same_votes = [conf for value, conf, _ms in candidates if value == latest_value]
        accept = latest_conf >= 0.82 or len(same_votes) >= 2 or previous_value is None
        if accept:
            self.last_count_seen_ms[field_name] = latest_ms
            return latest_value, _clamp(max(latest_conf, _mean(same_votes)))

        if previous_value is not None:
            return previous_value, _decay_confidence(getattr(self.last_stable_counter.confidence, field_name) if self.last_stable_counter else 0.0)
        return latest_value, _clamp(latest_conf * 0.82)

    def _stable_penalty(self, field_name: Literal["ourPenalty", "enemyPenalty"]) -> tuple[int | None, float]:
        values: list[tuple[int, float]] = []
        for reading in reversed(self.raw_history):
            value = getattr(reading.counter, field_name)
            confidence = getattr(reading.counter.confidence, field_name)
            if value is not None and confidence >= 0.62:
                values.append((int(value), float(confidence)))
        if not values:
            return None, 0.0
        latest_value, latest_conf = values[0]
        same = [confidence for value, confidence in values if value == latest_value]
        if latest_conf >= 0.78 or len(same) >= 2:
            return latest_value, _clamp(max(latest_conf, _mean(same)))
        return None, 0.0

    def _recent_delta(self, now_ms: int, our_count: int | None, enemy_count: int | None) -> dict[str, int | None]:
        def delta_for(field_name: Literal["ourCount", "enemyCount"], current: int | None, window_ms: int) -> int | None:
            if current is None:
                return None
            target = now_ms - window_ms
            older: RankedObjectiveReading | None = None
            for reading in self.stable_history:
                if reading.timestampMs <= target:
                    older = reading
                else:
                    break
            if older is None:
                return None
            previous = getattr(older.counter, field_name)
            if previous is None:
                return None
            return int(current) - int(previous)

        return {
            "ourCountDelta5s": delta_for("ourCount", our_count, 5000),
            "enemyCountDelta5s": delta_for("enemyCount", enemy_count, 5000),
        }

    def _smooth_objective(self, raw: RankedObjectiveReading, objective: dict[str, object]) -> dict[str, object]:
        marker = _marker_from_objective(objective)
        if marker.visible and marker.confidence >= 0.58:
            self.last_objective_seen_ms = raw.timestampMs
            self.last_objective_state = objective
            return objective

        if self.last_objective_state is None:
            return objective

        age_ms = raw.timestampMs - self.last_objective_seen_ms
        if 0 <= age_ms <= self.stale_ms and raw.rule in {"tower_control", "rainmaker"}:
            merged = dict(objective)
            merged["freshness"] = "stale"
            merged["lastSeenMsAgo"] = int(age_ms)
            previous_position = self.last_objective_state.get("screenPosition")
            if isinstance(previous_position, Mapping):
                merged["screenPosition"] = {
                    "visible": False,
                    "x": None,
                    "y": None,
                    "distanceClass": "offscreen",
                }
            return merged
        return objective

    def _append_stable(self, reading: RankedObjectiveReading) -> None:
        self.stable_history.append(reading)
        cutoff = reading.timestampMs - self.history_ms
        while self.stable_history and self.stable_history[0].timestampMs < cutoff:
            self.stable_history.popleft()

    def _overall_confidence(
        self,
        raw: RankedObjectiveReading,
        counter: CounterState,
        objective: dict[str, object],
    ) -> float:
        count_conf = _mean([counter.confidence.ourCount, counter.confidence.enemyCount])
        objective_conf = float(objective.get("confidence", 0.0) or 0.0)
        freshness_weight = 1.0 if counter.freshness == "fresh" else 0.72 if counter.freshness == "stale" else 0.35
        return _clamp((0.78 * count_conf + 0.22 * objective_conf) * freshness_weight)

    def _events(self, reading: RankedObjectiveReading) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        counter = reading.counter
        confidence = float(reading.quality.get("overallConfidence", 0.0) or 0.0)

        if counter.freshness == "unknown":
            if self.counter_unreadable_since_ms is None:
                self.counter_unreadable_since_ms = reading.timestampMs
            if reading.timestampMs - self.counter_unreadable_since_ms >= 1000:
                events.append({"event": "counter_unreadable", "severity": 2, "confidence": round(max(0.25, 1.0 - confidence), 4)})
        else:
            self.counter_unreadable_since_ms = None

        if counter.leader in {"our", "enemy"} and counter.leader != self.last_leader and self.last_leader not in {"unknown", "tie"}:
            event = "our_lead_taken" if counter.leader == "our" else "enemy_lead_taken"
            events.append(
                {
                    "event": event,
                    "severity": 4,
                    "basis": {"ourCount": counter.ourCount, "enemyCount": counter.enemyCount},
                    "confidence": round(confidence, 4),
                }
            )
        if counter.leader != "unknown":
            self.last_leader = counter.leader

        delta10 = self._delta_10s(reading.timestampMs)
        our_delta_10 = delta10.get("ourCountDelta10s")
        enemy_delta_10 = delta10.get("enemyCountDelta10s")
        if our_delta_10 is not None and our_delta_10 <= -10 and confidence >= 0.75:
            events.append(
                {
                    "event": "our_big_push",
                    "severity": 4,
                    "basis": {"ourCountDelta10s": our_delta_10, "ourCount": counter.ourCount, "enemyCount": counter.enemyCount},
                    "confidence": round(confidence, 4),
                }
            )
        if enemy_delta_10 is not None and enemy_delta_10 <= -10 and confidence >= 0.75:
            events.append(
                {
                    "event": "enemy_big_push",
                    "severity": 4,
                    "basis": {"enemyCountDelta10s": enemy_delta_10, "enemyCount": counter.enemyCount, "ourCount": counter.ourCount},
                    "confidence": round(confidence, 4),
                }
            )

        near_goal = _near_goal_event(reading.objective, counter)
        if near_goal is not None and confidence >= 0.70:
            events.append({"event": near_goal, "severity": 5, "confidence": round(confidence, 4)})

        if self._is_stalled(reading):
            events.append({"event": "objective_stalled", "severity": 2, "confidence": round(confidence, 4)})

        if counter.isOvertime is True and not self.overtime_emitted:
            events.append({"event": "overtime_started", "severity": 4, "confidence": round(max(0.65, confidence), 4)})
            self.overtime_emitted = True
        if counter.isKnockout is True and not self.knockout_emitted:
            events.append({"event": "knockout", "severity": 5, "confidence": round(confidence, 4)})
            self.knockout_emitted = True

        return events

    def _delta_10s(self, now_ms: int) -> dict[str, int | None]:
        current = self.last_stable_counter
        if current is None:
            return {"ourCountDelta10s": None, "enemyCountDelta10s": None}
        target = now_ms - 10000
        older: RankedObjectiveReading | None = None
        for reading in self.stable_history:
            if reading.timestampMs <= target:
                older = reading
            else:
                break
        if older is None:
            return {"ourCountDelta10s": None, "enemyCountDelta10s": None}

        def make_delta(field_name: Literal["ourCount", "enemyCount"]) -> int | None:
            current_value = getattr(current, field_name)
            previous_value = getattr(older.counter, field_name)
            if current_value is None or previous_value is None:
                return None
            return int(current_value) - int(previous_value)

        return {"ourCountDelta10s": make_delta("ourCount"), "enemyCountDelta10s": make_delta("enemyCount")}

    def _is_stalled(self, reading: RankedObjectiveReading) -> bool:
        if len(self.stable_history) < 2:
            return False
        target = reading.timestampMs - 15000
        older: RankedObjectiveReading | None = None
        for item in self.stable_history:
            if item.timestampMs <= target:
                older = item
            else:
                break
        if older is None:
            return False
        if reading.counter.ourCount != older.counter.ourCount or reading.counter.enemyCount != older.counter.enemyCount:
            return False
        objective = reading.objective
        state_text = _objective_state_text(objective)
        return state_text not in {"unknown", "neutral", "free"}


def build_objective(
    rule: RankedRule,
    counter: CounterState,
    marker: MarkerReading,
    freshness: Freshness = "fresh",
    last_seen_ms_ago: int | None = None,
) -> dict[str, object]:
    if rule == "splat_zones":
        return build_splat_zones_objective(counter, freshness, last_seen_ms_ago)
    if rule == "tower_control":
        return build_tower_objective(counter, marker, freshness, last_seen_ms_ago)
    if rule == "rainmaker":
        return build_rainmaker_objective(counter, marker, freshness, last_seen_ms_ago)
    if rule == "clam_blitz":
        return build_clam_objective(counter, marker, freshness, last_seen_ms_ago)
    return build_unknown_objective(rule)


def build_unknown_objective(rule: RankedRule) -> dict[str, object]:
    return {"kind": rule, "confidence": 0.0, "freshness": "unknown", "lastSeenMsAgo": None}


def build_splat_zones_objective(counter: CounterState, freshness: Freshness, last_seen_ms_ago: int | None) -> dict[str, object]:
    active = active_scoring_side(counter)
    if active == "our":
        zone_control = "our_control"
        confidence = _clamp(0.35 + 0.65 * counter.confidence.ourCount)
    elif active == "enemy":
        zone_control = "enemy_control"
        confidence = _clamp(0.35 + 0.65 * counter.confidence.enemyCount)
    elif counter.freshness == "fresh":
        zone_control = "unknown"
        confidence = 0.35
    else:
        zone_control = "unknown"
        confidence = 0.0
    return {
        "kind": "splat_zones",
        "zoneControl": zone_control,
        "zoneCount": {"totalZones": None, "ourControlledZones": None, "enemyControlledZones": None},
        "position": None,
        "confidence": confidence,
        "freshness": freshness,
        "lastSeenMsAgo": last_seen_ms_ago,
    }


def build_tower_objective(counter: CounterState, marker: MarkerReading, freshness: Freshness, last_seen_ms_ago: int | None) -> dict[str, object]:
    active = active_scoring_side(counter)
    progress = progress_for_active_side(counter, active)
    return {
        "kind": "tower_control",
        "towerOwner": active if active in {"our", "enemy"} else ("neutral" if counter.freshness == "fresh" else "unknown"),
        "towerProgress": progress_dict(active, progress),
        "screenPosition": marker.screen_position_dict(),
        "confidence": objective_confidence(counter, marker, progress),
        "freshness": freshness,
        "lastSeenMsAgo": last_seen_ms_ago,
    }


def build_rainmaker_objective(counter: CounterState, marker: MarkerReading, freshness: Freshness, last_seen_ms_ago: int | None) -> dict[str, object]:
    active = active_scoring_side(counter)
    if active == "our":
        state = "our_carrier"
    elif active == "enemy":
        state = "enemy_carrier"
    elif counter.freshness == "fresh":
        state = "unknown"
    else:
        state = "unknown"
    progress = progress_for_active_side(counter, active)
    return {
        "kind": "rainmaker",
        "rainmakerState": state,
        "rainmakerProgress": {
            "side": progress_side(active, progress),
            "progressPercent": progress,
            "nearGoal": None if progress is None else progress >= 80,
        },
        "screenPosition": marker.screen_position_dict(),
        "confidence": objective_confidence(counter, marker, progress),
        "freshness": freshness,
        "lastSeenMsAgo": last_seen_ms_ago,
    }


def build_clam_objective(counter: CounterState, marker: MarkerReading, freshness: Freshness, last_seen_ms_ago: int | None) -> dict[str, object]:
    active = active_scoring_side(counter)
    our_goal = "open" if active == "enemy" else "closed" if counter.freshness == "fresh" else "unknown"
    enemy_goal = "open" if active == "our" else "closed" if counter.freshness == "fresh" else "unknown"
    visible_power_clam = None
    position: dict[str, object] | None = None
    if marker.visible:
        visible_power_clam = {
            "visible": True,
            "x": None if marker.x is None else round(float(marker.x), 4),
            "y": None if marker.y is None else round(float(marker.y), 4),
            "owner": "unknown",
        }
        position = {
            "type": marker.kind if marker.kind in {"visible_power_clam", "goal_marker"} else "visible_power_clam",
            "x": round(float(marker.x or 0.0), 4),
            "y": round(float(marker.y or 0.0), 4),
            "confidence": round(float(marker.confidence), 4),
        }
    return {
        "kind": "clam_blitz",
        "goalState": {"ourGoal": our_goal, "enemyGoal": enemy_goal},
        "clamState": {
            "playerClams": None,
            "playerHasPowerClam": None,
            "visiblePowerClam": visible_power_clam,
        },
        "position": position,
        "confidence": objective_confidence(counter, marker, None),
        "freshness": freshness,
        "lastSeenMsAgo": last_seen_ms_ago,
    }


def active_scoring_side(counter: CounterState) -> Literal["our", "enemy", "unknown"]:
    if counter.progressDirection == "our_count_decreasing":
        return "our"
    if counter.progressDirection == "enemy_count_decreasing":
        return "enemy"
    return "unknown"


def progress_for_active_side(counter: CounterState, active: str) -> int | None:
    if active == "our" and counter.ourCount is not None:
        return int(max(0, min(100, 100 - counter.ourCount)))
    if active == "enemy" and counter.enemyCount is not None:
        return int(max(0, min(100, 100 - counter.enemyCount)))
    return None


def progress_dict(active: str, progress: int | None) -> dict[str, object]:
    checkpoint_index: int | None = None
    checkpoint_state = "unknown"
    if progress is not None:
        if progress >= 75:
            checkpoint_index = 3
        elif progress >= 50:
            checkpoint_index = 2
        elif progress >= 25:
            checkpoint_index = 1
        nearest = min((25, 50, 75), key=lambda value: abs(value - progress))
        if abs(nearest - progress) <= 4:
            checkpoint_state = "at_checkpoint"
        elif progress < nearest:
            checkpoint_state = "before_checkpoint"
        else:
            checkpoint_state = "after_checkpoint"
    return {
        "side": progress_side(active, progress),
        "progressPercent": progress,
        "checkpointIndex": checkpoint_index,
        "checkpointState": checkpoint_state,
    }


def progress_side(active: str, progress: int | None) -> Literal["our_side", "center", "enemy_side", "unknown"]:
    if progress is None or active not in {"our", "enemy"}:
        return "unknown"
    if progress < 30:
        return "center"
    if active == "our":
        return "enemy_side"
    return "our_side"


def objective_confidence(counter: CounterState, marker: MarkerReading, progress: int | None) -> float:
    count_conf = _mean([counter.confidence.ourCount, counter.confidence.enemyCount])
    progress_bonus = 0.12 if progress is not None else 0.0
    marker_bonus = 0.22 * marker.confidence if marker.visible else 0.0
    freshness_weight = 1.0 if counter.freshness == "fresh" else 0.70 if counter.freshness == "stale" else 0.35
    return _clamp((0.66 * count_conf + progress_bonus + marker_bonus) * freshness_weight)


def compute_leader(our_count: int | None, enemy_count: int | None) -> Leader:
    if our_count is None or enemy_count is None:
        return "unknown"
    if our_count < enemy_count:
        return "our"
    if enemy_count < our_count:
        return "enemy"
    return "tie"


def compute_progress_direction(our_delta_5s: int | None, enemy_delta_5s: int | None) -> ProgressDirection:
    our_delta = our_delta_5s if our_delta_5s is not None else 0
    enemy_delta = enemy_delta_5s if enemy_delta_5s is not None else 0
    if our_delta_5s is None and enemy_delta_5s is None:
        return "unknown"
    if our_delta <= -2 and our_delta < enemy_delta:
        return "our_count_decreasing"
    if enemy_delta <= -2 and enemy_delta < our_delta:
        return "enemy_count_decreasing"
    return "stable"


def suppress_lone_count_noise(left_count: NumberReading, right_count: NumberReading) -> tuple[NumberReading, NumberReading]:
    left_present = left_count.value is not None
    right_present = right_count.value is not None
    if left_present == right_present:
        return left_count, right_count

    present = left_count if left_present else right_count
    if len(present.text) == 1 and present.confidence < 0.90:
        suppressed = replace(present, value=None, confidence=0.0, text="")
        if left_present:
            return suppressed, right_count
        return left_count, suppressed
    return left_count, right_count


def detect_clam_goal_state_hint(
    frame: np.ndarray,
    ally_side: SideName,
    count_by_side: Mapping[SideName, NumberReading],
) -> ClamGoalStateHint | None:
    scores: dict[SideName, tuple[float, dict[str, float]]] = {}
    for side in ("left", "right"):
        reading = count_by_side.get(side)
        if reading is None or reading.value is None or reading.rect is None:
            continue
        scores[side] = clam_score_plate_highlight_score(frame, reading.rect)

    if not scores:
        return None

    ordered = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
    active_side = ordered[0][0]
    active_score, active_metrics = ordered[0][1]
    second_score = ordered[1][1][0] if len(ordered) > 1 else 0.0

    if active_score < 0.62 or active_score - second_score < 0.22:
        return None
    if active_metrics["darkFraction"] > 0.18:
        return None
    if active_metrics["brightFraction"] < 0.55 or active_metrics["coloredFraction"] < 0.42:
        return None

    if active_side == ally_side:
        goal_state = {"ourGoal": "closed", "enemyGoal": "open"}
    else:
        goal_state = {"ourGoal": "open", "enemyGoal": "closed"}
    confidence = _clamp((active_score - 0.52) / 0.38)
    return ClamGoalStateHint(
        activeScoreSide=active_side,
        goalState=goal_state,
        confidence=confidence,
        metrics={side: metrics for side, (_score, metrics) in scores.items()},
    )


def clam_score_plate_highlight_score(frame: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[float, dict[str, float]]:
    x1, y1, x2, y2 = rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(12, int(round(width * 0.40)))
    pad_y = max(8, int(round(height * 0.24)))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame.shape[1], x2 + pad_x)
    crop_y2 = min(frame.shape[0], y2 + pad_y)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        metrics = {"score": 0.0, "coloredFraction": 0.0, "vividFraction": 0.0, "brightFraction": 0.0, "darkFraction": 1.0}
        return 0.0, metrics

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = (saturation > 65) & (value > 90)
    vivid = (saturation > 100) & (value > 120)
    bright = value > 145
    dark = value < 80
    colored_fraction = float(colored.mean())
    vivid_fraction = float(vivid.mean())
    bright_fraction = float(bright.mean())
    dark_fraction = float(dark.mean())
    score = (
        0.42 * colored_fraction
        + 0.30 * (1.0 - dark_fraction)
        + 0.18 * bright_fraction
        + 0.10 * vivid_fraction
    )
    metrics = {
        "score": round(float(score), 4),
        "coloredFraction": round(colored_fraction, 4),
        "vividFraction": round(vivid_fraction, 4),
        "brightFraction": round(bright_fraction, 4),
        "darkFraction": round(dark_fraction, 4),
    }
    return float(score), metrics


def apply_clam_goal_state_hint(objective: dict[str, object], hint: ClamGoalStateHint) -> dict[str, object]:
    if objective.get("kind") != "clam_blitz":
        return objective
    merged = dict(objective)
    merged["goalState"] = dict(hint.goalState)
    merged["confidence"] = max(float(merged.get("confidence", 0.0) or 0.0), _clamp(0.45 + 0.45 * hint.confidence))
    return merged


def merge_static_clam_goal_state(objective: dict[str, object], raw_objective: Mapping[str, object]) -> dict[str, object]:
    if objective.get("kind") != "clam_blitz" or raw_objective.get("kind") != "clam_blitz":
        return objective
    raw_goal = raw_objective.get("goalState")
    current_goal = objective.get("goalState")
    if not isinstance(raw_goal, Mapping) or not isinstance(current_goal, Mapping):
        return objective
    raw_has_open = raw_goal.get("ourGoal") == "open" or raw_goal.get("enemyGoal") == "open"
    current_has_open = current_goal.get("ourGoal") == "open" or current_goal.get("enemyGoal") == "open"
    if not raw_has_open or current_has_open:
        return objective
    merged = dict(objective)
    merged["goalState"] = dict(raw_goal)
    merged["confidence"] = max(float(merged.get("confidence", 0.0) or 0.0), float(raw_objective.get("confidence", 0.0) or 0.0) * 0.95)
    return merged


def count_candidate_score(reading: NumberReading) -> float:
    if reading.value is None:
        return -1.0
    score = reading.confidence + 0.035 * min(3, len(reading.text))
    if reading.text in {"0", "1"}:
        score -= 0.10
    if reading.rect is not None:
        x1, y1, x2, y2 = reading.rect
        height = y2 - y1
        if height < 28:
            score -= 0.10
    if reading.source.startswith("search"):
        score += 0.05
    return float(score)


def dedupe_number_readings(readings: Sequence[NumberReading]) -> list[NumberReading]:
    output: list[NumberReading] = []
    for reading in sorted(readings, key=count_candidate_score, reverse=True):
        if reading.rect is None:
            output.append(reading)
            continue
        duplicate = False
        for existing in output:
            if existing.rect is not None and rect_iou(reading.rect, existing.rect) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            output.append(reading)
    return output


def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1, area_a + area_b - inter))


def sample_video(
    video_path: Path,
    detector: RankedObjectiveDetector | None = None,
    context: TrackerContext | Mapping[str, object] | None = None,
    sample_interval: float = 0.20,
    start: float = 0.0,
    end: float | None = None,
    smooth: bool = True,
) -> list[RankedObjectiveUpdate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    detector = detector or RankedObjectiveDetector()
    tracker = RankedObjectiveTracker(detector) if smooth else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0
    stop = duration if end is None else min(end, duration)

    updates: list[RankedObjectiveUpdate] = []
    timestamp = max(0.0, start)
    while timestamp <= stop + 1e-6:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        if tracker is not None:
            update = tracker.read_frame(frame, context=context, timestamp=timestamp, frame_index=frame_index)
        else:
            reading = detector.read_frame(frame, context=context, timestamp=timestamp, frame_index=frame_index)
            update = RankedObjectiveUpdate(reading, [])
        updates.append(update)
        timestamp += sample_interval

    cap.release()
    return updates


def write_jsonl(path: Path, updates: Iterable[RankedObjectiveUpdate], include_debug: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for update in updates:
            file.write(json.dumps(update.to_dict(include_debug=include_debug), ensure_ascii=False) + "\n")


def write_csv(path: Path, updates: Iterable[RankedObjectiveUpdate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "timestamp_ms",
        "frame_index",
        "rule",
        "our_count",
        "enemy_count",
        "our_penalty",
        "enemy_penalty",
        "leader",
        "progress_direction",
        "our_delta_5s",
        "enemy_delta_5s",
        "counter_freshness",
        "objective_kind",
        "objective_state",
        "marker_visible",
        "marker_x",
        "marker_y",
        "confidence",
        "events",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for update in updates:
            writer.writerow(update.to_row())


def draw_debug_overlay(frame: np.ndarray, reading: RankedObjectiveReading) -> np.ndarray:
    out = frame.copy()
    for side, candidates in SIDE_COUNT_ROI_CANDIDATES.items():
        for name, roi in candidates:
            x1, y1, x2, y2 = scaled_rect(frame.shape, roi)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 1)
            cv2.putText(out, f"{side} {name}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    for side, roi in SIDE_PENALTY_ROIS.items():
        x1, y1, x2, y2 = scaled_rect(frame.shape, roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(out, f"{side} pen", (x1, max(18, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, cv2.LINE_AA)

    marker = _marker_from_objective(reading.objective)
    if marker.rect is not None and marker.visible:
        x1, y1, x2, y2 = marker.rect
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 3)
        cv2.putText(out, f"marker {marker.confidence:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 80), 2, cv2.LINE_AA)

    label = (
        f"{reading.rule} our={reading.counter.ourCount} enemy={reading.counter.enemyCount} "
        f"{reading.counter.leader} {reading.counter.progressDirection} conf={float(reading.quality.get('overallConfidence', 0.0)):.2f}"
    )
    cv2.putText(out, label, (24, out.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (24, out.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_counter_text_mask(crop: np.ndarray) -> np.ndarray:
    return make_counter_text_masks(crop)[-1][1]


def make_counter_text_masks(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_strict = (val >= 150) & (sat <= 90)
    white = (val >= 132) & (sat <= 125)
    white_soft = (val >= 168) & (sat <= 150)
    bright = val >= 178
    variants = [
        ("white_strict", white_strict),
        ("white", white),
        ("white_soft", white_soft),
        ("bright", bright),
    ]
    masks: list[tuple[str, np.ndarray]] = []
    for name, variant in variants:
        mask = np.where(variant, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        masks.append((name, mask))
    return masks


def extract_number_blobs(mask: np.ndarray, max_digits: int = 3) -> list[DigitBlob]:
    height, width = mask.shape[:2]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs: list[DigitBlob] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
        if area < max(14, int(width * height * 0.0015)):
            continue
        if area > int(width * height * 0.20):
            continue
        if h < max(10, int(height * 0.18)) or h > int(height * 0.88):
            continue
        if w < 3 or w > int(width * 0.48):
            continue
        if y > int(height * 0.76):
            continue
        aspect = w / max(1, h)
        if aspect > 1.10:
            continue
        touches = int(x <= 1) + int(y <= 1) + int(x + w >= width - 1) + int(y + h >= height - 1)
        if touches >= 2 and area > width * height * 0.03:
            continue
        blobs.append(DigitBlob(x=x, y=y, w=w, h=h, area=area))

    blobs.sort(key=lambda blob: blob.x)
    projected = extract_projection_digit_blobs(mask, max_digits=max_digits)
    if len(projected) > len(blobs):
        blobs = projected
    if len(blobs) > max_digits:
        # Prefer components that look like count digits over small "remaining" text or UI badges.
        blobs.sort(key=lambda blob: (blob.h * 2.0 + math.sqrt(blob.area) - abs((blob.y + blob.h * 0.5) / max(1, height) - 0.62) * 20.0), reverse=True)
        blobs = sorted(blobs[:max_digits], key=lambda blob: blob.x)
    return blobs


def extract_projection_digit_blobs(mask: np.ndarray, max_digits: int = 3) -> list[DigitBlob]:
    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return []
    active = mask > 0
    column_counts = active.sum(axis=0)
    thresholds = [
        max(5, int(round(height * 0.34))),
        max(5, int(round(height * 0.28))),
        max(4, int(round(height * 0.22))),
    ]
    best: list[DigitBlob] = []
    for threshold in thresholds:
        runs: list[tuple[int, int]] = []
        in_run = False
        start = 0
        for index, count in enumerate(column_counts):
            if count > threshold and not in_run:
                start = index
                in_run = True
            if in_run and (count <= threshold or index == width - 1):
                end = index if count <= threshold else index + 1
                if end - start >= 3:
                    runs.append((start, end))
                in_run = False

        blobs: list[DigitBlob] = []
        for start, end in runs:
            segment = active[:, start:end]
            ys, xs = np.where(segment)
            if len(xs) < 14:
                continue
            x = int(start + xs.min())
            y = int(ys.min())
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            area = int(len(xs))
            if h < max(10, int(height * 0.20)):
                continue
            if h > int(height * 0.92):
                continue
            if w < 3 or w > int(width * 0.42):
                continue
            if y > int(height * 0.78):
                continue
            aspect = w / max(1, h)
            if aspect > 0.95:
                continue
            blobs.append(DigitBlob(x=x, y=y, w=w, h=h, area=area))

        if len(blobs) > len(best):
            best = sorted(blobs, key=lambda blob: blob.x)
        if len(best) >= max_digits:
            break
    if len(best) > max_digits:
        best.sort(key=lambda blob: (blob.h, blob.area), reverse=True)
        best = sorted(best[:max_digits], key=lambda blob: blob.x)
    return best


def normalize_glyph(mask: np.ndarray, blob: DigitBlob) -> np.ndarray:
    target_h, target_w = DIGIT_TEMPLATE_SIZE
    pad = 2
    x1 = max(0, blob.x - pad)
    y1 = max(0, blob.y - pad)
    x2 = min(mask.shape[1], blob.x + blob.w + pad)
    y2 = min(mask.shape[0], blob.y + blob.h + pad)
    glyph = mask[y1:y2, x1:x2]
    if glyph.size == 0:
        return np.zeros(DIGIT_TEMPLATE_SIZE, dtype=np.float32)

    scale = min((target_w - 4) / max(1, glyph.shape[1]), (target_h - 4) / max(1, glyph.shape[0]))
    resized_w = max(1, int(round(glyph.shape[1] * scale)))
    resized_h = max(1, int(round(glyph.shape[0] * scale)))
    resized = cv2.resize(glyph, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(DIGIT_TEMPLATE_SIZE, dtype=np.float32)
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized.astype(np.float32) / 255.0
    return canvas


def extract_slot_glyphs(mask: np.ndarray, digit_count: int) -> list[np.ndarray]:
    slots = COUNT_SLOT_LAYOUTS.get(digit_count)
    if slots is None:
        return []
    glyphs: list[np.ndarray] = []
    for slot in slots:
        x1, y1, x2, y2 = scaled_rect(mask.shape, slot)
        slot_mask = mask[y1:y2, x1:x2]
        ys, xs = np.where(slot_mask > 0)
        if len(xs) < max(16, int(slot_mask.shape[0] * slot_mask.shape[1] * 0.018)):
            return []
        blob = DigitBlob(
            x=int(xs.min()),
            y=int(ys.min()),
            w=int(xs.max() - xs.min() + 1),
            h=int(ys.max() - ys.min() + 1),
            area=int(len(xs)),
        )
        if blob.h < max(8, int(slot_mask.shape[0] * 0.22)):
            return []
        glyphs.append(normalize_glyph(slot_mask, blob))
    return glyphs


def slot_layout_rect(shape: Sequence[int], digit_count: int) -> tuple[int, int, int, int] | None:
    slots = COUNT_SLOT_LAYOUTS.get(digit_count)
    if slots is None:
        return None
    rects = [scaled_rect(shape, slot) for slot in slots]
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def scaled_rect(shape: Sequence[int], rel_rect: Sequence[float]) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = rel_rect
    left = max(0, min(width - 1, int(round(width * x1))))
    top = max(0, min(height - 1, int(round(height * y1))))
    right = max(left + 1, min(width, int(round(width * x2))))
    bottom = max(top + 1, min(height, int(round(height * y2))))
    return left, top, right, bottom


def crop_rel(frame: np.ndarray, rel_rect: Sequence[float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = scaled_rect(frame.shape, rel_rect)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _load_timer_ocr() -> object | None:
    private_dir = Path(__file__).resolve().parents[1]
    if str(private_dir) not in sys.path:
        sys.path.append(str(private_dir))
    try:
        from match_time_ocr.timer_ocr import TimerOCR

        return TimerOCR()
    except Exception:
        return None


def _timer_debug(timer_reading: object | None) -> dict[str, object] | None:
    if timer_reading is None:
        return None
    return {
        "kind": getattr(timer_reading, "kind", None),
        "text": getattr(timer_reading, "text", None),
        "seconds": getattr(timer_reading, "seconds", None),
        "confidence": round(float(getattr(timer_reading, "confidence", 0.0)), 4),
    }


def _marker_from_objective(objective: Mapping[str, object]) -> MarkerReading:
    position = objective.get("screenPosition")
    if not isinstance(position, Mapping):
        position = objective.get("position") if isinstance(objective.get("position"), Mapping) else None
    if not isinstance(position, Mapping):
        return MarkerReading.offscreen()
    visible = bool(position.get("visible", False))
    x = position.get("x")
    y = position.get("y")
    distance = str(position.get("distanceClass", "unknown"))
    distance_class: DistanceClass = distance if distance in {"near", "mid", "far", "offscreen", "unknown"} else "unknown"  # type: ignore[assignment]
    confidence = float(objective.get("confidence", 0.0) or 0.0)
    return MarkerReading(
        visible=visible,
        x=float(x) if isinstance(x, (int, float)) else None,
        y=float(y) if isinstance(y, (int, float)) else None,
        distanceClass=distance_class,
        confidence=confidence,
        rect=None,
    )


def _near_goal_event(objective: Mapping[str, object], counter: CounterState) -> str | None:
    kind = objective.get("kind")
    active = active_scoring_side(counter)
    progress: int | None = None
    if kind == "tower_control":
        tower_progress = objective.get("towerProgress")
        if isinstance(tower_progress, Mapping) and isinstance(tower_progress.get("progressPercent"), int):
            progress = int(tower_progress["progressPercent"])
    elif kind == "rainmaker":
        rainmaker_progress = objective.get("rainmakerProgress")
        if isinstance(rainmaker_progress, Mapping) and isinstance(rainmaker_progress.get("progressPercent"), int):
            progress = int(rainmaker_progress["progressPercent"])

    if active == "our" and (progress is not None and progress >= 80 or counter.ourCount is not None and counter.ourCount <= 20):
        return "our_near_goal"
    if active == "enemy" and (progress is not None and progress >= 80 or counter.enemyCount is not None and counter.enemyCount <= 20):
        return "enemy_near_goal"
    return None


def _objective_state_text(objective: Mapping[str, object]) -> str:
    for key in ("zoneControl", "towerOwner", "rainmakerState"):
        value = objective.get(key)
        if isinstance(value, str):
            return value
    goal_state = objective.get("goalState")
    if isinstance(goal_state, Mapping):
        return f"ourGoal={goal_state.get('ourGoal')};enemyGoal={goal_state.get('enemyGoal')}"
    return "unknown"


def _offset_rect(
    rect: tuple[int, int, int, int] | None,
    base: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    bx1, by1, _bx2, _by2 = base
    return x1 + bx1, y1 + by1, x2 + bx1, y2 + by1


def _compute_knockout(our_count: int | None, enemy_count: int | None) -> bool | None:
    if our_count is None and enemy_count is None:
        return None
    return our_count == 0 or enemy_count == 0


def _plausible_count_transition(previous: int, candidate: int, age_ms: int) -> bool:
    if not 0 <= candidate <= 100:
        return False
    if candidate > previous + 2:
        return False
    elapsed_sec = max(0.0, age_ms / 1000.0)
    max_drop = max(4, int(round(18 * elapsed_sec + 5)))
    return previous - candidate <= max_drop


def _decay_confidence(value: float) -> float:
    return _clamp(float(value) * 0.82)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if value is not None]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


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
