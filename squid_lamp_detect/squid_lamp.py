from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


SideName = Literal["left", "right"]
SemanticSide = Literal["ally", "enemy"]
SlotState = Literal["alive", "down", "unknown"]
HudState = Literal["match", "non_match"]


# 1920x1080 Splatoon 3 capture layout, expressed as relative coordinates.
# The right side is the player's team in the captured videos in ../videos.
SLOT_CENTERS: dict[SideName, tuple[float, float, float, float]] = {
    "left": (0.265, 0.323, 0.382, 0.440),
    "right": (0.562, 0.620, 0.680, 0.735),
}
SLOT_CENTER_Y = 0.067
SLOT_WIDTH = 0.052
SLOT_HEIGHT = 0.080

TIMER_ROI = (0.455, 0.035, 0.545, 0.110)

COLOR_ROIS: dict[SideName, tuple[tuple[str, tuple[float, float, float, float], int, int, float], ...]] = {
    "left": (
        ("thin_line", (0.235, 0.140, 0.490, 0.155), 180, 100, 1.00),
        ("score_or_line", (0.235, 0.132, 0.490, 0.150), 150, 100, 0.90),
        ("lamp_icons", (0.245, 0.015, 0.485, 0.125), 100, 70, 0.70),
    ),
    "right": (
        ("thin_line", (0.510, 0.140, 0.780, 0.155), 180, 100, 1.00),
        ("score_or_line", (0.510, 0.132, 0.780, 0.150), 150, 100, 0.90),
        ("lamp_icons", (0.535, 0.015, 0.775, 0.125), 100, 70, 0.70),
    ),
}


@dataclass
class InkColor:
    rgb: tuple[int, int, int] | None
    hsv: tuple[int, int, int] | None
    confidence: float
    source: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "rgb": list(self.rgb) if self.rgb is not None else None,
            "hsv": list(self.hsv) if self.hsv is not None else None,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass
class SlotReading:
    index: int
    state: SlotState
    confidence: float
    down_score: float
    color_fraction: float
    special_ready: bool | None = None
    special_confidence: float = 0.0
    special_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "down_score": round(float(self.down_score), 4),
            "color_fraction": round(float(self.color_fraction), 4),
            "special_ready": self.special_ready,
            "special_confidence": round(float(self.special_confidence), 4),
            "special_score": round(float(self.special_score), 4),
        }


@dataclass
class SideReading:
    side: SideName
    alive: int | None
    down: int | None
    unknown: int
    ink_color: InkColor
    confidence: float
    slots: list[SlotReading]

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "alive": self.alive,
            "down": self.down,
            "unknown": self.unknown,
            "ink_color": self.ink_color.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "slots": [slot.to_dict() for slot in self.slots],
        }


@dataclass
class LampReading:
    timestamp: float
    frame_index: int
    timestamp_ms: int
    hud_state: HudState
    ally_alive: int | None
    enemy_alive: int | None
    ally_down: int | None
    enemy_down: int | None
    ally_ink_color: InkColor
    enemy_ink_color: InkColor
    confidence: float
    ally_side: SideName
    sides: dict[SideName, SideReading]
    smoothed: bool = False

    def to_dict(self) -> dict[str, object]:
        ally_side_reading = self.sides[self.ally_side]
        enemy_side = "left" if self.ally_side == "right" else "right"
        enemy_side_reading = self.sides[enemy_side]
        return {
            "timestamp": round(float(self.timestamp), 4),
            "timestamp_ms": int(self.timestamp_ms),
            "frame_index": int(self.frame_index),
            "hud_state": self.hud_state,
            "ally_alive": self.ally_alive,
            "enemy_alive": self.enemy_alive,
            "ally_down": self.ally_down,
            "enemy_down": self.enemy_down,
            "ally_unknown": ally_side_reading.unknown,
            "enemy_unknown": enemy_side_reading.unknown,
            "ally_special_ready": [slot.special_ready for slot in ally_side_reading.slots],
            "enemy_special_ready": [slot.special_ready for slot in enemy_side_reading.slots],
            "ally_ink_color": self.ally_ink_color.to_dict(),
            "enemy_ink_color": self.enemy_ink_color.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "ally_side": self.ally_side,
            "smoothed": self.smoothed,
            "sides": {side: reading.to_dict() for side, reading in self.sides.items()},
        }

    def to_row(self) -> dict[str, object]:
        ally_side_reading = self.sides[self.ally_side]
        enemy_side = "left" if self.ally_side == "right" else "right"
        enemy_side_reading = self.sides[enemy_side]
        return {
            "timestamp": round(float(self.timestamp), 4),
            "timestamp_ms": int(self.timestamp_ms),
            "frame_index": int(self.frame_index),
            "hud_state": self.hud_state,
            "ally_alive": self.ally_alive,
            "enemy_alive": self.enemy_alive,
            "ally_down": self.ally_down,
            "enemy_down": self.enemy_down,
            "ally_unknown": ally_side_reading.unknown,
            "enemy_unknown": enemy_side_reading.unknown,
            "ally_special_ready": _special_text(ally_side_reading.slots),
            "enemy_special_ready": _special_text(enemy_side_reading.slots),
            "ally_color_rgb": _rgb_text(self.ally_ink_color.rgb),
            "enemy_color_rgb": _rgb_text(self.enemy_ink_color.rgb),
            "ally_color_confidence": round(float(self.ally_ink_color.confidence), 4),
            "enemy_color_confidence": round(float(self.enemy_ink_color.confidence), 4),
            "confidence": round(float(self.confidence), 4),
            "smoothed": self.smoothed,
        }


@dataclass
class ColorCandidate:
    color: InkColor
    pixel_count: int
    saturated_fraction: float
    peak_ratio: float


def _load_timer_ocr() -> object | None:
    private_dir = Path(__file__).resolve().parents[1]
    if str(private_dir) not in sys.path:
        sys.path.append(str(private_dir))
    try:
        from match_time_ocr.timer_ocr import TimerOCR

        return TimerOCR()
    except Exception:
        return None


def _rgb_text(rgb: tuple[int, int, int] | None) -> str:
    if rgb is None:
        return ""
    return ",".join(str(int(channel)) for channel in rgb)


def _special_text(slots: Sequence[SlotReading]) -> str:
    values = []
    for slot in slots:
        if slot.special_ready is None:
            values.append("")
        else:
            values.append("1" if slot.special_ready else "0")
    return ",".join(values)


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


def hue_distance(a: int, b: int) -> int:
    diff = abs(int(a) - int(b)) % 180
    return min(diff, 180 - diff)


def _smooth_hue_histogram(hues: np.ndarray, radius: int = 5) -> np.ndarray:
    hist = np.bincount(hues.astype(np.uint8), minlength=180).astype(np.float32)
    smoothed = np.zeros(180, dtype=np.float32)
    offsets = np.arange(-radius, radius + 1)
    for idx in range(180):
        smoothed[idx] = float(hist[(idx + offsets) % 180].sum())
    return smoothed


def _dominant_color_from_crop(
    crop: np.ndarray,
    source: str,
    sat_min: int,
    val_min: int,
    source_weight: float,
) -> ColorCandidate:
    if crop.size == 0:
        return ColorCandidate(InkColor(None, None, 0.0, source), 0, 0.0, 0.0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat >= sat_min) & (val >= val_min)
    pixel_count = int(mask.sum())
    area = int(mask.size)
    saturated_fraction = pixel_count / max(1, area)
    if pixel_count < 40:
        return ColorCandidate(InkColor(None, None, 0.0, source), pixel_count, saturated_fraction, 0.0)

    hues = hue[mask]
    smoothed = _smooth_hue_histogram(hues)
    peak_hue = int(smoothed.argmax())
    hue_dist = np.minimum((hue.astype(np.int16) - peak_hue) % 180, (peak_hue - hue.astype(np.int16)) % 180)
    near = mask & (hue_dist <= 12)
    near_count = int(near.sum())
    peak_ratio = near_count / max(1, pixel_count)
    if near_count < 20:
        return ColorCandidate(InkColor(None, None, 0.0, source), pixel_count, saturated_fraction, peak_ratio)

    median_bgr = np.median(crop[near], axis=0)
    median_rgb = tuple(int(round(channel)) for channel in median_bgr[::-1])
    median_hsv = np.median(hsv[near], axis=0)
    color_hsv = tuple(int(round(channel)) for channel in median_hsv)
    count_score = min(1.0, math.log1p(pixel_count) / math.log(12000))
    density_score = min(1.0, saturated_fraction / 0.18)
    confidence = (0.62 * peak_ratio + 0.25 * density_score + 0.13 * count_score) * source_weight
    confidence = float(max(0.0, min(1.0, confidence)))
    return ColorCandidate(
        InkColor(median_rgb, color_hsv, confidence, source),
        pixel_count,
        saturated_fraction,
        peak_ratio,
    )


def estimate_side_color(frame: np.ndarray, side: SideName) -> InkColor:
    candidates: list[ColorCandidate] = []
    for source, roi, sat_min, val_min, weight in COLOR_ROIS[side]:
        crop, _rect = crop_rel(frame, roi)
        candidates.append(_dominant_color_from_crop(crop, source, sat_min, val_min, weight))

    valid = [candidate for candidate in candidates if candidate.color.rgb is not None]
    if not valid:
        return InkColor(None, None, 0.0, "unknown")

    valid.sort(key=lambda candidate: candidate.color.confidence, reverse=True)
    return valid[0].color


def detect_hud_confidence(frame: np.ndarray) -> float:
    crop, _rect = crop_rel(frame, TIMER_ROI)
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    crop_h, crop_w = crop.shape[:2]

    black = (val < 55).astype(np.uint8) * 255
    text = (((val > 145) & (sat < 95)) | ((val > 120) & (sat > 60) & (hue >= 15) & (hue <= 45))).astype(np.uint8) * 255
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(text, 8)
    text_components = 0
    text_area = 0
    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
        if area < 18:
            continue
        if not (crop_h * 0.22 <= h <= crop_h * 0.70):
            continue
        if not (3 <= w <= crop_w * 0.28):
            continue
        if y <= 1 or y > crop_h * 0.78:
            continue
        text_components += 1
        text_area += area

    if text_components < 2:
        return 0.0

    black_count, _black_labels, black_stats, _black_centroids = cv2.connectedComponentsWithStats(black, 8)
    best_black = 0.0
    for idx in range(1, black_count):
        _x, _y, w, h, area = [int(value) for value in black_stats[idx]]
        if w >= crop_w * 0.45 and h >= crop_h * 0.35:
            best_black = max(best_black, area / max(1, crop_w * crop_h))

    text_score = min(1.0, text_components / 3.0) * 0.65 + min(1.0, text_area / max(1, crop_w * crop_h * 0.08)) * 0.35
    black_score = min(1.0, best_black / 0.32)
    confidence = 0.72 * text_score + 0.28 * black_score
    return float(max(0.0, min(1.0, confidence)))


def detect_timer_like_digits(frame: np.ndarray) -> float:
    crop, _rect = crop_rel(frame, TIMER_ROI)
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (((val > 145) & (sat < 95)) | ((val > 120) & (sat > 60) & (hue >= 15) & (hue <= 45))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    crop_h, crop_w = mask.shape[:2]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    large: list[tuple[int, int, int, int, int]] = []
    colon_like = 0
    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
        if area < 20:
            continue
        if h >= crop_h * 0.35 and area >= 50 and w <= crop_w * 0.28:
            large.append((x, y, w, h, area))
        elif crop_h * 0.06 <= h <= crop_h * 0.18 and area >= 18 and crop_w * 0.34 <= x <= crop_w * 0.48:
            colon_like += 1

    large.sort(key=lambda item: item[0])
    if len(large) != 3 or colon_like < 2:
        return 0.0
    left = large[0][0]
    right = max(x + w for x, _y, w, _h, _area in large)
    span = right - left
    heights = [h for _x, _y, _w, h, _area in large]
    if not (crop_w * 0.42 <= span <= crop_w * 0.66):
        return 0.0
    if max(heights) - min(heights) > crop_h * 0.15:
        return 0.0
    return 0.72


def _slot_rect(frame: np.ndarray, side: SideName, index: int) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    cx = int(round(width * SLOT_CENTERS[side][index]))
    cy = int(round(height * SLOT_CENTER_Y))
    slot_w = int(round(width * SLOT_WIDTH))
    slot_h = int(round(height * SLOT_HEIGHT))
    left = max(0, cx - slot_w // 2)
    top = max(0, cy - slot_h // 2)
    right = min(width, cx + slot_w // 2)
    bottom = min(height, cy + slot_h // 2)
    return left, top, right, bottom


def _team_color_fraction(slot: np.ndarray, color: InkColor) -> float:
    if color.hsv is None:
        hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        return float(((sat > 110) & (val > 90)).mean())

    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    target_hue = int(color.hsv[0])
    dist = np.minimum((hue - target_hue) % 180, (target_hue - hue) % 180)
    return float(((dist <= 15) & (sat > 75) & (val > 65)).mean())


def _component_features(mask: np.ndarray) -> tuple[float, float]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    slot_h, slot_w = mask.shape[:2]
    best_ratio = 0.0
    best_extent = 0.0

    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
        ratio = area / max(1, slot_h * slot_w)
        extent = (w / max(1, slot_w)) * (h / max(1, slot_h))
        touches = int(x <= 1) + int(y <= 1) + int(x + w >= slot_w - 1) + int(y + h >= slot_h - 1)
        if touches >= 3 and ratio > 0.45:
            # Stage/background patches can fill the whole slot when a fixed ROI lands off the lamp.
            continue
        if area > best_ratio * slot_h * slot_w:
            best_ratio = float(ratio)
            best_extent = float(extent)

    return best_ratio, best_extent


def _diagonal_x_score(mask: np.ndarray) -> tuple[float, float, float, float]:
    slot_h, slot_w = mask.shape[:2]
    yy, xx = np.mgrid[0:slot_h, 0:slot_w]
    diag_width = max(6, int(round(min(slot_h, slot_w) * 0.09)))
    dist_a = np.abs(yy - (slot_h - 1) * xx / max(1, slot_w - 1))
    dist_b = np.abs(yy - ((slot_h - 1) - (slot_h - 1) * xx / max(1, slot_w - 1)))
    diag_a = dist_a < diag_width
    diag_b = dist_b < diag_width
    center = (xx > slot_w * 0.08) & (xx < slot_w * 0.92) & (yy > slot_h * 0.05) & (yy < slot_h * 0.95)
    diagonal = (diag_a | diag_b) & center
    outside = (~(diag_a | diag_b)) & center
    density_a = float(mask[diag_a & center].mean() / 255.0) if np.any(diag_a & center) else 0.0
    density_b = float(mask[diag_b & center].mean() / 255.0) if np.any(diag_b & center) else 0.0
    outside_density = float(mask[outside].mean() / 255.0) if np.any(outside) else 0.0
    diagonal_density = float(mask[diagonal].mean() / 255.0) if np.any(diagonal) else 0.0
    score = min(density_a, density_b) - 0.45 * outside_density
    score = max(score, diagonal_density - 0.70 * outside_density)
    return float(max(0.0, min(1.0, score))), density_a, density_b, outside_density


def _special_badge_score(slot: np.ndarray, color: InkColor) -> float:
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    slot_h, slot_w = slot.shape[:2]
    y1 = int(round(slot_h * 0.55))
    y2 = int(round(slot_h * 0.98))
    x1 = int(round(slot_w * 0.05))
    x2 = int(round(slot_w * 0.95))
    region = hsv[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    hue = region[:, :, 0].astype(np.int16)
    sat = region[:, :, 1]
    val = region[:, :, 2]
    if color.hsv is not None:
        target_hue = int(color.hsv[0])
        dist = np.minimum((hue - target_hue) % 180, (target_hue - hue) % 180)
        team_color = (dist <= 18) & (sat > 70) & (val > 60)
    else:
        team_color = np.zeros_like(sat, dtype=bool)

    white = (sat < 85) & (val > 150)
    colored = (sat > 75) & (val > 95) & (~team_color)
    mask = ((white | colored) & (~team_color)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    region_h, region_w = mask.shape[:2]
    best_score = 0.0
    for idx in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[idx]]
        if area < 18 or w < 4 or h < 4:
            continue
        if w > region_w * 0.75 or h > region_h * 0.90:
            continue

        box_sat = sat[y : y + h, x : x + w]
        box_val = val[y : y + h, x : x + w]
        bottomness = (y + h) / max(1, region_h)
        compactness = area / max(1, w * h)
        size_score = min(1.0, area / 260.0)
        white_fraction = float(((box_sat < 85) & (box_val > 150)).mean())
        saturated_fraction = float(((box_sat > 95) & (box_val > 90)).mean())
        icon_fraction = max(white_fraction, saturated_fraction)
        shape_score = min(1.0, min(w, h) / max(1, min(region_w, region_h) * 0.32))
        score = (
            0.27 * size_score
            + 0.40 * bottomness
            + 0.14 * compactness
            + 0.07 * min(1.0, icon_fraction * 1.4)
            + 0.12 * shape_score
        )
        best_score = max(best_score, score)

    return float(max(0.0, min(1.0, best_score)))


def classify_slot(slot: np.ndarray, index: int, color: InkColor, special_allowed: bool = True) -> SlotReading:
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    neutral = ((sat < 80) & (val > 45) & (val < 220)).astype(np.uint8) * 255
    neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    component_ratio, component_extent = _component_features(neutral)
    diagonal_score, density_a, density_b, _outside_density = _diagonal_x_score(neutral)
    color_fraction = _team_color_fraction(slot, color)

    down_score = 0.0
    if component_ratio >= 0.30 and component_extent >= 0.55:
        down_score = max(down_score, 0.92)
    if component_ratio >= 0.18 and component_extent >= 0.40 and diagonal_score >= 0.045:
        down_score = max(down_score, 0.74)
    if diagonal_score >= 0.19 and component_ratio >= 0.10:
        down_score = max(down_score, 0.76)
    if min(density_a, density_b) >= 0.50 and component_ratio >= 0.16:
        down_score = max(down_score, 0.82)
    if diagonal_score >= 0.34 and min(density_a, density_b) >= 0.55:
        down_score = max(down_score, 0.72)

    special_score = _special_badge_score(slot, color) if special_allowed else 0.0

    # Strong team-colored lamp pixels usually mean the weapon is visible and no X overlay is present.
    if color_fraction >= 0.28 and down_score < 0.85:
        down_score *= 0.72
    if color_fraction >= 0.40 and diagonal_score < 0.12 and down_score < 0.85:
        down_score *= 0.55
    if color_fraction >= 0.45 and down_score < 0.96:
        down_score *= 0.50
    if color_fraction >= 0.28 and diagonal_score < 0.035 and down_score >= 0.85:
        down_score *= 0.45

    if (
        down_score < 0.62
        and component_ratio >= 0.20
        and component_extent >= 0.40
        and special_score >= 0.80
        and _outside_density >= 0.24
    ):
        down_score = max(down_score, 0.68)

    down_score = float(max(0.0, min(1.0, down_score)))
    if down_score >= 0.62:
        return SlotReading(index, "down", down_score, down_score, color_fraction)
    if 0.45 <= down_score < 0.62:
        return SlotReading(index, "unknown", 1.0 - abs(0.535 - down_score), down_score, color_fraction)

    alive_confidence = max(0.45, min(0.95, 1.0 - down_score))
    if color_fraction < 0.04 and component_ratio < 0.05 and special_score < 0.40:
        return SlotReading(index, "unknown", 0.35, down_score, color_fraction)
    special_ready = special_score >= 0.67
    special_confidence = special_score if special_ready else max(0.50, 1.0 - special_score)
    return SlotReading(
        index,
        "alive",
        alive_confidence,
        down_score,
        color_fraction,
        special_ready=special_ready,
        special_confidence=special_confidence,
        special_score=special_score,
    )


class SquidLampDetector:
    def __init__(
        self,
        ally_side: SideName = "right",
        min_hud_confidence: float = 0.35,
        use_timer_ocr: bool = True,
    ) -> None:
        if ally_side not in {"left", "right"}:
            raise ValueError("ally_side must be 'left' or 'right'")
        self.ally_side: SideName = ally_side
        self.enemy_side: SideName = "left" if ally_side == "right" else "right"
        self.min_hud_confidence = min_hud_confidence
        self.timer_ocr = _load_timer_ocr() if use_timer_ocr else None

    def read_frame(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = -1) -> LampReading:
        timestamp_ms = int(round(timestamp * 1000))
        legacy_hud_confidence = detect_hud_confidence(frame)
        timer_reading = None
        if self.timer_ocr is not None:
            read_frame = getattr(self.timer_ocr, "read_frame", None)
            if read_frame is not None:
                timer_reading = read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        if self.timer_ocr is not None:
            if timer_reading is None:
                timer_like_confidence = detect_timer_like_digits(frame)
                if timer_like_confidence >= 0.60:
                    hud_confidence = 0.70 * timer_like_confidence + 0.30 * legacy_hud_confidence
                else:
                    hud_confidence = min(legacy_hud_confidence, 0.20)
            else:
                timer_confidence = float(getattr(timer_reading, "confidence", 0.0))
                hud_confidence = 0.70 * timer_confidence + 0.30 * legacy_hud_confidence
        else:
            hud_confidence = legacy_hud_confidence
        if hud_confidence < self.min_hud_confidence:
            unknown_color = InkColor(None, None, 0.0, "unknown")
            sides = {
                "left": self._unknown_side("left", unknown_color),
                "right": self._unknown_side("right", unknown_color),
            }
            return LampReading(
                timestamp=timestamp,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                hud_state="non_match",
                ally_alive=None,
                enemy_alive=None,
                ally_down=None,
                enemy_down=None,
                ally_ink_color=unknown_color,
                enemy_ink_color=unknown_color,
                confidence=hud_confidence,
                ally_side=self.ally_side,
                sides=sides,
            )

        timer_seconds = getattr(timer_reading, "seconds", None)
        # Splatoon specials cannot realistically be ready in the very first seconds.
        special_allowed = not (isinstance(timer_seconds, int) and timer_seconds > 270)
        colors = {
            "left": estimate_side_color(frame, "left"),
            "right": estimate_side_color(frame, "right"),
        }
        sides = {
            "left": self._read_side(frame, "left", colors["left"], special_allowed=special_allowed),
            "right": self._read_side(frame, "right", colors["right"], special_allowed=special_allowed),
        }
        return self._build_reading(
            timestamp=timestamp,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            hud_state="match",
            hud_confidence=hud_confidence,
            sides=sides,
            smoothed=False,
        )

    def _unknown_side(self, side: SideName, color: InkColor) -> SideReading:
        slots = [SlotReading(index, "unknown", 0.0, 0.0, 0.0) for index in range(4)]
        return SideReading(side, None, None, 4, color, 0.0, slots)

    def _read_side(
        self,
        frame: np.ndarray,
        side: SideName,
        color: InkColor,
        special_allowed: bool = True,
    ) -> SideReading:
        slots: list[SlotReading] = []
        for index in range(4):
            x1, y1, x2, y2 = _slot_rect(frame, side, index)
            slot = frame[y1:y2, x1:x2]
            slots.append(classify_slot(slot, index, color, special_allowed=special_allowed))
        return _make_side_reading(side, color, slots)

    def _build_reading(
        self,
        timestamp: float,
        frame_index: int,
        timestamp_ms: int,
        hud_state: HudState,
        hud_confidence: float,
        sides: dict[SideName, SideReading],
        smoothed: bool,
    ) -> LampReading:
        ally = sides[self.ally_side]
        enemy = sides[self.enemy_side]
        side_confidences = [ally.confidence, enemy.confidence, ally.ink_color.confidence, enemy.ink_color.confidence]
        confidence = float(max(0.0, min(1.0, 0.35 * hud_confidence + 0.65 * np.mean(side_confidences))))
        return LampReading(
            timestamp=timestamp,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            hud_state=hud_state,
            ally_alive=ally.alive,
            enemy_alive=enemy.alive,
            ally_down=ally.down,
            enemy_down=enemy.down,
            ally_ink_color=ally.ink_color,
            enemy_ink_color=enemy.ink_color,
            confidence=confidence,
            ally_side=self.ally_side,
            sides=sides,
            smoothed=smoothed,
        )


def _make_side_reading(side: SideName, color: InkColor, slots: list[SlotReading]) -> SideReading:
    unknown = sum(1 for slot in slots if slot.state == "unknown")
    alive = sum(1 for slot in slots if slot.state == "alive")
    down = sum(1 for slot in slots if slot.state == "down")
    if unknown >= 3:
        alive_value: int | None = None
        down_value: int | None = None
    else:
        alive_value = alive
        down_value = down

    slot_confidence = float(np.mean([slot.confidence for slot in slots])) if slots else 0.0
    completeness = 1.0 - unknown / max(1, len(slots))
    confidence = float(max(0.0, min(1.0, slot_confidence * (0.55 + 0.45 * completeness))))
    return SideReading(side, alive_value, down_value, unknown, color, confidence, slots)


@dataclass
class _StableColor:
    color: InkColor | None = None
    pending: InkColor | None = None
    pending_count: int = 0

    def update(self, candidate: InkColor, min_confidence: float = 0.45) -> InkColor:
        if candidate.rgb is None or candidate.hsv is None or candidate.confidence < min_confidence:
            return self.color or candidate

        if self.color is None or self.color.rgb is None or self.color.hsv is None or self.color.confidence < 0.35:
            self.color = candidate
            self.pending = None
            self.pending_count = 0
            return self.color

        distance = hue_distance(self.color.hsv[0], candidate.hsv[0])
        if distance <= 28:
            old_weight = 10.0 + max(0.0, self.color.confidence) * 6.0
            new_weight = max(0.25, candidate.confidence * 0.75)
            rgb = tuple(
                int(round((self.color.rgb[idx] * old_weight + candidate.rgb[idx] * new_weight) / (old_weight + new_weight)))
                for idx in range(3)
            )
            hsv = tuple(
                int(round((self.color.hsv[idx] * old_weight + candidate.hsv[idx] * new_weight) / (old_weight + new_weight)))
                for idx in range(3)
            )
            confidence = max(self.color.confidence * 0.96, min(0.98, candidate.confidence))
            self.color = InkColor(rgb, hsv, confidence, "smoothed")
            self.pending = None
            self.pending_count = 0
            return self.color

        if candidate.confidence >= 0.88:
            if self.pending is not None and self.pending.hsv is not None and hue_distance(self.pending.hsv[0], candidate.hsv[0]) <= 18:
                self.pending_count += 1
            else:
                self.pending = candidate
                self.pending_count = 1
            if self.pending_count >= 3:
                self.color = candidate
                self.pending = None
                self.pending_count = 0
        return self.color

    def reset(self) -> None:
        self.color = None
        self.pending = None
        self.pending_count = 0


class SquidLampTracker:
    def __init__(
        self,
        detector: SquidLampDetector,
        window_size: int = 5,
        reset_after_missing: int = 8,
    ) -> None:
        self.detector = detector
        self.window_size = max(1, int(window_size))
        self.reset_after_missing = max(1, int(reset_after_missing))
        self.history: deque[LampReading] = deque(maxlen=self.window_size)
        self.colors = {"left": _StableColor(), "right": _StableColor()}
        self.missing_count = 0

    def read_frame(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = -1) -> LampReading:
        raw = self.detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        return self.update(raw)

    def update(self, raw: LampReading) -> LampReading:
        if raw.hud_state != "match":
            self.missing_count += 1
            if self.missing_count >= self.reset_after_missing:
                self.reset()
            return raw

        self.missing_count = 0
        sides: dict[SideName, SideReading] = {}
        for side_name, side in raw.sides.items():
            stable_color = self.colors[side_name].update(side.ink_color)
            slots = [replace(slot) for slot in side.slots]
            sides[side_name] = _make_side_reading(side_name, stable_color, slots)

        staged = self.detector._build_reading(
            timestamp=raw.timestamp,
            frame_index=raw.frame_index,
            timestamp_ms=raw.timestamp_ms,
            hud_state=raw.hud_state,
            hud_confidence=detect_hud_confidence_from_reading(raw),
            sides=sides,
            smoothed=False,
        )
        self.history.append(staged)
        smoothed_sides = {side: self._smooth_side(side) for side in ("left", "right")}
        return self.detector._build_reading(
            timestamp=raw.timestamp,
            frame_index=raw.frame_index,
            timestamp_ms=raw.timestamp_ms,
            hud_state=raw.hud_state,
            hud_confidence=max(0.35, raw.confidence),
            sides=smoothed_sides,
            smoothed=True,
        )

    def _smooth_side(self, side: SideName) -> SideReading:
        side_readings = [reading.sides[side] for reading in self.history if reading.hud_state == "match"]
        if not side_readings:
            return SideReading(side, None, None, 4, InkColor(None, None, 0.0, "unknown"), 0.0, [])

        latest = side_readings[-1]
        slots: list[SlotReading] = []
        for index in range(4):
            votes = {"alive": 0.0, "down": 0.0, "unknown": 0.0}
            special_votes = {True: 0.0, False: 0.0}
            score_sum = 0.0
            color_sum = 0.0
            special_score_sum = 0.0
            weight_sum = 0.0
            special_weight_sum = 0.0
            for offset, reading in enumerate(reversed(side_readings)):
                slot = reading.slots[index]
                recency = 3.0 / (1.0 + offset * 1.15)
                weight = recency * max(0.05, slot.confidence)
                votes[slot.state] += weight
                score_sum += slot.down_score * weight
                color_sum += slot.color_fraction * weight
                weight_sum += weight
                if slot.special_ready is not None:
                    special_weight = recency * max(0.05, slot.special_confidence)
                    special_votes[slot.special_ready] += special_weight
                    special_score_sum += slot.special_score * special_weight
                    special_weight_sum += special_weight
            state = max(votes, key=votes.get)
            total_vote = sum(votes.values()) or 1.0
            confidence = votes[state] / total_vote
            special_ready: bool | None = None
            special_confidence = 0.0
            special_score = 0.0
            if state == "alive" and special_weight_sum > 0:
                special_ready = special_votes[True] >= special_votes[False]
                special_total = special_votes[True] + special_votes[False]
                special_confidence = max(special_votes[True], special_votes[False]) / max(1e-6, special_total)
                special_score = special_score_sum / max(1e-6, special_weight_sum)
            slots.append(
                SlotReading(
                    index=index,
                    state=state,  # type: ignore[arg-type]
                    confidence=float(confidence),
                    down_score=float(score_sum / max(1e-6, weight_sum)),
                    color_fraction=float(color_sum / max(1e-6, weight_sum)),
                    special_ready=special_ready,
                    special_confidence=float(special_confidence),
                    special_score=float(special_score),
                )
            )
        return _make_side_reading(side, latest.ink_color, slots)

    def reset(self) -> None:
        self.history.clear()
        for color in self.colors.values():
            color.reset()
        self.missing_count = 0


def detect_hud_confidence_from_reading(reading: LampReading) -> float:
    if reading.hud_state != "match":
        return reading.confidence
    side_values = [side.confidence for side in reading.sides.values()]
    if not side_values:
        return reading.confidence
    return float(max(0.35, min(1.0, reading.confidence + 0.10 * np.mean(side_values))))


def sample_video(
    video_path: Path,
    detector: SquidLampDetector,
    sample_interval: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
    smooth: bool = True,
) -> list[LampReading]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0
    stop = duration if end is None else min(end, duration)
    tracker = SquidLampTracker(detector) if smooth else None

    readings: list[LampReading] = []
    timestamp = max(0.0, start)
    while timestamp <= stop + 1e-6:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        if tracker is not None:
            readings.append(tracker.read_frame(frame, timestamp=timestamp, frame_index=frame_index))
        else:
            readings.append(detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index))
        timestamp += sample_interval

    cap.release()
    return readings


def write_jsonl(path: Path, readings: Iterable[LampReading]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for reading in readings:
            file.write(json.dumps(reading.to_dict(), ensure_ascii=False) + "\n")


def write_csv(path: Path, readings: Iterable[LampReading]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "timestamp_ms",
        "frame_index",
        "hud_state",
        "ally_alive",
        "enemy_alive",
        "ally_down",
        "enemy_down",
        "ally_unknown",
        "enemy_unknown",
        "ally_special_ready",
        "enemy_special_ready",
        "ally_color_rgb",
        "enemy_color_rgb",
        "ally_color_confidence",
        "enemy_color_confidence",
        "confidence",
        "smoothed",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for reading in readings:
            writer.writerow(reading.to_row())


def draw_debug_overlay(frame: np.ndarray, reading: LampReading) -> np.ndarray:
    out = frame.copy()
    for side_name, side in reading.sides.items():
        color_bgr = (80, 220, 80) if side_name == reading.ally_side else (80, 80, 255)
        for slot in side.slots:
            x1, y1, x2, y2 = _slot_rect(frame, side_name, slot.index)
            if slot.state == "down":
                rect_color = (0, 0, 255)
            elif slot.state == "alive":
                rect_color = (0, 220, 0)
            else:
                rect_color = (0, 220, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), rect_color, 2)
            if slot.special_ready is True:
                cv2.circle(out, (x2 - 10, y2 - 10), 7, (0, 255, 255), -1)
                cv2.circle(out, (x2 - 10, y2 - 10), 7, (0, 0, 0), 1)
            cv2.putText(
                out,
                f"{slot.index}:{slot.state[0]} {slot.down_score:.2f}/{slot.special_score:.2f}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                rect_color,
                1,
                cv2.LINE_AA,
            )
        if side.ink_color.rgb is not None:
            rgb = side.ink_color.rgb
            swatch = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            x = 28 if side_name == "left" else frame.shape[1] - 168
            y = 24
            cv2.rectangle(out, (x, y), (x + 34, y + 34), swatch, -1)
            cv2.rectangle(out, (x, y), (x + 34, y + 34), color_bgr, 2)
            cv2.putText(
                out,
                f"{side_name} {side.alive}/{side.down} {side.ink_color.confidence:.2f}",
                (x + 42, y + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        out,
        f"ally={reading.ally_alive} enemy={reading.enemy_alive} conf={reading.confidence:.2f}",
        (frame.shape[1] // 2 - 190, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def reading_to_event(reading: LampReading) -> dict[str, object]:
    state = "unknown"
    severity = 0.0
    if reading.ally_alive is not None and reading.enemy_alive is not None:
        diff = reading.ally_alive - reading.enemy_alive
        if diff > 0:
            state = "advantage"
        elif diff < 0:
            state = "disadvantage"
        else:
            state = "even"
        severity = min(1.0, abs(diff) / 4.0)
    return {
        "event": "squid_lamp_update",
        "ally_alive": reading.ally_alive,
        "enemy_alive": reading.enemy_alive,
        "ally_down": reading.ally_down,
        "enemy_down": reading.enemy_down,
        "ally_unknown": reading.sides[reading.ally_side].unknown,
        "enemy_unknown": reading.sides["left" if reading.ally_side == "right" else "right"].unknown,
        "enemy_special_ready": [
            slot.special_ready for slot in reading.sides["left" if reading.ally_side == "right" else "right"].slots
        ],
        "state": state,
        "severity": round(float(severity), 4),
        "confidence": round(float(reading.confidence), 4),
        "timestamp_ms": reading.timestamp_ms,
    }


def ink_color_event(reading: LampReading) -> dict[str, object]:
    return {
        "event": "ink_color_update",
        "ally_color": reading.ally_ink_color.to_dict(),
        "enemy_color": reading.enemy_ink_color.to_dict(),
        "confidence": round(
            float(min(reading.ally_ink_color.confidence, reading.enemy_ink_color.confidence)),
            4,
        ),
        "timestamp_ms": reading.timestamp_ms,
    }


def dataclass_to_json(data: object) -> str:
    return json.dumps(asdict(data), ensure_ascii=False)
