from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


DIGIT_TEMPLATE_SIZE = (42, 28)  # height, width
TIME_TEXT_ROI = (0.470, 0.048, 0.535, 0.095)  # x1, y1, x2, y2
OVERTIME_ROI = (0.435, 0.040, 0.565, 0.125)
DIGIT_SLOTS = (
    (0.08, 0.02, 0.32, 0.98),
    (0.42, 0.02, 0.66, 0.98),
    (0.61, 0.02, 0.88, 0.98),
)


@dataclass
class DigitBlob:
    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class TimerReading:
    timestamp: float
    frame_index: int
    kind: str
    text: str
    seconds: int | None
    confidence: float
    inferred: bool = False

    def to_row(self) -> dict[str, str | int | float | None | bool]:
        return asdict(self)


def format_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


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


def make_timer_text_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    white = (val >= 135) & (sat <= 115)
    yellow = (val >= 120) & (sat >= 45) & (hue >= 15) & (hue <= 45)
    mask = np.where(white | yellow, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def extract_digit_blobs(mask: np.ndarray) -> list[DigitBlob]:
    height, width = mask.shape[:2]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs: list[DigitBlob] = []

    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 35:
            continue
        if h < max(16, int(height * 0.35)):
            continue
        if y <= 2 and h < int(height * 0.70):
            continue
        if w < 4 or w > int(width * 0.38):
            continue
        if h > int(height * 0.95):
            continue
        if y > int(height * 0.72):
            continue
        blobs.append(DigitBlob(x=x, y=y, w=w, h=h, area=area))

    blobs.sort(key=lambda blob: blob.x)
    return blobs


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


def normalize_slot_glyph(mask: np.ndarray, rel_rect: Sequence[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = scaled_rect(mask.shape, rel_rect)
    slot = mask[y1:y2, x1:x2]
    ys, xs = np.where(slot > 0)
    if len(xs) < 35:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    blob = DigitBlob(x=left, y=top, w=right - left, h=bottom - top, area=len(xs))
    return normalize_glyph(slot, blob)


def extract_slot_glyphs(frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    crop, _rect = crop_rel(frame, TIME_TEXT_ROI)
    mask = make_timer_text_mask(crop)
    glyphs: list[np.ndarray] = []
    for slot in DIGIT_SLOTS:
        glyph = normalize_slot_glyph(mask, slot)
        if glyph is None:
            return [], mask
        glyphs.append(glyph)
    return glyphs, mask


def extract_digit_glyphs(frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list[DigitBlob]]:
    crop, _rect = crop_rel(frame, TIME_TEXT_ROI)
    mask = make_timer_text_mask(crop)
    blobs = extract_digit_blobs(mask)
    glyphs = [normalize_glyph(mask, blob) for blob in blobs]
    return glyphs, mask, blobs


def load_digit_templates(templates_dir: Path) -> dict[str, np.ndarray]:
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


class TimerOCR:
    def __init__(
        self,
        templates_dir: Path | None = None,
        min_digit_score: float = 0.70,
    ) -> None:
        self.templates_dir = templates_dir or Path(__file__).resolve().parent / "templates" / "digits"
        self.templates = load_digit_templates(self.templates_dir)
        self.min_digit_score = min_digit_score
        missing = [digit for digit in "0123456789" if digit not in self.templates]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                f"digit templates are missing ({joined}); run train_templates.py first"
            )

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

    def read_digit_glyphs(self, glyphs: Sequence[np.ndarray], timestamp: float, frame_index: int) -> TimerReading | None:
        if len(glyphs) != 3:
            return None
        classified = [self.classify_digit(glyph) for glyph in glyphs]
        digits = [digit for digit, _score in classified]
        scores = [score for _digit, score in classified]
        if not all(digits) or min(scores) < self.min_digit_score:
            return None

        seconds_tens = int(digits[1])
        if seconds_tens > 5:
            return None

        seconds = int(digits[0]) * 60 + seconds_tens * 10 + int(digits[2])
        return TimerReading(
            timestamp=timestamp,
            frame_index=frame_index,
            kind="time",
            text=f"{digits[0]}:{digits[1]}{digits[2]}",
            seconds=seconds,
            confidence=min(scores),
        )

    def read_frame(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = -1) -> TimerReading | None:
        glyphs, _mask, blobs = extract_digit_glyphs(frame)

        reading = self.read_digit_glyphs(glyphs, timestamp, frame_index)
        if reading is not None:
            return reading

        slot_glyphs, _slot_mask = extract_slot_glyphs(frame)
        reading = self.read_digit_glyphs(slot_glyphs, timestamp, frame_index)
        if reading is not None and reading.confidence >= max(self.min_digit_score, 0.82):
            return reading

        overtime_confidence = detect_overtime(frame)
        if overtime_confidence >= 0.50:
            return TimerReading(
                timestamp=timestamp,
                frame_index=frame_index,
                kind="overtime",
                text="延長中!",
                seconds=None,
                confidence=overtime_confidence,
            )

        return None


def detect_overtime(frame: np.ndarray) -> float:
    crop, _rect = crop_rel(frame, OVERTIME_ROI)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    orange_red = (val >= 90) & (sat >= 80) & ((hue <= 18) | (hue >= 170))
    mask = np.where(orange_red, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    crop_h, crop_w = mask.shape[:2]
    best = 0.0
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        center_x = float(centroids[idx][0]) / max(1, crop_w)
        if area < 300:
            continue
        if w < int(crop_w * 0.20) or h < int(crop_h * 0.18):
            continue
        if not 0.24 <= center_x <= 0.76:
            continue
        confidence = min(1.0, area / 1800.0)
        best = max(best, confidence)
    return best


def sample_video(
    video_path: Path,
    ocr: TimerOCR,
    sample_interval: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
) -> list[TimerReading | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0
    stop = duration if end is None else min(end, duration)

    readings: list[TimerReading | None] = []
    timestamp = max(0.0, start)
    while timestamp <= stop + 1e-6:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        readings.append(ocr.read_frame(frame, timestamp=timestamp, frame_index=frame_index))
        timestamp += sample_interval

    cap.release()
    return readings


def interpolate_readings(
    readings: Iterable[TimerReading | None],
    sample_interval: float,
    max_gap: float = 15.0,
    fps: float = 60.0,
) -> list[TimerReading | None]:
    output: list[TimerReading | None] = []
    last_known: TimerReading | None = None
    max_steps = int(math.floor(max_gap / max(sample_interval, 1e-6)))
    gap_steps = 0

    for reading in readings:
        if reading is not None:
            output.append(reading)
            if reading.kind in {"time", "overtime"}:
                last_known = reading
                gap_steps = 0
            continue

        if last_known is None or len(output) == 0:
            output.append(None)
            continue

        gap_steps += 1
        if gap_steps > max_steps:
            output.append(None)
            continue

        timestamp = last_known.timestamp + gap_steps * sample_interval
        frame_index = int(round(last_known.frame_index + gap_steps * sample_interval * fps))
        if last_known.kind == "overtime":
            output.append(
                TimerReading(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    kind="overtime",
                    text="延長中!",
                    seconds=None,
                    confidence=max(0.1, last_known.confidence * 0.5),
                    inferred=True,
                )
            )
            continue

        if last_known.seconds is None:
            output.append(None)
            continue
        seconds = last_known.seconds - int(round(gap_steps * sample_interval))
        if seconds < 0:
            output.append(None)
            continue
        output.append(
            TimerReading(
                timestamp=timestamp,
                frame_index=frame_index,
                kind="time",
                text=format_seconds(seconds),
                seconds=seconds,
                confidence=max(0.1, last_known.confidence * 0.5),
                inferred=True,
            )
        )

    return output


def write_csv(path: Path, readings: Iterable[TimerReading | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "frame_index", "kind", "text", "seconds", "confidence", "inferred"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for reading in readings:
            if reading is None:
                continue
            writer.writerow(reading.to_row())


def write_jsonl(path: Path, readings: Iterable[TimerReading | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for reading in readings:
            if reading is None:
                continue
            file.write(json.dumps(reading.to_row(), ensure_ascii=False) + "\n")
