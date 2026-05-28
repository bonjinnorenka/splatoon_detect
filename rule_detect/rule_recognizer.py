from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


Rule = Literal[
    "turf_war",
    "splat_zones",
    "tower_control",
    "rainmaker",
    "clam_blitz",
    "unknown",
]
Source = Literal["template", "unknown"]

RULES: tuple[Rule, ...] = (
    "turf_war",
    "splat_zones",
    "tower_control",
    "rainmaker",
    "clam_blitz",
)

RULE_LABELS: dict[Rule, str] = {
    "turf_war": "ナワバリバトル",
    "splat_zones": "ガチエリア",
    "tower_control": "ガチヤグラ",
    "rainmaker": "ガチホコバトル",
    "clam_blitz": "ガチアサリ",
    "unknown": "不明",
}

DEFAULT_RULE_TITLE_ROI = (0.36, 0.22, 0.64, 0.62)
DEFAULT_TEMPLATE_WIDTH = 320
DEFAULT_ACCEPT_THRESHOLD = 0.78


@dataclass(frozen=True)
class TemplateMatch:
    rule: Rule
    score: float
    path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "score": round(float(self.score), 4),
            "path": self.path,
        }


@dataclass(frozen=True)
class RuleRecognitionResult:
    rule: Rule
    confidence: float
    source: Source
    locked: bool
    frame_time_ms: int
    timestamp: float
    frame_index: int
    template_path: str | None = None
    candidate_rule: Rule | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 4)
        data["timestamp"] = round(float(self.timestamp), 4)
        return data


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


def preprocess_rule_roi(crop: np.ndarray, output_width: int = DEFAULT_TEMPLATE_WIDTH) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((1, output_width), dtype=np.uint8)

    scale = output_width / max(1, crop.shape[1])
    output_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized

    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
    return blurred.astype(np.uint8)


def crop_and_preprocess_rule_roi(
    frame: np.ndarray,
    rel_rect: Sequence[float] = DEFAULT_RULE_TITLE_ROI,
    output_width: int = DEFAULT_TEMPLATE_WIDTH,
) -> np.ndarray:
    crop, _rect = crop_rel(frame, rel_rect)
    return preprocess_rule_roi(crop, output_width=output_width)


def infer_rule_from_template_path(path: Path) -> Rule | None:
    stem = path.stem
    for rule in RULES:
        if stem == rule or stem.startswith(f"{rule}_"):
            return rule
    return None


def _read_template(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if expected_shape is not None and image.shape != expected_shape:
        image = cv2.resize(image, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


class RuleRecognizer:
    def __init__(
        self,
        templates_dir: Path | None = None,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        roi: Sequence[float] = DEFAULT_RULE_TITLE_ROI,
        output_width: int = DEFAULT_TEMPLATE_WIDTH,
    ) -> None:
        self.templates_dir = templates_dir or Path(__file__).resolve().parent / "templates" / "jp_1080p"
        self.accept_threshold = accept_threshold
        self.roi = tuple(float(value) for value in roi)
        self.output_width = int(output_width)
        self.templates = self._load_templates()

    def _load_templates(self) -> list[tuple[Rule, Path, np.ndarray]]:
        if not self.templates_dir.exists():
            return []

        templates: list[tuple[Rule, Path, np.ndarray]] = []
        for path in sorted(self.templates_dir.glob("*.png")):
            rule = infer_rule_from_template_path(path)
            if rule is None:
                continue
            image = _read_template(path)
            if image is not None:
                templates.append((rule, path, image))
        return templates

    def match_frame(self, frame: np.ndarray) -> TemplateMatch | None:
        roi = crop_and_preprocess_rule_roi(frame, self.roi, self.output_width)
        best: TemplateMatch | None = None

        for rule, path, template in self.templates:
            candidate = template
            if candidate.shape != roi.shape:
                candidate = cv2.resize(candidate, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

            result = cv2.matchTemplate(roi, candidate, cv2.TM_CCOEFF_NORMED)
            score = float(result.max()) if result.size else 0.0
            if not math.isfinite(score):
                score = 0.0
            if best is None or score > best.score:
                best = TemplateMatch(rule=rule, score=score, path=str(path))

        return best

    def recognize_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = -1,
    ) -> RuleRecognitionResult:
        best = self.match_frame(frame)
        if best is not None and best.score >= self.accept_threshold:
            return RuleRecognitionResult(
                rule=best.rule,
                confidence=best.score,
                source="template",
                locked=True,
                frame_time_ms=int(round(timestamp * 1000.0)),
                timestamp=timestamp,
                frame_index=frame_index,
                template_path=best.path,
                candidate_rule=best.rule,
            )

        return RuleRecognitionResult(
            rule="unknown",
            confidence=0.0 if best is None else best.score,
            source="unknown",
            locked=True,
            frame_time_ms=int(round(timestamp * 1000.0)),
            timestamp=timestamp,
            frame_index=frame_index,
            template_path=None if best is None else best.path,
            candidate_rule=None if best is None else best.rule,
        )

    def recognize_frames(
        self,
        frames: Iterable[tuple[np.ndarray, float, int]],
    ) -> RuleRecognitionResult:
        best_result: RuleRecognitionResult | None = None

        for frame, timestamp, frame_index in frames:
            result = self.recognize_frame(frame, timestamp=timestamp, frame_index=frame_index)
            if result.source == "template":
                return result
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        if best_result is not None:
            return best_result

        return RuleRecognitionResult(
            rule="unknown",
            confidence=0.0,
            source="unknown",
            locked=True,
            frame_time_ms=0,
            timestamp=0.0,
            frame_index=-1,
        )


def sample_video_frames(
    video_path: Path,
    start: float,
    end: float | None,
    sample_interval: float,
) -> list[tuple[np.ndarray, float, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    stop = min(duration, start if end is None else end)

    frames: list[tuple[np.ndarray, float, int]] = []
    timestamp = max(0.0, start)
    while timestamp <= stop + 1e-6:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((frame, timestamp, frame_index))
        timestamp += max(1e-6, sample_interval)

    cap.release()
    return frames


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
