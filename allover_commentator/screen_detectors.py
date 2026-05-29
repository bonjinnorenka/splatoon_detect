from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


BASE_WIDTH = 1920
BASE_HEIGHT = 1080
START_ROI_1080P = (760, 260, 400, 400)
DEATH_ROI_1080P = (780, 245, 340, 250)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def scale_roi(
    shape: Sequence[int],
    roi_1080p: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x, y, w, h = roi_1080p
    left = int(round(x * width / BASE_WIDTH))
    top = int(round(y * height / BASE_HEIGHT))
    right = int(round((x + w) * width / BASE_WIDTH))
    bottom = int(round((y + h) * height / BASE_HEIGHT))
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def crop_scaled_roi(frame: np.ndarray, roi_1080p: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = scale_roi(frame.shape, roi_1080p)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


@dataclass(frozen=True)
class BinaryPrediction:
    prediction: int
    probability: float
    confidence: float
    label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction": int(self.prediction),
            "probability": round(float(self.probability), 4),
            "confidence": round(float(self.confidence), 4),
            "label": self.label,
        }


class DeathDetector:
    """Python port of private/death_detect/death_detection.js."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parents[1] / "death_detect" / "death_detection_model.json"
        with path.open("r", encoding="utf-8") as file:
            self.model = json.load(file)
        self.coefficients = np.asarray(self.model["coefficients"], dtype=np.float64)
        self.mean = np.asarray(self.model["mean"], dtype=np.float64)
        self.scale = np.asarray(self.model["scale"], dtype=np.float64)
        self.intercept = float(self.model["intercept"])
        size = self.model.get("image_size", [64, 64])
        self.image_size = (int(size[0]), int(size[1]))

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        return crop_scaled_roi(frame, DEATH_ROI_1080P)

    def predict_crop(self, crop: np.ndarray) -> BinaryPrediction:
        if crop.size == 0:
            return BinaryPrediction(0, 0.0, 0.0, "Not Death")
        resized = cv2.resize(crop, self.image_size, interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        features = rgb.reshape(-1).astype(np.float64)
        standardized = (features - self.mean) / self.scale
        z = self.intercept + float(np.dot(self.coefficients, standardized))
        probability = _sigmoid(z)
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else 1.0 - probability
        label = "Death" if prediction == 1 else "Not Death"
        return BinaryPrediction(prediction, probability, confidence, label)

    def predict_frame(self, frame: np.ndarray) -> tuple[BinaryPrediction, np.ndarray, tuple[int, int, int, int]]:
        crop, rect = self.crop(frame)
        return self.predict_crop(crop), crop, rect


class StartDetector:
    """Small Python port of the browser logistic start detector."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parents[1] / "start_detect" / "logistic_model.json"
        with path.open("r", encoding="utf-8") as file:
            self.model = json.load(file)
        self.weights = np.asarray(self.model["weights"], dtype=np.float64)
        self.bias = float(self.model["bias"])
        scaler = self.model["scaler"]
        self.means = np.asarray(scaler["means"], dtype=np.float64)
        self.stds = np.asarray(scaler["stds"], dtype=np.float64)

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        return crop_scaled_roi(frame, START_ROI_1080P)

    def predict_crop(self, crop: np.ndarray) -> BinaryPrediction:
        if crop.size == 0:
            return BinaryPrediction(0, 0.0, 0.0, "not")
        resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64)
        features = self._features(gray)
        scaled = np.where(self.stds == 0.0, 0.0, (features - self.means) / self.stds)
        probability = _sigmoid(self.bias + float(np.dot(self.weights, scaled)))
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else 1.0 - probability
        label = "start" if prediction == 1 else "not"
        return BinaryPrediction(prediction, probability, confidence, label)

    def predict_frame(self, frame: np.ndarray) -> tuple[BinaryPrediction, np.ndarray, tuple[int, int, int, int]]:
        crop, rect = self.crop(frame)
        return self.predict_crop(crop), crop, rect

    @staticmethod
    def _features(gray: np.ndarray) -> np.ndarray:
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        min_value = float(np.min(gray))
        max_value = float(np.max(gray))

        edge_count = 0
        height, width = gray.shape[:2]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gradient_x = gray[y, x - 1] - gray[y, x + 1]
                gradient_y = gray[y - 1, x] - gray[y + 1, x]
                if math.hypot(float(gradient_x), float(gradient_y)) > 50:
                    edge_count += 1
        edge_density = edge_count / max(1, width * height)

        hist, _edges = np.histogram(gray, bins=8, range=(0, 256))
        hist_normalized = hist.astype(np.float64) / max(1, int(hist.sum()))
        return np.concatenate(
            [
                np.asarray([mean, std, min_value, max_value, edge_density], dtype=np.float64),
                hist_normalized,
            ]
        )
