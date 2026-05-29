from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


SideName = Literal["left", "right"]

# Same 1920x1080 Splatoon 3 HUD geometry used by squid_lamp_detect.
SLOT_CENTERS: dict[SideName, tuple[float, float, float, float]] = {
    "left": (0.265, 0.323, 0.382, 0.440),
    "right": (0.562, 0.620, 0.680, 0.735),
}
SLOT_CENTER_Y = 0.067
SLOT_WIDTH = 0.052
SLOT_HEIGHT = 0.080


@dataclass(frozen=True)
class WeaponTemplate:
    name: str
    weapon_class: str
    path: str
    bgr: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class _TemplateVariant:
    name: str
    weapon_class: str
    bgr: np.ndarray
    gray: np.ndarray
    lab: np.ndarray
    edge: np.ndarray
    mask: np.ndarray
    max_dim: int
    angle: float
    mask_area: int
    edge_norm: float


@dataclass
class MatchCandidate:
    weapon: str
    weapon_class: str
    score: float
    confidence: float
    location: tuple[int, int]
    size: tuple[int, int]
    angle: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "weapon": self.weapon,
            "weapon_class": self.weapon_class,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "location": [int(self.location[0]), int(self.location[1])],
            "size": [int(self.size[0]), int(self.size[1])],
            "angle": round(float(self.angle), 2),
            "components": {key: round(float(value), 4) for key, value in self.components.items()},
        }


@dataclass
class SlotWeaponPrediction:
    side: SideName
    index: int
    state: str
    weapon: str | None
    weapon_class: str | None
    confidence: float
    score: float
    candidates: list[MatchCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "index": self.index,
            "state": self.state,
            "weapon": self.weapon,
            "weapon_class": self.weapon_class,
            "confidence": round(float(self.confidence), 4),
            "score": round(float(self.score), 4),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def weapon_class_for_name(name: str) -> str:
    text = name.lower()
    if "blaster" in text or "s-blast" in text:
        return "blaster"
    if "roller" in text or "dynamo" in text or "flingza" in text or "big swig" in text:
        return "roller"
    if (
        "charger" in text
        or "e-liter" in text
        or "squiffer" in text
        or "bamboozler" in text
        or "goo tuber" in text
        or "snipewriter" in text
        or "splatterscope" in text
    ):
        return "charger"
    if (
        "slosher" in text
        or "bloblobber" in text
        or "explosher" in text
        or "sloshing machine" in text
        or "dread wringer" in text
    ):
        return "slosher"
    if "splatling" in text or "nautilus" in text or "ballpoint" in text or "hydra" in text:
        return "splatling"
    if "dualies" in text or "dualie" in text or "douser" in text or "tetra" in text:
        return "dualies"
    if "brella" in text:
        return "brella"
    if "brush" in text:
        return "brush"
    if "splatana" in text or "decavitator" in text:
        return "splatana"
    if "stringer" in text or "reef-lux" in text or "wellstring" in text:
        return "stringer"
    return "shooter"


def slot_rect(frame_shape: Sequence[int], side: SideName, index: int) -> tuple[int, int, int, int]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    cx = int(round(width * SLOT_CENTERS[side][index]))
    cy = int(round(height * SLOT_CENTER_Y))
    slot_w = int(round(width * SLOT_WIDTH))
    slot_h = int(round(height * SLOT_HEIGHT))
    left = max(0, cx - slot_w // 2)
    top = max(0, cy - slot_h // 2)
    right = min(width, cx + slot_w // 2)
    bottom = min(height, cy + slot_h // 2)
    return left, top, right, bottom


def expand_rect(
    rect: tuple[int, int, int, int],
    frame_shape: Sequence[int],
    pad_x: float = 1.35,
    pad_y: float = 1.25,
) -> tuple[int, int, int, int]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_w = (x2 - x1) * pad_x
    new_h = (y2 - y1) * pad_y
    left = max(0, int(round(cx - new_w * 0.5)))
    top = max(0, int(round(cy - new_h * 0.5)))
    right = min(width, int(round(cx + new_w * 0.5)))
    bottom = min(height, int(round(cy + new_h * 0.5)))
    return left, top, right, bottom


def crop_slot(
    frame: np.ndarray,
    side: SideName,
    index: int,
    pad_x: float = 1.35,
    pad_y: float = 1.25,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rect = expand_rect(slot_rect(frame.shape, side, index), frame.shape, pad_x=pad_x, pad_y=pad_y)
    x1, y1, x2, y2 = rect
    return frame[y1:y2, x1:x2], rect


def _alpha_bbox(alpha: np.ndarray, threshold: int = 16, pad: int = 6) -> tuple[int, int, int, int]:
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(alpha.shape[1], int(xs.max()) + 1 + pad)
    y2 = min(alpha.shape[0], int(ys.max()) + 1 + pad)
    return x1, y1, x2, y2


def _rotate_bound(bgr: np.ndarray, alpha: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(angle) < 1e-6:
        return bgr, alpha
    height, width = bgr.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w * 0.5 - center[0]
    matrix[1, 2] += new_h * 0.5 - center[1]
    rotated_bgr = cv2.warpAffine(bgr, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_alpha = cv2.warpAffine(alpha, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    x1, y1, x2, y2 = _alpha_bbox(rotated_alpha, threshold=8, pad=0)
    return rotated_bgr[y1:y2, x1:x2], rotated_alpha[y1:y2, x1:x2]


def _masked_multichannel_pearson(template: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 30
    if int(valid.sum()) < 20:
        return 0.0
    lhs = template[valid].astype(np.float32)
    rhs = patch[valid].astype(np.float32)
    lhs -= lhs.mean(axis=0, keepdims=True)
    rhs -= rhs.mean(axis=0, keepdims=True)
    lhs = lhs.ravel()
    rhs = rhs.ravel()
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm < 1e-6 or rhs_norm < 1e-6:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


def _edge_cosine(template_edge: np.ndarray, patch_edge: np.ndarray, template_norm: float) -> float:
    if template_norm < 1e-6:
        return 0.0
    lhs = template_edge.astype(np.float32)
    rhs = patch_edge.astype(np.float32)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm < 1e-6:
        return 0.0
    return float(np.dot(lhs.ravel(), rhs.ravel()) / (template_norm * rhs_norm))


def _masked_match_template(image: np.ndarray, template: np.ndarray, mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    try:
        result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask)
    except cv2.error:
        return 0.0, (0, 0)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(result)
    if not math.isfinite(max_value):
        return 0.0, (0, 0)
    return float(max(0.0, max_value)), (int(max_location[0]), int(max_location[1]))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class WeaponIconMatcher:
    def __init__(
        self,
        templates: list[WeaponTemplate],
        variant_sizes: Sequence[int] = (62, 82),
        variant_angles: Sequence[float] = (-8.0, 0.0, 8.0),
    ) -> None:
        self.templates = templates
        self.variant_sizes = tuple(int(value) for value in variant_sizes)
        self.variant_angles = tuple(float(value) for value in variant_angles)
        self.variants = self._build_variants()

    @classmethod
    def from_dir(
        cls,
        template_dir: Path,
        variant_sizes: Sequence[int] = (62, 82),
        variant_angles: Sequence[float] = (-8.0, 0.0, 8.0),
    ) -> "WeaponIconMatcher":
        templates: list[WeaponTemplate] = []
        for path in sorted(template_dir.glob("*.png")):
            rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            alpha = rgba[:, :, 3]
            x1, y1, x2, y2 = _alpha_bbox(alpha)
            bgr = rgba[y1:y2, x1:x2, :3]
            cropped_alpha = alpha[y1:y2, x1:x2]
            name = path.stem
            templates.append(
                WeaponTemplate(
                    name=name,
                    weapon_class=weapon_class_for_name(name),
                    path=str(path),
                    bgr=bgr,
                    alpha=cropped_alpha,
                )
            )
        if not templates:
            raise FileNotFoundError(f"no weapon templates found in {template_dir}")
        return cls(templates, variant_sizes=variant_sizes, variant_angles=variant_angles)

    def _build_variants(self) -> list[_TemplateVariant]:
        variants: list[_TemplateVariant] = []
        for template in self.templates:
            for angle in self.variant_angles:
                rotated_bgr, rotated_alpha = _rotate_bound(template.bgr, template.alpha, angle)
                height, width = rotated_bgr.shape[:2]
                base_max = max(height, width)
                if base_max <= 0:
                    continue
                for max_dim in self.variant_sizes:
                    scale = max_dim / base_max
                    new_w = max(6, int(round(width * scale)))
                    new_h = max(6, int(round(height * scale)))
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    bgr = cv2.resize(rotated_bgr, (new_w, new_h), interpolation=interpolation)
                    mask = cv2.resize(rotated_alpha, (new_w, new_h), interpolation=interpolation)
                    mask_u8 = (mask > 24).astype(np.uint8) * 255
                    if int(mask_u8.sum()) < 255 * 30:
                        continue
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    edge = cv2.Canny(gray, 45, 145)
                    edge = cv2.bitwise_and(edge, edge, mask=mask_u8)
                    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
                    if int((edge > 0).sum()) < 12:
                        continue
                    edge_norm = float(np.linalg.norm(edge.astype(np.float32)))
                    variants.append(
                        _TemplateVariant(
                            name=template.name,
                            weapon_class=template.weapon_class,
                            bgr=bgr,
                            gray=gray,
                            lab=lab,
                            edge=edge,
                            mask=mask_u8,
                            max_dim=int(max_dim),
                            angle=float(angle),
                            mask_area=int((mask_u8 > 0).sum()),
                            edge_norm=edge_norm,
                        )
                    )
        return variants

    def predict_crop(self, crop: np.ndarray, top_k: int = 5) -> list[MatchCandidate]:
        if crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        image_edge = cv2.Canny(gray, 45, 145)
        crop_h, crop_w = gray.shape[:2]
        crop_area = max(1, crop_h * crop_w)
        target_dim = min(crop_h, crop_w) * 0.68

        best_by_weapon: dict[str, MatchCandidate] = {}
        for variant in self.variants:
            tpl_h, tpl_w = variant.gray.shape[:2]
            if tpl_h > crop_h or tpl_w > crop_w:
                continue
            gray_corr, (x, y) = _masked_match_template(gray, variant.gray, variant.mask)
            patch_edge = image_edge[y : y + tpl_h, x : x + tpl_w]
            patch_lab = lab[y : y + tpl_h, x : x + tpl_w]
            shape_score = _edge_cosine(variant.edge, patch_edge, variant.edge_norm)
            if not math.isfinite(shape_score):
                continue
            color_corr = max(0.0, _masked_multichannel_pearson(variant.lab, patch_lab, variant.mask))
            size_sigma = max(8.0, target_dim * 0.35)
            scale_prior = math.exp(-((variant.max_dim - target_dim) ** 2) / (2.0 * size_sigma * size_sigma))
            match_cx = x + tpl_w * 0.5
            match_cy = y + tpl_h * 0.5
            dist = math.hypot((match_cx - crop_w * 0.5) / crop_w, (match_cy - crop_h * 0.50) / crop_h)
            location_prior = 1.0 - min(1.0, dist / 0.72)
            area_prior = min(1.0, variant.mask_area / max(1.0, crop_area * 0.045))
            score = (
                0.40 * float(gray_corr)
                + 0.25 * float(color_corr)
                + 0.15 * float(max(0.0, shape_score))
                + 0.12 * float(scale_prior)
                + 0.08 * float(location_prior)
            )
            score *= 0.82 + 0.18 * area_prior
            score = _clamp01(score)
            candidate = MatchCandidate(
                weapon=variant.name,
                weapon_class=variant.weapon_class,
                score=score,
                confidence=score,
                location=(int(x), int(y)),
                size=(int(tpl_w), int(tpl_h)),
                angle=variant.angle,
                components={
                    "shape": float(shape_score),
                    "gray_corr": float(gray_corr),
                    "color_corr": float(color_corr),
                    "scale_prior": float(scale_prior),
                    "location_prior": float(location_prior),
                    "area_prior": float(area_prior),
                },
            )
            previous = best_by_weapon.get(candidate.weapon)
            if previous is None or candidate.score > previous.score:
                best_by_weapon[candidate.weapon] = candidate

        candidates = sorted(best_by_weapon.values(), key=lambda item: item.score, reverse=True)[: max(1, top_k)]
        if candidates:
            second = candidates[1].score if len(candidates) > 1 else 0.0
            margin = max(0.0, candidates[0].score - second)
            for candidate in candidates:
                candidate.confidence = _clamp01(0.58 * candidate.score + 2.2 * margin)
        return candidates

    def predict_slot(
        self,
        crop: np.ndarray,
        side: SideName,
        index: int,
        state: str = "unknown",
        top_k: int = 5,
    ) -> SlotWeaponPrediction:
        candidates = self.predict_crop(crop, top_k=top_k)
        if not candidates:
            return SlotWeaponPrediction(side, index, state, None, None, 0.0, 0.0, [])
        best = candidates[0]
        return SlotWeaponPrediction(
            side=side,
            index=index,
            state=state,
            weapon=best.weapon,
            weapon_class=best.weapon_class,
            confidence=best.confidence,
            score=best.score,
            candidates=candidates,
        )

    def predict_frame(
        self,
        frame: np.ndarray,
        states: dict[str, list[str]] | None = None,
        top_k: int = 5,
        pad_x: float = 1.35,
        pad_y: float = 1.25,
    ) -> list[SlotWeaponPrediction]:
        predictions: list[SlotWeaponPrediction] = []
        for side in ("left", "right"):
            for index in range(4):
                state = "unknown"
                if states is not None:
                    side_states = states.get(side, [])
                    if index < len(side_states):
                        state = side_states[index]
                crop, _rect = crop_slot(frame, side, index, pad_x=pad_x, pad_y=pad_y)
                predictions.append(self.predict_slot(crop, side, index, state=state, top_k=top_k))
        return predictions


def iter_video_frames(
    video_path: Path,
    sample_interval: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
) -> Iterable[tuple[float, int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        stop = duration if end is None else min(end, duration)
        timestamp = max(0.0, start)
        while timestamp <= stop + 1e-6:
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            yield timestamp, frame_index, frame
            timestamp += sample_interval
    finally:
        cap.release()


def write_predictions_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_predictions_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "timestamp_ms",
        "frame_index",
        "hud_state",
        "side",
        "slot_index",
        "state",
        "weapon",
        "weapon_class",
        "confidence",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for prediction in row.get("slots", []):  # type: ignore[union-attr]
                if not isinstance(prediction, dict):
                    continue
                writer.writerow(
                    {
                        "timestamp": row.get("timestamp"),
                        "timestamp_ms": row.get("timestamp_ms"),
                        "frame_index": row.get("frame_index"),
                        "hud_state": row.get("hud_state"),
                        "side": prediction.get("side"),
                        "slot_index": prediction.get("index"),
                        "state": prediction.get("state"),
                        "weapon": prediction.get("weapon"),
                        "weapon_class": prediction.get("weapon_class"),
                        "confidence": prediction.get("confidence"),
                        "score": prediction.get("score"),
                    }
                )


def prediction_to_json(data: object) -> str:
    return json.dumps(asdict(data), ensure_ascii=False)
