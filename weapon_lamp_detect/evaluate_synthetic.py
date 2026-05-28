from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .weapon_lamp import WeaponIconMatcher, WeaponTemplate, weapon_class_for_name
except ImportError:
    from weapon_lamp import WeaponIconMatcher, WeaponTemplate, weapon_class_for_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "sample_data" / "Main Weapons"


@dataclass
class Metric:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total

    def add(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.correct += 1

    def to_dict(self) -> dict[str, int | float | None]:
        accuracy = self.accuracy
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": None if accuracy is None else round(float(accuracy), 4),
        }


def _resize_rgba(template: WeaponTemplate, max_dim: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = template.bgr.shape[:2]
    scale = max_dim / max(height, width)
    new_w = max(6, int(round(width * scale)))
    new_h = max(6, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return (
        cv2.resize(template.bgr, (new_w, new_h), interpolation=interpolation),
        cv2.resize(template.alpha, (new_w, new_h), interpolation=interpolation),
    )


def _rotate_rgba(bgr: np.ndarray, alpha: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = bgr.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w * 0.5 - center[0]
    matrix[1, 2] += new_h * 0.5 - center[1]
    out_bgr = cv2.warpAffine(bgr, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    out_alpha = cv2.warpAffine(alpha, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.where(out_alpha > 8)
    if len(xs) == 0:
        return out_bgr, out_alpha
    return out_bgr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], out_alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    dst_h, dst_w = dst.shape[:2]
    src_h, src_w = src.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(dst_w, x + src_w)
    y2 = min(dst_h, y + src_h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    alpha_f = (alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0)[:, :, None]
    dst[y1:y2, x1:x2] = (src[sy1:sy2, sx1:sx2].astype(np.float32) * alpha_f + dst[y1:y2, x1:x2].astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)


def render_synthetic_lamp(
    template: WeaponTemplate,
    rng: np.random.Generator,
    crop_size: tuple[int, int] = (128, 112),
    down: bool = False,
) -> np.ndarray:
    width, height = crop_size
    hue_color = rng.integers(0, 180)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue_color
    hsv[:, :, 1] = rng.integers(145, 235)
    hsv[:, :, 2] = rng.integers(145, 245)
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Simple squid-lamp-like blocks behind the weapon.
    cv2.rectangle(
        crop,
        (int(width * rng.uniform(0.00, 0.20)), int(height * rng.uniform(0.20, 0.48))),
        (int(width * rng.uniform(0.18, 0.42)), int(height * rng.uniform(0.75, 0.98))),
        (16, 16, 16),
        -1,
    )
    if rng.random() < 0.55:
        color = tuple(int(v) for v in cv2.cvtColor(np.array([[[hue_color, 210, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0])
        points = np.array(
            [
                [int(width * rng.uniform(0.20, 0.40)), int(height * rng.uniform(0.05, 0.18))],
                [int(width * rng.uniform(0.62, 0.95)), int(height * rng.uniform(0.25, 0.55))],
                [int(width * rng.uniform(0.18, 0.42)), int(height * rng.uniform(0.85, 1.02))],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(crop, points, color)

    icon_bgr, icon_alpha = _resize_rgba(template, float(rng.uniform(54, 88)))
    icon_bgr, icon_alpha = _rotate_rgba(icon_bgr, icon_alpha, float(rng.uniform(-11, 11)))
    contrast = float(rng.uniform(0.82, 1.14))
    brightness = float(rng.uniform(-13, 13))
    icon_bgr = np.clip(icon_bgr.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    icon_h, icon_w = icon_bgr.shape[:2]
    center_x = int(round(width * rng.uniform(0.42, 0.58)))
    center_y = int(round(height * rng.uniform(0.42, 0.57)))
    x = center_x - icon_w // 2 + int(round(rng.normal(0, width * 0.07)))
    y = center_y - icon_h // 2 + int(round(rng.normal(0, height * 0.06)))
    _alpha_blend(crop, icon_bgr, icon_alpha, x, y)

    if rng.random() < 0.35:
        bx = int(width * rng.uniform(0.06, 0.74))
        by = int(height * rng.uniform(0.62, 0.82))
        cv2.rectangle(crop, (bx, by), (min(width - 1, bx + 24), min(height - 1, by + 16)), (245, 245, 245), -1)
        cv2.rectangle(crop, (bx + 2, by + 2), (min(width - 1, bx + 22), min(height - 1, by + 14)), (30, 30, 30), 1)

    if down:
        thickness = int(rng.integers(7, 11))
        gray = int(rng.integers(125, 175))
        cv2.line(crop, (int(width * 0.18), int(height * 0.12)), (int(width * 0.82), int(height * 0.90)), (gray, gray, gray), thickness, cv2.LINE_AA)
        cv2.line(crop, (int(width * 0.82), int(height * 0.12)), (int(width * 0.18), int(height * 0.90)), (gray, gray, gray), thickness, cv2.LINE_AA)

    if rng.random() < 0.40:
        crop = cv2.GaussianBlur(crop, (3, 3), float(rng.uniform(0.15, 0.55)))
    noise = rng.normal(0, rng.uniform(1.0, 4.0), crop.shape)
    crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.60:
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(68, 92))])
        if ok:
            crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return crop


def evaluate(
    matcher: WeaponIconMatcher,
    samples_per_weapon: int,
    down_fraction: float,
    seed: int,
    top_k: int,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metrics = {
        "exact_top1": Metric(),
        "exact_top3": Metric(),
        "exact_top5": Metric(),
        "class_top1": Metric(),
        "class_top3": Metric(),
        "class_top5": Metric(),
    }
    by_state = {
        "alive": {"exact_top1": Metric(), "class_top1": Metric()},
        "down": {"exact_top1": Metric(), "class_top1": Metric()},
    }
    mistakes: list[dict[str, Any]] = []
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    sample_index = 0
    for template in matcher.templates:
        expected_class = weapon_class_for_name(template.name)
        for _ in range(samples_per_weapon):
            down = bool(rng.random() < down_fraction)
            state = "down" if down else "alive"
            crop = render_synthetic_lamp(template, rng, down=down)
            candidates = matcher.predict_crop(crop, top_k=max(5, top_k))
            names = [candidate.weapon for candidate in candidates]
            classes = [candidate.weapon_class for candidate in candidates]
            exact_top1 = len(names) > 0 and names[0] == template.name
            class_top1 = len(classes) > 0 and classes[0] == expected_class
            metrics["exact_top1"].add(exact_top1)
            metrics["exact_top3"].add(template.name in names[:3])
            metrics["exact_top5"].add(template.name in names[:5])
            metrics["class_top1"].add(class_top1)
            metrics["class_top3"].add(expected_class in classes[:3])
            metrics["class_top5"].add(expected_class in classes[:5])
            by_state[state]["exact_top1"].add(exact_top1)
            by_state[state]["class_top1"].add(class_top1)
            if not exact_top1 and len(mistakes) < 80:
                mistakes.append(
                    {
                        "expected": template.name,
                        "expected_class": expected_class,
                        "state": state,
                        "predicted": names[0] if names else None,
                        "predicted_class": classes[0] if classes else None,
                        "top5": [
                            {
                                "weapon": candidate.weapon,
                                "weapon_class": candidate.weapon_class,
                                "score": round(candidate.score, 4),
                                "confidence": round(candidate.confidence, 4),
                            }
                            for candidate in candidates[:5]
                        ],
                    }
                )
                if debug_dir is not None:
                    cv2.imwrite(str(debug_dir / f"mistake_{sample_index:05d}_{template.name}.jpg"), crop)
            sample_index += 1

    return {
        "templates": len(matcher.templates),
        "synthetic_samples": sample_index,
        "samples_per_weapon": samples_per_weapon,
        "down_fraction": down_fraction,
        "seed": seed,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "by_state": {
            state: {name: metric.to_dict() for name, metric in state_metrics.items()}
            for state, state_metrics in by_state.items()
        },
        "mistakes": mistakes,
        "note": "Synthetic samples are generated from sample_data templates, so this measures template-matching robustness, not true video accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weapon-lamp template matching on synthetic squid-lamp crops.")
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--samples-per-weapon", type=int, default=2)
    parser.add_argument("--down-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    args = parser.parse_args()

    matcher = WeaponIconMatcher.from_dir(args.template_dir)
    result = evaluate(
        matcher,
        samples_per_weapon=max(1, args.samples_per_weapon),
        down_fraction=max(0.0, min(1.0, args.down_fraction)),
        seed=args.seed,
        top_k=max(1, args.top_k),
        debug_dir=args.debug_dir,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
