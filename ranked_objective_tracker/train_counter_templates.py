from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .ranked_objective import (
        DEFAULT_TEMPLATE_DIR,
        DIGIT_TEMPLATE_SIZE,
        SIDE_COUNT_ROI_CANDIDATES,
        TrackerContext,
        crop_rel,
        extract_number_blobs,
        make_counter_text_masks,
        normalize_glyph,
    )
except ImportError:
    from ranked_objective import (
        DEFAULT_TEMPLATE_DIR,
        DIGIT_TEMPLATE_SIZE,
        SIDE_COUNT_ROI_CANDIDATES,
        TrackerContext,
        crop_rel,
        extract_number_blobs,
        make_counter_text_masks,
        normalize_glyph,
    )


DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "data" / "eval_samples" / "annotations.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "templates" / "counter_digits"


def add_glyph_samples(samples: dict[str, list[np.ndarray]], expected_text: str, glyphs: list[np.ndarray]) -> int:
    if len(glyphs) != len(expected_text):
        return 0
    added = 0
    for digit, glyph in zip(expected_text, glyphs):
        samples[digit].append(glyph)
        added += 1
    return added


def extract_training_glyphs(mask: np.ndarray, expected_text: str) -> list[np.ndarray]:
    blobs = extract_number_blobs(mask, max_digits=3)
    if len(blobs) == len(expected_text):
        return [normalize_glyph(mask, blob) for blob in blobs]
    return []


def load_annotations(path: Path) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    if not path.exists():
        return annotations
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        annotations.append(dict(json.loads(line)))
    return annotations


def side_counts(annotation: dict[str, object]) -> dict[str, int | None]:
    expected = annotation.get("expected")
    if not isinstance(expected, dict):
        return {"left": None, "right": None}
    side_data = expected.get("sideCounts")
    if isinstance(side_data, dict):
        return {
            "left": int(side_data["left"]) if isinstance(side_data.get("left"), int) else None,
            "right": int(side_data["right"]) if isinstance(side_data.get("right"), int) else None,
        }

    counter = expected.get("counter")
    if not isinstance(counter, dict):
        return {"left": None, "right": None}
    context = TrackerContext.from_mapping(annotation)
    our = counter.get("ourCount")
    enemy = counter.get("enemyCount")
    if not isinstance(our, int) or not isinstance(enemy, int):
        return {"left": None, "right": None}
    if context.ally_side == "right":
        return {"left": enemy, "right": our}
    return {"left": our, "right": enemy}


def annotation_image_path(annotation: dict[str, object], annotations_path: Path) -> Path | None:
    image = annotation.get("image")
    if not isinstance(image, str):
        return None
    path = Path(image)
    if path.is_absolute():
        return path
    return annotations_path.parent / path


def collect_samples(annotations: Iterable[dict[str, object]], annotations_path: Path, splits: set[str]) -> dict[str, list[np.ndarray]]:
    samples: dict[str, list[np.ndarray]] = {digit: [] for digit in "0123456789"}
    for annotation in annotations:
        split = str(annotation.get("split", "eval"))
        if split not in splits:
            continue
        image_path = annotation_image_path(annotation, annotations_path)
        if image_path is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        counts = side_counts(annotation)
        rule = str(annotation.get("rule", "rule_unknown"))
        expected = annotation.get("expected")
        count_bboxes = expected.get("countBboxes") if isinstance(expected, dict) else None
        for side, value in counts.items():
            if value is None:
                continue
            expected_text = str(int(value))
            bbox_sample_count = 0
            if isinstance(count_bboxes, dict) and isinstance(count_bboxes.get(side), list):
                x1, y1, x2, y2 = [int(v) for v in count_bboxes[side]]
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                for _mask_name, mask in make_counter_text_masks(crop):
                    bbox_sample_count += add_glyph_samples(samples, expected_text, extract_training_glyphs(mask, expected_text))
            if bbox_sample_count:
                continue
            for name, roi in SIDE_COUNT_ROI_CANDIDATES[side]:  # type: ignore[index]
                if rule == "tower_control":
                    if name != "tower":
                        continue
                elif name != "standard_plate":
                    continue
                crop, _rect = crop_rel(frame, roi)
                for _mask_name, mask in make_counter_text_masks(crop):
                    glyphs = extract_training_glyphs(mask, expected_text)
                    add_glyph_samples(samples, expected_text, glyphs)
                    if rule == "tower_control":
                        blobs = extract_number_blobs(mask, max_digits=3)
                        if len(blobs) == len(expected_text):
                            for digit, blob in zip(expected_text, blobs):
                                samples[digit].append(normalize_glyph(mask, blob))
    return samples


def load_fallback_templates(path: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for digit in "0123456789":
        image = cv2.imread(str(path / f"{digit}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if image.shape != DIGIT_TEMPLATE_SIZE:
            image = cv2.resize(image, (DIGIT_TEMPLATE_SIZE[1], DIGIT_TEMPLATE_SIZE[0]), interpolation=cv2.INTER_AREA)
        templates[digit] = image.astype(np.float32) / 255.0
    return templates


def save_templates(samples: dict[str, list[np.ndarray]], output_dir: Path, fallback_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback = load_fallback_templates(fallback_dir)
    counts: dict[str, int] = {}
    for digit in "0123456789":
        glyphs = samples[digit]
        counts[digit] = len(glyphs)
        if glyphs:
            stack = np.stack(glyphs, axis=0)
            template = np.median(stack, axis=0)
        elif digit in fallback:
            template = fallback[digit]
        else:
            template = np.zeros(DIGIT_TEMPLATE_SIZE, dtype=np.float32)
        image = np.clip(template * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{digit}.png"), image)

    metadata = {
        "source": str(DEFAULT_ANNOTATIONS),
        "template_size": list(DIGIT_TEMPLATE_SIZE),
        "counts": counts,
        "fallback_dir": str(fallback_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ranked counter digit templates from annotated frames.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fallback-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--splits", default="train", help="Comma separated annotation splits to use.")
    args = parser.parse_args()

    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    annotations = load_annotations(args.annotations)
    samples = collect_samples(annotations, args.annotations, splits)
    counts = save_templates(samples, args.output_dir, args.fallback_dir)
    print(json.dumps({"annotations": len(annotations), "splits": sorted(splits), "counts": counts, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
