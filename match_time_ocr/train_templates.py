from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from timer_ocr import DIGIT_TEMPLATE_SIZE, extract_digit_glyphs, format_seconds


DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "training_segments.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "templates" / "digits"


def resolve_path(value: str, config_path: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw

    bases = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        config_path.resolve().parent,
    ]
    for base in bases:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    return (Path(__file__).resolve().parent / raw).resolve()


def load_segments(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("training config must be a list")
    return data


def collect_digit_samples(config_path: Path, sample_step: float) -> dict[str, list[np.ndarray]]:
    samples: dict[str, list[np.ndarray]] = {digit: [] for digit in "0123456789"}

    for segment in load_segments(config_path):
        video_path = resolve_path(str(segment["video"]), config_path)
        starts_at = float(segment["starts_at"])
        start_seconds = int(segment.get("start_seconds", 300))
        duration = float(segment.get("duration", start_seconds))
        step = float(segment.get("sample_step", sample_step))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        elapsed = 0.0
        while elapsed < duration:
            timestamp = starts_at + elapsed
            expected_seconds = start_seconds - int(round(elapsed))
            if expected_seconds < 0:
                break
            expected_digits = format_seconds(expected_seconds).replace(":", "")

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
            ok, frame = cap.read()
            if not ok:
                elapsed += step
                continue

            glyphs, _mask, blobs = extract_digit_glyphs(frame)
            if len(blobs) == 3:
                for digit, glyph in zip(expected_digits, glyphs):
                    samples[digit].append(glyph)

            elapsed += step

        cap.release()

    return samples


def save_templates(samples: dict[str, list[np.ndarray]], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for digit in "0123456789":
        glyphs = samples[digit]
        counts[digit] = len(glyphs)
        if not glyphs:
            raise ValueError(f"no samples collected for digit {digit}")

        stack = np.stack(glyphs, axis=0)
        template = np.median(stack, axis=0)
        image = np.clip(template * 255.0, 0, 255).astype(np.uint8)
        if image.shape != DIGIT_TEMPLATE_SIZE:
            image = cv2.resize(image, (DIGIT_TEMPLATE_SIZE[1], DIGIT_TEMPLATE_SIZE[0]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f"{digit}.png"), image)

    return counts


def save_metadata(config_path: Path, output_dir: Path, counts: dict[str, int], sample_step: float) -> None:
    root = Path(__file__).resolve().parent
    resolved_config = config_path.resolve()
    try:
        source_config = str(resolved_config.relative_to(root))
    except ValueError:
        source_config = str(resolved_config)
    metadata = {
        "source_config": source_config,
        "template_size": list(DIGIT_TEMPLATE_SIZE),
        "sample_step": sample_step,
        "counts": counts,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Splatoon timer digit templates from captured videos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-step", type=float, default=1.0)
    args = parser.parse_args()

    samples = collect_digit_samples(args.config, args.sample_step)
    counts = save_templates(samples, args.output_dir)
    save_metadata(args.config, args.output_dir, counts, args.sample_step)
    print("saved templates:", args.output_dir)
    print("counts:", counts)


if __name__ == "__main__":
    main()
