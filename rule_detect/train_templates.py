from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

try:
    from .rule_recognizer import DEFAULT_TEMPLATE_WIDTH, RULES, crop_and_preprocess_rule_roi, write_json
except ImportError:
    from rule_recognizer import DEFAULT_TEMPLATE_WIDTH, RULES, crop_and_preprocess_rule_roi, write_json


DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "training_segments.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "templates" / "jp_1080p"


def resolve_path(value: str, config_path: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw

    bases = (
        Path.cwd(),
        Path(__file__).resolve().parent,
        config_path.resolve().parent,
    )
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


def _timestamps(segment: dict[str, Any]) -> list[float]:
    if "timestamps" in segment:
        values = segment["timestamps"]
        if not isinstance(values, list):
            raise ValueError("timestamps must be a list")
        return [float(value) for value in values]
    return [float(segment["timestamp"])]


def generate_templates(config_path: Path, output_dir: Path, output_width: int) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_template in output_dir.glob("*.png"):
        old_template.unlink()
    counts: dict[str, int] = {rule: 0 for rule in RULES}

    for segment in load_segments(config_path):
        rule = str(segment["rule"])
        if rule not in RULES:
            raise ValueError(f"unknown rule: {rule}")
        video_path = resolve_path(str(segment["video"]), config_path)
        timestamps = _timestamps(segment)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        for timestamp in timestamps:
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read {video_path} at {timestamp:.3f}s")

            image = crop_and_preprocess_rule_roi(frame, output_width=output_width)
            counts[rule] += 1
            output_path = output_dir / f"{rule}_{counts[rule]:02d}.png"
            cv2.imwrite(str(output_path), image)

        cap.release()

    metadata = {
        "source_config": str(config_path.resolve()),
        "template_width": output_width,
        "counts": counts,
    }
    write_json(output_dir / "metadata.json", metadata)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Splatoon 3 rule title templates from labeled start scenes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-width", type=int, default=DEFAULT_TEMPLATE_WIDTH)
    args = parser.parse_args()

    counts = generate_templates(args.config, args.output_dir, args.output_width)
    print("saved templates:", args.output_dir)
    print("counts:", counts)


if __name__ == "__main__":
    main()
