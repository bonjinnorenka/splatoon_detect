from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

try:
    from .rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer
    from .train_templates import resolve_path
except ImportError:
    from rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer
    from train_templates import resolve_path


DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "validation_segments.json"


def load_cases(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("validation config must be a list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Splatoon 3 rule recognition samples.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD)
    args = parser.parse_args()

    recognizer = RuleRecognizer(
        templates_dir=args.templates_dir,
        accept_threshold=args.accept_threshold,
    )
    if not recognizer.templates:
        raise FileNotFoundError("rule templates are missing; run rule_detect/train_templates.py first")

    failures: list[dict[str, object]] = []
    for case in load_cases(args.config):
        video_path = resolve_path(str(case["video"]), args.config)
        expected = str(case["rule"])
        timestamp = float(case["timestamp"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read {video_path} at {timestamp:.3f}s")

        result = recognizer.recognize_frame(frame, timestamp=timestamp, frame_index=frame_index)
        passed = result.rule == expected
        row = {
            "expected": expected,
            "actual": result.rule,
            "confidence": round(result.confidence, 4),
            "source": result.source,
            "timestamp": timestamp,
            "video": str(video_path),
        }
        print(json.dumps(row, ensure_ascii=False))
        if not passed:
            failures.append(row)

    if failures:
        raise SystemExit(f"validation failed: {len(failures)} sample(s)")

    print("validation passed")


if __name__ == "__main__":
    main()
