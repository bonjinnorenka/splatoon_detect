from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    from .squid_lamp import SquidLampDetector, SquidLampTracker, draw_debug_overlay
except ImportError:
    from squid_lamp import SquidLampDetector, SquidLampTracker, draw_debug_overlay


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ANNOTATIONS = SCRIPT_DIR / "data" / "eval_samples" / "annotations.jsonl"


@dataclass
class Metric:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total

    def to_dict(self) -> dict[str, int | float | None]:
        accuracy = self.accuracy
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": None if accuracy is None else round(float(accuracy), 4),
        }


def load_annotations(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line"] = line_number
            items.append(item)
    return items


def read_video_frame(video_path: Path, timestamp: float) -> tuple[Any, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_index = int(round(timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read {video_path} at {timestamp}s")
    return frame, frame_index


def get_slot_special(slot: Any) -> bool | None:
    return getattr(slot, "special_ready", None)


def evaluate(
    annotations: list[dict[str, Any]],
    annotation_path: Path,
    ally_side: str,
    smooth: bool,
    debug_dir: Path | None,
    extract_images: bool,
) -> dict[str, Any]:
    detector = SquidLampDetector(ally_side=ally_side)  # type: ignore[arg-type]
    tracker = SquidLampTracker(detector) if smooth else None
    metrics = {
        "hud": Metric(),
        "slot": Metric(),
        "enemy_special": Metric(),
    }
    mistakes: list[dict[str, Any]] = []

    image_root = annotation_path.parent
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for item in annotations:
        video_path = REPO_ROOT / item["video"]
        timestamp = float(item["timestamp"])
        frame, frame_index = read_video_frame(video_path, timestamp)
        if extract_images and item.get("image"):
            image_path = image_root / item["image"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            crop_h = max(1, int(round(frame.shape[0] * 0.24)))
            cv2.imwrite(str(image_path), frame[:crop_h])

        if tracker is not None:
            reading = tracker.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        else:
            reading = detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index)

        label = item["label"]
        expected_match = label["hud_state"] == "match"
        predicted_match = reading.hud_state == "match"
        metrics["hud"].total += 1
        if expected_match == predicted_match:
            metrics["hud"].correct += 1
        else:
            mistakes.append(
                {
                    "id": item["id"],
                    "type": "hud",
                    "expected": label["hud_state"],
                    "predicted": reading.hud_state,
                    "confidence": round(float(reading.confidence), 4),
                }
            )

        if expected_match:
            for side_name in ("left", "right"):
                expected_slots = label.get(f"{side_name}_slots", [])
                predicted_side = reading.sides[side_name]
                for index, expected_state in enumerate(expected_slots):
                    if expected_state is None:
                        continue
                    predicted_state = predicted_side.slots[index].state
                    metrics["slot"].total += 1
                    if expected_state == predicted_state:
                        metrics["slot"].correct += 1
                    else:
                        mistakes.append(
                            {
                                "id": item["id"],
                                "type": "slot",
                                "side": side_name,
                                "index": index,
                                "expected": expected_state,
                                "predicted": predicted_state,
                                "confidence": round(float(predicted_side.slots[index].confidence), 4),
                            }
                        )

            enemy_side = "left" if ally_side == "right" else "right"
            expected_specials = label.get("enemy_special_ready", [])
            predicted_enemy = reading.sides[enemy_side]
            for index, expected_ready in enumerate(expected_specials):
                if expected_ready is None:
                    continue
                predicted_ready = get_slot_special(predicted_enemy.slots[index])
                metrics["enemy_special"].total += 1
                if expected_ready == predicted_ready:
                    metrics["enemy_special"].correct += 1
                else:
                    mistakes.append(
                        {
                            "id": item["id"],
                            "type": "enemy_special",
                            "side": enemy_side,
                            "index": index,
                            "expected": expected_ready,
                            "predicted": predicted_ready,
                            "confidence": round(float(getattr(predicted_enemy.slots[index], "special_confidence", 0.0)), 4),
                        }
                    )

        if debug_dir is not None:
            overlay = draw_debug_overlay(frame, reading)
            crop_h = max(1, int(round(frame.shape[0] * 0.24)))
            debug_path = debug_dir / f"{item['id']}.jpg"
            cv2.imwrite(str(debug_path), overlay[:crop_h])

    return {
        "samples": len(annotations),
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate squid lamp detector against labeled sample frames.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    result = evaluate(
        annotations,
        annotation_path=args.annotations,
        ally_side=args.ally_side,
        smooth=args.smooth,
        debug_dir=args.debug_dir,
        extract_images=args.extract_images,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
