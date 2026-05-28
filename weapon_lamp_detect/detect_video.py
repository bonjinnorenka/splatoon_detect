from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

try:
    from .weapon_lamp import WeaponIconMatcher, crop_slot, iter_video_frames, write_predictions_csv, write_predictions_jsonl
except ImportError:
    from weapon_lamp import WeaponIconMatcher, crop_slot, iter_video_frames, write_predictions_csv, write_predictions_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_WEAPON_DIR = REPO_ROOT / "sample_data" / "Main Weapons"


def _load_squid_detector(ally_side: str):
    private_dir = SCRIPT_DIR.parent
    if str(private_dir) not in sys.path:
        sys.path.append(str(private_dir))
    try:
        from squid_lamp_detect.squid_lamp import SquidLampDetector

        return SquidLampDetector(ally_side=ally_side)  # type: ignore[arg-type]
    except Exception:
        return None


def _slot_states_from_reading(reading) -> dict[str, list[str]]:
    states: dict[str, list[str]] = {"left": [], "right": []}
    if reading is None:
        return states
    for side in ("left", "right"):
        side_reading = reading.sides.get(side)
        if side_reading is None:
            continue
        states[side] = [slot.state for slot in side_reading.slots]
    return states


def _draw_debug(frame, row: dict[str, object], output_path: Path) -> None:
    out = frame.copy()
    for prediction in row.get("slots", []):  # type: ignore[union-attr]
        if not isinstance(prediction, dict):
            continue
        side = prediction.get("side")
        index = prediction.get("index")
        if side not in {"left", "right"} or not isinstance(index, int):
            continue
        crop, rect = crop_slot(frame, side, index)
        _ = crop
        x1, y1, x2, y2 = rect
        color = (0, 220, 0) if prediction.get("state") == "alive" else (0, 190, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(prediction.get("weapon") or "unknown")
        conf = float(prediction.get("confidence") or 0.0)
        text = f"{side[0]}{index} {label[:18]} {conf:.2f}"
        cv2.putText(out, text, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_h = max(1, int(round(out.shape[0] * 0.22)))
    cv2.imwrite(str(output_path), out[:crop_h])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict main weapons from Splatoon 3 squid-lamp slots.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_WEAPON_DIR)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-max-frames", type=int, default=20)
    args = parser.parse_args()

    matcher = WeaponIconMatcher.from_dir(args.template_dir)
    squid_detector = _load_squid_detector(args.ally_side)
    rows: list[dict[str, object]] = []
    debug_count = 0

    for timestamp, frame_index, frame in iter_video_frames(
        args.video,
        sample_interval=args.sample_interval,
        start=args.start,
        end=args.end,
    ):
        reading = squid_detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index) if squid_detector is not None else None
        states = _slot_states_from_reading(reading)
        predictions = matcher.predict_frame(frame, states=states, top_k=max(1, args.top_k))
        row = {
            "timestamp": round(float(timestamp), 4),
            "timestamp_ms": int(round(timestamp * 1000)),
            "frame_index": int(frame_index),
            "hud_state": getattr(reading, "hud_state", "unknown"),
            "slots": [prediction.to_dict() for prediction in predictions],
        }
        rows.append(row)
        if args.debug_dir is not None and debug_count < max(0, args.debug_max_frames):
            stem = args.video.stem.replace(" ", "_")
            debug_path = args.debug_dir / f"{stem}_{int(round(timestamp * 1000)):08d}ms.jpg"
            _draw_debug(frame, row, debug_path)
            debug_count += 1

    if args.jsonl_path is not None:
        write_predictions_jsonl(args.jsonl_path, rows)
    if args.csv_path is not None:
        write_predictions_csv(args.csv_path, rows)
    if args.jsonl_path is None and args.csv_path is None:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))

    print(f"readings={len(rows)} templates={len(matcher.templates)} variants={len(matcher.variants)}", file=sys.stderr)


if __name__ == "__main__":
    main()
