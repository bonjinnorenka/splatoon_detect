from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

try:
    from .squid_lamp import SquidLampDetector, draw_debug_overlay, sample_video, write_csv, write_jsonl
except ImportError:
    from squid_lamp import SquidLampDetector, draw_debug_overlay, sample_video, write_csv, write_jsonl


def save_debug_frames(
    video_path: Path,
    readings,
    debug_dir: Path,
    max_frames: int,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    saved = 0
    for reading in readings:
        if saved >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(reading.frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        overlay = draw_debug_overlay(frame, reading)
        stem = video_path.stem.replace(" ", "_")
        path = debug_dir / f"{stem}_{reading.timestamp_ms:08d}ms.jpg"
        cv2.imwrite(str(path), overlay)
        saved += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Splatoon 3 squid lamps and match ink colors from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-max-frames", type=int, default=20)
    args = parser.parse_args()

    detector = SquidLampDetector(ally_side=args.ally_side)
    readings = sample_video(
        args.video,
        detector,
        sample_interval=args.sample_interval,
        start=args.start,
        end=args.end,
        smooth=not args.no_smooth,
    )

    if args.csv_path is not None:
        write_csv(args.csv_path, readings)
    if args.jsonl_path is not None:
        write_jsonl(args.jsonl_path, readings)
    if args.debug_dir is not None:
        save_debug_frames(args.video, readings, args.debug_dir, max(0, args.debug_max_frames))

    if args.csv_path is None and args.jsonl_path is None:
        for reading in readings:
            print(json.dumps(reading.to_dict(), ensure_ascii=False))

    match_count = sum(1 for reading in readings if reading.hud_state == "match")
    known_count = sum(1 for reading in readings if reading.ally_alive is not None and reading.enemy_alive is not None)
    print(f"readings={len(readings)} match_hud={match_count} known_counts={known_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
