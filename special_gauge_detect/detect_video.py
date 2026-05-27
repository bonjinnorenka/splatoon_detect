from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

try:
    from .special_gauge import (
        ROIConfig,
        SpecialGaugeDetector,
        draw_debug_overlay,
        load_death_events,
        sample_video,
        write_csv,
        write_jsonl,
    )
except ImportError:
    from special_gauge import (
        ROIConfig,
        SpecialGaugeDetector,
        draw_debug_overlay,
        load_death_events,
        sample_video,
        write_csv,
        write_jsonl,
    )


def save_debug_frames(video_path: Path, updates, debug_dir: Path, max_frames: int) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    saved = 0
    for update in updates:
        if saved >= max_frames:
            break
        reading = update.special_gauge
        if reading.state == "unknown" and not update.events:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(reading.frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        overlay = draw_debug_overlay(frame, reading)
        stem = video_path.stem.replace(" ", "_")
        path = debug_dir / f"{stem}_{reading.timestamp_ms:08d}ms_{reading.state}.jpg"
        cv2.imwrite(str(path), overlay)
        saved += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the local player's Splatoon 3 special gauge state.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--roi-x", type=float, default=ROIConfig.x_ratio)
    parser.add_argument("--roi-y", type=float, default=ROIConfig.y_ratio)
    parser.add_argument("--roi-w", type=float, default=ROIConfig.w_ratio)
    parser.add_argument("--roi-h", type=float, default=ROIConfig.h_ratio)
    parser.add_argument("--ready-threshold", type=float, default=0.58)
    parser.add_argument("--visible-threshold", type=float, default=0.43)
    parser.add_argument("--disable-near-ready", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--death-events", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path, default=None)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-max-frames", type=int, default=30)
    args = parser.parse_args()

    roi = ROIConfig(args.roi_x, args.roi_y, args.roi_w, args.roi_h)
    detector = SpecialGaugeDetector(
        roi=roi,
        ready_threshold=args.ready_threshold,
        visible_threshold=args.visible_threshold,
        enable_near_ready=not args.disable_near_ready,
    )
    death_events = load_death_events(args.death_events) if args.death_events is not None else None
    updates = sample_video(
        args.video,
        detector=detector,
        sample_interval=args.sample_interval,
        start=args.start,
        end=args.end,
        smooth=not args.no_smooth,
        death_events=death_events,
    )

    if args.csv_path is not None:
        write_csv(args.csv_path, updates)
    if args.jsonl_path is not None:
        write_jsonl(args.jsonl_path, updates, include_debug=args.include_debug)
    if args.debug_dir is not None:
        save_debug_frames(args.video, updates, args.debug_dir, max(0, args.debug_max_frames))

    if args.csv_path is None and args.jsonl_path is None:
        for update in updates:
            print(json.dumps(update.to_dict(include_debug=args.include_debug), ensure_ascii=False))

    states = {}
    for update in updates:
        state = update.special_gauge.state
        states[state] = states.get(state, 0) + 1
    event_count = sum(len(update.events) for update in updates)
    print(f"readings={len(updates)} states={states} events={event_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
