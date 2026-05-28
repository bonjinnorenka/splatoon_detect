from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

try:
    from .ranked_objective import (
        RankedObjectiveDetector,
        TrackerContext,
        draw_debug_overlay,
        sample_video,
        write_csv,
        write_jsonl,
    )
except ImportError:
    from ranked_objective import (
        RankedObjectiveDetector,
        TrackerContext,
        draw_debug_overlay,
        sample_video,
        write_csv,
        write_jsonl,
    )


def load_context(path: Path | None, rule: str, ally_side: str, time_left_sec: int | None) -> TrackerContext:
    data: dict[str, object] = {}
    if path is not None:
        data = dict(json.loads(path.read_text(encoding="utf-8")))
    data.setdefault("rule", rule)
    data.setdefault("ally_side", ally_side)
    data.setdefault("match_started", True)
    if time_left_sec is not None:
        data["time_left_sec"] = time_left_sec
    return TrackerContext.from_mapping(data)


def save_debug_frames(video_path: Path, updates, debug_dir: Path, max_frames: int) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    saved = 0
    for update in updates:
        if saved >= max_frames:
            break
        reading = update.reading
        if reading.counter.freshness == "unknown" and not update.events:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(reading.frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        overlay = draw_debug_overlay(frame, reading)
        stem = video_path.stem.replace(" ", "_")
        path = debug_dir / f"{stem}_{reading.timestampMs:08d}ms.jpg"
        cv2.imwrite(str(path), overlay)
        saved += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Splatoon 3 ranked counters and objective state from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--rule",
        choices=["splat_zones", "tower_control", "rainmaker", "clam_blitz", "rule_unknown"],
        default="rule_unknown",
    )
    parser.add_argument("--context-json", type=Path, default=None)
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--time-left-sec", type=int, default=None)
    parser.add_argument("--sample-interval", type=float, default=0.20)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--min-digit-score", type=float, default=0.62)
    parser.add_argument("--marker-min-confidence", type=float, default=0.88)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--disable-timer-ocr", action="store_true")
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path, default=None)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-max-frames", type=int, default=30)
    args = parser.parse_args()

    context = load_context(args.context_json, args.rule, args.ally_side, args.time_left_sec)
    detector = RankedObjectiveDetector(
        templates_dir=args.templates_dir,
        min_digit_score=args.min_digit_score,
        use_timer_ocr=not args.disable_timer_ocr,
        marker_min_confidence=args.marker_min_confidence,
    )
    updates = sample_video(
        args.video,
        detector=detector,
        context=context,
        sample_interval=args.sample_interval,
        start=args.start,
        end=args.end,
        smooth=not args.no_smooth,
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

    known = sum(1 for update in updates if update.reading.counter.ourCount is not None and update.reading.counter.enemyCount is not None)
    marker_visible = sum(1 for update in updates if bool(update.reading.raw.get("marker", {}).get("visible")) if isinstance(update.reading.raw.get("marker"), dict))
    event_count = sum(len(update.events) for update in updates)
    print(
        f"readings={len(updates)} known_counters={known} marker_visible={marker_visible} events={event_count}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
