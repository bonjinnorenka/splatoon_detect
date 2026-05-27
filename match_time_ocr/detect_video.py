from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

from timer_ocr import TimerOCR, interpolate_readings, sample_video, write_csv, write_jsonl


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    return fps


def write_stdout_csv(readings) -> None:
    fieldnames = ["timestamp", "frame_index", "kind", "text", "seconds", "confidence", "inferred"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for reading in readings:
        if reading is None:
            continue
        writer.writerow(reading.to_row())


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Splatoon match remaining time from a captured video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path, default=None)
    parser.add_argument("--no-interpolate", action="store_true")
    parser.add_argument("--interpolate-gap", type=float, default=15.0)
    parser.add_argument("--min-digit-score", type=float, default=0.70)
    args = parser.parse_args()

    ocr = TimerOCR(templates_dir=args.templates_dir, min_digit_score=args.min_digit_score)
    fps = get_fps(args.video)
    raw = sample_video(
        args.video,
        ocr,
        sample_interval=args.sample_interval,
        start=args.start,
        end=args.end,
    )
    readings = raw
    if not args.no_interpolate:
        readings = interpolate_readings(raw, args.sample_interval, max_gap=args.interpolate_gap, fps=fps)

    if args.csv_path is not None:
        write_csv(args.csv_path, readings)
    if args.jsonl_path is not None:
        write_jsonl(args.jsonl_path, readings)
    if args.csv_path is None and args.jsonl_path is None:
        write_stdout_csv(readings)

    found = sum(1 for reading in readings if reading is not None)
    overtime = sum(1 for reading in readings if reading is not None and reading.kind == "overtime")
    print(f"readings={found} overtime={overtime}", file=sys.stderr)


if __name__ == "__main__":
    main()
