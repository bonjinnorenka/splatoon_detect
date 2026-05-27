from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from timer_ocr import TimerOCR


@dataclass(frozen=True)
class Sample:
    video: Path
    timestamp: float
    expected_kind: str
    expected_text: str


def read_at(ocr: TimerOCR, video_path: Path, timestamp: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_index = int(round(timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read {video_path} at {timestamp}")
    return ocr.read_frame(frame, timestamp=timestamp, frame_index=frame_index)


def main() -> None:
    root = Path(__file__).resolve().parent
    videos_dir = (root / "../../videos").resolve()
    ocr = TimerOCR(root / "templates" / "digits")
    samples = [
        Sample(videos_dir / "2026-02-01 21-18-24.mp4", 20.0, "time", "5:00"),
        Sample(videos_dir / "2026-02-01 21-18-24.mp4", 310.0, "time", "0:10"),
        Sample(videos_dir / "2026-02-01 21-18-24.mp4", 320.0, "overtime", "延長中!"),
        Sample(videos_dir / "2026-02-06 22-39-39.mkv", 13.0, "time", "5:00"),
        Sample(videos_dir / "2026-02-06 22-39-39.mkv", 20.0, "time", "4:53"),
        Sample(videos_dir / "2026-02-06 22-39-39.mkv", 2100.0, "overtime", "延長中!"),
    ]

    failures: list[str] = []
    for sample in samples:
        reading = read_at(ocr, sample.video, sample.timestamp)
        actual = "None" if reading is None else f"{reading.kind} {reading.text}"
        expected = f"{sample.expected_kind} {sample.expected_text}"
        print(f"{sample.video.name} @{sample.timestamp:.1f}s: {actual}")
        if reading is None or reading.kind != sample.expected_kind or reading.text != sample.expected_text:
            failures.append(f"{sample.video.name} @{sample.timestamp:.1f}s expected {expected}, got {actual}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
