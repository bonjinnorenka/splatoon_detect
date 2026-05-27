from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from .squid_lamp import SquidLampDetector, draw_debug_overlay
except ImportError:
    from squid_lamp import SquidLampDetector, draw_debug_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract top HUD crops from videos for squid lamp annotation/training."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/hud_crops"))
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--top-height", type=float, default=0.20)
    parser.add_argument("--with-overlay", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.jsonl"
    detector = SquidLampDetector(ally_side=args.ally_side)

    with metadata_path.open("w", encoding="utf-8") as metadata:
        for video in args.videos:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise FileNotFoundError(video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps else 0.0
            stop = duration if args.end is None else min(args.end, duration)
            timestamp = max(0.0, args.start)
            while timestamp <= stop + 1e-6:
                frame_index = int(round(timestamp * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    break
                reading = detector.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
                crop_h = max(1, int(round(frame.shape[0] * args.top_height)))
                crop = frame[:crop_h, :]
                stem = video.stem.replace(" ", "_")
                image_name = f"{stem}_{int(round(timestamp * 1000)):08d}ms.jpg"
                image_path = args.out_dir / image_name
                cv2.imwrite(str(image_path), crop)
                overlay_name = None
                if args.with_overlay:
                    overlay_name = f"{stem}_{int(round(timestamp * 1000)):08d}ms_overlay.jpg"
                    overlay = draw_debug_overlay(frame, reading)
                    cv2.imwrite(str(args.out_dir / overlay_name), overlay[:crop_h, :])

                metadata.write(
                    json.dumps(
                        {
                            "video": str(video),
                            "image": image_name,
                            "overlay": overlay_name,
                            "timestamp": timestamp,
                            "timestamp_ms": int(round(timestamp * 1000)),
                            "frame_index": frame_index,
                            "prediction": reading.to_dict(),
                            "label": {
                                "left": ["unknown", "unknown", "unknown", "unknown"],
                                "right": ["unknown", "unknown", "unknown", "unknown"],
                                "enemy_special_ready": [None, None, None, None],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                timestamp += args.sample_interval
            cap.release()

    print(metadata_path)


if __name__ == "__main__":
    main()
