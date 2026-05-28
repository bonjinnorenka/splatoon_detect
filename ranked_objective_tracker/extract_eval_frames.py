from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from .ranked_objective import RankedObjectiveDetector, TrackerContext, draw_debug_overlay
except ImportError:
    from ranked_objective import RankedObjectiveDetector, TrackerContext, draw_debug_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract draft evaluation frames for ranked objective tracker labeling.")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "eval_samples")
    parser.add_argument("--rule", choices=["splat_zones", "tower_control", "rainmaker", "clam_blitz", "rule_unknown"], default="rule_unknown")
    parser.add_argument("--ally-side", choices=["left", "right"], default="right")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--with-overlay", action="store_true")
    args = parser.parse_args()

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    detector = RankedObjectiveDetector(use_timer_ocr=False)
    context = TrackerContext.from_mapping({"rule": args.rule, "ally_side": args.ally_side, "match_started": True})
    annotations: list[dict[str, object]] = []
    saved = 0

    for video in args.videos:
        if saved >= args.max_frames:
            break
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0.0
        stop = duration if args.end is None else min(args.end, duration)
        timestamp = max(0.0, args.start)
        while timestamp <= stop + 1e-6 and saved < args.max_frames:
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            reading = detector.read_frame(frame, context=context, timestamp=timestamp, frame_index=frame_index)
            stem = video.stem.replace(" ", "_")
            image_name = f"{stem}_{int(round(timestamp * 1000)):08d}ms.jpg"
            image_path = images_dir / image_name
            output_frame = draw_debug_overlay(frame, reading) if args.with_overlay else frame
            cv2.imwrite(str(image_path), output_frame)
            annotations.append(
                {
                    "id": image_path.stem,
                    "image": f"images/{image_name}",
                    "video": str(video),
                    "timestamp": round(float(timestamp), 4),
                    "frame_index": frame_index,
                    "rule": args.rule,
                    "ally_side": args.ally_side,
                    "expected": {
                        "counter": {
                            "ourCount": None,
                            "enemyCount": None,
                            "leader": "unknown",
                            "progressDirection": "unknown",
                        },
                        "objective": {
                            "screenPosition": {
                                "visible": False,
                                "x": None,
                                "y": None,
                            }
                        },
                    },
                    "draft_prediction": reading.to_dict(include_debug=True),
                }
            )
            saved += 1
            timestamp += args.sample_interval
        cap.release()

    annotations_path = args.out_dir / "annotations.draft.jsonl"
    with annotations_path.open("w", encoding="utf-8") as file:
        for annotation in annotations:
            file.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    print(f"saved_frames={saved} annotations={annotations_path}")


if __name__ == "__main__":
    main()
