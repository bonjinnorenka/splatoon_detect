from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer, sample_video_frames, write_json
except ImportError:
    from rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer, sample_video_frames, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize the Splatoon 3 match rule from the start animation.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    recognizer = RuleRecognizer(
        templates_dir=args.templates_dir,
        accept_threshold=args.accept_threshold,
    )
    if not recognizer.templates:
        raise FileNotFoundError("rule templates are missing; run rule_detect/train_templates.py first")

    frames = sample_video_frames(
        args.video,
        start=args.start,
        end=args.end,
        sample_interval=args.sample_interval,
    )
    result = recognizer.recognize_frames(frames)
    data = result.to_dict()

    if args.json_path is not None:
        write_json(args.json_path, data)
    print(json.dumps(data, ensure_ascii=False))
    print(
        f"rule={result.rule} confidence={result.confidence:.4f} source={result.source}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
