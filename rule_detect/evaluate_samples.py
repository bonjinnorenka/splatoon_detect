from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

try:
    from .rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer, crop_and_preprocess_rule_roi
    from .train_templates import DEFAULT_CONFIG as DEFAULT_TRAINING_CONFIG
    from .train_templates import resolve_path
except ImportError:
    from rule_recognizer import DEFAULT_ACCEPT_THRESHOLD, RuleRecognizer, crop_and_preprocess_rule_roi
    from train_templates import DEFAULT_CONFIG as DEFAULT_TRAINING_CONFIG
    from train_templates import resolve_path


DEFAULT_NEGATIVE_CONFIG = Path(__file__).resolve().parent / "data" / "negative_segments.json"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list")
    return data


def expand_positive_samples(config_path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for segment in load_json_list(config_path):
        timestamps = segment.get("timestamps", [segment.get("timestamp")])
        if not isinstance(timestamps, list):
            raise ValueError("timestamps must be a list")
        for timestamp in timestamps:
            samples.append(
                {
                    "rule": str(segment["rule"]),
                    "video": str(segment["video"]),
                    "timestamp": float(timestamp),
                }
            )
    return samples


def read_frame(video: str, timestamp: float, config_path: Path) -> tuple[Any, int]:
    video_path = resolve_path(video, config_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_index = int(round(timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read {video_path} at {timestamp:.3f}s")
    return frame, frame_index


def evaluate_leave_one_out(
    samples: list[dict[str, object]],
    config_path: Path,
    accept_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rois = []
    for sample in samples:
        frame, _frame_index = read_frame(str(sample["video"]), float(sample["timestamp"]), config_path)
        rois.append(crop_and_preprocess_rule_roi(frame))

    class_counts = Counter(str(sample["rule"]) for sample in samples)
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        expected = str(sample["rule"])
        if class_counts[expected] <= 1:
            rows.append(
                {
                    **sample,
                    "predicted": "skipped",
                    "candidate_rule": None,
                    "confidence": 0.0,
                    "reason": "no same-class peer for leave-one-out",
                }
            )
            continue

        roi = rois[index]
        best_rule: str | None = None
        best_score = 0.0
        for other_index, other_sample in enumerate(samples):
            if index == other_index:
                continue
            template = rois[other_index]
            if template.shape != roi.shape:
                template = cv2.resize(template, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max()) if result.size else 0.0
            if not math.isfinite(score):
                score = 0.0
            if score > best_score:
                best_rule = str(other_sample["rule"])
                best_score = score

        predicted = best_rule if best_score >= accept_threshold else "unknown"
        rows.append(
            {
                **sample,
                "predicted": predicted,
                "candidate_rule": best_rule,
                "confidence": round(best_score, 4),
            }
        )

    eligible = [row for row in rows if row["predicted"] != "skipped"]
    correct = sum(1 for row in eligible if row["predicted"] == row["rule"])
    wrong = sum(1 for row in eligible if row["predicted"] not in {row["rule"], "unknown"})
    unknown = sum(1 for row in eligible if row["predicted"] == "unknown")
    skipped = len(rows) - len(eligible)
    summary = {
        "mode": "leave_one_out",
        "samples": len(rows),
        "eligible": len(eligible),
        "skipped": skipped,
        "correct": correct,
        "unknown": unknown,
        "wrong": wrong,
        "accuracy": round(correct / max(1, len(eligible)), 4),
        "coverage": round((correct + wrong) / max(1, len(eligible)), 4),
    }
    return summary, rows


def evaluate_with_templates(
    samples: list[dict[str, object]],
    config_path: Path,
    recognizer: RuleRecognizer,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    by_rule: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        frame, frame_index = read_frame(str(sample["video"]), float(sample["timestamp"]), config_path)
        result = recognizer.recognize_frame(frame, timestamp=float(sample["timestamp"]), frame_index=frame_index)
        expected = str(sample["rule"])
        row = {
            **sample,
            "predicted": result.rule,
            "confidence": round(result.confidence, 4),
            "candidate_rule": result.candidate_rule,
        }
        rows.append(row)
        by_rule[expected][result.rule] += 1

    correct = sum(1 for row in rows if row["predicted"] == row["rule"])
    wrong = len(rows) - correct
    summary = {
        "mode": "template_set",
        "samples": len(rows),
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(correct / max(1, len(rows)), 4),
        "by_rule": {rule: dict(counter) for rule, counter in sorted(by_rule.items())},
    }
    return summary, rows


def evaluate_negatives(
    config_path: Path,
    recognizer: RuleRecognizer,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for segment in load_json_list(config_path):
        timestamp = float(segment["timestamp"])
        frame, frame_index = read_frame(str(segment["video"]), timestamp, config_path)
        result = recognizer.recognize_frame(frame, timestamp=timestamp, frame_index=frame_index)
        rows.append(
            {
                **segment,
                "predicted": result.rule,
                "confidence": round(result.confidence, 4),
                "candidate_rule": result.candidate_rule,
            }
        )

    false_positive = sum(1 for row in rows if row["predicted"] != "unknown")
    summary = {
        "mode": "negative",
        "samples": len(rows),
        "unknown": len(rows) - false_positive,
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / max(1, len(rows)), 4),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Splatoon 3 rule recognition on labeled start and negative frames.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--negative-config", type=Path, default=DEFAULT_NEGATIVE_CONFIG)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD)
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    positives = expand_positive_samples(args.training_config)
    recognizer = RuleRecognizer(templates_dir=args.templates_dir, accept_threshold=args.accept_threshold)
    if not recognizer.templates:
        raise FileNotFoundError("rule templates are missing; run rule_detect/train_templates.py first")

    summaries = []
    problem_rows: list[dict[str, object]] = []

    loo_summary, loo_rows = evaluate_leave_one_out(positives, args.training_config, args.accept_threshold)
    summaries.append(loo_summary)
    problem_rows.extend(row for row in loo_rows if row["predicted"] not in {row["rule"], "skipped"})

    template_summary, template_rows = evaluate_with_templates(positives, args.training_config, recognizer)
    summaries.append(template_summary)
    problem_rows.extend(row for row in template_rows if row["predicted"] != row["rule"])

    negative_summary, negative_rows = evaluate_negatives(args.negative_config, recognizer)
    summaries.append(negative_summary)
    problem_rows.extend(row for row in negative_rows if row["predicted"] != "unknown")

    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))

    if args.show_rows:
        for row in loo_rows + template_rows + negative_rows:
            print(json.dumps(row, ensure_ascii=False))
    elif problem_rows:
        for row in problem_rows:
            print(json.dumps(row, ensure_ascii=False))

    if template_summary["wrong"] or negative_summary["false_positive"] or loo_summary["wrong"]:
        raise SystemExit("evaluation failed")


if __name__ == "__main__":
    main()
