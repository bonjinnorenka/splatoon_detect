from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    from .weapon_lamp import WeaponIconMatcher, weapon_class_for_name
except ImportError:
    from weapon_lamp import WeaponIconMatcher, weapon_class_for_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "sample_data" / "Main Weapons"


@dataclass
class Metric:
    total: int = 0
    correct: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.correct += 1

    def to_dict(self) -> dict[str, int | float | None]:
        if self.total == 0:
            return {"correct": self.correct, "total": self.total, "accuracy": None}
        return {"correct": self.correct, "total": self.total, "accuracy": round(self.correct / self.total, 4)}


def _load_samples(session_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("samples.json must contain a list")
    return [item for item in data if isinstance(item, dict)]


def _load_labels(session_dir: Path) -> dict[str, dict[str, Any]]:
    current = session_dir / "labels_current.json"
    if current.exists():
        data = json.loads(current.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(key): value for key, value in data.items() if isinstance(value, dict)}

    labels: dict[str, dict[str, Any]] = {}
    jsonl = session_dir / "labels.jsonl"
    if not jsonl.exists():
        return labels
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict) and data.get("sample_id"):
            labels[str(data["sample_id"])] = data
    return labels


def evaluate_session(
    session_dir: Path,
    matcher: WeaponIconMatcher,
    top_k: int,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    samples = _load_samples(session_dir)
    labels = _load_labels(session_dir)
    metrics = {
        "exact_top1": Metric(),
        "exact_top3": Metric(),
        "exact_top5": Metric(),
        "class_top1": Metric(),
        "class_top3": Metric(),
        "class_top5": Metric(),
    }
    mistakes: list[dict[str, Any]] = []
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    evaluated = 0
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        label = labels.get(sample_id)
        if not label or label.get("status") != "labeled":
            continue
        expected = str(label.get("weapon") or "").strip()
        if not expected:
            continue
        crop_path = session_dir / str(sample.get("crop") or "")
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        candidates = matcher.predict_crop(crop, top_k=max(5, top_k))
        names = [candidate.weapon for candidate in candidates]
        classes = [candidate.weapon_class for candidate in candidates]
        expected_class = weapon_class_for_name(expected)
        exact_top1 = bool(names and names[0] == expected)
        class_top1 = bool(classes and classes[0] == expected_class)
        metrics["exact_top1"].add(exact_top1)
        metrics["exact_top3"].add(expected in names[:3])
        metrics["exact_top5"].add(expected in names[:5])
        metrics["class_top1"].add(class_top1)
        metrics["class_top3"].add(expected_class in classes[:3])
        metrics["class_top5"].add(expected_class in classes[:5])

        if not exact_top1 and len(mistakes) < 100:
            mistakes.append(
                {
                    "sample_id": sample_id,
                    "expected": expected,
                    "expected_class": expected_class,
                    "predicted": names[0] if names else None,
                    "predicted_class": classes[0] if classes else None,
                    "video": sample.get("video"),
                    "timestamp": sample.get("timestamp"),
                    "side": sample.get("side"),
                    "index": sample.get("index"),
                    "state": sample.get("state"),
                    "top5": [candidate.to_dict() for candidate in candidates[:5]],
                }
            )
            if debug_dir is not None:
                cv2.imwrite(str(debug_dir / f"{sample_id}_{expected}.jpg"), crop)
        evaluated += 1

    return {
        "session_dir": str(session_dir),
        "samples": len(samples),
        "labels": len(labels),
        "evaluated": evaluated,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weapon-lamp predictions against a labeling session.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variant-sizes", type=int, nargs="+", default=[62, 82])
    parser.add_argument("--variant-angles", type=float, nargs="+", default=[-8.0, 0.0, 8.0])
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    args = parser.parse_args()

    matcher = WeaponIconMatcher.from_dir(args.template_dir, variant_sizes=args.variant_sizes, variant_angles=args.variant_angles)
    result = evaluate_session(args.session_dir, matcher, top_k=max(1, args.top_k), debug_dir=args.debug_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
