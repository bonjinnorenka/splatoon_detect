from __future__ import annotations

import sys
from pathlib import Path

import cv2

try:
    from .special_gauge import GaugeReading, ROIConfig, SpecialGaugeDetector, SpecialGaugeTracker
except ImportError:
    from special_gauge import GaugeReading, ROIConfig, SpecialGaugeDetector, SpecialGaugeTracker


SAMPLE_EXPECTATIONS: dict[str, set[str]] = {
    "2026-02-01_21-18-24_0005s.png": {"unknown"},
    "2026-02-01_21-18-24_0030s.png": {"not_ready", "near_ready"},
    "2026-02-01_21-18-24_0120s.png": {"not_ready", "near_ready"},
    "2026-02-01_21-18-24_0180s.png": {"not_ready", "near_ready"},
    "2026-02-01_21-18-24_0240s.png": {"ready"},
    "2026-02-01_21-18-24_0300s.png": {"not_ready", "near_ready"},
    "2026-02-01_21-18-24_0360s.png": {"unknown"},
    "2026-02-06_22-39-39_0005s.png": {"unknown"},
    "2026-02-06_22-39-39_0030s.png": {"not_ready", "near_ready"},
    "2026-02-06_22-39-39_0060s.png": {"not_ready", "near_ready"},
    "2026-02-06_22-39-39_0120s.png": {"not_ready", "near_ready"},
    "2026-02-06_22-39-39_0180s.png": {"not_ready", "near_ready"},
    "2026-02-06_22-39-39_0240s.png": {"ready"},
    "2026-02-06_22-39-39_0300s.png": {"not_ready", "near_ready"},
    "2026-02-06_22-39-39_0360s.png": {"unknown"},
}


def stable_state_for_frame(detector: SpecialGaugeDetector, frame, timestamp: float) -> GaugeReading:
    tracker = SpecialGaugeTracker(detector)
    result = None
    for index in range(5):
        result = tracker.read_frame(frame, timestamp=timestamp + index * 0.1, frame_index=index)
    assert result is not None
    return result.special_gauge


def validate_images(samples_dir: Path) -> list[str]:
    detector = SpecialGaugeDetector(roi=ROIConfig())
    failures: list[str] = []
    for filename, expected in SAMPLE_EXPECTATIONS.items():
        path = samples_dir / filename
        if not path.exists():
            print(f"skip missing sample: {path}")
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            failures.append(f"{filename}: could not read image")
            continue
        reading = stable_state_for_frame(detector, frame, timestamp=0.0)
        ok = reading.state in expected
        marker = "OK" if ok else "NG"
        print(
            f"{marker} {filename}: state={reading.state} conf={reading.confidence:.3f} "
            f"raw={reading.raw_state} ready_score={float(reading.features.get('ready_score', 0.0)):.3f} "
            f"visible={float(reading.features.get('visible_score', 0.0)):.3f}"
        )
        if not ok:
            failures.append(f"{filename}: expected {sorted(expected)}, got {reading.state}")

        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            resized_reading = stable_state_for_frame(detector, resized, timestamp=0.0)
            resized_ok = resized_reading.state in expected
            marker = "OK" if resized_ok else "NG"
            print(
                f"{marker} {filename}@720p: state={resized_reading.state} "
                f"conf={resized_reading.confidence:.3f} raw={resized_reading.raw_state} "
                f"ready_score={float(resized_reading.features.get('ready_score', 0.0)):.3f} "
                f"visible={float(resized_reading.features.get('visible_score', 0.0)):.3f}"
            )
            if not resized_ok:
                failures.append(f"{filename}@720p: expected {sorted(expected)}, got {resized_reading.state}")
    return failures


def reading(timestamp_ms: int, state: str, confidence: float = 0.92, visible: bool = True) -> GaugeReading:
    return GaugeReading(
        timestamp=timestamp_ms / 1000.0,
        frame_index=timestamp_ms,
        timestamp_ms=timestamp_ms,
        state=state,  # type: ignore[arg-type]
        confidence=confidence,
        roi_visible=visible,
    )


def validate_events() -> list[str]:
    failures: list[str] = []
    detector = SpecialGaugeDetector()

    tracker = SpecialGaugeTracker(detector, window_size=1, ready_votes=1, state_votes=1)
    ready_update = tracker.update(reading(1000, "ready"))
    if not any(event.get("type") == "special_ready" for event in ready_update.events):
        failures.append("special_ready did not fire")
    death_events = tracker.handle_external_event({"type": "player_death", "timestamp_ms": 2500})
    if not any(event.get("type") == "death_with_special_ready" for event in death_events):
        failures.append("death_with_special_ready did not fire")

    tracker = SpecialGaugeTracker(detector, window_size=1, ready_votes=1, state_votes=1)
    tracker.update(reading(1000, "ready"))
    used_update = tracker.update(reading(1800, "not_ready"))
    if not any(event.get("type") == "special_used" for event in used_update.events):
        failures.append("special_used did not fire")
    if used_update.special_gauge.state != "using":
        failures.append("state did not become using after special_used")

    tracker = SpecialGaugeTracker(detector, window_size=1, ready_votes=1, state_votes=1)
    tracker.update(reading(1000, "ready"))
    tracker.handle_external_event({"type": "player_death", "timestamp_ms": 1500})
    after_death_update = tracker.update(reading(1600, "not_ready"))
    if any(event.get("type") == "special_used" for event in after_death_update.events):
        failures.append("special_used fired after a death event")

    tracker = SpecialGaugeTracker(detector, window_size=1, ready_votes=1, state_votes=1)
    tracker.update(reading(0, "ready"))
    unused_update = tracker.update(reading(15100, "ready"))
    if not any(event.get("type") == "special_ready_unused_too_long" for event in unused_update.events):
        failures.append("special_ready_unused_too_long did not fire")

    return failures


def main() -> None:
    repo_private = Path(__file__).resolve().parents[1]
    samples_dir = repo_private / "debug_timer_samples"
    failures = validate_images(samples_dir)
    failures.extend(validate_events())

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("validation passed")


if __name__ == "__main__":
    main()
