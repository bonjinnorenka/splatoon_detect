from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from match_time_ocr.timer_ocr import TimerOCR
from ranked_objective_tracker import RankedObjectiveDetector, RankedObjectiveTracker
from rule_detect import RuleRecognizer
from special_gauge_detect import SpecialGaugeDetector, SpecialGaugeTracker
from squid_lamp_detect import SquidLampDetector, SquidLampTracker

from .commenter import Commenter, ImagePayload
from .screen_detectors import DeathDetector, StartDetector


RULE_CHOICES = ("auto", "unknown", "turf_war", "splat_zones", "tower_control", "rainmaker", "clam_blitz")


@dataclass
class FrameSnapshot:
    timestamp: float
    timestamp_ms: int
    frame_index: int
    frame: np.ndarray


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"video not found: {video_path}", file=sys.stderr)
        return 2

    if args.comments_jsonl is not None:
        args.comments_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.state_jsonl is not None:
        args.state_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    commenter = Commenter(disabled=args.no_gemini, timeout_sec=args.gemini_timeout)
    runner = VideoCommentaryRunner(args=args, commenter=commenter)
    return runner.run(video_path)


class VideoCommentaryRunner:
    def __init__(self, *, args: argparse.Namespace, commenter: Commenter) -> None:
        self.args = args
        self.commenter = commenter
        self.timer_ocr = TimerOCR()
        self.lamp_tracker = SquidLampTracker(SquidLampDetector(ally_side=args.ally_side))
        self.special_tracker = SpecialGaugeTracker(SpecialGaugeDetector())
        self.death_detector = DeathDetector()
        self.start_detector = StartDetector()
        self.rule_recognizer = RuleRecognizer()
        self.ranked_tracker = RankedObjectiveTracker(RankedObjectiveDetector())

        self.death_votes: deque[int] = deque(maxlen=max(1, int(args.death_consecutive)))
        self.start_votes: deque[int] = deque(maxlen=max(1, int(args.start_consecutive)))
        self.frame_buffer: deque[FrameSnapshot] = deque()
        self.recent_events: deque[dict[str, object]] = deque(maxlen=30)
        self.last_death_ms = -10**12
        self.match_death_count = 0
        self.rule_state = {
            "rule": args.rule if args.rule != "auto" else "unknown",
            "confidence": 1.0 if args.rule not in {"auto", "unknown"} else 0.0,
            "source": "cli" if args.rule != "auto" else "unknown",
            "locked": args.rule not in {"auto", "unknown"},
        }

    def run(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"failed to open video: {video_path}", file=sys.stderr)
            return 2

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            print("video fps is unavailable", file=sys.stderr)
            return 2
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else None
        start = max(0.0, float(self.args.start))
        stop = float(self.args.end) if self.args.end is not None else duration
        if stop is None:
            print("--end is required when video duration is unavailable", file=sys.stderr)
            return 2
        sample_interval = max(0.01, float(self.args.sample_interval))

        comments_file = self.args.comments_jsonl.open("w", encoding="utf-8") if self.args.comments_jsonl else None
        state_file = self.args.state_jsonl.open("w", encoding="utf-8") if self.args.state_jsonl else None
        try:
            timestamp = start
            processed = 0
            comments = 0
            while timestamp <= stop + 1e-6:
                frame_index = int(round(timestamp * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                timestamp_ms = int(round(timestamp * 1000.0))
                update = self.process_frame(frame, timestamp, timestamp_ms, frame_index)
                processed += 1

                if state_file is not None:
                    write_jsonl(state_file, update["state"])
                comment = update.get("comment")
                if isinstance(comment, dict):
                    comments += 1
                    if comments_file is not None:
                        write_jsonl(comments_file, comment)
                    if comment.get("speak") and comment.get("line"):
                        print(f"{format_timestamp(timestamp)} {comment['line']}")

                timestamp += sample_interval

            print(f"processed={processed} comments={comments} gemini_enabled={self.commenter.gemini_enabled}", file=sys.stderr)
        finally:
            cap.release()
            if comments_file is not None:
                comments_file.close()
            if state_file is not None:
                state_file.close()
        return 0

    def process_frame(self, frame: np.ndarray, timestamp: float, timestamp_ms: int, frame_index: int) -> dict[str, object]:
        self._append_frame(FrameSnapshot(timestamp, timestamp_ms, frame_index, frame.copy()))
        timer = self.timer_ocr.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        self._update_rule(frame, timestamp, frame_index)
        lamp = self.lamp_tracker.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        special_update = self.special_tracker.read_frame(frame, timestamp=timestamp, frame_index=frame_index)
        ranked_update = self._read_ranked_objective(frame, timestamp, frame_index, timer, lamp)

        death_prediction, death_crop, death_rect = self.death_detector.predict_frame(frame)
        death_detected = self._death_detected(death_prediction, timestamp_ms)
        start_prediction, _start_crop, start_rect = self.start_detector.predict_frame(frame)
        start_detected = False if death_detected else self._start_detected(start_prediction)

        events: list[dict[str, object]] = []
        for event in special_update.events:
            normalized = self._event_with_time(event, timestamp_ms)
            events.append(normalized)
        if ranked_update is not None:
            for event in ranked_update.events:
                normalized = self._event_with_time(event, timestamp_ms)
                events.append(normalized)

        if start_detected:
            self.match_death_count = 0
            self.special_tracker.reset()
            self.death_votes.clear()
            event = {"type": "match_start_detected", "confidence": round(float(start_prediction.probability), 4), "timestamp_ms": timestamp_ms}
            events.append(event)

        comment: dict[str, object] | None = None
        if death_detected:
            self.match_death_count += 1
            death_event = {
                "type": "player_death",
                "confidence": round(float(death_prediction.confidence), 4),
                "timestamp_ms": timestamp_ms,
                "match_death_count": self.match_death_count,
            }
            events.append(death_event)
            for event in self.special_tracker.handle_external_event(death_event):
                events.append(self._event_with_time(event, timestamp_ms))
            request = self._build_comment_request(
                timestamp=timestamp,
                timestamp_ms=timestamp_ms,
                frame_index=frame_index,
                timer=timer,
                lamp=lamp,
                special_update=special_update,
                ranked_update=ranked_update,
                death_prediction=death_prediction.to_dict(),
                events=events,
            )
            images = self._death_images(timestamp, timestamp_ms, death_crop)
            comment_result = self.commenter.generate(request, images)
            comment = {
                "timestamp": round(float(timestamp), 4),
                "timestamp_ms": timestamp_ms,
                "frame_index": frame_index,
                "event": death_event,
                **comment_result,
                "request": request,
            }

        for event in events:
            self.recent_events.append(event)

        state = {
            "timestamp": round(float(timestamp), 4),
            "timestamp_ms": timestamp_ms,
            "frame_index": frame_index,
            "match": {
                **self.rule_state,
                "time_left_sec": None if timer is None else timer.seconds,
                "time_text": None if timer is None else timer.text,
                "time_kind": None if timer is None else timer.kind,
                "phase": match_phase(None if timer is None else timer.seconds),
            },
            "squid_lamps": lamp.to_dict(),
            "special_gauge": special_update.to_dict(),
            "ranked_objective": None if ranked_update is None else ranked_update.to_dict(),
            "death_detection": {
                **death_prediction.to_dict(),
                "detected": bool(death_detected),
                "roi_rect": list(death_rect),
                "match_death_count": self.match_death_count,
            },
            "start_detection": {
                **start_prediction.to_dict(),
                "detected": bool(start_detected),
                "roi_rect": list(start_rect),
            },
            "events": events,
        }
        return {"state": state, "comment": comment}

    def _update_rule(self, frame: np.ndarray, timestamp: float, frame_index: int) -> None:
        if self.rule_state.get("locked"):
            return
        result = self.rule_recognizer.recognize_frame(frame, timestamp=timestamp, frame_index=frame_index)
        if result.rule != "unknown":
            self.rule_state = {
                "rule": result.rule,
                "confidence": round(float(result.confidence), 4),
                "source": result.source,
                "locked": True,
            }

    def _read_ranked_objective(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        timer: Any,
        lamp: Any,
    ) -> Any | None:
        rule = ranked_rule_from_match_rule(str(self.rule_state.get("rule", "unknown")))
        if rule is None:
            return None
        context = {
            "match_started": getattr(lamp, "hud_state", "") == "match",
            "rule": rule,
            "rule_confidence": float(self.rule_state.get("confidence", 0.0) or 0.0),
            "rule_locked": bool(self.rule_state.get("locked", False)),
            "time_left_sec": None if timer is None else timer.seconds,
            "ally_side": self.args.ally_side,
            "squid_lamps": {
                "ally_alive": getattr(lamp, "ally_alive", None),
                "enemy_alive": getattr(lamp, "enemy_alive", None),
                "ally_down": getattr(lamp, "ally_down", None),
                "enemy_down": getattr(lamp, "enemy_down", None),
                "confidence": round(float(getattr(lamp, "confidence", 0.0)), 4),
            },
        }
        return self.ranked_tracker.read_frame(frame, context=context, timestamp=timestamp, frame_index=frame_index)

    def _death_detected(self, prediction: Any, timestamp_ms: int) -> bool:
        high = prediction.prediction == 1 and prediction.confidence >= self.args.death_threshold
        self.death_votes.append(1 if high else 0)
        if len(self.death_votes) < self.death_votes.maxlen:
            return False
        if not all(value == 1 for value in self.death_votes):
            return False
        if timestamp_ms - self.last_death_ms < int(self.args.death_cooldown_sec * 1000):
            return False
        self.last_death_ms = timestamp_ms
        self.death_votes.clear()
        return True

    def _start_detected(self, prediction: Any) -> bool:
        high = prediction.prediction == 1 and prediction.probability >= self.args.start_threshold
        self.start_votes.append(1 if high else 0)
        if len(self.start_votes) < self.start_votes.maxlen:
            return False
        if not all(value == 1 for value in self.start_votes):
            return False
        self.start_votes.clear()
        return True

    def _append_frame(self, snapshot: FrameSnapshot) -> None:
        self.frame_buffer.append(snapshot)
        cutoff = snapshot.timestamp - float(self.args.ring_seconds)
        while self.frame_buffer and self.frame_buffer[0].timestamp < cutoff:
            self.frame_buffer.popleft()

    def _build_comment_request(
        self,
        *,
        timestamp: float,
        timestamp_ms: int,
        frame_index: int,
        timer: Any,
        lamp: Any,
        special_update: Any,
        ranked_update: Any | None,
        death_prediction: dict[str, object],
        events: list[dict[str, object]],
    ) -> dict[str, object]:
        facts = ["player_dead"]
        event_types = {str(event.get("type", "")) for event in events}
        if "death_with_special_ready" in event_types:
            facts.append("death_with_special_ready")
        if timer is not None and getattr(timer, "seconds", None) is not None and int(timer.seconds) <= 30:
            facts.append("late_game_death")
        if timer is not None and getattr(timer, "kind", "") == "overtime":
            facts.append("overtime_death")
        objective_summary = ranked_objective_summary(ranked_update)
        if objective_summary.get("kind") == "tower_control":
            facts.append("tower_control_death")
            if objective_summary.get("tower_owner") == "enemy":
                facts.append("enemy_tower_push_death")
            if objective_summary.get("tower_visible") is True:
                facts.append("tower_visible_on_screen")

        inferences: list[dict[str, object]] = []
        ally_alive = getattr(lamp, "ally_alive", None)
        enemy_alive = getattr(lamp, "enemy_alive", None)
        if isinstance(ally_alive, int) and isinstance(enemy_alive, int) and ally_alive < enemy_alive:
            inferences.append(
                {
                    "type": "outnumbered_death",
                    "confidence": round(min(0.9, max(0.55, float(getattr(lamp, "confidence", 0.0)))), 4),
                    "basis": {"ally_alive": ally_alive, "enemy_alive": enemy_alive},
                }
            )
        if objective_summary.get("kind") == "tower_control" and objective_summary.get("tower_owner") == "enemy":
            confidence = float(objective_summary.get("confidence", 0.0) or 0.0)
            inferences.append(
                {
                    "type": "died_while_enemy_tower_pushing",
                    "confidence": round(min(0.92, max(0.55, confidence)), 4),
                    "basis": {
                        "tower_owner": objective_summary.get("tower_owner"),
                        "tower_progress": objective_summary.get("tower_progress"),
                        "screen_position": objective_summary.get("screen_position"),
                    },
                }
            )

        return {
            "timestamp_sec": round(float(timestamp), 4),
            "timestamp_ms": timestamp_ms,
            "frame_index": frame_index,
            "match": {
                **self.rule_state,
                "time_left_sec": None if timer is None else timer.seconds,
                "time_text": None if timer is None else timer.text,
                "time_kind": None if timer is None else timer.kind,
                "phase": match_phase(None if timer is None else timer.seconds),
            },
            "player": {
                "alive": False,
                "special_gauge": special_update.special_gauge.gauge_dict(),
                "death_event": {
                    "detected": True,
                    "match_death_count": self.match_death_count,
                    "confidence": death_prediction.get("confidence"),
                },
            },
            "squid_lamps": {
                "ally_alive": lamp.ally_alive,
                "enemy_alive": lamp.enemy_alive,
                "ally_down": lamp.ally_down,
                "enemy_down": lamp.enemy_down,
                "confidence": round(float(lamp.confidence), 4),
                "hud_state": lamp.hud_state,
            },
            "objective": objective_summary,
            "facts": facts,
            "inferences": inferences,
            "recent_events": recent_events_for_prompt(self.recent_events, timestamp_ms, events),
            "comment_policy": {
                "style": "harsh_short_roast",
                "target": "player_decision_only",
                "avoid": [
                    "protected_class",
                    "real_player_name_attack",
                    "personality_attack",
                    "sexual_content",
                    "self_harm",
                    "slurs",
                ],
            },
        }

    def _death_images(self, timestamp: float, timestamp_ms: int, death_crop: np.ndarray) -> list[ImagePayload]:
        targets = [
            ("t_minus_2s", timestamp - 2.0),
            ("t_minus_1s", timestamp - 1.0),
            ("death_frame", timestamp),
        ]
        images: list[ImagePayload] = []
        used_snapshots: set[int] = set()
        for label, target in targets:
            snapshot = closest_snapshot(self.frame_buffer, target)
            if snapshot is None:
                continue
            key = snapshot.frame_index
            if key in used_snapshots and label != "death_frame":
                continue
            used_snapshots.add(key)
            encoded = encode_jpeg(snapshot.frame, max_width=768, quality=82)
            if encoded is not None:
                images.append(ImagePayload(label=label, data=encoded))
                self._write_debug_image(timestamp_ms, label, encoded)

        crop_encoded = encode_jpeg(death_crop, max_width=512, quality=88)
        if crop_encoded is not None:
            images.append(ImagePayload(label="death_roi", data=crop_encoded))
            self._write_debug_image(timestamp_ms, "death_roi", crop_encoded)
        return images

    def _write_debug_image(self, timestamp_ms: int, label: str, data: bytes) -> None:
        if self.args.debug_dir is None:
            return
        path = self.args.debug_dir / f"{timestamp_ms:08d}ms_{label}.jpg"
        path.write_bytes(data)

    @staticmethod
    def _event_with_time(event: dict[str, object], timestamp_ms: int) -> dict[str, object]:
        normalized = dict(event)
        if "event" in normalized and "type" not in normalized:
            normalized["type"] = normalized.pop("event")
        normalized.setdefault("timestamp_ms", timestamp_ms)
        return normalized


def closest_snapshot(buffer: Iterable[FrameSnapshot], target_timestamp: float) -> FrameSnapshot | None:
    best: FrameSnapshot | None = None
    best_distance = float("inf")
    for snapshot in buffer:
        distance = abs(snapshot.timestamp - target_timestamp)
        if distance < best_distance:
            best = snapshot
            best_distance = distance
    return best


def ranked_rule_from_match_rule(rule: str) -> str | None:
    if rule in {"splat_zones", "tower_control", "rainmaker", "clam_blitz"}:
        return rule
    return None


def ranked_objective_summary(ranked_update: Any | None) -> dict[str, object]:
    if ranked_update is None:
        return {"kind": "none", "available": False}
    reading = ranked_update.reading
    objective = dict(reading.objective)
    counter = reading.counter.to_dict()
    result: dict[str, object] = {
        "kind": objective.get("kind", reading.rule),
        "available": True,
        "counter": {
            "our_count": counter.get("ourCount"),
            "enemy_count": counter.get("enemyCount"),
            "leader": counter.get("leader"),
            "progress_direction": counter.get("progressDirection"),
            "recent_delta": counter.get("recentDelta"),
            "freshness": counter.get("freshness"),
        },
        "confidence": reading.quality.get("overallConfidence", objective.get("confidence", 0.0)),
        "events": [event.get("type", event.get("event", "unknown")) for event in ranked_update.events],
    }
    if objective.get("kind") == "tower_control":
        screen_position = objective.get("screenPosition")
        if not isinstance(screen_position, dict):
            screen_position = {}
        result.update(
            {
                "tower_owner": objective.get("towerOwner"),
                "tower_progress": objective.get("towerProgress"),
                "tower_visible": bool(screen_position.get("visible", False)),
                "screen_position": {
                    "visible": bool(screen_position.get("visible", False)),
                    "x": screen_position.get("x"),
                    "y": screen_position.get("y"),
                    "distance_class": screen_position.get("distanceClass"),
                },
            }
        )
    else:
        result["objective"] = objective
    return result


def encode_jpeg(frame: np.ndarray, *, max_width: int, quality: int) -> bytes | None:
    if frame.size == 0:
        return None
    image = frame
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / max(1, width)
        image = cv2.resize(image, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return data.tobytes()


def recent_events_for_prompt(
    stored: Iterable[dict[str, object]],
    timestamp_ms: int,
    current: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    combined = list(stored)[-8:] + list(current)
    result: list[dict[str, object]] = []
    for event in combined[-10:]:
        event_ms = int(event.get("timestamp_ms", timestamp_ms) or timestamp_ms)
        item = {
            "t_minus_sec": round(max(0, timestamp_ms - event_ms) / 1000.0, 3),
            "type": event.get("type", event.get("event", "unknown")),
            "confidence": event.get("confidence"),
        }
        if "match_death_count" in event:
            item["match_death_count"] = event["match_death_count"]
        result.append(item)
    return result


def match_phase(time_left_sec: int | None) -> str:
    if time_left_sec is None:
        return "unknown"
    if time_left_sec <= 0:
        return "overtime"
    if time_left_sec <= 30:
        return "end_game"
    if time_left_sec <= 90:
        return "late_game"
    return "mid_game"


def format_timestamp(timestamp: float) -> str:
    seconds = max(0, int(round(timestamp)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def write_jsonl(file: Any, item: dict[str, object]) -> None:
    file.write(json.dumps(item, ensure_ascii=False, default=json_default, separators=(",", ":")))
    file.write("\n")
    file.flush()


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run existing Splatoon screen recognizers and generate short death comments.")
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--ally-side", choices=("left", "right"), default="right")
    parser.add_argument("--rule", choices=RULE_CHOICES, default="auto")
    parser.add_argument("--comments-jsonl", type=Path, default=None)
    parser.add_argument("--state-jsonl", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--no-gemini", action="store_true", help="Use template fallback even if GEMINI_API_KEY/GEMINI_MODEL are set.")
    parser.add_argument("--gemini-timeout", type=float, default=20.0)
    parser.add_argument("--death-threshold", type=float, default=0.90)
    parser.add_argument("--death-consecutive", type=int, default=2)
    parser.add_argument("--death-cooldown-sec", type=float, default=10.0)
    parser.add_argument("--start-threshold", type=float, default=0.90)
    parser.add_argument("--start-consecutive", type=int, default=10)
    parser.add_argument("--ring-seconds", type=float, default=2.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
