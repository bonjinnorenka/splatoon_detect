from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import cv2

try:
    from .ranked_objective import (
        CounterConfidence,
        CounterState,
        RankedObjectiveDetector,
        RankedObjectiveReading,
        RankedObjectiveTracker,
        TrackerContext,
        compute_progress_direction,
    )
except ImportError:
    from ranked_objective import (
        CounterConfidence,
        CounterState,
        RankedObjectiveDetector,
        RankedObjectiveReading,
        RankedObjectiveTracker,
        TrackerContext,
        compute_progress_direction,
    )


DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "data" / "eval_samples" / "annotations.jsonl"
SEQUENCE_SAMPLE_INTERVAL_MS = 5000
RANKED_RULES = ("splat_zones", "tower_control", "rainmaker", "clam_blitz")


def load_annotations(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    annotations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        annotations.append(dict(json.loads(line)))
    return annotations


def split_selected(annotation: dict[str, object], splits: set[str] | None) -> bool:
    return splits is None or str(annotation.get("split", "eval")) in splits


def empty_metrics(metric_names: Iterable[str]) -> dict[str, dict[str, object]]:
    return {name: {"correct": 0, "total": 0, "accuracy": None} for name in metric_names}


def build_coverage(annotations: Iterable[dict[str, object]], splits: set[str] | None = None) -> dict[str, object]:
    by_rule: dict[str, dict[str, int]] = {
        rule: {
            "annotations": 0,
            "counterSamples": 0,
            "sideCountLabels": 0,
            "counterLabels": 0,
            "objectiveLabels": 0,
            "goalStateLabels": 0,
            "markerLabels": 0,
            "visibleMarkerLabels": 0,
            "staticObjectiveState": {},
            "staticMarkerState": {},
        }
        for rule in RANKED_RULES
    }
    by_split: dict[str, int] = {}
    evaluated_annotations = 0

    for annotation in annotations:
        if not split_selected(annotation, splits):
            continue
        evaluated_annotations += 1
        split_name = str(annotation.get("split", "eval"))
        by_split[split_name] = by_split.get(split_name, 0) + 1
        rule = str(annotation.get("rule", "rule_unknown"))
        item = by_rule.setdefault(
            rule,
            {
                "annotations": 0,
                "counterSamples": 0,
                "sideCountLabels": 0,
                "counterLabels": 0,
                "objectiveLabels": 0,
                "goalStateLabels": 0,
                "markerLabels": 0,
                "visibleMarkerLabels": 0,
                "staticObjectiveState": {},
                "staticMarkerState": {},
            },
        )
        item["annotations"] += 1

        expected = annotation.get("expected")
        if not isinstance(expected, dict):
            continue
        side_counts = expected.get("sideCounts")
        if isinstance(side_counts, dict):
            item["sideCountLabels"] += sum(1 for side in ("left", "right") if side in side_counts)
        counter = expected.get("counter")
        if isinstance(counter, dict):
            if isinstance(counter.get("ourCount"), int) and isinstance(counter.get("enemyCount"), int):
                item["counterSamples"] += 1
            item["counterLabels"] += sum(1 for field_name in ("ourCount", "enemyCount", "leader", "progressDirection") if field_name in counter)

        objective = expected.get("objective")
        if isinstance(objective, dict):
            item["objectiveLabels"] += sum(1 for field_name in ("zoneControl", "towerOwner", "rainmakerState") if field_name in objective)
            static_objective_state = item.get("staticObjectiveState")
            if isinstance(static_objective_state, dict):
                for label in _static_objective_labels(objective):
                    _increment_coverage(static_objective_state, label)
            if "goalState" in objective:
                item["goalStateLabels"] += 1
                goal_state = objective.get("goalState")
                if isinstance(goal_state, dict) and isinstance(static_objective_state, dict):
                    _increment_coverage(static_objective_state, f"goalState.our={goal_state.get('ourGoal')}.enemy={goal_state.get('enemyGoal')}")
            static_marker_state = item.get("staticMarkerState")
            for position in _expected_marker_positions(objective):
                item["markerLabels"] += 1
                if _marker_expected_visible(position):
                    item["visibleMarkerLabels"] += 1
                if isinstance(static_marker_state, dict):
                    _increment_coverage(static_marker_state, _static_marker_label(position))

    return {
        "evaluatedAnnotationSamples": evaluated_annotations,
        "bySplit": by_split,
        "byRule": by_rule,
    }


def annotation_image_path(annotation: dict[str, object], annotations_path: Path = DEFAULT_ANNOTATIONS) -> Path | None:
    image_value = annotation.get("image")
    if not isinstance(image_value, str):
        return None
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return annotations_path.parent / image_path


def expected_counter(annotation: dict[str, object]) -> dict[str, int | str] | None:
    expected = annotation.get("expected")
    if not isinstance(expected, dict):
        return None
    counter = expected.get("counter")
    if not isinstance(counter, dict):
        return None
    our = counter.get("ourCount")
    enemy = counter.get("enemyCount")
    leader = counter.get("leader")
    if not isinstance(our, int) or not isinstance(enemy, int):
        return None
    if not isinstance(leader, str):
        leader = "unknown"
    return {"ourCount": our, "enemyCount": enemy, "leader": leader}


def _expected_marker_positions(objective: dict[str, object]) -> list[object]:
    positions: list[object] = []
    if "screenPosition" in objective:
        positions.append(objective.get("screenPosition"))
    if "position" in objective:
        positions.append(objective.get("position"))
    clam_state = objective.get("clamState")
    if isinstance(clam_state, dict) and "visiblePowerClam" in clam_state:
        positions.append(clam_state.get("visiblePowerClam"))
    return positions


def _marker_expected_visible(position: object) -> bool:
    if position is None:
        return False
    if not isinstance(position, dict):
        return False
    return bool(position.get("visible", True))


def _static_objective_labels(objective: dict[str, object]) -> list[str]:
    labels: list[str] = []
    for field_name in ("zoneControl", "towerOwner", "rainmakerState"):
        if field_name in objective:
            labels.append(f"{field_name}={objective.get(field_name)}")
    return labels


def _static_marker_label(position: object) -> str:
    if position is None:
        return "null"
    if not isinstance(position, dict):
        return "other"
    if bool(position.get("visible", True)):
        marker_type = position.get("type")
        return f"visible:{marker_type}" if isinstance(marker_type, str) else "visible"
    distance_class = position.get("distanceClass")
    return f"offscreen:{distance_class}" if isinstance(distance_class, str) else "offscreen"


def evaluate_annotations(annotations: Iterable[dict[str, object]], detector: RankedObjectiveDetector, splits: set[str] | None = None) -> dict[str, object]:
    metrics = {
        "side_count": {"correct": 0, "total": 0},
        "count": {"correct": 0, "total": 0},
        "leader": {"correct": 0, "total": 0},
        "progress_direction": {"correct": 0, "total": 0},
        "objective_state": {"correct": 0, "total": 0},
        "goal_state": {"correct": 0, "total": 0},
        "marker": {"correct": 0, "total": 0},
    }
    mistakes: list[dict[str, object]] = []

    for annotation in annotations:
        if splits is not None and str(annotation.get("split", "eval")) not in splits:
            continue
        image_path = annotation_image_path(annotation)
        if image_path is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            mistakes.append({"id": annotation.get("id", annotation.get("image")), "type": "missing_image", "image": str(image_path)})
            continue

        context = TrackerContext.from_mapping(
            {
                "rule": annotation.get("rule", "rule_unknown"),
                "ally_side": annotation.get("ally_side", "right"),
                "match_started": annotation.get("match_started", True),
            }
        )
        reading = detector.read_frame(frame, context=context, timestamp=float(annotation.get("timestamp", 0.0)), frame_index=int(annotation.get("frame_index", -1)))
        expected = annotation.get("expected")
        if not isinstance(expected, dict):
            continue
        side_counts = expected.get("sideCounts")
        if isinstance(side_counts, dict):
            for side_name in ("left", "right"):
                if side_name not in side_counts:
                    continue
                target = side_counts[side_name]
                raw_side = reading.raw.get(side_name)
                actual = None
                if isinstance(raw_side, dict):
                    raw_count = raw_side.get("count")
                    if isinstance(raw_count, dict):
                        actual = raw_count.get("value")
                metrics["side_count"]["total"] += 1
                ok = isinstance(actual, int) and isinstance(target, int) and abs(actual - target) <= 1
                if ok:
                    metrics["side_count"]["correct"] += 1
                else:
                    mistakes.append({"id": annotation.get("id"), "type": f"{side_name}Count", "expected": target, "actual": actual})
        counter = expected.get("counter")
        if isinstance(counter, dict):
            for field_name in ("ourCount", "enemyCount"):
                if field_name in counter:
                    metrics["count"]["total"] += 1
                    actual = getattr(reading.counter, field_name)
                    target = counter[field_name]
                    ok = isinstance(actual, int) and isinstance(target, int) and abs(actual - target) <= 1
                    if ok:
                        metrics["count"]["correct"] += 1
                    else:
                        mistakes.append({"id": annotation.get("id"), "type": field_name, "expected": target, "actual": actual})
            if "leader" in counter:
                metrics["leader"]["total"] += 1
                if reading.counter.leader == counter["leader"]:
                    metrics["leader"]["correct"] += 1
                else:
                    mistakes.append({"id": annotation.get("id"), "type": "leader", "expected": counter["leader"], "actual": reading.counter.leader})
            if "progressDirection" in counter:
                metrics["progress_direction"]["total"] += 1
                if reading.counter.progressDirection == counter["progressDirection"]:
                    metrics["progress_direction"]["correct"] += 1
                else:
                    mistakes.append(
                        {
                            "id": annotation.get("id"),
                            "type": "progressDirection",
                            "expected": counter["progressDirection"],
                            "actual": reading.counter.progressDirection,
                        }
                    )

        objective = expected.get("objective")
        if isinstance(objective, dict):
            evaluate_static_objective(annotation.get("id"), objective, reading.objective, metrics, mistakes)

    for item in metrics.values():
        total = item["total"]
        item["accuracy"] = None if total == 0 else item["correct"] / total
    return {"metrics": metrics, "mistakes": mistakes}


def evaluate_static_objective(
    annotation_id: object,
    expected: dict[str, object],
    actual: dict[str, object],
    metrics: dict[str, dict[str, object]],
    mistakes: list[dict[str, object]],
) -> None:
    for field_name in ("zoneControl", "towerOwner", "rainmakerState"):
        if field_name not in expected:
            continue
        metrics["objective_state"]["total"] += 1
        if actual.get(field_name) == expected.get(field_name):
            metrics["objective_state"]["correct"] += 1
        else:
            mistakes.append({"id": annotation_id, "type": field_name, "expected": expected.get(field_name), "actual": actual.get(field_name)})

    if "goalState" in expected:
        metrics["goal_state"]["total"] += 1
        if actual.get("goalState") == expected.get("goalState"):
            metrics["goal_state"]["correct"] += 1
        else:
            mistakes.append({"id": annotation_id, "type": "goalState", "expected": expected.get("goalState"), "actual": actual.get("goalState")})

    if "screenPosition" in expected:
        evaluate_marker(annotation_id, expected.get("screenPosition"), actual.get("screenPosition"), metrics, mistakes, "screenPosition")
    if "position" in expected:
        evaluate_marker(annotation_id, expected.get("position"), actual.get("position"), metrics, mistakes, "position")

    clam_state = expected.get("clamState")
    actual_clam_state = actual.get("clamState")
    if isinstance(clam_state, dict) and "visiblePowerClam" in clam_state:
        actual_power_clam = actual_clam_state.get("visiblePowerClam") if isinstance(actual_clam_state, dict) else None
        evaluate_marker(annotation_id, clam_state.get("visiblePowerClam"), actual_power_clam, metrics, mistakes, "visiblePowerClam")


def evaluate_marker(
    annotation_id: object,
    expected_position: object,
    actual_position: object,
    metrics: dict[str, dict[str, object]],
    mistakes: list[dict[str, object]],
    field_name: str,
) -> None:
    metrics["marker"]["total"] += 1
    ok = marker_matches(expected_position, actual_position)
    if ok:
        metrics["marker"]["correct"] += 1
    else:
        mistakes.append({"id": annotation_id, "type": field_name, "expected": expected_position, "actual": actual_position})


def marker_matches(expected_position: object, actual_position: object) -> bool:
    if expected_position is None:
        if actual_position is None:
            return True
        if isinstance(actual_position, dict):
            return bool(actual_position.get("visible")) is False
        return False
    if not isinstance(expected_position, dict):
        return expected_position == actual_position

    expected_visible = bool(expected_position.get("visible", True))
    if actual_position is None:
        return expected_visible is False
    if not isinstance(actual_position, dict):
        return False

    actual_visible = bool(actual_position.get("visible", True))
    if actual_visible != expected_visible:
        return False
    if not expected_visible:
        return True

    ex = expected_position.get("x")
    ey = expected_position.get("y")
    ax = actual_position.get("x")
    ay = actual_position.get("y")
    return all(isinstance(value, (int, float)) for value in (ex, ey, ax, ay)) and abs(float(ax) - float(ex)) <= 0.10 and abs(float(ay) - float(ey)) <= 0.10


def evaluate_sequences(annotations: Iterable[dict[str, object]], detector: RankedObjectiveDetector, splits: set[str] | None = None) -> dict[str, object]:
    metrics = {
        "progress_direction": {"correct": 0, "total": 0},
        "objective_state": {"correct": 0, "total": 0},
        "sequence_count": {"correct": 0, "total": 0},
    }
    mistakes: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    coverage = {
        "progressDirection": {},
        "objectiveState": {},
    }

    for annotation in annotations:
        if splits is not None and str(annotation.get("split", "eval")) not in splits:
            continue
        if expected_counter(annotation) is None:
            continue
        video = str(annotation.get("video", ""))
        rule = str(annotation.get("rule", "rule_unknown"))
        ally_side = str(annotation.get("ally_side", "right"))
        groups.setdefault((video, rule, ally_side), []).append(annotation)

    evaluated_segments = 0
    for (video, rule, ally_side), group in groups.items():
        group.sort(key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
        segment: list[dict[str, object]] = []
        previous_counter: dict[str, int | str] | None = None
        for annotation in group:
            current_counter = expected_counter(annotation)
            if current_counter is None:
                continue
            if previous_counter is not None and _is_counter_reset(previous_counter, current_counter):
                evaluated_segments += _evaluate_sequence_segment(segment, detector, metrics, mistakes, coverage)
                segment = []
            segment.append(annotation)
            previous_counter = current_counter
        evaluated_segments += _evaluate_sequence_segment(segment, detector, metrics, mistakes, coverage)

    for item in metrics.values():
        total = item["total"]
        item["accuracy"] = None if total == 0 else item["correct"] / total
    return {"metrics": metrics, "mistakes": mistakes, "segments": evaluated_segments, "sequenceCoverage": coverage}


def evaluate_annotations_by_rule(annotations: Iterable[dict[str, object]], detector: RankedObjectiveDetector, splits: set[str] | None = None) -> dict[str, object]:
    rows = list(annotations)
    result: dict[str, object] = {}
    metric_names = ("side_count", "count", "leader", "progress_direction", "objective_state", "goal_state", "marker")
    for rule in RANKED_RULES:
        rule_rows = [annotation for annotation in rows if str(annotation.get("rule", "rule_unknown")) == rule]
        if rule_rows:
            result[rule] = evaluate_annotations(rule_rows, detector, splits=splits)
        else:
            result[rule] = {"metrics": empty_metrics(metric_names), "mistakes": []}
    return result


def evaluate_sequences_by_rule(annotations: Iterable[dict[str, object]], detector: RankedObjectiveDetector, splits: set[str] | None = None) -> dict[str, object]:
    rows = list(annotations)
    result: dict[str, object] = {}
    metric_names = ("progress_direction", "objective_state", "sequence_count")
    for rule in RANKED_RULES:
        rule_rows = [annotation for annotation in rows if str(annotation.get("rule", "rule_unknown")) == rule]
        if rule_rows:
            result[rule] = evaluate_sequences(rule_rows, detector, splits=splits)
        else:
            result[rule] = {
                "metrics": empty_metrics(metric_names),
                "mistakes": [],
                "segments": 0,
                "sequenceCoverage": {"progressDirection": {}, "objectiveState": {}},
            }
    return result


def coverage_requirement_failures(
    coverage: dict[str, object],
    per_rule_sequence_result: dict[str, object],
    require_full_objective_states: bool = False,
) -> list[str]:
    failures: list[str] = []
    by_rule = coverage.get("byRule")
    if not isinstance(by_rule, dict):
        return ["missing coverage.byRule"]

    for rule in RANKED_RULES:
        item = by_rule.get(rule)
        if not isinstance(item, dict):
            failures.append(f"{rule}: missing coverage")
            continue
        if int(item.get("annotations", 0) or 0) <= 0:
            failures.append(f"{rule}: no annotations")
        if int(item.get("counterSamples", 0) or 0) <= 0:
            failures.append(f"{rule}: no counter samples")
        if int(item.get("sideCountLabels", 0) or 0) <= 0:
            failures.append(f"{rule}: no side-count labels")

        sequence = per_rule_sequence_result.get(rule)
        if not isinstance(sequence, dict):
            failures.append(f"{rule}: missing sequence result")
            continue
        metrics = sequence.get("metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{rule}: missing sequence metrics")
            continue
        progress = metrics.get("progress_direction")
        objective = metrics.get("objective_state")
        if not isinstance(progress, dict) or int(progress.get("total", 0) or 0) <= 0:
            failures.append(f"{rule}: no sequence progressDirection labels")
        if not isinstance(objective, dict) or int(objective.get("total", 0) or 0) <= 0:
            failures.append(f"{rule}: no sequence objective_state labels")

    if require_full_objective_states:
        expected_states = {
            "splat_zones": {"zoneControl=our_control", "zoneControl=enemy_control", "zoneControl=unknown"},
            "tower_control": {"towerOwner=our", "towerOwner=enemy", "towerOwner=neutral"},
            "rainmaker": {"rainmakerState=our_carrier", "rainmakerState=enemy_carrier", "rainmakerState=unknown"},
            "clam_blitz": {
                "goalState.our=open.enemy=closed",
                "goalState.our=closed.enemy=open",
                "goalState.our=closed.enemy=closed",
            },
        }
        for rule, states in expected_states.items():
            sequence = per_rule_sequence_result.get(rule)
            sequence_coverage = sequence.get("sequenceCoverage") if isinstance(sequence, dict) else None
            objective_state = sequence_coverage.get("objectiveState") if isinstance(sequence_coverage, dict) else None
            actual_states = set(objective_state) if isinstance(objective_state, dict) else set()
            missing = sorted(states - actual_states)
            if missing:
                failures.append(f"{rule}: missing sequence objective states {missing}")

    return failures


def static_coverage_gaps(coverage: dict[str, object]) -> list[str]:
    gaps: list[str] = []
    by_rule = coverage.get("byRule")
    if not isinstance(by_rule, dict):
        return ["missing coverage.byRule"]

    for rule in ("tower_control", "rainmaker"):
        item = by_rule.get(rule)
        if not isinstance(item, dict):
            continue
        if int(item.get("visibleMarkerLabels", 0) or 0) == 0:
            gaps.append(f"{rule}: no visible screen marker labels")

    splat_zones = by_rule.get("splat_zones")
    if isinstance(splat_zones, dict):
        static_state = splat_zones.get("staticObjectiveState")
        if isinstance(static_state, dict):
            if not any(key in static_state for key in ("zoneControl=our_control", "zoneControl=enemy_control", "zoneControl=neutral", "zoneControl=contested")):
                gaps.append("splat_zones: no non-unknown static zoneControl labels")

    tower = by_rule.get("tower_control")
    if isinstance(tower, dict):
        static_state = tower.get("staticObjectiveState")
        if isinstance(static_state, dict):
            if not any(key in static_state for key in ("towerOwner=our", "towerOwner=enemy")):
                gaps.append("tower_control: no non-neutral static towerOwner labels")

    rainmaker = by_rule.get("rainmaker")
    if isinstance(rainmaker, dict):
        static_state = rainmaker.get("staticObjectiveState")
        if isinstance(static_state, dict):
            if not any(key in static_state for key in ("rainmakerState=our_carrier", "rainmakerState=enemy_carrier", "rainmakerState=free", "rainmakerState=shielded")):
                gaps.append("rainmaker: no non-unknown static rainmakerState labels")

    clam = by_rule.get("clam_blitz")
    if isinstance(clam, dict):
        static_state = clam.get("staticObjectiveState")
        if isinstance(static_state, dict):
            if not any(key in static_state for key in ("goalState.our=open.enemy=closed", "goalState.our=closed.enemy=open")):
                gaps.append("clam_blitz: no static open goalState labels")

    return gaps


def _evaluate_sequence_segment(
    segment: list[dict[str, object]],
    detector: RankedObjectiveDetector,
    metrics: dict[str, dict[str, object]],
    mistakes: list[dict[str, object]],
    coverage: dict[str, dict[str, int]],
) -> int:
    if len(segment) < 2:
        return 0
    rule = str(segment[0].get("rule", "rule_unknown"))
    ally_side = str(segment[0].get("ally_side", "right"))
    context = TrackerContext.from_mapping({"rule": rule, "ally_side": ally_side, "match_started": True})
    tracker = RankedObjectiveTracker(detector, fresh_ms=SEQUENCE_SAMPLE_INTERVAL_MS + 1000, stale_ms=SEQUENCE_SAMPLE_INTERVAL_MS + 1000)
    previous_expected: dict[str, int | str] | None = None

    for index, annotation in enumerate(segment):
        image_path = annotation_image_path(annotation)
        if image_path is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            mistakes.append({"id": annotation.get("id"), "type": "sequence_missing_image", "image": str(image_path)})
            continue
        timestamp = index * SEQUENCE_SAMPLE_INTERVAL_MS / 1000.0
        update = tracker.read_frame(frame, context=context, timestamp=timestamp, frame_index=int(annotation.get("frame_index", -1)))
        expected = expected_counter(annotation)
        if expected is None:
            continue

        metrics["sequence_count"]["total"] += 2
        count_ok = True
        for field_name in ("ourCount", "enemyCount"):
            actual = getattr(update.reading.counter, field_name)
            target = expected[field_name]
            ok = isinstance(actual, int) and isinstance(target, int) and abs(actual - target) <= 1
            if ok:
                metrics["sequence_count"]["correct"] += 1
            else:
                count_ok = False
                mistakes.append({"id": annotation.get("id"), "type": f"sequence_{field_name}", "expected": target, "actual": actual})

        if previous_expected is None or not count_ok:
            previous_expected = expected
            continue

        expected_direction = compute_progress_direction(
            int(expected["ourCount"]) - int(previous_expected["ourCount"]),
            int(expected["enemyCount"]) - int(previous_expected["enemyCount"]),
        )
        _increment_coverage(coverage["progressDirection"], expected_direction)
        metrics["progress_direction"]["total"] += 1
        if update.reading.counter.progressDirection == expected_direction:
            metrics["progress_direction"]["correct"] += 1
        else:
            mistakes.append(
                {
                    "id": annotation.get("id"),
                    "type": "sequence_progressDirection",
                    "expected": expected_direction,
                    "actual": update.reading.counter.progressDirection,
                }
            )

        expected_objective = expected_sequence_objective_state(annotation, rule, expected_direction)
        actual_objective = actual_objective_state(update.reading.objective)
        if expected_objective is not None:
            _increment_coverage(coverage["objectiveState"], _objective_state_label(expected_objective))
            metrics["objective_state"]["total"] += 1
            if actual_objective == expected_objective:
                metrics["objective_state"]["correct"] += 1
            else:
                mistakes.append({"id": annotation.get("id"), "type": "sequence_objective_state", "expected": expected_objective, "actual": actual_objective})

        previous_expected = expected
    return 1


def _increment_coverage(items: dict[str, int], key: str) -> None:
    items[key] = items.get(key, 0) + 1


def _objective_state_label(state: object) -> str:
    if not isinstance(state, dict):
        return str(state)
    if "zoneControl" in state:
        return f"zoneControl={state.get('zoneControl')}"
    if "towerOwner" in state:
        return f"towerOwner={state.get('towerOwner')}"
    if "rainmakerState" in state:
        return f"rainmakerState={state.get('rainmakerState')}"
    goal_state = state.get("goalState")
    if isinstance(goal_state, dict):
        return f"goalState.our={goal_state.get('ourGoal')}.enemy={goal_state.get('enemyGoal')}"
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _is_counter_reset(previous: dict[str, int | str], current: dict[str, int | str]) -> bool:
    return int(current["ourCount"]) > int(previous["ourCount"]) + 2 or int(current["enemyCount"]) > int(previous["enemyCount"]) + 2


def expected_sequence_objective_state(annotation: dict[str, object], rule: str, progress_direction: str) -> object | None:
    if rule == "clam_blitz":
        expected = annotation.get("expected")
        objective = expected.get("objective") if isinstance(expected, dict) else None
        goal_state = objective.get("goalState") if isinstance(objective, dict) else None
        if isinstance(goal_state, dict) and (goal_state.get("ourGoal") == "open" or goal_state.get("enemyGoal") == "open"):
            return {"goalState": {"ourGoal": goal_state.get("ourGoal"), "enemyGoal": goal_state.get("enemyGoal")}}
    return expected_objective_state(rule, progress_direction)


def expected_objective_state(rule: str, progress_direction: str) -> object | None:
    if rule == "splat_zones":
        if progress_direction == "our_count_decreasing":
            return {"zoneControl": "our_control"}
        if progress_direction == "enemy_count_decreasing":
            return {"zoneControl": "enemy_control"}
        return {"zoneControl": "unknown"}
    if rule == "tower_control":
        if progress_direction == "our_count_decreasing":
            return {"towerOwner": "our"}
        if progress_direction == "enemy_count_decreasing":
            return {"towerOwner": "enemy"}
        return {"towerOwner": "neutral"}
    if rule == "rainmaker":
        if progress_direction == "our_count_decreasing":
            return {"rainmakerState": "our_carrier"}
        if progress_direction == "enemy_count_decreasing":
            return {"rainmakerState": "enemy_carrier"}
        return {"rainmakerState": "unknown"}
    if rule == "clam_blitz":
        if progress_direction == "our_count_decreasing":
            return {"goalState": {"ourGoal": "closed", "enemyGoal": "open"}}
        if progress_direction == "enemy_count_decreasing":
            return {"goalState": {"ourGoal": "open", "enemyGoal": "closed"}}
        return {"goalState": {"ourGoal": "closed", "enemyGoal": "closed"}}
    return None


def actual_objective_state(objective: dict[str, object]) -> object | None:
    kind = objective.get("kind")
    if kind == "splat_zones":
        return {"zoneControl": objective.get("zoneControl")}
    if kind == "tower_control":
        return {"towerOwner": objective.get("towerOwner")}
    if kind == "rainmaker":
        return {"rainmakerState": objective.get("rainmakerState")}
    if kind == "clam_blitz":
        return {"goalState": objective.get("goalState")}
    return None


def synthetic_tracker_checks() -> list[str]:
    failures: list[str] = []
    detector = RankedObjectiveDetector(use_timer_ocr=False)
    tracker = RankedObjectiveTracker(detector)
    template = RankedObjectiveReading(
        timestamp=0.0,
        frame_index=0,
        timestampMs=0,
        rule="tower_control",
        counter=CounterState(
            ourCount=70,
            enemyCount=50,
            leader="enemy",
            isOvertime=False,
            isKnockout=False,
            confidence=CounterConfidence(ourCount=0.95, enemyCount=0.95),
            freshness="fresh",
            lastSeenMsAgo=0,
        ),
        objective={"kind": "tower_control", "confidence": 0.0},
        quality={"overallConfidence": 0.9, "source": ["synthetic"], "warnings": []},
    )

    sequence = [
        (0, 70, 50),
        (1000, 70, 49),
        (2000, 70, 82),
        (3000, 70, 48),
        (5000, 70, 46),
        (6000, 70, 44),
    ]
    last_update = None
    for timestamp_ms, our_count, enemy_count in sequence:
        raw = replace(
            template,
            timestamp=timestamp_ms / 1000.0,
            timestampMs=timestamp_ms,
            counter=replace(
                template.counter,
                ourCount=our_count,
                enemyCount=enemy_count,
                confidence=CounterConfidence(ourCount=0.95, enemyCount=0.95),
            ),
        )
        last_update = tracker.update(raw)
        if timestamp_ms == 2000 and last_update.reading.counter.enemyCount == 82:
            failures.append("single-frame impossible enemy count jump was accepted")

    if last_update is None:
        failures.append("synthetic tracker produced no updates")
        return failures
    if last_update.reading.counter.progressDirection != "enemy_count_decreasing":
        failures.append(f"expected enemy_count_decreasing, got {last_update.reading.counter.progressDirection}")
    if last_update.reading.objective.get("towerOwner") != "enemy":
        failures.append(f"expected towerOwner enemy, got {last_update.reading.objective.get('towerOwner')}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ranked objective tracker samples and state logic.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--split", default="eval", help="Comma separated annotation splits to evaluate, or 'all'.")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail if the evaluated annotations do not cover all ranked rules and key sequence objective states.",
    )
    args = parser.parse_args()

    detector = RankedObjectiveDetector(templates_dir=args.templates_dir, use_timer_ocr=False)
    annotations = load_annotations(args.annotations)
    splits = None if args.split == "all" else {item.strip() for item in args.split.split(",") if item.strip()}
    coverage = build_coverage(annotations, splits=splits)
    annotation_result = evaluate_annotations(annotations, detector, splits=splits)
    sequence_result = evaluate_sequences(annotations, detector, splits=splits)
    per_rule_annotation_result = evaluate_annotations_by_rule(annotations, detector, splits=splits)
    per_rule_sequence_result = evaluate_sequences_by_rule(annotations, detector, splits=splits)
    static_gaps = static_coverage_gaps(coverage)
    coverage_failures = coverage_requirement_failures(
        coverage,
        per_rule_sequence_result,
        require_full_objective_states=args.require_full_coverage,
    )
    synthetic_failures = synthetic_tracker_checks()
    result = {
        "annotation_samples": len(annotations),
        "evaluated_splits": "all" if splits is None else sorted(splits),
        "coverage": coverage,
        "annotation_result": annotation_result,
        "sequence_result": sequence_result,
        "per_rule_annotation_result": per_rule_annotation_result,
        "per_rule_sequence_result": per_rule_sequence_result,
        "coverage_requirements": {
            "passed": len(coverage_failures) == 0,
            "requireFullObjectiveStates": bool(args.require_full_coverage),
            "failures": coverage_failures,
        },
        "static_coverage_gaps": static_gaps,
        "synthetic": {
            "passed": len(synthetic_failures) == 0,
            "failures": synthetic_failures,
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(text + "\n", encoding="utf-8")
    if synthetic_failures or (args.require_full_coverage and coverage_failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
