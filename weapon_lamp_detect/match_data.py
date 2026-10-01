"""Shared, versioned match annotations. Predictions never become ground truth."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATES = ROOT / "sample_data" / "Main Weapons"
SLOTS = tuple(f"{side}{i}" for side in ("left", "right") for i in range(4))
STATUSES = {"labeled", "unknown", "occluded", "skip"}
DEFAULT_HUD_OFFSET = 20.0
# Measured on the original 1920x1080 OBS frame: lamp centers ~565,655,745,835
# and ~1085,1175,1265,1355. The previous 134px crops overlapped adjacent icons.
HUD_CENTERS = {"left": (.294, .341, .388, .435), "right": (.565, .612, .659, .706)}
HUD_CROP_PROFILE = "obs_1080p_lamps_v2"


def default_geometry():
    return {f"{side}{i}": [cx, .067, .052, .092]
            for side, centers in HUD_CENTERS.items() for i, cx in enumerate(centers)}

CATEGORIES = dict(zip(
    ["シューター", "ローラー", "チャージャー", "スロッシャー", "スピナー", "マニューバー", "シェルター", "ブラスター", "フデ", "ストリンガー", "ワイパー"],
    ["shooter", "roller", "charger", "slosher", "splatling", "dualies", "brella", "blaster", "brush", "stringer", "splatana"],
))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def local_path(value):
    """Accept native Windows paths on Windows and translate drive paths on WSL."""
    value = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return Path(value).expanduser().resolve()


def discover_videos(inputs):
    paths = set()
    extensions = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}
    for value in inputs:
        path = local_path(value)
        if path.is_dir():
            paths.update(p.resolve() for p in path.rglob("*") if p.suffix.lower() in extensions)
        elif path.is_file():
            paths.add(path)
        else:
            raise ValueError(f"動画またはフォルダがありません: {path}")
    return sorted(paths)


def video_metadata(path):
    path = local_path(path)
    stat = path.stat()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f"動画metadataが不正です: {path}")
        # Bounded content fingerprint, independent of filename/path. No full 16GB hash.
        digest = hashlib.sha256(str(stat.st_size).encode())
        with path.open("rb") as stream:
            for position in (0, max(0, stat.st_size // 2 - 32768), max(0, stat.st_size - 65536)):
                stream.seek(position)
                digest.update(stream.read(65536))
        return {"video_id": digest.hexdigest()[:24], "path": str(path), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "duration": frames / fps, "fps": fps,
                "frame_count": frames, "width": int(cap.get(3)), "height": int(cap.get(4)),
                "identity_method": "sha256(size + first/middle/last 64KiB)"}
    finally:
        cap.release()


def frame_at(video, timestamp=None, frame_index=None):
    if frame_index is None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        frame_index = round(timestamp * video["fps"])
    frame_index = max(0, min(video["frame_count"] - 1, int(frame_index)))
    cap = cv2.VideoCapture(video["path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"frameを読めません: {frame_index}")
        return frame, frame_index, frame_index / video["fps"]
    finally:
        cap.release()


def image_bytes(frame, extension=".jpg"):
    ok, encoded = cv2.imencode(extension, frame)
    if not ok:
        raise ValueError("image encoding failed")
    return encoded.tobytes()


def write_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes(frame, path.suffix))


def squid_detector(ally_side="right"):
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    from squid_lamp_detect.squid_lamp import SquidLampDetector
    return SquidLampDetector(ally_side=ally_side)


def canonical_crop(crop):
    # Keep weapon aspect ratios when the corrected crop is narrower than the old ROI.
    height, width = crop.shape[:2]
    scale = min(134/width, 108/height)
    new_w, new_h = max(1, round(width*scale)), max(1, round(height*scale))
    resized = cv2.resize(crop, (new_w,new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((108,134,3), dtype=crop.dtype)
    x, y = (134-new_w)//2, (108-new_h)//2
    canvas[y:y+new_h,x:x+new_w] = resized
    return canvas


def calibrated_crop(frame, side, index, geometry=None):
    setting = (geometry or {}).get(f"{side}{index}", default_geometry()[f"{side}{index}"])
    cx, cy, width, height = setting
    h, w = frame.shape[:2]
    rect = (max(0, round((cx-width/2)*w)), max(0, round((cy-height/2)*h)),
            min(w, round((cx+width/2)*w)), min(h, round((cy+height/2)*h)))
    x1, y1, x2, y2 = rect
    if x2-x1 < 8 or y2-y1 < 8:
        raise ValueError("cropが小さすぎます")
    return frame[y1:y2, x1:x2], rect


def calibrated_slot(frame, reading, side, index, geometry=None):
    if reading.hud_state != "match":
        return reading.sides[side].slots[index]
    from squid_lamp_detect.squid_lamp import classify_slot
    crop, _ = calibrated_crop(frame, side, index, geometry)
    # These corrected boxes already surround one lamp; do not shrink them again.
    return classify_slot(crop, index, reading.sides[side].ink_color)


def validate_geometry(geometry, video):
    if not isinstance(geometry, dict) or set(geometry) - set(SLOTS):
        raise ValueError("invalid crop geometry")
    for setting in geometry.values():
        if not isinstance(setting, list) or len(setting) != 4 or not all(math.isfinite(float(n)) for n in setting):
            raise ValueError("crop geometry must be [cx,cy,width,height]")
        cx, cy, width, height = map(float, setting)
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and .005 <= width <= .25 and .005 <= height <= .4):
            raise ValueError("crop geometry out of range")
        if min(1, cx+width/2)-max(0, cx-width/2) < 8/video["width"] or min(1, cy+height/2)-max(0, cy-height/2) < 8/video["height"]:
            raise ValueError("crop geometry too small")


class Catalog:
    def __init__(self, template_dir=TEMPLATES, catalog_path=None):
        from weapon_lamp_detect.weapon_lamp import weapon_class_for_name
        self.entries = {p.stem: {"name": p.stem, "display_name": p.stem,
                                "weapon_class": weapon_class_for_name(p.stem), "official_template": str(p.resolve())}
                        for p in Path(template_dir).glob("*.png")}
        self.aliases = {}
        mapping = read_json(HERE / "weapon_aliases.json")
        bundled = HERE / "weapon_catalog.json"
        source = local_path(catalog_path) if catalog_path else (ROOT / "wepons.json")
        if catalog_path is not None and not source.is_file():
            raise ValueError(f"指定した武器catalogがありません: {source}")
        self.catalog_path = (source if source.is_file() else bundled).resolve()
        data = read_json(self.catalog_path)
        # Annotation choices come exclusively from the user's JSON, not template stems.
        # Keep template-only entries internally for existing annotations/evaluation.
        self.label_entries = []
        for item in data["weapons"]:
            jp = item["name"]
            en = mapping.get(jp, jp)
            entry = self.entries.setdefault(en, {"name": en, "official_template": None})
            entry.update(display_name=jp, category=item["category"], weapon_class=CATEGORIES[item["category"]])
            self.label_entries.append(dict(entry))
            self.aliases[jp] = en
        self.aliases.update({name: name for name in self.entries})
        self.label_names = {entry["name"] for entry in self.label_entries}

    def resolve(self, name):
        name = str(name).strip()
        if name not in self.aliases:
            raise ValueError(f"武器名がcatalogにありません: {name}")
        return self.aliases[name]

    def to_list(self):
        return [dict(entry) for entry in self.label_entries]

    def resolve_label(self, name):
        """Only JSON-listed weapons may be saved as new annotation labels."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("武器名が未入力です。JSONの候補から選択してください")
        resolved = self.aliases.get(name.strip())
        if resolved not in self.label_names:
            raise ValueError(f"無効な武器名「{name}」です。{self.catalog_path.name} の候補から選択してください")
        return resolved


def blank_match(video, start, end, source="manual", candidate=None, hud_offset=DEFAULT_HUD_OFFSET):
    start = max(0.0, min(float(start), video["duration"] - 1 / video["fps"]))
    end = min(video["duration"], max(start + 1 / video["fps"], float(end)))
    if not math.isfinite(hud_offset) or hud_offset < 0:
        raise ValueError("HUD offset must be finite and ≥ 0")
    ref_index = round(min(end - 1 / video["fps"], start + hud_offset) * video["fps"])
    ref = ref_index / video["fps"]
    return {"schema_version": 1, "match_id": f'{video["video_id"]}_{round(start * video["fps"]):09d}',
            "video_id": video["video_id"], "video_path": video["path"], "video_metadata": video,
            "start_timestamp": start, "end_timestamp": end, "reference_timestamp": ref,
            "reference_frame_index": ref_index, "reference_source": "initial_hud_offset", "ally_side": "right",
            "start_source": source, "start_candidate": candidate,
            "start_event": "stage_intro" if source == "hog_svm" else "manual_boundary",
            "hud_offset_seconds": hud_offset, "crop_profile": HUD_CROP_PROFILE,
            "boundaries_confirmed": False, "confirmed": False, "rejected": False,
            "slots": {slot: {"weapon": None, "status": "unknown", "reviewed": False} for slot in SLOTS},
            "geometry": default_geometry(), "notes": "", "created_at": now(), "updated_at": now(), "revision": 0}


def validate_match(value, video, catalog):
    value = dict(value)
    start, end, ref = (float(value[key]) for key in ("start_timestamp", "end_timestamp", "reference_timestamp"))
    if not all(math.isfinite(n) for n in (start, end, ref)) or not 0 <= start <= ref < end <= video["duration"]:
        raise ValueError("0 ≤ start ≤ reference < end ≤ duration が必要です")
    index = int(value["reference_frame_index"])
    if not 0 <= index < video["frame_count"] or abs(index / video["fps"] - ref) > 1 / video["fps"]:
        raise ValueError("reference timestampとframe indexが一致しません")
    if value.get("ally_side") not in {"left", "right"} or set(value["slots"]) != set(SLOTS):
        raise ValueError("ally side / 8 slotsが不正です")
    value["slots"] = {key: dict(slot) for key, slot in value["slots"].items()}
    geometry = value.get("geometry", {})
    validate_geometry(geometry, video)
    offset = float(value.get("hud_offset_seconds", DEFAULT_HUD_OFFSET))
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("HUD待ち時間が不正です")
    for key, slot in value["slots"].items():
        if slot["status"] not in STATUSES:
            raise ValueError("slot statusが不正です")
        try:
            if slot.get("weapon") is not None or slot["status"] == "labeled":
                resolved = catalog.resolve_label(slot.get("weapon"))
            slot["weapon"] = resolved if slot["status"] == "labeled" else None
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}（保存していません）") from exc
    if value.get("confirmed") and (not value.get("boundaries_confirmed") or not all(s.get("reviewed") for s in value["slots"].values())):
        raise ValueError("確定には開始・終了の確認と8スロットの人力入力が必要です")
    return value


class MatchStore:
    def __init__(self, directory, catalog):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.videos_path = self.directory / "videos.json"
        self.videos = read_json(self.videos_path) if self.videos_path.exists() else {}

    def add_videos(self, inputs):
        for path in discover_videos(inputs):
            video = video_metadata(path)
            self.videos[video["video_id"]] = video
        atomic_json(self.videos_path, self.videos)

    def upgrade_unedited_matches(self, hud_offset=DEFAULT_HUD_OFFSET):
        """Fix previous auto defaults only; preserve every human-edited revision."""
        updated = 0
        for match in self.matches():
            if match.get("revision") != 1 or match.get("confirmed") or match.get("rejected"):
                continue
            if match.get("geometry") or match.get("notes") or any(s.get("reviewed") for s in match["slots"].values()):
                continue
            video = self.videos[match["video_id"]]
            old_ref = min(match["end_timestamp"]-1/video["fps"], match["start_timestamp"]+5)
            if abs(match["reference_timestamp"]-old_ref) > 1/video["fps"]:
                continue
            defaults = blank_match(video, match["start_timestamp"], match["end_timestamp"],
                                   match["start_source"], match.get("start_candidate"), hud_offset)
            for key in ("reference_timestamp", "reference_frame_index", "reference_source", "start_event",
                        "hud_offset_seconds", "crop_profile", "geometry"):
                match[key] = defaults[key]
            self.save(match)
            updated += 1
        return updated

    def matches(self):
        rows = [read_json(path) for path in sorted((self.directory / "matches").glob("*.json"))]
        return sorted(rows, key=lambda m: (m["video_path"], m["start_timestamp"]))

    def path(self, match_id):
        if not re.fullmatch(r"[a-f0-9]{24}_[0-9]{9,}", match_id):
            raise ValueError("invalid match id")
        return self.directory / "matches" / (match_id + ".json")

    def save(self, value):
        path = self.path(value["match_id"])
        old = read_json(path) if path.exists() else None
        video = self.videos[value["video_id"]]
        if old and int(value.get("revision", -1)) != old["revision"]:
            raise ValueError("別画面で更新済みです。再読込してください")
        if old and old["video_id"] != value["video_id"]:
            raise ValueError("match video cannot change")
        value = validate_match(value, video, self.catalog)
        value.update(video_path=video["path"], video_metadata=video,
                     created_at=old["created_at"] if old else now(), updated_at=now(),
                     revision=(old["revision"] if old else 0) + 1)
        if old:
            atomic_json(self.directory / "history" / value["match_id"] / f'{old["revision"]:06d}.json', old)
        atomic_json(path, value)
        return value
