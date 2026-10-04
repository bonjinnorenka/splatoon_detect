"""Separate acquisition records from human labels to avoid concurrent overwrites."""
from __future__ import annotations

import copy
import hashlib
import re
import threading
from pathlib import Path

from live_weapon_collect.io_utils import FileLock, atomic_json
from weapon_lamp_detect.match_data import (
    SLOTS, STATUSES, default_geometry, local_path, now, read_json, validate_geometry,
)


class LiveStore:
    def __init__(self, directory, catalog):
        self.directory = local_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.lock = threading.RLock()

    def match_dir(self, match_id):
        if not isinstance(match_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", match_id):
            raise ValueError("不正なmatch idです")
        return self.directory / "matches" / match_id

    def capture(self, match_id):
        value = read_json(self.match_dir(match_id) / "capture.json")
        if value["match_id"] != match_id:
            raise ValueError("match idが一致しません")
        return value

    def get(self, match_id):
        capture = self.capture(match_id)
        path = self.match_dir(match_id) / "annotation.json"
        label = read_json(path) if path.exists() else {
            "schema_version": 1, "match_id": match_id, "revision": 0,
            "ally_side": "right", "reference_ordinal": 0, "geometry": default_geometry(),
            "confirmed": False, "rejected": False, "opening_reviewed": False, "notes": "",
            "slots": {s: {"weapon": None, "status": "unknown", "reviewed": False} for s in SLOTS},
            "created_at": capture["created_at"], "updated_at": capture["created_at"],
        }
        return {"capture": capture, "annotation": label}

    def list(self):
        result = []
        for path in sorted((self.directory / "matches").glob("*/capture.json")):
            value = self.get(path.parent.name)
            c, a = value["capture"], value["annotation"]
            result.append({"match_id": c["match_id"], "created_at": c["created_at"],
                           "session_id": c["session_id"], "frame_count": len(c["frames"]),
                           "capture_status": c["status"], "source": c["source"],
                           "needs_review": c.get("needs_review", False) and not a["opening_reviewed"],
                           "confirmed": a["confirmed"], "rejected": a["rejected"],
                           "revision": a["revision"], "updated_at": a["updated_at"]})
        return result

    def frame(self, match_id, ordinal):
        c = self.capture(match_id)
        ordinal = int(ordinal)
        if not 0 <= ordinal < len(c["frames"]):
            raise ValueError("保存済みframeがありません")
        row = c["frames"][ordinal]
        path = (self.match_dir(match_id) / row["path"]).resolve()
        if not path.is_relative_to(self.match_dir(match_id).resolve()):
            raise ValueError("frame pathが不正です")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("保存frameが変更されています")
        return c, row, path

    def validate(self, value):
        if not isinstance(value, dict):
            raise ValueError("annotation objectが必要です")
        value = copy.deepcopy(value)
        c = self.capture(value["match_id"])
        if value.get("ally_side") not in {"left", "right"} or set(value.get("slots", {})) != set(SLOTS):
            raise ValueError("ally side / 8スロットが不正です")
        for key in ("confirmed", "rejected", "opening_reviewed"):
            if not isinstance(value.get(key), bool):
                raise ValueError(f"{key}はbooleanで指定してください")
        ordinal = value.get("reference_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(c["frames"]):
            raise ValueError("referenceは保存済みframeから選んでください")
        validate_geometry(value.get("geometry", {}), c["resolution"])
        for key, slot in value["slots"].items():
            if not isinstance(slot, dict) or slot.get("status") not in STATUSES or type(slot.get("reviewed")) is not bool:
                raise ValueError(f"{key}: status / reviewedが不正です")
            # Even hidden non-null invalid names must not enter saved annotations.
            if slot.get("weapon") is not None or slot["status"] == "labeled":
                try:
                    resolved = self.catalog.resolve_label(slot.get("weapon"))
                except ValueError as exc:
                    raise ValueError(f"{key}: {exc}（保存していません）") from exc
            slot["weapon"] = resolved if slot["status"] == "labeled" else None
            if slot["status"] == "labeled":
                slot["weapon_class"] = self.catalog.entries[resolved]["weapon_class"]
            else:
                slot.pop("weapon_class", None)
        if value["confirmed"]:
            if value["rejected"] or not value["opening_reviewed"] or not all(s["reviewed"] for s in value["slots"].values()):
                raise ValueError("確定には開始HUDの確認と8スロットの人力確認が必要です")
            if c["status"] == "collecting":
                raise ValueError("収集完了後に確定してください（下書き保存は可能です）")
        notes = value.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError("notesが不正です")
        # Explicit allowlist: predictions/capture metadata cannot become ground truth.
        return {k: value[k] for k in ("match_id", "ally_side", "reference_ordinal", "geometry",
                                     "confirmed", "rejected", "opening_reviewed", "slots", "notes")}

    def save(self, value):
        with self.lock, FileLock(self.directory / "annotation.lock", "別アプリが保存中です。少し待って保存してください"):
            return self._save(value)

    def _save(self, value):
        old = self.get(value["match_id"])["annotation"]
        if type(value.get("revision")) is not int or value["revision"] != old["revision"]:
            raise ValueError("別画面で更新されています。再読込してください（上書きしていません）")
        clean = self.validate(value)
        directory = self.match_dir(clean["match_id"])
        clean.update(schema_version=1, revision=old["revision"] + 1,
                     created_at=old["created_at"], updated_at=now())
        atomic_json(directory / "history" / f'{old["revision"]:06d}.json', old)
        atomic_json(directory / "annotation.json", clean)
        atomic_json(self.directory / "resume.json", {"match_id": clean["match_id"]})
        return clean

    def undo(self, match_id, revision):
        with self.lock, FileLock(self.directory / "annotation.lock", "別アプリが保存中です。少し待ってください"):
            old = self.get(match_id)["annotation"]
            if revision != old["revision"]:
                raise ValueError("更新競合です。再読込してください")
            path = self.match_dir(match_id) / "history" / f'{old["revision"] - 1:06d}.json'
            if not path.exists():
                raise ValueError("戻せる履歴がありません")
            previous = read_json(path)
            previous["revision"] = old["revision"]
            return self._save(previous)

    def resume(self):
        path = self.directory / "resume.json"
        return read_json(path) if path.exists() else {}
