"""Small Windows-safe wrappers around the existing atomic JSON writer."""
from __future__ import annotations

import os
import time
from pathlib import Path

from weapon_lamp_detect.match_data import atomic_json as _atomic_json


def atomic_json(path, value):
    # Windows readers can briefly prevent replace(); retain old data and retry.
    for attempt in range(6):
        try:
            return _atomic_json(path, value)
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(.025 * (attempt + 1))


class FileLock:
    """Nonblocking OS lock released on crash; no stale PID lockfile removal."""
    def __init__(self, path, message):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise ValueError(message) from exc

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
