"""Source-side mutex for Windows/WSL shared video sessions.

The WSL UNC provider need not implement Windows byte-range locks. Atomic
directory creation still coordinates review processes on both operating systems.
Never steal an existing lock: after a crash an operator must confirm that all
writers have stopped before removing the marker.
"""
from __future__ import annotations

import json
import os
import socket
import uuid
import warnings
from pathlib import Path

from weapon_lamp_detect.match_data import now, read_json


class VideoSaveBusy(ValueError):
    pass


class VideoSaveLock:
    def __init__(self, directory):
        self.path = Path(directory) / 'annotation.review.lock'
        self.owner_path = self.path / 'owner.json'
        self.token = uuid.uuid4().hex
        self.closed = False
        try:
            self.path.mkdir()  # Atomic, including across Windows / WSL UNC.
        except FileExistsError as exc:
            try:
                owner = read_json(self.owner_path)
                detail = f"host={owner.get('host')} / pid={owner.get('pid')} / 開始={owner.get('created_at')}"
            except (OSError, ValueError, TypeError, AttributeError):
                detail = '所有者情報なし（取得直後または中断されたロック）'
            raise VideoSaveBusy(
                f'元動画ラベルの保存ロックを取得できません。再試行してください。{detail}\n'
                f'ロック: {self.path}\n'
                '残留している場合は、全確認アプリを終了してからREADMEの復旧手順を実行してください。'
            ) from exc
        except OSError as exc:
            # A permissions/provider error is NOT evidence of an active writer.
            raise OSError(f'元動画ラベルの保存ロックを作成できません: {self.path}: {exc}') from exc
        try:
            with self.owner_path.open('x', encoding='utf-8') as stream:
                json.dump({'token': self.token, 'host': socket.gethostname(),
                           'pid': os.getpid(), 'created_at': now()}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            # Only this process created the directory; nobody else may acquire it.
            try:
                self.owner_path.unlink(missing_ok=True)
                self.path.rmdir()
            except OSError as cleanup:
                warnings.warn(f'保存ロック初期化失敗後のマーカーが残っています: {self.path}: {cleanup}')
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if read_json(self.owner_path).get('token') != self.token:
                raise ValueError('所有者が変わったため削除しません')
            self.owner_path.unlink()
            self.path.rmdir()
        except (OSError, ValueError, AttributeError) as exc:
            # A completed GT commit must not be misreported as a failed save.
            # Retain unowned markers; never recursively delete this directory.
            warnings.warn(f'保存ロックを解除できませんでした: {self.path}: {exc}')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
