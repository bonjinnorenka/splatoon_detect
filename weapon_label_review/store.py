"""Index match slots by weapon; reuse original validation/history for corrections."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import cv2

from live_weapon_collect.io_utils import FileLock, atomic_json
from live_weapon_collect.store import LiveStore
from weapon_label_review.locking import VideoSaveLock
from weapon_lamp_detect.build_dataset import cv2_read
from weapon_lamp_detect.match_data import (
    Catalog, MatchStore, SLOTS, STATUSES, calibrated_crop, frame_at,
    image_bytes, local_path, now, read_json, video_metadata,
)

UNASSIGNED = "__unassigned__"


def signature(annotation, key):
    value = {"slot": annotation['slots'][key], "geometry": annotation.get('geometry', {}).get(key),
             "reference": annotation.get('reference_ordinal', annotation.get('reference_frame_index')),
             "confirmed": annotation['confirmed'], "rejected": annotation['rejected']}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ReviewStore:
    def __init__(self, data_dir, live_dirs=(), video_dirs=(), catalog=None, video_root=None, include_drafts=False):
        self.catalog = catalog or Catalog()
        self.directory = local_path(data_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.video_root = local_path(video_root) if video_root else None
        self.include_drafts = include_drafts
        self.lock = threading.RLock()
        self.sources, self.items, self.warnings = {}, {}, []
        for kind, directories in [('live', live_dirs), ('video', video_dirs)]:
            for supplied in directories:
                directory = local_path(supplied)
                if directory == self.directory:
                    raise ValueError('確認履歴のdata-dirは元ラベルのsession保存先と別にしてください')
                if not directory.is_dir():
                    raise ValueError(f"入力元がありません（作成していません）: {directory}")
                source_id = hashlib.sha256((kind + str(directory)).encode()).hexdigest()[:16]
                if source_id in self.sources:
                    continue
                store = LiveStore(directory, self.catalog) if kind == 'live' else MatchStore(directory, self.catalog)
                self.sources[source_id] = {'kind': kind, 'directory': directory, 'store': store}
        if not self.sources:
            raise ValueError('--live-dir または --video-dir を指定してください')
        self.refresh()

    def get_annotation(self, source, match_id):
        if source['kind'] == 'live':
            return source['store'].get(match_id)['annotation']
        return read_json(source['store'].path(match_id))

    def refresh(self):
        with self.lock:
            items, warnings = {}, []
            for source_id, source in self.sources.items():
                base = source['directory'] / 'matches'
                paths = base.glob('*/annotation.json') if source['kind'] == 'live' else base.glob('*.json')
                for path in sorted(paths):
                    try:
                        annotation = read_json(path)
                        if annotation.get('rejected') or (not self.include_drafts and not annotation.get('confirmed')):
                            continue
                        match_id = annotation['match_id']
                        if source['kind'] == 'live':
                            capture = source['store'].capture(match_id)
                            if not capture['frames'] or capture['status'] == 'collecting':
                                continue
                        if set(annotation['slots']) != set(SLOTS):
                            raise ValueError('8スロットがありません')
                        for key in SLOTS:
                            slot = annotation['slots'][key]
                            weapon = UNASSIGNED
                            if slot['status'] == 'labeled':
                                try:
                                    weapon = self.catalog.resolve_label(slot['weapon'])
                                except ValueError as exc:
                                    warnings.append(f'{path} / {key}: {exc}')
                            item_id = hashlib.sha256(f'{source_id}/{match_id}/{key}'.encode()).hexdigest()[:32]
                            review_path = self.directory / 'reviews' / f'{item_id}.json'
                            checked = False
                            if review_path.exists():
                                review = read_json(review_path)
                                checked = review.get('fingerprint') == signature(annotation, key) and review.get('checked') is True
                            items[item_id] = {'item_id': item_id, 'source_id': source_id, 'kind': source['kind'],
                                             'match_id': match_id, 'slot': key, 'weapon': weapon,
                                             'status': slot['status'], 'revision': annotation['revision'],
                                             'confirmed': annotation['confirmed'], 'checked': checked}
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        warnings.append(f'{path}: {exc}（一覧から除外）')
            self.items, self.warnings = items, warnings
            return items

    def state(self):
        self.refresh()
        groups = []
        for entry in [*self.catalog.to_list(), {'name': UNASSIGNED, 'display_name': '未指定 / Unknown / Occluded / Skip', 'category': 'その他'}]:
            rows = [r for r in self.items.values() if r['weapon'] == entry['name']]
            groups.append({**entry, 'slots': len(rows), 'matches': len({(r['source_id'], r['match_id']) for r in rows}),
                           'checked': sum(r['checked'] for r in rows), 'remaining': sum(not r['checked'] for r in rows),
                           'by_kind': {kind: {'slots': sum(r['kind']==kind for r in rows),
                                             'matches': len({(r['source_id'],r['match_id']) for r in rows if r['kind']==kind}),
                                             'remaining': sum(r['kind']==kind and not r['checked'] for r in rows)}
                                       for kind in ('live','video')}})
        resume_path = self.directory / 'resume.json'
        return {'weapons': groups, 'items': len(self.items), 'checked': sum(r['checked'] for r in self.items.values()),
                'sources': [{'kind': s['kind'], 'directory': str(s['directory'])} for s in self.sources.values()],
                'data_dir': str(self.directory), 'warnings': self.warnings,
                'resume': read_json(resume_path) if resume_path.exists() else {}}

    def group(self, weapon):
        self.refresh()
        return sorted([dict(r) for r in self.items.values() if r['weapon'] == weapon],
                      key=lambda r: (r['kind'], r['match_id'], r['slot']))

    def lookup(self, item_id):
        if item_id not in self.items:
            self.refresh()
        if item_id not in self.items:
            raise ValueError('対象がなくなりました。一覧を更新してください')
        with self.lock:
            item = self.items.get(item_id)
            if item is None:
                raise ValueError('対象がなくなりました。一覧を更新してください')
            source = self.sources[item['source_id']]
            annotation = self.get_annotation(source, item['match_id'])
            return item, source, annotation

    def frames(self, item, source, annotation):
        if source['kind'] == 'live':
            capture = source['store'].capture(item['match_id'])
            return [{k: row[k] for k in ('ordinal', 'timestamp', 'frame_index')} for row in capture['frames']]
        video = annotation.get('video_metadata') or source['store'].videos[annotation['video_id']]
        first = annotation['reference_timestamp']
        return [{'ordinal': i, 'timestamp': first + i, 'frame_index': round((first+i)*video['fps'])}
                for i in range(5) if first+i < min(annotation['end_timestamp'], video['duration'])]

    def item(self, item_id):
        item, source, a = self.lookup(item_id)
        slot = a['slots'][item['slot']]
        return {**item, 'revision': a['revision'], 'label': slot,
                'display_name': self.catalog.entries.get(slot.get('weapon'), {}).get('display_name', slot.get('weapon')),
                'reference_ordinal': a.get('reference_ordinal', 0), 'frames': self.frames(item, source, a),
                'source_directory': str(source['directory'])}

    @staticmethod
    @lru_cache(maxsize=8)
    def decode(path, mtime, size):
        return cv2_read(path)

    @staticmethod
    @lru_cache(maxsize=8)
    def decode_video(path, mtime, size, index, fps, frame_count):
        return frame_at({'path': path, 'fps': fps, 'frame_count': frame_count}, frame_index=index)[0]

    @staticmethod
    @lru_cache(maxsize=8)
    def verify_video(path, mtime, size, expected_id):
        if video_metadata(path)['video_id'] != expected_id:
            raise ValueError('元動画の識別子が一致しません。同名の別動画を表示・修正しません')

    def frame(self, item, source, annotation, ordinal):
        rows = self.frames(item, source, annotation)
        if type(ordinal) is not int or not 0 <= ordinal < len(rows):
            raise ValueError('frame番号が不正です')
        if source['kind'] == 'live':
            _, _, path = source['store'].frame(item['match_id'], ordinal)  # Includes SHA256 verification.
            stat = path.stat()
            return self.decode(str(path), stat.st_mtime_ns, stat.st_size)
        video = dict(annotation.get('video_metadata') or source['store'].videos[annotation['video_id']])
        path = local_path(video['path'])
        if not path.is_file() and self.video_root:
            # Explicit user override; no source metadata/path is rewritten.
            filename = str(video['path']).replace('\\', '/').rsplit('/', 1)[-1]
            path = self.video_root / filename
        if not path.is_file():
            raise ValueError(f'元動画が見つかりません: {video["path"]}。--video-root で動画フォルダを指定してください')
        stat = path.stat()
        self.verify_video(str(path), stat.st_mtime_ns, stat.st_size, annotation['video_id'])
        return self.decode_video(str(path), stat.st_mtime_ns, stat.st_size, rows[ordinal]['frame_index'], video['fps'], video['frame_count'])

    def image(self, item_id, ordinal=None, kind='crop'):
        item, source, a = self.lookup(item_id)
        ordinal = a.get('reference_ordinal', 0) if ordinal is None else ordinal
        frame = self.frame(item, source, a, ordinal)
        key = item['slot']
        crop, rect = calibrated_crop(frame, key[:-1], int(key[-1]), a.get('geometry'))
        if kind == 'crop':
            return image_bytes(crop)
        if kind not in {'hud', 'context'}:
            raise ValueError('画像種別が不正です')
        marked = frame.copy()
        x1, y1, x2, y2 = rect
        cv2.rectangle(marked, (x1,y1), (x2,y2), (0,220,255), 3)
        if kind == 'hud':
            marked = marked[:round(frame.shape[0]*.16)]
        else:
            marked = cv2.resize(marked, (960, round(960*marked.shape[0]/marked.shape[1])))
        return image_bytes(marked)

    def save_source(self, source, value):
        if source['kind'] == 'live':
            return source['store'].save(value)
        with VideoSaveLock(source['directory']):
            return source['store'].save(value)

    def check_revision(self, value, expected):
        if type(expected) is not int or expected != value['revision']:
            raise ValueError('別画面で更新されています。再読込してください（上書きしていません）')

    def mark(self, item_id, revision):
        with self.lock, FileLock(self.directory/'review.lock', '別の確認アプリが保存中です'):
            item, source, a = self.lookup(item_id)
            self.check_revision(a, revision)
            self.frame(item, source, a, a.get('reference_ordinal', 0))
            # Re-read after image I/O: don't mark a concurrently edited label.
            self.check_revision(self.get_annotation(source, item['match_id']), revision)
            atomic_json(self.directory/'reviews'/f'{item_id}.json', {'checked': True, 'fingerprint': signature(a,item['slot']), 'updated_at': now()})
            return {'ok': True, 'revision': revision}

    def correct(self, item_id, revision, weapon, status):
        if status not in STATUSES:
            raise ValueError('statusが不正です（保存していません）')
        # Refuse invalid hidden names too, just like the original application.
        resolved = self.catalog.resolve_label(weapon) if weapon is not None or status == 'labeled' else None
        with self.lock, FileLock(self.directory/'review.lock', '別の確認アプリが保存中です'):
            item, source, before = self.lookup(item_id)
            self.check_revision(before, revision)
            self.frame(item, source, before, before.get('reference_ordinal', 0))
            value = copy.deepcopy(before)
            slot = value['slots'][item['slot']]
            slot.update(weapon=resolved if status=='labeled' else None, status=status, reviewed=True)
            if status == 'labeled':
                slot['weapon_class'] = self.catalog.entries[resolved]['weapon_class']
            else:
                slot.pop('weapon_class', None)
            operation = uuid.uuid4().hex
            backup = {'operation': operation, 'source_id': item['source_id'], 'item_id': item_id,
                      'match_id': item['match_id'], 'slot': item['slot'], 'before': before,
                      'after_revision': revision+1, 'created_at': now(), 'status': 'prepared'}
            path = self.directory/'changes'/f'{operation}.json'
            atomic_json(path, backup)  # A durable full-match backup BEFORE touching the source.
            saved = self.save_source(source, value)
            backup.update(status='applied', after_revision=saved['revision'])
            atomic_json(path, backup)
            atomic_json(self.directory/'reviews'/f'{item_id}.json', {'checked': True, 'fingerprint': signature(saved,item['slot']), 'updated_at': now()})
            self.refresh()
            return {'ok': True, 'operation': operation, 'revision': saved['revision'], 'weapon': saved['slots'][item['slot']]['weapon']}

    def undo(self, operation):
        if not isinstance(operation, str) or len(operation)!=32 or any(c not in '0123456789abcdef' for c in operation):
            raise ValueError('操作IDが不正です')
        with self.lock, FileLock(self.directory/'review.lock', '別の確認アプリが保存中です'):
            path = self.directory/'changes'/f'{operation}.json'
            change = read_json(path)
            if change['status'] != 'applied':
                raise ValueError('この修正は取消済みまたは未完了です')
            source = self.sources[change['source_id']]
            current = self.get_annotation(source, change['match_id'])
            self.check_revision(current, change['after_revision'])
            value = copy.deepcopy(change['before'])
            value['revision'] = current['revision']
            saved = self.save_source(source, value)
            atomic_json(self.directory/'reviews'/f'{change["item_id"]}.json', {'checked': False, 'updated_at': now()})
            change.update(status='undone', undo_revision=saved['revision'], updated_at=now())
            atomic_json(path, change)
            self.refresh()
            return {'ok': True, 'revision': saved['revision']}

    def resume(self, weapon, item_id=None):
        if weapon != UNASSIGNED:
            self.catalog.resolve_label(weapon)
        atomic_json(self.directory/'resume.json', {'weapon': weapon, 'item_id': item_id})
