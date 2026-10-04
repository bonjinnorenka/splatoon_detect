"""Local browser app for weapon-first review; never opens a camera."""
from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

from live_weapon_collect.app import LiveHandler
from weapon_label_review.locking import VideoSaveBusy
from weapon_label_review.store import ReviewStore
from weapon_lamp_detect.match_data import Catalog

HERE = Path(__file__).resolve().parent


class ReviewHandler(LiveHandler):
    def send(self, value, status=200, content_type='application/json; charset=utf-8'):
        try:
            return super().send(value, status, content_type)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers cancel obsolete image requests on navigation. Do not send
            # another error response to the same dead connection. POST commits
            # have already finished before send() and must not be rolled back.
            self.close_connection = True

    def do_GET(self):
        try:
            self.local_request()
            url = urlsplit(self.path)
            q = {k:v[0] for k,v in parse_qs(url.query).items()}
            store = self.server.store
            if url.path == '/':
                return self.send((HERE/'ui.html').read_bytes(), content_type='text/html; charset=utf-8')
            if url.path == '/api/state':
                return self.send(store.state())
            if url.path == '/api/items':
                return self.send(store.group(q['weapon']))
            if url.path == '/api/item':
                return self.send(store.item(q['item_id']))
            if url.path == '/api/image':
                return self.send(store.image(q['item_id'], int(q['ordinal']) if 'ordinal' in q else None, q.get('kind','crop')),
                                 content_type='image/jpeg')
            return self.send({'error':'not found'},404)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.send({'error':str(exc)},400)

    def do_POST(self):
        try:
            self.local_request()
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                raise ValueError('別サイトからの保存を拒否しました')
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                raise ValueError('origin mismatch')
            if not self.headers.get('Content-Type','').startswith('application/json'):
                raise ValueError('application/json required')
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 100000:
                raise ValueError('invalid body length')
            value = json.loads(self.rfile.read(length))
            store = self.server.store
            if self.path == '/api/check':
                return self.send(store.mark(value['item_id'],value['revision']))
            if self.path == '/api/correct':
                return self.send(store.correct(value['item_id'],value['revision'],value.get('weapon'),value['status']))
            if self.path == '/api/undo':
                return self.send(store.undo(value['operation']))
            if self.path == '/api/resume':
                store.resume(value['weapon'],value.get('item_id'))
                return self.send({'ok':True})
            return self.send({'error':'not found'},404)
        except VideoSaveBusy as exc:
            self.send({'error':str(exc)},409)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.send({'error':str(exc)},400)


def main():
    parser = argparse.ArgumentParser(description='武器ごとにライブ・動画の人力ラベルを確認・修正する別アプリ')
    parser.add_argument('--live-dir', action='append', type=Path, default=[], help='ライブsession保存先。複数指定可')
    parser.add_argument('--video-dir', action='append', type=Path, default=[], help='既存動画ラベラーのsession保存先。複数指定可')
    parser.add_argument('--video-root', type=Path, help='元動画の保存フォルダ（WindowsでLinuxの動画pathを補う場合）')
    default_data = Path(os.environ['LOCALAPPDATA'])/'splatoon-weapon-review' if os.name=='nt' and os.environ.get('LOCALAPPDATA') else HERE/'data'
    parser.add_argument('--data-dir', type=Path, default=default_data, help='確認履歴・修正前バックアップ・resumeの保存先。元画像は複製しない')
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--include-drafts', action='store_true', help='確定済みだけでなく下書きも表示（確定フラグは自動変更しない）')
    parser.add_argument('--port', type=int, default=8783)
    a = parser.parse_args()
    if not a.live_dir and not a.video_dir:
        parser.error('--live-dir または --video-dir が必要です')
    cv2.setNumThreads(1)
    store = ReviewStore(a.data_dir,a.live_dir,a.video_dir,Catalog(catalog_path=a.catalog),a.video_root,a.include_drafts)
    server = ThreadingHTTPServer(('127.0.0.1',a.port),ReviewHandler)
    server.store = store
    print(f'http://127.0.0.1:{a.port}/  確認履歴={store.directory}',flush=True)
    print('カメラを開きません。正しい確認は元ラベルを変更せず、修正保存だけ元GTへ反映します。',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
