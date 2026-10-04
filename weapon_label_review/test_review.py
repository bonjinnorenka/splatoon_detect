"""Use temporary data only: user labels, recordings and live images stay untouched."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from live_weapon_collect.test_live import fixture
from weapon_label_review.app import ReviewHandler
from weapon_label_review.locking import VideoSaveLock
from weapon_label_review.store import ReviewStore, UNASSIGNED
from weapon_lamp_detect.match_data import Catalog, MatchStore, atomic_json, blank_match, read_json, video_metadata


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='weapon-review-日本語 空白-')
        self.root = Path(self.tmp.name)
        self.catalog = Catalog()
        self.live, self.writer = fixture(self.root/'live', self.catalog)
        a = self.live.get(self.writer.record['match_id'])['annotation']
        a.update(confirmed=True, opening_reviewed=True)
        for slot in a['slots'].values():
            slot.update(weapon='ボールドマーカー', status='labeled', reviewed=True)
        self.live.save(a)
        self.video = MatchStore(self.root/'video',self.catalog)
        movie = self.root/'test.avi'
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'MJPG'), 2, (320,180))
        if not writer.isOpened():
            self.skipTest('MJPG writer unavailable')
        for i in range(12):
            frame = np.full((180,320,3),128,np.uint8)
            cv2.rectangle(frame,(80,8),(135,35),(20+i,10,220),-1)
            writer.write(frame)
        writer.release()
        self.movie = movie
        metadata = video_metadata(movie)
        self.video.add_videos([movie])
        a = blank_match(metadata,0,metadata['duration'],hud_offset=0)
        a.update(confirmed=True,boundaries_confirmed=True)
        for slot in a['slots'].values():
            slot.update(weapon='ボールドマーカー',status='labeled',reviewed=True)
        self.video_match = self.video.save(a)
        self.review = ReviewStore(self.root/'review',[self.live.directory],[self.video.directory],self.catalog)
        self.live_item = next(i for i in self.review.group('Sploosh-o-matic') if i['kind']=='live')
        self.video_item = next(i for i in self.review.group('Sploosh-o-matic') if i['kind']=='video')

    def tearDown(self):
        self.tmp.cleanup()

    def live_annotation(self):
        return self.live.get(self.live_item['match_id'])['annotation']

    def test_index_counts_match_slots_not_frames(self):
        state = self.review.state()
        group = next(w for w in state['weapons'] if w['name']=='Sploosh-o-matic')
        self.assertEqual((group['matches'],group['slots'],group['remaining']),(2,16,16))
        self.assertEqual(len(self.review.item(self.live_item['item_id'])['frames']),5)
        self.assertEqual(len(self.review.item(self.video_item['item_id'])['frames']),5)

    def test_mark_does_not_write_original_gt_or_capture(self):
        paths=[self.writer.path/'annotation.json',self.writer.path/'capture.json',self.writer.path/'frames/00.png']
        before=[p.read_bytes() for p in paths]
        self.review.mark(self.live_item['item_id'],self.live_item['revision'])
        self.assertEqual(before,[p.read_bytes() for p in paths])
        self.assertEqual(self.review.state()['checked'],1)
        resumed=ReviewStore(self.root/'review',[self.live.directory],[self.video.directory],self.catalog)
        self.assertEqual(resumed.state()['checked'],1)

    def test_live_correction_only_one_slot_keeps_confirmed_and_images(self):
        original=self.live_annotation()
        capture=(self.writer.path/'capture.json').read_bytes()
        result=self.review.correct(self.live_item['item_id'],original['revision'],'シャープマーカー','labeled')
        after=self.live_annotation()
        key=self.live_item['slot']
        self.assertEqual(after['slots'][key]['weapon'],'Splash-o-matic')
        self.assertTrue(after['confirmed'])
        for k in after['slots']:
            if k!=key:self.assertEqual(after['slots'][k],original['slots'][k])
        self.assertEqual(after['geometry'],original['geometry'])
        self.assertEqual((self.writer.path/'capture.json').read_bytes(),capture)
        backup=read_json(self.root/'review/changes'/f'{result["operation"]}.json')
        self.assertEqual(backup['before'],original)
        self.assertTrue((self.writer.path/'history'/f'{original["revision"]:06d}.json').exists())
        self.assertEqual(len(self.review.group('Splash-o-matic')),1)

    def test_video_correction_and_undo_reuse_original_history(self):
        item=self.video_item
        original=read_json(self.video.path(item['match_id']))
        result=self.review.correct(item['item_id'],item['revision'],'シャープマーカー','labeled')
        after=read_json(self.video.path(item['match_id']))
        self.assertEqual(after['slots'][item['slot']]['weapon'],'Splash-o-matic')
        self.assertTrue(after['confirmed'])
        self.review.undo(result['operation'])
        undone=read_json(self.video.path(item['match_id']))
        self.assertEqual(undone['slots'],original['slots'])
        self.assertEqual(undone['revision'],original['revision']+2)
        self.assertTrue((self.video.directory/'history'/item['match_id']/f'{original["revision"]:06d}.json').exists())

    def test_unknown_and_skip_can_be_restored_to_weapon(self):
        item=self.live_item
        result=self.review.correct(item['item_id'],item['revision'],None,'skip')
        self.assertEqual(len(self.review.group(UNASSIGNED)),1)
        self.review.correct(item['item_id'],result['revision'],'ボールドマーカー','labeled')
        self.assertFalse(self.review.group(UNASSIGNED))

    def test_invalid_names_and_status_do_not_save_or_create_backup(self):
        original=(self.writer.path/'annotation.json').read_bytes()
        for weapon,status in [('シューター','labeled'),('fake','unknown'),('', 'labeled'),(None,'wrong')]:
            with self.subTest(weapon=weapon,status=status), self.assertRaises(ValueError):
                self.review.correct(self.live_item['item_id'],self.live_item['revision'],weapon,status)
        self.assertEqual((self.writer.path/'annotation.json').read_bytes(),original)
        self.assertFalse((self.root/'review/changes').exists())

    def test_stale_revision_refused_and_external_slot_edit_invalidates_check(self):
        item=self.live_item
        self.review.mark(item['item_id'],item['revision'])
        a=self.live_annotation();a['slots'][item['slot']]['weapon']='シャープマーカー'
        self.live.save(a)
        self.assertEqual(self.review.state()['checked'],0)
        for action in (lambda:self.review.mark(item['item_id'],item['revision']),
                       lambda:self.review.correct(item['item_id'],item['revision'],'ボールドマーカー','labeled')):
            with self.assertRaisesRegex(ValueError,'別画面'):
                action()

    def test_unrelated_slot_edit_does_not_reset_checked_fingerprint(self):
        item=self.live_item
        self.review.mark(item['item_id'],item['revision'])
        a=self.live_annotation()
        other=next(k for k in a['slots'] if k!=item['slot'])
        a['slots'][other]['weapon']='シャープマーカー';self.live.save(a)
        self.assertEqual(self.review.state()['checked'],1)

    def test_undo_refuses_overwriting_later_edits(self):
        item=self.live_item
        result=self.review.correct(item['item_id'],item['revision'],'シャープマーカー','labeled')
        a=self.live_annotation();a['notes']='another edit';self.live.save(a)
        before=(self.writer.path/'annotation.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'別画面'):
            self.review.undo(result['operation'])
        self.assertEqual(before,(self.writer.path/'annotation.json').read_bytes())

    def test_live_undo_once_preserves_older_labels(self):
        item=self.live_item; original=self.live_annotation()
        result=self.review.correct(item['item_id'],item['revision'],'シャープマーカー','labeled')
        self.review.undo(result['operation'])
        self.assertEqual(self.live_annotation()['slots'],original['slots'])
        with self.assertRaisesRegex(ValueError,'取消済み'):
            self.review.undo(result['operation'])

    def test_broken_image_blocks_check_and_correction(self):
        item=self.live_item;before=(self.writer.path/'annotation.json').read_bytes()
        (self.writer.path/'frames/00.png').write_bytes(b'broken')
        for action in (lambda:self.review.mark(item['item_id'],item['revision']),
                       lambda:self.review.correct(item['item_id'],item['revision'],'シャープマーカー','labeled')):
            with self.assertRaisesRegex(ValueError,'変更'):
                action()
        self.assertEqual(before,(self.writer.path/'annotation.json').read_bytes())

    def test_image_endpoints_live_and_video(self):
        for item in (self.live_item,self.video_item):
            for kind in ('crop','hud','context'):
                data=self.review.image(item['item_id'],0,kind)
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
        with self.assertRaises(ValueError):self.review.image('../escape')
        with self.assertRaises(ValueError):self.review.image(self.live_item['item_id'],20)

    def test_drafts_and_rejected_excluded_by_default(self):
        a=self.live_annotation();a['confirmed']=False;self.live.save(a)
        self.assertEqual(self.review.state()['items'],8)
        drafts=ReviewStore(self.root/'draft-review',[self.live.directory],[],self.catalog,include_drafts=True)
        self.assertEqual(drafts.state()['items'],8)
        a=self.live_annotation();a['rejected']=True;self.live.save(a)
        self.assertEqual(drafts.state()['items'],0)

    def test_http_validation_and_image_without_socket(self):
        server=SimpleNamespace(store=self.review)
        def request(path,value=None,origin=None):
            h=ReviewHandler.__new__(ReviewHandler);h.server=server;h.path=path
            h.request_version='HTTP/1.1';h.command='POST' if value is not None else 'GET';h.requestline=h.command+' '+path
            raw=json.dumps(value).encode() if value is not None else b''
            h.headers={'Host':'127.0.0.1:8783','Content-Type':'application/json','Content-Length':str(len(raw))}
            if origin:h.headers['Origin']=origin
            h.rfile=io.BytesIO(raw);h.wfile=io.BytesIO()
            h.do_POST() if value is not None else h.do_GET()
            headers,body=h.wfile.getvalue().split(b'\r\n\r\n',1)
            return int(headers.split()[1]),body
        item=self.live_item
        status,body=request('/api/state');self.assertEqual(status,200)
        status,body=request('/api/image?item_id='+item['item_id']);self.assertEqual(status,200)
        payload={'item_id':item['item_id'],'revision':item['revision'],'weapon':'シューター','status':'labeled'}
        status,body=request('/api/correct',payload);self.assertEqual(status,400)
        self.assertIn('無効な武器名',json.loads(body)['error'])
        self.assertEqual(request('/api/correct',payload,'http://example.com')[0],400)

    def handler(self, path, value=None, output=None):
        h = ReviewHandler.__new__(ReviewHandler)
        h.server = SimpleNamespace(store=self.review)
        h.path = path
        h.request_version = 'HTTP/1.1'
        h.command = 'POST' if value is not None else 'GET'
        h.requestline = h.command+' '+path
        raw = json.dumps(value).encode() if value is not None else b''
        h.headers = {'Host':'127.0.0.1:8783', 'Content-Type':'application/json', 'Content-Length':str(len(raw))}
        h.rfile = io.BytesIO(raw)
        h.wfile = output if output is not None else io.BytesIO()
        h.send_response = Mock(wraps=h.send_response)
        return h

    def test_video_busy_returns_conflict_without_touching_original_gt(self):
        item = self.video_item
        path = self.video.path(item['match_id'])
        before = path.read_bytes()
        with VideoSaveLock(self.video.directory) as owner:
            marker = owner.owner_path.read_bytes()
            h = self.handler('/api/correct', {'item_id':item['item_id'], 'revision':item['revision'],
                                             'weapon':'Splash-o-matic', 'status':'labeled'})
            h.do_POST()
            h.send_response.assert_called_once_with(409)
            body = h.wfile.getvalue().split(b'\r\n\r\n',1)[1]
            self.assertIn('annotation.review.lock', json.loads(body)['error'])
            self.assertEqual(owner.owner_path.read_bytes(), marker)
        self.assertEqual(path.read_bytes(), before)
        backups = list((self.review.directory/'changes').glob('*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(read_json(backups[0])['status'], 'prepared')
        # The same revision is safe to retry after the competing writer stops.
        result = self.review.correct(item['item_id'], item['revision'], 'Splash-o-matic', 'labeled')
        self.assertEqual(result['revision'], item['revision']+1)
        self.assertFalse((self.video.directory/'annotation.review.lock').exists())

    def test_video_save_avoids_source_byte_lock_but_keeps_review_mutex(self):
        from live_weapon_collect.io_utils import FileLock
        def native_only(path, message):
            if Path(path).parent == self.video.directory:
                raise OSError('Windows UNC byte-range locking unavailable')
            return FileLock(path, message)
        item = self.video_item
        with patch('weapon_label_review.store.FileLock', side_effect=native_only):
            result = self.review.correct(item['item_id'], item['revision'], 'Splash-o-matic', 'labeled')
            self.review.undo(result['operation'])
        self.assertEqual(read_json(self.video.path(item['match_id']))['slots'][item['slot']]['weapon'], 'Sploosh-o-matic')

    def test_cancelled_image_does_not_attempt_second_response(self):
        class Disconnected:
            def __init__(self, error, on_write):
                self.calls = 0
                self.error, self.on_write = error, on_write
            def write(self, data):
                self.calls += 1
                if self.calls == self.on_write:
                    raise self.error
        for error in (BrokenPipeError('closed'), ConnectionResetError('reset'), ConnectionAbortedError(10053, 'aborted')):
            for on_write in (1, 2):  # Failure in HTTP headers or in JPEG body.
                with self.subTest(error=type(error).__name__, on_write=on_write):
                    output = Disconnected(error, on_write)
                    h = self.handler('/api/image?item_id='+self.live_item['item_id'], output=output)
                    h.do_GET()
                    self.assertTrue(h.close_connection)
                    self.assertEqual(output.calls, on_write)
                    h.send_response.assert_called_once_with(200)

    def test_post_disconnection_does_not_undo_completed_save(self):
        output = SimpleNamespace(write=Mock(side_effect=ConnectionAbortedError(10053, 'aborted')))
        item = self.video_item
        h = self.handler('/api/correct', {'item_id':item['item_id'], 'revision':item['revision'],
                                         'weapon':'Splash-o-matic', 'status':'labeled'}, output)
        h.do_POST()
        self.assertTrue(h.close_connection)
        h.send_response.assert_called_once_with(200)
        output.write.assert_called_once()
        saved = read_json(self.video.path(item['match_id']))
        self.assertEqual(saved['revision'], item['revision']+1)
        self.assertEqual(saved['slots'][item['slot']]['weapon'], 'Splash-o-matic')
        backups = list((self.review.directory/'changes').glob('*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(read_json(backups[0])['status'], 'applied')
        self.assertTrue((self.video.directory/'history'/item['match_id']/f'{item["revision"]:06d}.json').exists())

    def test_bad_json_visible_warning_and_missing_source_not_created(self):
        (self.writer.path/'annotation.json').write_bytes(b'bad json')
        self.assertTrue(self.review.state()['warnings'])
        missing=self.root/'nonexistent source'
        with self.assertRaisesRegex(ValueError,'入力元がありません'):
            ReviewStore(self.root/'other review',[missing],[],self.catalog)
        self.assertFalse(missing.exists())

    def test_video_root_override_validates_video_identity(self):
        a=read_json(self.video.path(self.video_item['match_id']))
        a['video_metadata']['path']='/missing Linux path/test.avi'
        atomic_json(self.video.path(a['match_id']),a)
        self.review.video_root=self.root
        self.assertTrue(self.review.image(self.video_item['item_id'],0))
        other=self.root/'wrong folder';other.mkdir()
        writer=cv2.VideoWriter(str(other/'test.avi'),cv2.VideoWriter_fourcc(*'MJPG'),2,(320,180))
        for _ in range(12):writer.write(np.full((180,320,3),40,np.uint8))
        writer.release()
        self.review.video_root=other
        with self.assertRaisesRegex(ValueError,'識別子が一致しません'):
            self.review.image(self.video_item['item_id'],0)

    def test_review_directory_cannot_overwrite_source_resume(self):
        before=(self.live.directory/'resume.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'元ラベルのsession保存先と別'):
            ReviewStore(self.live.directory,[self.live.directory],[],self.catalog)
        self.assertEqual(before,(self.live.directory/'resume.json').read_bytes())


if __name__=='__main__':unittest.main()
