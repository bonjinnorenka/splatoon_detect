"""CPU-only tests; no camera, new libraries, or user recordings required."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from live_weapon_collect.app import LiveHandler, LiveServer
from live_weapon_collect.capture import (
    CaptureConfig, CaptureWriter, CollectorLock, LatestCapture, OpeningGate, capture_worker, recover_interrupted,
)
from live_weapon_collect.export_dataset import export_dataset
from live_weapon_collect.store import LiveStore
from squid_lamp_detect.squid_lamp import SlotReading
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop, verify_sources
from weapon_lamp_detect.match_data import Catalog, SLOTS, read_json


def fixture(directory, catalog, frames=5):
    cfg = CaptureConfig()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    reading = SimpleNamespace(hud_state="non_match", sides={
        side: SimpleNamespace(slots=[SlotReading(i, "unknown", 0., 0., 0.) for i in range(4)])
        for side in ("left", "right")})
    writer = CaptureWriter(directory, "testsession", "0",
                           {"timing": "monotonic elapsed", "frame_index_definition": "successful read counter"},
                           cfg, (frame, 1200, 20.), "intro_hud_timer", 0.)
    for i in range(frames):
        writer.append((frame, 1200 + 60*i, 20.+i), reading)
    writer.finish()
    return LiveStore(directory, catalog), writer


class GateTests(unittest.TestCase):
    def test_intro_waits_for_hud_and_settle(self):
        g = OpeningGate(CaptureConfig())
        self.assertIsNone(g.observe(0., True, False, "non_match"))
        self.assertEqual(g.phase, "waiting_hud")
        self.assertIsNone(g.observe(7., False, True, "match"))
        self.assertIsNone(g.observe(15., False, True, "match"))
        self.assertIsNone(g.observe(15.5, False, True, "match"))
        self.assertEqual(g.observe(16., False, True, "match"), "capture")

    def test_nonopening_hud_keeps_provisional_opening_not_late_frames(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        for t in (10., 15., 19.5):
            self.assertIsNone(g.observe(t, False, False, "match"))
        self.assertEqual(g.observe(20., False, False, "match"), "capture_unverified")
        self.assertFalse(g.verified)
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        self.assertEqual(g.observe(36., False, False, "match"), "timeout")
        self.assertEqual(g.blocked_until, 38.)

    def test_dedup_requires_cooldown_and_nonmatch(self):
        g = OpeningGate(CaptureConfig())
        g.manual(0.)
        g.verified = True
        g.finish(4.)
        for t in (70., 71., 72., 73., 80.):
            self.assertIsNone(g.observe(t, True, True, "match"))
        self.assertEqual(g.phase, "cooldown")
        for t in (81., 81.5, 82., 82.5):
            g.observe(t, False, False, "non_match")
        g.observe(83., True, False, "non_match")
        self.assertEqual(g.phase, "waiting_hud")

    def test_unverified_capture_does_not_block_fresh_intro(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        for t in (1., 2., 3., 4.):
            g.observe(t, False, False, "non_match")
        self.assertEqual(g.observe(20., False, False, "non_match"), "capture_unverified")
        g.finish(24.)
        g.observe(27., True, False, "match")  # Even a wrong HUD-state cannot block a new intro.
        self.assertEqual(g.intro, 27.)
        self.assertEqual(g.phase, "waiting_hud")

    def test_continuous_intro_is_not_collected_repeatedly(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        self.assertEqual(g.observe(20., True, False, "non_match"), "capture_unverified")
        g.finish(24.)
        for t in range(26, 100):
            self.assertIsNone(g.observe(t, True, False, "non_match"))

    def test_new_intro_replaces_false_pending_candidate(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        for t in range(1, 6):
            g.observe(t, False, False, "non_match")
        g.observe(10., True, False, "non_match")
        self.assertEqual(g.intro, 10.)
        self.assertIsNone(g.observe(20., False, False, "non_match"))
        self.assertEqual(g.observe(30., False, False, "non_match"), "capture_unverified")

    def test_verified_read_at_fallback_deadline_takes_priority(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0., True, False, "non_match")
        g.observe(19., False, True, "match")
        g.observe(19.5, False, True, "match")
        self.assertEqual(g.observe(20., False, True, "match"), "capture")
        self.assertTrue(g.verified)

    def test_invalid_hud_resets_consecutive_reads(self):
        g = OpeningGate(CaptureConfig())
        g.observe(0, True, False, "non_match")
        g.observe(15, False, True, "match")
        g.observe(15.5, False, False, "match")
        self.assertIsNone(g.observe(16, False, True, "match"))
        self.assertIsNone(g.observe(16.5, False, True, "match"))
        self.assertEqual(g.observe(17, False, True, "match"), "capture")

    def test_manual_does_not_reuse_old_intro(self):
        g = OpeningGate(CaptureConfig())
        g.intro = 0
        g.phase = "cooldown"
        g.manual(300)
        self.assertIsNone(g.last_intro)

    def test_config_limits(self):
        for kwargs in ({"interval": float("nan")}, {"frames": 100}, {"hud_delay_min": 40}, {"fourcc": "abc"},
                       {"fallback_offset": float("nan")}, {"fallback_offset": 36}, {"fallback_offset": 5}):
            with self.assertRaises(ValueError):
                CaptureConfig(**kwargs).validate()


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="live-日本語 空白-")
        self.root = Path(self.tmp.name)
        self.store, self.writer = fixture(self.root, self.catalog)
        self.match_id = self.writer.record["match_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def annotation(self):
        a = self.store.get(self.match_id)["annotation"]
        for s in a["slots"].values():
            s.update(weapon="ボールドマーカー", status="labeled", reviewed=True)
        a.update(opening_reviewed=True, confirmed=True)
        return a

    def test_labels_resume_and_capture_untouched(self):
        capture_path = self.writer.path / "capture.json"
        before = capture_path.read_bytes()
        saved = self.store.save(self.annotation())
        reloaded = LiveStore(self.root, self.catalog)
        self.assertEqual(reloaded.get(self.match_id)["annotation"], saved)
        self.assertEqual(reloaded.resume()["match_id"], self.match_id)
        self.assertEqual(capture_path.read_bytes(), before)
        self.assertEqual(saved["slots"]["left0"]["weapon"], "Sploosh-o-matic")

    def test_invalid_tag_and_nonnull_invalid_unknown_refuse_all_writes(self):
        a = self.annotation()
        saved = self.store.save(a)
        path = self.writer.path / "annotation.json"
        before = path.read_bytes()
        for status, name in (("labeled", "シューター"), ("labeled", "架空の武器"), ("unknown", "wrong")):
            bad = copy.deepcopy(saved)
            bad["slots"]["left0"].update(status=status, weapon=name)
            with self.assertRaisesRegex(ValueError, "保存していません"):
                self.store.save(bad)
            self.assertEqual(before, path.read_bytes())

    def test_unknown_requires_human_review_for_confirmation(self):
        a = self.annotation()
        a["slots"]["left0"].update(weapon=None, status="unknown", reviewed=False)
        with self.assertRaises(ValueError):
            self.store.save(a)
        a["slots"]["left0"]["reviewed"] = True
        self.assertTrue(self.store.save(a)["confirmed"])

    def test_draft_can_be_saved_while_collecting(self):
        self.writer.record["status"] = "collecting"
        self.writer.finish("collecting")
        a = self.annotation()
        with self.assertRaisesRegex(ValueError, "収集完了"):
            self.store.save(a)
        a["confirmed"] = False
        self.assertFalse(self.store.save(a)["confirmed"])

    def test_stale_revision_and_undo(self):
        old = self.annotation()
        saved = self.store.save(old)
        with self.assertRaisesRegex(ValueError, "別画面"):
            self.store.save(old)
        undone = self.store.undo(self.match_id, saved["revision"])
        self.assertFalse(undone["confirmed"])
        self.assertEqual(undone["revision"], 2)

    def test_new_match_does_not_overwrite_old_labels(self):
        saved = self.store.save(self.annotation())
        _, second = fixture(self.root, self.catalog)
        self.assertNotEqual(self.match_id, second.record["match_id"])
        self.assertEqual(saved, self.store.get(self.match_id)["annotation"])
        self.assertEqual(len(self.store.list()), 2)

    def test_reference_geometry_and_path_validation(self):
        a = self.annotation()
        a["reference_ordinal"] = 20
        with self.assertRaises(ValueError):
            self.store.save(a)
        a["reference_ordinal"] = 0
        a["geometry"]["left0"] = [float("nan"), .06, .05, .09]
        with self.assertRaises(ValueError):
            self.store.save(a)
        with self.assertRaises(ValueError):
            self.store.get("../escape")

    def test_recover_partial_preserves_existing_labels_and_frames(self):
        saved = self.store.save(self.annotation())
        self.writer.finish("collecting")
        recover_interrupted(self.root)
        self.assertEqual(self.store.capture(self.match_id)["status"], "interrupted")
        self.assertEqual(len(self.store.capture(self.match_id)["frames"]), 5)
        self.assertEqual(self.store.get(self.match_id)["annotation"], saved)

    def test_source_hash_tampering_refused(self):
        path = self.writer.path / "frames" / "00.png"
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "変更"):
            self.store.frame(self.match_id, 0)

    def test_export_existing_crop_and_template_pipeline(self):
        self.store.save(self.annotation())
        output = self.root / "export"
        self.assertEqual(export_dataset(self.root, output, self.catalog), 40)
        metadata, rows = load_dataset(output)
        verify_sources(metadata)
        self.assertEqual(sample_crop(output, rows[0], metadata).shape, (108, 134, 3))
        self.assertEqual({r["match_id"] for r in rows}, {self.match_id})
        self.assertEqual({r["opening_frame_ordinal"] for r in rows}, set(range(5)))
        self.assertEqual(rows[0]["weapon_label"], "Sploosh-o-matic")
        self.assertFalse(metadata["videos"])
        (output / rows[0]["context"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "異なります"):
            verify_sources(metadata)

    def test_export_never_uses_unconfirmed_labels(self):
        with self.assertRaisesRegex(ValueError, "人力確定"):
            export_dataset(self.root, self.root / "export", self.catalog)

    def test_exclusive_capture_lock_released(self):
        lock = CollectorLock(self.root)
        try:
            with self.assertRaises(ValueError):
                CollectorLock(self.root)
        finally:
            lock.close()
        again = CollectorLock(self.root)
        again.close()

    def test_http_invalid_names_warn_and_frame_endpoint(self):
        # Exercise the real handler without requiring sockets (sandbox restriction).
        server = LiveServer.__new__(LiveServer)
        server.store = self.store
        def request(path, value):
            handler = LiveHandler.__new__(LiveHandler)
            data = json.dumps(value).encode()
            handler.server, handler.path = server, path
            handler.request_version, handler.command, handler.requestline = "HTTP/1.1", "POST", "POST " + path
            handler.headers = {"Host": "127.0.0.1:8780", "Content-Type": "application/json", "Content-Length": str(len(data))}
            handler.rfile, handler.wfile = io.BytesIO(data), io.BytesIO()
            handler.do_POST()
            headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
            return int(headers.split()[1]), json.loads(body)
        a = self.annotation()
        a["slots"]["right3"]["weapon"] = "シューター"
        status, body = request("/api/save", a)
        self.assertEqual(status, 400)
        self.assertIn("保存していません", body["error"])
        self.assertFalse((self.writer.path / "annotation.json").exists())
        status, body = request("/api/frame", {"match_id": self.match_id})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["slots"]), 8)
        self.assertIn("hud", body)
        server.matcher_lock = threading.Lock()
        server.candidate_matcher = "hybrid"
        server.candidate_info = {"engine": "hybrid", "configuration": "blend_soft", "weapon_count": 1}
        candidate = SimpleNamespace(weapon="Sploosh-o-matic", to_dict=lambda: {"weapon": "Sploosh-o-matic", "score": .8})
        sides = []
        def infer(crop, **kwargs):
            sides.append(kwargs['side'])
            return [candidate]
        server.matcher = SimpleNamespace(predict_crop=infer)
        status, body = request("/api/candidates", {"match_id": self.match_id})
        self.assertEqual(status, 200)
        self.assertEqual(body["slots"][0]["candidates"][0]["display_name"], "ボールドマーカー")
        self.assertEqual(body["candidate_model"]["engine"], "hybrid")
        self.assertEqual(sides, ["left"] * 4 + ["right"] * 4)
        self.assertFalse((self.writer.path / "annotation.json").exists())


class ReaderTests(unittest.TestCase):
    def test_ocr_failure_or_no_alive_slots_does_not_lose_opening(self):
        for mode in ("absent", "unreadable", "error", "valid_but_zero_alive"):
            with self.subTest(mode=mode):
                frame = np.zeros((180, 320, 3), np.uint8)
                reading = SimpleNamespace(hud_state="match" if mode == "valid_but_zero_alive" else "non_match", sides={})
                def timer_read(*args, **kwargs):
                    if mode == "error":
                        raise ValueError("OCR test failure")
                    return SimpleNamespace(kind="time" if mode == "valid_but_zero_alive" else "unknown",
                                           seconds=300 if mode == "valid_but_zero_alive" else None)
                detector = SimpleNamespace(read_frame=lambda *args: reading,
                                           timer_ocr=None if mode == "absent" else SimpleNamespace(read_frame=timer_read))
                if mode == "error":
                    def hud_read(*args):
                        if detector.timer_ocr is not None:
                            raise ValueError("internal OCR test failure")
                        return reading
                    detector.read_frame = hud_read
                class Reader:
                    error = None
                    ended = False
                    metadata = {"timing": "simulated"}
                    thread = SimpleNamespace(start=lambda: None)
                    index = -1
                    def get(self):
                        self.index += 1
                        self.ended = self.index >= 110  # A continuous intro must not create another capture.
                        return frame, self.index*30, self.index*.5
                with tempfile.TemporaryDirectory() as directory, \
                     patch("live_weapon_collect.capture.LatestCapture", return_value=Reader()), \
                     patch("live_weapon_collect.capture.StartPredictor", return_value=SimpleNamespace(predict=lambda frame: {"is_start": True})), \
                     patch("live_weapon_collect.capture.squid_detector", return_value=detector), \
                     patch("live_weapon_collect.capture.calibrated_slot", side_effect=lambda frame, reading, side, i: SlotReading(i,"unknown",0.,0.,0.)):
                    capture_worker(directory, CaptureConfig(), queue.Queue(), threading.Event())
                    store = LiveStore(directory, Catalog())
                    matches = store.list()
                    self.assertEqual(len(matches), 1)
                    value = store.get(matches[0]["match_id"])
                    capture = value["capture"]
                    self.assertEqual(capture["status"], "complete")
                    self.assertEqual(len(capture["frames"]), 5)
                    self.assertFalse(value["annotation"]["confirmed"])
                    if mode == "valid_but_zero_alive":
                        self.assertEqual(capture["trigger"], "intro_hud_timer")
                        self.assertFalse(capture["needs_review"])
                    else:
                        self.assertEqual(capture["trigger"], "intro_offset_unverified")
                        self.assertTrue(matches[0]["needs_review"])
                        self.assertEqual([r["timestamp"] for r in capture["frames"]], [20.,21.,22.,23.,24.])
                        self.assertTrue(capture["warnings"])
                        self.assertTrue(capture["detection_history"])
                        if mode == "error":
                            self.assertEqual(capture["detection_history"][-1]["timer_error"], "OCR test failure")
                            self.assertEqual(capture["detection_history"][-1]["hud_error"], "internal OCR test failure")

    def test_complete_worker_persists_unlabeled_frames(self):
        cfg = CaptureConfig()
        frame = np.zeros((1080, 1920, 3), np.uint8)
        reading = SimpleNamespace(hud_state="match", sides={})
        detector = SimpleNamespace(
            read_frame=lambda *args: reading,
            timer_ocr=SimpleNamespace(read_frame=lambda *args, **kwargs: SimpleNamespace(kind="time", seconds=300)))
        class Reader:
            error = None
            ended = False
            metadata = {"timing": "simulated", "frame_index_definition": "simulated"}
            thread = SimpleNamespace(start=lambda: None)
            index = -1
            def get(self):
                self.index += 1
                if self.index > 50:
                    self.ended = True
                return frame, self.index*30, self.index*.5
        with tempfile.TemporaryDirectory() as directory, \
             patch("live_weapon_collect.capture.LatestCapture", return_value=Reader()), \
             patch("live_weapon_collect.capture.StartPredictor", return_value=SimpleNamespace(predict=lambda frame: {"is_start": True})), \
             patch("live_weapon_collect.capture.squid_detector", return_value=detector), \
             patch("live_weapon_collect.capture.calibrated_slot", side_effect=lambda frame, reading, side, i: SlotReading(i,"alive",1.,0.,1.)):
            capture_worker(directory, cfg, queue.Queue(), threading.Event())
            store = LiveStore(directory, Catalog())
            matches = store.list()
            self.assertEqual(len(matches), 1)
            value = store.get(matches[0]["match_id"])
            self.assertEqual(value["capture"]["status"], "complete")
            self.assertEqual(len(value["capture"]["frames"]), 5)
            self.assertFalse(value["annotation"]["confirmed"])
            self.assertTrue(all(s["weapon"] is None and not s["reviewed"] for s in value["annotation"]["slots"].values()))
            times = [r["timestamp"] for r in value["capture"]["frames"]]
            self.assertTrue(all(b-a >= cfg.interval for a,b in zip(times,times[1:])))
            self.assertLess(times[-1]-times[0], cfg.frames*cfg.interval)

    def test_replay_drains_frames_and_metadata_without_camera(self):
        stop = threading.Event()
        frames = [np.zeros((180, 320, 3), np.uint8) for _ in range(3)]
        fake = SimpleNamespace(isOpened=lambda: True, get=lambda prop: 100. if prop == 5 else 0.,
                               getBackendName=lambda: "fake", release=lambda: None)
        fake.read = lambda: (True, frames.pop()) if frames else (False, None)
        reader = LatestCapture(CaptureConfig(source="日本語 動画.mp4", replay=True), stop)
        with patch("live_weapon_collect.capture.cv2.VideoCapture", return_value=fake):
            reader.thread.start()
            reader.thread.join(timeout=2)
        self.assertIsNone(reader.error)
        self.assertTrue(reader.ended)
        self.assertEqual(reader.get()[1:], (2, .02))
        self.assertIn("decoded", reader.metadata["frame_index_definition"])


if __name__ == "__main__":
    unittest.main()
