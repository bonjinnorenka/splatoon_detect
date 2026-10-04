"""Filesystem-only tests; never touch user session markers or labels."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weapon_label_review.locking import VideoSaveBusy, VideoSaveLock


class VideoLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='video-lock-日本語 空白-')
        self.directory = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mutex_and_release_do_not_use_byte_range_lock(self):
        legacy = self.directory/'annotation.lock'
        legacy.write_bytes(b'0')
        with patch('live_weapon_collect.io_utils.FileLock', side_effect=AssertionError('OS locking forbidden')):
            with VideoSaveLock(self.directory) as first:
                before = first.owner_path.read_bytes()
                with self.assertRaisesRegex(VideoSaveBusy, 'pid='):
                    VideoSaveLock(self.directory)
                self.assertEqual(first.owner_path.read_bytes(), before)
            self.assertFalse(first.path.exists())
            first.close()  # Idempotent; doesn't remove the next owner's lock.
            with VideoSaveLock(self.directory) as second:
                first.close()
                self.assertTrue(second.path.exists())
        self.assertEqual(legacy.read_bytes(), b'0')

    def test_second_process_cannot_enter(self):
        code = ('from pathlib import Path; import sys; '
                'from weapon_label_review.locking import VideoSaveLock; '
                'VideoSaveLock(Path(sys.argv[1]))')
        with VideoSaveLock(self.directory) as owner:
            before = owner.owner_path.read_bytes()
            result = subprocess.run([sys.executable, '-c', code, str(self.directory)],
                                    capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('VideoSaveBusy', result.stderr)
            self.assertEqual(owner.owner_path.read_bytes(), before)

    def test_stale_or_incomplete_marker_is_not_automatically_deleted(self):
        marker = self.directory/'annotation.review.lock'
        marker.mkdir()
        for value in (None, {'token':'dead', 'pid':0, 'host':'other machine'}):
            if value is not None:
                (marker/'owner.json').write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(VideoSaveBusy):
                VideoSaveLock(self.directory)
            self.assertTrue(marker.exists())
            if value is not None:
                self.assertEqual(json.loads((marker/'owner.json').read_text()), value)

    def test_release_cannot_remove_a_different_owner(self):
        owner = VideoSaveLock(self.directory)
        owner.owner_path.write_text(json.dumps({'token':'different'}), encoding='utf-8')
        with self.assertWarnsRegex(UserWarning, '所有者が変わった'):
            owner.close()
        self.assertTrue(owner.path.exists())

    def test_permission_failure_is_not_reported_as_busy(self):
        with patch.object(Path, 'mkdir', side_effect=PermissionError('access denied')):
            with self.assertRaisesRegex(OSError, '作成できません.*access denied'):
                VideoSaveLock(self.directory)

    def test_initialization_failure_cleans_only_created_marker(self):
        with patch('weapon_label_review.locking.os.fsync', side_effect=OSError('sync failed')):
            with self.assertRaisesRegex(OSError, 'sync failed'):
                VideoSaveLock(self.directory)
        self.assertFalse((self.directory/'annotation.review.lock').exists())

    def test_exception_in_body_releases_marker(self):
        with self.assertRaisesRegex(ValueError, 'save refused'):
            with VideoSaveLock(self.directory):
                raise ValueError('save refused')
        self.assertFalse((self.directory/'annotation.review.lock').exists())


if __name__ == '__main__':
    unittest.main()
