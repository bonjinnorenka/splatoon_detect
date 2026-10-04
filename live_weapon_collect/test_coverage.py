"""Coverage counts distinct human-confirmed matches, not players/frames/templates."""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from live_weapon_collect.app import LiveHandler, LiveServer
from live_weapon_collect.coverage import CoverageIndex, csv_text
from weapon_lamp_detect.match_data import Catalog, SLOTS, atomic_json


def annotation(match_id, weapons=("ボールドマーカー",), confirmed=True, rejected=False, revision=1):
    slots = {key: {"weapon": None, "status": "unknown", "reviewed": True} for key in SLOTS}
    for key, weapon in zip(SLOTS, weapons):
        slots[key] = {"weapon": weapon, "status": "labeled", "reviewed": True}
    return {"match_id": match_id, "confirmed": confirmed, "rejected": rejected,
            "revision": revision, "slots": slots}


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="武器集計 空白-")
        self.root = Path(self.tmp.name)
        self.live, self.video = self.root / "live", self.root / "video"
        self.live.mkdir()
        self.video.mkdir()
        self.index = CoverageIndex(self.live, self.catalog, [self.video])

    def tearDown(self):
        self.tmp.cleanup()

    def video_match(self, value, root=None):
        path = (root or self.video) / "matches" / f'{value["match_id"]}.json'
        atomic_json(path, value)
        return path

    def live_match(self, value, status="complete", frames=5):
        directory = self.live / "matches" / value["match_id"]
        atomic_json(directory / "annotation.json", value)
        atomic_json(directory / "capture.json", {"match_id": value["match_id"], "status": status,
                                                 "frames": [{"ordinal": i} for i in range(frames)]})
        return directory / "annotation.json"

    def weapon(self, report, name="Sploosh-o-matic"):
        return next(w for w in report["weapons"] if w["name"] == name)

    def test_zero_weapons_are_listed_in_catalog_only(self):
        report = self.index.snapshot()
        self.assertEqual(len(report["weapons"]), len(self.catalog.to_list()))
        self.assertEqual({w["name"] for w in report["weapons"]}, self.catalog.label_names)
        self.assertEqual(report["summary"]["weapons_missing"], len(self.catalog.to_list()))
        self.assertTrue(all(w["remaining_matches"] == 3 and w["status"] == "missing" for w in report["weapons"]))

    def test_duplicate_slots_and_multiple_frames_count_one_match(self):
        self.live_match(annotation("live1", ("ボールドマーカー", "Sploosh-o-matic")*4))
        w = self.weapon(self.index.snapshot())
        self.assertEqual(w["match_count"], 1)
        self.assertEqual(w["live_match_count"], 1)
        self.assertEqual(w["remaining_matches"], 2)

    def test_live_and_video_are_combined_and_targets_change(self):
        self.live_match(annotation("live1"))
        self.video_match(annotation("video1"))
        report = self.index.snapshot(2)
        w = self.weapon(report)
        self.assertEqual((w["match_count"], w["video_match_count"], w["live_match_count"]), (2, 1, 1))
        self.assertEqual(w["status"], "at_target")
        self.assertEqual(self.weapon(self.index.snapshot(3))["remaining_matches"], 1)

    def test_drafts_rejected_collecting_empty_and_unreviewed_do_not_count(self):
        self.video_match(annotation("draft", confirmed=False))
        self.video_match(annotation("rejected", rejected=True))
        self.live_match(annotation("collecting"), status="collecting")
        self.live_match(annotation("empty"), frames=0)
        a = annotation("unreviewed")
        a["slots"]["left0"]["reviewed"] = False
        self.video_match(a)
        self.assertEqual(self.weapon(self.index.snapshot())["match_count"], 0)

    def test_unknown_occluded_skip_and_invalid_tags_do_not_count(self):
        a = annotation("m1", ("ボールドマーカー", "シューター", "シャープマーカー", "unknown"))
        for key, status in zip(("left0", "left2", "left3"), ("unknown", "occluded", "skip")):
            a["slots"][key]["status"] = status
        self.video_match(a)
        report = self.index.snapshot()
        self.assertEqual(report["summary"]["weapons_with_data"], 0)
        self.assertEqual(len(report["warnings"]), 1)

    def test_same_match_copies_do_not_double_count_and_latest_revision_wins(self):
        a = annotation("m1")
        self.video_match(a)
        other = self.root / "other"
        path = self.video_match(a, other)
        index = CoverageIndex(self.live, self.catalog, [self.video, other, other])
        report = index.snapshot()
        self.assertEqual(self.weapon(report)["match_count"], 1)
        self.assertEqual(report["summary"]["duplicate_copies_ignored"], 1)
        a.update(confirmed=False, revision=2)
        atomic_json(path, a)
        self.assertEqual(self.weapon(index.snapshot())["match_count"], 0)

    def test_equal_revision_conflicts_are_reported_not_added_as_two_weapons(self):
        self.video_match(annotation("m1"))
        other = self.root / "other"
        self.video_match(annotation("m1", ("シャープマーカー",)), other)
        report = CoverageIndex(self.live, self.catalog, [self.video, other]).snapshot()
        self.assertEqual(report["summary"]["weapons_with_data"], 0)
        self.assertEqual(report["summary"]["unique_confirmed_matches"], 0)
        self.assertIn("異なるラベル", report["warnings"][0])

    def test_interrupted_human_confirmed_images_are_explicitly_counted(self):
        self.live_match(annotation("partial"), status="interrupted", frames=1)
        w = self.weapon(self.index.snapshot())
        self.assertEqual(w["match_count"], 1)
        self.assertEqual(w["incomplete_match_count"], 1)

    def test_edits_undo_and_file_removal_refresh_cache_without_writing_sources(self):
        path = self.video_match(annotation("m1"))
        original = path.read_bytes()
        self.assertEqual(self.weapon(self.index.snapshot())["match_count"], 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.weapon(self.index.snapshot())["match_count"], 1)
        atomic_json(path, annotation("m1", ("シャープマーカー",), revision=2))
        report = self.index.snapshot()
        self.assertEqual(self.weapon(report)["match_count"], 0)
        self.assertEqual(self.weapon(report, "Splash-o-matic")["match_count"], 1)
        path.unlink()
        self.assertEqual(self.weapon(self.index.snapshot(), "Splash-o-matic")["match_count"], 0)

    def test_missing_and_malformed_sources_are_visible(self):
        bad = self.video / "matches" / "invalid.json"
        bad.parent.mkdir()
        bad.write_text('{"match_id":"bad"}', encoding="utf-8")
        report = CoverageIndex(self.live, self.catalog, [self.video, self.root / "missing"]).snapshot()
        self.assertEqual(len(report["warnings"]), 2)
        self.assertFalse(report["sources"][-1]["exists"])

    def test_csv_includes_japanese_names_and_filters_met_target(self):
        self.video_match(annotation("v1"))
        self.live_match(annotation("l1"))
        report = self.index.snapshot(2)
        all_rows = list(csv.reader(io.StringIO(csv_text(report))))
        missing_rows = list(csv.reader(io.StringIO(csv_text(report, True))))
        self.assertEqual(len(all_rows), len(self.catalog.to_list())+1)
        self.assertTrue(any(row[0] == "ボールドマーカー" for row in all_rows))
        self.assertFalse(any(row[0] == "ボールドマーカー" for row in missing_rows))

    def test_invalid_targets(self):
        for target in (0, 1, 4, 2.5, "3", True):
            with self.assertRaises(ValueError):
                self.index.snapshot(target)

    def test_coverage_http_json_and_csv_without_sockets(self):
        self.video_match(annotation("v1"))
        server = LiveServer.__new__(LiveServer)
        server.coverage = self.index
        def get(path):
            h = LiveHandler.__new__(LiveHandler)
            h.server, h.path = server, path
            h.request_version, h.command, h.requestline = "HTTP/1.1", "GET", "GET " + path
            h.headers = {"Host": "127.0.0.1:8780"}
            h.wfile = io.BytesIO()
            h.do_GET()
            headers, body = h.wfile.getvalue().split(b"\r\n\r\n", 1)
            return int(headers.split()[1]), body
        status, data = get("/api/coverage?target=2")
        self.assertEqual(status, 200)
        import json
        self.assertEqual(self.weapon(json.loads(data))["match_count"], 1)
        status, data = get("/api/coverage.csv?target=3&missing_only=1")
        self.assertEqual(status, 200)
        self.assertIn("ボールドマーカー", data.decode("utf-8-sig"))
        self.assertEqual(get("/api/coverage?target=0")[0], 400)


if __name__ == "__main__":
    unittest.main()
