"""Workflow/integrity tests. Artificial fixtures are NOT real OBS accuracy evidence."""
from __future__ import annotations

import json
import io
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weapon_lamp_detect.build_dataset import build_dataset, load_dataset
from weapon_lamp_detect.build_obs_templates import build_templates, make_split
from weapon_lamp_detect.detect_matches import StartPredictor, register_candidates
from weapon_lamp_detect.evaluate_poc import aggregate, evaluate, metrics, prediction, validate_split
from weapon_lamp_detect.label_matches import MatchHandler
from weapon_lamp_detect.compare_runs import compare_runs, paired_rows
from weapon_lamp_detect.check_region_training import check_training
from weapon_lamp_detect.region_matcher import REGION, masked_zncc, weapon_region
from weapon_lamp_detect.match_data import Catalog, MatchStore, SLOTS, atomic_json, blank_match, calibrated_crop, calibrated_slot, canonical_crop, default_geometry, frame_at, video_metadata


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="weapon-poc-test-")
        self.root = Path(self.temp.name)
        self.catalog = Catalog()
        # Artificial fixtures have prescribed states; real classification is checked separately.
        self.state_patch = patch("weapon_lamp_detect.build_dataset.calibrated_slot",
                                 side_effect=lambda f,r,s,i,g: r.sides[s].slots[i])
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.temp.cleanup()

    def video(self):
        path = self.root / "OBS テスト 空白.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        self.assertTrue(writer.isOpened())
        random = np.random.default_rng(9)
        # Every frame is an artificial HUD-like fixture, not manually labeled OBS footage.
        base = random.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        for _ in range(120):
            writer.write(base)
        writer.release()
        return path

    def store(self):
        store = MatchStore(self.root / "session", self.catalog)
        store.add_videos([self.video()])
        return store, next(iter(store.videos.values()))

    def confirmed(self, store, video, start, end):
        match = blank_match(video, start, end)
        match.update(reference_timestamp=start+1, reference_frame_index=round((start+1)*video["fps"]),
                     boundaries_confirmed=True, confirmed=True)
        for i, key in enumerate(SLOTS):
            match["slots"][key] = {"weapon": "スプラシューター" if i % 2 == 0 else "シャープマーカー",
                                   "status": "labeled", "reviewed": True}
        return store.save(match)

    def test_catalog_and_path_independent_identity(self):
        self.assertEqual(self.catalog.resolve("スプラシューター"), "Splattershot")
        self.assertEqual(self.catalog.entries[self.catalog.resolve("ジムワイパー")]["weapon_class"], "splatana")
        first = self.video()
        copy = self.root / "other.avi"
        shutil.copyfile(first, copy)
        self.assertEqual(video_metadata(first)["video_id"], video_metadata(copy)["video_id"])
        with self.assertRaises(ValueError):
            self.catalog.resolve("made up weapon")
        frame, index, t = frame_at(video_metadata(first), frame_index=15)
        self.assertEqual((index, t), (15, 1.5))
        self.assertEqual(frame.shape[:2], (360, 640))

    def test_label_choices_are_json_weapon_names_not_classes_or_template_extras(self):
        data = json.loads(self.catalog.catalog_path.read_text(encoding="utf-8"))
        choices = self.catalog.to_list()
        self.assertEqual([entry["display_name"] for entry in choices],
                         [weapon["name"] for weapon in data["weapons"]])
        self.assertEqual(len(choices), data["count"])
        self.assertTrue(all(entry["category"] == weapon["category"]
                            for entry, weapon in zip(choices, data["weapons"])))
        with self.assertRaises(ValueError):
            self.catalog.resolve("シューター")
        custom = self.root / "武器名 空白.json"
        custom.write_text(json.dumps({"weapons": [{"name": "スプラシューター", "category": "シューター"}]},
                                     ensure_ascii=False), encoding="utf-8")
        catalog = Catalog(catalog_path=custom)
        self.assertEqual(catalog.catalog_path, custom.resolve())
        self.assertEqual([entry["display_name"] for entry in catalog.to_list()], ["スプラシューター"])
        self.assertEqual(catalog.resolve("スプラシューター"), "Splattershot")
        # Existing canonical annotations remain usable even outside this UI list.
        self.assertEqual(catalog.resolve("Splash-o-matic"), "Splash-o-matic")
        with self.assertRaisesRegex(ValueError, "無効な武器名"):
            catalog.resolve_label("Splash-o-matic")
        self.assertEqual(catalog.resolve_label("スプラシューター"), "Splattershot")
        self.assertEqual(catalog.resolve_label("Splattershot"), "Splattershot")
        choices[0]["display_name"] = "edited externally"
        self.assertNotEqual(self.catalog.to_list()[0]["display_name"], "edited externally")

    def test_explicit_missing_catalog_does_not_silently_fallback(self):
        with self.assertRaisesRegex(ValueError, "指定した武器catalogがありません"):
            Catalog(catalog_path=self.root / "missing.json")

    def test_foreground_removes_ink_not_neutral_weapon_details(self):
        images = []
        for hue in (10,110):
            hsv = np.full((108,134,3),(hue,255,220),np.uint8)
            image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            image[38:65,40:94] = (230,230,230)
            image[45:60,65:90] = (20,20,20)
            images.append(image)
        plain = [weapon_region(i) for i in images]
        clean = [weapon_region(i,True,True) for i in images]
        self.assertFalse(np.array_equal(plain[0],plain[1]))
        self.assertTrue(np.array_equal(clean[0][0],clean[1][0]))
        self.assertEqual(clean[0][0].shape[:2],(REGION[3]-REGION[1],REGION[2]-REGION[0]))
        self.assertGreater(np.mean(clean[0][1]>0),.5)
        self.assertTrue(np.any(np.all(clean[0][0]==(20,20,20),axis=2)))
        self.assertTrue(np.any(np.all(clean[0][0]==(230,230,230),axis=2)))
        flat = np.full((108,134,3),80,np.uint8)
        region,mask = weapon_region(flat,True,True)
        self.assertTrue(np.all(region==80))
        self.assertFalse(np.any(mask))

    def test_masked_zncc_brightness_invariance_and_flat_rejection(self):
        rng = np.random.default_rng(11)
        template = rng.integers(10,90,(12,15),dtype=np.uint8)
        image = np.zeros((22,30),np.uint8)
        image[4:16,7:22] = template*2+20
        mask = np.full(template.shape,255,np.uint8)
        result = masked_zncc(image,template,mask)
        _,score,_,point = cv2.minMaxLoc(result)
        self.assertEqual(point,(7,4))
        self.assertAlmostEqual(score,1.,places=5)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(masked_zncc(np.full(image.shape,200,np.uint8),template,mask)==-1))
        self.assertIsNone(masked_zncc(image,template,np.zeros_like(mask)))

    def test_hud_default_geometry_and_twenty_second_reference(self):
        video = {"video_id": "a"*24, "path": "fixture", "duration": 90, "fps": 60,
                 "frame_count": 5400, "width": 1920, "height": 1080}
        match = blank_match(video, 2.5, 90, "hog_svm")
        self.assertEqual(match["start_timestamp"], 2.5)
        self.assertEqual(match["start_event"], "stage_intro")
        self.assertEqual(match["reference_timestamp"], 22.5)
        self.assertEqual(match["reference_frame_index"], 1350)
        self.assertEqual(blank_match(video, 2.5, 90, hud_offset=15)["reference_timestamp"], 17.5)
        frame = np.zeros((1080,1920,3), np.uint8)
        for side, expected in (("left", [565,655,745,835]), ("right", [1085,1175,1265,1355])):
            rects = []
            for i in range(4):
                crop, rect = calibrated_crop(frame, side, i)
                rects.append(rect)
                self.assertLessEqual(abs((rect[0]+rect[2])/2-expected[i]), 1)
                self.assertLessEqual(crop.shape[1], 101)
                self.assertLessEqual(crop.shape[0], 101)
            for a,b in zip(rects,rects[1:]):
                self.assertLessEqual(a[2]-b[0], 12)
        reading = self.detector().read_frame()
        reading.sides["left"].ink_color = object()
        with patch("squid_lamp_detect.squid_lamp.classify_slot", return_value=reading.sides["left"].slots[0]) as classify:
            calibrated_slot(frame, reading, "left", 0)
            self.assertEqual(classify.call_args.args[0].shape, calibrated_crop(frame,"left",0)[0].shape)
        normalized = canonical_crop(np.full((100,100,3),255,np.uint8))
        self.assertTrue(np.all(normalized[:,:13] == 0))
        self.assertTrue(np.all(normalized[:,13:121] == 255))

    def test_upgrade_only_unedited_legacy_defaults(self):
        store, video = self.store()
        match = blank_match(video, 1, 12, "hog_svm")
        match.update(geometry={}, reference_timestamp=6, reference_frame_index=60)
        for key in ("hud_offset_seconds", "crop_profile", "reference_source", "start_event"):
            match.pop(key)
        saved = store.save(match)
        self.assertEqual(store.upgrade_unedited_matches(),1)
        updated = store.matches()[0]
        self.assertEqual(updated["reference_timestamp"], 11.9)
        self.assertEqual(updated["geometry"], default_geometry())
        self.assertEqual(updated["start_timestamp"], saved["start_timestamp"])
        self.assertEqual(updated["slots"], saved["slots"])
        custom = blank_match(video, 0, 5)
        custom.update(geometry={}, reference_timestamp=1, reference_frame_index=10)
        store.save(custom)
        self.assertEqual(store.upgrade_unedited_matches(),0)
        self.assertEqual(store.matches()[0]["reference_timestamp"],1)
        custom = store.matches()[0]
        custom.update(reference_timestamp=4.9,reference_frame_index=49)
        store.save(custom)
        self.assertEqual(store.upgrade_unedited_matches(),0)

    def test_sampling_uses_match_hud_wait_unless_cli_overrides(self):
        store, video = self.store()
        match = self.confirmed(store, video, 0, 11)
        match["hud_offset_seconds"] = 2.5
        store.save(match)
        with patch("weapon_lamp_detect.build_dataset.squid_detector", side_effect=lambda _: self.detector()):
            metadata = build_dataset(store.directory, self.root / "match_offset", max_frames_per_match=1)
            overridden = build_dataset(store.directory, self.root / "override_offset", start_offset=1, max_frames_per_match=1)
        _, rows = load_dataset(self.root / "match_offset")
        self.assertEqual(rows[0]["timestamp"],2.5)
        self.assertEqual(metadata["effective_start_offsets"][match["match_id"]],2.5)
        self.assertEqual(overridden["effective_start_offsets"][match["match_id"]],1)

    def test_store_confirmation_revision_resume_and_candidate_protection(self):
        store, video = self.store()
        m = blank_match(video, 1, 5)
        m["confirmed"] = True
        with self.assertRaises(ValueError):
            store.save(m)
        m["confirmed"] = False
        m = store.save(m)
        stale = dict(m)
        m["notes"] = "human boundary correction"
        saved = store.save(m)
        self.assertEqual(saved["created_at"], m["created_at"])
        with self.assertRaises(ValueError):
            store.save(stale)
        resumed = MatchStore(store.directory, self.catalog)
        self.assertEqual(resumed.matches()[0]["notes"], saved["notes"])
        register_candidates(store, video, {"candidates": [{"timestamp": 1, "score": 2}]})
        self.assertEqual(store.matches()[0]["notes"], saved["notes"])
        with self.assertRaises(ValueError):
            store.path("../../escape")
        m = dict(saved)
        m["reference_timestamp"] = float("nan")
        with self.assertRaises(ValueError):
            store.save(m)

    def test_invalid_weapon_save_preserves_annotation_and_history(self):
        store, video = self.store()
        saved = self.confirmed(store, video, 0, 5)
        path = store.path(saved["match_id"])
        before = path.read_bytes()
        history = list((store.directory / "history").rglob("*.json"))
        template_only = next(name for name in self.catalog.entries if name not in self.catalog.label_names)
        for name in ("シューター", "存在しない武器", "スプラシュー", template_only, "", None, 123):
            bad = json.loads(before)
            bad["slots"]["left0"]["weapon"] = name
            for confirmed in (False, True):
                bad["confirmed"] = confirmed
                with self.assertRaisesRegex(ValueError, "left0:.*保存していません"):
                    store.save(bad)
                # Exercise the HTTP save handler too, without opening a socket.
                body = json.dumps(bad).encode()
                handler = MatchHandler.__new__(MatchHandler)
                handler.path = "/api/save"
                handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
                handler.rfile = io.BytesIO(body)
                handler.server = SimpleNamespace(store=store, lock=threading.RLock())
                responses = []
                handler.send = lambda value, status=200: responses.append((value, status))
                handler.do_POST()
                self.assertEqual(responses[0][1], 400)
                self.assertIn("left0", responses[0][0]["error"])
                self.assertFalse((store.directory / "resume.json").exists())
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list((store.directory / "history").rglob("*.json")), history)
        # A failed request cannot mutate the last confirmed ground truth.
        self.assertTrue(store.matches()[0]["confirmed"])
        valid = json.loads(before)
        valid["slots"]["left0"]["weapon"] = "ボールドマーカー"
        updated = store.save(valid)
        self.assertEqual(updated["slots"]["left0"]["weapon"], "Sploosh-o-matic")
        self.assertEqual(updated["revision"], saved["revision"] + 1)

    def test_missing_human_labels_and_invalid_sampling(self):
        store, video = self.store()
        store.save(blank_match(video, 0, 5))
        with self.assertRaisesRegex(ValueError, "人力確定"):
            build_dataset(store.directory, self.root / "dataset")
        with self.assertRaises(ValueError):
            build_dataset(store.directory, self.root / "dataset", interval=0)

    @staticmethod
    def detector():
        slots = [SimpleNamespace(state="alive" if i != 1 else "down", special_score=0) for i in range(4)]
        reading = SimpleNamespace(hud_state="match", sides={s: SimpleNamespace(slots=slots) for s in ("left", "right")})
        return SimpleNamespace(read_frame=lambda *args, **kwargs: reading)

    def test_full_dataset_templates_compare_and_leak_guard(self):
        store, video = self.store()
        self.confirmed(store, video, 0, 5)
        self.confirmed(store, video, 6, 11)
        dataset, templates, report = (self.root / n for n in ("dataset", "templates", "report"))
        with patch("weapon_lamp_detect.build_dataset.squid_detector", side_effect=lambda _: self.detector()):
            build_dataset(store.directory, dataset, interval=.3, start_offset=.5, end_offset=.1,
                          states=("alive", "down", "unknown"), export=True, max_frames_per_match=12)
        metadata, rows = load_dataset(dataset)
        self.assertEqual(len(rows), 192)
        self.assertEqual({r["state"] for r in rows}, {"alive", "down"})
        self.assertTrue((dataset / rows[0]["crop"]).exists())
        manifest = build_templates(dataset, templates)
        selected = validate_split(manifest, rows, dataset)
        self.assertEqual(len(selected), 96)
        result = evaluate(dataset, report, "compare", templates)
        for method in ("official", "obs"):
            self.assertEqual(result["methods"][method]["single_frame"]["metrics"]["support"], 96)
            self.assertEqual(result["methods"][method]["multi_frame"]["mean_score"]["10"]["eligible_groups"], 8)
            self.assertIn("margins", result["methods"][method]["single_frame"])
        self.assertTrue((report / "predictions.jsonl").exists())
        parallel_dir = self.root / "parallel_report"
        parallel = evaluate(dataset, parallel_dir, "compare", templates, workers=2)
        self.assertEqual(result["methods"], parallel["methods"])
        self.assertEqual((report / "predictions.jsonl").read_bytes(),
                         (parallel_dir / "predictions.jsonl").read_bytes())
        process_dir = self.root / "process_report"
        process = evaluate(dataset, process_dir, "compare", templates, workers=2, executor="process")
        self.assertEqual(result["methods"],process["methods"])
        self.assertEqual((report/"predictions.jsonl").read_bytes(),(process_dir/"predictions.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "候補集合"):
            evaluate(dataset, self.root / "uncollected", weapons=["ボールドマーカー"])
        with self.assertRaisesRegex(ValueError, "workers"):
            evaluate(dataset, self.root / "bad_workers", workers=0)
        with self.assertRaisesRegex(ValueError, "候補集合"):
            evaluate(dataset, self.root / "small_limit", max_weapons=1)
        with self.assertRaisesRegex(ValueError, "候補数"):
            build_templates(dataset, self.root / "small_template_limit", max_weapons=1)
        regions_dir = self.root / "regions_report"
        frozen = check_training(dataset,templates,regions_dir,workers=2)
        self.assertTrue(set(frozen["training_check_sample_ids"]).isdisjoint(selected))
        self.assertEqual(frozen["training_match_ids"],manifest["split"]["train_match_ids"])
        leak_dir=self.root/"leaked_configuration"
        atomic_json(leak_dir/"configuration_frozen.json",{**frozen,"training_check_sample_ids":[next(iter(selected))]})
        with self.assertRaisesRegex(ValueError,"調整用sample"):
            evaluate(dataset,leak_dir,"regions",templates)
        changed_dir=self.root/"changed_configuration"
        atomic_json(changed_dir/"configuration_frozen.json",{**frozen,"region":[0,0,134,108]})
        with self.assertRaisesRegex(ValueError,"固定設定"):
            evaluate(dataset,changed_dir,"regions",templates)
        regions = evaluate(dataset, regions_dir, "regions", templates, workers=2)
        for method in ("official_region","official_foreground","obs_region","obs_foreground"):
            self.assertEqual(regions["methods"][method]["single_frame"]["metrics"]["support"],96)
            self.assertEqual(regions["provenance"][method]["region"],list(REGION))
        self.assertEqual(regions["provenance"]["obs_foreground"]["templates_manifest"]["split"],manifest["split"])
        self.assertTrue(list((regions_dir/"processed_templates"/"obs_foreground").glob("*.png")))
        single_foreground = evaluate(dataset,self.root/"single_foreground","obs_foreground",templates,workers=2)
        self.assertEqual(set(single_foreground["methods"]),{"obs_foreground"})
        self.assertEqual(single_foreground["methods"]["obs_foreground"],regions["methods"]["obs_foreground"])
        self.assertEqual(single_foreground["provenance"]["obs_foreground"]["region"],list(REGION))
        with self.assertRaisesRegex(ValueError,"上書き"):
            check_training(dataset,templates,regions_dir)
        comparison = compare_runs(report,regions_dir,self.root/"improvement")
        self.assertEqual(len(comparison["methods"]),6)
        self.assertTrue(comparison["paired_comparison"])
        self.assertTrue((self.root/"improvement"/"comparison.md").exists())
        unchanged=(self.root/"improvement"/"predictions.jsonl").read_bytes()
        compare_runs(report,regions_dir,self.root/"improvement",refresh_summary=True)
        self.assertEqual((self.root/"improvement"/"predictions.jsonl").read_bytes(),unchanged)
        with self.assertRaisesRegex(ValueError,"上書き"):
            compare_runs(report,regions_dir,self.root/"improvement")
        example = json.loads((report/"predictions.jsonl").read_text().splitlines()[0])
        changed = {**example,"method":"new","state":"unknown"}
        with self.assertRaisesRegex(ValueError,"一致"):
            paired_rows([example],[changed])
        with self.assertRaisesRegex(ValueError,"sample集合"):
            paired_rows([example],[{**changed,"sample_id":"different"}])
        manifest["split"]["evaluation_sample_ids"].append(manifest["split"]["template_sample_ids"][0])
        with self.assertRaisesRegex(ValueError, "split"):
            validate_split(manifest, rows, dataset)

    def test_alive_only_sparse_cap_and_metadata_only_regeneration(self):
        store, video = self.store()
        match = self.confirmed(store, video, 0, 5)
        match["slots"]["left0"] = {"weapon": None, "status": "occluded", "reviewed": True}
        store.save(match)
        dataset = self.root / "dataset"
        with patch("weapon_lamp_detect.build_dataset.squid_detector", side_effect=lambda _: self.detector()):
            build_dataset(store.directory, dataset, interval=1, start_offset=.5, end_offset=.5, max_samples=9)
        _, rows = load_dataset(dataset)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r["state"] == "alive" and "crop" not in r for r in rows))
        self.assertTrue(all(r["side"]+str(r["slot_index"]) != "left0" for r in rows))
        from weapon_lamp_detect.build_dataset import sample_crop
        meta, _ = load_dataset(dataset)
        self.assertEqual(sample_crop(dataset, rows[0], meta).shape[:2], (108, 134))

    def test_calibrated_geometry_survives_export_and_regeneration(self):
        store, video = self.store()
        match = self.confirmed(store, video, 0, 5)
        match["geometry"] = {"left0": [.3, .07, .06, .1]}
        saved = store.save(match)
        self.assertEqual(saved["geometry"], match["geometry"])
        frame, _, _ = frame_at(video, frame_index=10)
        crop, rect = calibrated_crop(frame, "left", 0, match["geometry"])
        self.assertEqual(rect, (173, 7, 211, 43))
        self.assertEqual(crop.shape[:2], (36, 38))
        bad = dict(saved)
        bad["geometry"] = {"left0": [.3, .07, -1, .1]}
        with self.assertRaises(ValueError):
            store.save(bad)
        dataset = self.root / "calibrated_dataset"
        with patch("weapon_lamp_detect.build_dataset.squid_detector", side_effect=lambda _: self.detector()), \
             patch("weapon_lamp_detect.build_dataset.calibrated_slot", side_effect=lambda f,r,s,i,g: r.sides[s].slots[i]):
            build_dataset(store.directory, dataset, interval=1, start_offset=1, end_offset=1, max_frames_per_match=1)
        metadata, rows = load_dataset(dataset)
        left0 = next(r for r in rows if r["side"] == "left" and r["slot_index"] == 0)
        self.assertEqual(left0["rect"], list(rect))
        from weapon_lamp_detect.build_dataset import sample_crop
        self.assertEqual(sample_crop(dataset, left0, metadata).shape[:2], (108,134))

    def test_relocated_source_and_edited_annotation_validation(self):
        store, video = self.store()
        match = self.confirmed(store, video, 0, 5)
        moved = self.root / "移動後の動画 空白.avi"
        Path(video["path"]).rename(moved)
        store.add_videos([moved])
        dataset = self.root / "moved_dataset"
        with patch("weapon_lamp_detect.build_dataset.squid_detector", side_effect=lambda _: self.detector()):
            build_dataset(store.directory, dataset, interval=1, start_offset=1, end_offset=1, max_frames_per_match=1)
        _, rows = load_dataset(dataset)
        self.assertEqual(rows[0]["source_video"], str(moved))
        from weapon_lamp_detect.match_data import atomic_json
        match["slots"]["left0"]["reviewed"] = False
        atomic_json(store.path(match["match_id"]), match)
        with self.assertRaisesRegex(ValueError, "人力入力"):
            build_dataset(store.directory, self.root / "invalid_dataset")

    def test_within_match_is_explicit_and_never_cross_match(self):
        rows = [{"sample_id": f"s{i}", "match_id": "m", "video_id": "v", "frame_index": i,
                 "weapon_label": "Splattershot", "state": "alive", "side": "left", "slot_index": 0} for i in range(5)]
        with self.assertRaises(ValueError):
            make_split(rows, ["Splattershot"])
        split = make_split(rows, ["Splattershot"], True)
        self.assertEqual(set(split["evaluation_scope"].values()), {"within_match_other_timestamp"})
        self.assertFalse(set(split["template_sample_ids"]) & set(split["evaluation_sample_ids"]))

    def test_temporal_state_separation_margin_and_support(self):
        rows = []
        for i in range(12):
            row = {"sample_id": str(i), "match_id": "m", "side": "left", "slot_index": 0,
                   "frame_index": i, "state": "alive", "weapon_label": "A", "weapon_class": "shooter"}
            cs = [{"weapon": "A", "weapon_class": "shooter", "score": .8, "confidence": .6},
                  {"weapon": "B", "weapon_class": "roller", "score": .5, "confidence": .4}]
            rows.append(prediction(row, cs, "official", "cross_match"))
        self.assertAlmostEqual(rows[0]["margin"], .3)
        result = aggregate(rows)
        for n in (1, 3, 5, 10):
            self.assertEqual(result["mean_score"][str(n)]["metrics"]["support"], 1)
            self.assertEqual(result["mean_score"][str(n)]["common_cohort"]["metrics"]["exact_top1"]["accuracy"], 1)
        rows[-1]["state"] = "down"
        result = aggregate(rows)
        self.assertEqual(result["mean_score"]["3"]["insufficient_groups"], 1)
        self.assertIsNone(metrics([])["exact_top1"]["accuracy"])

    def test_hog_json_matches_existing_pickle_prediction(self):
        from start_detect import hog_svm_predictor as existing
        sys.modules.setdefault("hog_svm_predictor", existing)
        model = existing.HOGSVMImagePredictor().load_pickle(Path(existing.__file__).parent / "hog_svm_model.pkl")
        predictor = StartPredictor()
        fixture = next((Path(existing.__file__).parent / "data" / "start").glob("*.png"))
        crop = cv2.imread(str(fixture))
        frame = np.zeros((1080, 1920, 3), np.uint8)
        frame[260:660, 760:1160] = cv2.resize(crop, (400, 400))
        expected = model.predict_image(frame[260:660, 760:1160])
        actual = predictor.predict(frame)
        self.assertAlmostEqual(actual["score"], expected["score"], places=7)
        self.assertEqual(actual["is_start"], expected["prediction"] == 1)
        self.assertAlmostEqual(actual["probability"], expected["start_probability"], places=7)


if __name__ == "__main__":
    unittest.main()
