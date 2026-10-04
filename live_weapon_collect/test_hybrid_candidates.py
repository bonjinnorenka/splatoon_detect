"""Portable templates preserve the frozen matcher and never use live labels."""
import copy
import shutil
import unittest
from pathlib import Path

import numpy as np

from live_weapon_collect.hybrid_candidates import bundle_info, load, prepare
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.hybrid_opening_matcher import HybridOpeningMatcher
from weapon_lamp_detect.match_data import atomic_json, read_json
from weapon_lamp_detect import test_opening_bank as bank_tests


class HybridCandidateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = bank_tests.OpeningBankTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.experiment = self.root / "experiment"
        self.bank = self.fixture.bank()
        atomic_json(self.experiment / "configuration_frozen.json", {"selected_configuration": "blend_soft"})
        atomic_json(self.experiment / "full_training" / "templates.json", self.bank)
        self.output = self.root / "portable"

    def tearDown(self):
        self.fixture.tearDown()

    def test_moved_bundle_scores_equal_original_on_both_sides(self):
        info = prepare(self.experiment, self.output)
        moved = self.root / "日本語 空白/moved bundle"
        shutil.copytree(self.output, moved)
        original = HybridOpeningMatcher(self.bank, "blend_soft")
        restored = load(moved)
        self.assertEqual(info["weapon_count"], 2)
        self.assertEqual(bundle_info(moved)["configuration"], "blend_soft")
        metadata, rows = load_dataset(self.fixture.dataset)
        crop = sample_crop(self.fixture.dataset, rows[0], metadata)
        for side in ("left", "right"):
            expected = original.predict_crop(crop, 5, side)
            actual = restored.predict_crop(crop, 5, side)
            self.assertEqual([c.weapon for c in expected], [c.weapon for c in actual])
            np.testing.assert_allclose([c.score for c in expected], [c.score for c in actual], atol=1e-7)
        for engine in restored.engines.values():
            self.assertTrue(all(Path(entry['dataset']).is_relative_to(moved) for entry, _ in engine.templates))

    def test_only_selected_sources_are_exported_and_provenance_kept(self):
        original = copy.deepcopy(self.bank)
        info = prepare(self.experiment, self.output)
        expected = {i for e in original["templates"] for i in e["sample_ids"]}
        actual = {r['sample_id'] for path in self.output.glob('dataset_*') for r in load_dataset(path)[1]}
        self.assertEqual(actual, expected)
        self.assertFalse(any(i.startswith('test_') for i in actual))
        self.assertEqual([e['source_frames'] for e in info['bank']['templates']],
                         [e['source_frames'] for e in original['templates']])
        self.assertEqual(read_json(self.experiment/'full_training/templates.json'), original)

    def test_missing_bundle_has_warning_not_official_fallback(self):
        self.assertFalse(bundle_info(self.output)['available'])
        with self.assertRaisesRegex(ValueError, 'hybrid見本がありません'):
            load(self.output)

    def test_corrupt_crop_refused(self):
        info = prepare(self.experiment, self.output)
        path = next(name for name in info['files'] if name.endswith('.png'))
        (self.output/path).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, '破損'):
            load(self.output)

    def test_outside_dataset_path_refused(self):
        info = prepare(self.experiment, self.output)
        info['bank']['templates'][0]['dataset'] = '../outside'
        atomic_json(self.output/'bundle.json', info)
        with self.assertRaisesRegex(ValueError, '保存先の外'):
            load(self.output)

    def test_existing_output_refused(self):
        prepare(self.experiment, self.output)
        before = (self.output/'bundle.json').read_bytes()
        with self.assertRaisesRegex(ValueError, '上書きしません'):
            prepare(self.experiment, self.output)
        self.assertEqual(before, (self.output/'bundle.json').read_bytes())

    def test_corrupt_manifest_does_not_break_status(self):
        prepare(self.experiment, self.output)
        (self.output/'bundle.json').write_bytes(b'incomplete json')
        info = bundle_info(self.output)
        self.assertFalse(info['available'])
        self.assertIn('収集・人力入力は継続', info['notice'])


if __name__ == '__main__':
    unittest.main()
