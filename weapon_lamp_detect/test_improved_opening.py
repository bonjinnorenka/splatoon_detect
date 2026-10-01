"""Safety/aggregation tests; accuracy must be measured on actual OBS crops."""
import copy
import json
import unittest

import cv2
import numpy as np

from weapon_lamp_detect.evaluate_improved_opening import decisions
from weapon_lamp_detect.opening_matcher import CONFIGURATIONS, OpeningMatcher, processed_region, quality
from weapon_lamp_detect.test_opening import row


class ImprovedOpeningTests(unittest.TestCase):
    def samples(self):
        rows = [row(0, scores=(.9,.1)), row(1, scores=(.8,.2)), row(2, scores=(.0,1.))]
        for r, weight in zip(rows, (1.,1.,.02)):
            r['image_quality'] = {'weight':weight,'white_fraction':1-weight,'contrast':80.}
        return rows

    def test_whiteout_quality_is_lower_without_weapon_label(self):
        clear = np.full((108,134,3),(220,200,0),np.uint8)
        cv2.rectangle(clear,(30,40),(90,60),(20,20,20),-1)
        white = np.full_like(clear,255)
        self.assertGreater(quality(clear)['weight'],quality(white)['weight'])
        self.assertEqual(quality(white)['white_fraction'],1.)
        self.assertEqual(quality(white)['weight'],.02)

    def test_mean_reproduces_existing_score_mean(self):
        r = decisions(self.samples(),3,'mean')[0]
        self.assertAlmostEqual(r['best_score'],( .9+.8)/3)
        self.assertEqual(r['aggregation_weights'],{'m_0_0':1/3,'m_1_0':1/3,'m_2_0':1/3})

    def test_quality_downweights_bad_frame_and_retains_denominator(self):
        r = decisions(self.samples(),3,'quality')[0]
        self.assertTrue(r['correct'])
        self.assertAlmostEqual(r['best_score'],1.7/2.02)
        self.assertEqual(r['usable_frame_count'],3)
        self.assertEqual(r['frame_indices'],[0,60,120])
        self.assertEqual(r['decision_timestamp'],102)

    def test_median_is_robust_to_single_flash(self):
        r = decisions(self.samples(),3,'median')[0]
        self.assertTrue(r['correct'])
        self.assertAlmostEqual(r['best_score'],.8)
        self.assertAlmostEqual(r['second_best_score'],.2)

    def test_best_quality_tie_uses_first_bounded_frame(self):
        r = decisions(self.samples(),3,'best_quality')[0]
        self.assertAlmostEqual(r['best_score'],.9)
        self.assertEqual(r['aggregation_weights']['m_0_0'],1.)

    def test_no_late_rescue_and_no_state_filter(self):
        rows = self.samples()
        rows[0]['state']='down'; rows[1]['state']='unknown'
        rows.append({**row(40,scores=(1.,0.)),'image_quality':{'weight':1.}})
        for policy in ('mean','quality','median','best_quality'):
            r=decisions(rows,3,policy)[0]
            self.assertEqual(r['frame_indices'],[0,60,120])
            self.assertEqual(r['usable_frame_count'],3)

    def test_reserved_source_and_no_candidates_remain_unavailable(self):
        rows=self.samples()
        rows[0]['reserved_template_frame']=True
        rows[1]['all_candidates']=[]
        for policy in ('mean','quality','median','best_quality'):
            r=decisions(rows,3,policy)[0]
            self.assertEqual(r['usable_frame_count'],1)
            self.assertFalse(r['correct'])
            self.assertTrue(r['insufficient_frames'])
        for r in rows:
            r['all_candidates']=[]
        self.assertIsNone(decisions(rows,3,'quality')[0]['predicted'])

    def test_inputs_not_mutated_and_json_finite(self):
        rows=self.samples();original=copy.deepcopy(rows)
        for policy in ('mean','quality','median','best_quality'):
            json.dumps(decisions(rows,3,policy),allow_nan=False)
            self.assertEqual(rows,original)

    def test_all_processed_shapes_and_input_unchanged(self):
        image=np.random.default_rng(23).integers(0,255,(108,134,3),dtype=np.uint8)
        original=image.copy()
        for config,settings in CONFIGURATIONS.items():
            processed=processed_region(image,config)
            x1,y1,x2,y2=settings['region']
            self.assertEqual(processed.shape,(y2-y1,x2-x1,3))
            np.testing.assert_array_equal(image,original)

    def test_flat_templates_and_exemplars_are_finite(self):
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        fixture=AdditionalDataTests();fixture.setUp()
        try:
            _,_,_,_,source=fixture.fixtures()
            original=json.loads((source/'templates.json').read_text())
            for config in CONFIGURATIONS:
                matcher=OpeningMatcher(source,config)
                candidates=matcher.predict_crop(np.full((108,134,3),255,np.uint8),5,'left')
                self.assertTrue(candidates)
                self.assertTrue(all(0<=c.score<=1 for c in candidates))
                json.dumps([c.to_dict() for c in candidates],allow_nan=False)
                self.assertEqual(matcher.manifest['templates'],original['templates'])
        finally:
            fixture.tearDown()


if __name__=='__main__':
    unittest.main()
