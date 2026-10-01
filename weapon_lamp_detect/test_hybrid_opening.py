import copy
import unittest
from unittest.mock import patch

from weapon_lamp_detect.hybrid_opening_matcher import CONFIGURATIONS, HybridOpeningMatcher, fuse, hybrid_bank
from weapon_lamp_detect.weapon_lamp import MatchCandidate


class HybridOpeningTests(unittest.TestCase):
    def candidates(self):
        scores={'legacy_same':(.8,.75),'legacy_shared':(.9,.76),'local_same':(.2,.9),'local_shared':(.3,.95)}
        return {key:[MatchCandidate(name,'shooter',score,score,(0,0),(106,72),0,{}) for name,score in zip(('A','B'),values)] for key,values in scores.items()}

    def test_soft_side_and_feature_blend_are_numeric_not_pair_specific(self):
        r=fuse(self.candidates(),'blend_soft',5)
        self.assertEqual(r[0].weapon,'B')
        self.assertAlmostEqual(r[0].score,.75*.75+.25*.92)
        self.assertAlmostEqual(r[1].score,.75*.87+.25*.27)

    def test_hard_side_ignores_opposite_boost(self):
        r=fuse(self.candidates(),'legacy_hard',5)
        self.assertEqual(r[0].weapon,'A')
        self.assertAlmostEqual(r[0].score,.8)

    def test_zero_penalty_reuses_better_opposite_side(self):
        r=fuse(self.candidates(),'legacy_shared',5)
        self.assertAlmostEqual(r[0].score,.9)

    def test_fusion_does_not_mutate_components(self):
        raw=self.candidates();original=copy.deepcopy(raw)
        for name in CONFIGURATIONS:
            r=fuse(raw,name,2)
            self.assertTrue(all(0<=c.score<=1 for c in r))
            self.assertEqual([c.to_dict() for c in raw['legacy_same']],[c.to_dict() for c in original['legacy_same']])

    def test_historical_test_sources_absent_from_training_and_whole_fold_excluded(self):
        from weapon_lamp_detect.test_opening_bank import OpeningBankTests
        fixture=OpeningBankTests();fixture.setUp()
        try:
            for match,index in [('train',14),('train2',34)]:
                fixture.original['templates'].append({'weapon':'A','weapon_class':'shooter','side':'left','sample_ids':[f'{match}_{index}'],
                    'source_frames':[{'video_id':'v','match_id':match,'frame_index':index,'timestamp':float(index),'side':'left','slot_index':0}]})
            with patch('weapon_lamp_detect.hybrid_opening_matcher.source_protocol',return_value=fixture.protocol),patch('weapon_lamp_detect.opening_template_bank.source_protocol',return_value=fixture.protocol):
                bank=hybrid_bank(fixture.root,fixture.dataset,training_only=True)
            self.assertTrue(any(e.get('historical_source') for e in bank['templates']))
            self.assertTrue(all(f['match_id'] in {'train','train2'} for e in bank['templates'] for f in e['source_frames']))
            matcher=HybridOpeningMatcher(bank,excluded_matches=('train',))
            for engine in matcher.engines.values():
                self.assertTrue(all(f['match_id']=='train2' for e,_ in engine.templates for f in e['source_frames']))
        finally:
            fixture.tearDown()

    def test_full_cv_excludes_match_and_keeps_missing_weapons_in_denominator(self):
        from weapon_lamp_detect.test_opening_bank import OpeningBankTests
        from weapon_lamp_detect.evaluate_opening_cv import fold
        fixture=OpeningBankTests();fixture.setUp()
        try:
            bank=fixture.bank()
            historical={**bank,'templates':[{**e,'dataset':str(fixture.dataset)} for e in fixture.original['templates']]}
            raw,record=fold((bank,historical,fixture.dataset,'fallback','blend_soft'))
            self.assertEqual(record['missing_expected_weapons'],['B'])
            self.assertNotIn('fallback',record['template_source_matches'])
            self.assertEqual(record['inferred_template_frame_overlap'],0)
            self.assertEqual(record['evaluated_slots'],1)
            self.assertEqual(len(raw),15)
            self.assertTrue(all(not r['correct'] for r in raw))
            self.assertTrue(all(r['template_missing_in_fold'] for r in raw))
        finally:
            fixture.tearDown()


if __name__=='__main__':unittest.main()
