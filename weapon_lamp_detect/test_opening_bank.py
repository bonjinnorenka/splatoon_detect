"""Prove opening template sources and LOMO validation exclude evaluated matches."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from weapon_lamp_detect.match_data import atomic_json, write_image
from weapon_lamp_detect.opening_template_bank import OpeningBankMatcher, make_bank
from weapon_lamp_detect.evaluate_opening_bank import training_fold


class OpeningBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='opening-bank-')
        self.root=Path(self.temp.name);self.dataset=self.root/'dataset';self.dataset.mkdir()
        rows=[]
        for match,base,weapon in [('train',10,'A'),('train2',30,'A'),('test',20,'A'),('fallback',90,'B')]:
            for ordinal in range(5):
                index=base+ordinal
                image=np.full((108,134,3),(220,200,0),np.uint8)
                cv2.rectangle(image,(35,40),(90,60),(30,30,30),-1)
                crop=f'{match}/{index}.png';write_image(self.dataset/crop,image)
                rows.append({'sample_id':f'{match}_{index}','video_id':'v','match_id':match,'frame_index':index,
                             'timestamp':float(index),'opening_frame_ordinal':ordinal,'weapon_label':weapon,
                             'weapon_class':'shooter','state':'alive','side':'left','slot_index':0,'crop':crop,
                             'context':crop,'rect':[0,0,134,108]})
        (self.dataset/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        atomic_json(self.dataset/'dataset.json',{'videos':{},'opening_window_seconds':5,'sampling_interval':1})
        self.original={'dataset':str(self.dataset),'weapons':['A','B'],
                       'templates':[{'weapon':'B','weapon_class':'shooter','side':'left','sample_ids':['fallback_90'],
                                     'source_frames':[{'video_id':'v','match_id':'fallback','frame_index':90,'timestamp':90.,'side':'left','slot_index':0}]}]}
        self.protocol=(self.root,self.original,{'train','train2'},{'test','fallback'},{('v',90)},{('fallback','B')})

    def tearDown(self):
        self.temp.cleanup()

    def bank(self,training_only=False):
        with patch('weapon_lamp_detect.opening_template_bank.source_protocol',return_value=self.protocol):
            return make_bank(self.root,self.dataset,training_only=training_only)

    def test_only_training_first_two_frames_are_opening_templates(self):
        bank=self.bank()
        entries=[e for e in bank['templates'] if not e.get('legacy_fallback')]
        self.assertEqual({i for e in entries for i in e['sample_ids']},{'train_10','train_11','train2_30','train2_31'})
        self.assertTrue(all(f['match_id'] in {'train','train2'} for e in entries for f in e['source_frames']))
        self.assertEqual(bank['fallback_match_weapon'],[['fallback','B']])
        self.assertNotIn(['v',20],bank['template_frame_keys'])

    def test_training_only_has_no_test_fallback(self):
        bank=self.bank(True)
        self.assertFalse(any(e.get('legacy_fallback') for e in bank['templates']))
        self.assertEqual({e['weapon'] for e in bank['templates']},{'A'})

    def test_lomo_matcher_removes_whole_match_not_only_timestamp(self):
        bank=self.bank(True)
        matcher=OpeningBankMatcher(bank,excluded_matches=('train',))
        self.assertTrue(matcher.templates)
        self.assertTrue(all(f['match_id']=='train2' for e,_ in matcher.templates for f in e['source_frames']))

    def test_lomo_validation_uses_excluded_match_later_opening_frames(self):
        raw=training_fold((self.bank(True),self.dataset,'train'))
        self.assertTrue(raw)
        self.assertEqual({r['sample_id'] for r in raw},{'train_13','train_14'})
        self.assertEqual({r['training_original_ordinal'] for r in raw},{3,4})
        self.assertEqual({r['scope'] for r in raw},{'training_leave_match_out'})

    def test_all_configs_return_finite_scores_and_preserve_provenance(self):
        from weapon_lamp_detect.evaluate_opening_bank import SETTINGS
        bank=self.bank()
        for settings in SETTINGS.values():
            matcher=OpeningBankMatcher(bank,**settings)
            image=cv2.imdecode(np.frombuffer((self.dataset/'test/20.png').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            candidates=matcher.predict_crop(image,5,'left')
            self.assertEqual({c.weapon for c in candidates},{'A','B'})
            json.dumps([c.to_dict() for c in candidates],allow_nan=False)
            self.assertEqual(matcher.manifest,bank)

    def test_whiteout_template_rejected_without_test_or_label_tuning(self):
        for index in (10,11):
            write_image(self.dataset/f'train/{index}.png',np.full((108,134,3),255,np.uint8))
        bank=self.bank(True)
        self.assertFalse(any(f['match_id']=='train' for e in bank['templates'] for f in e['source_frames']))
        self.assertTrue(bank['templates'])


if __name__=='__main__':unittest.main()
