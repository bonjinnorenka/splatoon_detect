"""Opening-window and decision-unit tests; synthetic accuracy is not OBS evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from weapon_lamp_detect.build_opening_dataset import opening_timer,prepare
from weapon_lamp_detect.evaluate_opening import opening_decisions, source_protocol
from weapon_lamp_detect.evaluate_poc import prediction
from weapon_lamp_detect.match_data import atomic_json, default_geometry


def row(ordinal,state='alive',scores=(.8,.2),slot=0,reserved=False):
    sample={'sample_id':f'm_{ordinal}_{slot}','match_id':'m','side':'left','slot_index':slot,
            'weapon_label':'A','weapon_class':'shooter','state':state,'opening_frame_ordinal':ordinal,
            'frame_index':ordinal*60,'timestamp':100+ordinal,'opening_timestamp':100,
            'crop':'crop.png','context':'frame.jpg','rect':[1,2,31,42],
            'reserved_template_frame':reserved}
    candidates=[] if scores is None else [{'weapon':w,'weapon_class':'shooter','score':v,'confidence':v}
                                           for w,v in zip(('A','B'),scores)]
    candidates.sort(key=lambda c:-c['score'])
    return prediction(sample,candidates,'obs_opening_baseline_foreground','cross_match')


class OpeningTests(unittest.TestCase):
    def test_only_first_global_frames_not_later_alive(self):
        rows=[row(0,'down',None),row(1,'unknown',None),row(2,'down',None),row(30,'alive')]
        result=opening_decisions(rows,3)[0]
        self.assertIsNone(result['predicted'])
        self.assertEqual(result['usable_frame_count'],0)
        self.assertEqual(result['frame_indices'],[0,60,120])
        self.assertEqual(result['decision_timestamp'],102)

    def test_auto_down_unknown_do_not_drop_clear_opening_weapon(self):
        result=opening_decisions([row(0,'down'),row(1,'unknown'),row(2,'alive')],3)[0]
        self.assertTrue(result['correct'])
        self.assertEqual(result['usable_frame_count'],3)

    def test_same_slot_denominator_for_1_3_5_frames(self):
        rows=[row(i,slot=s) for s in (0,1) for i in range(5)]
        for n in (1,3,5):
            decisions=opening_decisions(rows,n)
            self.assertEqual(len(decisions),2)
            self.assertTrue(all(r['correct'] for r in decisions))
            self.assertTrue(all(len(r['frame_indices'])==n for r in decisions))

    def test_reserved_template_frame_not_inferred_against_itself(self):
        rows=[row(0,reserved=True),row(1,scores=(.2,.8)),row(2,scores=(.2,.8))]
        result=opening_decisions(rows,3)[0]
        self.assertFalse(result['correct'])
        self.assertEqual(result['usable_frame_count'],2)
        self.assertTrue(result['insufficient_frames'])

    def test_score_mean_and_margin(self):
        result=opening_decisions([row(0,scores=(.9,.1)),row(1,scores=(.7,.2)),row(2,scores=(.8,.3))],3)[0]
        self.assertAlmostEqual(result['best_score'],.8)
        self.assertAlmostEqual(result['second_best_score'],.2)
        self.assertAlmostEqual(result['margin'],.6)

    def test_duplicate_and_changed_labels_rejected(self):
        with self.assertRaises(ValueError):
            opening_decisions([row(0),row(0)],3)
        with self.assertRaises(ValueError):
            opening_decisions([row(0),{**row(1),'expected':'B'}],3)

    def test_opening_timer_is_not_arbitrary_late_time(self):
        for value in (300,299,295,180,175):
            self.assertTrue(opening_timer(value))
        for value in (None,0,90,170,210,294):
            self.assertFalse(opening_timer(value))

    def test_dense_dataset_is_hard_bounded_and_keeps_all_slots(self):
        class Capture:
            def set(self,*args):
                pass
            def read(self):
                return True,np.zeros((270,480,3),np.uint8)
            def release(self):
                pass
        detector=SimpleNamespace(
            read_frame=lambda *a:SimpleNamespace(hud_state='match'),
            timer_ocr=SimpleNamespace(read_frame=lambda frame,timestamp,frame_index:
                                     SimpleNamespace(kind='time',seconds=300) if timestamp>=15 else None))
        slot=SimpleNamespace(state='down',special_score=0.)
        with tempfile.TemporaryDirectory(prefix='weapon-opening-test-') as name:
            root=Path(name);source=root/'source';source.mkdir()
            match={'match_id':'m','video_id':'v','confirmed':True,'rejected':False,'start_timestamp':2.,
                   'end_timestamp':40.,'hud_offset_seconds':20.,'ally_side':'right','revision':1,
                   'geometry':default_geometry(),'slots':{f'{s}{i}':{'status':'labeled','weapon':'Splattershot'} for s in ('left','right') for i in range(4)}}
            atomic_json(source/'dataset.json',{'videos':{'v':{'path':'synthetic.avi','fps':10,'width':480,'height':270}},'matches':[match],
                                             'samples_per_match':{'m':8},'max_frames_per_match':1,
                                             'effective_start_offsets':{'m':20.},'excluded':{'label_occluded':8}})
            (source/'samples.jsonl').write_text('')
            original=(source/'dataset.json').read_bytes()
            with patch('weapon_lamp_detect.build_opening_dataset.verify_sources'),patch('weapon_lamp_detect.build_opening_dataset.squid_detector',return_value=detector),patch('weapon_lamp_detect.build_opening_dataset.cv2.VideoCapture',return_value=Capture()),patch('weapon_lamp_detect.build_opening_dataset.calibrated_slot',return_value=SimpleNamespace(state='alive',special_score=0.)):
                metadata=prepare(source,root/'opening',5.,1.,1)
            rows=[json.loads(l) for l in (root/'opening/samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),40)
            self.assertEqual(metadata['samples_per_match'],{'m':40})
            self.assertEqual(metadata['max_frames_per_match'],5)
            self.assertEqual(metadata['effective_start_offsets'],{'m':14.})
            self.assertEqual(metadata['excluded'],{})
            self.assertEqual(metadata['source_dataset_config']['samples_per_match'],{'m':8})
            self.assertEqual(metadata['source_dataset_config']['max_frames_per_match'],1)
            self.assertEqual({r['opening_frame_ordinal'] for r in rows},{0,1,2,3,4})
            self.assertEqual(max(r['seconds_after_opening'] for r in rows),4)
            self.assertTrue(all(16<=r['timestamp']<=20 for r in rows))
            self.assertFalse(metadata['opening_detection']['m']['fallback'])
            self.assertEqual((source/'dataset.json').read_bytes(),original)

    def test_source_split_validation_is_reused(self):
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        from weapon_lamp_detect.evaluate_added_data import growth_manifest
        fixture=AdditionalDataTests();fixture.setUp()
        try:
            dataset,metadata,rows,protocol,_=fixture.fixtures()
            growth_manifest(fixture.root,protocol,dataset,metadata,rows,'augmented')
            _,_,train,test,reserved,_=source_protocol(fixture.root,'independent')
            self.assertTrue(train.isdisjoint(test))
            self.assertIn('new_train',train)
            self.assertIn('new_test_b',test)
            self.assertIn(('synthetic',21),reserved)
        finally:
            fixture.tearDown()


if __name__=='__main__':
    unittest.main()
