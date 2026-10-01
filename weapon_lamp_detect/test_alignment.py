"""Geometry/provenance tests, not evidence of real OBS recognition accuracy."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from weapon_lamp_detect.adaptive_crop import locate_lamp, registered_rect
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.detail_matcher import DetailMatcher, detail_region
from weapon_lamp_detect.evaluate_alignment import paired, rebase_manifest
from weapon_lamp_detect.match_data import atomic_json, write_image
from weapon_lamp_detect.view_errors import ErrorServer


class AlignmentTests(unittest.TestCase):
    def test_lamp_envelope_does_not_need_weapon_name(self):
        frame=np.full((180,360,3),35,np.uint8)
        cv2.fillPoly(frame,[np.array([[180,40],[145,98],[151,117],[210,117],[215,98]])],(230,220,0))
        cv2.rectangle(frame,(164,80),(197,91),(230,230,230),-1)
        result=locate_lamp(frame,[130,38,230,138])
        self.assertTrue(result['valid'],result)
        self.assertLessEqual(result['bbox'][0],151)
        self.assertGreaterEqual(result['bbox'][2],210)

    def test_uniform_scene_ink_is_not_a_lamp(self):
        frame=np.full((180,360,3),(230,220,0),np.uint8)
        self.assertFalse(locate_lamp(frame,[130,38,230,138])['valid'])

    def test_blank_frame_is_not_a_lamp(self):
        self.assertFalse(locate_lamp(np.zeros((180,360,3),np.uint8),[130,38,230,138])['valid'])

    def test_registration_tracks_uniform_scale_and_shift(self):
        rect=[100,20,200,120]
        reference={'valid':True,'bbox':[110,30,190,110]}
        current={'valid':True,'bbox':[107,25,203,121]}
        result,quality=registered_rect(rect,reference,current,(240,400,3))
        self.assertTrue(quality['accepted'])
        self.assertAlmostEqual(quality['scale'],1.2)
        self.assertEqual(result,[95,13,215,133])

    def test_unreliable_registration_retains_original(self):
        rect=[100,20,200,120]
        for current in ({'valid':False},{'valid':True,'bbox':[50,0,240,140]},
                        {'valid':True,'bbox':[220,30,300,110]}):
            result,quality=registered_rect(rect,{'valid':True,'bbox':[110,30,190,110]},current,(240,400,3))
            self.assertEqual(result,rect)
            self.assertFalse(quality['accepted'])

    def test_detail_mask_has_fixed_shape_and_retains_weapon(self):
        image=np.full((108,134,3),(230,220,0),np.uint8)
        cv2.rectangle(image,(35,42),(95,62),(40,40,40),-1)
        region,mask=detail_region(image,True)
        self.assertEqual(region.shape,(56,106,3))
        self.assertEqual(mask.shape,(56,106))
        self.assertGreater((mask[14:34,21:81]>0).mean(),.9)

    def test_pairing_allows_only_documented_geometry_change(self):
        a={'sample_id':'a','expected':'A','weapon_label':'A','state':'alive','scope':'cross_match',
           'frame_index':12,'rect':[1,2,31,42],'correct':False}
        b={**a,'original_rect':a['rect'],'rect':[2,2,32,42],'correct':True}
        self.assertEqual(paired([a],[b]),{'improved':1})
        for change in ({'expected':'B'},{'frame_index':13},{'state':'down'},{'original_rect':[0,0,10,10]}):
            with self.assertRaises(ValueError):
                paired([a],[{**b,**change}])
        with self.assertRaises(ValueError):
            paired([a],[b,b])

    def test_rebase_preserves_template_ids_and_split(self):
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        fixture=AdditionalDataTests()
        fixture.setUp()
        try:
            dataset,metadata,rows,_,source=fixture.fixtures()
            original=json.loads((source/'templates.json').read_text())
            corrected=fixture.root/'registered'
            corrected.mkdir()
            for row in rows:
                write_image(corrected/row['crop'],np.full((108,134,3),100,np.uint8))
            (corrected/'samples.jsonl').write_text(''.join(json.dumps({**r,'original_rect':[1,2,31,42],'rect':[2,2,32,42]})+'\n' for r in rows))
            atomic_json(corrected/'dataset.json',{**metadata,'source_dataset_sha256':dataset_digest(dataset)})
            rebase_manifest(source,corrected,fixture.root/'new_templates')
            result=json.loads((fixture.root/'new_templates/templates.json').read_text())
            self.assertEqual(original['templates'],result['templates'])
            self.assertEqual(original['split'],result['split'])
            self.assertNotEqual(original['dataset_sha256'],result['dataset_sha256'])
        finally:
            fixture.tearDown()

    def test_scale_variants_fit_search_window(self):
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        fixture=AdditionalDataTests();fixture.setUp()
        try:
            _,_,_,_,source=fixture.fixtures()
            matcher=DetailMatcher(source,True)
            crop=np.random.default_rng(9).integers(0,255,(108,134,3),dtype=np.uint8)
            candidates=matcher.predict_crop(crop,5,side='left')
            self.assertEqual(len(candidates),1)
            self.assertTrue(0<=candidates[0].score<=1)
        finally:
            fixture.tearDown()

    def test_flexible_unit_scale_reproduces_baseline(self):
        from weapon_lamp_detect.flexible_matcher import FlexibleMatcher
        from weapon_lamp_detect.region_matcher import OBSRegionMatcher
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        fixture=AdditionalDataTests();fixture.setUp()
        try:
            _,_,_,_,source=fixture.fixtures()
            baseline=OBSRegionMatcher(source,True)
            flexible=FlexibleMatcher(source,'shared')
            flexible.settings={'scales':[1.],'same_side_only':True}
            image=np.random.default_rng(123).integers(0,255,(108,134,3),dtype=np.uint8)
            a=baseline.predict_crop(image,5,side='left')
            b=flexible.predict_crop(image,5,side='left')
            self.assertEqual([c.weapon for c in a],[c.weapon for c in b])
            for x,y in zip(a,b):
                self.assertAlmostEqual(x.score,y.score,places=7)
        finally:
            fixture.tearDown()

    def test_flexible_enlarged_variants_fit_window(self):
        from weapon_lamp_detect.flexible_matcher import FlexibleMatcher
        from weapon_lamp_detect.test_added_data import AdditionalDataTests
        fixture=AdditionalDataTests();fixture.setUp()
        try:
            _,_,_,_,source=fixture.fixtures()
            for config in ('scaled','scaled_shared'):
                matcher=FlexibleMatcher(source,config)
                crop=np.random.default_rng(9).integers(0,255,(108,134,3),dtype=np.uint8)
                result=matcher.predict_crop(crop,5,side='right')
                self.assertEqual(len(result),1)
                self.assertIn(result[0].components['scale'],(.85,1.,1.15))
                json.dumps(result[0].to_dict(),allow_nan=False)
        finally:
            fixture.tearDown()

    def test_viewer_original_detail_and_hud_images(self):
        with tempfile.TemporaryDirectory(prefix='weapon-alignment-viewer-') as name:
            root=Path(name);dataset=root/'dataset';dataset.mkdir()
            image=np.full((108,134,3),(220,200,50),np.uint8)
            write_image(dataset/'crop.png',image)
            write_image(dataset/'original_crop.png',image)
            write_image(dataset/'context.jpg',np.zeros((1080,1920,3),np.uint8))
            atomic_json(dataset/'dataset.json',{'videos':{}})
            row={'sample_id':'a','method':'obs_detail_foreground','crop':'crop.png','context':'context.jpg',
                 'rect':[100,20,200,120],'original_rect':[99,20,199,120],
                 'decision_frames':[{'crop':'crop.png','context':'context.jpg','rect':[100,20,200,120]}]}
            (dataset/'samples.jsonl').write_text(json.dumps(row)+'\n')
            (root/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            atomic_json(root/'report.json',{'dataset':str(dataset),'methods':{}})
            server=ErrorServer(('127.0.0.1',0),root)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                prefix=f'http://127.0.0.1:{server.server_port}/api/image?id=obs_detail_foreground:a'
                for query in ('&processed=1','&original=1','&context=1&hud=1','&decision_frame=0','&decision_frame=0&processed=1'):
                    with urlopen(prefix+query,timeout=5) as response:
                        decoded=cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
                        self.assertIsNotNone(decoded)
                        if 'processed' in query:
                            self.assertEqual(decoded.shape[:2],(56,106))
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
