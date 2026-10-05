"""Temporary synthetic fixtures test mechanics, not real recognition accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from weapon_cnn.data import digest,freeze,sha,split_matches,training_weights,validate_rows,verify,write_rows
from weapon_cnn.evaluate import cluster_interval,whole_match_accuracy,run as evaluate
from weapon_cnn.model import HEIGHT,WIDTH,WeaponCNN,augment,image_input,load_model
from weapon_cnn.predict import WeaponCNNMatcher
from weapon_cnn.train import arrays,atomic_checkpoint,check_protocol,run as train
from weapon_lamp_detect.match_data import Catalog,atomic_json,write_image


def fixture_rows(count=12):
    result=[];weapons=['Sploosh-o-matic','Splash-o-matic','N-ZAP 85']
    for m in range(count):
        for slot in range(2):
            for frame in range(5):
                weapon=weapons[(m+slot)%3]
                result.append({'sample_id':f'm{m}_{slot}_{frame}','match_id':f'm{m}','side':'left','slot_index':slot,
                               'weapon_label':weapon,'weapon_class':'shooter','label_status':'labeled','match_revision':1,
                               'frame_identity':f'm{m}:f{frame}','opening_frame_ordinal':frame,'frame_index':m*100+frame,
                               'opening_timestamp':m*300,'seconds_after_opening':frame,'timestamp':m*300+frame,
                               'source_video':'synthetic only','video_id':'v','state':'alive','rect':[0,0,99,99],
                               'image_quality':{'weight':.2 if frame==0 else 1.},'cnn_source_kind':'live'})
    return result


class MechanicsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_whole_match_split_is_deterministic_disjoint_and_has_all_training_classes(self):
        rows=fixture_rows();a=split_matches(rows,seed=7);b=split_matches(rows,seed=7)
        self.assertEqual(a,b)
        parts={k:set(v) for k,v in a['match_ids'].items()}
        self.assertFalse(parts['train']&parts['test']);self.assertFalse(parts['validation']&parts['test'])
        self.assertEqual(set.union(*parts.values()),{r['match_id'] for r in rows})
        self.assertTrue(all(s['train']>=1 for s in a['per_weapon_matches'].values()))

    def test_singleton_match_and_all_its_frames_stay_in_training(self):
        rows=fixture_rows()
        for row in rows:
            if row['match_id']=='m0' and row['slot_index']==0:row['weapon_label']='Order Shot Replica'
        split=split_matches(rows)
        self.assertIn('m0',split['match_ids']['train'])
        self.assertEqual(split['singleton_training_only'],['Order Shot Replica'])
        self.assertNotIn('m0',split['match_ids']['test'])

    def test_duplicate_source_frame_between_matches_is_rejected(self):
        rows=fixture_rows();rows[-1]['frame_identity']=rows[0]['frame_identity']
        with self.assertRaisesRegex(ValueError,'同じ元frame'):validate_rows(rows)

    def test_late_frame_is_not_a_replacement_for_missing_opening_frame(self):
        rows=fixture_rows();rows[0]['opening_frame_ordinal']=10
        with self.assertRaisesRegex(ValueError,'開始5frame'):validate_rows(rows)

    def test_changed_ground_truth_within_slot_is_rejected(self):
        rows=fixture_rows();rows[0]['weapon_label']='Different'
        with self.assertRaisesRegex(ValueError,'武器が変わっています'):validate_rows(rows)

    def test_class_balancing_uses_matches_and_slots_not_raw_frame_frequency(self):
        rows=fixture_rows();weights=training_weights(rows)
        self.assertAlmostEqual(weights.sum(),1.)
        for weapon in {r['weapon_label'] for r in rows}:
            self.assertAlmostEqual(sum(w for w,r in zip(weights,rows) if r['weapon_label']==weapon),1/3)
        self.assertGreater(weights[1],weights[0])

    def test_roi_shapes_and_cnn_forward_are_cpu_finite(self):
        crop=np.full((99,99,3),128,np.uint8)
        pixels=image_input(crop);self.assertEqual(pixels.shape,(3,HEIGHT,WIDTH));self.assertEqual(pixels.dtype,np.uint8)
        model=WeaponCNN(3).eval();x=torch.from_numpy(np.stack([pixels,pixels])).float()/255
        outputs=model(x);self.assertEqual(outputs.shape,(2,3));self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(outputs.device.type,'cpu')

    def test_train_augmentation_is_bounded(self):
        x=augment(torch.rand(4,3,HEIGHT,WIDTH))
        self.assertTrue(torch.isfinite(x).all());self.assertGreaterEqual(float(x.min()),0);self.assertLessEqual(float(x.max()),1)

    def test_bootstrap_clusters_matches_not_frames(self):
        rows=[{'match_id':'a','correct':True}]*5+[{'match_id':'b','correct':False}]*5
        ci=cluster_interval(rows,draws=200)
        self.assertEqual(ci['matches'],2);self.assertEqual(ci['unit'],'whole match bootstrap')

    def test_eight_slot_accuracy_is_not_slot_accuracy(self):
        rows=[{'match_id':m,'side':side,'slot_index':slot,'correct':not(m=='b' and side=='left' and slot==0)}
              for m in ('a','b') for side in ('left','right') for slot in range(4)]
        rows.append({'match_id':'incomplete','side':'left','slot_index':0,'correct':True})
        result=whole_match_accuracy(rows)
        self.assertEqual(result['accuracy'],.5);self.assertEqual(result['support'],2)
        self.assertEqual(result['excluded_incomplete_matches'],1)
        with self.assertRaisesRegex(ValueError,'重複'):whole_match_accuracy(rows+[rows[0]])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory(prefix='weapon-cnn-日本語 空白-')
        self.root=Path(self.temp.name);self.dataset=self.root/'dataset';self.dataset.mkdir()
        self.rows=fixture_rows();catalog=Catalog();images={}
        (self.dataset/'catalog.json').write_bytes(catalog.catalog_path.read_bytes())
        for row in self.rows:
            rel=f"crops/{row['sample_id']}.png";row.update(crop=rel,context=rel)
            image=np.zeros((99,99,3),np.uint8)
            cv2.rectangle(image,(20,30),(70,60),(100,200,150),-1)
            write_image(self.dataset/rel,image);images[rel]=sha(self.dataset/rel)
        write_rows(self.dataset/'samples.jsonl',self.rows)
        atomic_json(self.dataset/'dataset.json',{'samples':len(self.rows),'matches':[],'videos':{},'images':images,
                                               'catalog_sha256':sha(self.dataset/'catalog.json')})
        self.protocol_path=self.root/'protocol.json';self.protocol=freeze(self.dataset,self.protocol_path,seed=9)

    def tearDown(self):self.temp.cleanup()

    def test_copied_hashes_and_catalog_are_verified(self):
        verify(self.dataset)
        (self.dataset/self.rows[0]['crop']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'path/hash'):verify(self.dataset)

    def test_protocol_detects_group_leakage_and_dataset_edits(self):
        p=copy.deepcopy(self.protocol);p['match_ids']['test'].append(p['match_ids']['train'][0])
        with self.assertRaisesRegex(ValueError,'データリーク'):check_protocol(self.dataset,p,self.rows)
        write_rows(self.dataset/'samples.jsonl',self.rows[:-1])
        with self.assertRaisesRegex(ValueError,'変更されています'):check_protocol(self.dataset,self.protocol,self.rows)

    def test_checkpoint_predict_and_validation_selection_never_load_test_pixels(self):
        seen=[]
        def checked_arrays(dataset,rows,classes):
            seen.extend(r['match_id'] for r in rows)
            return arrays(dataset,rows,classes)
        output=self.root/'run'
        with patch('weapon_cnn.train.arrays',side_effect=checked_arrays):
            history=train(self.dataset,self.protocol_path,output,epochs=1,batch_size=16,threads=2)
        self.assertFalse(set(seen)&set(self.protocol['match_ids']['test']))
        self.assertFalse(history['test_evaluated'])
        model,checkpoint=load_model(output/'best.pt')
        self.assertFalse(model.training)
        matcher=WeaponCNNMatcher(output/'best.pt',threads=2)
        result=matcher.predict_crop(np.full((99,99,3),100,np.uint8),top_k=3)
        self.assertEqual(len(result['top_k']),3);self.assertGreaterEqual(result['margin'],0)
        self.assertAlmostEqual(sum(c['score'] for c in result['top_k']),1.,places=5)
        before=model.features[1].running_mean.clone()
        with torch.inference_mode():model(torch.rand(2,3,HEIGHT,WIDTH))
        self.assertTrue(torch.equal(before,model.features[1].running_mean))
        report=evaluate(self.dataset,output/'best.pt',self.protocol_path,self.root/'evaluation',threads=2)
        self.assertFalse(report['test_used_for_selection'])
        self.assertEqual(report['methods'][report['primary_method']]['metrics']['support'],len(self.protocol['match_ids']['test'])*2)
        self.assertTrue((self.root/'evaluation/predictions.jsonl').exists())


if __name__=='__main__':unittest.main()
