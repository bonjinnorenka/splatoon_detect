"""Synthetic split/provenance tests; do not represent measured OBS accuracy."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_added_data import freeze_added_matches, full_training, growth_manifest, paired, sha
from weapon_lamp_detect.evaluate_poc import validate_split
from weapon_lamp_detect.match_data import atomic_json, write_image


class AdditionalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="weapon-added-test-")
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_hash_split_stable_and_disjoint(self):
        matches=[f"new_{i}" for i in range(39)]
        train,test=freeze_added_matches(matches)
        self.assertEqual(len(train),13)
        self.assertEqual(len(test),26)
        self.assertTrue(set(train).isdisjoint(test))
        self.assertEqual((train,test),freeze_added_matches(list(reversed(matches))+matches))
        self.assertEqual(set(train+test),set(matches))
        with self.assertRaises(ValueError):
            freeze_added_matches(["only_one"])

    def fixtures(self):
        dataset=self.root/"dataset"
        dataset.mkdir()
        rows=[]
        data=[("old_train",1,"Splattershot","left"),("old_test",11,"Splattershot","left"),
              ("old_test",12,"Splattershot","right"),("new_train",21,"Splattershot","left"),
              ("new_train",22,"Splash-o-matic","right"),("new_test_b",31,"Splattershot","left"),
              ("new_test_b",32,"Octobrush","left"),("new_test_c",41,"Splash-o-matic","right"),
              ("new_test_c",42,"Octobrush","right")]
        random=np.random.default_rng(4)
        for match,index,weapon,side in data:
            row={"sample_id":f"{match}_{index}_{side}0","source_video":"synthetic.avi","video_id":"synthetic",
                 "match_id":match,"timestamp":float(index),"frame_index":index,"side":side,"slot_index":0,
                 "state":"alive","weapon_label":weapon,"weapon_class":"brush" if weapon=="Octobrush" else "shooter",
                 "crop":f"crops/{match}/{index}.png"}
            write_image(dataset/row["crop"],random.integers(0,256,(108,134,3),dtype=np.uint8))
            rows.append(row)
        (dataset/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        metadata={"videos":{},"matches":[]}
        atomic_json(dataset/"dataset.json",metadata)
        original_dir=self.root/"original_templates"
        original_dir.mkdir()
        write_image(original_dir/"median_000.png",np.zeros((108,134,3),dtype=np.uint8))
        split={"template_sample_ids":[rows[0]["sample_id"]],"evaluation_sample_ids":[r["sample_id"] for r in rows[1:3]],
               "train_match_ids":["old_train"],"evaluation_scope":{r["sample_id"]:"cross_match" for r in rows[1:3]},
               "within_match_template_sample_ids":[],"seed":7}
        manifest={"dataset":str(dataset),"dataset_sha256":dataset_digest(dataset),"weapons":["Splattershot","Splash-o-matic"],
                  "split":split,"templates":[{"weapon":"Splattershot","weapon_class":"shooter","side":"left", "path":"median_000.png",
                                                 "sample_ids":[rows[0]["sample_id"]]}]}
        atomic_json(original_dir/"templates.json",manifest)
        protocol={"baseline_templates":str(original_dir),"baseline_manifest_sha256":sha(original_dir/"templates.json"),
                  "baseline_weapons":manifest["weapons"],"expanded_weapons":manifest["weapons"]+["Octobrush"],
                  "new_train_match_ids":["new_train"],"new_test_match_ids":["new_test_b","new_test_c"],
                  "old_train_match_ids":["old_train"],"old_evaluation_sample_ids":split["evaluation_sample_ids"],"seed":7}
        atomic_json(self.root/"protocol.json",protocol)
        return dataset,metadata,rows,protocol,original_dir

    def test_growth_retains_original_and_never_uses_holdout_for_rare_weapon(self):
        dataset,metadata,rows,protocol,original=self.fixtures()
        directories={kind:growth_manifest(self.root,protocol,dataset,metadata,rows,kind) for kind in ("baseline","augmented","expanded")}
        manifests={k:json.loads((p/"templates.json").read_text()) for k,p in directories.items()}
        self.assertEqual(manifests["baseline"]["split"]["evaluation_sample_ids"],manifests["augmented"]["split"]["evaluation_sample_ids"])
        self.assertEqual(len(manifests["baseline"]["templates"]),1)
        self.assertEqual(len(manifests["augmented"]["templates"]),3)
        for k,m in manifests.items():
            self.assertEqual(sha(directories[k]/"median_000.png"),sha(original/"median_000.png"))
            selected=validate_split(m,rows,dataset)
            by_id={r["sample_id"]:r for r in rows}
            self.assertTrue(all(by_id[i]["match_id"] not in {"new_test_b","new_test_c"} for i in m["split"]["template_sample_ids"]))
            self.assertTrue(all(m["split"]["evaluation_scope"][i]=="cross_match" for i in selected))
        expanded=manifests["expanded"]
        self.assertIn("Octobrush",expanded["missing_obs_templates"])
        by_id={r["sample_id"]:r for r in rows}
        self.assertEqual(sum(by_id[i]["weapon_label"]=="Octobrush" for i in expanded["split"]["evaluation_sample_ids"]),2)

    def test_growth_rejects_changed_original_manifest(self):
        dataset,metadata,rows,protocol,original=self.fixtures()
        atomic_json(original/"templates.json",{"changed":True})
        with self.assertRaisesRegex(ValueError,"変更"):
            growth_manifest(self.root,protocol,dataset,metadata,rows,"baseline")

    def test_paired_comparison_requires_same_samples_and_labels(self):
        base={"sample_id":"a","source_video":"source","frame_index":10,"expected":"Splattershot",
              "state":"alive","scope":"cross_match","rect":[1,2,3,4],"match_revision":1,"correct":False}
        self.assertEqual(paired([base],[{**base,"correct":True}])["wrong_to_correct"],1)
        for changed in ({"sample_id":"b"},{"expected":"Octobrush"},{"state":"down"},{"scope":"within_match_other_timestamp"},{"match_revision":2}):
            with self.assertRaises(ValueError):
                paired([base],[{**base,**changed}])

    def test_all_added_training_never_evaluates_added_matches(self):
        protocol={"new_train_match_ids":["new_a"],"new_test_match_ids":["new_b","new_c"],
                  "old_evaluation_sample_ids":["old_test_1"],"old_train_match_ids":["old_train"]}
        atomic_json(self.root/"protocol.json",protocol)
        original_sha=sha(self.root/"protocol.json")
        with patch("weapon_lamp_detect.evaluate_added_data.run") as invoke:
            full_training(self.root,workers=2)
            invoke.assert_called_once_with(self.root/"full_additional_training",2)
        frozen=json.loads((self.root/"full_additional_training/protocol.json").read_text())
        self.assertEqual(frozen["new_train_match_ids"],["new_a","new_b","new_c"])
        self.assertEqual(frozen["new_test_match_ids"],[])
        self.assertEqual(frozen["old_evaluation_sample_ids"],["old_test_1"])
        self.assertEqual(sha(self.root/"protocol.json"),original_sha)


if __name__=="__main__":
    unittest.main()
