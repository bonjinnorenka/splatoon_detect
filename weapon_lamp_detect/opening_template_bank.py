"""Opening HUD templates with explicit training-match and source-frame provenance."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.build_obs_templates import dataset_digest
from weapon_lamp_detect.evaluate_opening import source_protocol
from weapon_lamp_detect.match_data import atomic_json, now
from weapon_lamp_detect.opening_matcher import CONFIGURATIONS, OpeningMatcher, features, processed_region, quality
from weapon_lamp_detect.region_matcher import OBSRegionMatcher, region_features, weapon_region


def make_bank(source, opening_dataset, kind='independent', training_only=False):
    """Use only the first two opening frames of the already chosen train matches.

    Rare within-match legacy fallbacks are retained only for evaluation, never
    used for training-only configuration validation. Test opening images are
    never templates. The candidate list and original scope remain auditable.
    """
    _, original, train, test, original_reserved, source_match_weapon = source_protocol(source, kind)
    metadata, rows = load_dataset(opening_dataset)
    grouped = defaultdict(list)
    for r in rows:
        if r['match_id'] in train and r['weapon_label'] in original['weapons'] and r['opening_frame_ordinal'] < 2:
            grouped[r['weapon_label'], r['side'], r['match_id'], r['slot_index']].append(r)
    entries = []
    for (weapon, side, match, slot), group in sorted(grouped.items()):
        group.sort(key=lambda r: r['opening_frame_ordinal'])
        # Near-total whiteout cannot be a useful source; never look at predictions.
        usable = [r for r in group if quality(sample_crop(opening_dataset,r,metadata))['white_fraction'] < .85]
        if not usable:
            continue
        entries.append({'weapon':weapon,'weapon_class':group[0]['weapon_class'],'side':side,
                        'sample_ids':[r['sample_id'] for r in usable], 'dataset':str(Path(opening_dataset).resolve()),
                        'source_frames':[{k:r[k] for k in ('video_id','match_id','timestamp','frame_index','side','slot_index')} for r in usable],
                        'method':'first two opening frames of predefined training match; median or exemplars chosen on training-only validation'})
    available = {e['weapon'] for e in entries}
    if not training_only:
        # This is explicitly same-match-other-timestamp for rare weapons, not
        # cross-match accuracy. Retain only unavailable weapon fallbacks.
        for entry in original['templates']:
            if entry['weapon'] not in available:
                entries.append({**entry,'dataset':original['dataset'],'legacy_fallback':True})
    reserved = {(f['video_id'],f['frame_index']) for e in entries for f in e['source_frames']}
    fallback_match_weapon = {(f['match_id'],e['weapon']) for e in entries for f in e['source_frames'] if e.get('legacy_fallback')}
    if any(f['match_id'] in test for e in entries if not e.get('legacy_fallback') for f in e['source_frames']):
        raise ValueError('test match used for opening template')
    return {'schema_version':1,'created_at':now(),'dataset':str(Path(opening_dataset).resolve()),
            'dataset_sha256':dataset_digest(opening_dataset),'weapons':original['weapons'],
            'templates':entries,'train_match_ids':sorted(train),'test_match_ids':sorted(test),
            'training_only':training_only,'template_frame_keys':[list(k) for k in sorted(reserved)],
            'original_reserved_frame_keys':[list(k) for k in sorted(original_reserved)],
            'original_source_match_weapon':[list(k) for k in sorted(source_match_weapon)],
            'fallback_match_weapon':[list(k) for k in sorted(fallback_match_weapon)],
            'source_policy':'first two opening frames only, assigned training matches; fallback explicitly within-match and disabled for training-only validation'}


class OpeningBankMatcher(OpeningMatcher):
    def __init__(self, bank, configuration='legacy', exemplars=False, shared=False, excluded_matches=()):
        self.configuration=configuration
        self.settings=CONFIGURATIONS.get(configuration,{'region':[14,20,120,92],'processing':'legacy'}).copy()
        self.settings['shared']=shared
        self.remove_ink=True
        self.manifest=bank
        self.templates=[]
        datasets={}
        excluded=set(excluded_matches)
        for entry in bank['templates']:
            if any(f['match_id'] in excluded for f in entry['source_frames']):
                continue
            directory=entry['dataset']
            if directory not in datasets:
                metadata,rows=load_dataset(directory)
                datasets[directory]=(metadata,{r['sample_id']:r for r in rows},{})
            metadata,by_id,cache=datasets[directory]
            images=[]
            for i in entry['sample_ids']:
                crop=sample_crop(directory,by_id[i],metadata,cache)
                images.append(weapon_region(crop,True) if configuration=='legacy' else processed_region(crop,configuration))
            composed=images if exemplars else [np.median(np.stack(images),axis=0).astype(np.uint8)]
            for image in composed:
                f=region_features(image) if configuration=='legacy' else features(image,self.settings['processing']=='local')
                self.templates.append((entry,f))
        self.side_names={(e['weapon'],e['side']) for e,_ in self.templates}

    def predict_crop(self,crop,top_k=5,side=None):
        if self.configuration=='legacy' and not self.settings['shared']:
            return OBSRegionMatcher.predict_crop(self,crop,top_k,side)
        if self.configuration=='legacy':
            # The legacy implementation's filter is controlled solely by these
            # side-name entries. Do not mirror weapon icons between teams.
            names=self.side_names
            try:
                self.side_names=set()
                return OBSRegionMatcher.predict_crop(self,crop,top_k,side)
            finally:
                self.side_names=names
        return super().predict_crop(crop,top_k,side)
