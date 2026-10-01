"""Fuse complementary CPU template scores, with a soft rather than hard side prior."""
from copy import deepcopy

from weapon_lamp_detect.evaluate_opening import source_protocol
from weapon_lamp_detect.opening_template_bank import OpeningBankMatcher, make_bank
from weapon_lamp_detect.region_matcher import finish


CONFIGURATIONS={
    'legacy_hard':{'local_weight':0.,'opposite_penalty':None},
    'legacy_shared':{'local_weight':0.,'opposite_penalty':0.},
    'legacy_soft':{'local_weight':0.,'opposite_penalty':.03},
    'blend_shared':{'local_weight':.25,'opposite_penalty':0.},
    'blend_soft':{'local_weight':.25,'opposite_penalty':.03},
    'blend_half':{'local_weight':.5,'opposite_penalty':.03},
    'local_soft':{'local_weight':1.,'opposite_penalty':.03},
}


def hybrid_bank(source,dataset,kind='independent',training_only=False):
    bank=make_bank(source,dataset,kind,training_only)
    _,original,train,*_=source_protocol(source,kind)
    bank=deepcopy(bank)
    for entry in original['templates']:
        if training_only and any(f['match_id'] not in train for f in entry['source_frames']):
            continue
        # Opening fallback may already contain this exact historical source.
        if any(e['sample_ids']==entry['sample_ids'] and e['dataset']==original['dataset'] for e in bank['templates']):
            continue
        bank['templates'].append({**entry,'dataset':original['dataset'],'historical_source':True})
    reserved={(f['video_id'],f['frame_index']) for e in bank['templates'] for f in e['source_frames']}
    bank['template_frame_keys']=[list(k) for k in sorted(reserved)]
    bank['source_policy']+='; combined with original OBS source entries, excluding test sources during tuning'
    return bank


def fuse(component_candidates,configuration,top_k):
    config=CONFIGURATIONS[configuration]
    score_maps={name:{c.weapon:c for c in candidates} for name,candidates in component_candidates.items()}
    names=set.intersection(*(set(candidates) for candidates in score_maps.values()))
    result=[]
    for name in names:
        scores={key:score_maps[key][name].score for key in score_maps}
        def prior(kind):
            same,shared=scores[kind+'_same'],scores[kind+'_shared']
            return same if config['opposite_penalty'] is None else max(same,shared-config['opposite_penalty'])
        weight=config['local_weight']
        score=(1-weight)*prior('legacy')+weight*prior('local')
        candidate=deepcopy(score_maps['legacy_same'][name])
        candidate.score=float(score);candidate.confidence=float(score)
        candidate.components={**scores,'local_weight':weight,'opposite_penalty':config['opposite_penalty'] if config['opposite_penalty'] is not None else -1.}
        result.append(candidate)
    return finish(result,top_k)


class HybridOpeningMatcher:
    def __init__(self,bank,configuration='blend_soft',excluded_matches=()):
        self.configuration=configuration;self.manifest=bank
        self.engines={
            'legacy_same':OpeningBankMatcher(bank,'legacy',excluded_matches=excluded_matches),
            'legacy_shared':OpeningBankMatcher(bank,'legacy',shared=True,excluded_matches=excluded_matches),
            'local_same':OpeningBankMatcher(bank,'local_core',exemplars=True,excluded_matches=excluded_matches),
            'local_shared':OpeningBankMatcher(bank,'local_core',exemplars=True,shared=True,excluded_matches=excluded_matches),
        }

    def components(self,crop,side):
        return {name:engine.predict_crop(crop,len(self.manifest['weapons']),side) for name,engine in self.engines.items()}

    def predict_crop(self,crop,top_k=5,side=None):
        return fuse(self.components(crop,side),self.configuration,top_k)
