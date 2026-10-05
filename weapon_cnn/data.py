"""Reuse live crop export, OBS crops and validation; freeze whole-match splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from live_weapon_collect.export_dataset import export_dataset
from weapon_lamp_detect.build_dataset import cv2_read, load_dataset
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json, write_image
from weapon_lamp_detect.opening_matcher import quality


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def prepare(snapshot, output):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('既存datasetを上書きしません。新しいoutputを指定してください')
    catalog = Catalog(catalog_path=snapshot/'wepons.json')
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    rows, matches, images, videos, excluded = [], [], {}, {}, {}
    # The established exporter verifies every source PNG, human status and GT.
    # Its large lossless context copies are temporary, not a second permanent dataset.
    with tempfile.TemporaryDirectory(prefix='live-crop-export-', dir=output) as tmp:
        live = Path(tmp)/'live'
        export_dataset(snapshot/'live_session', live, catalog, states=('alive','down','unknown'))
        for kind, source in [('live',live), ('obs',snapshot/'obs_opening_crops')]:
            metadata, source_rows = load_dataset(source)
            matches.extend({**m,'cnn_source_kind':kind} for m in metadata['matches']
                           if any(r['match_id']==m['match_id'] for r in source_rows))
            videos.update(metadata.get('videos',{}))
            excluded[kind] = metadata.get('excluded',{})
            seen_contexts = set()
            for original in source_rows:
                if original['label_status'] != 'labeled':
                    raise ValueError('非labeled sampleを学習に混ぜません')
                row = dict(original)
                row['weapon_label'] = catalog.resolve_label(row['weapon_label'])
                row['weapon_class'] = catalog.entries[row['weapon_label']]['weapon_class']
                row['cnn_source_kind'] = kind
                crop = f"crops/{kind}/{row['match_id']}/{Path(row['crop']).name}"
                destination = output/crop
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source/row['crop'], destination)
                images[crop] = sha(destination)
                row['image_quality'] = quality(cv2_read(destination))
                context = f"frames/{kind}/{row['match_id']}/{row['frame_index']:09d}.jpg"
                if context not in seen_contexts:
                    frame = cv2_read(source/row['context'])
                    write_image(output/context, frame)
                    images[context] = sha(output/context)
                    seen_contexts.add(context)
                row['crop'], row['context'] = crop, context
                # Provenance only, never a network/feature input. Avoid temp paths.
                row['source_dataset'] = f'{kind}_snapshot'
                row['frame_identity'] = ('live:'+row['source_image_sha256'] if kind=='live'
                                         else f"obs:{row['video_id']}:{row['frame_index']}")
                rows.append(row)
            print('prepared',kind,len(source_rows),'crops',flush=True)
    rows.sort(key=lambda r:(r['match_id'],r['opening_frame_ordinal'],r['side'],r['slot_index']))
    validate_rows(rows)
    shutil.copyfile(snapshot/'wepons.json',output/'catalog.json')
    write_rows(output/'samples.jsonl', rows)
    atomic_json(output/'dataset.json', {
        'schema_version':1,'created_at':now(),'source_kind':'cnn_prepared_opening_crops',
        'snapshot':str(snapshot),'matches':matches,'videos':videos,'samples':len(rows),
        'samples_per_match':dict(Counter(r['match_id'] for r in rows)),
        'images':images,'excluded':excluded,'catalog_sha256':sha(output/'catalog.json'),
        'states':['alive','down','unknown'],'source_note':'Human-confirmed opening slots only. Automatic states never gate samples. Crop/context paths are relative; original videos are not needed.',
    })
    return {'samples':len(rows),'matches':len({r['match_id'] for r in rows}),'weapons':len({r['weapon_label'] for r in rows})}


def validate_rows(rows):
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('空datasetまたはsample ID重複')
    owners, slots = {}, defaultdict(list)
    for r in rows:
        owner = owners.setdefault(r['frame_identity'],r['match_id'])
        if owner != r['match_id']:
            raise ValueError('同じ元frameが別matchにあります。分割前に人間が重複を確認してください')
        if r['state'] not in {'alive','down','unknown'} or r['label_status']!='labeled':
            raise ValueError('state / label statusが不正です')
        slots[r['match_id'],r['side'],r['slot_index']].append(r)
    for group in slots.values():
        if len({r['weapon_label'] for r in group})!=1:
            raise ValueError('同一match slotの武器が変わっています')
        if sorted(r['opening_frame_ordinal'] for r in group)!=list(range(5)):
            raise ValueError('開始5frameが揃っていません（途中frameで補充しません）')


def digest(dataset):
    dataset = Path(dataset)
    return hashlib.sha256((dataset/'samples.jsonl').read_bytes()+(dataset/'dataset.json').read_bytes()).hexdigest()


def verify(dataset):
    dataset = Path(dataset).resolve()
    metadata,rows = load_dataset(dataset)
    validate_rows(rows)
    catalog = Catalog(catalog_path=dataset/'catalog.json')
    if sha(dataset/'catalog.json')!=metadata['catalog_sha256']:
        raise ValueError('catalogが変更されています')
    for path,expected in metadata['images'].items():
        resolved = (dataset/path).resolve()
        if not resolved.is_relative_to(dataset) or sha(resolved)!=expected:
            raise ValueError(f'保存画像のpath/hashが不正です: {path}')
    for r in rows:
        catalog.resolve_label(r['weapon_label'])
        if r['crop'] not in metadata['images'] or r['context'] not in metadata['images']:
            raise ValueError('画像manifestがありません')
    return metadata,rows


def split_matches(rows, seed=20261004, test_fraction=.25, validation_fraction=.10, attempts=32):
    """Class-aware whole-match split, selected using labels, never model scores."""
    validate_rows(rows)
    if not 0<test_fraction<.5 or not 0<validation_fraction<.3 or attempts<1:
        raise ValueError('分割比率 / attemptsが不正です')
    matches=sorted({r['match_id'] for r in rows});weapons=sorted({r['weapon_label'] for r in rows})
    if len(matches)<5:
        raise ValueError('train/validation/testを分離するには少なくとも5試合が必要です')
    ids={m:i for i,m in enumerate(matches)};labels={w:i for i,w in enumerate(weapons)}
    presence=np.zeros((len(matches),len(weapons)),np.int32)
    for r in rows:presence[ids[r['match_id']],labels[r['weapon_label']]]=1
    totals=presence.sum(0);weights=1/np.sqrt(totals);eligible=totals>=2
    rng=np.random.default_rng(seed);best=None
    for _ in range(attempts):
        remaining=totals.copy();chosen=[];covered=np.zeros(len(weapons),bool)
        available=np.ones(len(matches),bool)
        for step in range(int(np.ceil(len(matches)*.35))):
            valid=available & np.all(remaining-presence>=1,axis=1)
            gains=(presence*(weights*(~covered))).sum(1)
            if not valid.any() or (step>=np.ceil(len(matches)*test_fraction) and np.max(gains[valid])<1e-9):
                break
            # Prefer at least 2 training matches where possible, without dropping
            # rare-but-testable classes solely to make the reported accuracy high.
            scarcity=((remaining==2)&(totals>=3))
            score=10*gains-.2*(presence*scarcity).sum(1)+rng.uniform(0,.15,len(matches))
            score[~valid]=-np.inf;i=int(score.argmax())
            if step>=np.ceil(len(matches)*test_fraction) and gains[i]<1e-9:break
            chosen.append(i);available[i]=False;remaining-=presence[i];covered|=presence[i]>0
        rank=(int(np.sum(covered&eligible)),-sum((remaining<2)&(totals>=3)),-abs(len(chosen)-round(len(matches)*test_fraction)))
        if best is None or rank>best[0]:best=(rank,chosen)
    test=set(best[1]);remaining=totals-presence[list(test)].sum(0)
    available=np.array([i not in test for i in range(len(matches))]);val=[];covered=np.zeros(len(weapons),bool)
    for _ in range(max(1,round(len(matches)*validation_fraction))):
        desired=np.where(totals>=3,2,1)
        # Existing shortfalls after test selection cannot be made worse by val.
        valid=available & np.all((remaining-presence>=desired)|(presence==0),axis=1)
        if not valid.any():break
        score=(presence*(weights*(~covered))).sum(1)+rng.uniform(0,.05,len(matches))
        score[~valid]=-np.inf;i=int(score.argmax());val.append(i);available[i]=False
        remaining-=presence[i];covered|=presence[i]>0
    if not val or not test:raise ValueError('安全なvalidation/test分割を作れません')
    parts={'train':[m for i,m in enumerate(matches) if available[i]],
           'validation':[matches[i] for i in sorted(val)],'test':[matches[i] for i in sorted(test)]}
    support={w:{kind:sum(presence[ids[m],labels[w]] for m in ms).item() if ms else 0 for kind,ms in parts.items()}
             for w in weapons}
    for w in weapons:
        if support[w]['train']<1:raise ValueError('未学習クラスを主評価に混ぜません')
    return {'seed':seed,'classes':weapons,'match_ids':parts,'per_weapon_matches':support,
            'singleton_training_only':[w for w,n in zip(weapons,totals) if n==1],
            'test_unrepresented':[w for w in weapons if support[w]['test']==0],
            'policy':'Whole match split, class-aware greedy coverage with fixed seed; every class remains in train. Singleton matches stay in training. Validation preserves 2 train matches when possible. No frame/slot random split.',
            'scope':'cross_match, not cross_video/cross_session; fixed historical recordings, not untouched newly collected data.'}


def freeze(dataset, path, seed=20261004):
    path=Path(path)
    if path.exists():raise ValueError('既存protocolを上書きしません')
    _,rows=verify(dataset);protocol=split_matches(rows,seed)
    protocol.update(dataset_sha256=digest(dataset),created_at=now())
    atomic_json(path,protocol)
    return protocol


def training_weights(rows):
    """Uniform class -> match -> slot; quality distributes only within a slot."""
    by_class=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,r in enumerate(rows):by_class[r['weapon_label']][r['match_id']][r['side'],r['slot_index']].append(i)
    weights=np.zeros(len(rows),np.float64)
    for matches in by_class.values():
        for slots in matches.values():
            for indices in slots.values():
                q=np.array([max(.02,rows[i]['image_quality']['weight']) for i in indices])
                weights[indices]=q/q.sum()/len(matches)/len(slots)
    return weights/weights.sum()


def main():
    p=argparse.ArgumentParser(description='既存収集画像からCNN用datasetと試合単位固定分割を作成')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('snapshot',type=Path);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('split');a.add_argument('dataset',type=Path);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,default=20261004)
    args=p.parse_args()
    if args.command=='prepare':print(prepare(args.snapshot,args.output))
    else:
        value=freeze(args.dataset,args.output,args.seed)
        print({k:len(v) for k,v in value['match_ids'].items()},'classes',len(value['classes']),'test absent',value['test_unrepresented'])


if __name__=='__main__':main()
