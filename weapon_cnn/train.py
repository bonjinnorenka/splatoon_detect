"""CPU training; select checkpoints on validation only, never evaluate test here."""
from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from weapon_cnn.data import digest, sha, training_weights, verify
from weapon_cnn.model import WeaponCNN, HEIGHT, WIDTH, augment, image_input
from weapon_lamp_detect.build_dataset import cv2_read
from weapon_lamp_detect.match_data import Catalog, atomic_json, now, read_json


def atomic_checkpoint(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def check_protocol(dataset, protocol, rows):
    if protocol['dataset_sha256']!=digest(dataset):raise ValueError('固定分割後にdatasetが変更されています')
    parts={k:set(v) for k,v in protocol['match_ids'].items()}
    if set(parts)!={'train','validation','test'} or not all(parts.values()):raise ValueError('分割が空です')
    if any(parts[a]&parts[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('match単位のデータリークがあります')
    if set.union(*parts.values())!={r['match_id'] for r in rows}:raise ValueError('分割のmatchがdatasetと一致しません')
    seen={r['weapon_label'] for r in rows if r['match_id'] in parts['train']}
    if seen!=set(protocol['classes']):raise ValueError('全candidate classが学習側にありません')
    return parts


def arrays(dataset, rows, classes):
    labels={w:i for i,w in enumerate(classes)}
    x=torch.from_numpy(np.stack([image_input(cv2_read(Path(dataset)/r['crop'])) for r in rows]))
    y=torch.tensor([labels[r['weapon_label']] for r in rows],dtype=torch.long)
    return x,y


@torch.inference_mode()
def probabilities(model, x, batch_size=128):
    model.eval()
    return torch.cat([model(x[start:start+batch_size].float()/255).softmax(1)
                      for start in range(0,len(x),batch_size)]).numpy()


def validation_metrics(probs, labels, rows):
    labels=np.asarray(labels);chosen=probs.argmax(1)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['match_id'],r['side'],r['slot_index']].append(i)
    truth,predicted=[],[]
    for indices in groups.values():
        indices=sorted(indices,key=lambda i:rows[i]['opening_frame_ordinal'])[:5]
        q=np.array([rows[i]['image_quality']['weight'] for i in indices])
        truth.append(labels[indices[0]])
        predicted.append(int(np.average(probs[indices],weights=q,axis=0).argmax()))
    truth,predicted=np.asarray(truth),np.asarray(predicted)
    correct=truth==predicted
    macro=float(np.mean([np.mean(correct[truth==label]) for label in np.unique(truth)]))
    return {'single_frame_top1':float(np.mean(chosen==labels)),
            'quality_5frame_top1':float(correct.mean()),'quality_5frame_macro':macro,
            'log_loss':float(-np.log(probs[np.arange(len(labels)),labels].clip(1e-12)).mean()),
            'match_slots':len(truth),'weapons':len(np.unique(truth))}


def run(dataset, protocol_path, output, epochs=60, batch_size=96, threads=8, seed=20261004, patience=15):
    dataset,protocol_path,output=map(lambda p:Path(p).resolve(),(dataset,protocol_path,output))
    if output.exists():raise ValueError('実験を上書きしません。新しいoutputを指定してください')
    if min(epochs,batch_size,threads,patience)<1:raise ValueError('epochs/batch/threads/patienceは正数です')
    metadata,rows=verify(dataset);protocol=read_json(protocol_path);parts=check_protocol(dataset,protocol,rows)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    train=[r for r in rows if r['match_id'] in parts['train']]
    val=[r for r in rows if r['match_id'] in parts['validation']]
    # Test pixels and predictions are not loaded by the trainer.
    x,y=arrays(dataset,train,protocol['classes']);vx,vy=arrays(dataset,val,protocol['classes'])
    weights=torch.from_numpy(training_weights(train))
    model=WeaponCNN(len(protocol['classes']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00008)
    catalog=Catalog(catalog_path=dataset/'catalog.json')
    settings={'epochs':epochs,'batch_size':batch_size,'threads':threads,'seed':seed,'patience':patience,
              'optimizer':'AdamW','learning_rate':.002,'weight_decay':.0003,'label_smoothing':.04,
              'sampling':'uniform weapon -> match -> slot; quality-weighted frame within slot',
              'selection':'validation quality-weighted first-5-frame macro accuracy, then micro, then lower log loss',
              'test_used_for_selection':False,'device':'cpu','pretrained':False}
    code={n:sha(Path(__file__).parent/n) for n in ('model.py','train.py','data.py')}
    checkpoint_base={'architecture':'weapon_residual_v1','input_size':[HEIGHT,WIDTH],
                     'classes':protocol['classes'],
                     'class_information':[{k:catalog.entries[w][k] for k in ('name','display_name','weapon_class')} for w in protocol['classes']],
                     'dataset_sha256':digest(dataset),'protocol_sha256':sha(protocol_path),
                     'algorithm_sha256':code,'settings':settings,'torch_version':str(torch.__version__),
                     'untrained_catalog_weapons':sorted(catalog.label_names-set(protocol['classes'])),
                     'singleton_training_only':protocol['singleton_training_only']}
    output.mkdir(parents=True);atomic_json(output/'protocol.json',protocol)
    atomic_json(output/'settings.json',{**settings,'parameters':sum(p.numel() for p in model.parameters()),'algorithm_sha256':code})
    history=[];best=None;best_epoch=0;started=time.perf_counter()
    print('CPU training:',len(train),'train crops /',len(val),'validation crops;',len(protocol['classes']),'classes;',threads,'threads',flush=True)
    for epoch in range(1,epochs+1):
        tick=time.perf_counter();model.train();loss_sum=0.;total=0
        sampled=torch.multinomial(weights,len(train),replacement=True)
        for indices in sampled.split(batch_size):
            batch=augment(x[indices].float()/255)
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(batch),y[indices],label_smoothing=.04)
            if not torch.isfinite(loss):raise ValueError('学習lossが非有限値です')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();loss_sum+=float(loss.detach())*len(indices);total+=len(indices)
        val_result=validation_metrics(probabilities(model,vx),vy.numpy(),val)
        rank=(val_result['quality_5frame_macro'],val_result['quality_5frame_top1'],-val_result['log_loss'])
        record={'epoch':epoch,'train_loss':loss_sum/total,'validation':val_result,
                'learning_rate':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-tick}
        if best is None or rank>best:
            best=rank;best_epoch=epoch
            atomic_checkpoint(output/'best.pt',{**checkpoint_base,'state_dict':model.state_dict(),
                                               'epoch':epoch,'validation':val_result,'created_at':now()})
        scheduler.step();history.append(record)
        atomic_json(output/'history.json',{'status':'training','best_epoch':best_epoch,'epochs':history})
        print(f"epoch {epoch:03d}: loss={record['train_loss']:.4f}, val 5frame={val_result['quality_5frame_top1']:.3%}, macro={val_result['quality_5frame_macro']:.3%}, frame={val_result['single_frame_top1']:.3%}, {record['seconds']:.1f}s; best={best_epoch}",flush=True)
        if epoch>=25 and epoch-best_epoch>=patience:break
    result={'status':'complete','best_epoch':best_epoch,'elapsed_seconds':time.perf_counter()-started,
            'test_evaluated':False,'epochs':history,'checkpoint_sha256':sha(output/'best.pt')}
    atomic_json(output/'history.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description='CPU CNNを学習。モデル選択はvalidationのみ')
    p.add_argument('dataset',type=Path);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=96)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=20261004);p.add_argument('--patience',type=int,default=15)
    a=p.parse_args();run(a.dataset,a.protocol,a.output,a.epochs,a.batch_size,a.threads,a.seed,a.patience)


if __name__=='__main__':main()
