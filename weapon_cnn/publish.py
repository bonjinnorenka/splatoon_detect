"""Package a fixed experiment; no training or test-driven model selection."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from weapon_cnn.data import sha, write_rows
from weapon_cnn.evaluate import assessment, whole_match_accuracy
from weapon_lamp_detect.match_data import Catalog, atomic_json, read_json


def publish(evaluation, run, output, model_output):
    evaluation,run,output,model_output=map(lambda p:Path(p).resolve(),(evaluation,run,output,model_output))
    if output.exists() or model_output.exists():raise ValueError('公開済み成果物を上書きしません')
    report=read_json(evaluation/'report.json')
    if sha(run/'best.pt')!=report['checkpoint_sha256'] or sha(run/'protocol.json')!=report['protocol_sha256']:
        raise ValueError('評価とcheckpoint/protocolが一致しません')
    # Viewer needs top-k, not the full 173-way score vector in every prediction.
    rows=[]
    with (evaluation/'predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row=json.loads(line);row.pop('all_candidates',None);rows.append(row)
    primary=[r for r in rows if r['method']==report['primary_method']]
    report['primary_whole_match_accuracy']=whole_match_accuracy(primary)
    output.mkdir(parents=True);model_output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(run/'best.pt',model_output)
    report['packaged_checkpoint']=str(model_output)
    atomic_json(output/'report.json',report)
    write_rows(output/'predictions.jsonl',rows)
    for name in ('protocol.json','settings.json','history.json'):shutil.copyfile(run/name,output/name)
    shutil.copyfile(evaluation/'per_weapon.csv',output/'per_weapon.csv')
    dataset=Path(report['dataset']);catalog=Catalog(catalog_path=dataset/'catalog.json')
    (output/'assessment.md').write_text(assessment(report,catalog),encoding='utf-8')
    error_rows=[];lines=['# 先頭5frame集約で残った誤判定','',
                       '画像は使用した先頭frame。予測は5frame集約。画像を見て正解ラベルやモデルを変更していません。','',
                       'チャージャー／スコープ系3件、Orderブラシ／ワイパー系2件。高confidenceの誤りもあり、scoreだけによる自動確定には注意が必要です。',
                       '原因は未確定で、見た目の近さ・少数データ・cropでの細部欠落を切り分けるには追加の独立試合が必要です。','']
    for number,row in enumerate((r for r in primary if not r['correct']),1):
        entry=dict(row)
        for key in ('crop','context'):
            rel=f"errors/{number:02d}_{key}{Path(row[key]).suffix}"
            destination=output/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(dataset/row[key],destination);entry['exported_'+key]=rel
        error_rows.append(entry)
        expected=catalog.entries[row['expected']]['display_name'];predicted=catalog.entries[row['predicted']]['display_name']
        lines.extend([f"## {number}. {expected} → {predicted}",'',
                      f"match `{row['match_id']}` / {row['side']}{row['slot_index']} / score {row['best_score']:.4f} / margin {row['margin']:.4f}",'',
                      f"![crop]({entry['exported_crop']})",'',f"[context frame]({entry['exported_context']})",''])
    atomic_json(output/'errors.json',error_rows)
    (output/'errors.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(output/'artifacts.json',{
        'model_sha256':sha(model_output),'dataset_sha256':report['dataset_sha256'],
        'protocol_sha256':report['protocol_sha256'],
        'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'note':'Exact validation-selected checkpoint; no refit. Compact predictions retain top-5 for all evaluated rows. Error images are copies. Report dataset path must exist for browser viewer; portable error gallery needs no dataset.'})
    print('Packaged',len(rows),'predictions;',len(error_rows),'primary errors;',model_output,flush=True)


def main():
    p=argparse.ArgumentParser(description='固定CNNと実測結果をコピー。再学習・元ラベル変更はしません')
    p.add_argument('evaluation',type=Path);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-output',type=Path,required=True)
    a=p.parse_args();publish(a.evaluation,a.run,a.output,a.model_output)


if __name__=='__main__':main()
