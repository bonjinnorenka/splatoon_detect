"""Portable CPU crop inference; probabilities are not calibrated confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from weapon_cnn.model import image_input, load_model
from weapon_lamp_detect.build_dataset import cv2_read


class WeaponCNNMatcher:
    def __init__(self,checkpoint,threads=4):
        if threads<1:raise ValueError('threads >= 1')
        torch.set_num_threads(threads)
        self.model,self.checkpoint=load_model(checkpoint)

    @torch.inference_mode()
    def predict_crops(self,crops,top_k=5):
        if not crops:return []
        if top_k<1:raise ValueError('top_k >= 1')
        x=torch.from_numpy(np.stack([image_input(crop) for crop in crops])).float()/255
        probabilities=self.model(x).softmax(1).numpy();result=[]
        for row in probabilities:
            order=np.argsort(-row)
            margin=float(row[order[0]]-row[order[1]]) if len(order)>1 else None
            ranked=[{**self.checkpoint['class_information'][i],
                     'weapon':self.checkpoint['classes'][i],'score':float(row[i]),'confidence':float(row[i])}
                    for i in order[:top_k]]
            result.append({'top_k':ranked,'best_score':ranked[0]['score'],
                           'second_best_score':float(row[order[1]]) if len(order)>1 else None,
                           'margin':margin,'confidence_note':'uncalibrated softmax probability; not a validated Unknown threshold'})
        return result

    def predict_crop(self,crop,top_k=5):return self.predict_crops([crop],top_k)[0]


def main():
    p=argparse.ArgumentParser(description='保存CNNでHUD crop PNGをCPU推論。ラベルは変更しません')
    p.add_argument('crop',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=5);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();matcher=WeaponCNNMatcher(a.checkpoint,a.threads)
    print(json.dumps(matcher.predict_crop(cv2_read(a.crop),a.top_k),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
