"""CPU-only detail matching. Configuration is selected on training matches only."""
from pathlib import Path
import math

import cv2
import numpy as np

from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.match_data import canonical_crop, read_json, write_image
from weapon_lamp_detect.region_matcher import finish, masked_zncc
from weapon_lamp_detect.weapon_lamp import MatchCandidate, _masked_multichannel_pearson
from squid_lamp_detect.squid_lamp import _dominant_color_from_crop


DETAIL_REGION = (14, 28, 120, 84)
DETAIL_SCALES = (.90, 1., 1.10)
DETAIL_SHIFT = 6
DETAIL_CONFIG = {"region":list(DETAIL_REGION),"scales":list(DETAIL_SCALES),
                 "shift":DETAIL_SHIFT,"hue_distance":8,"ink_erosion":3,
                 "score_weights":{"gray":.65,"lab":.15,"edge":.20},
                 "mask_note":"eroded team-color interior excluded from template correlation; boundaries retained"}


def detail_region(crop, return_mask=False):
    """Narrower hue removal preserves more colored weapon/boundary pixels."""
    image=canonical_crop(crop)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    color=_dominant_color_from_crop(image,"weapon_crop",100,70,1.).color
    ink=np.zeros(image.shape[:2],np.uint8)
    if color.hsv is not None and color.confidence>=.55:
        delta=np.abs(hsv[:,:,0].astype(np.int16)-color.hsv[0])
        ink=((np.minimum(delta,180-delta)<=8)&(hsv[:,:,1]>75)&(hsv[:,:,2]>65)).astype(np.uint8)*255
        ink=cv2.erode(ink,np.ones((3,3),np.uint8))
    valid=((ink==0)&np.any(image!=0,axis=2)).astype(np.uint8)*255
    neutral=image.copy()
    neutral[ink>0]=128
    x1,y1,x2,y2=DETAIL_REGION
    region=neutral[y1:y2,x1:x2]
    return (region,valid[y1:y2,x1:x2]) if return_mask else region


class DetailMatcher:
    def __init__(self,directory,exemplars=False,debug_dir=None):
        self.manifest=read_json(Path(directory)/"templates.json")
        metadata,rows=load_dataset(self.manifest["dataset"])
        by_id={r["sample_id"]:r for r in rows}
        self.templates=[]
        self.exemplars=exemplars
        cache={}
        for entry in self.manifest["templates"]:
            pairs=[detail_region(sample_crop(self.manifest["dataset"],by_id[i],metadata,cache),True)
                   for i in entry["sample_ids"]]
            if not exemplars:
                pairs=[(np.median(np.stack([a for a,_ in pairs]),axis=0).astype(np.uint8),
                        (np.mean(np.stack([m>0 for _,m in pairs]),axis=0)>=.5).astype(np.uint8)*255)]
            for source_index,(image,mask) in enumerate(pairs):
                if debug_dir:
                    write_image(Path(debug_dir)/f"{Path(entry['path']).stem}_{source_index}.png",image)
                for scale in DETAIL_SCALES:
                    h,w=image.shape[:2]
                    size=(round(w*scale),round(h*scale))
                    bgr=cv2.resize(image,size,interpolation=cv2.INTER_LINEAR)
                    m=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
                    if (m>0).sum()<40:
                        m=np.full(m.shape,255,np.uint8)
                    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                    edge=cv2.Canny(gray,45,145)
                    self.templates.append((entry,source_index,scale,gray,cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB),edge,m))
        self.side_names={(e["weapon"],e["side"]) for e,*_ in self.templates}

    def predict_crop(self,crop,top_k=5,side=None):
        image=detail_region(crop)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        edge=cv2.Canny(gray,45,145)
        # Padding permits enlarged scale variants without sampling neighboring HUD.
        s=DETAIL_SHIFT+math.ceil(max(gray.shape)*(max(DETAIL_SCALES)-1)/2)
        target=cv2.copyMakeBorder(gray,s,s,s,s,cv2.BORDER_REFLECT_101)
        pl=cv2.copyMakeBorder(lab,s,s,s,s,cv2.BORDER_REFLECT_101)
        pe=cv2.copyMakeBorder(edge,s,s,s,s,cv2.BORDER_CONSTANT,value=0)
        best={}
        for entry,source_index,scale,tg,tl,te,mask in self.templates:
            if side and entry["side"]!=side and (entry["weapon"],side) in self.side_names:
                continue
            h,w=tg.shape
            cx=s+(gray.shape[1]-w)//2
            cy=s+(gray.shape[0]-h)//2
            ox,oy=cx-DETAIL_SHIFT,cy-DETAIL_SHIFT
            window=target[oy:cy+DETAIL_SHIFT+h,ox:cx+DETAIL_SHIFT+w]
            corrmap=masked_zncc(window,tg,mask)
            if corrmap is None:
                continue
            _,corr,_,(x,y)=cv2.minMaxLoc(corrmap)
            x,y=x+ox,y+oy
            corr=max(0.,corr)
            color=max(0.,_masked_multichannel_pearson(tl,pl[y:y+h,x:x+w],mask))
            active=(mask>0)
            a,b=te[active].astype(np.float32),pe[y:y+h,x:x+w][active].astype(np.float32)
            norm=float(np.linalg.norm(a)*np.linalg.norm(b))
            shape=float(a@b/norm) if norm>1e-6 else 0.
            score=float(np.clip(.65*corr+.15*color+.20*shape,0,1))
            candidate=MatchCandidate(entry["weapon"],entry["weapon_class"],score,score,
                                     (DETAIL_REGION[0]+x-s,DETAIL_REGION[1]+y-s),(w,h),0,
                                     {"gray_zncc":corr,"lab_pearson":color,"edge_cosine":shape,
                                      "scale":scale,"source_index":source_index,
                                      "mask_fraction":float(active.mean())})
            if candidate.weapon not in best or score>best[candidate.weapon].score:
                best[candidate.weapon]=candidate
        return finish(best.values(),top_k)
