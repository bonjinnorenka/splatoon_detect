"""Keep original crops; handle scale/position inside matching, not ink geometry."""
import math

import cv2
import numpy as np

from weapon_lamp_detect.region_matcher import OBSRegionMatcher, REGION, SEARCH_SHIFT, finish, region_features, weapon_region
from weapon_lamp_detect.weapon_lamp import MatchCandidate, _edge_cosine, _masked_multichannel_pearson


CONFIGURATIONS={
    "shared": {"scales":[1.],"same_side_only":False},
    "scaled": {"scales":[.85,1.,1.15],"same_side_only":True},
    "scaled_shared": {"scales":[.85,1.,1.15],"same_side_only":False},
}


class FlexibleMatcher(OBSRegionMatcher):
    def __init__(self,directory,configuration="scaled",debug_dir=None):
        super().__init__(directory,True,debug_dir)
        self.configuration=configuration
        self.settings=CONFIGURATIONS[configuration]
        self.variants=[]
        for entry,(gray,lab,edge) in self.templates:
            # Reconstruct BGR from the already preprocessed median Lab image.
            image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            for scale in self.settings["scales"]:
                h,w=gray.shape
                if scale==1.:
                    features=(gray,lab,edge)  # Exact baseline features, no roundtrip.
                else:
                    features=region_features(cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR))
                self.variants.append((entry,scale,features))

    def predict_crop(self,crop,top_k=5,side=None):
        gray,lab,edge=region_features(weapon_region(crop,True))
        s=SEARCH_SHIFT+math.ceil(max(gray.shape)*(max(self.settings["scales"])-1)/2)
        target=cv2.copyMakeBorder(gray,s,s,s,s,cv2.BORDER_REFLECT_101)
        pl=cv2.copyMakeBorder(lab,s,s,s,s,cv2.BORDER_REFLECT_101)
        pe=cv2.copyMakeBorder(edge,s,s,s,s,cv2.BORDER_CONSTANT,value=0)
        best={}
        for entry,scale,(tg,tl,te) in self.variants:
            if (self.settings["same_side_only"] and side and entry["side"]!=side
                    and (entry["weapon"],side) in self.side_names):
                continue
            h,w=tg.shape
            cx=s+(gray.shape[1]-w)//2;cy=s+(gray.shape[0]-h)//2
            ox,oy=cx-SEARCH_SHIFT,cy-SEARCH_SHIFT
            window=target[oy:cy+SEARCH_SHIFT+h,ox:cx+SEARCH_SHIFT+w]
            scores=cv2.matchTemplate(window,tg,cv2.TM_CCOEFF_NORMED)
            scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
            _,corr,_,(x,y)=cv2.minMaxLoc(scores)
            x,y=x+ox,y+oy
            corr=max(0.,corr)
            mask=np.full(tg.shape,255,np.uint8)
            color=max(0.,_masked_multichannel_pearson(tl,pl[y:y+h,x:x+w],mask))
            shape=max(0.,_edge_cosine(te,pe[y:y+h,x:x+w],float(np.linalg.norm(te.astype(np.float32)))))
            score=float(np.clip(.65*corr+.15*color+.20*shape,0,1))
            candidate=MatchCandidate(entry["weapon"],entry["weapon_class"],score,score,
                                     (REGION[0]+x-s,REGION[1]+y-s),(w,h),0,
                                     {"gray_zncc":corr,"lab_pearson":color,"edge_cosine":shape,
                                      "scale":scale,"template_side_right":float(entry["side"]=="right"),
                                      "shift_x":x-cx,"shift_y":y-cy})
            if candidate.weapon not in best or score>best[candidate.weapon].score:
                best[candidate.weapon]=candidate
        return finish(best.values(),top_k)
