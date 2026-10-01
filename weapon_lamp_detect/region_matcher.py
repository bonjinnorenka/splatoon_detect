"""Fixed weapon-region / ink-removal ablations. No CNN or test-label tuning."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.match_data import canonical_crop, read_json, write_image
from weapon_lamp_detect.weapon_lamp import (MatchCandidate, WeaponIconMatcher, _edge_cosine,
                                          _masked_multichannel_pearson)
from squid_lamp_detect.squid_lamp import _dominant_color_from_crop

# Coordinates in the common 134x108 canvas. Exclude the top tip, bottom badge
# row and letterbox margins. Config is fixed before held-out evaluation.
REGION = (14, 20, 120, 92)
SIZES = (54, 70)
SEARCH_SHIFT = 4


def weapon_region(image, remove_ink=False, return_mask=False):
    image = canonical_crop(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    if remove_ink:
        color = _dominant_color_from_crop(image, "weapon_crop", 100, 70, 1.0).color
        if color.hsv is not None and color.confidence >= .55:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.int16)
            distance = np.minimum((hue-color.hsv[0]) % 180, (color.hsv[0]-hue) % 180)
            # Same hue/S/V criterion as the existing squid lamp team-color test.
            mask = ((distance <= 15) & (hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 65)).astype(np.uint8)*255
            image = image.copy()
            image[mask > 0] = 128
    x1, y1, x2, y2 = REGION
    region = image[y1:y2, x1:x2]
    return (region, mask[y1:y2, x1:x2]) if return_mask else region


def masked_zncc(image, template, mask):
    """Zero-mean masked correlation, unlike legacy brightness-biased CCORR."""
    weight = (mask > 0).astype(np.float32)
    n = float(weight.sum())
    if n < 20:
        return None
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    centered = (template-float((template*weight).sum()/n))*weight
    norm = float(np.sum(centered*centered))
    if norm < 1e-6:
        return None
    numerator = cv2.matchTemplate(image, centered, cv2.TM_CCORR)
    sums = cv2.matchTemplate(image, weight, cv2.TM_CCORR)
    squares = cv2.matchTemplate(image*image, weight, cv2.TM_CCORR)
    variance = np.maximum(0, squares-sums*sums/n)
    denominator = np.sqrt(variance*norm)
    result = np.full(numerator.shape, -1, np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-5)
    result[variance < n*4] = -1  # Flat patches cannot identify a weapon.
    return np.clip(np.nan_to_num(result, nan=-1, posinf=-1, neginf=-1), -1, 1)


def finish(candidates, top_k):
    result = sorted(candidates, key=lambda c: (-c.score, c.weapon))[:max(1, top_k)]
    margin = result[0].score-result[1].score if len(result)>1 else 0
    for c in result:
        c.confidence = min(1., .58*c.score+2.2*margin)
    return result


class OfficialRegionMatcher(WeaponIconMatcher):
    def __init__(self, templates, remove_ink=False):
        self.remove_ink = remove_ink
        # Reuse official loading, alpha masks, scales, rotations and edge builder.
        super().__init__(templates, variant_sizes=SIZES, variant_angles=(-8, 0, 8))

    def predict_crop(self, crop, top_k=5):
        crop = weapon_region(crop, self.remove_ink)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        edge = cv2.Canny(gray, 45, 145)
        best = {}
        for v in self.variants:
            h, w = v.gray.shape
            if h > gray.shape[0] or w > gray.shape[1]:
                continue
            # Raw gray on both sides avoids full-lamp histogram equalization.
            template_gray = cv2.cvtColor(v.bgr, cv2.COLOR_BGR2GRAY)
            correlations = masked_zncc(gray, template_gray, v.mask)
            if correlations is None:
                continue
            _, corr, _, (x, y) = cv2.minMaxLoc(correlations)
            if corr < 0:
                corr = 0.
            color = max(0., _masked_multichannel_pearson(v.lab, lab[y:y+h,x:x+w], v.mask))
            shape = max(0., _edge_cosine(v.edge, edge[y:y+h,x:x+w], v.edge_norm))
            score = float(np.clip(.55*corr+.25*color+.20*shape,0,1))
            c = MatchCandidate(v.name, v.weapon_class, score, score,
                               (x+REGION[0], y+REGION[1]), (w,h), v.angle,
                               {"gray_zncc": corr, "lab_pearson": color, "edge_cosine": shape})
            if c.weapon not in best or c.score>best[c.weapon].score:
                best[c.weapon]=c
        return finish(best.values(), top_k)


def region_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    edge = cv2.Canny(gray,45,145)
    return gray, lab, edge


class OBSRegionMatcher:
    def __init__(self, directory, remove_ink=False, debug_dir=None):
        directory = Path(directory)
        self.manifest = read_json(directory / "templates.json")
        metadata, rows = load_dataset(self.manifest["dataset"])
        by_id = {r["sample_id"]:r for r in rows}
        self.remove_ink = remove_ink
        self.templates = []
        cache = {}
        for entry in self.manifest["templates"]:
            # Same exact source samples as B. Remove background before median,
            # never learn from held-out crops or their labels.
            images = [weapon_region(sample_crop(self.manifest["dataset"],by_id[i],metadata,cache),remove_ink)
                      for i in entry["sample_ids"]]
            image = np.median(np.stack(images),axis=0).astype(np.uint8)
            self.templates.append((entry,region_features(image)))
            if debug_dir:
                write_image(Path(debug_dir)/entry["path"],image)
        self.side_names = {(entry["weapon"],entry["side"]) for entry,_ in self.templates}

    def predict_crop(self,crop,top_k=5,side=None):
        image = weapon_region(crop,self.remove_ink)
        gray,lab,edge = region_features(image)
        # A small, bounded translation search compensates HUD bobbing, not
        # arbitrary matching elsewhere in the lamp or surrounding scene.
        h,w = gray.shape
        s = SEARCH_SHIFT
        target = cv2.copyMakeBorder(gray,s,s,s,s,cv2.BORDER_REFLECT_101)
        padded_lab = cv2.copyMakeBorder(lab,s,s,s,s,cv2.BORDER_REFLECT_101)
        padded_edge = cv2.copyMakeBorder(edge,s,s,s,s,cv2.BORDER_CONSTANT,value=0)
        mask = np.full(gray.shape,255,np.uint8)
        best={}
        for entry,(tg,tl,te) in self.templates:
            if side and entry["side"]!=side and (entry["weapon"],side) in self.side_names:
                continue
            scores=cv2.matchTemplate(target,tg,cv2.TM_CCOEFF_NORMED)
            scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
            _,corr,_,(x,y)=cv2.minMaxLoc(scores)
            corr=max(0.,corr)
            color=max(0.,_masked_multichannel_pearson(tl,padded_lab[y:y+h,x:x+w],mask))
            shape=max(0.,_edge_cosine(te,padded_edge[y:y+h,x:x+w],float(np.linalg.norm(te.astype(np.float32)))))
            score=float(np.clip(.65*corr+.15*color+.20*shape,0,1))
            c=MatchCandidate(entry["weapon"],entry["weapon_class"],score,score,
                             (REGION[0]+x-s,REGION[1]+y-s),(w,h),0,
                             {"gray_zncc":corr,"lab_pearson":color,"edge_cosine":shape,
                              "shift_x":x-s,"shift_y":y-s})
            if c.weapon not in best or c.score>best[c.weapon].score:
                best[c.weapon]=c
        return finish(best.values(),top_k)
