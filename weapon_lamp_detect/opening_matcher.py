"""Opening-only, CPU template ablations which preserve weapon-coloured edges.

All templates come from the supplied, already separated source manifest. No
weapon-specific thresholds or evaluation labels enter preprocessing/quality.
"""
from pathlib import Path

import cv2
import numpy as np

from squid_lamp_detect.squid_lamp import _dominant_color_from_crop
from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.match_data import canonical_crop, read_json
from weapon_lamp_detect.region_matcher import finish, weapon_region
from weapon_lamp_detect.weapon_lamp import MatchCandidate, _masked_multichannel_pearson


CONFIGURATIONS = {
    "raw": {"region": [14, 20, 120, 92], "processing": "raw"},
    "raw_core": {"region": [14, 28, 120, 84], "processing": "raw"},
    "smooth_core": {"region": [14, 28, 120, 84], "processing": "smooth"},
    "local_core": {"region": [14, 28, 120, 84], "processing": "local"},
    "local_exemplars": {"region": [14, 28, 120, 84], "processing": "local", "exemplars": True},
    "local_shared": {"region": [14, 28, 120, 84], "processing": "local", "shared": True},
}


def quality(crop):
    """Label-blind washout measurements, not calibrated match confidence."""
    image = canonical_crop(crop)[28:84, 14:120]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # GO flashes are nearly achromatic and bright. Do not penalize vivid ink.
    clipped = float(np.mean((hsv[:, :, 2] >= 240) & (hsv[:, :, 1] <= 32)))
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    # A continuous weight avoids a test-set-derived hard rejection threshold.
    weight = float(np.clip((1 - clipped) ** 2 * min(1., contrast / 80.), .02, 1.))
    return {"white_fraction": clipped, "contrast": contrast, "weight": weight}


def processed_region(crop, configuration):
    settings = CONFIGURATIONS[configuration]
    image = canonical_crop(crop)
    if settings["processing"] == "smooth":
        color = _dominant_color_from_crop(image, "weapon_crop", 100, 70, 1.).color
        if color.hsv is not None and color.confidence >= .55:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.int16)
            distance = np.minimum((hue - color.hsv[0]) % 180, (color.hsv[0] - hue) % 180)
            ink = ((distance <= 8) & (hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 65)).astype(np.uint8)
            # Keep boundaries and thin coloured gun parts, unlike blanket HSV.
            ink = cv2.erode(ink, np.ones((3, 3), np.uint8))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            image = image.copy()
            image[(ink > 0) & (cv2.magnitude(gx, gy) < 40)] = 128
    x1, y1, x2, y2 = settings["region"]
    return image[y1:y2, x1:x2]


def features(image, local=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if local:
        gray = gray - cv2.GaussianBlur(gray, (0, 0), 3.)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    return gray, cv2.cvtColor(image, cv2.COLOR_BGR2LAB), gradient


class OpeningMatcher:
    def __init__(self, directory, configuration="local_core"):
        self.configuration = configuration
        self.settings = CONFIGURATIONS[configuration]
        self.manifest = read_json(Path(directory) / "templates.json")
        metadata, rows = load_dataset(self.manifest["dataset"])
        by_id = {r["sample_id"]: r for r in rows}
        self.templates = []
        cache = {}
        for entry in self.manifest["templates"]:
            images = [processed_region(sample_crop(self.manifest["dataset"], by_id[i], metadata, cache), configuration)
                      for i in entry["sample_ids"]]
            composed = images if self.settings.get("exemplars") else [np.median(np.stack(images), axis=0).astype(np.uint8)]
            for image in composed:
                self.templates.append((entry, features(image, self.settings["processing"] == "local")))
        self.side_names = {(entry["weapon"], entry["side"]) for entry, _ in self.templates}

    def predict_crop(self, crop, top_k=5, side=None):
        image = processed_region(crop, self.configuration)
        gray, lab, gradient = features(image, self.settings["processing"] == "local")
        s = 4
        padded = [cv2.copyMakeBorder(f, s, s, s, s, cv2.BORDER_REFLECT_101) for f in (gray, lab, gradient)]
        mask = np.full(gray.shape, 255, np.uint8)
        h, w = gray.shape
        best = {}
        for entry, (tg, tl, te) in self.templates:
            if not self.settings.get("shared") and side and entry["side"] != side and (entry["weapon"], side) in self.side_names:
                continue
            corr = cv2.matchTemplate(padded[0], tg, cv2.TM_CCOEFF_NORMED)
            corr = np.nan_to_num(corr, nan=-1, posinf=-1, neginf=-1)
            _, g, _, (x, y) = cv2.minMaxLoc(corr)
            g = max(0., g) if float(tg.std()) > .1 else 0.
            color = max(0., _masked_multichannel_pearson(tl, padded[1][y:y+h, x:x+w], mask))
            patch = padded[2][y:y+h, x:x+w]
            denom = float(np.linalg.norm(te) * np.linalg.norm(patch))
            edge = float(np.sum(te * patch) / denom) if denom > 1e-6 else 0.
            score = float(np.clip(.65*g + .15*color + .20*edge, 0, 1))
            candidate = MatchCandidate(entry["weapon"], entry["weapon_class"], score, score,
                                       (self.settings["region"][0]+x-s, self.settings["region"][1]+y-s), (w, h), 0,
                                       {"gray_zncc": g, "lab_pearson": color, "gradient_cosine": edge,
                                        "shift_x": x-s, "shift_y": y-s})
            if candidate.weapon not in best or score > best[candidate.weapon].score:
                best[candidate.weapon] = candidate
        return finish(best.values(), top_k)
