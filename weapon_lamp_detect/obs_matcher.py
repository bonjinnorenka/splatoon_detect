"""CPU-only OBS crop baseline. Deliberately simple and separate from A."""
from __future__ import annotations

import cv2
import numpy as np

from weapon_lamp_detect.build_dataset import cv2_read
from weapon_lamp_detect.match_data import canonical_crop, read_json
from weapon_lamp_detect.weapon_lamp import MatchCandidate, _masked_multichannel_pearson


def features(image):
    image = canonical_crop(image)
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    edge = cv2.Canny(gray, 45, 145)
    return gray, lab, edge


class OBSMatcher:
    def __init__(self, directory):
        from pathlib import Path
        self.manifest = read_json(Path(directory) / "templates.json")
        self.templates = [(entry, features(cv2_read(Path(directory) / entry["path"])))
                          for entry in self.manifest["templates"]]

    def predict_crop(self, crop, top_k=5, side=None):
        gray, lab, edge = features(crop)
        mask = np.full(gray.shape, 255, np.uint8)
        best = {}
        for entry, (tpl_gray, tpl_lab, tpl_edge) in self.templates:
            # Templates retain HUD side; when that side is unavailable, allow the other.
            if side and entry["side"] != side and any(e["weapon"] == entry["weapon"] and e["side"] == side for e, _ in self.templates):
                continue
            corr = float(cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
            color = max(0.0, _masked_multichannel_pearson(tpl_lab, lab, mask))
            edge_norm = float(np.linalg.norm(tpl_edge.astype(np.float32)) * np.linalg.norm(edge.astype(np.float32)))
            edge_score = float(np.sum(tpl_edge.astype(np.float32) * edge)) / edge_norm if edge_norm else 0.0
            score = max(0.0, min(1.0, .65 * max(0, corr) + .20 * edge_score + .15 * color))
            c = MatchCandidate(entry["weapon"], entry["weapon_class"], score, score, (0, 0), (134, 108), 0,
                               {"gray_pearson": corr, "edge_cosine": edge_score, "lab_pearson": color})
            if c.weapon not in best or c.score > best[c.weapon].score:
                best[c.weapon] = c
        candidates = sorted(best.values(), key=lambda c: (-c.score, c.weapon))[:top_k]
        margin = candidates[0].score - candidates[1].score if len(candidates) > 1 else 0
        for c in candidates:
            c.confidence = min(1, .58 * c.score + 2.2 * margin)
        return candidates
