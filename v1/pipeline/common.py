"""Shared paths and ink extraction for the Jurek-Hand pipeline."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "Niepotrzebne" / "work"
PAGES = WORK / "pages"
SEG = WORK / "seg"
LABELS = ROOT / "data" / "labels"
DIST = ROOT / "font"

MIN_AREA = 6
ROI = {"p-005": (130, 1990)}


def page_ids():
    return sorted(p.stem for p in PAGES.glob("p-*.jpg"))


def darkness(img):
    bg = cv2.dilate(img, np.ones((31, 31), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    n = np.clip(img.astype(np.float32) / np.maximum(bg, 1), 0, 1)
    return 1 - (n[..., 1] + n[..., 2]) / 2


def thresholds(d):
    gridded = ((d > 0.15) & (d < 0.35)).mean() > 0.04
    return (0.30, 0.45) if gridded else (0.18, 0.35)


def ink_components(d, pid=None):
    low, seed = thresholds(d)
    mask = (d > low).astype(np.uint8)
    if pid in ROI:
        x0, x1 = ROI[pid]
        mask[:, :x0] = 0
        mask[:, x1:] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab.ravel(), d.ravel())
    keep = [i for i in range(1, n) if peak[i] > seed and stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    return lab, stats, keep
