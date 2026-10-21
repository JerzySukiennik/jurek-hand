"""Split each page into ink components and render numbered bands for labeling."""
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from common import PAGES, SEG, page_ids, darkness, ink_components

COLORS = [(200, 30, 30), (20, 120, 20), (180, 0, 160), (0, 110, 200), (200, 110, 0), (90, 60, 30)]
BAND_MIN, BAND_MAX, CTX = 50, 170, 40


def cut_bands(mask):
    prof = gaussian_filter1d(mask.sum(1).astype(np.float32), 6)
    h, cuts, y = len(prof), [0], 0
    while h - y > BAND_MAX:
        lo, hi = y + BAND_MIN, min(h, y + BAND_MAX)
        y = lo + int(np.argmin(prof[lo:hi]))
        cuts.append(y)
    cuts.append(h)
    return cuts


def render_band(pid, bi, y0, y1, lab, comps, img_h):
    top, bot = max(0, y0 - CTX), min(img_h, y1 + CTX)
    sub = lab[top:bot]
    out = np.full(sub.shape + (3,), 255, np.uint8)
    out[sub > 0] = (215, 215, 215)
    placed = []
    for k, c in enumerate(comps):
        if c["band"] != bi:
            continue
        col = COLORS[k % len(COLORS)]
        out[sub == c["lab"]] = col
        x, y, w, h = c["bbox"]
        tx, ty = x, y - top - 3
        if w * h < 120:
            tx, ty = x + w + 2, y - top + h
        for _ in range(12):
            r = (tx, ty - 11, tx + 8 * len(str(c["id"])), ty)
            if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in placed):
                break
            ty -= 11
        placed.append(r)
        cv2.putText(out, str(c["id"]), (tx, max(10, ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.line(out, (0, y0 - top), (out.shape[1], y0 - top), (255, 200, 120), 1)
    cv2.line(out, (0, y1 - top), (out.shape[1], y1 - top), (255, 200, 120), 1)
    cv2.imwrite(str(SEG / f"{pid}_band{bi:02d}.png"), out)


def main():
    SEG.mkdir(parents=True, exist_ok=True)
    for pid in page_ids():
        img = cv2.imread(str(PAGES / f"{pid}.jpg"))
        d = darkness(img)
        lab, stats, keep = ink_components(d, pid)
        clean = np.isin(lab, keep)
        cuts = cut_bands(clean)
        comps = []
        for i in keep:
            x, y, w, h, a = (int(v) for v in stats[i])
            cy = y + h / 2
            band = int(np.searchsorted(cuts, cy, side="right") - 1)
            comps.append({"lab": int(i), "bbox": [x, y, w, h], "area": a, "band": band})
        comps.sort(key=lambda c: (c["band"], c["bbox"][0]))
        for n, c in enumerate(comps, 1):
            c["id"] = n
        np.savez_compressed(SEG / f"{pid}.npz", lab=lab.astype(np.int32), d=d.astype(np.float16))
        (SEG / f"{pid}.json").write_text(json.dumps({"cuts": cuts, "comps": comps}))
        for bi in range(len(cuts) - 1):
            render_band(pid, bi, cuts[bi], cuts[bi + 1], lab, comps, img.shape[0])
        print(pid, "components", len(comps), "bands", len(cuts) - 1)


if __name__ == "__main__":
    main()
