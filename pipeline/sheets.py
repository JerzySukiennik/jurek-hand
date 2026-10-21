"""Render zoomed, numbered labeling sheets from segmented pages."""
import json
import sys
import cv2
import numpy as np
from common import SEG, page_ids

COLORS = [(200, 30, 30), (20, 120, 20), (180, 0, 160), (0, 110, 200), (200, 110, 0), (60, 60, 60)]
TARGET_H, CHUNK_W, SHEET_H, PAD = 100, 1700, 1500, 30


def is_edge(c, shape):
    x, y, w, h = c["bbox"]
    H, W = shape
    touches = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
    return touches and (w > 0.25 * W or h > 0.25 * H)


def band_chunks(pid, lab, comps, cuts, b):
    mine = [c for c in comps if c["band"] == b and not is_edge(c, lab.shape)]
    if len(mine) < 2:
        return []
    hs = [c["bbox"][3] for c in mine if c["bbox"][3] > 15]
    s = float(np.clip(TARGET_H / (np.median(hs) if hs else 50), 1.0, 2.6))
    x0 = max(0, min(c["bbox"][0] for c in mine) - PAD)
    x1 = min(lab.shape[1], max(c["bbox"][0] + c["bbox"][2] for c in mine) + PAD)
    y0 = max(0, min(min(c["bbox"][1] for c in mine), cuts[b]) - PAD)
    y1 = min(lab.shape[0], max(max(c["bbox"][1] + c["bbox"][3] for c in mine), cuts[b + 1]) + PAD)
    sub = lab[y0:y1, x0:x1]
    img = np.full(sub.shape + (3,), 255, np.uint8)
    img[sub > 0] = (220, 220, 220)
    for k, c in enumerate(mine):
        img[sub == c["lab"]] = COLORS[k % len(COLORS)]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    top = 26
    img = np.pad(img, ((top, 0), (0, 0), (0, 0)), constant_values=255)
    placed = []
    for k, c in enumerate(mine):
        col = COLORS[k % len(COLORS)]
        x, y, w, h = c["bbox"]
        txt = str(c["id"])
        tw = 10 * len(txt) + 2
        bx0, by0 = int((x - x0) * s), int((y - y0) * s) + top
        by1 = int((y - y0 + h) * s) + top
        cands = [(bx0 + dx * tw, by0 - 3 - dy * 14) for dy in range(4) for dx in range(4)]
        cands += [(bx0 + dx * tw, by1 + 15 + dy * 14) for dy in range(2) for dx in range(4)]
        if w * h < 150:
            cands.insert(0, (int((x - x0 + w) * s) + 3, by1))
        r = None
        for tx, ty in cands:
            ty = max(13, ty)
            q0 = (tx, ty - 13, tx + tw, ty + 1)
            if not any(q0[0] < q[2] and q[0] < q0[2] and q0[1] < q[3] and q[1] < q0[3] for q in placed):
                r = q0
                break
        if r is None:
            tx, ty = cands[0][0], max(13, cands[0][1])
            r = (tx, ty - 13, tx + tw, ty + 1)
        placed.append(r)
        cv2.putText(img, txt, (r[0], r[3] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (r[0], r[3] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    out = []
    for cx in range(0, img.shape[1], CHUNK_W - 120):
        part = img[:, cx:cx + CHUNK_W]
        part = np.pad(part, ((0, 0), (0, CHUNK_W - part.shape[1]), (0, 0)), constant_values=255)
        cv2.putText(part, f"{pid} b{b}", (CHUNK_W - 130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        out.append(part)
        if cx + CHUNK_W >= img.shape[1]:
            break
    return out


def main(pids):
    for pid in pids:
        meta = json.loads((SEG / f"{pid}.json").read_text())
        lab = np.load(SEG / f"{pid}.npz")["lab"]
        chunks = []
        for b in range(len(meta["cuts"]) - 1):
            chunks += band_chunks(pid, lab, meta["comps"], meta["cuts"], b)
        sheets, cur, h = [], [], 0
        for ch in chunks:
            if cur and h + ch.shape[0] > SHEET_H:
                sheets.append(cur)
                cur, h = [], 0
            cur.append(np.pad(ch, ((0, 4), (0, 0), (0, 0)), constant_values=180))
            h += ch.shape[0] + 4
        if cur:
            sheets.append(cur)
        for f in SEG.glob(f"z_{pid}_*.png"):
            f.unlink()
        for i, s in enumerate(sheets):
            cv2.imwrite(str(SEG / f"z_{pid}_{i}.png"), np.vstack(s))
        print(pid, "sheets", len(sheets))


if __name__ == "__main__":
    main(sys.argv[1:] or page_ids())
