"""Contact sheets of extracted glyphs, one row per character, normalized to cap height."""
import pickle
import sys
import cv2
import numpy as np
from common import WORK

H = 70


def tile(g):
    s = H / g["cap"]
    x0, y0, x1, y1 = g["box"]
    img = (255 - np.clip(g["soft"] * 300, 0, 255)).astype(np.uint8)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    top = int(round((g["base"] - g["cap"] * 1.35 - y0) * s))
    canvas = np.full((int(H * 2.0), img.shape[1] + 8), 255, np.uint8)
    for yy in range(img.shape[0]):
        ty = yy - top
        if 0 <= ty < canvas.shape[0]:
            canvas[ty, 4:4 + img.shape[1]] = np.minimum(canvas[ty, 4:4 + img.shape[1]], img[yy])
    base_y = int(round(H * 1.35))
    canvas[base_y, :] = np.minimum(canvas[base_y, :], 200)
    canvas[base_y - H, :] = np.minimum(canvas[base_y - H, :], 225)
    cv2.putText(canvas, f"{g['page'][2:]}:{g['ids'][0]}", (2, canvas.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 120, 1)
    return canvas


def main(chars=None):
    data = pickle.load(open(WORK / "glyphs.pkl", "rb"))
    by = {}
    for g in data["glyphs"]:
        by.setdefault(g["char"], []).append(g)
    keys = sorted(by) if not chars else [c for c in chars if c in by]
    rows = []
    for ch in keys:
        tiles = [tile(g) for g in by[ch]]
        row = np.hstack([np.full((tiles[0].shape[0], 40), 255, np.uint8)] + tiles)
        cv2.putText(row, ch if ch.isascii() else "?", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        rows.append(row[:, :1900])
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 2), (0, w - r.shape[1])), constant_values=180) for r in rows]
    out = WORK / "review"
    out.mkdir(exist_ok=True)
    for f in out.glob("rev_*.png"):
        f.unlink()
    per = 10
    for i in range(0, len(rows), per):
        cv2.imwrite(str(out / f"rev_{i // per:02d}.png"), np.vstack(rows[i:i + per]))
    print("rows", len(rows), "->", keys)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
