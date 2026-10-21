"""Render every outline variant straight from the built font, one row per character."""
import pickle
import sys
import cv2
import numpy as np
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from common import DIST, WORK

H = 90


def render(gs, name, wght):
    pen = FreeTypePen(None)
    gs[name].draw(pen)
    s = H / 700
    w = max(int(gs[name].width * s) + 4, 10)
    img = pen.array(width=w, height=int(H * 1.9), transform=(s, 0, 0, s, 2, H * 0.55), contain=False, evenOdd=False)
    return (255 - np.clip(img * 255, 0, 255)).astype(np.uint8)


def main(wght=400, chars=None):
    font = TTFont(DIST / "JurekHand[wght].ttf")
    gs = font.getGlyphSet(location={"wght": wght})
    order = font.getGlyphOrder()
    rows = {}
    for n in order:
        if ".v" in n:
            base = n.split(".v")[0]
            rows.setdefault(base, []).append(n)
    keys = [k for k in sorted(rows) if not chars or k in chars]
    src = pickle.load(open(WORK / "report.pkl", "rb")).get("sources", {})
    lines = []
    for k in keys:
        tiles = []
        for j, n in enumerate(rows[k]):
            t = render(gs, n, wght)
            lab = src.get(k, [""] * 9)[j] if j < len(src.get(k, [])) else ""
            t = np.pad(t, ((0, 0), (0, max(0, 70 - t.shape[1]))), constant_values=255)
            cv2.putText(t, lab, (2, t.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 90, 1)
            tiles.append(t)
        tiles = [np.pad(t, ((0, 0), (0, 6)), constant_values=255) for t in tiles]
        row = np.hstack([np.full((tiles[0].shape[0], 110), 255, np.uint8)] + tiles)
        row[int(H * 1.35), :] = np.minimum(row[int(H * 1.35), :], 210)
        cv2.putText(row, k[:12], (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        lines.append(row)
    w = max(l.shape[1] for l in lines)
    lines = [np.pad(l, ((0, 2), (0, w - l.shape[1])), constant_values=170) for l in lines]
    for i in range(0, len(lines), 12):
        cv2.imwrite(str(WORK / "review" / f"variants_{i // 12}.png"), np.vstack(lines[i:i + 12]))


if __name__ == "__main__":
    main(400, sys.argv[1].split(",") if len(sys.argv) > 1 else None)
