"""Side-by-side check: original glyph ink vs reconstructed stroke model."""
import pickle
import sys
import cv2
import numpy as np
from common import WORK

H = 70


def draw_model(strokes, k=1.0, scale=H / 700, pad=10):
    allp = np.vstack([s["pts"] for s in strokes]) if strokes else np.zeros((1, 2))
    rmax = max((s["r"].max() for s in strokes), default=10) * k
    xmin, xmax = allp[:, 0].min() - rmax, allp[:, 0].max() + rmax
    w = int((xmax - xmin) * scale) + 2 * pad
    img = np.full((int(H * 2.0), max(w, 20)), 255, np.uint8)
    base = int(H * 1.35)

    def tp(p):
        return (int(round((p[0] - xmin) * scale)) + pad, int(round(base - p[1] * scale)))

    for s in strokes:
        pts, r = s["pts"], s["r"] * k
        if s.get("closed"):
            pts, r = np.vstack([pts, pts[:1]]), np.concatenate([r, r[:1]])
        if len(pts) == 1:
            cv2.circle(img, tp(pts[0]), max(1, int(r[0] * scale)), 0, -1, cv2.LINE_AA)
        for i in range(len(pts) - 1):
            n = int(np.linalg.norm(pts[i + 1] - pts[i]) / 3) + 1
            for t in np.linspace(0, 1, n + 1):
                q = pts[i] * (1 - t) + pts[i + 1] * t
                rr = r[i] * (1 - t) + r[i + 1] * t
                cv2.circle(img, tp(q), max(1, int(round(rr * scale))), 0, -1, cv2.LINE_AA)
    img[base, :] = np.minimum(img[base, :], 200)
    img[base - H, :] = np.minimum(img[base - H, :], 225)
    return img


def main(chars):
    data = pickle.load(open(WORK / "glyphs.pkl", "rb"))
    strokes = pickle.load(open(WORK / "strokes.pkl", "rb"))
    from review import tile
    rows = []
    for ch in chars:
        idx = [i for i, g in enumerate(data["glyphs"]) if g["char"] == ch][:12]
        pair = []
        for i in idx:
            a = tile(data["glyphs"][i])
            b = draw_model(strokes[i])
            pair.append(np.hstack([a, np.full((a.shape[0], 4), 160, np.uint8), b, np.full((a.shape[0], 16), 255, np.uint8)]))
        if pair:
            rows.append(np.hstack(pair)[:, :1900])
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 3), (0, w - r.shape[1])), constant_values=120) for r in rows]
    cv2.imwrite(str(WORK / "review" / "compare.png"), np.vstack(rows))


if __name__ == "__main__":
    main(sys.argv[1])
