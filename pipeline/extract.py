"""Cut labeled glyphs out of the pages and normalize them to a common baseline and cap height."""
import json
import pickle
import cv2
import numpy as np
from common import SEG, LABELS, WORK, thresholds

NO_METRICS = set("ĄĘiQJ,.:;-+=/!?()'\"_*")
MARGIN = 6


def parse(path):
    lines = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        seq = []
        for tok in raw.split():
            if tok in ("|", "_"):
                seq.append(tok)
                continue
            ids, ch = tok.split(":", 1)
            seq.append((tuple(int(i) for i in ids.split("+")), ch))
        lines.append(seq)
    return lines


def cut(lab, d, comps_by_id, ids, low):
    labs = [comps_by_id[i]["lab"] for i in ids]
    xs = [comps_by_id[i]["bbox"] for i in ids]
    x0 = max(0, min(b[0] for b in xs) - MARGIN)
    y0 = max(0, min(b[1] for b in xs) - MARGIN)
    x1 = min(lab.shape[1], max(b[0] + b[2] for b in xs) + MARGIN)
    y1 = min(lab.shape[0], max(b[1] + b[3] for b in xs) + MARGIN)
    sub = lab[y0:y1, x0:x1]
    own = np.isin(sub, labs)
    other = (sub > 0) & ~own
    near = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    soft = np.where(near & ~other, d[y0:y1, x0:x1].astype(np.float32), 0)
    inner = np.argwhere(own)
    return {"box": [x0, y0, x1, y1], "mask": own, "soft": soft, "parts": np.where(own, sub, 0),
            "ink": [int(x0 + inner[:, 1].min()), int(y0 + inner[:, 0].min()),
                    int(x0 + inner[:, 1].max() + 1), int(y0 + inner[:, 0].max() + 1)], "low": low}


def line_metrics(items):
    ref = [g for g in items if g["char"] not in NO_METRICS]
    if len(ref) < 2:
        ref = [g for g in items if g["char"] not in ",.:;-+=/"] or items
    tops = np.array([g["ink"][1] for g in ref], float)
    bots = np.array([g["ink"][3] for g in ref], float)
    cx = np.array([(g["ink"][0] + g["ink"][2]) / 2 for g in ref], float)
    cap = float(np.median(bots - tops))
    if len(ref) >= 4 and np.ptp(cx) > cap:
        slope = float(np.clip(np.polyfit(cx, bots, 1)[0], -0.15, 0.15))
    else:
        slope = 0.0
    base0 = float(np.median(bots - slope * cx))
    return cap, slope, base0


def excluded():
    f = LABELS / "exclude.txt"
    return set(f.read_text().split()) if f.exists() else set()


def main():
    glyphs, lines_out = [], []
    skip = excluded()
    for lf in sorted(LABELS.glob("p-*.txt")):
        pid = lf.stem
        meta = json.loads((SEG / f"{pid}.json").read_text())
        z = np.load(SEG / f"{pid}.npz")
        lab, d = z["lab"], z["d"].astype(np.float32)
        low, _ = thresholds(d)
        cmap = {c["id"]: c for c in meta["comps"]}
        for li, seq in enumerate(parse(lf)):
            items = []
            for tok in seq:
                if isinstance(tok, str):
                    items.append(tok)
                    continue
                ids, ch = tok
                if f"{pid}:{ids[0]}" in skip:
                    items.append("|")
                    continue
                g = cut(lab, d, cmap, ids, low)
                g.update(page=pid, line=li, char=ch, ids=ids)
                items.append(g)
            gl = [g for g in items if isinstance(g, dict)]
            cap, slope, base0 = line_metrics(gl)
            for g in gl:
                cx = (g["ink"][0] + g["ink"][2]) / 2
                g["cap"], g["base"] = cap, base0 + slope * cx
                g["idx"] = len(glyphs)
                glyphs.append(g)
            lines_out.append([t if isinstance(t, str) else t["idx"] for t in items])
    with open(WORK / "glyphs.pkl", "wb") as f:
        pickle.dump({"glyphs": glyphs, "lines": lines_out}, f)
    counts = {}
    for g in glyphs:
        counts[g["char"]] = counts.get(g["char"], 0) + 1
    print(len(glyphs), "glyphs")
    print(" ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
