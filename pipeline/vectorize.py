"""Turn glyph bitmaps into stroke models: smoothed centerlines with a per-point pen radius."""
import pickle
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from common import WORK

CAP = 700
HI = 260
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def hires(g):
    u = HI / g["cap"]
    soft = cv2.resize(g["soft"], None, fx=u, fy=u, interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(soft, (0, 0), u * 0.35)
    m = soft > g["low"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= u * u * 3
    m = keep[lab]
    holes = ndimage.binary_fill_holes(m) & ~m
    hn, hl, hs, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = np.zeros(hn, bool)
    small[1:] = hs[1:, cv2.CC_STAT_AREA] < u * u * 4
    return m | small[hl], u


def pixel_graph(skel):
    pts = set(zip(*[a.tolist() for a in np.nonzero(skel)]))
    adj = {}
    for y, x in pts:
        nb = []
        for dy, dx in N8:
            q = (y + dy, x + dx)
            if q not in pts:
                continue
            if dy and dx and ((y + dy, x) in pts or (y, x + dx) in pts):
                continue
            nb.append(q)
        adj[(y, x)] = nb
    return adj


def trace(adj):
    deg = {p: len(v) for p, v in adj.items()}
    nodes = {p for p, d in deg.items() if d != 2}
    cid = {}
    for p in nodes:
        if p in cid:
            continue
        cid[p] = p
        stack = [p]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b in nodes and b not in cid and deg[a] >= 3 and deg[b] >= 3:
                    cid[b] = p
                    stack.append(b)
    groups = {}
    for p, c in cid.items():
        groups.setdefault(c, []).append(p)
    center = {c: np.mean(v, axis=0) for c, v in groups.items()}
    seen, edges = set(), []
    for p in nodes:
        for q in adj[p]:
            if (p, q) in seen or (q in nodes and cid[q] == cid[p]):
                continue
            path = [p, q]
            seen.update({(p, q), (q, p)})
            prev, cur = p, q
            while cur not in nodes:
                nxt = [r for r in adj[cur] if r != prev]
                if not nxt:
                    break
                r = nxt[0]
                seen.update({(cur, r), (r, cur)})
                path.append(r)
                prev, cur = cur, r
            pts = [np.array(t, float) for t in path]
            a, b = cid[p], cid.get(cur, cur)
            pts[0] = center[a]
            if b in center:
                pts[-1] = center[b]
            edges.append({"a": a, "b": b, "pts": pts})
    for p in adj:
        if deg[p] == 2 and not any((p, q) in seen for q in adj[p]):
            path, prev, cur = [p], p, adj[p][0]
            seen.update({(p, cur), (cur, p)})
            while cur != p:
                path.append(cur)
                nxt = [r for r in adj[cur] if r != prev and (cur, r) not in seen]
                if not nxt:
                    break
                r = nxt[0]
                seen.update({(cur, r), (r, cur)})
                prev, cur = cur, r
            edges.append({"a": ("loop", p), "b": ("loop", p), "pts": [np.array(t, float) for t in path], "cycle": True})
    return edges


def node_degree(edges):
    deg = {}
    for e in edges:
        if e.get("cycle"):
            continue
        deg[e["a"]] = deg.get(e["a"], 0) + 1
        deg[e["b"]] = deg.get(e["b"], 0) + 1
    return deg


def length(pts):
    p = np.array(pts)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def simplify_graph(edges, dt):
    for _ in range(6):
        deg = node_degree(edges)
        changed = False
        for e in list(edges):
            if e.get("cycle") or len(edges) <= 1:
                continue
            for end, other in (("a", "b"), ("b", "a")):
                if deg.get(e[end]) == 1 and deg.get(e[other], 0) >= 3:
                    j = e["pts"][0] if other == "a" else e["pts"][-1]
                    rj = dt[int(round(j[0])), int(round(j[1]))]
                    if length(e["pts"]) < 1.6 * rj + 3:
                        edges.remove(e)
                        changed = True
                    break
        deg = node_degree(edges)
        for n, d in deg.items():
            if d != 2:
                continue
            inc = [e for e in edges if not e.get("cycle") and n in (e["a"], e["b"])]
            if len(inc) == 1:
                e = inc[0]
                if e["a"] == e["b"]:
                    e["cycle"] = True
                    e["pts"] = e["pts"][:-1]
                    changed = True
                continue
            e1, e2 = inc
            p1 = e1["pts"] if e1["b"] == n else e1["pts"][::-1]
            p2 = e2["pts"] if e2["a"] == n else e2["pts"][::-1]
            a = e1["a"] if e1["b"] == n else e1["b"]
            b = e2["b"] if e2["a"] == n else e2["a"]
            edges.remove(e1)
            edges.remove(e2)
            ne = {"a": a, "b": b, "pts": p1 + p2[1:]}
            if a == b:
                ne["cycle"] = True
                ne["pts"] = ne["pts"][:-1]
            edges.append(ne)
            changed = True
            break
        if not changed:
            break
    return edges


def resample(p, step, closed):
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return p[[0, -1]] if not closed else p[:-1]
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    out = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
    return out[:-1] if closed else out


def smooth(p, sigma, closed):
    if len(p) < 4:
        return p
    if closed:
        return ndimage.gaussian_filter1d(p, sigma, axis=0, mode="wrap")
    k = min(len(p) - 1, int(3 * sigma) + 1)
    head = 2 * p[0] - p[1:k + 1][::-1]
    tail = 2 * p[-1] - p[-k - 1:-1][::-1]
    q = ndimage.gaussian_filter1d(np.vstack([head, p, tail]), sigma, axis=0, mode="nearest")
    return q[k:k + len(p)]


def rdp(p, eps):
    if len(p) < 3:
        return np.arange(len(p))
    keep = np.zeros(len(p), bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        nrm = np.linalg.norm(ab)
        seg = p[i + 1:j] - a
        if nrm < 1e-9:
            dist = np.linalg.norm(seg, axis=1)
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / nrm
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.nonzero(keep)[0]


def stroke_from_edge(e, dt, ends_free):
    closed = bool(e.get("cycle"))
    raw = np.array(e["pts"])
    r_raw = np.array([dt[int(round(y)), int(round(x))] for y, x in raw])
    core = r_raw.copy()
    if not closed:
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
        inner = r_raw[(s > r_raw.max()) & (s < s[-1] - r_raw.max())]
        mid = np.median(inner) if len(inner) else np.median(r_raw)
        rj_a, rj_b = r_raw[0], r_raw[-1]
        if not ends_free[0]:
            core[s < rj_a * 1.2] = mid
        if not ends_free[1]:
            core[s > s[-1] - rj_b * 1.2] = mid
    p = resample(raw, 1.5, closed)
    s_raw = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
    if closed:
        s_raw = np.concatenate([s_raw, [s_raw[-1] + np.linalg.norm(raw[-1] - raw[0])]])
        core = np.concatenate([core, core[:1]])
    s_new = np.linspace(0, s_raw[-1], len(p) + (1 if closed else 0))[:len(p)]
    r = np.interp(s_new, s_raw, core)
    p = smooth(p, 2.5, closed)
    r = ndimage.gaussian_filter1d(r, 4, mode="wrap" if closed else "nearest")
    return p, r, closed


def vectorize(g):
    m, u = hires(g)
    dt = ndimage.distance_transform_edt(m)
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    x0, y0 = g["box"][0], g["box"][1]
    s = CAP / g["cap"]

    def to_font(p):
        fx = ((p[:, 1] / u) + x0 - g["ink"][0]) * s
        fy = (g["base"] - ((p[:, 0] / u) + y0)) * s
        return np.stack([fx, fy], 1)

    strokes = []
    for c in range(1, n):
        cm = lab == c
        sk = skeletonize(cm)
        rmax = float(dt[cm].max())
        if sk.sum() < 2.5 * rmax or cm.sum() < 2.2 * np.pi * rmax ** 2:
            ys, xs = np.nonzero(cm)
            center = np.array([[ys.mean(), xs.mean()]])
            strokes.append({"pts": to_font(center), "r": np.array([np.sqrt(cm.sum() / np.pi) / u * s]), "closed": False, "dot": True})
            continue
        edges = simplify_graph(trace(pixel_graph(sk)), dt)
        deg = node_degree(edges)
        for e in edges:
            free = (deg.get(e["a"], 0) <= 1, deg.get(e["b"], 0) <= 1)
            p, r, closed = stroke_from_edge(e, dt, free)
            fp = to_font(p)
            fr = r / u * s
            keep = rdp(fp, 1.2)
            if closed and len(keep) < 4:
                keep = np.linspace(0, len(fp) - 1, 6).astype(int)
            strokes.append({"pts": fp[keep], "r": fr[keep], "closed": closed, "dot": False})
    return strokes


def main():
    data = pickle.load(open(WORK / "glyphs.pkl", "rb"))
    out = []
    for g in data["glyphs"]:
        try:
            st = vectorize(g)
        except Exception as ex:
            print("fail", g["page"], g["ids"], g["char"], ex)
            st = []
        out.append(st)
    pickle.dump(out, open(WORK / "strokes.pkl", "wb"))
    pts = sum(len(s["pts"]) for st in out for s in st)
    print("glyphs", len(out), "strokes", sum(len(s) for s in out), "points", pts)


if __name__ == "__main__":
    main()
