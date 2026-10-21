"""Build the Jurek Hand variable font (wght axis) from stroke models."""
import pickle
import random
import unicodedata
import numpy as np
from fontTools import varLib
from fontTools.agl import UV2AGL
from fontTools.designspaceLib import AxisDescriptor, DesignSpaceDocument, InstanceDescriptor, SourceDescriptor
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.otlLib.builder import buildStatTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._g_l_y_f import flagOverlapSimple
from common import WORK, DIST

FAMILY = "Jurek Hand"
STAMP = 3843849600
CAP = 700
R0 = 17.0
SLOTS = 6
MASTERS = [(100, 0.45), (400, 1.0), (900, 2.6)]
INSTANCES = [(100, "Thin"), (200, "ExtraLight"), (300, "Light"), (400, "Regular"), (500, "Medium"),
             (600, "SemiBold"), (700, "Bold"), (800, "ExtraBold"), (900, "Black")]
ACUTE_SRC = {"Á", "Í", "Ć"}
COMPOSE_ACUTE = {"Ś": "S", "Ź": "Z", "Ń": "N", "Ó": "O", "Ć": "C", "É": "E", "Á": "A", "Í": "I", "Ú": "U", "Ý": "Y"}
COMPOSE_DOT = {"Ż": "Z"}


def gname(ch):
    cp = ord(ch)
    return UV2AGL.get(cp, f"uni{cp:04X}")


def bounds(strokes):
    p = np.vstack([s["pts"] for s in strokes])
    return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()


def copy_strokes(strokes, dx=0.0, dy=0.0, sx=1.0, sy=1.0, rk=1.0):
    out = []
    for s in strokes:
        q = dict(s)
        q["pts"] = s["pts"] * [sx, sy] + [dx, dy]
        q["r"] = s["r"] * rk
        out.append(q)
    return out


def normalize_radius(strokes):
    body = [s for s in strokes if not s.get("dot")] or strokes
    med = float(np.median(np.concatenate([s["r"] for s in body])))
    out = []
    for s in strokes:
        q = dict(s)
        mod = np.clip(s["r"] / med, 0.6, 1.6) if not s.get("dot") else np.clip(s["r"] / med, 0.8, 2.2)
        q["r"] = R0 * mod
        out.append(q)
    return out


def descriptor(strokes, n=24):
    img = np.zeros((n, n), np.float32)
    x0, x1, _, _ = bounds(strokes)
    w = max(x1 - x0, 1)
    for s in strokes:
        for p in s["pts"]:
            ix = int(np.clip((p[0] - x0) / w * (n - 1), 0, n - 1))
            iy = int(np.clip((1.3 * CAP - p[1]) / (1.6 * CAP) * (n - 1), 0, n - 1))
            img[iy, ix] += 1
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 1.2)
    return (img / (np.linalg.norm(img) + 1e-9)).ravel()


def pick_variants(samples, k):
    if len(samples) <= k:
        return list(range(len(samples)))
    d = np.array([descriptor(s) for s in samples])
    dist = 1 - d @ d.T
    rng = random.Random(len(samples))
    best, best_cost = None, 1e9
    for _ in range(40):
        med = rng.sample(range(len(samples)), k)
        for _ in range(20):
            assign = np.argmin(dist[:, med], axis=1)
            new = []
            for j in range(k):
                members = np.nonzero(assign == j)[0]
                if len(members) == 0:
                    new.append(med[j])
                    continue
                new.append(int(members[np.argmin(dist[np.ix_(members, members)].sum(1))]))
            if new == med:
                break
            med = new
        cost = dist[:, med].min(1).sum()
        if cost < best_cost:
            best, best_cost = list(med), cost
    return best


def split_points(pts, rmax, closed):
    n = len(pts)
    cuts = set()
    for i in range(1, n - 1):
        a, b = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la < 1e-6 or lb < 1e-6:
            continue
        ang = np.arccos(np.clip(a @ b / (la * lb), -1, 1))
        if ang > np.radians(50):
            cuts.add(i)
        elif ang > 1e-3 and min(la, lb) / ang < 1.15 * rmax[i]:
            cuts.add(i)
    return sorted(cuts)


def pieces(stroke, rmax):
    pts, r = stroke["pts"], stroke["r"]
    if stroke.get("closed"):
        pts = np.vstack([pts, pts[:2]])
        r = np.concatenate([r, r[:2]])
        rmax = np.concatenate([rmax, rmax[:2]])
    cuts = split_points(pts, rmax, False)
    bounds_ = [0] + cuts + [len(pts) - 1]
    out = []
    for a, b in zip(bounds_[:-1], bounds_[1:]):
        if b > a:
            out.append((a, b))
    return pts, r, out


def envelope(pts, r):
    n = len(pts)
    t = np.zeros_like(pts)
    for i in range(n):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        d = b - a
        t[i] = d / (np.linalg.norm(d) + 1e-9)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    dr = np.gradient(r, s) if n > 1 and s[-1] > 0 else np.zeros(n)
    dr = np.clip(dr, -0.7, 0.7)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    c = np.sqrt(1 - dr ** 2)[:, None]
    left = pts + r[:, None] * (nrm * c - t * dr[:, None])
    right = pts + r[:, None] * (-nrm * c - t * dr[:, None])
    return left, right, t


def cap_points(center, radius, tangent, forward):
    k = 1 / np.cos(np.radians(22.5))
    base = np.arctan2(tangent[1], tangent[0])
    angs = [base + np.pi / 2 - np.radians(a) for a in (22.5, 67.5, 112.5, 157.5)]
    if not forward:
        angs = [a + np.pi for a in angs]
    return [center + radius * k * np.array([np.cos(a), np.sin(a)]) for a in angs]


def piece_contour(pts, r):
    if len(pts) == 1 or np.linalg.norm(pts[-1] - pts[0]) < 1e-3 and len(pts) <= 2:
        return circle_contour(pts[0], r[0])
    left, right, t = envelope(pts, r)
    c = [(left[0], True)]
    c += [(p, False) for p in left[1:-1]]
    c += [(left[-1], True)]
    c += [(p, False) for p in cap_points(pts[-1], r[-1], t[-1], True)]
    c += [(right[-1], True)]
    c += [(p, False) for p in right[1:-1][::-1]]
    c += [(right[0], True)]
    c += [(p, False) for p in cap_points(pts[0], r[0], t[0], False)]
    return c


def circle_contour(center, radius):
    k = 1 / np.cos(np.radians(22.5))
    return [(center + radius * k * np.array([np.cos(a), np.sin(a)]), False)
            for a in np.radians([-22.5, -67.5, -112.5, -157.5, 157.5, 112.5, 67.5, 22.5])]


def stroke_contours(strokes, kmul, rmax_k):
    out = []
    for s in strokes:
        if s.get("dot") or len(s["pts"]) == 1:
            out.append(circle_contour(s["pts"][0], s["r"][0] * kmul))
            continue
        pts, r, segs = pieces(s, s["r"] * rmax_k)
        rm = np.maximum(r * kmul, 5.0)
        for a, b in segs:
            out.append(piece_contour(pts[a:b + 1], rm[a:b + 1]))
    return out


def draw_glyph(contours, dx):
    pen = TTGlyphPen(None)
    for c in contours:
        pts = [(int(round(p[0] + dx)), int(round(p[1]))) for p, _ in c]
        on = [o for _, o in c]
        if not any(on):
            pen.qCurveTo(*pts, None)
            pen.closePath()
            continue
        i0 = on.index(True)
        pts, on = pts[i0:] + pts[:i0], on[i0:] + on[:i0]
        pen.moveTo(pts[0])
        buf = []
        for p, o in zip(pts[1:], on[1:]):
            if o:
                pen.qCurveTo(*buf, p) if buf else pen.lineTo(p)
                buf = []
            else:
                buf.append(p)
        if buf:
            pen.qCurveTo(*buf, pts[0])
        pen.closePath()
    g = pen.glyph()
    if g.numberOfContours > 0:
        g.flags[0] |= flagOverlapSimple
    return g


def fit_spacing(glyphs, lines, core, chars):
    idx = {c: i for i, c in enumerate(chars)}
    rows, rhs, words = [], [], []
    for seq in lines:
        prev, sep = None, None
        for t in seq:
            if isinstance(t, str):
                sep = t if prev is not None else None
                if t == "|":
                    prev = None
                continue
            if prev is not None:
                ga, gb = glyphs[prev], glyphs[t]
                s = CAP / ga["cap"]
                ra = ga["ink"][0] * s + core[prev][1]
                lb = gb["ink"][0] * s + core[t][0]
                gap = lb - ra
                if sep == "_":
                    words.append((ga["char"], gb["char"], gap))
                elif ga["char"] in idx and gb["char"] in idx and -100 < gap < 500:
                    row = np.zeros(2 * len(chars))
                    row[2 * idx[ga["char"]] + 1] = 1
                    row[2 * idx[gb["char"]]] = 1
                    rows.append(row)
                    rhs.append(gap)
            prev, sep = t, None
    rhs = np.array(rhs)
    m = float(np.median(rhs)) / 2
    lam = 0.6
    A = np.vstack([np.array(rows), lam * np.eye(2 * len(chars))])
    y = np.concatenate([rhs, lam * np.full(2 * len(chars), m)])
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    sol = np.clip(sol, -0.05 * CAP, 0.35 * CAP)
    sb = {c: (sol[2 * i], sol[2 * i + 1]) for c, i in idx.items()}
    space = np.median([g - sb.get(a, (m, m))[1] - sb.get(b, (m, m))[0] for a, b, g in words]) if words else 0.45 * CAP
    return sb, m, float(space), len(rows)


def mark_placement(samples):
    acute, offs = [], []
    for st, g in samples:
        if g["char"] not in ACUTE_SRC:
            continue
        top = [s for s in st if s["pts"][:, 1].min() > 0.92 * CAP and np.ptp(s["pts"], axis=0).max() > 50]
        body = [s for s in st if all(s is not t for t in top)]
        if not top or not body:
            continue
        bx0, bx1, _, by1 = bounds(body)
        ax0, ax1, ay0, _ = bounds(top)
        cx = (ax0 + ax1) / 2
        acute.append(copy_strokes(top, dx=-cx, dy=-ay0))
        offs.append(((cx - (bx0 + bx1) / 2) / max(bx1 - bx0, 1), ay0 - by1))
    rel = float(np.clip(np.median([o[0] for o in offs]), -0.1, 0.3)) if offs else 0.1
    gap = float(np.clip(np.median([o[1] for o in offs]), 30, 80)) if offs else 60
    return acute, rel, gap


def i_dots(samples):
    dots = []
    for st, g in samples:
        if g["char"] != "i" or len(st) < 2:
            continue
        hi = max(st, key=lambda s: s["pts"][:, 1].mean())
        rest = [s for s in st if s is not hi]
        _, _, _, top = bounds(rest)
        x0, x1, y0, y1 = bounds([hi])
        if (x1 - x0) < 140 and (y1 - y0) < 140:
            dots.append((copy_strokes([hi], dx=-(x0 + x1) / 2, dy=-(y0 + y1) / 2), (y0 + y1) / 2 - top, rest))
    return dots


def compose(base, mark, rel, gap):
    bx0, bx1, _, by1 = bounds(base)
    _, _, my0, _ = bounds(mark)
    cx = (bx0 + bx1) / 2 + rel * (bx1 - bx0)
    return base + copy_strokes(mark, dx=cx, dy=by1 + gap - my0)


def synth_stroke(p0, p1, r=R0):
    pts = np.array([p0, p1], float)
    return {"pts": pts, "r": np.array([r, r]), "closed": False, "dot": False}


def main():
    data = pickle.load(open(WORK / "glyphs.pkl", "rb"))
    strokes = pickle.load(open(WORK / "strokes.pkl", "rb"))
    glyphs = data["glyphs"]
    norm = [normalize_radius(st) if st else st for st in strokes]
    core = [(bounds(st)[0], bounds(st)[1]) if st else (0, 0) for st in norm]

    by_char = {}
    for i, (g, st) in enumerate(zip(glyphs, norm)):
        if st:
            by_char.setdefault(g["char"], []).append((st, g, i))

    acute, rel, gap = mark_placement([(st, g) for st, g, _ in sum(by_char.values(), [])])
    dots = i_dots([(st, g) for st, g, _ in by_char.get("i", [])])
    print("acute marks", len(acute), "rel", round(rel, 3), "gap", round(gap, 1), "i-dots", len(dots))

    by_char.pop("I", None)
    synthetic = set()
    variants, sources = {}, {}
    for ch, items in by_char.items():
        sts = [it[0] for it in items]
        k = min(SLOTS, len(sts))
        pick = pick_variants(sts, k)
        variants[ch] = [sts[j] for j in pick]
        sources[ch] = [f"{items[j][1]['page'][2:]}:{items[j][1]['ids'][0]}" for j in pick]

    for ch in "-=+":
        if ch not in variants:
            continue
        cy = [(bounds(v)[2] + bounds(v)[3]) / 2 for v in variants[ch]]
        mid = float(np.median(cy))
        variants[ch] = [copy_strokes(v, dy=(mid - c) * 0.9) for v, c in zip(variants[ch], cy)]

    for ch, base in {**COMPOSE_ACUTE, **COMPOSE_DOT}.items():
        if base not in variants:
            continue
        have = variants.get(ch, [])
        need = max(0, min(SLOTS, len(variants[base])) - len(have))
        made = []
        for j in range(need):
            b = variants[base][j % len(variants[base])]
            if ch in COMPOSE_DOT and dots:
                mark, dgap, _ = dots[j % len(dots)]
                made.append(compose(b, mark, 0.0, dgap - 30))
            elif acute:
                made.append(compose(b, acute[j % len(acute)], rel, gap))
        if made:
            variants[ch] = have + made
            sources[ch] = sources.get(ch, []) + [f"{base}+mark"] * len(made)
            if not have:
                synthetic.add(ch)

    if "O" in variants and "Q" not in variants:
        qs = []
        for b in variants["O"]:
            x0, x1, y0, y1 = bounds(b)
            w = x1 - x0
            qs.append(b + [synth_stroke((x0 + 0.55 * w, y0 + 0.18 * (y1 - y0)), (x1 + 0.12 * w, y0 - 0.12 * (y1 - y0)))])
        variants["Q"] = qs
        synthetic.add("Q")
    if "." in variants:
        dot = variants["."][0]
        _, _, dy0, dy1 = bounds(dot)
        if "!" not in variants and "/" in variants:
            out = []
            for st in variants["/"][:3]:
                x0, x1, y0, y1 = bounds(st)
                top = copy_strokes(st, dx=-(x0 + x1) / 2, dy=-y0)
                h = y1 - y0
                top = copy_strokes(top, sy=0.62 * CAP / h, sx=0.5, dy=0.3 * CAP)
                out.append(top + copy_strokes(dot, dx=-(bounds(dot)[0] + bounds(dot)[1]) / 2))
            variants["!"] = out
            synthetic.add("!")
        if ";" not in variants and "," in variants:
            out = []
            for d in variants["."][:3]:
                cx = (bounds(d)[0] + bounds(d)[1]) / 2
                out.append(copy_strokes(d, dx=-cx, dy=0.42 * CAP) + copy_strokes(variants[","][0], dx=-(bounds(variants[","][0])[0] + bounds(variants[","][0])[1]) / 2))
            variants[";"] = out
            synthetic.add(";")
    if acute:
        if "'" not in variants:
            variants["'"] = [copy_strokes(a, dy=0.72 * CAP) for a in acute]
            synthetic.add("'")
        if '"' not in variants:
            variants['"'] = [copy_strokes(a, dy=0.72 * CAP) + copy_strokes(a, dx=0.16 * CAP, dy=0.72 * CAP) for a in acute]
            synthetic.add('"')

    chars = sorted(variants)
    spacing, mean_sb, space_w, npairs = fit_spacing(glyphs, data["lines"], core, chars)
    space_w = float(np.clip(space_w, 0.28 * CAP, 0.62 * CAP))
    print("spacing pairs", npairs, "mean sb", round(mean_sb, 1), "space", round(space_w, 1))

    order = [".notdef", "space"]
    outline_names, slot_map = {}, {}
    for ch in chars:
        base = gname(ch)
        outline_names[ch] = [f"{base}.v{j}" for j in range(len(variants[ch]))]
        order += outline_names[ch]
    for ch in chars + [" "]:
        base = gname(ch) if ch != " " else "space"
        names = [base] + [f"{base}.s{j}" for j in range(1, SLOTS)]
        slot_map[ch] = names
        order += [n for n in names if n not in order]

    seq_rng = random.Random(7)
    slot_variant = {}
    for ch in chars:
        k = len(variants[ch])
        seq = list(range(k)) * (SLOTS // k + 1)
        seq = seq[:SLOTS]
        for _ in range(50):
            seq_rng.shuffle(seq)
            if all(seq[i] != seq[(i + 1) % SLOTS] for i in range(SLOTS)) or k == 1:
                break
        slot_variant[ch] = seq

    cmap = {32: "space", 0xA0: "space"}
    for ch in chars:
        cmap[ord(ch)] = slot_map[ch][0]
        low = ch.lower()
        if low != ch and len(low) == 1 and low not in variants:
            cmap[ord(low)] = slot_map[ch][0]
    if "i" in variants:
        cmap[ord("i")] = slot_map["i"][0]
        cmap[ord("I")] = slot_map["i"][0]

    rk_max = max(k for _, k in MASTERS)
    masters = []
    for wght, kmul in MASTERS:
        delta = R0 * kmul - R0
        glyf, hmtx = {}, {}
        glyf[".notdef"] = TTGlyphPen(None).glyph()
        hmtx[".notdef"] = (int(space_w), 0)
        for ch in chars:
            lsb, rsb = spacing.get(ch, (mean_sb, mean_sb))
            for j, st in enumerate(variants[ch]):
                x0, x1, _, _ = bounds(st)
                dx = lsb + delta - x0
                g = draw_glyph(stroke_contours(st, kmul, rk_max), dx)
                name = outline_names[ch][j]
                glyf[name] = g
                hmtx[name] = (int(round(lsb + (x1 - x0) + rsb + 2 * delta)), 0)
        for ch in chars + [" "]:
            for slot, name in enumerate(slot_map[ch]):
                if ch == " ":
                    glyf[name] = TTGlyphPen(None).glyph()
                    hmtx[name] = (int(round(space_w)), 0)
                    continue
                target = outline_names[ch][slot_variant[ch][slot]]
                pen = TTGlyphPen(glyf)
                pen.addComponent(target, (1, 0, 0, 1, 0, 0))
                glyf[name] = pen.glyph()
                hmtx[name] = hmtx[target]
        fb = FontBuilder(1000, isTTF=True)
        fb.setupGlyphOrder(order)
        fb.setupCharacterMap(cmap)
        fb.setupGlyf(glyf)
        gt = fb.font["glyf"]
        metrics = {}
        for name in order:
            gg = gt[name]
            gg.recalcBounds(gt)
            metrics[name] = (hmtx[name][0], getattr(gg, "xMin", 0) if gg.numberOfContours else 0)
        fb.setupHorizontalMetrics(metrics)
        fb.setupHorizontalHeader(ascent=950, descent=-350)
        fb.setupNameTable({"familyName": FAMILY, "styleName": "Regular", "uniqueFontIdentifier": f"{FAMILY} {wght}",
                           "fullName": FAMILY, "psName": "JurekHand-Regular", "version": "Version 1.000",
                           "designer": "Jurek (handwriting), built by Claude"})
        fb.setupOS2(sTypoAscender=950, sTypoDescender=-350, sTypoLineGap=0, usWinAscent=1100, usWinDescent=450,
                    sCapHeight=CAP, sxHeight=CAP, usWeightClass=wght, fsSelection=0x40 | 0x80, achVendID="JRKH", version=4)
        fb.setupPost()
        fb.setupHead(unitsPerEm=1000, created=STAMP, modified=STAMP)
        masters.append((wght, fb.font))
        print("master", wght, "glyphs", len(order))

    doc = DesignSpaceDocument()
    ax = AxisDescriptor()
    ax.tag, ax.name, ax.minimum, ax.default, ax.maximum = "wght", "Weight", 100, 400, 900
    doc.addAxis(ax)
    for wght, font in masters:
        src = SourceDescriptor()
        src.font, src.location, src.name = font, {"Weight": wght}, f"master{wght}"
        src.filename = f"master{wght}.ttf"
        doc.addSource(src)
    for wght, nm in INSTANCES:
        ins = InstanceDescriptor()
        ins.familyName, ins.styleName, ins.location = FAMILY, nm, {"Weight": wght}
        doc.addInstance(ins)
    built = varLib.build(doc)
    vf = built[0] if isinstance(built, tuple) else built
    buildStatTable(vf, [{"tag": "wght", "name": "Weight", "values": [
        {"value": w, "name": n, "flags": 0x2 if w == 400 else 0} for w, n in INSTANCES]}])

    classes = []
    for slot in range(SLOTS):
        classes.append("[" + " ".join(slot_map[ch][slot] for ch in chars + [" "]) + "]")
    fea = "".join(f"@S{i} = {c};\n" for i, c in enumerate(classes))
    fea += "feature calt {\n  lookup cycle {\n"
    for i in range(SLOTS - 1):
        fea += f"    sub @S{i} @S0' by @S{i + 1};\n"
    fea += "  } cycle;\n} calt;\n"
    addOpenTypeFeaturesFromString(vf, fea)

    DIST.mkdir(exist_ok=True)
    vf.save(DIST / "JurekHand[wght].ttf")
    vf.flavor = "woff2"
    vf.save(DIST / "JurekHand[wght].woff2")
    have = sorted(set(chars))
    missing_pl = [c for c in "AĄBCĆDEĘFGHJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ0123456789.,:;!?-+=/()'\"" if c not in variants]
    report = {
        "chars": have, "synthetic": sorted(synthetic), "missing": missing_pl,
        "variants": {c: len(v) for c, v in variants.items()}, "sources": {gname(c): v for c, v in sources.items()}, "samples": {c: len(v) for c, v in by_char.items()},
        "space": space_w, "pairs": npairs,
    }
    pickle.dump(report, open(WORK / "report.pkl", "wb"))
    import json
    import time
    build_id = int(time.time())
    (DIST / "info.json").write_text(json.dumps({
        "chars": have, "synthetic": sorted(synthetic), "variants": report["variants"],
        "samples": {c: len(v) for c, v in by_char.items()}, "glyphs": len(order), "pages": len({g["page"] for g in glyphs}),
        "letters": len(glyphs), "build": build_id}, ensure_ascii=False))
    print("synthetic", "".join(sorted(synthetic)), "missing", "".join(missing_pl))
    print("saved", DIST)


if __name__ == "__main__":
    main()
