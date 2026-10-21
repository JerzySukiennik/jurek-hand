"""Original note lines vs the same text typed in Jurek Hand, rendered with HarfBuzz (calt on)."""
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import PAGES, WORK, DIST

CAP_PX = 56
LINES = [
    ("p-000", [8, 9], "ODP. W SOBOTĘ PRZEJECHAŁ"),
    ("p-000", [10], "28 KM, W NIEDZIELĘ 24"),
    ("p-000", [11], "A W PIĄTEK 8."),
    ("p-006", [3], "EN LA ESCUELA TENGO MUCHOS CLASES:"),
    ("p-006", [7], "POR LA TARDE VUELVO A CASA,"),
    ("p-002", [2], "FRYTKI - 2X - 7 ZŁ"),
    ("p-003", [3], "TERAZ:"),
]


def crop_line(glyphs, pid, lines):
    gs = [g for g in glyphs if g["page"] == pid and g["line"] in lines]
    x0 = min(g["ink"][0] for g in gs)
    x1 = max(g["ink"][2] for g in gs)
    cap = float(np.median([g["cap"] for g in gs]))
    base = float(np.median([g["base"] for g in gs]))
    img = cv2.imread(str(PAGES / f"{pid}.jpg"))
    y0, y1 = int(base - 1.45 * cap), int(base + 0.45 * cap)
    from common import darkness
    band = darkness(img[max(0, y0 + int(0.3 * cap)):int(base + 0.1 * cap)]) > 0.5
    cols = band.sum(0) > 1
    step = int(1.6 * cap)
    while x1 + step < img.shape[1] and cols[int(x1):int(x1) + step].any():
        x1 = int(x1) + 1 + int(np.nonzero(cols[int(x1):int(x1) + step])[0].max())
    x0, x1 = max(0, int(x0 - 0.5 * cap)), min(img.shape[1], int(x1 + 0.5 * cap))
    crop = img[max(0, y0):y1, x0:x1]
    s = CAP_PX / cap
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    d = darkness(img[max(0, y0):y1, x0:x1])
    ink = img[max(0, y0):y1, x0:x1][d > 0.5]
    color = tuple(int(c) for c in np.median(ink, axis=0)[::-1]) if len(ink) else (40, 40, 120)
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), color


def render_text(text, color, wght=400):
    size = int(CAP_PX / 0.7)
    font = ImageFont.truetype(str(DIST / "JurekHand[wght].ttf"), size, layout_engine=ImageFont.Layout.RAQM)
    font.set_variation_by_axes([wght])
    w = int(font.getlength(text, features=["calt"])) + 2 * CAP_PX
    img = Image.new("RGB", (w, int(CAP_PX * 1.9)), (250, 249, 246))
    ImageDraw.Draw(img).text((CAP_PX // 2, int(CAP_PX * 1.45)), text, font=font, fill=color, anchor="ls", features=["calt"])
    return img


def main():
    glyphs = pickle.load(open(WORK / "glyphs.pkl", "rb"))["glyphs"]
    pairs, meta = [], []
    (DIST / "ref").mkdir(exist_ok=True)
    for n, (pid, lines, text) in enumerate(LINES):
        ref, color = crop_line(glyphs, pid, lines)
        ref.save(DIST / "ref" / f"{n}.png")
        meta.append({"src": f"ref/{n}.png", "text": text, "color": "#%02x%02x%02x" % color, "cap": CAP_PX})
        pairs.append((ref, render_text(text, color)))
    import json
    (DIST / "ref" / "lines.json").write_text(json.dumps(meta, ensure_ascii=False))
    W = max(max(a.width, b.width) for a, b in pairs) + 160
    H = sum(a.height + b.height + 40 for a, b in pairs) + 40
    out = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    y = 20
    for a, b in pairs:
        dr.text((16, y + a.height // 2), "notatka", font=label, fill=(140, 140, 140), anchor="lm")
        out.paste(a, (140, y))
        y += a.height
        dr.text((16, y + b.height // 2), "font", font=label, fill=(140, 140, 140), anchor="lm")
        out.paste(b, (140, y))
        y += b.height + 40
        dr.line((16, y - 20, W - 16, y - 20), fill=(225, 225, 225), width=1)
    out.save(DIST / "porownanie.png")
    print("saved", DIST / "porownanie.png", out.size)


if __name__ == "__main__":
    main()
