# Jurek Hand

A variable handwriting font (`wght` 100–900) built 1:1 from scanned school notes.

Each glyph is traced from real ink: the stroke centerline plus a per-point pen radius taken from the distance transform. Outlines are regenerated from that model, so every weight master is point-compatible by construction. Up to six real versions of each character are kept and rotated through a `calt` cycle, so repeated letters never look identical. Side bearings and word space are fitted by least squares to the gaps measured in the notes.

Live: https://jurek-hand.gzowo.fun. The scanned notes and labels stay private; the built font and the comparison crops are in `font/`.

## Pipeline

All scripts live in `pipeline/` and run with the project venv (Python 3.12, `opencv-python-headless`, `scikit-image`, `scipy`, `fonttools[woff]`, `pillow`, `freetype-py`).

1. `segment.py` – page images → ink components, numbered line bands
2. `sheets.py` – zoomed labeling sheets
3. label by hand in `data/labels/p-XXX.txt` (`ids:char`, `|` break, `_` word space), drop bad samples in `exclude.txt`
4. `extract.py` – cut glyphs, per-line baseline and cap height
5. `review.py` – contact sheets per character
6. `vectorize.py` – skeleton graph → smoothed strokes with radius
7. `build.py` – variant picking, composed accents, spacing fit, three masters → `font/JurekHand[wght].ttf/.woff2`
8. `variants_sheet.py`, `compare_refs.py` – checks against the original notes

`index.html` is the specimen page: type, weight and size sliders, variant toggle, notes vs font, every glyph with its variants.
