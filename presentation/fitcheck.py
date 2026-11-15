"""Rough text-overflow check without a renderer (wider fallback fonts => conservative).

uv run --with python-pptx --with pillow python fitcheck.py deck.pptx
"""

import sys

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-{}.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif{}.ttf"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def font(serif, bold, pt):
    path = (
        SERIF.format("-Bold" if bold else "")
        if serif
        else SANS.format("Bold" if bold else "Regular")
    )
    return ImageFont.truetype(path, int(pt * 10))  # 10x for precision


def lines_needed(text, f, width_pt):
    n = 0
    for para in text.split("\n"):
        cur, n = "", n + 1
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if f.getlength(t) / 10 > width_pt and cur:
                n, cur = n + 1, w
            else:
                cur = t
    return n


def inherited_size(shape, slide):
    if not shape.is_placeholder:
        return None
    try:
        lay = slide.slide_layout.placeholders.get(idx=shape.placeholder_format.idx)
        rpr = lay.text_frame._txBody.find(f".//{A}defRPr")
        if rpr is not None and rpr.get("sz"):
            return int(rpr.get("sz")) / 100
        for p in lay.text_frame.paragraphs:
            for r in p.runs:
                if r.font.size:
                    return r.font.size.pt
    except Exception:
        pass
    return None


prs = Presentation(sys.argv[1])
bad = 0
for i, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        w_pt, h_pt = Emu(sh.width).pt, Emu(sh.height).pt
        base = inherited_size(sh, slide) or 18
        is_title = sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type)
        need = 0.0
        for p in sh.text_frame.paragraphs:
            txt = "".join(r.text for r in p.runs)
            pt = max([r.font.size.pt for r in p.runs if r.font.size] or [base])
            bold = any(r.font.bold for r in p.runs) or is_title
            serif = is_title or any((r.font.name or "") == "Cambria" for r in p.runs)
            bullet = p._pPr is not None and p._pPr.find(f"{A}buChar") is not None
            n = lines_needed(txt, font(serif, bold, pt), w_pt - (18 if bullet else 0)) if txt else 1
            need += n * pt * 1.2 + (p.space_after.pt if p.space_after else 0)
        if need > h_pt * 1.02:
            bad += 1
            print(f"slide {i}: '{sh.text_frame.text[:50]}' needs {need:.0f}pt, box {h_pt:.0f}pt")
print("overflow candidates:", bad)
