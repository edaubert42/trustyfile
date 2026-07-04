"""
Overlay / edit-trace detector (Module: overlay) — Phase 2.

Detects the "opaque box over a value + retype" tampering attack purely from PDF
structure (no rendering, no network). See docs/superpowers/specs/2026-07-04-
phase2-overlay-detector-design.md. Geometry alone is SOFT (heuristic); it
escalates to HARD only with a tampering-provenance corroborator.
"""
import logging
import unicodedata
import fitz

logger = logging.getLogger(__name__)

OPAQUE_FILL_MIN_OPACITY = 0.90
BG_COLOR_MATCH_TOL = 0.06
CONTAINMENT_MIN = 0.8
CONFLICT_IOU_MIN = 0.6
LOCALIZED_FILL_MAX_AREA_FRAC = 0.10
LINE_HEIGHT_LO, LINE_HEIGHT_HI = 0.5, 2.5
MAX_FILL_ASPECT = 25.0
HORIZONTAL_TOL_DEG = 4.0
SIZE_MATCH_FRAC = 0.10
INTERACTIVE_EDITOR_PRODUCERS = (
    "acrobat", "foxit", "pdf-xchange", "pdfxchange", "libreoffice",
    "nitro", "sejda", "pdfescape", "master pdf",
)


def iou(r1: fitz.Rect, r2: fitz.Rect) -> float:
    inter = r1 & r2
    if inter.is_empty or inter.is_infinite:
        return 0.0
    ia = inter.get_area()
    union = r1.get_area() + r2.get_area() - ia
    return ia / union if union > 0 else 0.0


def containment(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    if inter.is_empty or inner.get_area() <= 0:
        return 0.0
    return inter.get_area() / inner.get_area()


def is_opaque(fill_opacity) -> bool:
    return fill_opacity is None or fill_opacity >= OPAQUE_FILL_MIN_OPACITY


def color_matches_bg(color, bg) -> bool:
    if color is None or bg is None:
        return False
    return all(abs(c - b) <= BG_COLOR_MATCH_TOL for c, b in zip(color, bg))


def is_localized_fill(rect: fitz.Rect, page_rect: fitz.Rect, median_line_h: float) -> bool:
    w, h = abs(rect.width), abs(rect.height)
    if w <= 2 or h <= 2:
        return False  # hairline rule / underline
    if page_rect.get_area() > 0 and rect.get_area() > LOCALIZED_FILL_MAX_AREA_FRAC * page_rect.get_area():
        return False  # background, not a patch
    if median_line_h > 0 and not (LINE_HEIGHT_LO * median_line_h <= h <= LINE_HEIGHT_HI * median_line_h):
        return False  # not value-sized (drops full-height bands)
    if max(w, h) / max(min(w, h), 0.01) > MAX_FILL_ASPECT:
        return False  # extreme aspect = rule/band
    return True


def span_text(span: dict) -> str:
    out = []
    for c in span.get("chars", []):
        try:
            out.append(chr(c[0]))
        except (TypeError, ValueError):
            pass
    return "".join(out)


def _has_combining_mark(s: str) -> bool:
    return any(unicodedata.category(ch) in ("Mn", "Mc") for ch in s)


def char_class(s: str) -> str:
    stripped = "".join(ch for ch in s if not ch.isspace() and ch not in "€$£.,-")
    if _has_combining_mark(s) or s != unicodedata.normalize("NFC", s):
        return "excluded"
    if not stripped:
        return "excluded"
    if all(ch.isdigit() for ch in stripped):
        return "numeric"
    if all(ch.isalpha() for ch in stripped):
        return "alpha"
    return "mixed"


def norm(s: str) -> str:
    return "".join(s.split()).replace(" ", "").replace(" ", "").casefold()


def is_horizontal(span_dir, page_angle_deg: float) -> bool:
    import math
    dx, dy = span_dir
    ang = math.degrees(math.atan2(dy, dx))          # span text angle
    # normalize both angles to [0,360) and compare to the page baseline (page_angle_deg)
    diff = (ang - page_angle_deg) % 360.0
    diff = min(diff, 360.0 - diff)
    return diff <= HORIZONTAL_TOL_DEG


def page_bg_color(page) -> tuple:
    """Earliest fill covering >=90% of the page (or full-page from origin); else white."""
    pr = page.rect
    best = None
    for d in page.get_drawings():
        if d.get("type") in ("f", "fs") and d.get("fill") is not None:
            r = d["rect"]
            if r.get_area() >= 0.9 * pr.get_area():
                if best is None or d.get("seqno", 0) < best[0]:
                    best = (d.get("seqno", 0), tuple(d["fill"]))
    return best[1] if best else (1.0, 1.0, 1.0)
