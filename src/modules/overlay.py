"""
Overlay / edit-trace detector (Module: overlay) — Phase 2.

Detects the "opaque box over a value + retype" tampering attack purely from PDF
structure (no rendering, no network). See docs/superpowers/specs/2026-07-04-
phase2-overlay-detector-design.md. Geometry alone is SOFT (heuristic); it
escalates to HARD only with a tampering-provenance corroborator.
"""
import logging
import re
import statistics
import unicodedata
import fitz

from src.models import Flag, ModuleResult

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
    # Collapse ASCII whitespace via split(), then explicitly strip the no-break
    # spaces PDFs use as thousands separators: U+202F (narrow no-break space)
    # and U+00A0 (no-break space). Critical for French amounts, whose thousands
    # separator is typically U+202F, not a plain space -- matching "1 200,00"
    # against "1200,00" must not depend on split() incidentally handling
    # Unicode whitespace (a future refactor could silently break it).
    return "".join(s.split()).replace("\u202f", "").replace("\xa0", "").casefold()


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


# =============================================================================
# SCORING
# =============================================================================
# Same weights as every other module (CLAUDE.md scoring table): each flag
# subtracts its severity's points from a starting score of 100.
SEVERITY_POINTS = {"low": 5, "medium": 15, "high": 30, "critical": 50}


# =============================================================================
# PLACEHOLDER GUARD  (the single most FP-sensitive piece of Phase 2)
# =============================================================================
# A legitimate template / mail-merge engine produces the EXACT structural
# signature of the fraud: it paints a placeholder ("0,00", "XXX"), draws a white
# box over it, then stamps the real value on top — so the paint order
# (placeholder seqno < box seqno < value seqno) is identical to a cover-and-
# retype forgery. The ONLY reliable discriminator is WHAT gets covered:
#   - fraud   covers a REAL value  (e.g. "1 200,00") -> tampering
#   - template covers a PLACEHOLDER (e.g. "0,00")     -> benign scaffolding
# So Signals 1 and 3 must NOT fire when the covered/hidden text is a placeholder.

# Textual placeholder tokens (already lower-cased; compared against norm()).
_PLACEHOLDER_TOKENS = {
    "xxx", "x", "x.xx", "x,xx", "n/a", "na", "tbd", "-", "--", "---", "___",
}

# "Zeros and separators only" pattern, applied to the norm()'d string (which has
# already had ALL whitespace, U+202F and U+00A0 removed and been case-folded).
# The currency set includes the real glyphs (€ $ £ ¥) AND U+00B7 (MIDDLE DOT):
# PyMuPDF's base-14 Helvetica decodes an inserted "€" back to U+00B7 via
# get_texttrace, so a "0,00 €" placeholder surfaces as "0,00·". Including U+00B7
# lets the guard recognise that placeholder regardless of font encoding. This
# only ever ADDS currency-symbol tolerance — a real value like "1 200,00" still
# contains the non-zero digits 1 and 2, which are NOT in the class, so it can
# never be mistaken for a placeholder (the fraud/template discriminator holds).
_PLACEHOLDER_ZEROS_RE = re.compile(r"[0.,\s€$£¥·]*")


def is_placeholder(text: str) -> bool:
    """True if `text` is benign template scaffolding rather than a real value.

    A placeholder is one of: empty, a zeros-and-separators string ("0,00",
    "0.00 €", ""), or a known token ("XXX", "N/A", "-", ...). When the COVERED
    text of a cover-and-retype pattern is a placeholder, the pattern is honest
    template generation, not tampering — Signals 1 and 3 stay silent.

    Examples:
        >>> is_placeholder("0,00 €")     # template placeholder
        True
        >>> is_placeholder("XXX")
        True
        >>> is_placeholder("1 200,00 €") # a real value
        False
    """
    n = norm(text)
    if not n:
        return True
    if _PLACEHOLDER_ZEROS_RE.fullmatch(n):
        return True
    return n in _PLACEHOLDER_TOKENS


# =============================================================================
# AMOUNT PARSING  (for the human-readable delta in the cover-and-retype message)
# =============================================================================
# Money-like run: a digit then digits / whitespace / separators. \s already
# covers the no-break (U+00A0) and narrow no-break (U+202F) spaces PDFs use
# as thousands separators, which _money_value() strips before parsing.
_MONEY_RE = re.compile(r"\d[\d\s.,]*")


def _money_value(text: str) -> float | None:
    """Parse the numeric amount out of a span's text, or None if not money-like.

    Handles the French convention (space/U+202F thousands separator, comma
    decimal) as well as the anglo convention (comma thousands, dot decimal).
    Returns a float so callers can compute a delta between two amounts.
    """
    m = _MONEY_RE.search(text or "")
    if not m:
        return None
    raw = m.group(0)
    # Drop every kind of thousands whitespace separator.
    s = raw.replace(" ", "").replace(" ", "").replace("\xa0", "")
    if "," in s and "." in s:
        # Both present: the LAST separator is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # French: 1.200,00
        else:
            s = s.replace(",", "")                      # Anglo: 1,200.00
    elif "," in s:
        s = s.replace(",", ".")                         # French decimal comma
    try:
        return float(s)
    except ValueError:
        return None


# =============================================================================
# PER-PAGE EXTRACTION
# =============================================================================
def _page_angle(page) -> float:
    """Baseline text angle for the page, derived from page.rotation.

    get_texttrace() reports each span's `dir` in the page's *rotated* coordinate
    system, so horizontal body text on a 90°-rotated page still reads dir≈(1,0).
    We keep the baseline at 0 for the common 0/180 case and only shift for the
    90/270 landscape rotations, matching is_horizontal()'s comparison.
    """
    rot = int(getattr(page, "rotation", 0) or 0) % 360
    return 90.0 if rot in (90, 270) else 0.0


def _painted_spans(page) -> list[dict]:
    """Text spans that are actually painted (type 0=fill, 2=fill+stroke).

    Invisible OCR text (type 3) and clipping spans are dropped: they cover
    nothing and must not be treated as hidden content (legit OCR sandwich).
    """
    spans = []
    for s in page.get_texttrace():
        if s.get("type") in (0, 2):
            spans.append(s)
    return spans


def _candidate_fills(page, bg, median_line_h: float,
                     provenance_rects: list) -> list[dict]:
    """Opaque, value-sized, background-coloured fills NOT owned by a form/annot.

    These are the only rectangles that could plausibly be a "cover box". Each
    filter is a false-positive guard:
      - is_opaque:        a translucent tint can't hide text underneath
      - color_matches_bg: a coloured band (zebra/header) is decoration, not a patch
      - is_localized_fill: rules out full-width bands and page backgrounds
      - not in provenance: form-field / annotation backgrounds are legitimate
    """
    fills = []
    for d in page.get_drawings():
        if d.get("type") not in ("f", "fs") or d.get("fill") is None:
            continue
        rect = d["rect"]
        if not is_opaque(d.get("fill_opacity")):
            continue
        if not color_matches_bg(d["fill"], bg):
            continue
        if not is_localized_fill(rect, page.rect, median_line_h):
            continue
        if any(containment(rect, pr) >= CONTAINMENT_MIN for pr in provenance_rects):
            continue  # a form field / annotation background, not a cover box
        fills.append(d)
    return fills


def _extract_page(page, page_angle: float) -> dict:
    """Gather everything the three signals need from a single page.

    Returns a dict with painted spans, candidate cover-fills, provenance rects
    (widget + annotation boxes to exclude), the page background colour and the
    median line height (used to reject fills that aren't value-sized).
    """
    spans = _painted_spans(page)

    # median painted-span height — the yardstick for "value-sized" fills.
    heights = [abs(s["bbox"][3] - s["bbox"][1]) for s in spans if s.get("bbox")]
    median_line_h = statistics.median(heights) if heights else 12.0

    # Provenance rects: content living inside a widget (form field) or an
    # annotation is authored through a legitimate channel, not the page content
    # stream, so a fill sitting inside one must not be read as a cover box.
    provenance_rects = []
    try:
        for w in page.widgets():
            provenance_rects.append(w.rect)
    except Exception:
        pass
    try:
        for a in page.annots():
            provenance_rects.append(a.rect)
    except Exception:
        pass

    bg = page_bg_color(page)
    fills = _candidate_fills(page, bg, median_line_h, provenance_rects)

    return {
        "spans": spans,
        "fills": fills,
        "provenance_rects": provenance_rects,
        "median_line_h": median_line_h,
        "page_angle": page_angle,
        "bg": bg,
    }


# =============================================================================
# THE THREE GEOMETRY SIGNALS
# =============================================================================
def _pair_is_conflict(a: dict, b: dict, page_angle: float) -> bool:
    """Do two spans hold conflicting versions of the SAME field?

    Shared predicate for Signals 2 and 3: different text, but the same *kind* of
    content (both numeric / both alpha, never punctuation-only), both horizontal,
    and matching size — i.e. one plausibly replaced the other. char_class returns
    "excluded" for accented / combining text so French prose never conflicts.
    """
    ta, tb = span_text(a), span_text(b)
    if norm(ta) == norm(tb):
        return False                       # identical text = faux-bold, not a conflict
    ca = char_class(ta)
    if ca == "excluded" or ca != char_class(tb):
        return False
    if not (is_horizontal(a["dir"], page_angle) and is_horizontal(b["dir"], page_angle)):
        return False
    sa, sb = a.get("size", 0.0), b.get("size", 0.0)
    if max(sa, sb) <= 0 or abs(sa - sb) / max(sa, sb) > SIZE_MATCH_FRAC:
        return False                       # different font sizes = different fields
    return True


def _signal_cover_and_retype(fills: list[dict], spans: list[dict], page_angle: float):
    """Signal 3 — the headline attack: a value covered by a box, retyped on top.

    For each candidate fill F, look for two spans A (painted BEFORE F) and B
    (painted AFTER F) that both sit inside F and hold conflicting values. The
    paint order A.seqno < F.seqno < B.seqno is the smoking gun: the original was
    hidden, then a new value stamped over the box.

    PLACEHOLDER GUARD: if the COVERED span A is a placeholder ("0,00", "XXX"),
    this is a legit template stamping a real value onto scaffolding, not fraud —
    skip it. Returns (hits, consumed_span_ids, consumed_fill_ids); consumed items
    are withheld from Signals 1 and 2 so one event yields one flag.
    """
    hits = []
    consumed_spans: set[int] = set()
    consumed_fills: set[int] = set()

    for f in fills:
        frect = f["rect"]
        inside = [s for s in spans if containment(fitz.Rect(s["bbox"]), frect) >= CONTAINMENT_MIN]
        under = [s for s in inside if s.get("seqno", 0) < f.get("seqno", 0)]
        over = [s for s in inside if s.get("seqno", 0) > f.get("seqno", 0)]
        found = None
        for a in under:
            if is_placeholder(span_text(a)):
                continue                   # covered a placeholder -> benign template fill
            for b in over:
                if _pair_is_conflict(a, b, page_angle):
                    found = (a, b)
                    break
            if found:
                break
        if found:
            a, b = found
            hits.append({"fill": f, "under": a, "over": b})
            consumed_fills.add(id(f))
            consumed_spans.add(id(a))
            consumed_spans.add(id(b))
    return hits, consumed_spans, consumed_fills


def _signal_opaque_fill_over_text(fills: list[dict], spans: list[dict],
                                  consumed_spans: set, consumed_fills: set) -> list[dict]:
    """Signal 1 — an opaque box painted OVER still-present live text.

    A candidate fill F covers a painted span T that was drawn BEFORE it
    (T.seqno < F.seqno) and is still extractable underneath: recoverable hidden
    content. PLACEHOLDER GUARD: a covered placeholder is benign scaffolding, not
    hidden content, so skip it. Fills/spans already claimed by Signal 3 are
    skipped to avoid double-reporting the same event.
    """
    hits = []
    for f in fills:
        if id(f) in consumed_fills:
            continue
        frect = f["rect"]
        for t in spans:
            if id(t) in consumed_spans:
                continue
            if t.get("seqno", 0) >= f.get("seqno", 0):
                continue                   # painted after the box = on top, not hidden
            if containment(fitz.Rect(t["bbox"]), frect) < CONTAINMENT_MIN:
                continue
            if is_placeholder(span_text(t)):
                continue                   # covered placeholder = benign template scaffolding
            hits.append({"fill": f, "hidden": t})
            break                          # one report per fill is enough
    return hits


def _signal_conflicting_text(spans: list[dict], page_angle: float,
                             consumed_spans: set) -> list[dict]:
    """Signal 2 — two conflicting values stacked at the same spot (no cover box).

    Two painted spans overlap heavily (iou >= CONFLICT_IOU_MIN) yet hold
    different values of the same kind: a leftover original still peeking through
    a retype. Spans consumed by Signal 3 are excluded. Identical text (faux-bold)
    is not a conflict (handled inside _pair_is_conflict). PLACEHOLDER GUARD:
    symmetric with Signals 1/3 — if the lower-seqno ("under"/original) span of
    the pair is a placeholder, this is benign template scaffolding (a real
    value painted over a placeholder), not a genuine conflict, so skip it. In
    real fraud the lower-seqno span is the original REAL value, never a
    placeholder, so genuine conflicts still fire.
    """
    hits = []
    live = [s for s in spans if id(s) not in consumed_spans]
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            if iou(fitz.Rect(a["bbox"]), fitz.Rect(b["bbox"])) < CONFLICT_IOU_MIN:
                continue
            if _pair_is_conflict(a, b, page_angle):
                under = a if a.get("seqno", 0) <= b.get("seqno", 0) else b
                if is_placeholder(span_text(under)):
                    continue
                hits.append({"a": a, "b": b})
    return hits


# =============================================================================
# FLAG CONSTRUCTION  (SOFT codes only — HARD escalation is Task 4)
# =============================================================================
def _cover_and_retype_flag(hit: dict, page_num: int) -> Flag:
    """Build the OVERLAY_COVER_AND_RETYPE flag, with a value delta when possible."""
    under_txt = span_text(hit["under"]).strip()
    over_txt = span_text(hit["over"]).strip()
    details = {"page": page_num, "covered": under_txt, "replacement": over_txt}
    old_v, new_v = _money_value(under_txt), _money_value(over_txt)
    if old_v is not None and new_v is not None:
        delta = new_v - old_v
        details["delta"] = delta
        msg = (f"A value appears to have been covered and retyped on page "
               f"{page_num + 1}: \"{under_txt}\" was hidden and \"{over_txt}\" "
               f"placed over it (a change of {delta:+,.2f}).")
    else:
        msg = (f"A value appears to have been covered and retyped on page "
               f"{page_num + 1}: \"{under_txt}\" was hidden and \"{over_txt}\" "
               f"placed over it.")
    return Flag(severity="critical", code="OVERLAY_COVER_AND_RETYPE",
                message=msg, details=details)


def _analyze_page(page, page_angle: float, page_num: int) -> list[Flag]:
    """Run the three signals on one page and turn the hits into SOFT flags."""
    extracted = _extract_page(page, page_angle)
    spans, fills = extracted["spans"], extracted["fills"]

    flags: list[Flag] = []

    # Signal 3 first: it consumes the fills/spans it explains so Signals 1 and 2
    # don't re-report the same cover-and-retype event under a weaker code.
    hits3, consumed_spans, consumed_fills = _signal_cover_and_retype(fills, spans, page_angle)
    for hit in hits3:
        flags.append(_cover_and_retype_flag(hit, page_num))

    for hit in _signal_opaque_fill_over_text(fills, spans,
                                             consumed_spans, consumed_fills):
        t = span_text(hit["hidden"]).strip()
        flags.append(Flag(
            severity="high", code="OVERLAY_OPAQUE_FILL_OVER_TEXT",
            message=(f"An opaque box on page {page_num + 1} covers still-present "
                     f"text (\"{t}\"), which remains recoverable underneath."),
            details={"page": page_num, "hidden": t},
        ))

    for hit in _signal_conflicting_text(spans, page_angle, consumed_spans):
        ta, tb = span_text(hit["a"]).strip(), span_text(hit["b"]).strip()
        flags.append(Flag(
            severity="high", code="OVERLAY_CONFLICTING_TEXT",
            message=(f"Two conflicting values overlap at the same position on "
                     f"page {page_num + 1}: \"{ta}\" and \"{tb}\"."),
            details={"page": page_num, "values": [ta, tb]},
        ))

    return flags


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def analyze_overlay(pdf_data) -> ModuleResult:
    """Detect opaque-box cover-and-retype tampering from PDF structure alone.

    SOFT (heuristic geometry) only — corroboration/HARD escalation is Task 4.
    Never raises: a malformed / encrypted / textless PDF yields a partial
    ModuleResult with reduced confidence instead of an exception (CLAUDE.md).
    """
    flags: list[Flag] = []
    confidence = 1.0

    if not getattr(pdf_data, "extraction_ok", True):
        # Extraction already failed upstream; nothing to analyse. The analyzer
        # short-circuits this path in production, but a direct caller must still
        # get a well-formed, low-confidence result (never a crash).
        return ModuleResult(module="overlay", flags=[], score=100, confidence=0.5)

    doc = None
    try:
        doc = fitz.open(pdf_data.file_path)
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                flags.extend(_analyze_page(page, _page_angle(page), page_num))
            except Exception as exc:  # one bad page must not sink the whole module
                logger.warning("overlay: page %d analysis failed: %s", page_num, exc)
                confidence = min(confidence, 0.5)
    except Exception as exc:  # malformed / encrypted / unreadable — fail soft
        logger.warning("overlay: analysis failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = 0.5
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    score = max(0, score)

    return ModuleResult(module="overlay", flags=flags, score=score, confidence=confidence)
