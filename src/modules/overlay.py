"""
Overlay / edit-trace detector (Module: overlay) — Phase 2.

Detects the "opaque box over a value + retype" tampering attack purely from PDF
structure (no rendering, no network). See docs/superpowers/specs/2026-07-04-
phase2-overlay-detector-design.md. Geometry alone is SOFT (heuristic); it
escalates to HARD only with a tampering-provenance corroborator.
"""
import difflib
import logging
import re
import statistics
import struct
import fitz

from src.models import Flag, ModuleResult
from src.modules.structure import find_eof_positions
from src.modules.span_conflict import (
    HORIZONTAL_TOL_DEG, SIZE_MATCH_FRAC, is_interactive_editor,
    span_text, char_class, norm, amount_token, is_horizontal,
    is_placeholder, page_angle as _page_angle,
    pair_is_conflict as _pair_is_conflict,
    money_value,
)

logger = logging.getLogger(__name__)

OPAQUE_FILL_MIN_OPACITY = 0.90
BG_COLOR_MATCH_TOL = 0.06
CONTAINMENT_MIN = 0.8
CONFLICT_IOU_MIN = 0.6
# Share of the shorter span's glyphs that must sit on a glyph of the other span.
# A retype stacks new characters on the old ones; a label/value grid written as
# two text runs only looks stacked at span level (FACA invoices: 0% of glyphs
# overlap, every corpus hit was this shape). Low on purpose, so a retype
# placed slightly off still counts.
CONFLICT_GLYPH_OVERLAP_MIN = 0.25
LOCALIZED_FILL_MAX_AREA_FRAC = 0.10
LINE_HEIGHT_LO, LINE_HEIGHT_HI = 0.5, 2.5
MAX_FILL_ASPECT = 25.0
FULL_WIDTH_BAND_FRAC = 0.85  # image width >= this * page width => header/logo band, not a patch

# Font-size tolerance for Signal 3 (cover-and-retype) only. Two texts that
# merely overlap must match within SIZE_MATCH_FRAC (10 %) to count as the same
# field. Signal 3 has stronger evidence binding them — a box painted AFTER the
# original and BEFORE the new text, on the same row — and a forger typing with
# an editor's "add text" tool gets its default size (10 or 12 pt over an 11 pt
# original: 9-17 % off). Measured on the real-tool corpus: retypes over
# LibreOffice / Chromium bases were 13-14 % smaller and slipped through.
COVER_SIZE_MATCH_FRAC = 0.20


def boxes_disjoint(a, b) -> bool:
    """True when two (x0, y0, x1, y1) boxes share no area at all.

    Why this exists: the signals below compare every span with every other
    span (or with every fill / image). On a dense page that is hundreds of
    thousands of pairs, and almost all of them are nowhere near each other.
    Building two fitz.Rect objects and intersecting them costs ~25 microseconds
    in PyMuPDF's Python layer; four float comparisons on plain tuples cost
    ~0.1. So callers ask this first and only build Rects for the few pairs
    that actually touch.

    It is a pure shortcut, never a different answer: when it returns True the
    intersection is empty (zero or negative width or height), which is exactly
    the case where iou() and containment() return 0.0. Boxes that merely touch
    along an edge count as disjoint, as they do for fitz.Rect.is_empty.

    Args:
        a, b: Any 4-item sequences (tuple, list). Pass tuples, not fitz.Rect:
            indexing a Rect is itself slow.

    Examples:
        boxes_disjoint((0, 0, 10, 10), (20, 0, 30, 10)) -> True   (far apart)
        boxes_disjoint((0, 0, 10, 10), (10, 0, 20, 10)) -> True   (edge only)
        boxes_disjoint((0, 0, 10, 10), (5, 5, 15, 15))  -> False  (overlap)
    """
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


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
# PER-PAGE EXTRACTION
# =============================================================================


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
        fbox = tuple(frect)                # plain tuple for the cheap pre-reject
        inside = [s for s in spans
                  if not boxes_disjoint(s["bbox"], fbox)
                  and containment(fitz.Rect(s["bbox"]), frect) >= CONTAINMENT_MIN]
        under = [s for s in inside if s.get("seqno", 0) < f.get("seqno", 0)]
        over = [s for s in inside if s.get("seqno", 0) > f.get("seqno", 0)]
        found = _find_cover_pair(under, over, page_angle)
        # Embedded-span fallback: the box covers only PART of a wider span (label
        # + amount + currency as one text object, e.g. "Gesamtbetrag: 571,04 €"),
        # so the original never sits fully inside the box. Match a span the box
        # sits WITHIN, comparing the amount portions rather than the whole text.
        if found is None:
            found = _find_embedded_cover_pair(f, frect, over, spans, page_angle)
        if found:
            a, b = found
            hits.append({"fill": f, "under": a, "over": b})
            consumed_fills.add(id(f))
            consumed_spans.add(id(a))
            consumed_spans.add(id(b))
    return hits, consumed_spans, consumed_fills


def _find_cover_pair(under: list, over: list, page_angle: float):
    """Standard path: original A and retype B are each fully inside the box."""
    for a in under:
        if is_placeholder(span_text(a)):
            continue                       # covered a placeholder -> benign template fill
        for b in over:
            if _pair_is_conflict(a, b, page_angle, size_frac=COVER_SIZE_MATCH_FRAC):
                return (a, b)
    return None


def _find_embedded_cover_pair(f, frect, over: list, spans: list, page_angle: float):
    """Embedded path: the box sits WITHIN a wider original span. A qualifying
    original A is painted before the box, has the box contained in its bbox, and
    carries an amount that differs from a retyped amount B inside the box."""
    over_amt = [b for b in over if amount_token(span_text(b))]
    if not over_amt:
        return None
    for a in spans:
        if a.get("seqno", 0) >= f.get("seqno", 0):
            continue                       # A must be painted before the box
        if not _box_on_span_row(frect, fitz.Rect(a["bbox"])):
            continue                       # the box must sit on A's row, within its width
        for b in over_amt:
            if _amounts_conflict(a, b, page_angle, size_frac=COVER_SIZE_MATCH_FRAC):
                return (a, b)
    return None


def _box_on_span_row(frect: fitz.Rect, sbbox: fitz.Rect) -> bool:
    """True if fill `frect` sits on span `sbbox`'s text row and within its width.

    Row containment (vertical overlap + horizontal containment) rather than area
    containment: a cover box drawn from a search-rect is often a little taller
    than the text-span bbox, so strict area-containment would miss a box that
    plainly covers part of the line. FP-safe: the box must still lie inside the
    span's horizontal extent (>= CONTAINMENT_MIN of the box width) on the same
    row — i.e. it genuinely covers that span's text, not a neighbour's."""
    y_overlap = min(frect.y1, sbbox.y1) - max(frect.y0, sbbox.y0)
    if y_overlap <= 0.3 * min(frect.height, sbbox.height):
        return False                       # not the same row
    x_inside = min(frect.x1, sbbox.x1) - max(frect.x0, sbbox.x0)
    return frect.width > 0 and x_inside / frect.width >= CONTAINMENT_MIN


def _amounts_conflict(a: dict, b: dict, page_angle: float,
                      size_frac: float = SIZE_MATCH_FRAC) -> bool:
    """Conflict test on the AMOUNT portion of two spans (for the embedded path).

    True when both spans carry a horizontal, same-size monetary amount, the
    covered (A) amount is a real value (not a placeholder), and the two amounts
    differ NUMERICALLY. Comparing parsed numeric values (via money_value) — not
    whole-span text — is what lets a label-wrapped original ("Gesamtbetrag:
    571,04 €") conflict with a bare retype ("690,25"), while REJECTING dates /
    phone numbers / ids (which money_value returns None for) and treating two
    equal amounts written in different styles ("1.200,00" vs "1 200,00") as the
    SAME value, not a spurious change."""
    at, bt = span_text(a), span_text(b)
    av, bv = money_value(at), money_value(bt)
    if av is None or bv is None:
        return False                       # a non-amount (date/phone/id) never conflicts
    a_tok = amount_token(at)
    if a_tok and is_placeholder(a_tok):
        return False                       # covered a placeholder -> benign scaffolding
    if av == bv:
        return False                       # same value (any styling) = re-stamp, not a change
    da, db = a.get("dir", (1.0, 0.0)), b.get("dir", (1.0, 0.0))
    if not (is_horizontal(da, page_angle) and is_horizontal(db, page_angle)):
        return False
    sa, sb = a.get("size", 0.0), b.get("size", 0.0)
    if max(sa, sb) <= 0 or abs(sa - sb) / max(sa, sb) > size_frac:
        return False                       # different font size = different field
    return True


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
        fbox = tuple(frect)                # plain tuple for the cheap pre-reject
        for t in spans:
            if id(t) in consumed_spans:
                continue
            if t.get("seqno", 0) >= f.get("seqno", 0):
                continue                   # painted after the box = on top, not hidden
            if boxes_disjoint(t["bbox"], fbox):
                continue                   # nowhere near the box: containment is 0
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
            # Almost every pair of spans on a page is far apart: skip those
            # on four float comparisons before paying for two fitz.Rect.
            if boxes_disjoint(a["bbox"], b["bbox"]):
                continue
            if iou(fitz.Rect(a["bbox"]), fitz.Rect(b["bbox"])) < CONFLICT_IOU_MIN:
                continue
            if _is_layered_icon(a, b):
                continue
            if _glyph_overlap(a, b) < CONFLICT_GLYPH_OVERLAP_MIN:
                continue                   # boxes overlap, letters do not
            if _pair_is_conflict(a, b, page_angle):
                under = a if a.get("seqno", 0) <= b.get("seqno", 0) else b
                if is_placeholder(span_text(under)):
                    continue
                hits.append({"a": a, "b": b})
    return hits


def _glyph_boxes(span: dict) -> list:
    """The non-empty box of every visible character of a span (spaces dropped)."""
    boxes = []
    for char in span.get("chars", ()):
        x0, y0, x1, y1 = char[3]
        if char[0] != 32 and x1 > x0 and y1 > y0:
            boxes.append(fitz.Rect(x0, y0, x1, y1))
    return boxes


def _glyph_overlap(a: dict, b: dict) -> float:
    """Share of the shorter span's glyphs lying on a glyph of the other span.

    A glyph counts when at least 30% of its area is covered by one glyph of
    the other span. Spans without character data return 1.0, so the signal
    keeps its old behaviour on them.

    Examples:
        "1 250,00" retyped over "1 850,00" at the same spot -> about 1.0
        labels "Série : Type Mine :" and values "SUZ FX359LZ" written as two
        text runs whose boxes cover the same grid      -> 0.0
    """
    glyphs_a, glyphs_b = _glyph_boxes(a), _glyph_boxes(b)
    small, big = sorted((glyphs_a, glyphs_b), key=len)
    if not small:
        return 1.0
    covered = 0
    for glyph in small:
        for other in big:
            if glyph.intersects(other) and (glyph & other).get_area() >= 0.3 * glyph.get_area():
                covered += 1
                break
    return covered / len(small)


def _is_layered_icon(a: dict, b: dict) -> bool:
    """True when two stacked spans are single non-digit glyphs: a layered icon.

    Icon fonts draw a multi-colour pictogram by stacking glyphs at the same
    spot, one per colour. Real case: a genuine EDF invoice's logo is "b" then
    "a" in the PictosSIMM font, 80 pt, same position — which Signal 2 read as
    two conflicting values and flagged critical.

    Digits are NOT excused: a single digit retyped over another is exactly
    the edit this signal exists for, and icon fonts map pictograms to letters
    and symbols, not to the digits of an amount.

    Examples:
        "b" over "a"   -> True   (icon layers)
        "4" over "1"   -> False  (a retyped digit)
        "AB" over "CD" -> False  (real text)
    """
    ta, tb = span_text(a).strip(), span_text(b).strip()
    return (len(ta) == 1 and len(tb) == 1
            and not ta.isdigit() and not tb.isdigit())


# =============================================================================
# CORROBORATION  (Task 4 — provenance-gated HARD escalation)
# =============================================================================
# Geometry alone (Signals 1-3) is a heuristic: it can look identical to honest
# template scaffolding once the placeholder guard is defeated by coincidence.
# Escalating a SOFT finding to a HARD one (which the evidence ledger clamps to
# score <=15) requires actual corroborating evidence that the file passed
# through a hand-editing tool — not just "the file has more than one revision",
# because legitimate multi-session tools (report engines resaving the same PDF
# to correct a value) ALSO produce incremental updates. So:
#   C2 (HARD corroborator) = an interactive-editor Producer/Creator substring
#   C1 (context only)      = more than one %%EOF (incremental update exists)
# Only C2 may drive escalation. See docs/superpowers/specs/2026-07-04-phase2-
# overlay-detector-design.md and the Task-1 design review for the rationale.
def _corroboration(pdf_data, raw_bytes: bytes | None) -> dict:
    """Gather HARD-escalation context: editor provenance and revision count.

    `producer_is_editor` (C2) is the ONLY signal allowed to escalate a SOFT
    overlay finding to a `_CONFIRMED` HARD code — see the module-level comment
    above. `has_incremental_updates` is returned for context/messaging only;
    callers must NOT OR it into the escalation decision.

    Never raises: malformed metadata or unreadable bytes yield False, not an
    exception (CLAUDE.md "never crash on malformed PDFs").
    """
    producer_is_editor = False
    try:
        doc = fitz.open(pdf_data.file_path)
        try:
            meta = doc.metadata or {}
            haystack = f"{meta.get('producer') or ''} {meta.get('creator') or ''}".casefold()
            producer_is_editor = is_interactive_editor(haystack)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("overlay: corroboration metadata read failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)

    has_incremental_updates = False
    try:
        if raw_bytes is not None:
            has_incremental_updates = len(find_eof_positions(raw_bytes)) > 1
    except Exception as exc:
        logger.warning("overlay: corroboration EOF scan failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)

    return {
        "producer_is_editor": producer_is_editor,
        "has_incremental_updates": has_incremental_updates,
    }


# =============================================================================
# SIGNAL 4 (Task 5) — INCREMENTAL-REVISION ROLLBACK + VALUE DIFF
# =============================================================================
# Signals 1-3 only ever see the FINAL document. If a forger DELETES the
# original value outright (e.g. via a real redaction that strips the
# underlying text) instead of merely painting a box over it, there is no
# ghost text left underneath and no covering box to find — Signals 1-3 are
# structurally blind to this variant. PDF incremental saves make a rollback
# possible: each "%%EOF" marker delimits a complete, independently-openable
# prior version of the file (raw_bytes[:eof] IS that version), so we can
# reopen the version that existed BEFORE the edit and diff its extracted
# words, position-by-position, against the version AFTER.
#
# HARD vs SOFT mirrors Task 4's C2 gate exactly (see `_corroboration` above):
# only an interactive-editor Producer/Creator on the revision that made the
# change may escalate to a `_CONFIRMED` HARD code. The same document resaved
# by the SAME non-editor tool (e.g. a report engine correcting a value across
# sessions) must stay SOFT — never a fraud accusation for honest, ambiguous
# multi-session edits.
ROW_Y_TOL = 3.0  # pt; how far apart two words' vertical centres may be and still share a row

# Reflow guard (page level). A targeted edit leaves almost every row of the
# page byte-for-byte identical; a regenerated / re-laid-out revision changes
# most of them. We only trust a row diff when the page around it is stable.
#   - MIN_STABLE_ROWS: too few rows => nothing to compare the change against.
#   - MIN_STABLE_FRACTION: measured on the corpus, a name change leaves 12/14
#     rows stable (86 %), the real invoice_FRW 32/38 (84 %). 75 % keeps both
#     while rejecting a page where a quarter of the rows moved.
MIN_STABLE_ROWS = 3
MIN_STABLE_FRACTION = 0.75

# A bare number fragment ("1", "200,00", "1.250"). French amounts are split
# into several words by the thousands space ("1 200,00" -> "1", "200,00"), so
# when only "1" changed we widen the reported change over the neighbouring
# fragments to show "1 200,00 -> 4 200,00" instead of "1 -> 4".
_NUMBER_FRAGMENT_RE = re.compile(r"[\d.,\u202f\u00a0]+")

# A date-shaped run ("19/08/1992", "19/08/ 1998", "2026-10-14"): used to keep
# dates out of the money-delta computation in _revision_value_flag.
_DATE_RE = re.compile(r"\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}|\d{4}-\d{2}-\d{2}")

# Currency words are allowed inside a "value" change ("200,00 EUR").
_CURRENCY_WORDS = frozenset({"eur", "euro", "euros", "usd", "gbp", "chf"})

# Value-bearing categories worth flagging when they change across a revision
# boundary: dates, IBAN, SIREN/SIRET/VAT-shaped digit runs, and (falling
# through to the last, most permissive alternative) any bare run containing a
# digit — which is what money amounts collapse to once currency symbols and
# separators are stripped by norm(). A pure-prose change (e.g. a company name
# typo fix) has no digit at all and never reaches this alternative.
_VALUE_BEARING_RE = re.compile(
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"        # date: dd/mm/yyyy, dd-mm-yy, ...
    r"|\d{4}-\d{2}-\d{2}"                         # ISO date: yyyy-mm-dd
    r"|[A-Za-z]{2}\d{2}[A-Za-z0-9]{10,30}"          # IBAN-shaped
    r"|\d{9}(?:\d{5})?"                            # SIREN (9 digits) / SIRET (14 digits)
    r"|\d[\d\s\u202f\u00a0.,]*"                 # money / bare number run (most permissive)
)


def _is_value_bearing(text: str) -> bool:
    """True if `text` looks like a value worth flagging when it changes across
    a revision boundary, rather than incidental prose.

    Examples:
        >>> _is_value_bearing("1")            # bare digit — still value-bearing
        True
        >>> _is_value_bearing("9 200,00")     # money
        True
        >>> _is_value_bearing("ACME")         # prose, no digit
        False
    """
    return bool(text) and bool(_VALUE_BEARING_RE.search(text))


def _is_value_change(old_text: str, new_text: str) -> bool:
    """True when BOTH sides of a change are values (amount, date, id), not prose.

    Both sides must carry a digit, and neither may contain a real word (3+
    letters, currency words excepted). That keeps an address rewrite
    ("11 allée des Peupliers" -> "26 avenue de la Grande Armée") on the TEXT
    path: it contains digits, but it is not "a value that changed".

    Examples:
        >>> _is_value_change("1 200,00", "4 200,00")
        True
        >>> _is_value_change("19/08/1992", "19/08/ 1998")
        True
        >>> _is_value_change("92380 Garches", "75017 Paris")   # town names are words
        False
        >>> _is_value_change("Louise", "Philippe")              # no digit at all
        False
    """
    for text in (old_text, new_text):
        if not _is_value_bearing(text):
            return False
        for word in re.findall(r"[^\W\d_]{3,}", text):     # runs of 3+ letters
            if word.casefold() not in _CURRENCY_WORDS:
                return False
    return True


def visual_rows(words: list[tuple]) -> list[dict]:
    """Group a page's words into visual rows by vertical position.

    Why not PyMuPDF's own (block_no, line_no)? Because retyped text is a NEW
    text object: PyMuPDF puts it in a new block, so "Total TTC : 4 200,00 EUR"
    is split into an old block ("Total TTC :") and a new one ("4 200,00 EUR")
    even though a reader sees one line. Grouping by y keeps it one row in both
    revisions, which is what makes the two revisions comparable.

    Args:
        words: `page.get_text("words")` tuples (x0, y0, x1, y1, text, ...).

    Returns:
        Rows top to bottom, each `{"cy": float, "words": [...], "toks": [...]}`;
        words sorted left to right, `toks` their norm()-ed texts. Rows whose
        words all normalise to "" (stray punctuation) are dropped.

    Example:
        Words "Total"(y=290) "TTC"(y=290) "4"(y=289.5) "IBAN"(y=320) give two
        rows: ["total", "ttc", "4"] and ["iban"].
    """
    rows: list[dict] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0])):
        cy = (w[1] + w[3]) / 2.0
        if rows and abs(cy - rows[-1]["cy"]) <= ROW_Y_TOL:
            rows[-1]["words"].append(w)       # same row (anchored on its first word)
        else:
            rows.append({"cy": cy, "words": [w]})
    for row in rows:
        row["words"].sort(key=lambda w: w[0])
        row["toks"] = [norm(w[4]) for w in row["words"]]
    return [r for r in rows if any(r["toks"])]


def _closest_row(row: dict, candidates: list[dict]) -> dict | None:
    """The candidate row at (nearly) the same height as `row`, or None."""
    near = [c for c in candidates if abs(c["cy"] - row["cy"]) <= ROW_Y_TOL]
    if not near:
        return None
    return min(near, key=lambda c: abs(c["cy"] - row["cy"]))


def _changed_segments(old_row: dict, new_row: dict) -> list[tuple[list, list]]:
    """Which words of a row were REPLACED or DELETED between two revisions.

    `difflib.SequenceMatcher` aligns the two token lists like `diff` aligns two
    files, and returns opcodes: "equal", "replace", "delete", "insert". We
    keep only "replace" and "delete": an "insert" means the old text is still
    all there, with something added — a stacked correction whose original is
    still recoverable, which Signal 3 (paint order) owns, not this signal.

    Returns:
        A list of (old_words, new_words) pairs — the slices that differ,
        widened over adjacent number fragments (see _NUMBER_FRAGMENT_RE).

    Example:
        old ["total", "ttc", "1", "200,00", "eur"]
        new ["total", "ttc", "4", "200,00", "eur"]
        -> one segment: (["1", "200,00"], ["4", "200,00"])
    """
    a, b = old_row["toks"], new_row["toks"]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    segments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        # Widen left, then right, while the neighbour is the SAME number
        # fragment on both sides ("1" | "200,00" stays together as an amount).
        while (i1 > 0 and j1 > 0 and a[i1 - 1] == b[j1 - 1]
               and _NUMBER_FRAGMENT_RE.fullmatch(a[i1 - 1])):
            i1, j1 = i1 - 1, j1 - 1
        while (i2 < len(a) and j2 < len(b) and a[i2] == b[j2]
               and _NUMBER_FRAGMENT_RE.fullmatch(a[i2])):
            i2, j2 = i2 + 1, j2 + 1
        segments.append((old_row["words"][i1:i2], new_row["words"][j1:j2]))
    return segments


def _count_runs(toks: list[str], rows: list[dict]) -> int:
    """How many times the token run `toks` appears, consecutively, in `rows`."""
    n = len(toks)
    count = 0
    for row in rows:
        r = row["toks"]
        for start in range(len(r) - n + 1):
            if r[start:start + n] == toks:
                count += 1
    return count


def _survives_elsewhere(old_words: list[tuple], old_rows: list[dict],
                        new_rows: list[dict]) -> bool:
    """True if every copy of the old text is still somewhere in the NEW revision.

    This is the "moved, not changed" guard. Inserting one line (e.g. a company
    name under the client's name) pushes every following line down; at a fixed
    height, the phone number now sits where the town used to be. Row-by-row
    that looks like "town -> phone", but the town did not disappear — it is one
    row lower. Only text that lost a copy was really removed.

    We COUNT copies instead of asking "does it exist anywhere?": invoices
    repeat the total ("Total TTC" and "Net à payer"). A forger who edits only
    one of them leaves the other — the old text exists 2 times before and 1
    time after, so a copy was removed and the change is real.

    Examples:
        "0786460343": 1 copy before, 1 copy after (one row lower) -> True
        "1 200,00":   2 copies before, 1 copy after               -> False
    """
    toks = [t for t in (norm(w[4]) for w in old_words) if t]
    if not toks:
        return True                         # nothing meaningful was removed
    return _count_runs(toks, new_rows) >= _count_runs(toks, old_rows)


def _words_text(words: list[tuple]) -> str:
    """Join words back into display text: ["1", "200,00"] -> "1 200,00"."""
    return " ".join(w[4] for w in words).strip()


def _words_bbox(words: list[tuple], fallback: list[tuple]) -> tuple:
    """Union bbox of `words` (or of `fallback` when the change was a pure deletion)."""
    ws = words or fallback
    return (min(w[0] for w in ws), min(w[1] for w in ws),
            max(w[2] for w in ws), max(w[3] for w in ws))


def _revision_value_flag(old_text: str, new_text: str, page_num: int, confirmed: bool,
                         nw: tuple) -> Flag:
    """Build the Signal-4 flag, with a money delta when both sides parse as amounts.

    `confirmed` picks the code exactly like `_cover_and_retype_flag`: the HARD
    `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` iff the revision that changed
    the value was saved by an interactive editor (C2), else the SOFT
    `OVERLAY_REVISION_VALUE_CHANGED`.

    `nw` starts with the bbox (x0, y0, x1, y1) of where the changed value now
    sits, for preview highlighting: a bare 4-tuple (what `_diff_revision_page`
    passes) or a full `get_text("words")` tuple — only `nw[:4]` is read.
    """
    old_v, new_v = money_value(old_text), money_value(new_text)
    # money_value() reads the DAY of a date as an amount ("19/08/1992" -> 19.0),
    # which turned a birth-year change into "a change of +0.00". A date has no
    # money delta, so we skip it there.
    if _DATE_RE.search(old_text) or _DATE_RE.search(new_text):
        old_v = new_v = None
    details = {"page": page_num, "old_value": old_text.strip(), "new_value": new_text.strip(),
              "bbox": tuple(nw[:4])}
    if old_v is not None and new_v is not None:
        delta = new_v - old_v
        details["delta"] = delta
        change_clause = f" (a change of {delta:+,.2f})"
    else:
        change_clause = ""

    base_msg = (f"A value differs between two saved revisions of this PDF on "
                f"page {page_num + 1}: \"{old_text.strip()}\" in the earlier "
                f"revision became \"{new_text.strip()}\" in the later "
                f"one{change_clause}. The earlier value was only recoverable "
                f"by rolling back to a prior incremental revision.")

    if confirmed:
        code = "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"
        msg = (base_msg + " The revision that changed it was saved by an "
               "interactive editing tool, corroborating that this was a "
               "manual edit rather than template generation.")
    else:
        code = "OVERLAY_REVISION_VALUE_CHANGED"
        msg = base_msg

    return Flag(severity="critical", code=code, message=msg, details=details)


def _revision_text_flag(changes: list[dict], page_num: int) -> Flag:
    """One SOFT flag for all non-value text rewritten on a page between revisions.

    Names and addresses are not numbers, but changing the client's name on an
    invoice is exactly the edit a forger makes (real case: invoice_FRW, the
    payer's name and address replaced). We emit ONE flag per page, however
    many rows changed: three rewritten address lines are one edit, not three
    pieces of evidence (the evidence ledger would otherwise stack them).

    Always SOFT and "high", never `_CONFIRMED`: the module escalates to HARD
    only for value changes with an interactive-editor corroborator; a text
    rewrite stays a REVIEW-level finding.

    Args:
        changes: [{"old": str, "new": str, "bbox": tuple}, ...] in page order.
        page_num: 0-based page index.
    """
    shown = "; ".join(f"\"{c['old']}\" -> \"{c['new'] or '(removed)'}\""
                      for c in changes[:3])
    more = f" (and {len(changes) - 3} more)" if len(changes) > 3 else ""
    msg = (f"Text differs between two saved revisions of this PDF on page "
           f"{page_num + 1}: {shown}{more}. The earlier text was only "
           f"recoverable by rolling back to a prior incremental revision.")
    # "instances" is the multi-region shape src/preview.py understands: one
    # highlighted box per rewritten row, not only the first.
    details = {
        "page": page_num,
        "old_value": changes[0]["old"],
        "new_value": changes[0]["new"],
        "instances": [{"page": page_num, "bbox": c["bbox"], "old": c["old"], "new": c["new"]}
                      for c in changes],
    }
    return Flag(severity="high", code="OVERLAY_REVISION_TEXT_CHANGED",
                message=msg, details=details)


def _diff_revision_page(old_page, new_page, page_num: int, confirmed: bool) -> list[Flag]:
    """Diff one page between two revisions, row by row, and emit Signal-4 flags.

    Steps:
      1. Group both revisions' words into visual rows (visual_rows).
      2. Pair each old row with the new row at the same height.
         Identical token lists -> a STABLE row.
      3. Reflow guard: if fewer than MIN_STABLE_ROWS / MIN_STABLE_FRACTION of
         the old rows are stable, the page was re-laid-out, not edited —
         return nothing rather than guess.
      4. In each non-stable row, find the replaced/deleted words
         (_changed_segments), drop those that merely moved (_survives_elsewhere)
         or were placeholders ("0,00", "XXX" — template fill, not tampering).
      5. Value changes (amount, date) -> one critical flag each
         (_revision_value_flag, HARD with C2). Everything else -> one SOFT
         OVERLAY_REVISION_TEXT_CHANGED for the page.

    Example (corpus `redact_retype_incr`): 14 rows, 13 stable, row
    "Total TTC : 1 200,00 EUR" -> "Total TTC : 4 200,00 EUR" yields
    OVERLAY_REVISION_VALUE_CHANGED "1 200,00" -> "4 200,00" (delta +3,000.00).
    """
    old_rows = visual_rows(old_page.get_text("words"))
    new_rows = visual_rows(new_page.get_text("words"))
    if not old_rows:
        return []

    stable = 0
    differing: list[tuple[dict, dict]] = []
    for row in old_rows:
        twin = _closest_row(row, new_rows)
        if twin is None:
            continue                      # row vanished or moved: not counted as stable
        if twin["toks"] == row["toks"]:
            stable += 1
        else:
            differing.append((row, twin))

    if not differing:
        return []
    if stable < MIN_STABLE_ROWS or stable < MIN_STABLE_FRACTION * len(old_rows):
        return []                         # reflowed page: a row diff would be noise

    flags: list[Flag] = []
    text_changes: list[dict] = []
    for old_row, new_row in differing:
        for old_ws, new_ws in _changed_segments(old_row, new_row):
            old_text, new_text = _words_text(old_ws), _words_text(new_ws)
            if is_placeholder(old_text):
                continue                  # "0,00" / "XXX" filled in = template, not tamper
            if _survives_elsewhere(old_ws, old_rows, new_rows):
                continue                  # moved down/up a row, not removed
            bbox = _words_bbox(new_ws, old_ws)
            if _is_value_change(old_text, new_text):
                flags.append(_revision_value_flag(
                    old_text, new_text, page_num, confirmed, bbox))
            else:
                text_changes.append({"old": old_text, "new": new_text, "bbox": bbox})

    if text_changes:
        flags.append(_revision_text_flag(text_changes, page_num))
    return flags


def _signal_revision_value_changed(pdf_data, raw_bytes: bytes | None) -> list[Flag]:
    """Signal 4 — roll back through incremental PDF revisions and diff values.

    Skips entirely (returns no flags) when the file has at most one %%EOF —
    no incremental revisions exist to roll back to, so full-save forgeries
    are left to Signals 1-3 (this is a bonus path, not a replacement).

    For each consecutive revision pair, both truncations are opened
    independently; a truncation that fails to open is skipped (self-
    correcting against a stray "%%EOF" that appears inside a content stream
    rather than at a real version boundary, per `find_eof_positions`'s
    documented behaviour).

    Never raises: any failure at any step yields no flag for that revision
    pair rather than an exception (CLAUDE.md "never crash on malformed PDFs").
    """
    flags: list[Flag] = []
    if raw_bytes is None:
        return flags

    try:
        positions = find_eof_positions(raw_bytes)
    except Exception as exc:
        logger.warning("overlay: revision-diff EOF scan failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        return flags

    if len(positions) <= 1:
        return flags  # no incremental revisions -> nothing to roll back to

    for old_eof, new_eof in zip(positions, positions[1:]):
        old_doc = None
        new_doc = None
        try:
            try:
                old_doc = fitz.open(stream=raw_bytes[:old_eof], filetype="pdf")
            except Exception:
                continue  # not a real prior-version boundary — skip this pair
            try:
                new_doc = fitz.open(stream=raw_bytes[:new_eof], filetype="pdf")
            except Exception:
                continue

            old_meta, new_meta = old_doc.metadata or {}, new_doc.metadata or {}
            old_producer = f"{old_meta.get('producer') or ''} {old_meta.get('creator') or ''}".casefold()
            new_producer = f"{new_meta.get('producer') or ''} {new_meta.get('creator') or ''}".casefold()
            # C2 ONLY, exactly like `_analyze_page`'s escalation: a same-producer
            # (or non-editor) resave must never be reported as HARD.
            confirmed = (new_producer != old_producer) and is_interactive_editor(new_producer)

            for page_num in range(min(old_doc.page_count, new_doc.page_count)):
                try:
                    flags.extend(_diff_revision_page(
                        old_doc[page_num], new_doc[page_num], page_num, confirmed))
                except Exception as exc:
                    logger.warning("overlay: revision-diff page %d failed: %s", page_num, exc)
        except Exception as exc:
            logger.warning("overlay: revision-diff comparison failed for %s: %s",
                           getattr(pdf_data, "file_path", "?"), exc)
        finally:
            if old_doc is not None:
                try:
                    old_doc.close()
                except Exception:
                    pass
            if new_doc is not None:
                try:
                    new_doc.close()
                except Exception:
                    pass

    return flags


# =============================================================================
# SIGNAL 5 (Task 6) — IMAGE PATCH OVER LIVE TEXT
# =============================================================================
# Signals 1-3 only ever look at VECTOR fills from page.get_drawings() — but the
# design notes flag that the MOST COMMON real-world form of "cover a value and
# retype" is cheaper than drawing a rectangle: paste a small raster image (a
# screenshot crop, a solid-colour PNG) over the value instead. That never shows
# up in get_drawings(); it shows up in page.get_image_info(). The detection
# shape is otherwise identical to Signal 1 (opaque-fill-over-text): find a
# LOCALIZED image placement that CONTAINS a still-painted text span underneath.
#
# This signal is ALWAYS SOFT, even with editor-Producer corroboration (unlike
# Signals 1/3/4's HARD escalation path): get_image_info() has no seqno field
# comparable to get_texttrace()'s, so paint order relative to the text below it
# is not provable the way it is for vector fills — we only know the image and
# the text occupy the same rectangle, not which one a viewer would render on
# top. Treating that as HARD would be a fraud accusation the geometry can't
# back up, so it stays a heuristic finding (REVIEW), never a HARD veto.
#
# Paint order IS used in one direction, as a false-positive guard: when
# page.get_bboxlog() shows the text painted AFTER the image, the image is a
# background and the text is visible (_text_painted_over_image). It is not
# used to escalate: the bbox log matches text by position, which is good
# enough to excuse a label but not to accuse anyone.
def _localized_image_placements(page) -> list[dict]:
    """Image placements small enough to plausibly be a value-covering patch.

    Two exclusions, both false-positive guards:
      - area < LOCALIZED_FILL_MAX_AREA_FRAC * page area: rules out full-page
        scans and the OCR-sandwich pattern (the image IS the page background,
        not a patch placed over something).
      - width < FULL_WIDTH_BAND_FRAC * page width: rules out a full-width
        header/logo band, which spans the page horizontally like a banner
        rather than sitting over a single value.
    """
    placements = []
    page_rect = page.rect
    if page_rect.get_area() <= 0:
        return placements
    infos = page.get_image_info(xrefs=True)
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)
        if rect.is_empty or rect.is_infinite:
            continue
        if rect.get_area() >= LOCALIZED_FILL_MAX_AREA_FRAC * page_rect.get_area():
            continue  # full-page scan / OCR-sandwich background, not a patch
        if page_rect.width > 0 and rect.width >= FULL_WIDTH_BAND_FRAC * page_rect.width:
            continue  # full-width header/logo band
        placements.append(info)
    return placements


def _amount_char_xrange(span: dict) -> tuple[float, float] | None:
    """Horizontal [x0, x1] of the FIRST monetary amount inside a span's text.

    Reconstructs the span string exactly as span_text() does (same per-char
    decode + skip), keeping each character's x-extent from its get_texttrace
    bbox (char = (unicode, glyph, origin, bbox)), then returns the x-span of the
    amount_token() substring. None if the span carries no amount or the chars
    lack usable bboxes. Lets the embedded image path require the patch to cover
    the NUMBER itself, not merely the line the number sits on.
    """
    tok = amount_token(span_text(span))
    if tok is None:
        return None
    decoded = []                           # (char, x0, x1) aligned to span_text()
    for c in span.get("chars", []):
        try:
            ch = chr(c[0])
        except (TypeError, ValueError):
            continue                        # matches span_text()'s skip -> indices stay aligned
        bbox = c[3] if len(c) >= 4 else None
        decoded.append((ch, bbox[0], bbox[2]) if bbox else (ch, None, None))
    s = "".join(d[0] for d in decoded)
    start = s.find(tok)
    if start < 0:
        return None
    sub = decoded[start:start + len(tok)]
    xs0 = [d[1] for d in sub if d[1] is not None]
    xs1 = [d[2] for d in sub if d[2] is not None]
    if not xs0 or not xs1:
        return None
    return (min(xs0), max(xs1))


def _image_covers_amount(irect: fitz.Rect, xrange: tuple[float, float]) -> bool:
    """True if image `irect` horizontally covers >= CONTAINMENT_MIN of the
    amount's own x-range — i.e. the patch actually hides the number, rather than
    landing on a label or a neighbouring word of the same line."""
    ax0, ax1 = xrange
    aw = ax1 - ax0
    if aw <= 0:
        return False
    overlap = min(irect.x1, ax1) - max(irect.x0, ax0)
    return overlap / aw >= CONTAINMENT_MIN


def _find_embedded_image_cover(irect: fitz.Rect, spans: list[dict]):
    """Embedded path for Signal 5: the image covers only the AMOUNT PORTION of a
    wider span (label + amount + currency as one text object), so whole-span
    containment missed it. Return the covered span, or None.

    FP guards, in order — each mirrors the Signal-3 embedded path:
      - amount_token present            (the span carries a monetary amount)
      - money_value(tok) is not None   (reject dates / phones / ids)
      - not is_placeholder(tok)         ("0,00" -> benign template scaffolding)
      - _box_on_span_row                (image sits on the span's row, within it)
      - _image_covers_amount            (image covers the NUMBER's x-range) <- the
                                         discriminator vs a logo elsewhere on the line
    """
    for t in spans:
        tok = amount_token(span_text(t))
        if tok is None:
            continue
        if money_value(tok) is None:
            continue                        # a non-amount (date/phone/id) is never a hidden value
        if is_placeholder(tok):
            continue                        # covered a placeholder -> benign scaffolding
        if not _box_on_span_row(irect, fitz.Rect(t["bbox"])):
            continue
        xrange = _amount_char_xrange(t)
        if xrange is None:
            continue
        if _image_covers_amount(irect, xrange):
            return t
    return None


# Two rectangles are "the same placement" in the bbox log when every edge
# agrees within this many points (get_image_info and get_bboxlog compute the
# same transform, so in practice they agree exactly; 1 pt absorbs rounding).
_SAME_RECT_TOL = 1.0


def _paint_log(page) -> list | None:
    """The page's drawing operations IN PAINT ORDER, or None if unavailable.

    `page.get_bboxlog()` returns one (kind, rect) entry per operation, in the
    order the content stream paints them: "fill-image", "fill-text",
    "fill-path", "ignore-text" (invisible text), ... That order is what tells
    a cover from a background: whatever is painted LATER is on top.
    """
    try:
        return page.get_bboxlog()
    except Exception as exc:
        logger.warning("overlay: get_bboxlog failed: %s", exc)
        return None


_TEXT_PAINT_KINDS = ("fill-text", "stroke-text")
_FLOAT32 = struct.Struct("f")


def _as_float32(value: float) -> float:
    """Round a Python float the way MuPDF stores it (C `float`, 32 bits).

    MuPDF compares coordinates as 32-bit floats. A centre computed in Python
    (64 bits) can sit a hair inside a box that the 32-bit value sits exactly
    on the edge of — and the edge is outside. Rounding first keeps this
    module's answer identical to fitz.Rect.contains(), which it replaces.

    Example:
        _as_float32(9.9999999999) -> 10.0
    """
    try:
        return _FLOAT32.unpack(_FLOAT32.pack(value))[0]
    except OverflowError:               # beyond float32 range: C gives +/-inf
        return float("inf") if value > 0 else float("-inf")


def _last_image_op(log: list, irect) -> int | None:
    """Index of the LAST paint operation that draws the image at `irect`.

    The same image can be painted several times; only the last one decides
    what ends up on top. None when the log holds no such operation.
    """
    last = None
    for index, (kind, r) in enumerate(log):
        if kind == "fill-image" and all(abs(a - b) <= _SAME_RECT_TOL for a, b in zip(r, irect)):
            last = index
    return last


def _first_text_op(log: list, span: dict) -> int | None:
    """Index of the FIRST text operation whose box contains the span's centre.

    Depends on the span only, not on any image, so the caller computes it once
    per span and reuses it for every image on the page.

    "Contains" follows MuPDF: left/top edges are inside, right/bottom edges
    are outside (x0 <= x < x1), compared as 32-bit floats.
    """
    x0, y0, x1, y1 = span["bbox"]
    cx = _as_float32((x0 + x1) / 2)
    cy = _as_float32((y0 + y1) / 2)
    for index, (kind, r) in enumerate(log):
        if (kind in _TEXT_PAINT_KINDS
                and _as_float32(r[0]) <= cx < _as_float32(r[2])
                and _as_float32(r[1]) <= cy < _as_float32(r[3])):
            return index                # the log is in paint order: first hit is the minimum
    return None


def _text_painted_over_image(log: list | None, irect: fitz.Rect, span: dict) -> bool:
    """True if `span` is painted AFTER every drawing of the image at `irect`.

    Then the image is a background (a shaded box, a frame) and the text sits
    on top of it, fully visible — not hidden under a patch. Real case: a
    genuine Avis d'impôt 2023 draws each information box as an opaque image
    (op 27) and then writes "Numéro fiscal" inside it (op 28); Signal 5 used
    to report the label as "covered".

    Returns False whenever the order cannot be established (no log, image or
    text not found in it): the caller then keeps the cautious behaviour.

    Example:
        log = [("fill-image", box), ("fill-text", label)]  -> True  (background)
        log = [("fill-text", label), ("fill-image", box)]  -> False (covered)
    """
    if not log:
        return False
    last_image = _last_image_op(log, irect)
    first_text = _first_text_op(log, span)
    if last_image is None or first_text is None:
        return False
    return first_text > last_image


def _signal_image_over_text(page, spans: list[dict]) -> list[dict]:
    """Signal 5 — a localized IMAGE placement covers still-present live text.

    For each localized image placement I, look for a painted span T (type 0/2,
    so invisible OCR text is never mistaken for hidden content) with
    containment(T.bbox, I.bbox) >= CONTAINMENT_MIN — the image sits directly
    over text that PyMuPDF can still extract. PLACEHOLDER GUARD: symmetric with
    Signals 1/3 — a covered placeholder ("0,00", "XXX") is benign template
    scaffolding, not hidden content, so it is skipped.

    EMBEDDED FALLBACK: when no whole span sits inside the image, the amount may
    be embedded in a WIDER span and the patch covers only the number
    ("Gesamtbetrag: 571,04 €"); _find_embedded_image_cover matches on the
    amount's own x-range. All five image-patch misses on the real-invoice corpus
    were this shape.
    """
    hits = []
    log = None
    for info in _localized_image_placements(page):
        irect = fitz.Rect(info["bbox"])
        ibox = tuple(irect)                       # plain tuple for the cheap pre-reject
        if log is None:
            log = _paint_log(page) or []          # computed once, only if needed
            # Where each span first appears in the paint log. It does not
            # depend on the image, so it is worked out once per page instead
            # of once per (image, span) pair — that product was the slowest
            # loop of the whole analysis (45 s on a 3-page invoice).
            first_text = {id(t): _first_text_op(log, t) for t in spans} if log else {}
        # Text painted ON TOP of the image is visible, not covered: drop it
        # before looking for a hidden span (same rule as
        # _text_painted_over_image, with both halves precomputed).
        last_image = _last_image_op(log, irect) if log else None
        if log and last_image is None and not info.get("xref"):
            # get_image_info also reports a SHADING (a gradient, operator
            # "sh") as an image with no xref, but the paint log records it as
            # "fill-shade", never "fill-image". A gradient is a background,
            # not a pasted patch: Costumes FA5237's grey header gradient,
            # painted before every line of text, was read as covering the
            # company name. A real inline image (also xref 0) is "fill-image".
            continue
        if last_image is None:
            under = list(spans)                   # order unknown: stay cautious
        else:
            under = [t for t in spans
                     if first_text[id(t)] is None or first_text[id(t)] <= last_image]
        covered = None
        for t in under:
            if boxes_disjoint(t["bbox"], ibox):
                continue                   # nowhere near the image: containment is 0
            if containment(fitz.Rect(t["bbox"]), irect) < CONTAINMENT_MIN:
                continue
            if is_placeholder(span_text(t)):
                continue                   # covered placeholder = benign template scaffolding
            covered = t
            break                          # one report per image is enough
        if covered is None:
            covered = _find_embedded_image_cover(irect, under)
        if covered is not None:
            hits.append({"image": info, "covered": covered})
    return hits


def _image_over_text_flag(hit: dict, page_num: int) -> Flag:
    """Build the Signal-5 flag. NEVER HARD — see the module comment above
    `_localized_image_placements` for why image paint order can't back a HARD code.

    `OVERLAY_IMAGE_OVER_VALUE` (critical) when the covered text is value-bearing
    (money/date/IBAN/SIRET/SIREN/VAT/bare number, via the same `_is_value_bearing`
    helper Signal 4 uses); otherwise the lower-severity `OVERLAY_IMAGE_OVER_TEXT`.
    """
    text = span_text(hit["covered"]).strip()
    details = {"page": page_num, "covered": text, "bbox": tuple(hit["image"]["bbox"])}
    if _is_value_bearing(text):
        return Flag(
            severity="critical",
            code="OVERLAY_IMAGE_OVER_VALUE",
            message=(f"An image on page {page_num + 1} is placed over a "
                     f"still-present value (\"{text}\"), which remains "
                     "recoverable underneath it."),
            details=details,
        )
    # HIGH — deliberately BELOW the value-bearing branch. This was briefly
    # bumped to critical (2026-07) to close a band-level miss, then reverted
    # after adversarial review: the "0 legit corpus FPs" justification only
    # held because the corpus has no stamped/signed documents — a legit
    # "PAID"/seal/signature IMAGE over live text is common and fires this
    # exact branch. FP-first: a lone hit stays GENUINE (92) with the flag
    # visible in the report; covering an actual VALUE stays critical above.
    return Flag(
        severity="high",
        code="OVERLAY_IMAGE_OVER_TEXT",
        message=(f"An image on page {page_num + 1} is placed over "
                 f"still-present text (\"{text}\"), which remains recoverable "
                 "underneath it."),
        details=details,
    )


# =============================================================================
# FLAG CONSTRUCTION
# =============================================================================
def _cover_and_retype_flag(hit: dict, page_num: int, confirmed: bool) -> Flag:
    """Build the cover-and-retype flag, with a value delta when possible.

    `confirmed` picks the code: `OVERLAY_COVER_AND_RETYPE_CONFIRMED` (HARD) iff
    an interactive-editor Producer corroborates the geometry (C2), else the
    plain `OVERLAY_COVER_AND_RETYPE` (SOFT). Exactly one of the two is ever
    emitted for a given hit — never both.
    """
    under_txt = span_text(hit["under"]).strip()
    over_txt = span_text(hit["over"]).strip()
    details = {"page": page_num, "covered": under_txt, "replacement": over_txt,
              "bbox": tuple(hit["fill"]["rect"])}
    old_v, new_v = money_value(under_txt), money_value(over_txt)
    if old_v is not None and new_v is not None:
        delta = new_v - old_v
        details["delta"] = delta
        change_clause = f" (a change of {delta:+,.2f})"
    else:
        change_clause = ""

    base_msg = (f"A value appears to have been covered and retyped on page "
                f"{page_num + 1}: \"{under_txt}\" was hidden and \"{over_txt}\" "
                f"placed over it{change_clause}.")

    if confirmed:
        code = "OVERLAY_COVER_AND_RETYPE_CONFIRMED"
        msg = (base_msg + " The document's Producer/Creator metadata shows it "
               "was saved by an interactive editing tool, corroborating that "
               "this was a manual edit rather than template generation.")
    else:
        code = "OVERLAY_COVER_AND_RETYPE"
        msg = base_msg

    return Flag(severity="critical", code=code, message=msg, details=details)


def _analyze_page(page, page_angle: float, page_num: int, corroboration: dict) -> list[Flag]:
    """Run the three signals on one page and turn the hits into flags."""
    extracted = _extract_page(page, page_angle)
    spans, fills = extracted["spans"], extracted["fills"]

    flags: list[Flag] = []

    # Signal 3 first: it consumes the fills/spans it explains so Signals 1 and 2
    # don't re-report the same cover-and-retype event under a weaker code.
    hits3, consumed_spans, consumed_fills = _signal_cover_and_retype(fills, spans, page_angle)
    # C2 ONLY. Do NOT `or` in corroboration["has_incremental_updates"] here —
    # that reintroduces the legit-multisession false positive the Task-1
    # design review caught (see module-level comment above `_corroboration`).
    confirmed = corroboration["producer_is_editor"]
    for hit in hits3:
        flags.append(_cover_and_retype_flag(hit, page_num, confirmed))

    for hit in _signal_opaque_fill_over_text(fills, spans,
                                             consumed_spans, consumed_fills):
        t = span_text(hit["hidden"]).strip()
        flags.append(Flag(
            # critical (not high): a lone geometry signal must reach the design's
            # REVIEW floor ("geometry alone -> REVIEW", phase2 design §4/§8). At
            # high (weight 1.5) with the -4.0 prior a lone occurrence scored
            # ~92/GENUINE; at critical (3.0) it lands ~73/REVIEW, consistent with
            # Signal 3. Still SOFT — only editor-Producer provenance escalates to
            # HARD. Fires 0x on the legit corpus, so it never demotes a legit doc.
            severity="critical", code="OVERLAY_OPAQUE_FILL_OVER_TEXT",
            message=(f"An opaque box on page {page_num + 1} covers still-present "
                     f"text (\"{t}\"), which remains recoverable underneath."),
            details={"page": page_num, "hidden": t, "bbox": tuple(hit["fill"]["rect"])},
        ))

    for hit in _signal_conflicting_text(spans, page_angle, consumed_spans):
        ta, tb = span_text(hit["a"]).strip(), span_text(hit["b"]).strip()
        union = fitz.Rect(hit["a"]["bbox"]) | fitz.Rect(hit["b"]["bbox"])
        flags.append(Flag(
            # critical (not high): same REVIEW-floor rationale as
            # OVERLAY_OPAQUE_FILL_OVER_TEXT above — two conflicting values at one
            # position is a smoking gun that must not read GENUINE. SOFT; provenance
            # still required for HARD. Fires 0x on the legit corpus.
            severity="critical", code="OVERLAY_CONFLICTING_TEXT",
            message=(f"Two conflicting values overlap at the same position on "
                     f"page {page_num + 1}: \"{ta}\" and \"{tb}\"."),
            details={"page": page_num, "values": [ta, tb], "bbox": tuple(union)},
        ))

    # Signal 5 (Task 6): raster version of the cover-and-retype attack. Its own
    # try/except so a get_image_info() hiccup on this page can't sink the
    # geometry signals above, which already ran successfully.
    try:
        for hit in _signal_image_over_text(page, spans):
            flags.append(_image_over_text_flag(hit, page_num))
    except Exception as exc:
        logger.warning("overlay: image-over-text signal failed on page %d: %s",
                       page_num, exc)

    for signal in (_signal_shadow_duplicate_text, _signal_freehand_stroke):
        try:
            flags.extend(signal(page, page_num))
        except Exception as exc:
            logger.warning("overlay: %s failed on page %d: %s",
                           signal.__name__, page_num, exc)

    return flags


# Shortest line worth comparing: shorter strings ("Total", "EUR") repeat
# honestly all over an invoice.
SHADOW_MIN_CHARS = 8
# Below this offset, a second drawing of the same text is the classic
# "fake bold" trick (text drawn twice, a fraction of a point apart).
SHADOW_MIN_OFFSET_PT = 1.0


def _signal_shadow_duplicate_text(page, page_num: int) -> list[Flag]:
    """The same line of text drawn twice, overlapping, a few points apart.

    Seen on a test invoice edited online: the date/amount block was pasted a
    second time 10 pt lower-right, on top of the original. Generators repeat
    a line in other places (headers, footers) but never on top of itself.
    Measured 2026-11-12: 0 hits in the 171 non-suspicious local samples.

    Example:
        "Total TTC ... 120,00 EUR" at (72, 218) and again at (82, 228)
        -> one OVERLAY_DUPLICATED_TEXT flag
    """
    seen: dict[str, list[fitz.Rect]] = {}
    hits = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if len(text) < SHADOW_MIN_CHARS:
                    continue
                rect = fitz.Rect(span["bbox"])
                for other in seen.get(text, []):
                    dx, dy = abs(rect.x0 - other.x0), abs(rect.y0 - other.y0)
                    offset = max(dx, dy)
                    # Same column, one line height lower: the next row of a
                    # table repeating its label (EDF: "Electricité Période
                    # unique" for two periods, 8.2 pt apart in a 7 pt font).
                    # Line boxes are taller than the line pitch, so they touch.
                    if dx <= SHADOW_MIN_OFFSET_PT and dy >= 0.9 * span.get("size", 0):
                        continue
                    if rect.intersects(other) and offset > SHADOW_MIN_OFFSET_PT:
                        hits.append({"text": text[:60], "offset_pt": round(offset, 1)})
                seen.setdefault(text, []).append(rect)
    if not hits:
        return []
    return [Flag(
        severity="high",
        code="OVERLAY_DUPLICATED_TEXT",
        message=(f"Text drawn twice on top of itself on page {page_num + 1}: "
                 f"'{hits[0]['text']}'"),
        details={"page": page_num + 1, "lines": hits[:5], "total": len(hits),
                 "explanation": "The same line appears a second time, shifted by a "
                                "few points over the original: pasted content."},
    )]


# A pen stroke is a stroked, open path of many Bézier curves. Rounded
# rectangles have 4 curves and are closed; logos are filled, not stroked.
FREEHAND_MIN_CURVES = 6


def _signal_freehand_stroke(page, page_num: int) -> list[Flag]:
    """A hand-drawn pen stroke flattened into the page content.

    When an online editor "flattens" an ink annotation, the stroke stops
    being an annotation and becomes page drawing — the annotation check no
    longer sees it. Measured 2026-11-12: 0 hits in the 171 non-suspicious
    local samples; 2 test invoices drawn on online.
    """
    strokes = []
    for d in page.get_drawings():
        ops = [item[0] for item in d.get("items", [])]
        if d.get("type") == "s" and not d.get("closePath") \
                and ops.count("c") >= FREEHAND_MIN_CURVES:
            r = d["rect"]
            strokes.append({"rect": [round(r.x0), round(r.y0), round(r.x1), round(r.y1)],
                            "curves": ops.count("c")})
    if not strokes:
        return []
    return [Flag(
        severity="high",
        code="OVERLAY_FREEHAND_DRAWING",
        message=f"Hand-drawn stroke on page {page_num + 1}",
        details={"page": page_num + 1, "strokes": strokes[:5],
                 "explanation": "A free pen stroke is part of the page: someone "
                                "drew on the document and flattened it."},
    )]


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def analyze_overlay(pdf_data) -> ModuleResult:
    """Detect opaque-box cover-and-retype tampering from PDF structure alone.

    Geometry (Signals 1-3) is heuristic and stays SOFT on its own; it escalates
    to a HARD `_CONFIRMED` code only when corroborated by an interactive-editor
    Producer/Creator (C2 — see `_corroboration`). Never raises: a malformed /
    encrypted / textless PDF yields a partial ModuleResult with reduced
    confidence instead of an exception (CLAUDE.md).
    """
    flags: list[Flag] = []
    confidence = 1.0

    if not getattr(pdf_data, "extraction_ok", True):
        # Extraction already failed upstream; nothing to analyse. The analyzer
        # short-circuits this path in production, but a direct caller must still
        # get a well-formed, low-confidence result (never a crash).
        return ModuleResult(module="overlay", flags=[], score=100, confidence=0.5)

    # Read the raw bytes ONCE, guarded: used by _corroboration for the
    # incremental-update count (context only, not an escalation driver) AND by
    # Signal 4's revision rollback (the actual byte prefixes it reopens).
    raw_bytes = None
    try:
        with open(pdf_data.file_path, "rb") as fh:
            raw_bytes = fh.read()
    except Exception as exc:
        logger.warning("overlay: raw byte read failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)

    corroboration = _corroboration(pdf_data, raw_bytes)

    doc = None
    try:
        doc = fitz.open(pdf_data.file_path)
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                flags.extend(_analyze_page(page, _page_angle(page), page_num, corroboration))
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

    # Signal 4 (Task 5): bonus path for the "deleted, not covered" tamper
    # variant. Belt-and-suspenders try/except even though the function itself
    # never raises — a rollback/diff failure must yield no revision flag, not
    # an exception.
    try:
        flags.extend(_signal_revision_value_changed(pdf_data, raw_bytes))
    except Exception as exc:
        logger.warning("overlay: revision-diff signal failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    score = max(0, score)

    return ModuleResult(module="overlay", flags=flags, score=score, confidence=confidence)
