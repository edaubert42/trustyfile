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
import fitz

from src.models import Flag, ModuleResult
from src.modules.structure import find_eof_positions
from src.modules.span_conflict import (
    HORIZONTAL_TOL_DEG, SIZE_MATCH_FRAC, INTERACTIVE_EDITOR_PRODUCERS,
    span_text, char_class, norm, amount_token, is_horizontal,
    is_placeholder, page_angle as _page_angle,
    pair_is_conflict as _pair_is_conflict,
    money_value as _money_value,
)

logger = logging.getLogger(__name__)

OPAQUE_FILL_MIN_OPACITY = 0.90
BG_COLOR_MATCH_TOL = 0.06
CONTAINMENT_MIN = 0.8
CONFLICT_IOU_MIN = 0.6
LOCALIZED_FILL_MAX_AREA_FRAC = 0.10
LINE_HEIGHT_LO, LINE_HEIGHT_HI = 0.5, 2.5
MAX_FILL_ASPECT = 25.0
FULL_WIDTH_BAND_FRAC = 0.85  # image width >= this * page width => header/logo band, not a patch


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
        inside = [s for s in spans if containment(fitz.Rect(s["bbox"]), frect) >= CONTAINMENT_MIN]
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
            if _pair_is_conflict(a, b, page_angle):
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
            if _amounts_conflict(a, b, page_angle):
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


def _amounts_conflict(a: dict, b: dict, page_angle: float) -> bool:
    """Conflict test on the AMOUNT portion of two spans (for the embedded path).

    True when both spans carry a horizontal, same-size monetary amount, the
    covered (A) amount is a real value (not a placeholder), and the two amounts
    differ NUMERICALLY. Comparing parsed numeric values (via _money_value) — not
    whole-span text — is what lets a label-wrapped original ("Gesamtbetrag:
    571,04 €") conflict with a bare retype ("690,25"), while REJECTING dates /
    phone numbers / ids (which _money_value returns None for) and treating two
    equal amounts written in different styles ("1.200,00" vs "1 200,00") as the
    SAME value, not a spurious change."""
    at, bt = span_text(a), span_text(b)
    av, bv = _money_value(at), _money_value(bt)
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
    if max(sa, sb) <= 0 or abs(sa - sb) / max(sa, sb) > SIZE_MATCH_FRAC:
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
            producer_is_editor = any(p in haystack for p in INTERACTIVE_EDITOR_PRODUCERS)
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
TOKEN_CENTRE_TOL = 3.0  # pt; PyMuPDF word-splitter jitter between independent revisions

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


def _word_centre(word: tuple) -> tuple[float, float]:
    x0, y0, x1, y1 = word[0], word[1], word[2], word[3]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _word_line_key(word: tuple) -> tuple:
    return (word[5], word[6])  # (block_no, line_no)


def _match_revision_words(old_words: list[tuple], new_words: list[tuple],
                          tol: float = TOKEN_CENTRE_TOL) -> list[tuple]:
    """Pair each OLD word with its counterpart at (nearly) the same position
    in the NEW revision, by bbox-centre proximity.

    When several NEW words fall within tolerance of one OLD word — e.g. an
    incremental save that APPENDS a corrected value on top of an original
    that is still physically present in the content stream — the candidate
    with IDENTICAL text is preferred. That "extra layer stacked at the same
    spot, original still recoverable" pattern is Signal 3's job (paint-order
    geometry); Signal 4 only needs to speak up when the OLD token has NO
    surviving twin at all (it was actually deleted, e.g. via redaction). This
    preference is what keeps the honest multi-session-correction fixture from
    being mis-paired into a spurious change.
    """
    used_new: set[int] = set()
    pairs = []
    for ow in old_words:
        ocx, ocy = _word_centre(ow)
        candidates = []
        for idx, nw in enumerate(new_words):
            if idx in used_new:
                continue
            ncx, ncy = _word_centre(nw)
            if abs(ocx - ncx) <= tol and abs(ocy - ncy) <= tol:
                candidates.append((idx, nw))
        if not candidates:
            continue
        exact = [c for c in candidates if norm(c[1][4]) == norm(ow[4])]
        idx, nw = exact[0] if exact else candidates[0]
        used_new.add(idx)
        pairs.append((ow, nw))
    return pairs


def _neighbours_stable(ow: tuple, pairs_by_old_id: dict, old_words_by_line: dict) -> bool:
    """Bbox-STABILITY guard: true iff every OTHER old word on ow's line has a
    matched NEW counterpart with IDENTICAL text.

    A genuine isolated tamper changes one value while everything around it
    (labels, other digits, punctuation) stays exactly the same. A reflowed
    revision (different layout, unrelated content) instead tends to leave
    neighbours unmatched (no candidate within tolerance at all) or altered —
    either of which fails this guard, preventing us from manufacturing a
    change out of two tokens that only coincidentally landed near each other.
    """
    key = _word_line_key(ow)
    siblings = [w for w in old_words_by_line.get(key, []) if w is not ow]
    if not siblings:
        return True  # isolated token on its own line — nothing to corroborate against
    for sib in siblings:
        match = pairs_by_old_id.get(id(sib))
        if match is None:
            return False  # a neighbour has no surviving counterpart -> unsafe to trust
        _, nw = match
        if norm(nw[4]) != norm(sib[4]):
            return False  # a neighbour also changed -> not an isolated, targeted edit
    return True


def _revision_value_flag(old_text: str, new_text: str, page_num: int, confirmed: bool,
                         nw: tuple) -> Flag:
    """Build the Signal-4 flag, with a money delta when both sides parse as amounts.

    `confirmed` picks the code exactly like `_cover_and_retype_flag`: the HARD
    `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` iff the revision that changed
    the value was saved by an interactive editor (C2), else the SOFT
    `OVERLAY_REVISION_VALUE_CHANGED`.

    `nw` is the NEW revision's word tuple from `page.get_text("words")`
    (x0,y0,x1,y1,text,block_no,line_no,word_no) — its first 4 elements are
    the bbox of where the changed value now sits, for preview highlighting.
    """
    old_v, new_v = _money_value(old_text), _money_value(new_text)
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


def _diff_revision_page(old_page, new_page, page_num: int, confirmed: bool) -> list[Flag]:
    """Diff one page's words between two revisions and emit Signal-4 flags."""
    old_words = old_page.get_text("words")
    new_words = new_page.get_text("words")

    pairs = _match_revision_words(old_words, new_words)
    pairs_by_old_id = {id(ow): (ow, nw) for ow, nw in pairs}
    old_words_by_line: dict = {}
    for ow in old_words:
        old_words_by_line.setdefault(_word_line_key(ow), []).append(ow)

    flags: list[Flag] = []
    for ow, nw in pairs:
        old_text, new_text = ow[4], nw[4]
        if norm(old_text) == norm(new_text):
            continue                                          # unchanged
        if not (_is_value_bearing(old_text) or _is_value_bearing(new_text)):
            continue                                          # prose, not a value
        if not _neighbours_stable(ow, pairs_by_old_id, old_words_by_line):
            continue                                          # unsafe pairing (reflow)
        flags.append(_revision_value_flag(old_text, new_text, page_num, confirmed, nw))
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
            confirmed = (new_producer != old_producer) and any(
                p in new_producer for p in INTERACTIVE_EDITOR_PRODUCERS)

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
      - _money_value(tok) is not None   (reject dates / phones / ids)
      - not is_placeholder(tok)         ("0,00" -> benign template scaffolding)
      - _box_on_span_row                (image sits on the span's row, within it)
      - _image_covers_amount            (image covers the NUMBER's x-range) <- the
                                         discriminator vs a logo elsewhere on the line
    """
    for t in spans:
        tok = amount_token(span_text(t))
        if tok is None:
            continue
        if _money_value(tok) is None:
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
    image_ops = [i for i, (kind, r) in enumerate(log)
                 if kind == "fill-image"
                 and all(abs(a - b) <= _SAME_RECT_TOL for a, b in zip(r, irect))]
    sbox = fitz.Rect(span["bbox"])
    centre = fitz.Point((sbox.x0 + sbox.x1) / 2, (sbox.y0 + sbox.y1) / 2)
    text_ops = [i for i, (kind, r) in enumerate(log)
                if kind in ("fill-text", "stroke-text") and fitz.Rect(r).contains(centre)]
    if not image_ops or not text_ops:
        return False
    return min(text_ops) > max(image_ops)


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
        if log is None:
            log = _paint_log(page) or []          # computed once, only if needed
        # Text painted ON TOP of the image is visible, not covered: drop it
        # before looking for a hidden span (see _text_painted_over_image).
        under = [t for t in spans if not _text_painted_over_image(log, irect, t)]
        covered = None
        for t in under:
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
    old_v, new_v = _money_value(under_txt), _money_value(over_txt)
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

    return flags


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
