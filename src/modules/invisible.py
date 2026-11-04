"""
Invisible-text mismatch detector (Module: invisible) — phase 7.

Detects Tr-mode-3 ("paint nothing") text that conflicts with, or hides from,
the visible page. See docs/superpowers/specs/2026-07-12-invisible-text-
mismatch-design.md. Invisible text is NOT suspicious per se (legit OCR
sandwich) — only conflicts/injections on digitally-authored pages are flagged.
"""
import logging
import re

import fitz

from src.models import Flag, ModuleResult
from src.modules.span_conflict import (
    span_text, amount_token, is_placeholder, pair_is_conflict, page_angle,
    money_value, is_interactive_editor,
)

logger = logging.getLogger(__name__)

SEVERITY_POINTS = {"low": 5, "medium": 15, "high": 30, "critical": 50}

# A page is analysed only when painted text exists AND invisible text is a
# minority of its text area — pure OCR sandwiches and hybrid scans are the
# own-OCR-reconciliation module's territory, not this one's (spec, gate).
INVISIBLE_DOMINANCE_MAX = 0.50
ROW_OVERLAP_MIN = 0.5

# IBAN shape after whitespace strip + uppercase (spec).
_IBAN_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}")

# Producers whose invisible layers may legitimately touch mixed pages
# (re-OCR tools). Damper: injection suppressed, conflict demoted (Task 6).
OCR_TOOL_PRODUCERS = ("ocrmypdf", "tesseract", "abbyy", "readiris",
                      "finereader", "naps2")


def _split_spans(page) -> tuple[list, list]:
    """Partition texttrace spans into (painted, invisible).

    Painted = fill/fill+stroke (types 0, 2); invisible = Tr 3 (type 3).
    Other types (stroke-only, clip variants) are ignored, matching overlay's
    convention. Verified on this PyMuPDF: render_mode=3 (Tr 3, "paint
    nothing") surfaces as get_texttrace() type==3, so it is covered. Tr
    modes 4-7 (clip variants) are NOT covered by this split -- a documented
    coverage gap, not FP-relevant (missing a clip-mode invisible span can
    only under-detect, never raise a false positive).
    """
    painted, invisible = [], []
    for s in page.get_texttrace():
        t = s.get("type")
        if t in (0, 2):
            painted.append(s)
        elif t == 3:
            invisible.append(s)
    return painted, invisible


def _area(spans: list) -> float:
    total = 0.0
    for s in spans:
        r = fitz.Rect(s["bbox"])
        total += max(0.0, r.width) * max(0.0, r.height)
    return total


def _is_digital_native(painted: list, invisible: list) -> bool:
    """Gate: painted text exists and invisible text is an area minority."""
    if not painted:
        return False
    if not invisible:
        return True
    total = _area(painted) + _area(invisible)
    return total > 0 and (_area(invisible) / total) < INVISIBLE_DOMINANCE_MAX


def _value_token(text: str) -> str | None:
    """A monetary amount or IBAN-shaped token in `text`, or None."""
    tok = amount_token(text)
    if tok:
        return tok
    m = _IBAN_RE.search("".join(text.split()).upper())
    return m.group() if m else None


def _rows_overlap(b1: fitz.Rect, b2: fitz.Rect) -> bool:
    """Vertical overlap >= ROW_OVERLAP_MIN of the smaller span height."""
    overlap = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    smaller = min(b1.height, b2.height)
    return smaller > 0 and (overlap / smaller) >= ROW_OVERLAP_MIN


def _near_horizontally(inv: fitz.Rect, pt: fitz.Rect, painted_span: dict) -> bool:
    """Horizontal overlap, or a gap under the painted span's glyph advance."""
    gap = max(pt.x0 - inv.x1, inv.x0 - pt.x1)
    if gap <= 0:
        return True
    nchars = max(1, len(painted_span.get("chars", [])))
    advance = pt.width / nchars
    return gap < advance


def _same_value(hidden_text: str, visible_text: str) -> bool:
    """True when both texts parse to the SAME monetary amount.

    "1 025,00" (French comma-decimal) and "1025.00" (period-decimal) are the
    same value in two representations -- norm() does not canonicalize decimal
    separators, so pair_is_conflict() alone would call this a conflict. Only
    skip when BOTH sides parse to a non-None float and those floats are
    equal; if either side doesn't parse as money (e.g. an IBAN), this falls
    through False and the existing conflict logic is unaffected.
    """
    hv, vv = money_value(hidden_text), money_value(visible_text)
    return hv is not None and hv == vv


def _find_conflicts(painted: list, invisible: list, angle: float) -> list[dict]:
    """Invisible value tokens that disagree with painted text on their row."""
    hits = []
    for inv in invisible:
        hidden = span_text(inv)
        if not _value_token(hidden) or is_placeholder(hidden):
            continue
        ibox = fitz.Rect(inv["bbox"])
        for p in painted:
            pbox = fitz.Rect(p["bbox"])
            if not _rows_overlap(ibox, pbox):
                continue
            if not _near_horizontally(ibox, pbox, p):
                continue
            visible = span_text(p)
            if pair_is_conflict(inv, p, angle) and not _same_value(hidden, visible):
                hits.append({"hidden": hidden.strip(),
                             "visible": visible.strip(),
                             "bbox": tuple(inv["bbox"])})
                break                      # one conflict per invisible span
    return hits


def _find_injections(painted: list, invisible: list) -> list[dict]:
    """Invisible value-bearing spans on rows with NO painted text at all.

    Distinct from `_find_conflicts`: a conflict needs a painted counterpart
    sharing the row; an injection is a value token hidden in a blank area
    where nothing is painted at all (whitespace injection).
    """
    hits = []
    for inv in invisible:
        hidden = span_text(inv)
        if not _value_token(hidden) or is_placeholder(hidden):
            continue
        ibox = fitz.Rect(inv["bbox"])
        if any(_rows_overlap(ibox, fitz.Rect(p["bbox"])) for p in painted):
            continue                      # row has visible text -> Task 3's job
        hits.append({"hidden": hidden.strip(), "bbox": tuple(inv["bbox"])})
    return hits


def _producer_haystack(pdf_data) -> str:
    """Lower-cased Producer+Creator, '' on any failure (never raises)."""
    try:
        doc = fitz.open(pdf_data.file_path)
        try:
            meta = doc.metadata or {}
            return (f"{meta.get('producer') or ''} "
                    f"{meta.get('creator') or ''}").casefold()
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("invisible: metadata read failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        return ""


def analyze_invisible(pdf_data) -> ModuleResult:
    """Flag invisible text that conflicts with or hides from the visible page.

    Never raises: malformed/encrypted/textless PDFs yield a partial
    ModuleResult with reduced confidence instead of an exception (CLAUDE.md).
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="invisible", flags=[], score=100,
                            confidence=0.5)

    flags: list[Flag] = []
    confidence = 1.0
    conflict_instances: list[dict] = []
    injection_instances: list[dict] = []

    haystack = _producer_haystack(pdf_data)
    producer_is_editor = is_interactive_editor(haystack)
    producer_is_ocr = any(p in haystack for p in OCR_TOOL_PRODUCERS)

    doc = None
    try:
        doc = fitz.open(pdf_data.file_path)
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                painted, invisible = _split_spans(page)
                if not _is_digital_native(painted, invisible):
                    continue
                angle = page_angle(page)
                for hit in _find_conflicts(painted, invisible, angle):
                    conflict_instances.append({"page": page_num, **hit})
                for hit in _find_injections(painted, invisible):
                    injection_instances.append({"page": page_num, **hit})
            except Exception as exc:  # one bad page must not sink the module
                logger.warning("invisible: page %d failed: %s", page_num, exc)
                confidence = min(confidence, 0.5)
    except Exception as exc:
        logger.warning("invisible: analysis failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = 0.5
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    if conflict_instances:
        n = len(conflict_instances)
        if producer_is_editor:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED", "critical"
        elif producer_is_ocr:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT", "medium"
        else:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT", "high"
        flags.append(Flag(
            severity=severity,
            code=code,
            message=(f"{n} invisible text value(s) disagree with the visible "
                     f"page (e.g. hidden '{conflict_instances[0]['hidden']}' "
                     f"vs visible '{conflict_instances[0]['visible']}')"),
            details={"count": n, "instances": conflict_instances,
                     "producer_is_editor": producer_is_editor,
                     "producer_is_ocr": producer_is_ocr},
        ))

    if injection_instances and not producer_is_ocr:
        n = len(injection_instances)
        flags.append(Flag(
            severity="medium",
            code="INVISIBLE_TEXT_INJECTION",
            message=(f"{n} invisible value-bearing text(s) hidden in blank "
                     f"areas (e.g. '{injection_instances[0]['hidden']}')"),
            details={"count": n, "instances": injection_instances},
        ))

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    return ModuleResult(module="invisible", flags=flags,
                        score=max(0, score), confidence=confidence)
