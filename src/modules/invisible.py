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
    INTERACTIVE_EDITOR_PRODUCERS,
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
    convention. Implementation note from the spec: verify how get_texttrace()
    reports Tr 4-7; if they surface as type 3 they are covered for free.
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

    doc = None
    try:
        doc = fitz.open(pdf_data.file_path)
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                painted, invisible = _split_spans(page)
                if not _is_digital_native(painted, invisible):
                    continue
                # Detection signals land here in later tasks.
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

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    return ModuleResult(module="invisible", flags=flags,
                        score=max(0, score), confidence=confidence)
