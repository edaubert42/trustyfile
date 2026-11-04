"""
Generate REALISTIC tampered PDFs from real invoice bases — with ground truth.

The synthetic corpus in build_corpus.py forges attacks on ONE hand-built invoice.
This module instead takes ANY real invoice PDF and re-creates the common
digital-edit frauds *on that real document*, so the detector is tested against
realistic layouts. Every output carries a manifest entry (source, attack, the
value before/after, the Producer used, and the expected band) so results are
self-checking.

Local only: pure PyMuPDF, no network, never sends the document anywhere.

Attacks produced per base (when an amount can be located):
  - cover_retype_plain : opaque box over the total + a bigger number, saved by a
                         NON-editor Producer -> geometry only -> expect REVIEW.
  - cover_retype_editor: same, but saved by an interactive-editor Producer (the
                         C2 corroborator) -> expect SUSPICIOUS (HARD).
  - image_patch        : an opaque image pasted over the total -> expect REVIEW.
  - metadata_backdate  : ModDate long after CreationDate + online-converter
                         Producer -> metadata tamper -> expect REVIEW/SUSPICIOUS.
"""
import json
import logging
import re
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

# A monetary amount inside a span: "1 200,00", "1,000.00", "200,00".
# \s covers regular / narrow / non-breaking FR thousands spaces. The
# (?<!\d)…(?!\d) anchors reject a match that sits INSIDE a longer digit run —
# e.g. "3.441.81" out of the share-capital "3.441.812" (the free_fiber bug).
_MONEY = re.compile(r"(?<!\d)\d[\d\s.,]*[.,]\d{2}(?!\d)")
# A day.month.year / day/month/year date — looks like an amount to _MONEY
# ("10.01.20") but must never be tampered as a total (the Teilrechnung bug).
_DATE = re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}$")
# An interactive PDF editor Producer — the C2 corroborator that makes a
# geometry finding HARD (see span_conflict.is_interactive_editor).
EDITOR_PRODUCER = "Adobe Acrobat Pro DC"
NON_EDITOR_PRODUCER = "ReportLab PDF Library"
CONVERTER_PRODUCER = "iLovePDF"


def _amount_value(word: str) -> float | None:
    """Parse a fitz 'word' into a number, or None if it isn't a money amount.

    Handles both FR ("1 200,00") and EN ("1,200.00") groupings by stripping
    thousands separators and normalising the decimal comma.
    """
    w = word.strip()
    if _DATE.match(w):
        return None                      # a date, not a monetary amount
    m = _MONEY.fullmatch(w)
    if not m:
        return None
    s = re.sub(r"\s", "", word)          # drop every kind of space (thousands sep)
    sep = max(s.rfind(","), s.rfind("."))  # last separator = decimal point
    digits = re.sub(r"[.,]", "", s[:sep]) + "." + s[sep + 1:]
    try:
        return float(digits)
    except ValueError:
        return None


# Total-label keywords in the corpus's three languages (FR / EN / DE), tiered.
# STRONG = an unambiguous grand-total label; WEAK = a generic total-ish word that
# also appears on subtotals/line groups. A fraudster edits the amount NEXT TO one
# of these, so anchoring the target to a keyword (not "largest number") is both
# realistic and the only way to hit a total that lives in a table column.
_KW_STRONG = re.compile(
    r"net\s*[àa]\s*payer|[àa]\s*payer|total\s*ttc|montant\s*ttc|montant\s*total|"
    r"amount\s*due|balance\s*due|total\s*due|grand\s*total|zahlbetrag|"
    r"gesamtbetrag|rechnungssumme|bruttosumme|zu\s*zahlen", re.I)
_KW_WEAK = re.compile(r"\btotal\b|montant|summe|gesamt|brutto|\bbetrag\b|payable|\bdue\b", re.I)


def _kw_tier(text: str) -> int:
    """2 for a strong grand-total label, 1 for a weak total-ish word, else 0."""
    if _KW_STRONG.search(text):
        return 2
    if _KW_WEAK.search(text):
        return 1
    return 0


def _anchor_score(a: fitz.Rect, keywords: list) -> tuple | None:
    """Best (tier, -distance) linking amount rect `a` to a nearby total-keyword,
    or None if no keyword anchors it.

    Anchored when the keyword is on the SAME ROW and to the LEFT of the amount
    (label ... value — the usual invoice layout, incl. table columns), or in the
    SAME COLUMN just ABOVE it. Distance is the gap between them, so the closest
    qualifying label wins.
    """
    lh = max(a.height, 1.0)
    ay = (a.y0 + a.y1) / 2
    ax = (a.x0 + a.x1) / 2
    best = None
    for krect, tier in keywords:
        ky = (krect.y0 + krect.y1) / 2
        same_row = abs(ky - ay) <= 0.7 * lh and krect.x1 <= a.x0 and (a.x0 - krect.x1) < 320
        kx = (krect.x0 + krect.x1) / 2
        below = abs(kx - ax) <= 2.5 * lh and 0 <= (a.y0 - krect.y1) < 2.2 * lh
        if not (same_row or below):
            continue
        dist = (a.x0 - krect.x1) if same_row else (a.y0 - krect.y1)
        cand = (tier, -dist)
        if best is None or cand > best:
            best = cand
    return best


def _intra_span_anchor(span_text: str, amount_start: int) -> tuple | None:
    """Anchor score when a total-keyword precedes the number in the SAME span
    (e.g. "Total à payer: 1 234,56 €"). Returns (tier, 0.0) or None."""
    for rx, tier in ((_KW_STRONG, 2), (_KW_WEAK, 1)):
        km = rx.search(span_text)
        if km and km.start() < amount_start:
            return (tier, 0.0)
    return None


def find_amount(page) -> dict | None:
    """Locate the invoice TOTAL on the page — the amount next to a total-keyword.

    Returns {rect, text, value, size, origin, font, span_text} for the best
    keyword-anchored amount, or None when no total-keyword anchors any amount
    (the generator then SKIPS the doc rather than tamper a random number — an
    honest corpus of confident total-tampers). We anchor to a keyword instead of
    picking the largest number because on real invoices the biggest token is
    often a date, reference, or legalese figure, not the total. `size`/`origin`
    let the retype match the original font; `rect` hugs the number itself (via
    search_for within the span) so a wide span doesn't make the cover box swallow
    the whole line.
    """
    spans = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans.extend(line.get("spans", []))

    keywords = [(fitz.Rect(s["bbox"]), _kw_tier(s.get("text", "")))
                for s in spans if _kw_tier(s.get("text", ""))]
    if not keywords:
        return None

    best = None          # (score_tuple, value) for tie-breaking
    best_hit = None
    for span in spans:
        span_rect = fitz.Rect(span["bbox"])
        span_text = span.get("text", "")
        for mm in _MONEY.finditer(span_text):
            val = _amount_value(mm.group())
            if val is None:
                continue
            tight = _tight_rect(page, mm.group(), span_rect)
            # anchor either across spans (label ... value) or WITHIN this span
            # ("Total à payer: 1 234,56 €" — keyword text precedes the number)
            cross = _anchor_score(tight, keywords)
            intra = _intra_span_anchor(span_text, mm.start())
            candidates = [s for s in (cross, intra) if s is not None]
            if not candidates:
                continue
            score = max(candidates)
            key = (score, val)              # strongest anchor, then largest value
            if best is None or key > best:
                best = key
                best_hit = {"rect": tight, "text": mm.group(), "value": val,
                            "size": span.get("size", 10.0),
                            "origin": span.get("origin", (span["bbox"][0], span["bbox"][3])),
                            "font": span.get("font", ""),
                            "span_text": span.get("text", mm.group())}
    return best_hit


def _tight_rect(page, needle: str, span_rect: fitz.Rect) -> fitz.Rect:
    """Rect hugging `needle` inside `span_rect`, or the span rect if not found.

    search_for locates the exact substring; clipping to the span disambiguates
    when the same number appears elsewhere on the page. Falls back to the full
    span rect (the old behaviour) if the glyphs don't round-trip (e.g. a font
    that renders digits via unusual encoding)."""
    try:
        hits = page.search_for(needle, clip=span_rect)
    except Exception:
        hits = None
    return hits[0] if hits else span_rect


def _match_base14(font: str) -> str:
    """Map the original font name to the closest base-14 face (helv/tiro/cour).

    We can't re-embed an arbitrary subset font, so we match the FAMILY (serif /
    monospace / sans) to avoid a font *mismatch* — otherwise a font-consistency
    detector could 'detect' the tamper via the wrong font instead of the
    cover-and-retype geometry, inflating apparent detection. Family match is an
    accepted approximation; exact glyphs are not reproduced.
    """
    n = (font or "").lower()
    if any(k in n for k in ("times", "serif", "georgia", "roman", "minion")):
        return "tiro"
    if any(k in n for k in ("cour", "mono", "consol")):
        return "cour"
    return "helv"


def _format_like(value: float, sample: str) -> str:
    """Format `value` in the SAME style as `sample` (FR '1 200,00' vs EN
    '1,200.00'), so the retyped amount blends in the way a real forgery would."""
    dec = "," if sample.rfind(",") > sample.rfind(".") else "."
    grp = " " if any(sp in sample for sp in (" ", " ", " ")) else ("," if dec == "." else ".")
    tmp = f"{value:,.2f}"                    # default: ',' thousands, '.' decimal
    return tmp.replace(",", "\0").replace(".", dec).replace("\0", grp)


def _find_total(doc) -> tuple:
    """Scan every page for the keyword-anchored total. Returns (page_index, hit)
    or (None, None) — a total can live on a later page of a multi-page invoice
    (e.g. a German Teilrechnung's summary page)."""
    for pno in range(doc.page_count):
        hit = find_amount(doc[pno])
        if hit is not None:
            return pno, hit
    return None, None


def _cover_and_retype(page, hit: dict, new_text, bg=(1, 1, 1)):
    """Draw an opaque background-coloured box over the amount and retype it at the
    SAME font family, size and baseline — the canonical 'cover a value and retype'
    edit, made realistic so the retyped value blends in the way a real forgery
    does (a size/font mismatch would be an artificial giveaway that a detector
    could key on instead of the geometry). NOTE: `bg` defaults to white; a total
    on a shaded cell would need the page background sampled — accepted limitation
    for the current white-invoice bases."""
    page.draw_rect(hit["rect"], color=bg, fill=bg, width=0)
    page.insert_text(fitz.Point(hit["rect"].x0, hit["origin"][1]), new_text,
                     fontname=_match_base14(hit.get("font", "")), fontsize=hit["size"])


def tamper_cover_retype(src: str, dst: str, producer: str) -> dict | None:
    """Cover the total and retype a 20% larger amount; save under `producer`.

    Returns a manifest dict, or None if no amount could be located in the base.
    """
    doc = fitz.open(src)
    try:
        pno, hit = _find_total(doc)
        if hit is None:
            return None
        old_text, val = hit["text"], hit["value"]
        new_val = round(val * 1.2 + 5, 2)
        # retype ONLY the number, matched to the original format (FR/EN) so it
        # blends in; the cover box hugs just the number (hit["rect"]), so any
        # currency symbol / label beside it stays visible and untouched.
        new_text = _format_like(new_val, old_text)
        _cover_and_retype(doc[pno], hit, new_text)
        doc.set_metadata({"producer": producer, "creator": "invoice tool",
                          "creationDate": "D:20250115103000", "modDate": "D:20250115103000"})
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        doc.save(dst)
    finally:
        doc.close()
    is_editor = "acrobat" in producer.lower()
    return {"attack": "cover_retype_editor" if is_editor else "cover_retype_plain",
            "old_value": old_text, "new_value": new_text, "producer": producer,
            "expect_band": "SUSPICIOUS" if is_editor else "REVIEW"}


def tamper_metadata_backdate(src: str, dst: str) -> dict:
    """Set ModDate 45 days after CreationDate and an online-converter Producer."""
    doc = fitz.open(src)
    try:
        doc.set_metadata({"producer": CONVERTER_PRODUCER,
                          "creationDate": "D:20250115103000",
                          "modDate": "D:20250301103000"})
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        doc.save(dst)
    finally:
        doc.close()
    return {"attack": "metadata_backdate", "producer": CONVERTER_PRODUCER,
            "expect_band": "REVIEW"}


def tamper_image_patch(src: str, dst: str) -> dict | None:
    """Paste a small opaque white image over the total (raster cover attack)."""
    doc = fitz.open(src)
    try:
        pno, hit = _find_total(doc)
        if hit is None:
            return None
        rect, old_text = hit["rect"], hit["text"]
        # a solid-white PNG the size of the amount, pasted over it
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, max(2, int(rect.width)),
                                                 max(2, int(rect.height))), False)
        pix.clear_with(255)
        doc[pno].insert_image(rect, pixmap=pix)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        doc.save(dst)
    finally:
        doc.close()
    return {"attack": "image_patch", "old_value": old_text, "expect_band": "REVIEW"}


def generate_from_dir(src_dir: str, out_dir: str) -> dict:
    """Produce all tampered variants for every PDF in `src_dir`.

    Writes the tampered PDFs plus a manifest.json (ground truth) into out_dir.
    Returns the manifest dict {generated: [...], skipped: [...]}.
    """
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated, skipped = [], []
    for pdf in sorted(src.glob("*.pdf")):
        stem = pdf.stem
        variants = [
            (f"{stem}__cover_plain.pdf", lambda s, d: tamper_cover_retype(s, d, NON_EDITOR_PRODUCER)),
            (f"{stem}__cover_editor.pdf", lambda s, d: tamper_cover_retype(s, d, EDITOR_PRODUCER)),
            (f"{stem}__image_patch.pdf", tamper_image_patch),
            (f"{stem}__metadata.pdf", tamper_metadata_backdate),
        ]
        for name, fn in variants:
            dst = out / name
            try:
                entry = fn(str(pdf), str(dst))
            except Exception as exc:  # never let one base sink the batch
                logger.warning("generate: %s failed on %s: %s", name, pdf.name, exc)
                entry = None
            if entry is None:
                skipped.append({"source": pdf.name, "output": name})
                continue
            entry.update({"source": pdf.name, "output": name})
            generated.append(entry)
    (out / "manifest.json").write_text(json.dumps(
        {"generated": generated, "skipped": skipped}, indent=2, ensure_ascii=False))
    return {"generated": generated, "skipped": skipped}


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO)
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "samples/legitimate/real"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "samples/suspicious/generated"
    result = generate_from_dir(src_dir, out_dir)
    print(f"generated {len(result['generated'])} tampered PDFs, "
          f"skipped {len(result['skipped'])} (no amount found)")
