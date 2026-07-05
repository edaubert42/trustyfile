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
# \s covers regular / narrow / non-breaking FR thousands spaces.
_MONEY = re.compile(r"\d[\d\s.,]*[.,]\d{2}")
# An interactive PDF editor Producer — the C2 corroborator that makes a
# geometry finding HARD (mirrors overlay.INTERACTIVE_EDITOR_PRODUCERS).
EDITOR_PRODUCER = "Adobe Acrobat Pro DC"
NON_EDITOR_PRODUCER = "ReportLab PDF Library"
CONVERTER_PRODUCER = "iLovePDF"


def _amount_value(word: str) -> float | None:
    """Parse a fitz 'word' into a number, or None if it isn't a money amount.

    Handles both FR ("1 200,00") and EN ("1,200.00") groupings by stripping
    thousands separators and normalising the decimal comma.
    """
    m = _MONEY.fullmatch(word.strip())
    if not m:
        return None
    s = re.sub(r"\s", "", word)          # drop every kind of space (thousands sep)
    sep = max(s.rfind(","), s.rfind("."))  # last separator = decimal point
    digits = re.sub(r"[.,]", "", s[:sep]) + "." + s[sep + 1:]
    try:
        return float(digits)
    except ValueError:
        return None


def find_amount(page) -> dict | None:
    """Locate the largest monetary amount on the page (usually the total).

    Returns {rect, text, value, size, origin} for the biggest amount found, or
    None. We pick the largest value because the grand total — the fraudster's
    usual target — is almost always the biggest number on an invoice. `size` and
    `origin` (the text baseline) are captured so a retype can MATCH the original
    font exactly — a real fraudster does, and a size mismatch would otherwise be
    a giveaway the detector keys on.
    """
    best = None
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for mm in _MONEY.finditer(span.get("text", "")):
                    val = _amount_value(mm.group())
                    if val is None:
                        continue
                    if best is None or val > best["value"]:
                        best = {"rect": fitz.Rect(span["bbox"]), "text": mm.group(),
                                "value": val, "size": span.get("size", 10.0),
                                "origin": span.get("origin", (span["bbox"][0], span["bbox"][3])),
                                "font": span.get("font", ""),
                                "span_text": span.get("text", mm.group())}
    return best


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
        hit = find_amount(doc[0])
        if hit is None:
            return None
        old_text, val = hit["text"], hit["value"]
        new_val = round(val * 1.2 + 5, 2)
        # rebuild the WHOLE span with only the number swapped, preserving the original
        # format and any currency suffix ("1 200,00 €" -> "1 445,00 €") so the retype
        # is realistic (same char class, size, baseline) rather than a giveaway.
        new_money = _format_like(new_val, old_text)
        new_text = hit["span_text"].replace(old_text, new_money, 1)
        _cover_and_retype(doc[0], hit, new_text)
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
        hit = find_amount(doc[0])
        if hit is None:
            return None
        rect, old_text = hit["rect"], hit["text"]
        # a solid-white PNG the size of the amount, pasted over it
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, max(2, int(rect.width)),
                                                 max(2, int(rect.height))), False)
        pix.clear_with(255)
        doc[0].insert_image(rect, pixmap=pix)
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
