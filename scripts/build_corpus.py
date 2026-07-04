"""
Build a calibration corpus: one clean legit invoice + forged tampered variants.

Legit samples may also be sourced online (public Factur-X / utility invoices —
see samples/README.md); this builder guarantees a reproducible local baseline so
the harness and CI always have data. Forging our own tampered documents to test
our own detector is intended (CLAUDE.md "Test Document Ideas").
"""
from datetime import datetime, timedelta
from pathlib import Path

import fitz


def _clean_invoice(path: str) -> None:
    """A coherent, legitimate-looking invoice: valid dates, normal producer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS", fontname="helv", fontsize=18)
    page.insert_text(fitz.Point(40, 85),
                     "SIRET: 552 081 317 00018  |  TVA: FR03552081317",
                     fontname="helv", fontsize=8)
    page.insert_text(fitz.Point(40, 140), "FACTURE  N° FA-2025-00847", fontname="helv", fontsize=14)
    page.insert_text(fitz.Point(350, 120), "Date de facturation: 15/01/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(350, 135), "Date d'échéance: 14/02/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(40, 300), "Prestation de conseil", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 360), "Total HT:", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 360), "1 000,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 380), "TVA (20%):", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 380), "200,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(40, 460), "IBAN: FR7630006000011234567890189",
                     fontname="helv", fontsize=9)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "ReportLab PDF Library",
        "creator": "ACME Billing System",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": creation.strftime("D:%Y%m%d%H%M%S"),  # equal → no modification
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _forge_metadata_dates(src: str, dst: str) -> None:
    """Tamper: ModDate 45 days after CreationDate + online-converter producer."""
    doc = fitz.open(src)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "iLovePDF",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": (creation + timedelta(days=45)).strftime("D:%Y%m%d%H%M%S"),
    })
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    doc.close()


def _forge_amount_overlay(src: str, dst: str) -> None:
    """Tamper: white-box over the total, new (higher) amount drawn on top."""
    doc = fitz.open(src)
    page = doc[0]
    # Cover "1 200,00 €" and draw "9 200,00 €" over it.
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)  # full save; also try incremental variant below
    doc.close()


def _forge_incremental_edit(src: str, dst: str) -> None:
    """Tamper: append an incremental update that overlays a new amount."""
    import shutil
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    doc = fitz.open(dst)
    page = doc[0]
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)  # keeps prior revision
    doc.close()


# --- Phase 2 overlay corpus additions ---

def _legit_template_overlay(src: str, dst: str) -> None:
    """LEGIT: a template tool stamps the real value over a '0,00' placeholder.
    Reproduces the fraud's seqno signature (placeholder < fill < value) but is honest."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "TEMPLATE INVOICE", fontname="helv", fontsize=16)
    # placeholder painted first (low seqno)
    page.insert_text(fitz.Point(460, 400), "0,00 €", fontname="helv", fontsize=12)
    # stamping pass: white box then real value (higher seqno) — legit generation
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creator": "ACME Report Engine"})
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_faux_bold(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    for dx in (0.0, 0.4):  # same string, tiny offset = synthetic bold
        page.insert_text(fitz.Point(40 + dx, 100), "TOTAL: 1 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_watermark(dst: str) -> None:
    # NOTE: page.insert_text()'s `rotate` only accepts multiples of 90 in this
    # PyMuPDF build (rotate=45 raises "bad rotate value"); rotate=90 is the
    # closest valid rotation and still exercises the same structural pattern
    # (large, greyed-out, rotated watermark text over the real content).
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 100), "Facture 1 200,00 €", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(150, 500), "SPECIMEN", fontname="helv", fontsize=60,
                     color=(0.7, 0.7, 0.7), rotate=90)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_zebra(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    # tint painted BEFORE text (legit background z-order)
    page.draw_rect(fitz.Rect(40, 390, 545, 408), color=None, fill=(0.9, 0.9, 0.95))
    page.insert_text(fitz.Point(460, 403), "1 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_knockout_header(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(0, 0, 595, 60), color=None, fill=(0.1, 0.1, 0.3))  # dark band first
    page.insert_text(fitz.Point(40, 40), "TOTAL", fontname="helv", fontsize=20, color=(1, 1, 1))
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_footer_band(dst: str) -> None:
    """LEGIT: full-width colored footer band painted BEFORE the SIRET/TVA legal text
    that sits on top of it. Exercises the "background rect precedes text" z-order
    guard (same shape as _legit_zebra/_legit_knockout_header, but full-width footer)."""
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 100), "Facture 1 200,00 €", fontname="helv", fontsize=12)
    # full-width footer tint drawn first (low seqno) — legit background
    page.draw_rect(fitz.Rect(0, 800, 595, 842), color=None, fill=(0.93, 0.93, 0.93))
    page.insert_text(fitz.Point(40, 820),
                     "SIRET: 552 081 317 00018  |  TVA: FR03552081317",
                     fontname="helv", fontsize=8)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_form_field(dst: str) -> None:
    """LEGIT: an annotation (freetext) whose rect overlaps a label.
    NOTE: headless PyMuPDF widget/AcroForm creation is awkward without a full
    appearance-stream setup, so this uses page.add_freetext_annot instead — it
    exercises the SAME annot-provenance exclusion path (content living in an
    Annot rather than the page content stream must not be mistaken for a
    cover-and-retype overlay)."""
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    annot = page.add_freetext_annot(fitz.Rect(455, 392, 545, 406), "1 200,00 €", fontsize=12)
    annot.update()
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_flattened_form(dst: str) -> None:
    """LEGIT: emulates a flattened form field — the field's background rect is
    drawn first, then the filled-in value at a higher seqno, authored by a
    typical form-filling producer. If this PyMuPDF build exposes doc.bake()
    (form flattening), use it to make the emulation more faithful; otherwise
    the hand-drawn background+value sequence already reproduces the structural
    pattern (background z-order + single authoring session)."""
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    # field background box first (low seqno) — how flattened form fields render
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(0.8, 0.8, 0.8), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": "Adobe Acrobat Form Filler", "creator": "Acrobat"})
    if hasattr(doc, "bake"):
        try:
            doc.bake()
        except Exception:
            pass  # NOTE: bake() unavailable/inapplicable in this headless build; hand-drawn emulation stands
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_french_accents(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 100),
                     "Réédité à Nîmes : intérêt, échéance, société, où çà où là œuvre",
                     fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_real_redaction(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    # NOTE: the brief's verbatim fixture only draws the value before redacting it,
    # which leaves the page with ZERO extractable text once apply_redactions() runs.
    # That accidentally triggers an unrelated pre-existing bug: src/modules/identity.py's
    # _has_face_photo() calls cv2.CascadeClassifier(...) unguarded outside its try/except,
    # and this environment's opencv-python-headless build has no CascadeClassifier at all,
    # so any textless PDF crashes the whole analyzer. Adding this label keeps the page
    # non-empty (out of scope to patch identity.py here) while still exercising the exact
    # same structural guard: the VALUE text is genuinely gone after apply_redactions().
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    page.add_redact_annot(fitz.Rect(455, 392, 545, 406), fill=(0, 0, 0))
    page.apply_redactions()  # REMOVES the underlying text
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_ocr_sandwich(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
    pix.clear_with(255)
    page.insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv",
                     fontsize=12, render_mode=3)  # invisible OCR text
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _legit_multisession_template_fill(dst: str) -> None:
    """LEGIT (boundary): built via _legit_template_overlay (placeholder < fill < value
    seqno signature), then a SECOND, incremental save that changes the value again
    while keeping the SAME producer metadata — emulates a legit multi-session
    template tool (e.g. draft saved, then re-opened and the amount corrected)
    rather than a single fraudulent edit session."""
    _legit_template_overlay("", dst)
    doc = fitz.open(dst)
    page = doc[0]
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creator": "ACME Report Engine"})
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()


def _forge_image_patch(src: str, dst: str) -> None:
    """TAMPER: cover the total with a small opaque image, leave the text underneath."""
    doc = fitz.open(src); page = doc[0]
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 90, 14)); pix.clear_with(255)
    page.insert_image(fitz.Rect(455, 392, 545, 406), pixmap=pix)  # opaque patch over total
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()


def _forge_text_deleted_retyped(src: str, dst: str) -> None:
    """TAMPER (boundary): remove the original via redaction, retype in place, incremental save.
    Signals 1-3 must NOT fire (no ghost); Signal 4 catches it via the revision diff."""
    import shutil
    Path(dst).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    doc = fitz.open(dst); page = doc[0]
    page.add_redact_annot(fitz.Rect(455, 392, 545, 406)); page.apply_redactions()
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP); doc.close()


corpus_expectations = {
    # tamper — must be caught
    "forged_amount_overlay.pdf":       {"label": "tamper", "expect_code": "OVERLAY_COVER_AND_RETYPE"},
    "forged_incremental_edit.pdf":     {"label": "tamper", "expect_code": "OVERLAY_COVER_AND_RETYPE_CONFIRMED", "max_band": "SUSPICIOUS"},
    "forged_image_patch_over_amount.pdf": {"label": "tamper", "expect_code": "OVERLAY_IMAGE_OVER_VALUE"},
    "forged_text_deleted_retyped.pdf": {"label": "tamper", "expect_code": "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"},
    # legit — must NOT flag
    "clean_invoice.pdf":               {"label": "legit", "expect_no_overlay": True},
    "legit_template_overlay.pdf":      {"label": "legit", "expect_no_overlay": True},
    "legit_faux_bold.pdf":             {"label": "legit", "expect_no_overlay": True},
    "legit_watermark.pdf":             {"label": "legit", "expect_no_overlay": True},
    "legit_zebra.pdf":                 {"label": "legit", "expect_no_overlay": True},
    "legit_knockout_header.pdf":       {"label": "legit", "expect_no_overlay": True},
    "legit_footer_band.pdf":           {"label": "legit", "expect_no_overlay": True},
    "legit_form_field.pdf":            {"label": "legit", "expect_no_overlay": True},
    "legit_real_redaction.pdf":        {"label": "legit", "expect_no_overlay": True},
    "legit_ocr_sandwich.pdf":          {"label": "legit", "expect_no_overlay": True},
    "legit_french_accents.pdf":        {"label": "legit", "expect_no_overlay": True},
    "legit_multisession_template_fill.pdf": {"label": "legit", "expect_no_overlay": True},
}


def build_corpus(out_dir: str) -> dict:
    out = Path(out_dir)
    legit = out / "legitimate" / "clean_invoice.pdf"
    _clean_invoice(str(legit))

    legitimate = [str(legit)]
    forged = []

    f1 = out / "suspicious" / "forged_metadata_dates.pdf"
    _forge_metadata_dates(str(legit), str(f1)); forged.append(str(f1))
    f2 = out / "suspicious" / "forged_amount_overlay.pdf"
    _forge_amount_overlay(str(legit), str(f2)); forged.append(str(f2))
    f3 = out / "suspicious" / "forged_incremental_edit.pdf"
    _forge_incremental_edit(str(legit), str(f3)); forged.append(str(f3))
    f4 = out / "suspicious" / "forged_image_patch_over_amount.pdf"
    _forge_image_patch(str(legit), str(f4)); forged.append(str(f4))
    f5 = out / "suspicious" / "forged_text_deleted_retyped.pdf"
    _forge_text_deleted_retyped(str(legit), str(f5)); forged.append(str(f5))

    lto = out / "legitimate" / "legit_template_overlay.pdf"
    _legit_template_overlay(str(legit), str(lto)); legitimate.append(str(lto))
    lfb = out / "legitimate" / "legit_faux_bold.pdf"
    _legit_faux_bold(str(lfb)); legitimate.append(str(lfb))
    lwm = out / "legitimate" / "legit_watermark.pdf"
    _legit_watermark(str(lwm)); legitimate.append(str(lwm))
    lz = out / "legitimate" / "legit_zebra.pdf"
    _legit_zebra(str(lz)); legitimate.append(str(lz))
    lkh = out / "legitimate" / "legit_knockout_header.pdf"
    _legit_knockout_header(str(lkh)); legitimate.append(str(lkh))
    lfoot = out / "legitimate" / "legit_footer_band.pdf"
    _legit_footer_band(str(lfoot)); legitimate.append(str(lfoot))
    lff = out / "legitimate" / "legit_form_field.pdf"
    _legit_form_field(str(lff)); legitimate.append(str(lff))
    lflat = out / "legitimate" / "legit_flattened_form.pdf"
    _legit_flattened_form(str(lflat)); legitimate.append(str(lflat))
    lrr = out / "legitimate" / "legit_real_redaction.pdf"
    _legit_real_redaction(str(lrr)); legitimate.append(str(lrr))
    locr = out / "legitimate" / "legit_ocr_sandwich.pdf"
    _legit_ocr_sandwich(str(locr)); legitimate.append(str(locr))
    lfr = out / "legitimate" / "legit_french_accents.pdf"
    _legit_french_accents(str(lfr)); legitimate.append(str(lfr))
    lms = out / "legitimate" / "legit_multisession_template_fill.pdf"
    _legit_multisession_template_fill(str(lms)); legitimate.append(str(lms))

    return {"legitimate": legitimate, "suspicious": forged}


if __name__ == "__main__":
    import sys
    result = build_corpus(sys.argv[1] if len(sys.argv) > 1 else "samples")
    print("Legitimate:", *result["legitimate"], sep="\n  ")
    print("Suspicious:", *result["suspicious"], sep="\n  ")
