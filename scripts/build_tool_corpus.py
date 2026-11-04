"""
Build a labelled corpus of genuine and tampered PDFs made with REAL tools.

Why this exists (2026-11-04): five real documents sent by the owner exposed
cases the older generators never produce — linearized ("Fast Web View") files
mistaken for edited ones, a recipient name/address replaced in a revision, an
edit appended after linearization, a birth year retyped in another font. This
script reproduces those situations with the PDF tools available locally, on
documents whose truth we know, so the detector can be measured on them.

Generators (how a genuine document is born):
    reportlab, pymupdf, libreoffice (PDF/A, carries XMP dates), chromium.
Genuine variants (must NOT be flagged as tampered):
    original, linearized (pikepdf), resaved (pikepdf full rewrite).
Tampered variants (must NOT come out GENUINE):
    redact_retype_incr     amount redacted + retyped, incremental save (PyMuPDF)
    cover_retype_full      white box + new amount, full rewrite: no revision left
    name_change_incr       recipient name/address replaced in a revision
    linearized_then_edited linearize, then an incremental edit (like fr_proof)
    font_midline_year      birth year retyped in a different font (attestation)
    stream_edit            amount bytes replaced inside the content stream (pikepdf)
    info_backdate          Info /CreationDate moved back 90 days, XMP left alone
Blind variant (must come out UNVERIFIABLE): flattened to an image (pdftoppm).

Nothing here is sent anywhere. Output goes to samples/ (gitignored).

Usage:
    python scripts/build_tool_corpus.py                 # -> samples/synthetic_tools/
    python scripts/build_tool_corpus.py --evaluate      # build, then score every file
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz
import pikepdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "samples" / "synthetic_tools"

logger = logging.getLogger("build_tool_corpus")

# --- Document content --------------------------------------------------------
# Real-looking but fictitious data; identifiers have valid check digits so the
# identifier checks stay quiet on the genuine files (Google France's public
# SIRET, the ISO 13616 example IBAN).
SIREN = "732829320"
SIRET = SIREN + "00074"
VAT = f"FR{(12 + 3 * (int(SIREN) % 97)) % 97:02d}{SIREN}"
IBAN = "FR76 3000 6000 0112 3456 7890 189"

INVOICE_LINES = [
    ("title", "FACTURE N° 2026-0412"),
    ("text", "Atelier Exemple SAS"),
    ("text", "8 rue des Lilas, 69003 Lyon"),
    ("text", f"SIRET : {SIRET}"),
    ("text", f"TVA intracommunautaire : {VAT}"),
    ("gap", ""),
    ("text", "Client :"),
    ("text", "Louise Martin"),                      # name_change target
    ("text", "11 allée des Peupliers, 92380 Garches"),  # name_change target
    ("gap", ""),
    ("text", "Date : 14/10/2026"),
    ("text", "Prestation de maintenance (5 jours)"),
    ("text", "Montant HT : 1 000,00 EUR"),
    ("text", "TVA 20 % : 200,00 EUR"),
    ("bold", "Total TTC : 1 200,00 EUR"),           # amount target
    ("gap", ""),
    ("text", f"IBAN : {IBAN}"),
]
ATTESTATION_LINES = [
    ("title", "ATTESTATION DE L'EMPLOYEUR"),
    ("gap", ""),
    ("text", "Je soussigné, Paul Bernard, gérant de la société Atelier Exemple SAS,"),
    ("text", f"SIRET {SIRET}, atteste que Madame Claire Durand,"),
    ("bold", "Date de naissance : 19/08/1992"),     # font_midline_year target
    ("text", "est employée en contrat à durée indéterminée depuis le 01/03/2021,"),
    ("text", "au poste de comptable, pour un salaire brut mensuel de 2 450,00 EUR."),
    ("gap", ""),
    ("text", "Fait pour servir et valoir ce que de droit."),
    ("text", "Fait à Lyon, le 15/10/2026."),
]
DOCS = {"invoice": INVOICE_LINES, "attestation": ATTESTATION_LINES}

AMOUNT_OLD, AMOUNT_NEW = "1 200,00 EUR", "4 200,00 EUR"
NAME_OLD = ["Louise Martin", "11 allée des Peupliers, 92380 Garches"]
NAME_NEW = ["Philippe Martin", "26 avenue de la Grande Armée, 75017 Paris"]
YEAR_OLD, YEAR_NEW = "1992", "1998"


# =============================================================================
# GENERATORS — four genuine ways a document is born
# =============================================================================

def gen_reportlab(lines, dst: Path) -> None:
    """ReportLab: base-14 Helvetica, plain-string content streams."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(dst), pagesize=A4)
    y = 780
    for kind, text in lines:
        if kind == "gap":
            y -= 10
            continue
        c.setFont("Helvetica-Bold" if kind in ("title", "bold") else "Helvetica", 15 if kind == "title" else 11)
        c.drawString(60, y, text)
        y -= 26 if kind == "title" else 17
    c.save()


def gen_pymupdf(lines, dst: Path) -> None:
    """PyMuPDF: base-14 fonts via insert_text."""
    doc = fitz.open()
    page = doc.new_page()
    y = 70
    for kind, text in lines:
        if kind == "gap":
            y += 10
            continue
        page.insert_text((60, y), text, fontsize=15 if kind == "title" else 11,
                         fontname="hebo" if kind in ("title", "bold") else "helv")
        y += 26 if kind == "title" else 17
    doc.set_metadata({"producer": "PyMuPDF", "creator": "Facturation Exemple 2.1",
                      "creationDate": fitz.get_pdf_now(), "modDate": fitz.get_pdf_now()})
    doc.save(dst)
    doc.close()


def gen_libreoffice(lines, dst: Path) -> None:
    """LibreOffice Writer -> PDF/A-2: embedded subset fonts, XMP metadata."""
    from xml.sax.saxutils import escape
    paras = []
    for kind, text in lines:
        style = {"title": "T", "bold": "B"}.get(kind, "P")
        paras.append(f'<text:p text:style-name="{style}">{escape(text)}</text:p>')
    fodt = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"
 office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.text">
 <office:meta><meta:creation-date>{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}</meta:creation-date></office:meta>
 <office:automatic-styles>
  <style:style style:name="T" style:family="paragraph"><style:text-properties fo:font-size="15pt" fo:font-weight="bold"/></style:style>
  <style:style style:name="B" style:family="paragraph"><style:text-properties fo:font-weight="bold"/></style:style>
 </office:automatic-styles>
 <office:body><office:text>{''.join(paras)}</office:text></office:body>
</office:document>"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.fodt"
        src.write_text(fodt, encoding="utf-8")
        subprocess.run(
            ["soffice", "--headless", "--convert-to",
             'pdf:writer_pdf_Export:{"SelectPdfVersion":{"type":"long","value":"2"}}',
             "--outdir", tmp, str(src)],
            check=True, capture_output=True, timeout=180,
        )
        shutil.move(Path(tmp) / "doc.pdf", dst)


def gen_chromium(lines, dst: Path) -> None:
    """Chromium 'print to PDF' (Skia/PDF), driven by Playwright."""
    from html import escape
    body = []
    for kind, text in lines:
        if kind == "gap":
            body.append("<div style='height:10px'></div>")
        elif kind == "title":
            body.append(f"<h2>{escape(text)}</h2>")
        else:
            weight = "bold" if kind == "bold" else "normal"
            body.append(f"<div style='font-weight:{weight}'>{escape(text)}</div>")
    html_doc = ("<html><body style='font-family:Arial, sans-serif; font-size:11pt; margin:40px'>"
                + "".join(body) + "</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page_html = Path(tmp) / "doc.html"
        page_html.write_text(html_doc, encoding="utf-8")
        script = (
            "const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
            "const p=await b.newPage();await p.goto(process.argv[1]);"
            "await p.pdf({path:process.argv[2],format:'A4'});await b.close()})()"
        )
        subprocess.run(["node", "-e", script, page_html.as_uri(), str(dst)],
                       check=True, capture_output=True, timeout=120)


GENERATORS = {
    "reportlab": gen_reportlab,
    "pymupdf": gen_pymupdf,
    "libreoffice": gen_libreoffice,
    "chromium": gen_chromium,
}


# How old the genuine documents are. A forger edits a document they RECEIVED,
# days or months after it was produced (the owner's real edited attestation:
# 143 days). Without this, base and edit share the same minute and every
# date-gap signal is silent — an unrealistic, too-hard corpus.
AGE_DAYS = 30


def _age_document(path: Path, days: int = AGE_DAYS) -> None:
    """Make a freshly generated file look produced `days` ago.

    Both copies of each date move together — Info /CreationDate and /ModDate,
    and the XMP CreateDate/ModifyDate when the file has XMP — so the genuine
    file stays internally consistent (no Info/XMP disagreement).
    """
    when = datetime.now(timezone.utc) - timedelta(days=days)
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.docinfo["/CreationDate"] = when.strftime("D:%Y%m%d%H%M%SZ")
        pdf.docinfo["/ModDate"] = when.strftime("D:%Y%m%d%H%M%SZ")
        meta = pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False)
        if "xmp:CreateDate" in meta:
            with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as m:
                for key in ("xmp:CreateDate", "xmp:ModifyDate", "xmp:MetadataDate"):
                    if key in m:
                        m[key] = when.isoformat(timespec="seconds")
        pdf.save(path)


# =============================================================================
# GENUINE VARIANTS — same content, different but innocent file layout
# =============================================================================

def var_linearized(src: Path, dst: Path) -> None:
    """'Fast Web View': a second %%EOF near the start, by construction."""
    with pikepdf.open(src) as pdf:
        pdf.save(dst, linearize=True)


def var_resaved(src: Path, dst: Path) -> None:
    """Full rewrite with object streams, content untouched."""
    with pikepdf.open(src) as pdf:
        pdf.save(dst, object_stream_mode=pikepdf.ObjectStreamMode.generate)


# =============================================================================
# TAMPERING — each returns True if it could be applied to this base
# =============================================================================

def _replace_text(page, old: str, new: str, fontname: str = "helv", bold: bool = False) -> bool:
    """Redact `old` (really removes it from the text layer) and type `new` in its place."""
    rects = page.search_for(old)
    if not rects:
        return False
    rect = rects[0]
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    size = max(6.0, rect.height * 0.78)
    page.insert_text((rect.x0, rect.y1 - rect.height * 0.22), new, fontsize=size,
                     fontname=("hebo" if bold else fontname))
    return True


def _incremental_edit(src: Path, dst: Path, edits: list[tuple[str, str]], **kw) -> bool:
    """Copy src, apply text edits, save INCREMENTALLY (revision kept, like Acrobat 'Save')."""
    shutil.copyfile(src, dst)
    doc = fitz.open(dst)
    ok = all(_replace_text(doc[0], old, new, **kw) for old, new in edits)
    if ok:
        doc.set_metadata({**doc.metadata, "modDate": fitz.get_pdf_now()})
        doc.saveIncr()
    doc.close()
    return ok


def t_redact_retype_incr(src: Path, dst: Path) -> bool:
    return _incremental_edit(src, dst, [(AMOUNT_OLD, AMOUNT_NEW)], bold=True)


def t_cover_retype_full(src: Path, dst: Path) -> bool:
    """White box over the amount + new amount, then a FULL save: the old text
    stays under the box, but no revision boundary is left in the file."""
    doc = fitz.open(src)
    page = doc[0]
    rects = page.search_for(AMOUNT_OLD)
    if not rects:
        return False
    r = rects[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text((r.x0, r.y1 - r.height * 0.22), AMOUNT_NEW, fontsize=max(6.0, r.height * 0.78), fontname="hebo")
    doc.save(dst, garbage=3, deflate=True)
    doc.close()
    return True


def t_name_change_incr(src: Path, dst: Path) -> bool:
    return _incremental_edit(src, dst, list(zip(NAME_OLD, NAME_NEW)))


def t_linearized_then_edited(src: Path, dst: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        lin = Path(tmp) / "lin.pdf"
        var_linearized(src, lin)
        return _incremental_edit(lin, dst, [(AMOUNT_OLD, AMOUNT_NEW) if "FACTURE" in fitz.open(lin)[0].get_text()
                                            else (YEAR_OLD, YEAR_NEW)], bold=True)


def t_font_midline_year(src: Path, dst: Path) -> bool:
    """Birth year retyped in a serif font on a sans-serif line (cf. fr_proof)."""
    return _incremental_edit(src, dst, [(YEAR_OLD, YEAR_NEW)], fontname="tiro")


def t_font_midline_year_full(src: Path, dst: Path) -> bool:
    """Same serif retype of the birth year, but a FULL save: no earlier
    revision is left to diff, so only the font evidence on the page remains."""
    doc = fitz.open(src)
    ok = _replace_text(doc[0], YEAR_OLD, YEAR_NEW, fontname="tiro")
    if ok:
        doc.save(dst, garbage=3, deflate=True)
    doc.close()
    return ok


def t_stream_edit(src: Path, dst: Path) -> bool:
    """Edit the amount where it is written in the content stream itself — no
    new object, no new font. Only works where text is stored as plain strings
    (ReportLab, PyMuPDF base-14); subset-font generators encode glyph ids."""
    with pikepdf.open(src) as pdf:
        changed = False
        for page in pdf.pages:
            for stream in ([page.Contents] if isinstance(page.Contents, pikepdf.Stream) else list(page.Contents)):
                data = stream.read_bytes()
                if b"1 200,00" in data:
                    stream.write(data.replace(b"1 200,00", b"4 200,00"))
                    changed = True
        if changed:
            pdf.save(dst)
        return changed


def t_info_backdate(src: Path, dst: Path) -> bool:
    """Move the Info /CreationDate back 90 days; leave the XMP copy alone.
    Only meaningful where an XMP CreateDate exists (LibreOffice PDF/A)."""
    with pikepdf.open(src) as pdf:
        meta = pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False)
        if "xmp:CreateDate" not in meta:
            return False
        when = datetime.now(timezone.utc) - timedelta(days=90 + AGE_DAYS)
        pdf.docinfo["/CreationDate"] = when.strftime("D:%Y%m%d%H%M%SZ")
        pdf.save(dst)
    return True


TAMPERS = {
    "redact_retype_incr": t_redact_retype_incr,
    "cover_retype_full": t_cover_retype_full,
    "name_change_incr": t_name_change_incr,
    "linearized_then_edited": t_linearized_then_edited,
    "font_midline_year": t_font_midline_year,
    "font_midline_year_full": t_font_midline_year_full,
    "stream_edit": t_stream_edit,
    "info_backdate": t_info_backdate,
}


def v_flattened(src: Path, dst: Path) -> bool:
    """Render to an image and wrap it in a PDF: what a scan or a 'print as image' gives."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(src), str(Path(tmp) / "p")],
                       check=True, capture_output=True, timeout=120)
        doc = fitz.open()
        img = fitz.open(Path(tmp) / "p.png")
        rect = img[0].rect
        page = doc.new_page(width=rect.width * 72 / 150, height=rect.height * 72 / 150)
        page.insert_image(page.rect, filename=str(Path(tmp) / "p.png"))
        doc.save(dst)
        doc.close()
    return True


# =============================================================================
# BUILD + EVALUATE
# =============================================================================

def build(out: Path) -> list[dict]:
    """Generate every base and variant; return the manifest (one dict per file)."""
    if out.exists():
        shutil.rmtree(out)
    for sub in ("genuine", "tampered", "blind"):
        (out / sub).mkdir(parents=True)
    manifest: list[dict] = []

    def add(path: Path, expect: str, generator: str, doc_type: str, variant: str) -> None:
        manifest.append({"file": str(path.relative_to(out)), "expect": expect,
                         "generator": generator, "doc_type": doc_type, "variant": variant})

    for doc_type, lines in DOCS.items():
        for gen_name, gen in GENERATORS.items():
            stem = f"{doc_type}_{gen_name}"
            base = out / "genuine" / f"{stem}_original.pdf"
            try:
                gen(lines, base)
                _age_document(base)
            except Exception as exc:  # a missing tool must not stop the rest
                logger.warning("generator %s failed: %s", gen_name, exc)
                continue
            add(base, "genuine", gen_name, doc_type, "original")
            for var_name, var in (("linearized", var_linearized), ("resaved", var_resaved)):
                dst = out / "genuine" / f"{stem}_{var_name}.pdf"
                var(base, dst)
                add(dst, "genuine", gen_name, doc_type, var_name)
            for t_name, tamper in TAMPERS.items():
                dst = out / "tampered" / f"{stem}_{t_name}.pdf"
                try:
                    applied = tamper(base, dst)
                except Exception as exc:
                    logger.warning("%s on %s failed: %s", t_name, stem, exc)
                    applied = False
                if applied:
                    add(dst, "tampered", gen_name, doc_type, t_name)
                elif dst.exists():
                    dst.unlink()
            dst = out / "blind" / f"{stem}_flattened.pdf"
            v_flattened(base, dst)
            add(dst, "unverifiable", gen_name, doc_type, "flattened")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def evaluate(out: Path, manifest: list[dict]) -> list[dict]:
    """Score every file and say whether the verdict matches the truth.

    Pass rules (deliberately the same bar the product promises):
        genuine      -> band GENUINE
        tampered     -> any band but GENUINE (REVIEW or SUSPICIOUS)
        unverifiable -> band UNVERIFIABLE (or SUSPICIOUS: never "fine")
    """
    sys.path.insert(0, str(ROOT))
    logging.disable(logging.CRITICAL)
    from src.analyzer import TrustyFileAnalyzer
    from src.scoring import collect_all_flags

    analyzer = TrustyFileAnalyzer(enable_external=False)
    for entry in manifest:
        result = analyzer.analyze(str(out / entry["file"]))
        band = result.band
        entry["band"], entry["score"] = band, result.trust_score
        entry["flags"] = [f.code for f in collect_all_flags(result.modules)]
        entry["ok"] = {
            "genuine": band == "GENUINE",
            "tampered": band != "GENUINE",
            "unverifiable": band in ("UNVERIFIABLE", "SUSPICIOUS"),
        }[entry["expect"]]
    logging.disable(logging.NOTSET)
    (out / "results.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _print_report(results: list[dict]) -> None:
    for expect in ("genuine", "tampered", "unverifiable"):
        rows = [r for r in results if r["expect"] == expect]
        if not rows:
            continue
        passed = sum(r["ok"] for r in rows)
        print(f"\n## {expect}: {passed}/{len(rows)} correct")
        for r in rows:
            mark = "ok " if r["ok"] else "XX "
            print(f"  {mark}{r['file']:58} {r['band']:12} {str(r['score']):>4}  {', '.join(r['flags'][:4])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluate", action="store_true", help="score every generated file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build(args.out)
    print(f"{len(manifest)} files written to {args.out}")
    if args.evaluate:
        _print_report(evaluate(args.out, manifest))


if __name__ == "__main__":
    main()
