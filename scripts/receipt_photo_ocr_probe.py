"""
Probe: does Tesseract read a PHOTOGRAPHED receipt's amounts exactly?

Arithmetic checks on expense receipts (lines sum to the total, VAT = net x
rate) are only safe if every amount is read exactly: one wrong digit and an
honest receipt "does not add up". This script measures that on a synthetic
80 mm receipt, photographed in simulation (rotation, blur, perspective,
sensor noise, JPEG), with two OCR settings:

  raw   pytesseract on the photo as is
  prep  2x upscale + Otsu binarisation, --psm 6 (one block of text)

Result on 2026-11-05 (tesseract 5, fra): 0/18 photos read exactly with
either setting, and wrong digits are CONSISTENT across settings (3,27 read
3,21 by both at 8 degrees), so a "two reads must agree" guard would not
catch them. See docs/superpowers/research/2026-11-05-expense-receipts-exploration.md.

Usage:
    python scripts/receipt_photo_ocr_probe.py
"""
import io
import re

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageFilter

# French money amounts ("36,00", "2 400,00"); percentages removed first.
_AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?: \d{3})+|\d+),(\d{2})(?![\d,])")


def line_amounts(line: str) -> list[float]:
    line = re.sub(r"\d+(?:[.,]\d+)?\s*%", " ", line)
    return [float(units.replace(" ", "") + "." + cents) for units, cents in _AMOUNT_RE.findall(line)]

RECEIPT = [
    ("CAFE DU COMMERCE", None),
    ("12 rue de la Paix 75002 PARIS", None),
    ("SIRET 732 829 320 00074", None),
    ("Le 14/03/2024 a 12:41  Ticket 0457", None),
    ("2 x Plat du jour      29,00", 29.00),
    ("1 x Cafe              2,50", 2.50),
    ("1 x Eau minerale      4,50", 4.50),
    ("TOTAL TTC            36,00", 36.00),
    ("dont TVA 10%          3,27", 3.27),
    ("TOTAL HT             32,73", 32.73),
    ("CB                   36,00", 36.00),
    ("Merci de votre visite", None),
]
EXPECTED = [amount for _, amount in RECEIPT if amount is not None]


def receipt_image(dpi: int = 300) -> Image.Image:
    """The receipt as a clean grayscale render (80 mm thermal roll, Courier)."""
    doc = fitz.open()
    page = doc.new_page(width=226, height=420)
    for i, (text, _) in enumerate(RECEIPT):
        page.insert_text((12, 30 + i * 22), text, fontsize=8.5, fontname="cour")
    pix = page.get_pixmap(dpi=dpi)
    doc.close()
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")


def photograph(img: Image.Image, angle: float, blur: float, persp: bool, seed: int) -> Image.Image:
    """Simulate a phone photo: paper on a table, tilt, perspective, noise, JPEG."""
    w, h = img.size
    canvas = Image.new("L", (int(w * 1.15), int(h * 1.1)), 90)
    canvas.paste(img.rotate(angle, fillcolor=90, resample=Image.BICUBIC), (int(w * 0.07), int(h * 0.05)))
    arr = np.asarray(canvas, dtype=np.float32)
    height, width = arr.shape
    if persp:
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = np.float32([[width * .05, height * .03], [width * .95, 0], [width, height], [0, height * .98]])
        arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (width, height), borderValue=90)
    arr = arr * 0.88 + 12 + np.random.default_rng(seed).normal(0, 6, arr.shape)
    photo = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    if blur:
        photo = photo.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=85)
    return Image.open(io.BytesIO(buf.getvalue()))


def ocr(photo: Image.Image, prep: bool) -> str:
    if not prep:
        return pytesseract.image_to_string(photo, lang="fra")
    gray = cv2.resize(np.array(photo.convert("L")), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(Image.fromarray(binary), lang="fra", config="--psm 6")


def main() -> None:
    clean = receipt_image()
    for prep in (False, True):
        exact = 0
        wrong_digits = 0
        cases = [(a, b, p) for a in (0, 3, 8) for b in (0, 1.2, 2.0) for p in (False, True)]
        for seed, (angle, blur, persp) in enumerate(cases):
            text = ocr(photograph(clean, angle, blur, persp, seed), prep)
            read = [a for line in text.splitlines() for a in line_amounts(line)]
            exact += read == EXPECTED
            # An amount read that is NOT on the receipt = a misread digit (the FP risk).
            wrong_digits += any(a not in EXPECTED for a in read)
        label = "prep" if prep else "raw "
        print(f"{label}: {exact}/{len(cases)} exact, {wrong_digits}/{len(cases)} with an invented amount")


if __name__ == "__main__":
    main()
