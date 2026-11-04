"""Signal 4 (overlay) — row-level revision diff.

Each test builds a small PDF, edits it in an INCREMENTAL save (so the earlier
revision is still in the file), and checks what `_signal_revision_value_changed`
reports. No dependency on the gitignored samples/ corpus: these run in CI.
"""
from types import SimpleNamespace

import fitz
import pytest

from src.modules.overlay import (
    _is_value_change,
    _signal_revision_value_changed,
)

LINES = [
    "FACTURE N 2026-0412",
    "Atelier Exemple SAS",
    "8 rue des Lilas, 69003 Lyon",
    "Client :",
    "Louise Martin",
    "11 allee des Peupliers, 92380 Garches",
    "Date : 14/10/2026",
    "Montant HT : 1 000,00 EUR",
    "TVA 20 % : 200,00 EUR",
    "Total TTC : 1 200,00 EUR",
    "Net a payer : 1 200,00 EUR",
]


def _base(path, lines=LINES):
    """Write a one-page PDF, one line every 18 pt, and return its path."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i, text in enumerate(lines):
        page.insert_text((60, 80 + 18 * i), text, fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _replace(page, old, new, bold=False, occurrence=0):
    """Really remove `old` (redaction) and type `new` at the same spot."""
    rect = page.search_for(old)[occurrence]
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    page.insert_text((rect.x0, rect.y1 - rect.height * 0.22), new,
                     fontsize=12, fontname="hebo" if bold else "helv")


def _flags(path):
    raw = path.read_bytes()
    return _signal_revision_value_changed(SimpleNamespace(file_path=str(path)), raw)


def test_amount_retyped_wider_is_caught_with_full_amount(tmp_path):
    # Bold retype is wider than the original: "EUR" moves ~4 pt to the right.
    # The old word-by-word pairing rejected this (neighbour not at the same
    # spot); the row diff sees "1 200,00" -> "4 200,00" on a stable page.
    pdf = _base(tmp_path / "a.pdf")
    doc = fitz.open(str(pdf))
    _replace(doc[0], "1 200,00 EUR", "4 200,00 EUR", bold=True)
    doc.saveIncr()
    doc.close()

    flags = _flags(pdf)
    assert [f.code for f in flags] == ["OVERLAY_REVISION_VALUE_CHANGED"]
    assert flags[0].details["old_value"] == "1 200,00"
    assert flags[0].details["new_value"] == "4 200,00"
    assert flags[0].details["delta"] == pytest.approx(3000.0)


def test_one_of_two_repeated_totals_changed_is_still_caught(tmp_path):
    # "1 200,00 EUR" is printed twice (Total TTC + Net a payer). The forger
    # edits only the first: the old text still exists once in the new
    # revision, but it existed TWICE before — a copy was removed.
    pdf = _base(tmp_path / "b.pdf")
    doc = fitz.open(str(pdf))
    _replace(doc[0], "1 200,00 EUR", "4 200,00 EUR", occurrence=0)
    doc.saveIncr()
    doc.close()

    codes = [f.code for f in _flags(pdf)]
    assert codes == ["OVERLAY_REVISION_VALUE_CHANGED"]


def test_name_and_address_change_is_one_text_flag(tmp_path):
    pdf = _base(tmp_path / "c.pdf")
    doc = fitz.open(str(pdf))
    _replace(doc[0], "Louise Martin", "Philippe Martin")
    _replace(doc[0], "11 allee des Peupliers, 92380 Garches",
             "26 avenue de la Grande Armee, 75017 Paris")
    doc.saveIncr()
    doc.close()

    flags = _flags(pdf)
    assert [f.code for f in flags] == ["OVERLAY_REVISION_TEXT_CHANGED"]
    assert flags[0].severity == "high"                  # SOFT, never HARD
    olds = [i["old"] for i in flags[0].details["instances"]]
    assert olds == ["Louise", "11 allee des Peupliers, 92380 Garches"]
    for inst in flags[0].details["instances"]:          # one preview box per row
        assert inst["page"] == 0 and len(inst["bbox"]) == 4


def test_rows_pushed_down_by_an_inserted_line_are_not_reported(tmp_path):
    # Real case (invoice_FRW): a company line is inserted under the client's
    # name, every following line moves down one row. Only the line that
    # really disappeared may be reported — not the ones that merely moved.
    lines = LINES[:6] + ["", ""] + LINES[6:]            # spare room at the bottom
    old = _base(tmp_path / "old.pdf", lines)
    doc = fitz.open(str(old))
    page = doc[0]
    # Remove the address row and the date row, re-type: new company line, then the
    # address one row LOWER (where the date was), then the date one row lower still.
    for text in ("11 allee des Peupliers, 92380 Garches", "Date : 14/10/2026"):
        page.add_redact_annot(page.search_for(text)[0], fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    y_addr = 80 + 18 * 5
    page.insert_text((60, y_addr), "MONNERET FORMATION", fontname="helv", fontsize=12)
    page.insert_text((60, y_addr + 18), "11 allee des Peupliers, 92380 Garches",
                     fontname="helv", fontsize=12)
    page.insert_text((60, y_addr + 36), "Date : 14/10/2026", fontname="helv", fontsize=12)
    doc.saveIncr()
    doc.close()

    assert _flags(old) == []     # nothing was removed: rows only moved / were added


def test_added_text_only_is_not_reported(tmp_path):
    # A "PAYE" stamp added in a later revision: nothing was replaced or removed.
    pdf = _base(tmp_path / "d.pdf")
    doc = fitz.open(str(pdf))
    doc[0].insert_text((400, 260), "PAYE le 20/10/2026", fontname="helv", fontsize=12)
    doc.saveIncr()
    doc.close()

    assert _flags(pdf) == []


def test_reflowed_page_is_not_diffed(tmp_path):
    # Second revision is a different document written over the first (a
    # generator reusing the file): same row heights, new text on 8 rows out
    # of 11. Only 3 rows are stable (27 % < 75 %), so the reflow guard refuses
    # to diff — a row-by-row comparison here would invent eight "changes".
    pdf = _base(tmp_path / "e.pdf")
    doc = fitz.open(str(pdf))
    page = doc[0]
    for text in LINES[3:]:
        page.add_redact_annot(page.search_for(text)[0], fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    for i in range(3, len(LINES)):
        page.insert_text((60, 80 + 18 * i), f"Ligne {i} contenu {i * 37} nouveau",
                         fontname="helv", fontsize=12)
    doc.saveIncr()
    doc.close()

    assert _flags(pdf) == []


def test_date_change_has_no_money_delta(tmp_path):
    # money_value() reads "14/10/2026" as 14.0: a date change used to be
    # reported as "a change of +0.00". Dates carry no delta.
    pdf = _base(tmp_path / "f.pdf")
    doc = fitz.open(str(pdf))
    _replace(doc[0], "14/10/2026", "14/10/2025")
    doc.saveIncr()
    doc.close()

    flags = _flags(pdf)
    assert [f.code for f in flags] == ["OVERLAY_REVISION_VALUE_CHANGED"]
    assert "delta" not in flags[0].details
    assert "change of" not in flags[0].message


@pytest.mark.parametrize("old, new, expected", [
    ("1 200,00", "4 200,00", True),
    ("19/08/1992", "19/08/ 1998", True),
    ("200,00 EUR", "900,00 EUR", True),      # currency words allowed
    ("92380 Garches", "75017 Paris", False),  # town names are words
    ("Louise", "Philippe", False),            # no digit at all
])
def test_is_value_change(old, new, expected):
    assert _is_value_change(old, new) is expected
