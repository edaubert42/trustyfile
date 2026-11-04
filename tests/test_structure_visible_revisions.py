"""Incremental updates that change nothing visible (metadata, attachments).

Real case: Intarsys ZUGFeRD sample invoices carry 4 incremental updates that
rewrite only the Info dictionary / XMP and attach the Factur-X XML; every
revision renders pixel-identical. They scored REVIEW ("edited 4 times").
"""
import fitz

from src.modules.structure import (
    _metadata_only_modification,
    check_incremental_updates,
    original_revision_unmodified,
    visible_content_unchanged,
)
from src.models import Flag

CREATED = "D:20140624120121+00'00'"


def _original(path, pages=1, mod_date=""):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 100), f"RECHNUNG page {i + 1}", fontname="helv", fontsize=14)
        page.insert_text((60, 140), "Gesamtbetrag 1.200,00 EUR", fontname="helv", fontsize=11)
    doc.set_metadata({"producer": "LibreOffice 4.2", "creator": "Calc",
                      "creationDate": CREATED, "modDate": mod_date})
    doc.save(str(path))
    doc.close()
    return path


def _metadata_updates(path, times=3, attach=True):
    """Re-save `times` times touching only metadata (and attach an XML file)."""
    for i in range(times):
        doc = fitz.open(str(path))
        doc.set_metadata({**doc.metadata, "modDate": f"D:2014100615{i:02d}00+00'00'"})
        if attach and i == times - 1:
            doc.embfile_add("ZUGFeRD-invoice.xml", b"<rsm:CrossIndustryDocument/>")
        doc.saveIncr()
        doc.close()


def _content_updates(path, times=3):
    """Re-save `times` times, each time changing what the page shows."""
    for i in range(times):
        doc = fitz.open(str(path))
        doc[0].insert_text((60, 200 + 20 * i), f"edit {i}", fontname="helv", fontsize=11)
        doc.saveIncr()
        doc.close()


def _incr_flag(path):
    return [f for f in check_incremental_updates(str(path), verify_signatures=False)
            if f.code == "STRUCT_INCREMENTAL_UPDATES"]


def test_metadata_only_updates_are_low(tmp_path):
    pdf = _original(tmp_path / "meta.pdf")
    _metadata_updates(pdf)
    assert visible_content_unchanged(str(pdf)) is True
    flags = _incr_flag(pdf)
    assert [f.severity for f in flags] == ["low"]
    assert flags[0].details["visible_content_changed"] is False


def test_content_updates_keep_their_weight(tmp_path):
    pdf = _original(tmp_path / "edit.pdf")
    _content_updates(pdf)
    assert visible_content_unchanged(str(pdf)) is False
    flags = _incr_flag(pdf)
    assert [f.severity for f in flags] == ["critical"]
    assert flags[0].details["visible_content_changed"] is True


def test_no_revision_means_unknown(tmp_path):
    assert visible_content_unchanged(str(_original(tmp_path / "plain.pdf"))) is None


def test_too_many_pages_means_unknown(tmp_path):
    pdf = _original(tmp_path / "long.pdf", pages=11)
    _metadata_updates(pdf, times=1, attach=False)
    assert visible_content_unchanged(str(pdf)) is None


MODIFIED = Flag(severity="high", code="META_DOCUMENT_MODIFIED",
                message="Document was modified 104 days after creation",
                details={"time_difference_hours": 2498.2})


def test_modified_flag_downgraded_when_provably_invisible(tmp_path):
    pdf = _original(tmp_path / "meta2.pdf")          # original: no ModDate
    _metadata_updates(pdf)
    assert original_revision_unmodified(str(pdf)) is True
    [flag] = _metadata_only_modification([MODIFIED], str(pdf))
    assert flag.severity == "low"
    assert flag.details["visible_content_changed"] is False
    assert MODIFIED.severity == "high"                # caller's flag not mutated


def test_modified_flag_kept_when_the_original_was_already_modified(tmp_path):
    # The original revision itself says "modified 3 months later": that edit
    # happened before any revision we can compare, so nothing is proven.
    pdf = _original(tmp_path / "meta3.pdf", mod_date="D:20140924120121+00'00'")
    _metadata_updates(pdf)
    assert original_revision_unmodified(str(pdf)) is False
    [flag] = _metadata_only_modification([MODIFIED], str(pdf))
    assert flag.severity == "high"


def test_modified_flag_kept_when_content_changed(tmp_path):
    pdf = _original(tmp_path / "edit2.pdf")
    _content_updates(pdf)
    [flag] = _metadata_only_modification([MODIFIED], str(pdf))
    assert flag.severity == "high"
