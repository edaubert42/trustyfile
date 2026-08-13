"""Format-agnostic detection of deleted (free) PDF objects — ghost data.

The old implementation grepped the raw bytes for classic ASCII xref free
entries (``nnnnnnnnnn ggggg f``). Two proven failures:

- PDF 1.5+ cross-reference STREAMS store entries as packed binary inside a
  compressed stream — no literal ``f`` anywhere, so the regex saw nothing
  even when free objects genuinely existed.
- Every classic xref section starts with the spec-mandated free-list head
  entry for object 0, which the regex counted as a "deletion" — one unit of
  built-in noise per revision.

`find_free_xrefs` resolves through MuPDF's xref abstraction instead, which
understands both formats. These tests pin all of that behaviour.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.modules.structure import check_object_streams, find_free_xrefs


def _build_classic_pdf_with_free_slots(path: Path, n_free: int) -> None:
    """Hand-craft a minimal classic-xref PDF with `n_free` real free slots.

    Objects 1-3 are a valid catalog/pages/page chain; objects 4..3+n_free
    exist only as type-0 (free) xref entries chained per spec (each free
    entry's offset field points to the next free object, the last points
    back to 0). Offsets are computed exactly so no repair pass is needed.
    """
    header = b"%PDF-1.4\n"
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>\nendobj\n",
    ]
    offsets = []
    pos = len(header)
    for obj in objs:
        offsets.append(pos)
        pos += len(obj)
    xref_pos = pos

    total = 4 + n_free
    xref = [f"xref\n0 {total}\n".encode()]
    first_free = 4 if n_free else 0
    xref.append(f"{first_free:010d} 65535 f \n".encode())
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n".encode())
    for i in range(n_free):
        nxt = 4 + i + 1 if i < n_free - 1 else 0
        xref.append(f"{nxt:010d} 00001 f \n".encode())
    trailer = (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    path.write_bytes(header + b"".join(objs) + b"".join(xref) + trailer)


class TestFindFreeXrefs:
    def test_clean_pdf_has_none(self, tmp_path):
        """Object 0's mandatory head entry must NOT count as a deletion."""
        p = tmp_path / "clean.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice 100 EUR")
        doc.save(str(p))
        doc.close()

        with fitz.open(str(p)) as doc:
            assert find_free_xrefs(doc) == []

    def test_classic_xref_free_slots_found(self, tmp_path):
        p = tmp_path / "free.pdf"
        _build_classic_pdf_with_free_slots(p, n_free=3)
        with fitz.open(str(p)) as doc:
            assert find_free_xrefs(doc) == [4, 5, 6]

    def test_inuse_null_object_counted_by_design(self, tmp_path):
        """An in-use object whose body is the literal ``null`` keyword is
        counted as ghost data ON PURPOSE: nulling an object's body is how
        some editors soft-delete content without renumbering. PyMuPDF
        cannot distinguish it from a free slot anyway (both return "null");
        this test pins that the ambiguity is accepted, not accidental."""
        header = b"%PDF-1.4\n"
        objs = [
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>\nendobj\n",
            b"4 0 obj\nnull\nendobj\n",
        ]
        offsets, pos = [], len(header)
        for obj in objs:
            offsets.append(pos)
            pos += len(obj)
        xref = [b"xref\n0 5\n", b"0000000000 65535 f \n"]
        xref += [f"{off:010d} 00000 n \n".encode() for off in offsets]
        trailer = (f"trailer\n<< /Size 5 /Root 1 0 R >>\n"
                   f"startxref\n{pos}\n%%EOF\n").encode()
        p = tmp_path / "inuse_null.pdf"
        p.write_bytes(header + b"".join(objs) + b"".join(xref) + trailer)

        with fitz.open(str(p)) as doc:
            assert find_free_xrefs(doc) == [4]

    def test_xref_stream_free_slots_found(self, tmp_path):
        """The case the byte-regex was blind to: free objects inside a
        PDF 1.5+ file whose xref is a compressed stream."""
        base = tmp_path / "base.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i}")
        doc.save(str(base), use_objstms=1)
        doc.close()

        # The edit must happen on the ON-DISK objstm file: only then does
        # MuPDF keep the original numbering and mark the deleted page's
        # objects free (an in-memory build gets renumbered on first save).
        p = tmp_path / "objstm.pdf"
        doc = fitz.open(str(base))
        doc.delete_page(1)
        doc.save(str(p), use_objstms=1)
        doc.close()

        raw = p.read_bytes()
        with fitz.open(str(p)) as doc:
            free = find_free_xrefs(doc)
        assert free, "deleting a page then saving as objstm must leave free slots"
        # The literal classic-format free entry must be absent from the file:
        # proves this fixture really exercises the xref-STREAM format.
        import re
        assert not re.findall(rb"\d{10}\s+\d{5}\s+f", raw)


class TestCheckObjectStreamsThreshold:
    def test_flag_fires_above_unsigned_threshold(self, tmp_path):
        p = tmp_path / "many_free.pdf"
        _build_classic_pdf_with_free_slots(p, n_free=6)  # threshold is 5
        flags = check_object_streams(str(p), has_trusted_signature=False)
        assert [f.code for f in flags] == ["STRUCT_DELETED_OBJECTS"]
        assert flags[0].severity == "medium"
        assert flags[0].details["free_objects"] == 6
        assert flags[0].details["free_xrefs"] == [4, 5, 6, 7, 8, 9]

    def test_signed_document_gets_higher_threshold(self, tmp_path):
        """6 free objects: flags unsigned, tolerated when trusted-signed."""
        p = tmp_path / "many_free.pdf"
        _build_classic_pdf_with_free_slots(p, n_free=6)
        assert check_object_streams(str(p), has_trusted_signature=True) == []

    def test_at_threshold_no_flag(self, tmp_path):
        p = tmp_path / "at_threshold.pdf"
        _build_classic_pdf_with_free_slots(p, n_free=5)
        assert check_object_streams(str(p), has_trusted_signature=False) == []

    def test_unreadable_file_no_flags(self, tmp_path):
        p = tmp_path / "garbage.pdf"
        p.write_bytes(b"not a pdf at all")
        assert check_object_streams(str(p)) == []
