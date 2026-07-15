"""
Module-level tests for analyze_twod_doc (audit item 14).

Uses parse_twod_doc on synthetic raw payloads (the same construction as the
module's __main__ harness) — no DataMatrix decoding, so these run without
libdmtx. The decode-dependent E2E lives in test_twod_doc_e2e.py.
"""
from src.models import ModuleResult
from src.modules.twod_doc import (
    GS,
    US,
    _flags_from_verification,
    analyze_twod_doc,
    parse_twod_doc,
    verify_2d_doc_against_pdf,
)

# Header: DC + version 03 + FR00 + 0001 + emit 2345 + sig 2345 + type 01 + perimeter 01
_HEADER = "DC03FR000001234523450101"


def _make_twod_doc(fields: dict[str, str]):
    """Build a parsed TwoDocData from {DI: value} using the real parser."""
    message = ""
    for di, value in fields.items():
        message += di + value
        if di != "24":                      # DI 24 (postal code) is fixed-length
            message += GS
    raw = _HEADER + message.rstrip(GS) + US + "FAKESIG"
    td = parse_twod_doc(raw)
    assert td is not None, "synthetic 2D-Doc failed to parse — fix the fixture"
    return td


MATCHING_TEXT = """AVIS D'IMPOT
JEAN DUPONT
123 RUE DE PARIS
75001 PARIS
Facture FAC-2024-001
"""


class TestFlagsFromVerification:
    def test_all_match_yields_consistent(self):
        td = _make_twod_doc({"10": "JEAN DUPONT", "25": "PARIS", "18": "FAC-2024-001"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        assert [f.code for f in flags] == ["TWOD_DOC_CONSISTENT"]
        assert flags[0].severity == "low"

    def test_single_field_no_consistent(self):
        """<2 comparisons must NOT earn the genuine flag (spec D2)."""
        td = _make_twod_doc({"25": "PARIS"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        assert _flags_from_verification(res) == []

    def test_fuzzy_field_mismatch_is_medium(self):
        """Name (confidence < 0.95) mismatch -> medium (spec D1)."""
        td = _make_twod_doc({"10": "MARIE MARTIN", "25": "PARIS", "18": "FAC-2024-001"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        mismatches = [f for f in flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].severity == "medium"
        assert mismatches[0].details["di"] == "10"
        assert "TWOD_DOC_CONSISTENT" not in [f.code for f in flags]

    def test_exact_field_mismatch_is_high(self):
        """Invoice number (confidence 1.0) mismatch -> high (spec D1)."""
        td = _make_twod_doc({"10": "JEAN DUPONT", "18": "FAC-9999-999"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        mismatches = [f for f in flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].severity == "high"

    def test_postal_code_mismatch_is_high(self):
        td = _make_twod_doc({"24": "69001", "25": "PARIS", "10": "JEAN DUPONT"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        highs = [f for f in _flags_from_verification(res)
                 if f.code == "TWOD_DOC_FIELD_MISMATCH" and f.severity == "high"]
        assert len(highs) == 1


class TestAnalyzeTwodDoc:
    def test_extraction_failed_neutral(self):
        class FakePDFData:
            extraction_ok = False
            file_path = "/nonexistent.pdf"
            text_by_page = []
        r = analyze_twod_doc(FakePDFData())
        assert isinstance(r, ModuleResult)
        assert r.module == "twod_doc" and r.flags == [] and r.score == 100

    def test_no_barcode_neutral(self, tmp_path):
        """A plain text-only PDF has no DataMatrix: neutral, never a penalty."""
        import fitz
        p = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Facture 100,00 EUR")
        doc.save(str(p)); doc.close()

        class FakePDFData:
            extraction_ok = True
            file_path = str(p)
            text_by_page = ["Facture 100,00 EUR"]
        r = analyze_twod_doc(FakePDFData())
        assert r.flags == [] and r.score == 100
        assert r.confidence <= 0.5


class TestEvidenceRegistration:
    def test_consistent_is_genuine_direction(self):
        from src.evidence import GENUINE_CODES, flag_to_evidence
        from src.models import Flag
        assert "TWOD_DOC_CONSISTENT" in GENUINE_CODES
        ev = flag_to_evidence(
            Flag(severity="low", code="TWOD_DOC_CONSISTENT", message="x"),
            "twod_doc")
        assert ev.direction == "genuine" and ev.kind == "soft"

    def test_mismatch_is_soft_tamper(self):
        from src.evidence import HARD_NEGATIVE_CODES, flag_to_evidence
        from src.models import Flag
        assert "TWOD_DOC_FIELD_MISMATCH" not in HARD_NEGATIVE_CODES
        ev = flag_to_evidence(
            Flag(severity="high", code="TWOD_DOC_FIELD_MISMATCH", message="x"),
            "twod_doc")
        assert ev.direction == "tamper" and ev.kind == "soft"


class TestAnalyzerRegistration:
    def test_module_runs_in_pipeline(self, tmp_path):
        """The analyzer must include a twod_doc ModuleResult."""
        import fitz
        p = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "hello facture invoice total ttc tva 20")
        doc.save(str(p)); doc.close()
        from src.analyzer import TrustyFileAnalyzer
        result = TrustyFileAnalyzer(enable_external=False).analyze(str(p))
        assert "twod_doc" in [m.module for m in result.modules]
