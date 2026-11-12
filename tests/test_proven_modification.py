"""Proven-modification rule: a file that is provably not the issuer's original
lands in SUSPICIOUS, whatever the sum of the other hints says.

Plan: docs/superpowers/plans/2026-11-12-scoring-provenance-rules.md (step 1).
"""

from src.evidence import PROVEN_MODIFICATION_CEILING, aggregate_evidence, flag_to_evidence
from src.models import Flag, ModuleResult
from src.scoring import create_analysis_result, proven_modification_codes


def _flag(code: str, severity: str = "high") -> Flag:
    return Flag(severity=severity, code=code, message=code)


def _result(flags: list[Flag], document_type: str = "invoice", **kw):
    return create_analysis_result(
        "hash", [ModuleResult(module="m", flags=flags, score=50, confidence=1.0)],
        document_type=document_type, **kw)


# --- the decision -------------------------------------------------------------

def test_revision_text_change_counts_for_any_genre():
    flags = [_flag("OVERLAY_REVISION_TEXT_CHANGED")]
    assert proven_modification_codes(flags, "scientific") == ["OVERLAY_REVISION_TEXT_CHANGED"]


def test_online_converter_counts_on_invoice_only():
    flags = [_flag("META_ONLINE_CONVERTER")]
    assert proven_modification_codes(flags, "invoice") == ["META_ONLINE_CONVERTER"]
    assert proven_modification_codes(flags, "scientific") == []


def test_short_date_gap_is_a_pipeline_not_a_proof():
    # medium = ModDate under 30 minutes after CreationDate (generate-then-view)
    assert proven_modification_codes([_flag("META_DOCUMENT_MODIFIED", "medium")], "invoice") == []
    assert proven_modification_codes([_flag("META_DOCUMENT_MODIFIED", "high")], "invoice") \
        == ["META_DOCUMENT_MODIFIED"]


def test_signature_excuses_the_date_gap_but_not_a_converter():
    signed = [_flag("STRUCT_SIGNATURE_UNVERIFIED", "low"), _flag("META_DOCUMENT_MODIFIED")]
    assert proven_modification_codes(signed, "invoice") == []
    signed.append(_flag("META_ONLINE_CONVERTER"))
    assert proven_modification_codes(signed, "invoice") == ["META_ONLINE_CONVERTER"]


def test_invalid_signature_is_no_excuse():
    flags = [_flag("STRUCT_SIGNATURE_INVALID"), _flag("META_DOCUMENT_MODIFIED")]
    assert proven_modification_codes(flags, "invoice") == ["META_DOCUMENT_MODIFIED"]


def test_scan_of_unknown_genre_still_counts():
    # A blind document has no text, so its genre is always "unknown".
    flags = [_flag("META_DOCUMENT_MODIFIED")]
    assert proven_modification_codes(flags, "unknown") == []
    assert proven_modification_codes(flags, "unknown", coverage_blind=True) \
        == ["META_DOCUMENT_MODIFIED"]


def test_low_severity_never_counts():
    flags = [_flag("STRUCT_HIDDEN_MODIFICATION", "low")]
    assert proven_modification_codes(flags, "invoice") == []


# --- the band -----------------------------------------------------------------

def test_single_converter_flag_is_suspicious_not_genuine():
    # Before the rule this exact ledger scored 92/GENUINE (investigation doc 1).
    r = _result([_flag("META_ONLINE_CONVERTER")])
    assert r.band == "SUSPICIOUS"
    assert r.trust_score == PROVEN_MODIFICATION_CEILING


def test_ceiling_never_raises_a_lower_score():
    ev = [flag_to_evidence(_flag("OVERLAY_REVISION_VALUE_CHANGED", "critical"), "m"),
          flag_to_evidence(_flag("OVERLAY_COVER_AND_RETYPE_CONFIRMED", "critical"), "m")]
    agg = aggregate_evidence(ev, proven_modification=True)
    assert agg.score <= 15


def test_modified_scan_is_suspicious_not_unverifiable():
    # Investigation doc 2: scan re-saved in Acrobat 844 days later.
    r = _result([_flag("META_DOCUMENT_MODIFIED")], document_type="unknown",
                coverage_blind=True)
    assert r.band == "SUSPICIOUS"


def test_same_flag_on_a_cv_keeps_its_old_band():
    r = _result([_flag("META_DOCUMENT_MODIFIED")], document_type="unknown")
    assert r.band == "GENUINE"


# --- e-invoice exemption ------------------------------------------------------

def test_einvoice_xml_excuses_the_date_gap():
    # Factur-X toolkits attach the XML days after creating the PDF.
    flags = [_flag("META_DOCUMENT_MODIFIED"), _flag("STRUCT_XMP_EDITOR_MISMATCH")]
    assert proven_modification_codes(flags, "invoice", einvoice_xml=True) == []


def test_einvoice_excuse_void_when_xml_contradicts_the_page():
    flags = [_flag("META_DOCUMENT_MODIFIED"), _flag("ARITH_XML_VISIBLE_MISMATCH", "critical")]
    assert proven_modification_codes(flags, "invoice", einvoice_xml=True) \
        == ["META_DOCUMENT_MODIFIED"]


def test_einvoice_never_excuses_a_converter():
    flags = [_flag("META_ONLINE_CONVERTER")]
    assert proven_modification_codes(flags, "invoice", einvoice_xml=True) \
        == ["META_ONLINE_CONVERTER"]


def test_einvoice_flag_is_read_from_module_details():
    mods = [ModuleResult(module="metadata", flags=[_flag("META_DOCUMENT_MODIFIED")],
                         score=50, confidence=1.0),
            ModuleResult(module="arithmetic", flags=[], score=100, confidence=1.0,
                         details={"einvoice_xml": True})]
    r = create_analysis_result("hash", mods, document_type="invoice")
    assert r.band == "GENUINE"


# --- virtual printer: origin unverifiable -> never GENUINE, never forced lower --

def test_virtual_printer_invoice_is_review_not_genuine():
    result = _result([_flag("META_VIRTUAL_PRINTER", "low")], "invoice")
    assert result.band == "REVIEW"


def test_virtual_printer_on_a_cv_changes_nothing():
    alone = _result([], "unknown")
    printed = _result([_flag("META_VIRTUAL_PRINTER", "low")], "unknown")
    assert (printed.band, printed.trust_score) == (alone.band, alone.trust_score)


def test_virtual_printer_plus_date_gap_is_suspicious():
    # EDF case: Print To PDF, then modified 36 minutes later.
    flags = [_flag("META_VIRTUAL_PRINTER", "low"), _flag("META_DOCUMENT_MODIFIED")]
    assert _result(flags, "invoice").band == "SUSPICIOUS"


def test_virtual_printer_detection():
    from src.modules.metadata import check_virtual_printer
    assert [f.code for f in check_virtual_printer("Microsoft: Print To PDF", None)] \
        == ["META_VIRTUAL_PRINTER"]
    assert [f.code for f in check_virtual_printer("PDFCreator 2.5.3", None)] \
        == ["META_VIRTUAL_PRINTER"]
    assert check_virtual_printer("Skia/PDF m120", "Chromium") == []
