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


def test_garbled_metadata_date_on_invoice_is_suspicious():
    assert _result([_flag("META_MALFORMED_DATE")], "invoice").band == "SUSPICIOUS"


# --- editor trace on an unknown genre -> REVIEW (edited_unknown_codes) -------

def test_acrobat_edit_on_unknown_genre_is_review_not_genuine():
    # Restaurant invoice without the word "facture", re-saved in Acrobat:
    # scored GENUINE 81 while classified "unknown" (2026-11-12).
    flags = [_flag("STRUCT_XMP_EDITOR_MISMATCH"), _flag("META_DOCUMENT_MODIFIED", "medium"),
             _flag("FONTS_MIDLINE_CHANGE", "medium")]
    r = _result(flags, "unknown")
    assert r.band == "REVIEW"
    assert r.forced_codes == {"STRUCT_XMP_EDITOR_MISMATCH": "REVIEW"}


def test_online_converter_on_unknown_genre_is_review():
    assert _result([_flag("META_ONLINE_CONVERTER")], "unknown").band == "REVIEW"


def test_editor_trace_on_a_known_non_issued_genre_changes_nothing():
    # A scientific article reworked by its publisher in Acrobat stays as before
    alone = _result([], "scientific")
    edited = _result([_flag("STRUCT_XMP_EDITOR_MISMATCH")], "scientific")
    assert edited.forced_codes == {}
    assert edited.band == alone.band


def test_einvoice_xml_still_excuses_acrobat_on_unknown_genre():
    # ZUGFeRD toolkits re-save the PDF to attach the XML
    r = create_analysis_result(
        "hash", [ModuleResult(module="arithmetic", flags=[_flag("STRUCT_XMP_EDITOR_MISMATCH")],
                              score=50, confidence=1.0, details={"einvoice_xml": True})],
        document_type="unknown")
    assert r.forced_codes == {}



# --- Acrobat edit marker, and the verified 2D-Doc that excuses it (2026-11-13) ---

_BARCODE = Flag(severity="medium", code="TWOD_DOC_SIGNATURE_VALID", message="valid")


def test_acrobat_text_edit_on_invoice_is_suspicious():
    res = _result([_flag("STRUCT_ACROBAT_TEXT_EDIT")])
    assert res.band == "SUSPICIOUS"
    assert res.forced_codes == {"STRUCT_ACROBAT_TEXT_EDIT": "SUSPICIOUS"}


def test_acrobat_text_edit_on_unknown_genre_is_review():
    res = _result([_flag("STRUCT_ACROBAT_TEXT_EDIT")], document_type="unknown")
    assert res.band == "REVIEW"


def test_acrobat_text_edit_on_an_article_is_only_a_hint():
    res = _result([_flag("STRUCT_ACROBAT_TEXT_EDIT")], document_type="scientific")
    assert res.band == "GENUINE"
    assert res.forced_codes == {}


def test_verified_barcode_excuses_the_acrobat_marker():
    # The tax notice case: template labels retouched in Acrobat by the issuer.
    with_marker = _result([_flag("STRUCT_ACROBAT_TEXT_EDIT"), _BARCODE],
                          document_type="tax_document")
    barcode_only = _result([_BARCODE], document_type="tax_document")
    assert with_marker.band == "GENUINE"
    assert with_marker.forced_codes == {}
    # The excused flag does not weigh either: same score as without it.
    assert with_marker.trust_score == barcode_only.trust_score


def test_verified_barcode_excuses_every_origin_signal():
    flags = [_flag("META_ONLINE_CONVERTER"), _flag("META_DOCUMENT_MODIFIED"),
             _flag("META_VIRTUAL_PRINTER", "low"), _BARCODE]
    res = _result(flags, document_type="tax_document")
    assert res.band == "GENUINE"
    assert res.forced_codes == {}


def test_verified_barcode_does_not_excuse_a_proven_content_change():
    # The barcode does not cover every field: text that changed between two
    # revisions is still proof, whatever the barcode says.
    res = _result([_flag("OVERLAY_REVISION_TEXT_CHANGED"), _BARCODE],
                  document_type="tax_document")
    assert res.band == "SUSPICIOUS"


def test_excused_flag_is_still_shown_to_the_reader():
    res = _result([_flag("STRUCT_ACROBAT_TEXT_EDIT"), _BARCODE],
                  document_type="tax_document")
    shown = [f.code for m in res.modules for f in m.flags]
    assert "STRUCT_ACROBAT_TEXT_EDIT" in shown


# --- rewrite engines (2026-11-13): PyMuPDF / qpdf bytes ------------------------

def test_file_rewritten_on_invoice_is_suspicious():
    res = _result([_flag("PROVENANCE_FILE_REWRITTEN")])
    assert res.band == "SUSPICIOUS"
    assert res.forced_codes == {"PROVENANCE_FILE_REWRITTEN": "SUSPICIOUS"}


def test_file_rewritten_on_unknown_genre_is_review():
    assert _result([_flag("PROVENANCE_FILE_REWRITTEN")], document_type="unknown").band == "REVIEW"


def test_file_rewritten_on_an_article_is_only_a_hint():
    res = _result([_flag("PROVENANCE_FILE_REWRITTEN")], document_type="scientific")
    assert res.band == "GENUINE"
    assert res.forced_codes == {}


def test_writer_unverified_invoice_is_review_not_genuine():
    res = _result([_flag("PROVENANCE_WRITER_UNVERIFIED", "low")], "invoice")
    assert res.band == "REVIEW"
    assert res.forced_codes == {"PROVENANCE_WRITER_UNVERIFIED": "REVIEW"}


def test_writer_unverified_outside_issued_documents_changes_nothing():
    for genre in ("unknown", "scientific"):
        alone = _result([], genre)
        flagged = _result([_flag("PROVENANCE_WRITER_UNVERIFIED", "low")], genre)
        assert (flagged.band, flagged.trust_score) == (alone.band, alone.trust_score), genre


def test_verified_barcode_excuses_both_rewrite_codes():
    for code, severity in (("PROVENANCE_FILE_REWRITTEN", "high"),
                           ("PROVENANCE_WRITER_UNVERIFIED", "low")):
        res = _result([_flag(code, severity), _BARCODE], document_type="tax_document")
        assert res.band == "GENUINE", code
        assert res.forced_codes == {}, code


# --- marks other editors leave in the page (2026-11-13) -------------------------

def test_editor_marks_on_invoice_are_suspicious():
    for code in ("STRUCT_EDITOR_MARKER", "STRUCT_EMPTY_TEXT_OPERATOR"):
        res = _result([_flag(code)])
        assert res.band == "SUSPICIOUS", code
        assert res.forced_codes == {code: "SUSPICIOUS"}, code


def test_editor_marks_on_unknown_genre_are_review():
    for code in ("STRUCT_EDITOR_MARKER", "STRUCT_EMPTY_TEXT_OPERATOR"):
        assert _result([_flag(code)], document_type="unknown").band == "REVIEW", code


def test_editor_marks_on_an_article_are_only_a_hint():
    for code in ("STRUCT_EDITOR_MARKER", "STRUCT_EMPTY_TEXT_OPERATOR"):
        res = _result([_flag(code)], document_type="scientific")
        assert res.band == "GENUINE", code
        assert res.forced_codes == {}, code
