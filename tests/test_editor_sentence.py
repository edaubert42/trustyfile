"""The editor-mismatch sentence names the tools and stays neutral on a scan.

Before 2026-11-12 it read "The document was modified by a different tool than
its creator" even on a ScanSnap scan whose two dates are identical.
"""
from src.models import Flag
from src.summary import get_flag_sentence


def _flag(**details):
    return Flag("medium", "STRUCT_XMP_EDITOR_MISMATCH", "m", details)


def test_scan_sentence_is_a_neutral_fact():
    flag = _flag(is_scanned=True, detected_editor="Adobe Acrobat")
    assert get_flag_sentence(flag, "fr") == "Le scan porte la trace d'un passage par Adobe Acrobat."
    assert get_flag_sentence(flag, "en") == "The scan bears the trace of Adobe Acrobat."


def test_document_sentence_names_both_tools():
    flag = _flag(is_scanned=False, detected_editor="Adobe Acrobat",
                 original_producer="macOS Quartz PDFContext")
    assert get_flag_sentence(flag, "fr") == (
        "Créé avec macOS Quartz PDFContext, puis enregistré avec Adobe Acrobat.")
