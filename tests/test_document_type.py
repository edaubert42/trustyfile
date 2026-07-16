"""Tests for the keyword document-type classifier (audit item 24)."""
from src.document_type import detect_document_type


class FakePDFData:
    def __init__(self, text: str):
        self.text_by_page = [text]


def test_clear_winner_classifies():
    text = "facture total ttc tva iban montant dû"
    assert detect_document_type(FakePDFData(text)) == "invoice"


def test_tie_returns_unknown():
    """3 invoice keywords vs 3 certificate keywords -> ambiguous -> unknown.

    Previously max() silently took the first dict key (audit item 24: the
    docstring promised tie->unknown; now the code delivers it).
    """
    text = ("facture total ttc tva "                       # 3 invoice hits
            "certificat certifie que homologation")        # 3 certificate hits
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_below_min_keywords_unknown():
    assert detect_document_type(FakePDFData("facture tva")) == "unknown"


def test_empty_text_unknown():
    assert detect_document_type(FakePDFData("")) == "unknown"
