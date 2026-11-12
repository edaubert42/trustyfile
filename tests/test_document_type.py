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
    text = ("facture iban échéance "                       # 3 invoice hits
            "certificat certifie que homologation")        # 3 certificate hits
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_below_min_keywords_unknown():
    assert detect_document_type(FakePDFData("facture tva")) == "unknown"


def test_empty_text_unknown():
    assert detect_document_type(FakePDFData("")) == "unknown"


# --- Pattern signals (PATTERNS) ---------------------------------------------

# Visible text of a real restaurant invoice typed in Word (2026-11-12). It
# never says "facture" and has no "total" line, so keywords alone found 2
# signals ("tva", "siret") and it fell into "unknown".
REPAS_DE_GROUPE = (
    "3,rue Saint-Saëns 29 200 BREST Siret : 789397759 00018 "
    "IRD Délégation Régionale Occitanie Numéro de Siret : 18000602500134 "
    "Numéro de TVA intracommunautaire : FR75180006025 "
    "FC A21012026 REPAS DE GROUPE du 21 JANVIER 2026 "
    "15 repas complets 375.14 Euros HT 10 % 293.25 euros TVA 10% 29.33 euros "
    "KERRAC'H SOCIETE sarl - RCS BREST B 789 397 759 - siege 3, rue Saint Saens"
)


def test_invoice_without_the_word_facture_is_an_invoice():
    assert detect_document_type(FakePDFData(REPAS_DE_GROUPE)) == "invoice"


def test_ht_inside_a_word_is_not_a_signal():
    # "light", "https", "night" contain "ht"; only "siret" and "tva" count
    text = "https://example.org light night siret tva"
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_amount_and_legal_form_alone_do_not_make_an_invoice():
    # A tax or legal notice can quote euros and name a SAS: 2 signals, not 3
    text = "Le loyer est de 1 200,00 euros, versé à la société Exemple SAS."
    assert detect_document_type(FakePDFData(text)) == "unknown"
