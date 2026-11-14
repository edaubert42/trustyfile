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


# Over 120 characters, so the keywords that follow sit outside the title area
_FILLER = "Bonjour, veuillez trouver ci-dessous les informations demandées lors de notre " \
          "échange de la semaine dernière, comme convenu ensemble. "


def test_below_min_keywords_unknown():
    assert detect_document_type(FakePDFData(_FILLER + "facture tva")) == "unknown"


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


_UK_PUB = ("The Crown Pub Ltd 12 High Street London. Group dinner 21 January 2026. "
           "15 set menus £375.14 Net £312.62 VAT 20% £62.52. "
           "Registered in England No 01234567. VAT Reg No GB123456789")


def test_english_receipt_without_the_word_invoice_is_an_invoice():
    # No word "invoice", but a bill number: the invoice identity
    assert detect_document_type(FakePDFData("Bill No 4471 " + _UK_PUB)) == "invoice"


# --- Invoice identity (word or number) ---------------------------------------

def test_accounting_signals_without_word_or_number_are_not_an_invoice():
    # VAT, amounts, Ltd, register: also a price grid. No identity -> unknown
    assert detect_document_type(FakePDFData(_UK_PUB)) == "unknown"


def test_price_grid_is_not_an_invoice():
    text = ("1 août 2025 Electricité - Option tarifaire Base. Abonnement mensuel (€) "
            "Prix HT 12,50 € TVA 20% 2,50 €. TotalEnergies SA RCS Nanterre 542 051 180")
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_invoice_number_without_the_word_facture_is_enough():
    assert detect_document_type(FakePDFData(REPAS_DE_GROUPE)) == "invoice"
    assert "facture" not in REPAS_DE_GROUPE.lower()


def test_us_bill_with_sales_tax_is_an_invoice():
    text = "Acme Supplies LLC. Bill 2026-118. 3 widgets $120.00 Sales tax 8.25% $9.90"
    assert detect_document_type(FakePDFData(text)) == "invoice"


def test_vat_inside_a_word_is_not_a_signal():
    # "private", "elevator", "activate" contain "vat"; "incl" is not "inc"
    text = "private elevator activate incl. siret tva"
    assert detect_document_type(FakePDFData(text)) == "unknown"


# --- Whole-word keywords, titles (2026-11-12 rework) -------------------------

def test_short_keyword_inside_a_word_does_not_count():
    # "api" in "rapide", "rest" in "restaurant", "table" in "acceptable"
    text = _FILLER + "un paiement rapide au restaurant, un prix acceptable, figure"
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_keyword_listed_in_two_languages_counts_once():
    # "attestation" is in the French and the English list
    text = _FILLER + "attestation attestation bonne foi"
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_verb_avoir_is_not_a_credit_note():
    text = _FILLER + "vous devez avoir une facture et un siret"
    assert detect_document_type(FakePDFData(text)) == "unknown"
    assert detect_document_type(FakePDFData(_FILLER + "Avoir n° 12 facture siret")) == "invoice"


def test_title_plus_one_keyword_classifies():
    text = "ATTESTATION D'ASSURANCE La compagnie certifie garantir M. Dupont. " + _FILLER \
           + "je soussigné"
    assert detect_document_type(FakePDFData(text)) == "attestation"


def test_document_that_names_itself_is_classified():
    # Title word ("facture", also a keyword) + title bonus = the 3-point minimum
    assert detect_document_type(FakePDFData("FACTURE " + _FILLER)) == "invoice"


def test_received_in_a_letter_opening_is_not_a_title():
    text = "Nous avons bien reçu votre courrier du 3 mars. " + _FILLER
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_title_beats_keywords_quoted_from_another_genre():
    # A technical spec that describes invoice fields
    text = ("Spécifications Techniques Date : 17/06/2024 interopérabilité " + _FILLER
            + "facture numéro de facture")
    assert detect_document_type(FakePDFData(text)) == "technical"


# --- Bank statements and official records (2026-11-13) ----------------------
# Both were "unknown" (capped at REVIEW) instead of an issued type: a known
# forged driving-licence record stayed REVIEW 73.

def test_english_bank_statement_is_a_bank_statement():
    text = ("Customer service information " + _FILLER + "Your Regular Checking "
            "Account summary Beginning balance on July 31 $54.11 Deposits and other "
            "additions 0.00 Withdrawals and other subtractions Ending balance $54.11")
    assert detect_document_type(FakePDFData(text)) == "bank_statement"


def test_french_bank_statement_is_a_bank_statement():
    text = ("RELEVÉ DE COMPTE n° 11 du 01/10/2026 au 31/10/2026 " + _FILLER
            + "Date d'opération Date de valeur Libellé Débit Crédit "
            "Ancien solde 1 204,10 Nouveau solde créditeur 980,55")
    assert detect_document_type(FakePDFData(text)) == "bank_statement"


def test_invoice_with_bank_details_stays_an_invoice():
    # Payment details at the foot of an invoice name an account, not a statement
    text = ("FACTURE n° FA-2026-118 " + _FILLER + "Total HT 100,00 € TVA 20% "
            "Total TTC 120,00 € Titulaire du compte : ACME SARL Numéro de compte 0001")
    assert detect_document_type(FakePDFData(text)) == "invoice"


def test_driving_licence_record_is_an_official_record():
    text = ("1/5 Relevé d'information intégral (RII) du permis de conduire " + _FILLER
            + "système national des permis de conduire solde de points Numéro (NEPH)")
    assert detect_document_type(FakePDFData(text)) == "official_record"


def test_subject_line_names_a_letter_as_an_attestation():
    # France Travail letter: the address block pushes "Objet" past the title area
    text = ("KCX2 FT56 Page 1/1 M. DUPONT JEAN 1 RUE DE LA PAIX 75002 PARIS " + _FILLER
            + "Objet : Attestation Monsieur DUPONT, France Travail certifie que vous "
            "avez été admis au bénéfice de l'allocation.")
    assert detect_document_type(FakePDFData(text)) == "attestation"


def test_subject_line_on_a_later_page_is_not_a_title():
    pages = FakePDFData(_FILLER + "nous restons à votre disposition")
    pages.text_by_page.append("Annexe Objet : Attestation de la banque")
    assert detect_document_type(pages) == "unknown"
