"""
Document Type Classifier for TrustyFile.

Detects the type of document (invoice, scientific paper, identity card, etc.)
by counting keyword matches in the extracted text. No ML — just keyword frequency.

Why keyword matching?
- Fast (no model to load, no inference time)
- Explainable (easy to debug which keywords triggered)
- Works offline, no dependencies
- Good enough for the 8 categories we care about

Usage:
    from src.document_type import detect_document_type
    doc_type = detect_document_type(pdf_data)
    # Returns e.g. "invoice", "scientific", "unknown"
"""

import logging
import re

from src.models import DocumentType

logger = logging.getLogger(__name__)


# =============================================================================
# KEYWORD TABLES
# =============================================================================

# Each type has a list of keywords to search for (case-insensitive).
# A document scores +1 for each keyword found (presence, not count).
# The type with the highest score wins. Tie → unknown.
# Minimum score to classify: 3 keywords must match.

KEYWORDS: dict[str, list[str]] = {
    "invoice": [
        # French
        "facture", "montant dû", "total ttc", "total ht", "tva", "siret",
        "iban", "numéro de facture", "régler avant", "échéance", "devis",
        "bon de commande", "avoir", "acompte", "solde",
        "total à payer", "total a payer", "hors taxes", "sous-total",
        "votre facture", "prix unitaire", "frais de livraison",
        # English
        "invoice", "amount due", "payment due", "vat number", "tax invoice",
        "billing address", "purchase order", "remittance", "receipt",
        "subtotal", "total due", "unit price",
        # German (ZUGFeRD / XRechnung invoices)
        "rechnung", "rechnungsnummer", "rechnungsdatum", "umsatzsteuer",
        "mwst", "ust-id", "nettobetrag", "bruttobetrag", "gesamtbetrag",
        "zahlbar",
        # Spanish
        "factura", "importe", "base imponible",
        # Transport / trade papers an issuer generates once, like an invoice
        # (investigation doc 1, a bill of lading, fell into legal_notice)
        "bill of lading", "connaissement", "shipper", "consignee",
        "port of loading", "port of discharge", "lettre de voiture",
    ],
    "identity": [
        # Termes exclusifs aux pièces d'identité officielles — ne pas ajouter ici
        # des termes génériques ("date de naissance", "nationalité", "nationality",
        # "date of birth"...) qui apparaissent aussi dans les CVs, formulaires, etc.
        # French
        "carte nationale d'identité", "passeport",
        "permis de conduire", "titre de séjour",
        # English
        "national identity card", "national id card",
        "driving licence", "driving license",
        # Marqueurs MRZ — impossibles dans tout autre contexte
        "machine readable zone", "<<<<<",
    ],
    "certificate": [
        # French
        "certificat", "certifie que", "certifions que", "homologation",
        "conformité", "attestons", "délivrons le présent",
        # English
        "certificate", "certify", "hereby certifies", "compliance",
        "certification", "this is to certify",
    ],
    "attestation": [
        # French
        "attestation", "je soussigné", "je soussignée", "certifie sur l'honneur",
        "domicilié", "l'intéressé", "atteste que", "bonne foi",
        # English
        "attestation", "i the undersigned", "hereby attest", "good faith",
    ],
    "tax_document": [
        # French
        "avis d'imposition", "revenus", "déclaration de revenus",
        "impôt sur le revenu", "dgfip", "direction générale des finances",
        "revenu fiscal de référence", "déclaration fiscale", "cotisation",
        # English
        "tax return", "tax assessment", "income tax", "fiscal year",
        "tax authority", "taxable income",
    ],
    "payslip": [
        # Wording of the French model payslip (bulletin de paie). Kept to
        # terms a tax notice or an invoice does not print: "impôt sur le
        # revenu" and "prélèvement à la source" also appear on a tax notice,
        # so they are deliberately NOT listed here.
        # French
        "bulletin de paie", "bulletin de salaire", "net à payer avant impôt",
        "net payé", "salaire de base", "salaire brut", "net imposable",
        "net social", "cotisations salariales", "part salariale",
        "part patronale", "convention collective", "congés payés",
        "date d'entrée", "matricule", "urssaf", "taux personnalisé",
        # English
        "payslip", "pay slip", "gross pay", "net pay",
    ],
    "technical": [
        # French — protocoles, specs, documentations techniques
        "protocole technique", "spécification technique", "cahier des charges",
        "référentiel technique", "documentation technique", "api", "endpoint",
        "authentification", "webservice", "swagger", "json", "xml",
        "requête http", "réponse http", "rest", "soap", "wsdl",
        "schéma xsd", "flux", "échange de données", "interopérabilité",
        "architecture technique", "environnement de production", "environnement de test",
        # English
        "technical specification", "technical protocol", "api documentation",
        "request", "response", "authentication", "authorization",
        "payload", "schema", "interface description",
    ],
    "scientific": [
        # French / universal
        "abstract", "résumé", "introduction", "methodology", "méthodologie",
        "references", "bibliographie", "keywords", "mots-clés", "doi:",
        "journal of", "proceedings", "arxiv", "issn", "peer-reviewed",
        "université", "laboratory", "fig.", "figure", "table",
        # English
        "conclusion", "discussion", "hypothesis", "experiment",
        "results", "citation", "literature review",
    ],
    "legal_notice": [
        # French
        "notice d'information", "conditions générales", "mentions légales",
        "règlement intérieur", "modalités", "en vigueur", "préambule",
        "cgu", "cgv", "article 1", "article 2", "conformément",
        # English
        "terms and conditions", "legal notice", "privacy policy",
        "terms of service", "general conditions", "in accordance with",
    ],
}


# =============================================================================
# PATTERN SIGNALS
# =============================================================================
# Some invoice signals are too short or too variable for a plain substring:
# "ht" is inside "light" and "https", and an amount is any number followed by
# a currency. A regex adds word boundaries and shapes. Each pattern found adds
# +1 to its type, exactly like a keyword.
#
# Why (2026-11-12): a restaurant invoice typed in Word ("FC A21012026", a
# HT/TVA table in euros, "sarl - RCS BREST" in the footer) matched only "tva"
# and "siret", fell into "unknown", and an Acrobat edit then scored GENUINE 81.
#
# Measured on the 205-PDF corpus: ht/ttc, a VAT rate, RCS and an
# intra-community VAT number appear on invoices only. An amount in euros also
# appears on tax notices and legal notices, a legal form on one technical spec:
# alone they never reach the 3-signal minimum.
#
# Each pattern is one MEANING in French and English ("TVA 10%" and "VAT 20%"
# are the same signal), so a bilingual invoice does not count it twice. An
# English receipt with no word "invoice" ("£375.14 ... VAT 20% ... Ltd ...
# Registered in England") was "unknown" too before the English forms.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "invoice": [
        # An amount with its currency: "375.14 Euros", "1 200,00 €", "43.8 EUR",
        # "£375.14", "$129.90", "120.00 GBP"
        re.compile(r"[£$]\s?\d"
                   r"|\d[\d  .,]*\s?(?:€|£|\$|euros?\b|eur\b|gbp\b|usd\b)"),
        # Net / gross markers as whole words: "HT 10 %", "Total TTC", "excl. VAT".
        # Not "light", "https" or "private".
        re.compile(r"\b(?:ht|ttc|vat)\b"),
        # A tax rate: "TVA 10%", "TVA à 5,5 %", "VAT 20%", "VAT @ 20%",
        # "Sales tax 8.25%", "GST 10%"
        re.compile(r"\b(?:tva|vat|sales tax|gst)\s*(?:à|@)?\s*\d{1,2}(?:[.,]\d+)?\s?%"),
        # Company forms. "sa" is left out (a French word), "inc" needs a word
        # boundary ("incl." does not match).
        re.compile(r"\b(?:sarl|sas|sasu|eurl|snc|ltd|llc|llp|inc|plc|gmbh)\b"),
        # Trade register: "RCS BREST B 789 397 759", "Registered in England
        # No 01234567", "Company No 01234567", "Company registration number"
        re.compile(r"\brcs\b|registered in (?:england|scotland|wales|northern ireland)"
                   r"|company (?:registration )?(?:no\b|number)"),
        # The seller's VAT number: "Numéro de TVA intracommunautaire", "N° TVA",
        # "VAT Reg No", "VAT registration number", "VAT No"
        re.compile(r"tva intracommunautaire|num[ée]ro de tva|n°\s?tva"
                   r"|vat (?:reg(?:istration)?\.? )?(?:no\b|number)"),
    ],
}


# =============================================================================
# CLASSIFIER
# =============================================================================

def detect_document_type(pdf_data) -> DocumentType:
    """
    Classify a document by counting keyword matches in its text content.

    The function joins all pages into one lowercase string, then counts
    how many distinct keywords from each category are present.

    The category with the highest keyword count is returned.
    If the winner has fewer than 3 matches, returns "unknown".
    A tie for the best score also returns "unknown".

    Args:
        pdf_data: PDFData object from the extractor (has .text_by_page list)

    Returns:
        DocumentType string, e.g. "invoice", "scientific", "unknown"

    Example:
        A document containing "facture", "total ttc", "tva", "iban"
        would score invoice=4, all others=0 → returns "invoice"
    """
    # Combine all pages into one lowercase string for easy searching
    full_text = " ".join(pdf_data.text_by_page).lower()

    if not full_text.strip():
        logger.debug("Document has no text — returning 'unknown'")
        return "unknown"

    # Score each document type by counting keyword hits
    scores: dict[str, int] = {}
    for doc_type, keywords in KEYWORDS.items():
        # Use a set to count distinct keyword matches (not total occurrences)
        hit_count = sum(1 for kw in keywords if kw.lower() in full_text)
        # Pattern signals count the same way: +1 per pattern found
        for pattern in PATTERNS.get(doc_type, []):
            if pattern.search(full_text):
                hit_count += 1
        scores[doc_type] = hit_count

    # Find the winner. A TIE between two types means the document is ambiguous
    # — classify as "unknown" rather than letting dict order pick arbitrarily
    # (doc-type confidence suppression must only apply to CLEAR classifications).
    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]

    logger.debug(f"Document type scores: {scores}")
    logger.debug(f"Best match: {winners} (score={best_score})")

    # Require at least 3 keyword matches to classify.
    # 2 is too low — generic words like "introduction" + "results" can appear
    # in any document and falsely trigger the "scientific" category.
    if best_score < 3 or len(winners) > 1:
        return "unknown"

    return winners[0]  # type: ignore[return-value]
