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
from src.models import DocumentType

logger = logging.getLogger(__name__)


# =============================================================================
# KEYWORD TABLES
# =============================================================================

# Each type has a list of keywords to search for (case-insensitive).
# A document scores +1 for each keyword found (presence, not count).
# The type with the highest score wins. Tie → unknown.
# Minimum score to classify: 2 keywords must match.

KEYWORDS: dict[str, list[str]] = {
    "invoice": [
        # French
        "facture", "montant dû", "total ttc", "total ht", "tva", "siret",
        "iban", "numéro de facture", "régler avant", "échéance", "devis",
        "bon de commande", "avoir", "acompte", "solde",
        # English
        "invoice", "amount due", "payment due", "vat number", "tax invoice",
        "billing address", "purchase order", "remittance",
    ],
    "identity": [
        # French
        "carte nationale d'identité", "carte nationale", "passeport", "passport",
        "permis de conduire", "titre de séjour", "date de naissance",
        "lieu de naissance", "nationalité", "délivré le",
        # English
        "national id", "driving licence", "driving license", "date of birth",
        "place of birth", "nationality", "issued on",
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
# CLASSIFIER
# =============================================================================

def detect_document_type(pdf_data) -> DocumentType:
    """
    Classify a document by counting keyword matches in its text content.

    The function joins all pages into one lowercase string, then counts
    how many distinct keywords from each category are present.

    The category with the highest keyword count is returned.
    If the winner has fewer than 3 matches, returns "unknown".

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
        scores[doc_type] = hit_count

    # Find the winner (type with max score)
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    logger.debug(f"Document type scores: {scores}")
    logger.debug(f"Best match: {best_type} (score={best_score})")

    # Require at least 3 keyword matches to classify.
    # 2 is too low — generic words like "introduction" + "results" can appear
    # in any document and falsely trigger the "scientific" category.
    if best_score < 3:
        return "unknown"

    return best_type  # type: ignore[return-value]
