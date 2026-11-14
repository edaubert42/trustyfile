"""
Corpus gate for the document-type classifier (src/document_type.py).

Since 2026-11-12 the type decides real verdicts: on an issued type (invoice,
attestation...) a provenance break forces SUSPICIOUS, on "unknown" it caps at
REVIEW, on any other type it is only a hint. So a wrong type now costs either
a missed forgery or a false alarm, and this file measures it.

Each labelled document lists the types it may receive. NOT_ISSUED means any
type outside scoring.ISSUED_DOCUMENT_TYPES, "unknown" included: the classifier
has no type for a menu or a cover letter, and that is fine as long as it does
not call it an invoice or an identity document.

The corpus is gitignored (real documents never leave the machine), so every
test here skips when it is absent. Documents without a text layer (scans) are
skipped: the classifier returns "unknown" for them by design.

KNOWN_MISSES lists today's errors with their cause. A fix that corrects one
must remove it (test_known_misses_are_still_misses), so progress is visible.
"""

import glob
import logging
import os
import typing

import pytest

from src.document_type import detect_document_type
from src.extractors.pdf_extractor import extract_pdf_data
from src.models import DocumentType
from src.scoring import ISSUED_DOCUMENT_TYPES

logging.disable(logging.CRITICAL)

ALL_TYPES = frozenset(typing.get_args(DocumentType))
NOT_ISSUED = ALL_TYPES - ISSUED_DOCUMENT_TYPES

# Folder prefix (under samples/) -> accepted types. Most specific prefix wins.
FOLDER_LABELS: dict[str, frozenset[str]] = {
    "doctypes/attestation/": frozenset({"attestation"}),
    "doctypes/certificate/": frozenset({"certificate"}),
    "doctypes/menu/": NOT_ISSUED,
    "doctypes/press_release/": NOT_ISSUED,
    "doctypes/letter/": NOT_ISSUED,
    "doctypes/bank_statement/": frozenset({"bank_statement"}),
    "doctypes/price_list/": NOT_ISSUED,
    "doctypes/technical/": NOT_ISSUED,
    "legitimate/web/zugferd/": frozenset({"invoice"}),
    "suspicious/editors/": frozenset({"invoice"}),
    "2d_doc/ants_2d-doc_cabspec": NOT_ISSUED,   # ANTS technical spec pages
}

INVOICE = frozenset({"invoice"})
ATTESTATION = frozenset({"attestation"})
RECORD = frozenset({"official_record"})
# Single files (path under samples/) -> accepted types.
FILE_LABELS: dict[str, frozenset[str]] = {
    # Local legit invoices
    "legitimate/45654.pdf": INVOICE,
    "legitimate/7071 Easy Monneret fact.pdf": INVOICE,
    "legitimate/82988103.pdf": INVOICE,
    "legitimate/Costumes Facture - FA5237 - Salon rétro mobile.pdf": INVOICE,
    "legitimate/FACA3322120235.pdf": INVOICE,
    "legitimate/FR2P2Y81AEUI.pdf": INVOICE,
    "legitimate/FRINV24000000923294.pdf": INVOICE,
    "legitimate/Facture_EDF_10194772319.pdf": INVOICE,
    "legitimate/Facture_FR62778926.pdf": INVOICE,
    "legitimate/Facture_FreePro_F202309008143_20230901.pdf": INVOICE,
    "legitimate/facAFR062784917_13926841.pdf": INVOICE,
    "legitimate/facture_999648.pdf": INVOICE,
    "legitimate/Impôts 2023.pdf": frozenset({"tax_document"}),
    # Local legit documents that are not issued
    "legitimate/Notice-congé-bailleur.pdf": NOT_ISSUED,
    "legitimate/Notice-dinformation-23.05.25.pdf": NOT_ISSUED,
    "legitimate/notice.pdf": NOT_ISSUED,          # ministry notice on the DALO appeal
    "legitimate/ProtocoleTechniqueFicoba3 v11.pdf": NOT_ISSUED,
    "legitimate/ad1180507.pdf": NOT_ISSUED,
    "legitimate/rediger-l-introduction-de-l-article-scientifique.pdf": NOT_ISSUED,
    "legitimate/reportage-8-circuit-article-scientifique.pdf": NOT_ISSUED,
    "legitimate/v29_n03_548.pdf": NOT_ISSUED,
    "legitimate/Outil6.pdf": NOT_ISSUED,
    "legitimate/en.subject.pdf": NOT_ISSUED,
    "legitimate/CV_AUBERT_FR_2026_P.pdf": NOT_ISSUED,
    # 2D-Doc folder: specs and one tax notice
    "2d_doc/code.pdf": NOT_ISSUED,
    "2d_doc/technical_spec.pdf": NOT_ISSUED,
    "2d_doc/id.pdf": NOT_ISSUED,                  # full ANTS spec, not an ID
    "2d_doc/test_taxes.pdf": frozenset({"tax_document"}),
    "2d_doc/test.pdf": RECORD,                    # driving-licence record (RII)
    # Official records and attestations an administration issues
    "legitimate/releve_information_integral_permis_de_conduire.pdf": RECORD,
    "legitimate/avis_de_situation.pdf": ATTESTATION,   # France Travail
    # Suspicious corpus: the genre is still known
    "suspicious/82988103_modified.pdf": INVOICE,
    "suspicious/FACA3322030235.pdf": INVOICE,
    "suspicious/Facture 1 .pdf": INVOICE,
    "suspicious/Facture ordi volé.pdf": INVOICE,
    "suspicious/Facture-15501760717992H.pdf": INVOICE,
    "suspicious/Receipt_29Jun2022_001045.pdf": INVOICE,
    "suspicious/demo_suspicious.pdf": INVOICE,
    "suspicious/eDF_Facture_20220905_155624 (AOUT 2022) - Copie.pdf": INVOICE,
    "suspicious/facture-2059303029.pdf": INVOICE,
    "suspicious/invoice_FRW0000308935.pdf": INVOICE,
    "suspicious/fr_proof_2025-2026_edaubert1.pdf": ATTESTATION,
    "suspicious/rii_permis_de_conduire_modif.pdf": RECORD,
    "suspicious/avis_de_situation.pdf": ATTESTATION,
}

# Path under samples/ -> why the classifier gets it wrong today.
KNOWN_MISSES: dict[str, str] = {
    # Baseline 2026-11-12, before the classifier rework
    "2d_doc/id.pdf":
        "ANTS spec quoting invoice, tax and payslip fields: invoice wins on count",
    "doctypes/attestation/carizy-attestation-multi-pro-rc-2026.pdf":
        "only 'attestation' (listed twice) + 'certifions que'; 'avoir' and "
        "'rest' match inside other words",
    "doctypes/certificate/carrlane-ec-conformity.pdf":
        "'hereby certifies' alone; one keyword is below the 3-hit minimum",
    "legitimate/web/zugferd/ZUGFeRDv1__correct__Intarsys__ZUGFeRD_1p0_COMFORT_Kraftfahrversicherung_Bruttopreise.pdf":
        "German insurance invoice: 'rechnung' and an amount only",
}

SAMPLES = "samples"
MIN_TEXT_CHARS = 50   # below this the document is a scan: always "unknown"


def _label(rel: str) -> frozenset[str] | None:
    if rel in FILE_LABELS:
        return FILE_LABELS[rel]
    prefixes = [p for p in FOLDER_LABELS if rel.startswith(p)]
    if not prefixes:
        return None
    return FOLDER_LABELS[max(prefixes, key=len)]


def _corpus() -> list[tuple[str, frozenset[str]]]:
    found = []
    for path in sorted(glob.glob(os.path.join(SAMPLES, "**", "*.pdf"), recursive=True)):
        rel = os.path.relpath(path, SAMPLES).replace(os.sep, "/")
        label = _label(rel)
        if label is not None:
            found.append((rel, label))
    return found


CORPUS = _corpus()


def _detect(rel: str) -> str | None:
    """Type of a corpus document, or None when it has no text layer."""
    data = extract_pdf_data(os.path.join(SAMPLES, rel))
    if len(" ".join(data.text_by_page).strip()) < MIN_TEXT_CHARS:
        return None
    return detect_document_type(data)


@pytest.mark.skipif(not CORPUS, reason="labelled corpus not present (gitignored)")
@pytest.mark.parametrize("rel,accepted", CORPUS, ids=[rel for rel, _ in CORPUS])
def test_corpus_doc_gets_an_accepted_type(rel, accepted):
    if rel in KNOWN_MISSES:
        pytest.skip(f"known miss: {KNOWN_MISSES[rel]}")
    got = _detect(rel)
    if got is None:
        pytest.skip("no text layer")
    assert got in accepted, f"{rel}: classified {got!r}, expected one of {sorted(accepted)}"


@pytest.mark.skipif(not CORPUS, reason="labelled corpus not present (gitignored)")
def test_known_misses_are_still_misses():
    """A fixed miss must leave KNOWN_MISSES, and a stale entry must not linger."""
    labels = dict(CORPUS)
    fixed, stale = [], []
    for rel in KNOWN_MISSES:
        if rel not in labels:
            stale.append(rel)
        elif _detect(rel) in labels[rel]:
            fixed.append(rel)
    assert not stale, f"KNOWN_MISSES names unlabelled or absent files: {stale}"
    assert not fixed, f"now classified correctly, remove from KNOWN_MISSES: {fixed}"
