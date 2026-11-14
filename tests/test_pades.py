"""PAdES: offline PDF signature verification (src/modules/pades.py).

Every signed PDF here is built in the test: a throwaway CA signs a seal
certificate for a fictitious company, pyHanko signs a PyMuPDF invoice with
it, and the test hands that CA to the verifier as its only trust root. So CI
checks the whole chain (integrity, coverage, diff after signing, trust,
signer on the page) without any real document.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz
import pytest
from asn1crypto import x509 as asn1_x509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from src.modules import pades
from src.modules.pades import company_identifiers, signer_on_page, verify_signatures
from src.modules.structure import check_signature

# Fictitious issuer: the textbook SIREN of the demo invoice. FR VAT key 44.
SIREN = "732829320"
VAT = "FR44732829320"
COMPANY = "EXEMPLE SERVICES SAS"


# --- Test PKI ---------------------------------------------------------------

def _name(common_name: str, org: str | None = None, org_id: str | None = None) -> x509.Name:
    attrs = [x509.NameAttribute(NameOID.COUNTRY_NAME, "FR"),
             x509.NameAttribute(NameOID.COMMON_NAME, common_name)]
    if org:
        attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org))
    if org_id:
        attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_IDENTIFIER, org_id))
    return x509.Name(attrs)


def _cert(subject, issuer, key, issuer_key, *, ca: bool) -> x509.Certificate:
    now = datetime.now(timezone.utc)
    builder = (x509.CertificateBuilder()
               .subject_name(subject).issuer_name(issuer)
               .public_key(key.public_key())
               .serial_number(x509.random_serial_number())
               .not_valid_before(now - timedelta(days=1))
               .not_valid_after(now + timedelta(days=30))
               .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
               .add_extension(x509.KeyUsage(
                   digital_signature=not ca, content_commitment=False,
                   key_encipherment=False, data_encipherment=False, key_agreement=False,
                   key_cert_sign=ca, crl_sign=ca, encipher_only=False,
                   decipher_only=False), critical=True))
    return builder.sign(issuer_key, hashes.SHA256())


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """A test CA and a seal certificate for COMPANY, as PEM files + the CA root."""
    out = tmp_path_factory.mktemp("pki")
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("Test Qualified CA")
    ca = _cert(ca_name, ca_name, ca_key, ca_key, ca=True)

    seal_key = ec.generate_private_key(ec.SECP256R1())
    seal = _cert(_name(COMPANY, COMPANY, f"VATFR-{VAT[2:]}"), ca_name, seal_key, ca_key, ca=False)

    pem = serialization.Encoding.PEM
    (out / "ca.pem").write_bytes(ca.public_bytes(pem))
    (out / "seal.pem").write_bytes(seal.public_bytes(pem))
    (out / "seal.key").write_bytes(seal_key.private_bytes(
        pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    root = asn1_x509.Certificate.load(ca.public_bytes(serialization.Encoding.DER))
    return {"dir": out, "roots": (root,)}


def _invoice(path: Path, issuer_line: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "FACTURE N° 2026-0042", fontsize=14)
    page.insert_text((72, 100), issuer_line)
    page.insert_text((72, 120), "Total TTC 1 200,00 EUR")
    doc.save(path)
    doc.close()


def _sign(pki, src: Path, dst: Path) -> None:
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import signers

    signer = signers.SimpleSigner.load(
        str(pki["dir"] / "seal.key"), str(pki["dir"] / "seal.pem"),
        ca_chain_files=[str(pki["dir"] / "ca.pem")])
    with open(src, "rb") as fh:
        writer = IncrementalPdfFileWriter(fh)
        with open(dst, "wb") as out:
            signers.sign_pdf(writer, signers.PdfSignatureMetadata(field_name="Seal"),
                             signer=signer, output=out)


@pytest.fixture(scope="module")
def signed(pki, tmp_path_factory):
    """An invoice naming COMPANY by its SIREN, signed by COMPANY's seal."""
    out = tmp_path_factory.mktemp("signed")
    _invoice(out / "plain.pdf", f"{COMPANY} - SIREN {SIREN[:3]} {SIREN[3:6]} {SIREN[6:]}")
    _sign(pki, out / "plain.pdf", out / "signed.pdf")
    return out / "signed.pdf"


# --- Reading the certificate's identity ------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("VAT-FR03552081317", ("FR03552081317", "552081317")),   # EDF's LuxTrust form
    ("VATFR-03552081317", ("FR03552081317", "552081317")),   # ETSI form
    ("NTRFR-552081317", ("552081317",)),
    ("VATDE-123456789", ("DE123456789",)),
    ("PSDFR-ACPR-12345", ()),                                # not a company id
    (None, ()),
])
def test_company_identifiers(raw, expected):
    assert company_identifiers(raw) == expected


def _check(**overrides) -> pades.SignatureCheck:
    base = dict(intact=True, valid=True, trusted=True, cert_expired=False,
                whole_file=True, untouched=True, modified_after=False,
                signer=COMPANY, organisation=COMPANY,
                identifiers=(VAT, SIREN), signed_at=None)
    return pades.SignatureCheck(**{**base, **overrides})


@pytest.mark.parametrize("page, expected", [
    (f"SIREN {SIREN[:3]} {SIREN[3:6]} {SIREN[6:]}", True),
    ("N° TVA FR 44 732 829 320", True),
    (f"Siège : {COMPANY.title()}, Paris", True),          # exact name, any case
    ("IBAN FR76 3000 6000 0173 2829 3200 180", False),    # digits inside a longer number
    ("Autre Société SARL - SIREN 552 081 317", False),
])
def test_signer_on_page(page, expected):
    assert signer_on_page(_check(), page) is expected


def test_short_organisation_name_does_not_match_alone():
    """A 3-letter O= would match too much text; identifiers still work."""
    assert not signer_on_page(_check(organisation="EDF", identifiers=()), "Facture EDF")


# --- Real signatures, built here --------------------------------------------

def test_valid_trusted_signature(pki, signed):
    [check] = verify_signatures(str(signed), roots=pki["roots"])
    assert (check.intact, check.valid, check.trusted) == (True, True, True)
    assert check.whole_file and check.untouched and not check.modified_after
    assert check.identifiers == (VAT, SIREN)
    assert signer_on_page(check, fitz.open(signed)[0].get_text())


def test_unknown_ca_is_not_trusted(signed):
    [check] = verify_signatures(str(signed), roots=())
    assert check.intact and check.valid and not check.trusted


def test_text_added_after_signing_is_a_modification(pki, signed, tmp_path):
    edited = tmp_path / "edited.pdf"
    edited.write_bytes(signed.read_bytes())
    doc = fitz.open(edited)
    doc[0].insert_text((72, 140), "Remise exceptionnelle -1 000,00 EUR")
    doc.saveIncr()
    doc.close()

    [check] = verify_signatures(str(edited), roots=pki["roots"])
    assert check.intact                  # the signed revision itself is untouched
    assert check.modified_after and not check.untouched


def test_signed_bytes_changed_is_a_modification(pki, signed, tmp_path):
    raw = signed.read_bytes()
    # One byte of the Producer, written in the first (signed) revision. Same
    # length, so every byte offset in the file stays valid.
    producer = b"/Producer(MuPDF"
    assert producer in raw[:raw.index(b"%%EOF")]
    forged = tmp_path / "forged.pdf"
    forged.write_bytes(raw.replace(producer, b"/Producer(MuXDF", 1))

    [check] = verify_signatures(str(forged), roots=pki["roots"])
    assert not check.intact and check.modified_after


def test_unsigned_and_unreadable_files(tmp_path):
    plain = tmp_path / "plain.pdf"
    _invoice(plain, "Nobody")
    assert verify_signatures(str(plain), roots=()) == []
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf")
    assert verify_signatures(str(junk), roots=()) is None


# --- From the summary to one flag -------------------------------------------

def _info(**overrides) -> dict:
    base = {"has_signature": True, "checked": True, "count": 1, "is_invalid": False,
            "modified_after": False, "is_valid": True, "is_trusted": True,
            "issuer_verified": True, "cert_expired": False,
            "signer": COMPANY, "signed_at": None}
    return {**base, **overrides}


@pytest.mark.parametrize("overrides, expected", [
    ({}, ("STRUCT_SIGNATURE_ISSUER_VERIFIED", "medium")),
    ({"issuer_verified": False}, ("STRUCT_SIGNATURE_TRUSTED", "low")),
    ({"issuer_verified": False, "cert_expired": True}, ("STRUCT_SIGNATURE_TRUSTED_EXPIRED", "low")),
    ({"issuer_verified": False, "is_trusted": False}, ("STRUCT_SIGNATURE_NOT_TRUSTED", "medium")),
    ({"modified_after": True}, ("STRUCT_SIGNATURE_MODIFIED_AFTER_SIGNING", "high")),
    ({"is_invalid": True, "is_valid": False, "issuer_verified": False},
     ("STRUCT_SIGNATURE_INVALID", "high")),
    ({"checked": False}, ("STRUCT_SIGNATURE_UNVERIFIABLE", "low")),
    ({"count": 0}, ("STRUCT_SIGNATURE_UNVERIFIED", "low")),
])
def test_check_signature_picks_the_strongest_finding(overrides, expected):
    [flag] = check_signature(_info(**overrides))
    assert (flag.code, flag.severity) == expected


def test_no_signature_no_flag():
    assert check_signature({"has_signature": False}) == []


# --- Through the whole analyzer ---------------------------------------------

@pytest.fixture
def trust_test_ca(monkeypatch, pki):
    monkeypatch.setattr(pades, "eutl_roots", lambda: pki["roots"])


def _analyse(path: Path):
    from src.analyzer import TrustyFileAnalyzer
    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(path))
    return result, {f.code for m in result.modules for f in m.flags}


def test_issuer_signed_invoice_is_genuine(trust_test_ca, signed):
    result, codes = _analyse(signed)
    assert "STRUCT_SIGNATURE_ISSUER_VERIFIED" in codes, codes
    assert result.band == "GENUINE", (result.band, result.trust_score, codes)


def test_invoice_signed_by_another_company_is_not_issuer_verified(trust_test_ca, pki, tmp_path):
    """Any qualified seal can sign anything: the page must name the signer."""
    plain, other = tmp_path / "plain.pdf", tmp_path / "other.pdf"
    _invoice(plain, "Autre Société SARL - SIREN 552 081 317")
    _sign(pki, plain, other)
    _, codes = _analyse(other)
    assert "STRUCT_SIGNATURE_ISSUER_VERIFIED" not in codes
    assert "STRUCT_SIGNATURE_TRUSTED" in codes, codes


def test_invoice_edited_after_signing_is_suspicious(trust_test_ca, signed, tmp_path):
    edited = tmp_path / "edited.pdf"
    edited.write_bytes(signed.read_bytes())
    doc = fitz.open(edited)
    doc[0].insert_text((72, 140), "Remise exceptionnelle -1 000,00 EUR")
    doc.saveIncr()
    doc.close()

    result, codes = _analyse(edited)
    assert "STRUCT_SIGNATURE_MODIFIED_AFTER_SIGNING" in codes, codes
    assert "STRUCT_SIGNATURE_ISSUER_VERIFIED" not in codes
    assert result.band == "SUSPICIOUS", (result.band, result.trust_score, codes)


# --- The bundled trust store ------------------------------------------------

def test_bundled_store_matches_its_index():
    index = json.loads((pades.EUTL_DIR / "index.json").read_text())
    files = sorted(p.name for p in pades.EUTL_DIR.glob("*.der"))
    assert files == sorted(index["certs"]) and index["count"] == len(files)
    assert len(pades.eutl_roots()) == len(files)
    # EDF's issuer: the store must not silently lose the CA real invoices use.
    assert any("LuxTrust Global Qualified CA 3" in s for s in index["certs"].values())
    # Website-only CAs are filtered out (scripts/fetch_eutl_certs.py).
    assert not any("Website Authentication" in s for s in index["certs"].values())


# --- What the issuer's whole-file signature outweighs (user decision 2026-11-14)

def _result(*flags):
    from src.models import Flag, ModuleResult
    from src.scoring import create_analysis_result

    modules = [ModuleResult(module=module, flags=[Flag(severity=sev, code=code, message=code)],
                            score=50, confidence=1.0)
               for module, sev, code in flags]
    return create_analysis_result("hash", modules, document_type="invoice")


ISSUER = ("structure", "medium", "STRUCT_SIGNATURE_ISSUER_VERIFIED")


def test_issuer_signature_outweighs_soft_page_findings():
    """The EDF case: a weight suffix read as a font change, an ELA region."""
    result = _result(ISSUER, ("fonts", "high", "FONTS_MIDLINE_CHANGE"),
                     ("forensics", "high", "FORENSICS_ELA_MAJOR_EDIT"),
                     ("structure", "high", "STRUCT_XMP_EDITOR_MISMATCH"))
    assert result.band == "GENUINE", (result.band, result.trust_score)


def test_without_the_signature_the_same_findings_count():
    result = _result(("fonts", "high", "FONTS_MIDLINE_CHANGE"),
                     ("forensics", "high", "FORENSICS_ELA_MAJOR_EDIT"))
    assert result.band != "GENUINE"


def test_proof_of_modification_still_wins_over_the_signature():
    result = _result(ISSUER, ("fonts", "high", "FONTS_MIDWORD_CHANGE"))
    assert result.band == "SUSPICIOUS", (result.band, result.trust_score)


def test_hard_tamper_evidence_still_wins_over_the_signature():
    result = _result(ISSUER, ("overlay", "critical", "OVERLAY_COVER_AND_RETYPE_CONFIRMED"))
    assert result.band != "GENUINE", (result.band, result.trust_score)


def test_outweighed_findings_are_not_given_as_reasons():
    """The reader must not see 'authentic' next to 'the text was modified'."""
    from src.models import Flag
    from src.summary import top_reasons

    sig = Flag(severity="medium", code="STRUCT_SIGNATURE_ISSUER_VERIFIED", message="")
    font = Flag(severity="high", code="FONTS_MIDLINE_CHANGE", message="")
    midword = Flag(severity="high", code="FONTS_MIDWORD_CHANGE", message="")
    assert top_reasons([font, sig]) == []
    assert top_reasons([font]) == [font]
    assert top_reasons([midword, font, sig]) == [midword]
