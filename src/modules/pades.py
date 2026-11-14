"""Offline verification of PDF signatures (PAdES), with pyHanko.

WHAT A PDF SIGNATURE PROVES
---------------------------
The signer hashes the bytes of the file (all of them except the hole the
signature sits in), encrypts that hash with a private key and stores the
result with its certificate. Checking it answers four questions:

  1. intact     — do the signed bytes still hash to the signed value?
                  If not, the file was changed after signing.
  2. coverage   — does the signature cover the whole file, or only an earlier
                  revision followed by appended ones?
  3. modified   — if revisions were appended after signing, did they only
                  add what PDF allows on a signed file (form fields,
                  annotations, timestamps), or did they change the content?
  4. trusted    — does the certificate chain to a CA of the EU trusted list
                  (src/data/eutl_certs, built by scripts/fetch_eutl_certs.py)?
                  Anyone can make a certificate that SAYS "EDF"; only a
                  qualified CA checked that it is EDF.

WHY PYHANKO
-----------
It is the reference PAdES validator in Python: it does all four checks
offline, including the diff analysis of revisions appended after signing,
which no other Python library offers. The previous code sent the PDF to the
EU DSS web API, which broke the rule that a document never leaves the
machine.

WHAT IS NOT CHECKED
-------------------
- Revocation: CRL/OCSP need the network. A revoked certificate is accepted.
- The signing time: most invoices carry no timestamp, only the time the
  signer's software wrote. Certificates are checked at that claimed time, so
  an expired certificate (EDF's expired in 2024) still validates. Claimed,
  not proven: TRUSTED_EXPIRED says so.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

EUTL_DIR = Path(__file__).resolve().parent.parent / "data" / "eutl_certs"


@dataclass(frozen=True)
class SignatureCheck:
    """What one embedded signature proves.

    Attributes:
        intact: The signed bytes are unchanged.
        valid: The signature value verifies with the certificate's key.
        trusted: The certificate chains to an EU qualified CA (at the claimed
            signing time) and allows document signing.
        cert_expired: The certificate is no longer valid today.
        whole_file: The signature covers every byte of the file.
        untouched: Nothing a reader sees changed since signing: the
            signature covers the whole file, or what was appended only
            adds long-term validation data (pyHanko LTA_UPDATES).
        modified_after: Content changed after signing: the signed bytes
            changed, or a later revision did more than PDF allows on a
            signed file.
        signer: The certificate's subject, readable.
        organisation: The subject's Organization (O=), if any.
        identifiers: Company identifiers read from the certificate
            (VAT number and/or SIREN), see company_identifiers().
        signed_at: The signing time the signer's software claims, if any.
    """

    intact: bool
    valid: bool
    trusted: bool
    cert_expired: bool
    whole_file: bool
    untouched: bool
    modified_after: bool
    signer: str
    organisation: str | None
    identifiers: tuple[str, ...]
    signed_at: datetime | None


@lru_cache(maxsize=1)
def eutl_roots() -> tuple:
    """Load the bundled EU qualified CA certificates, once per process.

    Returns:
        A tuple of asn1crypto certificates (770 on 2026-11-14), empty when
        the store is missing: every signature then reads as not trusted.
    """
    from asn1crypto import x509

    roots = []
    for path in sorted(EUTL_DIR.glob("*.der")):
        try:
            roots.append(x509.Certificate.load(path.read_bytes()))
        except Exception as e:   # one damaged file must not disable the rest
            logger.warning("EUTL certificate %s unreadable: %s", path.name, e)
    return tuple(roots)


def company_identifiers(organization_identifier: str | None) -> tuple[str, ...]:
    """Read the company identifiers out of a certificate's organizationIdentifier.

    ETSI EN 319 412-1 writes it as a 3-letter type, a country, a hyphen and the
    number: "VATFR-03552081317" (VAT number), "NTRFR-552081317" (national
    trade register: the SIREN in France). Some CAs write "VAT-FR03552081317".
    A French VAT number is "FR" + 2-character key + the 9-digit SIREN, so the
    SIREN is returned too: a French invoice may print either.

    Args:
        organization_identifier: The raw value, or None.

    Returns:
        The identifiers to look for on the page, without spaces.

    Examples:
        >>> company_identifiers("VAT-FR03552081317")
        ('FR03552081317', '552081317')
        >>> company_identifiers("NTRFR-552081317")
        ('552081317',)
        >>> company_identifiers("VATDE-123456789")
        ('DE123456789',)
        >>> company_identifiers(None)
        ()
    """
    if not organization_identifier:
        return ()
    match = re.fullmatch(r"(VAT|NTR)-?([A-Z]{2})-?([A-Z0-9]+)",
                         organization_identifier.strip().upper())
    if not match:
        return ()
    kind, country, number = match.groups()
    if kind == "NTR":
        return (number,)
    found = [country + number]
    if country == "FR" and len(number) == 11 and number[2:].isdigit():
        found.append(number[2:])      # FR + key (2) + SIREN (9)
    return tuple(found)


def _identifier_pattern(identifier: str) -> re.Pattern:
    """Regex for an identifier as printed: groups may be split by spaces or dots.

    Bounded so that it cannot match inside a longer number: "552081317" must
    not be found in the middle of an IBAN or a phone number.

    Example:
        "552081317" matches "SIREN 552 081 317" and "552.081.317",
        not "FR76 5520 8131 7000".
    """
    separated = r"[ . ]?".join(re.escape(ch) for ch in identifier)
    return re.compile(rf"(?<![A-Za-z0-9]){separated}(?![0-9])", re.IGNORECASE)


def _normalise_name(text: str) -> str:
    """Upper case, accents kept, runs of whitespace collapsed to one space."""
    return " ".join(text.upper().split())


def signer_on_page(check: SignatureCheck, page_text: str) -> bool:
    """Is the signer the issuer the page names?

    True when one of the certificate's company identifiers (VAT number or
    SIREN) is printed on the page, or its exact organisation name is.
    Anyone holding any EU qualified certificate can sign a forged invoice;
    this cross-check is what binds the signature to the issuer on the page
    (user rule 2026-11-14, as the 2D-Doc field match does).

    Example:
        EDF's certificate (VAT-FR03552081317, O=ELECTRICITE DE FRANCE) on a
        page printing "SIREN 552 081 317" -> True.
        The same signature on a page naming another company -> False.
    """
    for identifier in check.identifiers:
        if _identifier_pattern(identifier).search(page_text):
            return True
    if check.organisation:
        name = _normalise_name(check.organisation)
        # Whole words only: "EDF" must not match inside "EDFX".
        if len(name) >= 4 and re.search(rf"(?<!\w){re.escape(name)}(?!\w)",
                                        _normalise_name(page_text)):
            return True
    return False


def _attribute(cert, name: str) -> str | None:
    """One attribute of the certificate's subject, or None."""
    value = cert.subject.native.get(name)
    if isinstance(value, list):      # a multi-valued attribute: keep the first
        value = value[0] if value else None
    return value


def _check_one(embedded, roots: tuple) -> SignatureCheck:
    """Validate one embedded signature with pyHanko."""
    from pyhanko.sign.diff_analysis import ModificationLevel
    from pyhanko.sign.validation import validate_pdf_signature
    from pyhanko.sign.validation.settings import KeyUsageConstraints
    from pyhanko.sign.validation.status import SignatureCoverageLevel
    from pyhanko_certvalidator import ValidationContext

    # Check the chain at the time the signer claims, so that a certificate
    # that expired since (EDF's, in 2024) still validates. None: now.
    claimed = embedded.self_reported_timestamp
    context = ValidationContext(trust_roots=list(roots), allow_fetching=False,
                                moment=claimed)
    # pyHanko's default demands non_repudiation; an organisation's seal
    # (EDF's LuxTrust certificate) carries digital_signature only.
    usage = KeyUsageConstraints(key_usage={"digital_signature", "non_repudiation"},
                                match_all_key_usages=False)
    status = validate_pdf_signature(embedded, context, key_usage_settings=usage)

    cert = status.signing_cert
    expired = cert.not_valid_after < datetime.now(timezone.utc)
    level = status.modification_level
    whole_file = status.coverage == SignatureCoverageLevel.ENTIRE_FILE
    modified = (not status.intact) or (level is not None and level >= ModificationLevel.OTHER)
    return SignatureCheck(
        intact=bool(status.intact),
        valid=bool(status.valid),
        trusted=bool(status.trusted),
        cert_expired=expired,
        whole_file=whole_file,
        untouched=whole_file or (level is not None and level <= ModificationLevel.LTA_UPDATES),
        modified_after=modified,
        signer=cert.subject.human_friendly,
        organisation=_attribute(cert, "organization_name"),
        identifiers=company_identifiers(_attribute(cert, "organization_identifier")),
        signed_at=claimed,
    )


def verify_signatures(pdf_path: str, roots: tuple | None = None) -> list[SignatureCheck] | None:
    """Check every signature embedded in a PDF, offline.

    Args:
        pdf_path: The PDF to check.
        roots: Trust anchors; the bundled EU qualified CAs by default (tests
            pass their own).

    Returns:
        One SignatureCheck per signature, in file order ([] when the file
        holds no filled signature), or None when pyHanko could not read the
        file or a signature: we then know nothing, and say so.

    Example:
        An EDF invoice -> [SignatureCheck(intact=True, valid=True,
        trusted=True, cert_expired=True, whole_file=True,
        untouched=True, modified_after=False, signer="Common Name: ELECTRICITE DE FRANCE, ...",
        identifiers=("FR03552081317", "552081317"), ...)]
    """
    from pyhanko.pdf_utils.reader import PdfFileReader

    if roots is None:
        roots = eutl_roots()
    try:
        with open(pdf_path, "rb") as fh:
            reader = PdfFileReader(fh, strict=False)
            return [_check_one(sig, roots) for sig in reader.embedded_signatures]
    except Exception as e:   # malformed file or signature: never crash the analysis
        logger.warning("PAdES validation failed for %s: %s", pdf_path, e)
        return None
