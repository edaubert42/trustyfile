"""Build the offline store of EU qualified CA certificates (PAdES trust roots).

WHY THIS EXISTS
---------------
A PDF signature is only worth something if the signer's certificate chains to
an authority we trust. For the EU, that list is the EU trusted list (EUTL): a
"list of lists" (LOTL) published by the Commission that points to one trusted
list per member state, each naming the qualified trust service providers and
their CA certificates. EDF's invoices, for instance, are sealed with a
certificate issued by "LuxTrust Global Qualified CA 3", listed by Luxembourg.

TrustyFile's runtime rule is that an analysis NEVER touches the network. So,
as for the 2D-Doc certificates (scripts/fetch_twod_doc_certs.py), this script
runs at BUILD time: a developer runs it, reviews the diff, commits the result.
The runtime verifier (src/modules/pades.py) only reads the committed files.

    venv/bin/python scripts/fetch_eutl_certs.py

pyHanko (requirements-dev.txt: pyhanko[etsi]) does the fetching and checks the
XML signature of the LOTL and of every national list against the signers the
Commission publishes, so a tampered mirror cannot slip a certificate in.

Output: src/data/eutl_certs/
    <sha256 prefix>.der   one DER certificate per CA (16 hex chars of its sha256)
    index.json            {"fetched_at", "count", "certs": {file: subject}}:
                          the reviewable part of the diff

Measured 2026-11-14: 881 CAs listed, 770 kept (signatures and seals), ~30 s.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "data" / "eutl_certs"


async def fetch_ca_certificates() -> tuple[list, list]:
    """Download and parse the EU trusted lists.

    Returns:
        (certificates, errors): the asn1crypto certificates of every CA
        service listed, and the services pyHanko could not parse.
    """
    import aiohttp
    from pyhanko.sign.validation.qualified.eutl_fetch import lotl_to_registry

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        # None: fetch the LOTL from its official location.
        registry, errors = await lotl_to_registry(None, client)

    certs = []
    for authority in registry.known_certificate_authorities:
        # Some authorities are known only by name and key id, without a
        # certificate: nothing to anchor a chain on, so they are skipped.
        cert = getattr(authority, "certificate", None)
        if cert is not None and issues_signing_certificates(registry, authority):
            certs.append(cert)
    return certs, errors


def issues_signing_certificates(registry, authority) -> bool:
    """Does this CA issue certificates that can sign a document?

    Each listed service says which qualified certificates it issues: for
    e-signatures (a person), e-seals (an organisation, like EDF's invoices)
    or websites (TLS). A CA that only issues website certificates must not
    be able to vouch for a signed invoice. Measured 2026-11-14: 111 of the
    881 listed CAs are website-only.

    Example:
        "LuxTrust Global Qualified CA 3" (eseal + esign) -> True
        "Sectigo Qualified Website Authentication CA R35" (web only) -> False
    """
    from pyhanko.sign.validation.qualified.tsp import QcCertType

    signing = {QcCertType.QC_ESIGN, QcCertType.QC_ESEAL}
    # moment=None: every service definition, whatever its validity period.
    for service in registry.applicable_service_definitions(authority, None):
        issued = set(service.base_info.additional_info_certificate_type or ())
        if issued & signing:
            return True
    return False


def write_store(certs: list, out: Path = OUT) -> dict:
    """Replace the store with these certificates and return the index.

    The directory is emptied first, so a CA removed from the EUTL is removed
    from the store too. Files are named by content (sha256 prefix): a re-run
    with the same lists rewrites the same names, and the diff stays empty.

    Example:
        write_store([luxtrust_ca]) -> {"count": 1, "certs":
            {"3f2a...e1.der": "Common Name: LuxTrust Global Qualified CA 3, ..."}}
    """
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.der"):
        old.unlink()

    names: dict[str, str] = {}
    for cert in certs:
        name = f"{cert.sha256.hex()[:16]}.der"
        (out / name).write_bytes(cert.dump())
        names[name] = cert.subject.human_friendly

    index = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(names),
        "certs": dict(sorted(names.items())),
    }
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    return index


if __name__ == "__main__":
    certificates, parse_errors = asyncio.run(fetch_ca_certificates())
    if not certificates:
        # Never wipe a working store because the network was down.
        sys.exit("No certificate fetched: store left untouched.")
    written = write_store(certificates)
    print(f"{written['count']} CA certificates written to {OUT} "
          f"({len(parse_errors)} services pyHanko could not parse)")
