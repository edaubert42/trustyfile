"""Build the offline 2D-Doc certificate store from the ANTS trust list.

WHY THIS EXISTS
---------------
Verifying a 2D-Doc's ECDSA signature needs the issuing authority's X.509
certificate. The authoritative source is the ANTS TSL (trust list) — an
ETSI TS 102 231 XML document listing every accredited 2D-Doc certification
authority and its certificates. But TrustyFile's runtime constraint is that
a document analysis NEVER touches the network: the trust material must ship
with the repo.

So this script runs at BUILD time (a developer runs it, reviews the diff,
commits the result), and the runtime verifier (src/modules/twod_doc_crypto)
reads only the committed files.

    python scripts/fetch_twod_doc_certs.py

Output: src/data/twod_doc_certs/
    index.json          {ca_id: [{file, subject, serial, candidate_ids}]}
    <CA>_<n>.pem        one PEM per certificate

CERT-ID RESOLUTION
------------------
A 2D-Doc header names its signer as (CA-ID, Cert-ID) — e.g. ("FR04",
"0002") — but the TSL does not label certificates with those short IDs.
The de-facto matching rule (same as betagouv/2ddoc-parser) is to derive
CANDIDATE ids from each certificate and index on those:
    - trailing 4/5/6/8 hex digits of the serial number
    - trailing 4/5/6/8 hex digits of the SubjectKeyIdentifier
    - leading 4/6 hex chars of SHA-1(SubjectPublicKeyInfo)
    - the certificate's Common Name and its trailing 4 chars (several CAs
      name leaf certs literally "FR04-0002" style)
Candidates are computed HERE, once, so the runtime lookup is an exact
dict hit with no heuristics.

Some CAs publish their leaf (signing) certificates not in the TSL itself
but in a directory referenced by TSPInformationURI; those are fetched
best-effort (PEM blocks or raw DER) and merged in.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

# The TSL is signed XML from a government host, but it is still
# network-fetched input: parse with defusedxml like every other
# untrusted XML in this project.
from defusedxml import ElementTree as DET

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fetch_twod_doc_certs")

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = ROOT / "src" / "data" / "twod_doc_certs"

TSL_URL = "https://pub.ants.gouv.fr/2D-DOC/V1/PRD/01_TSL/tsl_signed.xml"

NS = {
    "tsl": "http://uri.etsi.org/02231/v2#",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}

# 2D-Doc CA-ID: 2-letter country + 2 alphanumerics (spec §5.2), e.g. FR01.
CA_ID_RE = re.compile(r"\b([A-Z]{2}[0-9A-Z]{2})\b")

# Only services in this status are accredited ("en accord").
STATUS_OK_SUFFIX = "/Svcstatus/inaccord"

FETCH_TIMEOUT = 15


def _http_get(url: str) -> bytes:
    # Some hosts (sp.ants.gouv.fr) 403 urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TrustyFile cert fetch)"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return resp.read()


def _fetch_leaf_blob(uri: str) -> bytes | None:
    """Fetch a leaf-certificate URI, preferring HTTPS.

    Several CAs publish plain-http URIs (and some hosts have no TLS at
    all). Transport here is NOT the trust boundary — every fetched leaf
    must pass `_is_pinned_or_chained` against the TSL-embedded CA certs
    before being stored — but https is still tried first.
    """
    candidates = [uri]
    if uri.startswith("http://"):
        candidates.insert(0, "https://" + uri[len("http://"):])
    for candidate in candidates:
        try:
            return _http_get(candidate)
        except Exception as exc:
            logger.debug("fetch failed (%s): %s", candidate, exc)
    return None


def _is_pinned_or_chained(leaf: "x509.Certificate",
                          tsl_certs: list["x509.Certificate"]) -> bool:
    """Accept a network-fetched cert only if the TSL vouches for it.

    True when the cert IS one of the TSL-embedded certs (byte-identical),
    or its signature verifies under one of them (i.e. the accredited CA
    issued it). This is what makes fetching over plain http safe: a
    tampered response cannot chain to the pinned CA.
    """
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

    leaf_der = leaf.public_bytes(serialization.Encoding.DER)
    for ca in tsl_certs:
        if ca.public_bytes(serialization.Encoding.DER) == leaf_der:
            return True
        try:
            pub = ca.public_key()
            if isinstance(pub, rsa.RSAPublicKey):
                pub.verify(leaf.signature, leaf.tbs_certificate_bytes,
                           padding.PKCS1v15(), leaf.signature_hash_algorithm)
                return True
            if isinstance(pub, ec.EllipticCurvePublicKey):
                pub.verify(leaf.signature, leaf.tbs_certificate_bytes,
                           ec.ECDSA(leaf.signature_hash_algorithm))
                return True
        except Exception:
            continue
    return False


def _candidate_ids(cert: x509.Certificate) -> list[str]:
    """Every short identifier a 2D-Doc header might use for this cert."""
    candidates: set[str] = set()

    serial_hex = format(cert.serial_number, "x").upper()
    for n in (4, 5, 6, 8):
        if len(serial_hex) >= n:
            candidates.add(serial_hex[-n:])

    try:
        ski = cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        ski_hex = ski.value.digest.hex().upper()
        for n in (4, 5, 6, 8):
            candidates.add(ski_hex[-n:])
    except x509.ExtensionNotFound:
        pass

    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    spki_sha1 = hashlib.sha1(spki).hexdigest().upper()
    candidates.add(spki_sha1[:4])
    candidates.add(spki_sha1[:6])

    # Several CAs put the 2D-Doc identifier straight into the CN
    # ("FR04-0002", "0002", ...): index the CN and its tail as well.
    try:
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
        if cn:
            cn_val = cn[0].value.strip().upper()
            candidates.add(cn_val)
            if len(cn_val) >= 4:
                candidates.add(cn_val[-4:])
    except Exception:
        pass

    return sorted(candidates)


def _certs_from_blob(blob: bytes) -> list[x509.Certificate]:
    """Extract certificates from an arbitrary HTTP response: PEM blocks,
    then raw DER, then PKCS#7 bundles (ariadnext publishes those).
    Best effort — returns []."""
    certs: list[x509.Certificate] = []
    pem_blocks = re.findall(
        rb"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----",
        blob, re.DOTALL,
    )
    for block in pem_blocks:
        try:
            certs.append(x509.load_pem_x509_certificate(block))
        except Exception:
            continue
    if not certs:
        try:
            certs.append(x509.load_der_x509_certificate(blob))
        except Exception:
            pass
    if not certs:
        try:
            from cryptography.hazmat.primitives.serialization import pkcs7
            certs.extend(pkcs7.load_der_pkcs7_certificates(blob))
        except Exception:
            pass
    if not certs:
        # Last resort: scan for embedded DER certificates. The ANTS and
        # ariadnext directories serve MIME multipart bodies ("--End"
        # boundaries with per-part Content-type headers) that wrap one DER
        # cert per part; a 0x30 0x82 SEQUENCE header + declared length
        # locates each candidate regardless of the wrapper format.
        i = 0
        while i < len(blob) - 4:
            if blob[i] == 0x30 and blob[i + 1] == 0x82:
                length = int.from_bytes(blob[i + 2:i + 4], "big") + 4
                try:
                    certs.append(x509.load_der_x509_certificate(blob[i:i + length]))
                    i += length
                    continue
                except Exception:
                    pass
            i += 1
    return certs


def main() -> int:
    logger.info("Fetching TSL: %s", TSL_URL)
    tsl_bytes = _http_get(TSL_URL)
    root = DET.fromstring(tsl_bytes)

    seq = root.findtext(".//tsl:TSLSequenceNumber", namespaces=NS)
    logger.info("TSL sequence number: %s", seq)

    by_ca: dict[str, list[x509.Certificate]] = {}

    for tsp in root.iterfind(".//tsl:TrustServiceProvider", NS):
        names = [el.text or "" for el in tsp.iterfind(".//tsl:TSPTradeName/tsl:Name", NS)]
        names += [el.text or "" for el in tsp.iterfind(".//tsl:TSPName/tsl:Name", NS)]
        ca_id = None
        for name in names:
            m = CA_ID_RE.search(name.upper())
            if m:
                ca_id = m.group(1)
                break
        if not ca_id:
            logger.warning("No CA-ID found in TSP names %r — skipped", names[:2])
            continue

        certs = by_ca.setdefault(ca_id, [])

        # TSL-embedded certificates: the trust anchors. NOTE: in the ANTS
        # TSL, DigitalId/X509Certificate lives in the *tsl* namespace, not
        # the XML-DSig one (which only signs the TSL document itself).
        tsl_certs: list[x509.Certificate] = []
        for svc in tsp.iterfind(".//tsl:TSPService", NS):
            status = svc.findtext(".//tsl:ServiceStatus", namespaces=NS) or ""
            if not status.endswith(STATUS_OK_SUFFIX):
                continue
            for b64 in svc.iterfind(".//tsl:ServiceDigitalIdentity//tsl:X509Certificate", NS):
                try:
                    tsl_certs.append(x509.load_der_x509_certificate(
                        base64.b64decode(b64.text or "")))
                except Exception as exc:
                    logger.warning("%s: undecodable TSL cert: %s", ca_id, exc)
        certs.extend(tsl_certs)

        # Leaf/signing certificates from the CA's published directory.
        # Transport may be plain http; trust comes from _is_pinned_or_chained
        # (identical to, or signed by, a TSL-embedded cert of this CA).
        uris = {(el.text or "").strip()
                for el in tsp.iterfind(".//tsl:TSPInformationURI/tsl:URI", NS)}
        for uri in sorted(u for u in uris if u.startswith(("http://", "https://"))):
            blob = _fetch_leaf_blob(uri)
            if blob is None:
                logger.info("%s: leaf fetch failed: %s", ca_id, uri)
                continue
            accepted = rejected = 0
            for cert in _certs_from_blob(blob):
                if _is_pinned_or_chained(cert, tsl_certs):
                    certs.append(cert)
                    accepted += 1
                else:
                    rejected += 1
            logger.info("%s: %s -> %d cert(s) accepted, %d rejected (no chain to TSL)",
                        ca_id, uri, accepted, rejected)

    # Dedupe per CA by fingerprint.
    deduped: dict[str, list[x509.Certificate]] = {}
    for ca_id, certs in sorted(by_ca.items()):
        seen: set[bytes] = set()
        kept = []
        for cert in certs:
            fp = cert.fingerprint(hashes.SHA256())
            if fp not in seen:
                seen.add(fp)
                kept.append(cert)
        deduped[ca_id] = kept

    # Cross-subject shared public keys. The live Certigna directory really
    # contains one (ENGIE and SNCF publishing byte-identical SPKIs) —
    # signatures cannot be attributed between such subjects, so the runtime
    # verifier surfaces the ambiguity in its detail string.
    spki_subjects: dict[bytes, set[str]] = {}
    for certs in deduped.values():
        for cert in certs:
            spki = cert.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)
            spki_subjects.setdefault(hashlib.sha256(spki).digest(), set()).add(
                cert.subject.rfc4514_string())

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    for old in STORE_DIR.glob("*.pem"):
        old.unlink()
    index: dict[str, list[dict]] = {}
    total = 0
    for ca_id, certs in deduped.items():
        entries = []
        for cert in certs:
            fname = f"{ca_id}_{len(entries)}.pem"
            (STORE_DIR / fname).write_bytes(
                cert.public_bytes(serialization.Encoding.PEM))
            subject = cert.subject.rfc4514_string()
            entry = {
                "file": fname,
                "subject": subject,
                "serial": format(cert.serial_number, "x").upper(),
                "not_before": cert.not_valid_before_utc.isoformat(),
                "not_after": cert.not_valid_after_utc.isoformat(),
                "candidate_ids": _candidate_ids(cert),
            }
            spki = cert.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)
            others = spki_subjects.get(hashlib.sha256(spki).digest(), set()) - {subject}
            if others:
                entry["shared_key_subjects"] = sorted(others)
            # CA directories publish TEST/SPECIMEN certificates whose keys
            # are shared for integration testing — a signature that resolves
            # to one must NEVER count as proof of authenticity. Kept in the
            # store (transparency, and so the verifier can say WHY) but
            # marked so the runtime refuses to grant "valid" through them.
            if re.search(r"\b(TEST|SPECIMEN)\b", subject, re.IGNORECASE) or \
                    re.search(r"(^|,)CN=0000(,|$)", subject):
                entry["test"] = True
            entries.append(entry)
            total += 1
        if entries:
            index[ca_id] = entries

    meta = {
        "source": TSL_URL,
        "tsl_sequence_number": seq,
        "certificate_count": total,
        "cas": sorted(index.keys()),
    }
    (STORE_DIR / "index.json").write_text(
        json.dumps({"meta": meta, "cas": index}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d certificate(s) for %d CA(s) to %s",
                total, len(index), STORE_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
