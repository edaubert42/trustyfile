"""ECDSA signature verification for French 2D-Doc barcodes.

WHAT THIS PROVES (and what it does not)
---------------------------------------
A 2D-Doc carries the document's key data plus an ECDSA signature produced
with the issuing authority's private key. If that signature verifies against
the authority's published certificate, the DATA INSIDE THE BARCODE is
authentic: nobody without the issuer's key can produce a valid signature
over modified data. This is the first hard, cryptographic, genuine-direction
evidence in TrustyFile — everything else is heuristics.

It does NOT prove the whole document: a fraudster can paste a genuine
barcode (their own tax notice's) into a forged PDF. The barcode's data would
verify — and contradict the forged visible text. That is why the HARD
POSITIVE flag is gated in twod_doc.py on BOTH a valid signature AND the
field-vs-visible-text consistency check passing.

MECHANICS (ANTS spec §3.5; same construction as betagouv/2ddoc-parser)
----------------------------------------------------------------------
- Signed payload: header + message zone — every byte of the barcode up to
  (excluding) the <US> (0x1F) separator, exactly as decoded.
- Signature: Base32 (RFC 4648) in C40-mode barcodes, decoding to the raw
  r‖s concatenation (NOT ASN.1 DER — must be re-encoded for `cryptography`).
- Curves: NIST P-256/P-384/P-521; hash follows the curve (SHA-256/384/512).
- Signer lookup: the header's (CA-ID, Cert-ID) resolved against the
  committed certificate store built by scripts/fetch_twod_doc_certs.py from
  the ANTS TSL. Fully offline at runtime.

FAILURE POSTURE
---------------
Every non-"valid" outcome distinguishes WHY, because they mean different
things for scoring:
- "invalid": cert resolved, bytes exact, math failed -> tamper-direction
  evidence (the caller decides severity).
- "cert_not_found": our snapshot may simply predate a new issuer -> neutral.
- "no_signature" / "malformed" / "unavailable" / "error": nothing provable
  either way -> neutral. This function never raises.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dep
    CRYPTOGRAPHY_AVAILABLE = False

# The committed certificate store (see scripts/fetch_twod_doc_certs.py).
STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "twod_doc_certs"

SignatureStatus = Literal[
    "valid",          # verified against an accredited cert, valid at emission
    "invalid",        # candidate cert(s) resolved, signature verifies with NONE
    "cert_not_found", # (CA-ID, Cert-ID) absent from the committed store
    "test_cert",      # verifies only through a published TEST/SPECIMEN cert
    "cert_window_mismatch",  # math verifies, but the cert was not valid at
                             # the document's emission date (possible replay
                             # with an old key — neutral, never "valid")
    "no_signature",   # barcode carries no signature zone
    "malformed",      # signature present but not decodable (bad Base32/size)
    "unavailable",    # cryptography lib or cert store missing
    "error",          # unexpected failure (never raises)
]


@dataclass
class SignatureCheck:
    """Outcome of verifying one 2D-Doc's signature.

    Attributes:
        status: see SignatureStatus.
        ca_id: the header's certification-authority identifier (e.g. FR04).
        cert_id: the header's certificate identifier (e.g. 0002).
        detail: human-oriented explanation of the outcome.
        cert_subject: subject of the certificate used, when one resolved.
    """
    status: SignatureStatus
    ca_id: str = ""
    cert_id: str = ""
    detail: str = ""
    cert_subject: str | None = None


class CertificateStore:
    """Read-only view over the committed ANTS certificate store.

    The store maps (CA-ID, Cert-ID) to X.509 certificates through candidate
    identifiers precomputed at build time (serial tails, SubjectKeyIdentifier
    tails, SPKI-hash prefixes, CN forms) — the runtime lookup is an exact
    dictionary hit, no heuristics. When the exact Cert-ID misses but the CA
    has exactly ONE certificate, that one is used (same fallback as the
    betagouv parser: several small CAs never rotated their key).
    """

    def __init__(self, store_dir: Path = STORE_DIR):
        self._dir = store_dir
        self._index: dict | None = None   # lazy; None = not loaded yet

    def _load(self) -> dict:
        if self._index is None:
            try:
                raw = json.loads(
                    (self._dir / "index.json").read_text(encoding="utf-8"))
                self._index = raw.get("cas", {})
            except Exception as exc:
                logger.warning("2D-Doc cert store unavailable: %s", exc)
                self._index = {}
        return self._index

    @property
    def available(self) -> bool:
        return bool(self._load())

    def resolve_candidates(self, ca_id: str, cert_id: str) -> list[dict]:
        """EVERY candidate entry for (ca_id, cert_id), certificate loaded.

        Candidate IDs are short derived identifiers (serial tails, SKI
        tails, hash prefixes), so the same 4-char cert_id can legitimately
        match several certificates — the committed store contains real
        collisions. Returning ALL of them lets the verifier try each key:
        only the true signer's public key can make the ECDSA math pass, so
        the ambiguity resolves itself and a collision can never brand a
        genuine document "invalid" through the wrong certificate.

        Entries marked ``test`` (published TEST/SPECIMEN certificates whose
        private keys are shared for integration testing) still resolve — so
        the verifier can report WHY — but the caller must never grant
        "valid" through them. The single-cert fallback ignores them.

        Returns:
            List of dicts: {"cert": Certificate, "test": bool,
            "shared_key_subjects": [str, ...]} — empty when nothing matches.
        """
        entries = self._load().get(ca_id.upper())
        if not entries:
            return []
        wanted = cert_id.upper()
        matches = [e for e in entries if wanted in e.get("candidate_ids", [])]
        if not matches:
            real = [e for e in entries if not e.get("test")]
            if len(real) == 1:
                matches = real  # single-cert CA fallback (never a test cert)
        out: list[dict] = []
        for entry in matches:
            try:
                pem = (self._dir / entry["file"]).read_bytes()
                out.append({
                    "cert": x509.load_pem_x509_certificate(pem),
                    "test": bool(entry.get("test")),
                    "shared_key_subjects": entry.get("shared_key_subjects", []),
                })
            except Exception as exc:
                logger.warning("2D-Doc cert %s unreadable: %s",
                               entry.get("file"), exc)
        return out


_DEFAULT_STORE = CertificateStore()

# Hash follows the curve (ANTS convention, mirrored from the official
# verifier implementations).
_HASH_BY_KEY_SIZE = {256: "SHA256", 384: "SHA384", 521: "SHA512"}


def _decode_base32_signature(signature: bytes) -> bytes | None:
    """Raw r||s bytes from the barcode's Base32 signature zone, or None.

    2D-Doc omits Base32 padding; b32decode requires it, so it is restored
    here. casefold tolerates the lowercase variants some encoders emit.
    """
    text = signature.decode("ascii", errors="strict").strip()
    if not text:
        return None
    pad = (-len(text)) % 8
    return base64.b32decode(text + "=" * pad, casefold=True)


def verify_twod_doc_signature(td, store: CertificateStore | None = None) -> SignatureCheck:
    """Verify one parsed 2D-Doc's ECDSA signature. Never raises.

    Args:
        td: a TwoDocData from parse_twod_doc — needs .header.raw,
            .header.ca_id, .header.cert_id, .raw_message, .signature.
        store: certificate store override (tests inject their own).

    Returns:
        SignatureCheck; status "valid" ONLY when the signature verified
        against the certificate resolved from the committed ANTS store.
    """
    ca_id = getattr(getattr(td, "header", None), "ca_id", "") or ""
    cert_id = getattr(getattr(td, "header", None), "cert_id", "") or ""

    if not CRYPTOGRAPHY_AVAILABLE:
        return SignatureCheck("unavailable", ca_id, cert_id,
                              "cryptography library not installed")
    store = store or _DEFAULT_STORE
    if not store.available:
        return SignatureCheck("unavailable", ca_id, cert_id,
                              "certificate store missing — run "
                              "scripts/fetch_twod_doc_certs.py")

    signature = getattr(td, "signature", None)
    if not signature:
        return SignatureCheck("no_signature", ca_id, cert_id,
                              "barcode carries no signature zone")

    try:
        try:
            raw_sig = _decode_base32_signature(signature)
        except Exception:
            raw_sig = None
        # r||s: two equally-sized big-endian integers. 64 bytes for P-256,
        # 96 for P-384, 132 for P-521 — always even, at least 64.
        if raw_sig is None or len(raw_sig) < 64 or len(raw_sig) % 2:
            return SignatureCheck("malformed", ca_id, cert_id,
                                  "signature zone is not valid Base32 r||s")

        candidates = store.resolve_candidates(ca_id, cert_id)
        if not candidates:
            return SignatureCheck(
                "cert_not_found", ca_id, cert_id,
                f"no certificate for ({ca_id}, {cert_id}) in the committed "
                "ANTS snapshot — issuer may postdate it")

        # The signed payload is every barcode byte before the <US>
        # separator, exactly as decoded. The scanner decodes the raw bytes
        # with latin-1 (a lossless byte<->str round trip), so latin-1 here
        # reconstructs the original bytes exactly.
        payload = (td.header.raw + td.raw_message).encode("latin-1")

        half = len(raw_sig) // 2
        r = int.from_bytes(raw_sig[:half], "big")
        s = int.from_bytes(raw_sig[half:], "big")
        der = encode_dss_signature(r, s)

        # Try EVERY candidate: short cert_ids collide (real collisions exist
        # in the committed store), and only the true signer's public key can
        # make the math pass — so trying all of them disambiguates for free,
        # and a collision can never brand a genuine document "invalid".
        winner: dict | None = None
        for entry in candidates:
            public_key = entry["cert"].public_key()
            if not isinstance(public_key, ec.EllipticCurvePublicKey):
                continue
            hash_name = _HASH_BY_KEY_SIZE.get(public_key.curve.key_size)
            if hash_name is None:
                continue
            try:
                public_key.verify(der, payload,
                                  ec.ECDSA(getattr(hashes, hash_name)()))
                winner = entry
                break
            except InvalidSignature:
                continue

        if winner is None:
            return SignatureCheck(
                "invalid", ca_id, cert_id,
                f"ECDSA verification failed against all "
                f"{len(candidates)} candidate certificate(s): the barcode "
                "data does not match its signature",
                candidates[0]["cert"].subject.rfc4514_string())

        cert = winner["cert"]
        subject = cert.subject.rfc4514_string()

        if winner["test"]:
            return SignatureCheck(
                "test_cert", ca_id, cert_id,
                "signature verifies only through a published TEST/SPECIMEN "
                "certificate — shared keys, proves nothing", subject)

        # Validity window: the certificate must have been valid when the
        # document was emitted (2D-Docs outlive their signing certs by
        # years, so "expired today" is normal — but a document whose
        # emission date falls OUTSIDE the cert's window was signed with a
        # key that was not accredited at that time: possible replay with a
        # compromised old key. Neutral, never "valid".
        # Residual risk, accepted: offline verification cannot check
        # revocation (CRLs), and an attacker who owns a compromised key can
        # also backdate the emission field. Full mitigation needs an online
        # revocation source.
        reference = getattr(getattr(td, "header", None), "emission_date", None)
        if reference is None:
            import datetime as _dt
            reference = _dt.datetime.now(_dt.timezone.utc).date()
        if not (cert.not_valid_before_utc.date() <= reference
                <= cert.not_valid_after_utc.date()):
            return SignatureCheck(
                "cert_window_mismatch", ca_id, cert_id,
                f"signature verifies, but the certificate was not valid on "
                f"the document's emission date ({reference.isoformat()}); "
                f"cert window {cert.not_valid_before_utc.date()} — "
                f"{cert.not_valid_after_utc.date()}", subject)

        detail = ("ECDSA signature verified against the ANTS-published "
                  "issuer certificate")
        if winner["shared_key_subjects"]:
            # The same public key is published under other subjects in the
            # CA directory — attribution between them is ambiguous.
            detail += ("; NOTE: key also published for "
                       + "; ".join(winner["shared_key_subjects"]))
        return SignatureCheck("valid", ca_id, cert_id, detail, subject)

    except Exception as exc:
        logger.warning("2D-Doc signature verification error: %s", exc)
        return SignatureCheck("error", ca_id, cert_id, type(exc).__name__)
