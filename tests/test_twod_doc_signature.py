"""2D-Doc ECDSA signature verification (Phase 8 crypto build).

The verifier cannot be tested against a real ANTS-signed document (none may
be committed — the corpus is gitignored and no public test vector carries a
private key we control). Instead these tests do what the spec does: generate
our own P-256 keypair, self-sign a certificate, build a certificate store in
tmp_path shaped exactly like scripts/fetch_twod_doc_certs.py's output, sign
real 2D-Doc payloads with our key, and run the REAL pipeline
(parse_twod_doc -> verify_twod_doc_signature) against it.

That validates every moving part except the authenticity of the committed
ANTS certificates themselves — which only a real signed sample can prove
(recorded as an accepted residual, same posture as the module's provisional
severities).
"""

from __future__ import annotations

import base64
import datetime
import json
from pathlib import Path

import pytest

cryptography = pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID

from src.evidence import GENUINE_CODES, HARD_POSITIVE_CODES, flag_to_evidence
from src.models import Flag
from src.modules.twod_doc import _signature_flags, parse_twod_doc
from src.modules.twod_doc_crypto import (
    CertificateStore,
    SignatureCheck,
    verify_twod_doc_signature,
)

US = "\x1f"
GS = "\x1d"

# Same synthetic header as tests/test_twod_doc_module.py:
# DC 03 | ca_id=FR00 | cert_id=0123 | dates 4523/4501 | doc type 01 | perimeter 01
_HEADER = "DC03FR000001234523450101"
_CA_ID = "FR00"
_CERT_ID = "0001"


# --- Test PKI -----------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FR00-0001")])
    # Wide validity window: the synthetic header's emission date ("4523"
    # hex = days since 2000-01-01) decodes to ~2048, and the verifier now
    # requires the cert to be valid AT EMISSION.
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
        .not_valid_after(datetime.datetime(2080, 1, 1, tzinfo=datetime.timezone.utc))
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.fixture()
def store(tmp_path, keypair) -> CertificateStore:
    """A store directory shaped like fetch_twod_doc_certs.py's output."""
    _, cert = keypair
    (tmp_path / "FR00_0.pem").write_bytes(
        cert.public_bytes(serialization.Encoding.PEM))
    index = {
        "meta": {"source": "test", "certificate_count": 1},
        "cas": {
            _CA_ID: [{
                "file": "FR00_0.pem",
                "subject": "CN=FR00-0001",
                "candidate_ids": [_CERT_ID, "FR00-0001"],
            }],
        },
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return CertificateStore(store_dir=tmp_path)


def _sign_payload(key, payload: str) -> str:
    """Base32 signature zone for `payload`, exactly as a 2D-Doc encodes it:
    raw r||s with fixed-size halves, Base32 without padding."""
    der = key.sign(payload.encode("latin-1"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    size = (key.curve.key_size + 7) // 8
    raw = r.to_bytes(size, "big") + s.to_bytes(size, "big")
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _signed_barcode(key, message: str) -> str:
    payload = _HEADER + message
    return payload + US + _sign_payload(key, payload)


_MESSAGE = "26FR0447050101"  # DI 26 (postal+country style payload used in module tests)


# --- Verifier outcomes --------------------------------------------------------


class TestVerifySignature:
    def test_valid_signature_verifies(self, keypair, store):
        key, _ = keypair
        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        check = verify_twod_doc_signature(td, store=store)
        assert check.status == "valid"
        assert check.ca_id == _CA_ID and check.cert_id == _CERT_ID
        assert check.cert_subject == "CN=FR00-0001"

    def test_tampered_message_is_invalid(self, keypair, store):
        """Change ONE character of the signed zone: math must fail."""
        key, _ = keypair
        barcode = _signed_barcode(key, _MESSAGE)
        tampered = barcode.replace("4705", "4706", 1)
        td = parse_twod_doc(tampered)
        assert verify_twod_doc_signature(td, store=store).status == "invalid"

    def test_tampered_header_is_invalid(self, keypair, store):
        """The header is inside the signed payload too (emission date here —
        same CA/cert so resolution still succeeds)."""
        key, _ = keypair
        barcode = _signed_barcode(key, _MESSAGE)
        tampered = barcode.replace("4523", "4524", 1)
        td = parse_twod_doc(tampered)
        assert verify_twod_doc_signature(td, store=store).status == "invalid"

    def test_unknown_ca_is_cert_not_found(self, keypair, store):
        key, _ = keypair
        barcode = _signed_barcode(key, _MESSAGE).replace("FR00", "FR99", 1)
        td = parse_twod_doc(barcode)
        assert verify_twod_doc_signature(td, store=store).status == "cert_not_found"

    def test_garbage_signature_is_malformed(self, store):
        td = parse_twod_doc(_HEADER + _MESSAGE + US + "@@@not-base32@@@")
        assert verify_twod_doc_signature(td, store=store).status == "malformed"

    def test_missing_signature_zone(self, store):
        td = parse_twod_doc(_HEADER + _MESSAGE)
        assert verify_twod_doc_signature(td, store=store).status == "no_signature"

    def test_missing_store_is_unavailable(self, keypair, tmp_path):
        key, _ = keypair
        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        empty = CertificateStore(store_dir=tmp_path / "nope")
        assert verify_twod_doc_signature(td, store=empty).status == "unavailable"

    def test_cert_expired_at_emission_is_window_mismatch(self, keypair, tmp_path):
        """Math verifies, but the cert's validity window excludes the
        document's emission date: possible replay with an old key.
        Neutral status, never 'valid'."""
        key, _ = keypair
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FR00-0001")])
        old_cert = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
            .not_valid_after(datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc))
            .sign(key, hashes.SHA256())
        )
        (tmp_path / "FR00_0.pem").write_bytes(
            old_cert.public_bytes(serialization.Encoding.PEM))
        index = {"cas": {_CA_ID: [{
            "file": "FR00_0.pem", "candidate_ids": [_CERT_ID]}]}}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")

        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))  # emission ~2048
        check = verify_twod_doc_signature(
            td, store=CertificateStore(store_dir=tmp_path))
        assert check.status == "cert_window_mismatch"

    def test_colliding_cert_ids_disambiguate_by_verification(self, keypair, tmp_path):
        """Two certs share the candidate id (real collisions exist in the
        committed store); the signer is listed SECOND. The verifier must try
        both and land on the one whose key verifies — never brand the
        document invalid through the wrong first match."""
        key, cert = keypair
        decoy_key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DECOY")])
        decoy = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(decoy_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
            .not_valid_after(datetime.datetime(2080, 1, 1, tzinfo=datetime.timezone.utc))
            .sign(decoy_key, hashes.SHA256())
        )
        (tmp_path / "FR00_0.pem").write_bytes(
            decoy.public_bytes(serialization.Encoding.PEM))
        (tmp_path / "FR00_1.pem").write_bytes(
            cert.public_bytes(serialization.Encoding.PEM))
        index = {"cas": {_CA_ID: [
            {"file": "FR00_0.pem", "candidate_ids": [_CERT_ID]},
            {"file": "FR00_1.pem", "candidate_ids": [_CERT_ID]},
        ]}}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")

        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        check = verify_twod_doc_signature(
            td, store=CertificateStore(store_dir=tmp_path))
        assert check.status == "valid"
        assert check.cert_subject == "CN=FR00-0001"

    def test_test_certificate_never_grants_valid(self, keypair, tmp_path):
        """A published TEST/SPECIMEN certificate has shared keys: a perfect
        signature through one proves nothing and must NOT read as valid."""
        key, cert = keypair
        (tmp_path / "FR00_0.pem").write_bytes(
            cert.public_bytes(serialization.Encoding.PEM))
        index = {"cas": {_CA_ID: [{
            "file": "FR00_0.pem", "candidate_ids": [_CERT_ID], "test": True}]}}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        check = verify_twod_doc_signature(
            td, store=CertificateStore(store_dir=tmp_path))
        assert check.status == "test_cert"

    def test_single_cert_fallback_skips_test_certs(self, keypair, tmp_path):
        """Fallback must not route unknown cert_ids through a test cert."""
        key, cert = keypair
        (tmp_path / "FR00_0.pem").write_bytes(
            cert.public_bytes(serialization.Encoding.PEM))
        index = {"cas": {_CA_ID: [{
            "file": "FR00_0.pem", "candidate_ids": ["OTHER"], "test": True}]}}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        check = verify_twod_doc_signature(
            td, store=CertificateStore(store_dir=tmp_path))
        assert check.status == "cert_not_found"

    def test_single_cert_ca_fallback(self, keypair, tmp_path):
        """Unknown cert_id but the CA has exactly one cert: that one is used
        (several small CAs never rotated their key)."""
        key, cert = keypair
        (tmp_path / "FR00_0.pem").write_bytes(
            cert.public_bytes(serialization.Encoding.PEM))
        index = {"cas": {_CA_ID: [{
            "file": "FR00_0.pem", "candidate_ids": ["SOMETHING-ELSE"]}]}}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        td = parse_twod_doc(_signed_barcode(key, _MESSAGE))
        check = verify_twod_doc_signature(
            td, store=CertificateStore(store_dir=tmp_path))
        assert check.status == "valid"


# --- IBAN cross-check (DI 31) — the payment-redirection binding ---------------


class TestIbanCrossCheck:
    """DI 31 was in the registry but never compared — the exact gap a
    'fraude au faux RIB' walked through (adversarial review 2026-08-13)."""

    IBAN = "FR7630006000011234567890189"

    def _verify(self, pdf_text):
        from src.modules.twod_doc import verify_2d_doc_against_pdf
        GS = "\x1d"
        td = parse_twod_doc(_HEADER + "31" + self.IBAN + GS + "26FR04")
        return verify_2d_doc_against_pdf(td, pdf_text)

    def test_matching_iban_passes_even_with_spacing(self):
        result = self._verify(
            "Merci de régler sur le compte FR76 3000 6000 0112 3456 7890 189")
        comp = [c for c in result.comparisons if c.di == "31"]
        assert len(comp) == 1 and comp[0].match is True

    def test_swapped_iban_is_a_mismatch(self):
        """Visible text shows a DIFFERENT account than the signed barcode."""
        result = self._verify(
            "Merci de régler sur le compte FR76 3000 4000 0298 7654 3210 943")
        comp = [c for c in result.comparisons if c.di == "31"]
        assert len(comp) == 1 and comp[0].match is False
        assert any("IBAN" in i for i in result.fraud_indicators)


# --- Flag gating (_signature_flags) -------------------------------------------


def _consistent_flag() -> Flag:
    return Flag(severity="medium", code="TWOD_DOC_CONSISTENT",
                message="all fields match", details={})


def _mismatch_flag() -> Flag:
    return Flag(severity="high", code="TWOD_DOC_FIELD_MISMATCH",
                message="field differs", details={})


class TestSignatureFlagGating:
    def test_valid_plus_consistent_is_hard_positive(self):
        check = SignatureCheck("valid", _CA_ID, _CERT_ID, cert_subject="CN=x")
        flags = _signature_flags(check, [_consistent_flag()])
        assert [f.code for f in flags] == ["TWOD_DOC_SIGNATURE_VALID"]

    def test_valid_without_consistency_emits_nothing(self):
        """A valid signature alone proves the barcode, not the page: without
        the field cross-check there is no binding — no positive flag."""
        check = SignatureCheck("valid", _CA_ID, _CERT_ID)
        assert _signature_flags(check, []) == []

    def test_valid_with_mismatch_emits_nothing_positive(self):
        """Authentic barcode + edited page: the mismatch flags already carry
        the tamper evidence; a positive flag here would be laundering."""
        check = SignatureCheck("valid", _CA_ID, _CERT_ID)
        assert _signature_flags(check, [_mismatch_flag()]) == []

    def test_invalid_is_critical_soft_flag(self):
        # CRITICAL, not high: a lone high scored 92/GENUINE under the -4.0-nat
        # prior, so a fabricated government barcode was reported as "no issues
        # detected". Still SOFT — see test_signature_invalid_is_soft_tamper —
        # so a resolver mismatch lands the document in REVIEW, never in the
        # fraud bands.
        check = SignatureCheck("invalid", _CA_ID, _CERT_ID)
        flags = _signature_flags(check, [_consistent_flag()])
        assert [f.code for f in flags] == ["TWOD_DOC_SIGNATURE_INVALID"]
        assert flags[0].severity == "critical"

    def test_invalid_signature_leaves_the_genuine_band(self):
        """The point of the severity bump: this must not read as legitimate."""
        from src.models import ModuleResult
        from src.scoring import create_analysis_result
        check = SignatureCheck("invalid", _CA_ID, _CERT_ID)
        res = create_analysis_result("h", [ModuleResult(
            module="twod_doc", score=50,
            flags=_signature_flags(check, [_consistent_flag()]))])
        assert res.band != "GENUINE"

    @pytest.mark.parametrize("status", [
        "cert_not_found", "test_cert", "no_signature", "malformed",
        "unavailable", "error"])
    def test_neutral_statuses_emit_no_flags(self, status):
        check = SignatureCheck(status, _CA_ID, _CERT_ID)
        assert _signature_flags(check, [_consistent_flag()]) == []


# --- Ledger registration -------------------------------------------------------


class TestEvidenceRegistration:
    def test_signature_valid_is_hard_positive(self):
        assert "TWOD_DOC_SIGNATURE_VALID" in HARD_POSITIVE_CODES
        ev = flag_to_evidence(
            Flag(severity="medium", code="TWOD_DOC_SIGNATURE_VALID",
                 message="x"), "twod_doc")
        assert ev.kind == "hard" and ev.direction == "genuine"

    def test_signature_valid_also_genuine_for_display(self):
        """The summary layer keys off GENUINE_CODES to keep genuine-direction
        flags out of tamper-theme dominance — membership is load-bearing."""
        assert "TWOD_DOC_SIGNATURE_VALID" in GENUINE_CODES

    def test_signature_invalid_is_soft_tamper(self):
        assert "TWOD_DOC_SIGNATURE_INVALID" not in HARD_POSITIVE_CODES
        ev = flag_to_evidence(
            Flag(severity="high", code="TWOD_DOC_SIGNATURE_INVALID",
                 message="x"), "twod_doc")
        assert ev.kind == "soft" and ev.direction == "tamper"

    def test_hard_positive_floors_final_score(self):
        """End to end through the ledger: a hard positive floors at 85."""
        from src.models import ModuleResult
        from src.scoring import create_analysis_result
        mr = ModuleResult(
            module="twod_doc",
            flags=[Flag(severity="medium", code="TWOD_DOC_SIGNATURE_VALID",
                        message="verified")],
            score=100, confidence=1.0)
        result = create_analysis_result(
            file_hash="x", module_results=[mr], analysis_time_ms=1)
        assert result.trust_score >= 85
        assert result.band == "GENUINE"

    def test_floor_withheld_under_heavy_tamper_evidence(self):
        """The RIB-fraud laundering scenario (adversarial review
        2026-08-13): genuine barcode + heavy independent tamper evidence
        (e.g. an edited IBAN the barcode does not cover). The hard-positive
        flag survives, but the 85 floor must NOT apply — otherwise the tool
        would launder exactly the fraud it exists to catch."""
        from src.models import ModuleResult
        from src.scoring import create_analysis_result
        twod = ModuleResult(
            module="twod_doc",
            flags=[Flag(severity="medium", code="TWOD_DOC_SIGNATURE_VALID",
                        message="verified")],
            score=100, confidence=1.0)
        overlay = ModuleResult(
            module="overlay",
            flags=[
                Flag(severity="high", code="OVERLAY_OPAQUE_FILL_OVER_TEXT",
                     message="covered value"),
                Flag(severity="high", code="FONTS_MIDLINE_CHANGE",
                     message="font swap inside the IBAN line"),
            ],
            score=40, confidence=1.0)
        structure = ModuleResult(
            module="structure",
            flags=[Flag(severity="high", code="STRUCT_HIDDEN_MODIFICATION",
                        message="revision diff shows a changed value")],
            score=40, confidence=1.0)

        with_pos = create_analysis_result(
            file_hash="x", module_results=[twod, overlay, structure],
            analysis_time_ms=1)
        without_pos = create_analysis_result(
            file_hash="x", module_results=[overlay, structure],
            analysis_time_ms=1)

        assert with_pos.trust_score < 85
        assert with_pos.band != "GENUINE"
        # The floor must be fully inert here: same score as if the hard
        # positive were absent (the flag itself stays visible in the ledger).
        assert with_pos.trust_score == without_pos.trust_score
