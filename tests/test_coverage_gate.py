"""Coverage gate (spec 2026-07-17): blind documents fail closed.

Empirical baseline this feature fixes: an image-only PDF (0 text chars)
scored 96/GENUINE/LOW with only META_NO_PRODUCER + IMAGES_IMAGE_ONLY_PDF.
"""
import fitz
import pytest

from src.analyzer import MIN_TEXT_CHARS_PER_PAGE, _coverage_blind
from src.models import ModuleResult


def test_threshold_is_strictly_below_50():
    assert MIN_TEXT_CHARS_PER_PAGE == 50
    blind_49, avg_49 = _coverage_blind(["x" * 49], 1, [])
    not_blind_50, avg_50 = _coverage_blind(["x" * 50], 1, [])
    assert blind_49 is True and avg_49 == 49.0
    assert not_blind_50 is False and avg_50 == 50.0


def test_average_is_over_page_count_and_ignores_whitespace():
    # 100 non-ws chars spread over 2 pages -> avg 50 -> not blind
    assert _coverage_blind(["x" * 100, ""], 2, [])[0] is False
    # 99 non-ws chars over 2 pages -> avg 49.5 -> blind
    assert _coverage_blind(["x" * 99, ""], 2, [])[0] is True
    # whitespace never counts
    assert _coverage_blind(["  \n\t " * 100], 1, [])[0] is True
    # zero pages must not divide by zero
    assert _coverage_blind([], 0, [])[0] is True


def test_identity_mrz_exemption():
    blind_doc = ["x" * 5]
    with_mrz = [ModuleResult(module="identity", details={"mrz_found": True})]
    without_mrz = [ModuleResult(module="identity", details={"mrz_found": False})]
    crashed = [ModuleResult(module="identity", details={})]
    other_module = [ModuleResult(module="images", details={"mrz_found": True})]

    assert _coverage_blind(blind_doc, 1, with_mrz)[0] is False
    assert _coverage_blind(blind_doc, 1, without_mrz)[0] is True   # fail closed
    assert _coverage_blind(blind_doc, 1, crashed)[0] is True       # fail closed
    assert _coverage_blind(blind_doc, 1, other_module)[0] is True  # only identity counts


from src.evidence import Evidence, aggregate_evidence


def _ev(code: str, weight: float, group: str) -> Evidence:
    return Evidence(code=code, weight=weight, kind="soft",
                    direction="tamper", group=group, message="")


def test_blind_empty_ledger_becomes_unverifiable():
    # Empty ledger scores 98/GENUINE today; blind must override it.
    agg = aggregate_evidence([], coverage_blind=True)
    assert agg.band == "UNVERIFIABLE"
    assert agg.risk_level == "UNVERIFIABLE"
    assert agg.score == 0 and agg.score_low == 0 and agg.score_high == 0


def test_blind_review_band_becomes_unverifiable():
    # One critical (3.0 nats): logit -4+3=-1 -> score 73 -> REVIEW -> override.
    agg = aggregate_evidence([_ev("X_CRIT", 3.0, "images")], coverage_blind=True)
    assert agg.band == "UNVERIFIABLE"


def test_blind_preserves_suspicious():
    # Two criticals in different groups: logit -4+3+3=2 -> score 12 -> SUSPICIOUS.
    ledger = [_ev("X_CRIT", 3.0, "images"), _ev("Y_CRIT", 3.0, "forensics")]
    agg = aggregate_evidence(ledger, coverage_blind=True)
    assert agg.band == "SUSPICIOUS"
    assert agg.score == aggregate_evidence(ledger, coverage_blind=False).score
    assert agg.ledger  # ledger is preserved for display


def test_extraction_failure_precedence_unchanged():
    agg = aggregate_evidence([], analyzable=False, coverage_blind=True)
    assert agg.band == "UNVERIFIABLE" and agg.score == 0


from src.analyzer import TrustyFileAnalyzer

INVOICE_TEXT = (
    "FACTURE N° 2026-0142\n"
    "ACME Services SARL — 12 rue de la Paix, 75002 Paris\n"
    "Total HT : 1 000,00 EUR — TVA 20% : 200,00 EUR\n"
    "Total TTC : 1 200,00 EUR\n"
)


def _write_text_pdf(path) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    doc.save(str(path))
    doc.close()


def _write_image_only_pdf(path) -> None:
    src = fitz.open()
    page = src.new_page()
    page.insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    pix = page.get_pixmap(dpi=150)
    src.close()
    doc = fitz.open()
    out = doc.new_page(width=595, height=842)
    out.insert_image(out.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def _analyze(path):
    return TrustyFileAnalyzer(
        enable_external=False, enable_qr_scan=False).analyze(str(path))


def test_image_only_scan_bands_unverifiable(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "UNVERIFIABLE"
    assert result.risk_level == "UNVERIFIABLE"
    assert result.trust_score == 0
    codes = [f.code for m in result.modules for f in m.flags]
    assert "COVERAGE_NO_TEXT_LAYER" in codes


def test_text_document_is_unaffected(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_text_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "GENUINE"
    codes = [f.code for m in result.modules for f in m.flags]
    assert "COVERAGE_NO_TEXT_LAYER" not in codes


# --- Task 6: corpus-level coverage gate (build_corpus) ---
from scripts.build_corpus import build_corpus

def test_no_legit_corpus_doc_is_blind(tmp_path):
    """Permanent invariant: every synthetic LEGITIMATE trap doc must carry a
    realistic text layer. If this fails after adding a new trap doc, make the
    DOC realistic (real documents have text) — never weaken the gate."""
    paths = build_corpus(str(tmp_path))
    analyzer = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False)
    blind = [p for p in paths["legitimate"]
             if analyzer.analyze(p).band == "UNVERIFIABLE"]
    assert not blind, f"legit trap docs banded UNVERIFIABLE: {blind}"


def test_corpus_unverifiable_category(tmp_path):
    paths = build_corpus(str(tmp_path))
    assert paths["unverifiable"], "build_corpus must produce the unverifiable category"
    analyzer = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False)
    for p in paths["unverifiable"]:
        assert analyzer.analyze(p).band == "UNVERIFIABLE", p


# --- Task 4: scan-specific UNVERIFIABLE verdict ---
from src.summary import generate_rich_summary


def test_blind_doc_gets_scan_specific_verdict(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    from src.summary import COVERAGE_VERDICT, COVERAGE_VERDICT_FR
    assert generate_rich_summary(result, "en").verdict == COVERAGE_VERDICT
    assert generate_rich_summary(result, "fr").verdict == COVERAGE_VERDICT_FR


def test_scan_reasons_do_not_repeat_the_headline(tmp_path):
    """The headline says "scan or photo": an image-only finding only repeats it."""
    from src.scoring import collect_all_flags
    from src.summary import SCAN_RESTATING_CODES, top_reasons
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "UNVERIFIABLE"
    flags = collect_all_flags(result.modules)
    assert any(f.code in SCAN_RESTATING_CODES and f.severity != "low" for f in flags)
    assert not [f for f in top_reasons(flags, band=result.band)
                if f.code in SCAN_RESTATING_CODES]


def test_unreadable_doc_keeps_generic_unverifiable_verdict(tmp_path):
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    result = _analyze(bad)
    assert result.band == "UNVERIFIABLE"
    verdict = generate_rich_summary(result, "en").verdict
    assert "could not be read or verified" in verdict
    from src.summary import COVERAGE_VERDICT
    assert verdict != COVERAGE_VERDICT


# --- Task 5: encrypted-file message rider ---
from src.summary import get_flag_sentence


def _write_encrypted_pdf(path) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="secret-user-pw", owner_pw="secret-owner-pw")
    doc.close()


def test_encrypted_file_message_mentions_password(tmp_path):
    pdf = tmp_path / "locked.pdf"
    _write_encrypted_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "UNVERIFIABLE"  # already true today
    flag = next(f for m in result.modules for f in m.flags
                if f.code == "EXTRACTION_FAILED")
    assert flag.details.get("encrypted") is True
    assert "password" in get_flag_sentence(flag, "en").lower()
    assert "mot de passe" in get_flag_sentence(flag, "fr").lower()


def test_garbage_file_message_stays_generic(tmp_path):
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    result = _analyze(bad)
    flag = next(f for m in result.modules for f in m.flags
                if f.code == "EXTRACTION_FAILED")
    assert not flag.details.get("encrypted")
    assert "valid PDF" in get_flag_sentence(flag, "en")


# --- Final-review fixes: ledger cleanliness + crash-vs-scan verdict ---
def test_coverage_flag_not_in_evidence_ledger(tmp_path):
    """The coverage flag's scoring role is the band override (coverage_blind),
    NOT ledger weight — it must not enter the ledger, else its 0.2 nats could
    nudge the band or trip the genuine-laundering threshold. It stays visible
    as a module flag for display + verdict selection."""
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    ledger_codes = [e.code for e in result.evidence_ledger]
    assert "COVERAGE_NO_TEXT_LAYER" not in ledger_codes
    flag_codes = [f.code for m in result.modules for f in m.flags]
    assert "COVERAGE_NO_TEXT_LAYER" in flag_codes


def test_crash_and_blind_doc_gets_generic_not_scan_verdict():
    """A doc that is UNVERIFIABLE by pipeline failure (EXTRACTION_FAILED) AND
    incidentally blind must read as a pipeline failure, not a scan."""
    from src.models import AnalysisResult, ModuleResult, Flag
    result = AnalysisResult(
        file_hash="x", trust_score=0, risk_level="UNVERIFIABLE",
        modules=[
            ModuleResult(module="analysis",
                         flags=[Flag("high", "EXTRACTION_FAILED", "x")], score=0),
            ModuleResult(module="coverage",
                         flags=[Flag("low", "COVERAGE_NO_TEXT_LAYER", "x")], score=0),
        ],
        band="UNVERIFIABLE",
    )
    verdict = generate_rich_summary(result, "en").verdict
    assert "could not be read or verified" in verdict
    assert "scanned or photographed" not in verdict
