"""
Tests for the scoring module.

The scoring system combines results from all analysis modules into a final
trust score (0-100) and risk level (LOW/MEDIUM/HIGH/CRITICAL).

We test:
1. Evidence-ledger scoring (create_analysis_result end-to-end)
2. Flag collection and sorting
3. Document-type suppression (per-flag-code down-weighting)
4. Summary generation

(Risk-level thresholds are exercised via the evidence ledger's own
band/risk-level mapping in src/evidence.py — the standalone score-to-risk-
level helper this file used to unit-test directly was deleted as dead code:
no production caller used it once create_analysis_result switched to the
ledger engine.)
"""

import pytest
from src.models import Flag, ModuleResult, AnalysisResult
from src.scoring import (
    collect_all_flags,
    create_analysis_result,
)


# =============================================================================
# HELPERS — Create test data easily
# =============================================================================

def make_flag(severity: str = "low", code: str = "TEST_FLAG", message: str = "Test") -> Flag:
    """Shortcut to create a Flag without typing all fields every time."""
    return Flag(severity=severity, code=code, message=message)


def make_result(
    module: str = "metadata",
    score: int = 100,
    confidence: float = 1.0,
    flags: list[Flag] | None = None,
) -> ModuleResult:
    """Shortcut to create a ModuleResult for testing."""
    return ModuleResult(
        module=module,
        score=score,
        confidence=confidence,
        flags=flags or [],
    )


# =============================================================================
# TEST collect_all_flags
# =============================================================================

class TestCollectAllFlags:
    """Tests for flag collection and sorting."""

    def test_empty_results(self):
        """No modules = no flags."""
        assert collect_all_flags([]) == []

    def test_no_flags_in_results(self):
        """Modules ran but found nothing suspicious."""
        results = [make_result(flags=[]), make_result(module="content", flags=[])]
        assert collect_all_flags(results) == []

    def test_collects_from_multiple_modules(self):
        """Flags from all modules are gathered into one list."""
        results = [
            make_result(flags=[make_flag("low", "FLAG_A")]),
            make_result(module="content", flags=[make_flag("high", "FLAG_B")]),
        ]
        flags = collect_all_flags(results)
        assert len(flags) == 2

    def test_sorted_by_severity(self):
        """
        Flags should be sorted: critical first, then high, medium, low.
        This ensures the most important findings appear at the top of reports.
        """
        results = [
            make_result(flags=[
                make_flag("low", "L1"),
                make_flag("critical", "C1"),
                make_flag("medium", "M1"),
                make_flag("high", "H1"),
            ]),
        ]
        flags = collect_all_flags(results)
        severities = [f.severity for f in flags]
        assert severities == ["critical", "high", "medium", "low"]


# =============================================================================
# TEST create_analysis_result (the most important tests)
# =============================================================================

class TestCreateAnalysisResult:
    """
    Tests for the final result builder (file hash / timing passthrough, and the
    baseline "clean document" case). Ledger-specific scoring behavior — how
    flags translate into a score/band, hard-negative clamps, abstaining clean
    modules, etc. — is covered by TestEvidenceLedgerScoring below.
    """

    def test_clean_document(self):
        """A document with no flags should get a high score in the GENUINE band.

        Under the evidence ledger a no-evidence document is not a flat 100 —
        the prior log-odds (-4.0) puts it at ~98, still comfortably GENUINE/LOW.
        """
        results = [
            make_result(module="metadata", score=100),
            make_result(module="content", score=100),
        ]
        analysis = create_analysis_result("abc123", results)
        assert analysis.band == "GENUINE"
        assert analysis.risk_level == "LOW"
        assert analysis.trust_score >= 80

    def test_file_hash_preserved(self):
        """The file hash should be passed through to the result."""
        analysis = create_analysis_result("sha256:deadbeef", [])
        assert analysis.file_hash == "sha256:deadbeef"

    def test_analysis_time_preserved(self):
        """Analysis time should be passed through to the result."""
        analysis = create_analysis_result("abc", [], analysis_time_ms=1234)
        assert analysis.analysis_time_ms == 1234


# =============================================================================
# TEST the evidence ledger (replaces the old weighted-average + hard-cap tests)
# =============================================================================

class TestEvidenceLedgerScoring:
    """The evidence ledger replaces the confidence-weighted average.

    Under the Phase-1 constants (prior log-odds -4.0; soft weights
    low .2/med .7/high 1.5/critical 3.0), a no-evidence document scores ~98
    (GENUINE). Modules score from FLAGS, not from their numeric `score`.
    """

    def test_clean_modules_score_genuine(self):
        results = [make_result("metadata", 100, 1.0), make_result("content", 100, 1.0)]
        r = create_analysis_result("h", results)
        assert r.band == "GENUINE" and r.risk_level == "LOW" and r.trust_score >= 80

    def test_serious_finding_not_diluted_by_clean_modules(self):
        # One critical-weight finding; everything else clean (abstains). Under the
        # ledger this lands in REVIEW/SUSPICIOUS, NOT GENUINE — the clean modules
        # cannot dilute it the way the old average did.
        results = [
            make_result("metadata", 100, 1.0),
            make_result("content", 100, 1.0),
            make_result("fonts", 100, 1.0),
            make_result("structure", 0, 1.0, flags=[make_flag("critical", "STRUCT_SUSPICIOUS")]),
        ]
        r = create_analysis_result("h", results)
        assert r.band in ("REVIEW", "SUSPICIOUS")
        assert r.trust_score < 80

    def test_hard_negative_flag_clamps(self):
        results = [make_result("overlay", 0, 1.0,
                   flags=[make_flag("critical", "OVERLAY_COVER_AND_RETYPE_CONFIRMED")])]
        r = create_analysis_result("h", results)
        assert r.trust_score <= 15 and r.risk_level == "CRITICAL"

    def test_extraction_failed_is_unverifiable(self):
        results = [make_result("extraction", 0, 1.0,
                   flags=[make_flag("high", "EXTRACTION_FAILED")])]
        r = create_analysis_result("h", results)
        assert r.band == "UNVERIFIABLE" and r.risk_level == "UNVERIFIABLE"

    def test_abstaining_clean_modules_do_not_change_score(self):
        # The real invariant: adding clean (abstaining) modules must not move the
        # score of a flagged document — neither diluting nor inflating it.
        flagged_only = create_analysis_result("h", [
            make_result("visual", 85, 1.0, flags=[make_flag("medium", "VISUAL_X")]),
        ])
        with_clean = create_analysis_result("h", [
            make_result(f"m{i}", 100, 1.0) for i in range(8)
        ] + [make_result("visual", 85, 1.0, flags=[make_flag("medium", "VISUAL_X")])])
        assert with_clean.trust_score == flagged_only.trust_score

    def test_result_carries_ledger_and_band_range(self):
        results = [make_result("structure", 0, 1.0, flags=[make_flag("high", "STRUCT_JS")])]
        r = create_analysis_result("h", results)
        assert len(r.evidence_ledger) == 1
        assert r.score_low <= r.trust_score <= r.score_high


def test_scanner_pipeline_handles_non_numeric_hours():
    from src.scoring import create_analysis_result
    from src.models import ModuleResult, Flag
    bad = ModuleResult(
        module="structure",
        flags=[Flag(
            severity="critical",
            code="META_DOCUMENT_MODIFIED",
            message="x",
            details={"time_difference_hours": "not-a-number"},
        )],
        score=0, confidence=1.0,
    )
    # Must not raise, even for an identity document (the branch that reads hours).
    result = create_analysis_result("hash", [bad], document_type="identity")
    assert result.trust_score is not None


def _scanner_fp_metadata():
    """The known-FP flag: scanner saved, viewer reopened ~6 min later."""
    from src.models import ModuleResult, Flag
    return ModuleResult(module="metadata", score=0, confidence=1.0, flags=[
        Flag("critical", "META_DOCUMENT_MODIFIED", "reopened",
             details={"time_difference_hours": 0.1})])


def _identity_module(mrz_found: bool):
    from src.models import ModuleResult
    return ModuleResult(module="identity", score=100, confidence=1.0,
                        flags=[], details={"mrz_found": mrz_found})


def test_scanner_pipeline_identity_stays_genuine():
    # Authentic scanned ID: the known-FP META_DOCUMENT_MODIFIED critical is
    # downgraded to negligible evidence, so the document stays GENUINE rather
    # than being flagged for review. Requires a real MRZ (see below).
    res = create_analysis_result(
        "h", [_scanner_fp_metadata(), _identity_module(mrz_found=True)],
        document_type="identity")
    assert res.band == "GENUINE" and res.trust_score >= 80


def test_scanner_fp_exemption_requires_a_real_mrz():
    """Security: the exemption must not be buyable by steering the classifier.

    document_type comes from keyword hits in attacker-supplied text, and this
    downgrade sits outside DOC_TYPE_CODE_MULTIPLIERS — so without an MRZ gate,
    sprinkling ID keywords into a tampered invoice bought a 38 -> 91 swing.
    """
    res = create_analysis_result(
        "h", [_scanner_fp_metadata(), _identity_module(mrz_found=False)],
        document_type="identity")
    assert res.band != "GENUINE"

    # ...and with no identity module at all (the pure label-steering case).
    res_no_module = create_analysis_result(
        "h", [_scanner_fp_metadata()], document_type="identity")
    assert res_no_module.band != "GENUINE"


def test_scanner_fp_downgrade_does_not_mutate_the_audit_trail():
    """The per-module breakdown must keep showing the module's real finding."""
    meta = _scanner_fp_metadata()
    create_analysis_result("h", [meta, _identity_module(mrz_found=True)],
                           document_type="identity")
    assert meta.flags[0].severity == "critical"


class TestDocTypeSuppression:
    def test_font_diversity_suppressed_for_scientific(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        e = Evidence("FONTS_HIGH_DIVERSITY", 1.5, "soft", "tamper", "fonts", "m")
        out = apply_doc_type_suppression(e, "scientific")
        assert out is None or out.weight < 1.5  # suppressed/down-weighted

    def test_hard_evidence_never_suppressed(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        e = Evidence("CONTENT_SIREN_VAT_MISMATCH", 3.0, "hard", "tamper", "content", "m")
        out = apply_doc_type_suppression(e, "scientific")
        assert out is not None and out.kind == "hard" and out.weight == 3.0

    def test_universal_code_never_suppressed_by_classifier(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        # A tampering code an attacker cannot dodge by reclassifying the doc.
        e = Evidence("STRUCT_JAVASCRIPT_DETECTED", 3.0, "soft", "tamper", "structure", "m")
        out = apply_doc_type_suppression(e, "technical")
        assert out is not None and out.weight == 3.0

    def test_hard_evidence_ignores_even_an_explicit_table_suppressor(self):
        # Security: the hard-kind short-circuit must win over the table. If an
        # attacker (or a future careless edit) puts a <1.0 multiplier for a HARD
        # code under some doc type, hard evidence must STILL pass at full weight.
        # This test fails if the `if evidence.kind == "hard"` bypass is removed.
        from unittest.mock import patch
        from src.scoring import apply_doc_type_suppression, DOC_TYPE_CODE_MULTIPLIERS
        from src.evidence import Evidence
        e = Evidence("CONTENT_SIREN_VAT_MISMATCH", 3.0, "hard", "tamper", "content", "m")
        with patch.dict(DOC_TYPE_CODE_MULTIPLIERS,
                        {"scientific": {"CONTENT_SIREN_VAT_MISMATCH": 0.1}}, clear=False):
            out = apply_doc_type_suppression(e, "scientific")
        assert out is not None and out.kind == "hard" and out.weight == 3.0


class TestHardPositiveEvidenceCounts:
    """A cryptographically verified signature must move the score.

    Regression: HARD_POSITIVE_FLOOR (85) was gated on net_tamper < 1.5 nats,
    but with PRIOR_LOG_ODDS = -4.0 the soft score only falls below 85 past
    ~2.27 nats — the withhold gate always fired first, so the floor could never
    bind. Hard evidence also never enters aggregate_soft. Measured over 20,000
    random ledgers: TWOD_DOC_SIGNATURE_VALID changed the score in zero of them.
    """

    @staticmethod
    def _sig_valid():
        return ModuleResult(module="twod_doc", score=50, flags=[
            Flag("medium", "TWOD_DOC_SIGNATURE_VALID", "signature verifies")])

    def test_verified_signature_beats_an_unremarkable_document(self):
        with_sig = create_analysis_result("h", [self._sig_valid()]).trust_score
        without = create_analysis_result(
            "h", [ModuleResult(module="twod_doc", score=100, flags=[])]).trust_score
        assert with_sig > without

    def test_verified_signature_does_not_launder_real_findings(self):
        """It proves the BARCODE's data, not the rest of the page."""
        tampered = [ModuleResult(module="fonts", score=50, flags=[
            Flag("high", "FONTS_MIDLINE_CHANGE", "retyped value")])]
        assert (create_analysis_result("h", tampered + [self._sig_valid()]).trust_score
                == create_analysis_result("h", tampered).trust_score)

    def test_hard_tamper_still_wins_over_a_verified_signature(self):
        res = create_analysis_result("h", [self._sig_valid(), ModuleResult(
            module="overlay", score=0, flags=[
                Flag("critical", "OVERLAY_COVER_AND_RETYPE_CONFIRMED", "covered")])])
        assert res.trust_score <= 15


class TestBlindedHardDetector:
    """Crashing a detector must never be cheaper than being caught by it."""

    @staticmethod
    def _crashed_overlay():
        return ModuleResult(module="overlay", score=100, confidence=0.0, flags=[
            Flag("low", "MODULE_ERROR", "overlay failed",
                 details={"module": "overlay", "error": "RuntimeError"})])

    def test_blinded_hard_detector_denies_the_genuine_band(self):
        res = create_analysis_result("h", [self._crashed_overlay()],
                                     hard_detector_blinded=True)
        assert res.band != "GENUINE"

    def test_module_error_is_not_tamper_evidence(self):
        """MODULE_ERROR is pipeline status; it must not nudge the ledger."""
        crashed = create_analysis_result("h", [self._crashed_overlay()]).trust_score
        clean = create_analysis_result(
            "h", [ModuleResult(module="overlay", score=100, flags=[])]).trust_score
        assert crashed == clean

    def test_suspicious_still_passes_through(self):
        """Surviving modules that DID find tamper evidence stand on their own."""
        res = create_analysis_result("h", [self._crashed_overlay(), ModuleResult(
            module="structure", score=0, flags=[
                Flag("critical", "STRUCT_JAVASCRIPT_DETECTED", "js"),
                Flag("critical", "STRUCT_HIDDEN_MODIFICATION", "hidden")])],
            hard_detector_blinded=True)
        assert res.band == "SUSPICIOUS"


class TestNonEvidenceCodes:
    """Document descriptions and pipeline status must not charge tamper nats."""

    def test_description_codes_are_excluded_from_the_ledger(self):
        from src.scoring import NON_EVIDENCE_CODES
        for code in ("META_NO_PRODUCER", "IMAGES_NO_IMAGES", "FONTS_SYSTEM_FONTS",
                     "COVERAGE_NO_TEXT_LAYER", "MODULE_ERROR"):
            assert code in NON_EVIDENCE_CODES

    def test_a_described_document_scores_like_a_silent_one(self):
        described = create_analysis_result("h", [
            ModuleResult(module="metadata", score=90,
                         flags=[Flag("low", "META_NO_PRODUCER", "no producer")]),
            ModuleResult(module="fonts", score=90,
                         flags=[Flag("low", "FONTS_SYSTEM_FONTS", "Arial")]),
            ModuleResult(module="images", score=90,
                         flags=[Flag("low", "IMAGES_NO_IMAGES", "text-only")]),
        ])
        silent = create_analysis_result("h", [ModuleResult(module="x", score=100)])
        assert described.trust_score == silent.trust_score

    def test_descriptions_cannot_switch_off_a_verified_signature(self):
        """The reason this matters: every low flag adds 0.2 nats to net_tamper,
        which gates the hard-positive grant at 0.7. Four description flags used to
        drop a cryptographically verified document from 100 to 96."""
        sig = ModuleResult(module="twod_doc", score=50, flags=[
            Flag("medium", "TWOD_DOC_SIGNATURE_VALID", "ECDSA verifies")])
        noise = [
            ModuleResult(module="metadata", score=90,
                         flags=[Flag("low", "META_NO_PRODUCER", "n")]),
            ModuleResult(module="fonts", score=90,
                         flags=[Flag("low", "FONTS_SYSTEM_FONTS", "n")]),
            ModuleResult(module="images", score=90,
                         flags=[Flag("low", "IMAGES_NO_IMAGES", "n")]),
        ]
        assert (create_analysis_result("h", [sig] + noise).trust_score
                == create_analysis_result("h", [sig]).trust_score)

    def test_excluded_flags_still_reach_the_reader(self):
        """Excluded from SCORING, not from the report."""
        mod = ModuleResult(module="metadata", score=90,
                           flags=[Flag("low", "META_NO_PRODUCER", "no producer")])
        res = create_analysis_result("h", [mod])
        shown = [f.code for m in res.modules for f in m.flags]
        assert "META_NO_PRODUCER" in shown
