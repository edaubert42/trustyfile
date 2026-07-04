"""
Tests for the scoring module.

The scoring system combines results from all analysis modules into a final
trust score (0-100) and risk level (LOW/MEDIUM/HIGH/CRITICAL).

We test:
1. Risk level thresholds (score → risk level)
2. Evidence-ledger scoring (create_analysis_result end-to-end)
3. Flag collection and sorting
4. Document-type suppression (per-flag-code down-weighting)
5. Summary generation
"""

import pytest
from src.models import Flag, ModuleResult, AnalysisResult
from src.scoring import (
    get_risk_level,
    collect_all_flags,
    count_flags_by_severity,
    create_analysis_result,
    generate_summary,
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
# TEST get_risk_level
# =============================================================================

class TestGetRiskLevel:
    """
    Tests for score → risk level conversion.

    Thresholds:
        80-100 → LOW
        50-79  → MEDIUM
        20-49  → HIGH
        0-19   → CRITICAL
    """

    # pytest.mark.parametrize runs the SAME test with DIFFERENT inputs.
    # Each tuple is (input_score, expected_risk_level).
    # This avoids writing 10 separate test functions that do the same thing.
    @pytest.mark.parametrize("score, expected", [
        (100, "LOW"),       # Perfect score
        (80, "LOW"),        # Exact boundary
        (85, "LOW"),        # Middle of LOW range
        (79, "MEDIUM"),     # Just below LOW threshold
        (50, "MEDIUM"),     # Exact boundary
        (65, "MEDIUM"),     # Middle of MEDIUM range
        (49, "HIGH"),       # Just below MEDIUM threshold
        (20, "HIGH"),       # Exact boundary
        (35, "HIGH"),       # Middle of HIGH range
        (19, "CRITICAL"),   # Just below HIGH threshold
        (0, "CRITICAL"),    # Minimum score
        (10, "CRITICAL"),   # Middle of CRITICAL range
    ])
    def test_thresholds(self, score, expected):
        assert get_risk_level(score) == expected


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
# TEST count_flags_by_severity
# =============================================================================

class TestCountFlagsBySeverity:
    """Tests for flag counting."""

    def test_empty_list(self):
        assert count_flags_by_severity([]) == {
            "critical": 0, "high": 0, "medium": 0, "low": 0
        }

    def test_counts_each_severity(self):
        flags = [
            make_flag("low"), make_flag("low"), make_flag("low"),
            make_flag("medium"), make_flag("medium"),
            make_flag("high"),
            make_flag("critical"),
        ]
        counts = count_flags_by_severity(flags)
        assert counts == {"critical": 1, "high": 1, "medium": 2, "low": 3}

    def test_unknown_severity_ignored(self):
        """A flag with an unexpected severity should not crash, just be ignored."""
        flags = [make_flag("unknown_severity")]
        counts = count_flags_by_severity(flags)
        # All counts should be 0 — unknown severity is not counted
        assert counts == {"critical": 0, "high": 0, "medium": 0, "low": 0}


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
        results = [make_result("content", 0, 1.0,
                   flags=[make_flag("critical", "CONTENT_SIREN_VAT_MISMATCH")])]
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


# =============================================================================
# TEST generate_summary
# =============================================================================

class TestGenerateSummary:
    """Tests for the human-readable summary."""

    def test_low_risk_summary(self):
        result = create_analysis_result("abc", [
            make_result(score=100),
        ])
        summary = generate_summary(result)
        assert "legitimate" in summary.lower()

    def test_medium_risk_summary(self):
        # Under the ledger, a bare score=65 with no flags contributes no
        # evidence (abstains) and would land GENUINE. To reach MEDIUM/REVIEW,
        # give it a single soft critical-weight flag: -4.0 + 3.0 = -1.0 logit
        # -> score ~73 (REVIEW/MEDIUM).
        result = create_analysis_result("abc", [
            make_result(score=0, flags=[make_flag("critical", "TEST_FLAG")]),
        ])
        summary = generate_summary(result)
        assert "verification" in summary.lower() or "concerns" in summary.lower()

    def test_critical_risk_summary(self):
        # Two criticals in ONE module would be group-capped (largest + 0.5*rest
        # = 4.5 nats -> score ~38, HIGH not CRITICAL). Put them in TWO SEPARATE
        # modules instead so each contributes its full 3.0 nats uncapped:
        # -4.0 + 3.0 + 3.0 = 2.0 logit -> score ~12 (CRITICAL/SUSPICIOUS).
        result = create_analysis_result("abc", [
            make_result(module="metadata", score=0, flags=[make_flag("critical", "CRIT_1")]),
            make_result(module="content", score=0, flags=[make_flag("critical", "CRIT_2")]),
        ])
        summary = generate_summary(result)
        assert "not trust" in summary.lower() or "fraud" in summary.lower()

    def test_unverifiable_summary_is_not_mislabeled_medium(self):
        # An extraction failure (encrypted/unreadable PDF) fails closed to
        # score=0, band/risk_level="UNVERIFIABLE". This is a SEPARATE state
        # from fraud: the summary must say "could not be verified", not fall
        # through to the MEDIUM ("some concerns") verdict.
        result = create_analysis_result("h", [
            make_result(module="extraction", score=0, confidence=1.0,
                        flags=[make_flag("high", "EXTRACTION_FAILED")]),
        ])
        assert result.risk_level == "UNVERIFIABLE"

        summary = generate_summary(result)
        summary_lower = summary.lower()
        assert (
            "could not be read" in summary_lower
            or "not be read" in summary_lower
            or "verified" in summary_lower
        )
        assert "some concerns" not in summary_lower

    def test_summary_includes_flag_descriptions(self):
        """Summary should describe the issues found (rich summary format)."""
        result = create_analysis_result("abc", [
            make_result(
                score=60,
                flags=[
                    make_flag("high", code="META_ONLINE_CONVERTER",
                              message="Document was processed by iLovePDF"),
                    make_flag("medium", code="CONTENT_ANACHRONISM_SERVICE",
                              message="Service date after invoice date"),
                ],
            ),
        ])
        summary = generate_summary(result)
        # Rich summary should describe the findings, not just list counts
        assert "converter" in summary.lower()
        assert "date" in summary.lower()


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


def test_scanner_pipeline_identity_stays_genuine():
    # Authentic scanned ID: scanner saved, viewer reopened ~6 min later. The
    # known-FP META_DOCUMENT_MODIFIED critical must be downgraded to negligible
    # evidence, so the document stays GENUINE rather than being flagged for review.
    from src.models import ModuleResult, Flag
    r = ModuleResult(module="metadata", score=0, confidence=1.0, flags=[
        Flag("critical", "META_DOCUMENT_MODIFIED", "reopened",
             details={"time_difference_hours": 0.1})])
    res = create_analysis_result("h", [r], document_type="identity")
    assert res.band == "GENUINE" and res.trust_score >= 80


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
