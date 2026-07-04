"""
Tests for the scoring module.

The scoring system combines results from all analysis modules into a final
trust score (0-100) and risk level (LOW/MEDIUM/HIGH/CRITICAL).

We test:
1. Risk level thresholds (score → risk level)
2. Weighted score calculation (the math)
3. Flag collection and sorting
4. Critical flag overrides (business rules)
5. Summary generation
"""

import pytest
from src.models import Flag, ModuleResult, AnalysisResult
from src.scoring import (
    get_risk_level,
    calculate_final_score,
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
# TEST calculate_final_score
# =============================================================================

class TestCalculateFinalScore:
    """
    Tests for the weighted average calculation.

    Formula: Σ(score × weight × confidence) / Σ(weight × confidence)
    """

    def test_empty_list_returns_100(self):
        """No modules analyzed = assume document is innocent."""
        assert calculate_final_score([]) == 100

    def test_single_module_full_confidence(self):
        """With one module at full confidence, the score IS the module score."""
        results = [make_result(module="metadata", score=75, confidence=1.0)]
        assert calculate_final_score(results) == 75

    def test_single_module_zero_confidence(self):
        """A module with zero confidence should be ignored entirely."""
        results = [make_result(module="metadata", score=0, confidence=0.0)]
        # Zero confidence = no data, so we assume innocent
        assert calculate_final_score(results) == 100

    def test_two_modules_equal_weight(self):
        """
        Two modules with weight 1.0 and full confidence = simple average.

        metadata has weight 1.0, fonts has weight 0.9.
        So the result is NOT a simple average — it's weighted.
        Let's use metadata (1.0) and "unknown" (1.0 default) to get equal weights.
        """
        results = [
            make_result(module="metadata", score=80, confidence=1.0),   # weight 1.0
            make_result(module="unknown_module", score=60, confidence=1.0),  # weight 1.0 (default)
        ]
        # (80×1.0 + 60×1.0) / (1.0 + 1.0) = 140/2 = 70
        assert calculate_final_score(results) == 70

    def test_weights_affect_result(self):
        """
        Modules with higher weights pull the score more toward their value.

        content has weight 1.2, visual has weight 0.8.
        If content scores low and visual scores high,
        the result should lean toward content's score.
        """
        results = [
            make_result(module="content", score=40, confidence=1.0),   # weight 1.2
            make_result(module="visual", score=100, confidence=1.0),   # weight 0.8
        ]
        # (40×1.2 + 100×0.8) / (1.2 + 0.8) = (48 + 80) / 2.0 = 64
        assert calculate_final_score(results) == 64

    def test_confidence_affects_result(self):
        """
        A module with low confidence should have less influence.

        If metadata (score=30, confidence=0.1) and content (score=90, confidence=1.0),
        the result should be close to 90 because metadata is barely trusted.
        """
        results = [
            make_result(module="metadata", score=30, confidence=0.1),  # weight 1.0
            make_result(module="content", score=90, confidence=1.0),   # weight 1.2
        ]
        # (30×1.0×0.1 + 90×1.2×1.0) / (1.0×0.1 + 1.2×1.0)
        # = (3 + 108) / (0.1 + 1.2) = 111 / 1.3 ≈ 85.38 → 85
        assert calculate_final_score(results) == 85

    def test_score_clamped_to_0_100(self):
        """Score should never go below 0 or above 100."""
        # Score of 100 with full weight should stay at 100
        results = [make_result(score=100, confidence=1.0)]
        assert calculate_final_score(results) <= 100

        # Score of 0 should stay at 0
        results = [make_result(score=0, confidence=1.0)]
        assert calculate_final_score(results) >= 0

    def test_all_modules_perfect(self):
        """All modules scoring 100 should give final score of 100."""
        results = [
            make_result(module="metadata", score=100),
            make_result(module="content", score=100),
            make_result(module="visual", score=100),
            make_result(module="fonts", score=100),
            make_result(module="images", score=100),
        ]
        assert calculate_final_score(results) == 100

    def test_all_modules_zero(self):
        """All modules scoring 0 should give final score of 0."""
        results = [
            make_result(module="metadata", score=0),
            make_result(module="content", score=0),
            make_result(module="visual", score=0),
            make_result(module="fonts", score=0),
            make_result(module="images", score=0),
        ]
        assert calculate_final_score(results) == 0


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
    Tests for the final result builder.

    The key business rules:
    - 1 critical flag → risk level forced to at least HIGH, score capped at 49
    - 2+ critical flags → risk level forced to CRITICAL, score capped at 19
    """

    def test_clean_document(self):
        """A document with no flags should get 100/LOW."""
        results = [
            make_result(module="metadata", score=100),
            make_result(module="content", score=100),
        ]
        analysis = create_analysis_result("abc123", results)
        assert analysis.trust_score == 100
        assert analysis.risk_level == "LOW"

    def test_one_critical_flag_caps_score(self):
        """
        Even if the weighted average gives a high score,
        one critical flag should cap it at 49 (HIGH).

        This prevents: "metadata found AI-generated content (critical),
        but everything else looks fine, so overall score is 85 (LOW)."
        That would be dangerous — one critical finding should NOT be diluted.
        """
        results = [
            make_result(
                module="metadata",
                score=50,
                flags=[make_flag("critical", "META_AI_GENERATED")],
            ),
            make_result(module="content", score=100),
            make_result(module="visual", score=100),
            make_result(module="fonts", score=100),
        ]
        analysis = create_analysis_result("abc123", results)
        # Score should be capped at 49 regardless of weighted average
        assert analysis.trust_score <= 49
        # Risk level should be at least HIGH
        assert analysis.risk_level in ("HIGH", "CRITICAL")

    def test_two_critical_flags_force_critical(self):
        """Two critical flags = CRITICAL risk, score capped at 19."""
        results = [
            make_result(
                module="metadata",
                score=60,
                flags=[make_flag("critical", "CRIT_1")],
            ),
            make_result(
                module="content",
                score=60,
                flags=[make_flag("critical", "CRIT_2")],
            ),
        ]
        analysis = create_analysis_result("abc123", results)
        assert analysis.trust_score <= 19
        assert analysis.risk_level == "CRITICAL"

    def test_non_critical_flags_dont_override(self):
        """High/medium/low flags reduce score but don't trigger the override."""
        results = [
            make_result(
                module="metadata",
                score=70,
                flags=[make_flag("high"), make_flag("medium"), make_flag("low")],
            ),
        ]
        analysis = create_analysis_result("abc123", results)
        # Score should be the actual calculated value, not overridden
        assert analysis.trust_score == 70
        assert analysis.risk_level == "MEDIUM"

    def test_file_hash_preserved(self):
        """The file hash should be passed through to the result."""
        analysis = create_analysis_result("sha256:deadbeef", [])
        assert analysis.file_hash == "sha256:deadbeef"

    def test_analysis_time_preserved(self):
        """Analysis time should be passed through to the result."""
        analysis = create_analysis_result("abc", [], analysis_time_ms=1234)
        assert analysis.analysis_time_ms == 1234


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
        result = create_analysis_result("abc", [
            make_result(score=65),
        ])
        summary = generate_summary(result)
        assert "verification" in summary.lower() or "concerns" in summary.lower()

    def test_critical_risk_summary(self):
        result = create_analysis_result("abc", [
            make_result(
                score=0,
                flags=[make_flag("critical"), make_flag("critical")],
            ),
        ])
        summary = generate_summary(result)
        assert "not trust" in summary.lower() or "fraud" in summary.lower()

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
