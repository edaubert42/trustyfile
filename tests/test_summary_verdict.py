"""
Regression tests for the summary verdict logic (whole-branch review, fix item 1).

Bug: `_get_dominant_theme` summed severity weights over ALL flags in a verdict
theme, including GENUINE-direction codes (ARITH_XML_CONSISTENT,
TWOD_DOC_CONSISTENT) that `_build_bullets` already excludes via
`src.evidence.GENUINE_CODES`. A positive "the e-invoice XML matches the
visible document" consistency flag could therefore make "tampering" win the
verdict headline, producing a false "this document may have been edited"
verdict even when there is zero real tamper evidence.
"""
from src.models import Flag, ModuleResult
from src.scoring import create_analysis_result
from src.summary import _get_dominant_theme, generate_rich_summary


class TestGetDominantThemeExcludesGenuineCodes:
    """Unit-level pin on the fixed function, independent of scoring internals."""

    def test_genuine_only_theme_is_not_dominant(self):
        """A theme whose ONLY occupant is a GENUINE_CODES flag must never be
        selected as dominant — there is nothing tamper-related to report."""
        grouped = {
            "tampering": [Flag(severity="medium", code="ARITH_XML_CONSISTENT",
                                message="genuine consistency signal")],
        }
        assert _get_dominant_theme(grouped) is None

    def test_genuine_theme_does_not_outrank_real_tamper_theme(self):
        """Even when the genuine-direction flag has higher raw severity than a
        real tamper flag living in another theme, the genuine one must not win."""
        grouped = {
            "tampering": [Flag(severity="critical", code="ARITH_XML_CONSISTENT",
                                message="genuine consistency signal")],
            "dates": [Flag(severity="low", code="CONTENT_VERY_OLD_DATE",
                            message="an old date")],
        }
        assert _get_dominant_theme(grouped) == "dates"

    def test_real_tamper_flag_in_same_theme_still_counts(self):
        """The fix must not blind the function to genuine tamper evidence that
        happens to share a theme with a genuine-direction flag."""
        grouped = {
            "tampering": [
                Flag(severity="medium", code="ARITH_XML_CONSISTENT", message="genuine"),
                Flag(severity="high", code="STRUCT_DELETED_OBJECTS", message="ghost data"),
            ],
        }
        assert _get_dominant_theme(grouped) == "tampering"


class TestVerdictNotDrivenByGenuineFlag:
    """End-to-end: a document whose only verdict-theme flag is the genuine
    ARITH_XML_CONSISTENT signal, plus enough non-verdict-theme noise to pull
    the risk level out of LOW (where no theme-specific verdict exists to
    expose the bug), must not receive a tampering-flavored verdict."""

    @staticmethod
    def _build_result():
        module_results = [
            ModuleResult(
                module="arithmetic",
                flags=[Flag(severity="medium", code="ARITH_XML_CONSISTENT",
                            message="The embedded e-invoice data matches the "
                                     "visible document (grand total verified) "
                                     "— machine-verified integrity evidence.")],
                score=100,
                confidence=1.0,
            ),
            # Non-verdict-theme noise ("images"/"fonts" are not in
            # _get_dominant_theme's verdict_themes set) that pulls the risk
            # level out of LOW without ever competing for the verdict theme.
            ModuleResult(
                module="images",
                flags=[Flag(severity="high", code="IMAGES_HEAVY_COMPRESSION",
                            message="heavily compressed image")],
                score=70,
                confidence=1.0,
            ),
            ModuleResult(
                module="fonts",
                flags=[Flag(severity="high", code="FONTS_SYSTEM_FONTS",
                            message="system fonts used")],
                score=70,
                confidence=1.0,
            ),
        ]
        return create_analysis_result(
            file_hash="sha256:test", module_results=module_results,
            document_type="invoice",
        )

    def test_risk_level_is_not_low(self):
        """Sanity check on the fixture itself: LOW has no theme-specific
        verdict, which would mask the bug regardless of dominant_theme."""
        result = self._build_result()
        assert result.risk_level in ("MEDIUM", "HIGH", "CRITICAL")

    def test_verdict_not_tampering_flavored(self):
        result = self._build_result()
        summary = generate_rich_summary(result)
        lowered = summary.verdict.lower()
        assert "edited" not in lowered
        assert "altered" not in lowered
        assert "modification" not in lowered
