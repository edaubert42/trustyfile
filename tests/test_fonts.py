"""
Tests for Module D: Font Analysis.

This module checks fonts for signs of PDF editing:
- Font diversity (too many = amateur editing)
- System fonts in professional invoices
- Font embedding issues
- Mixed subset/non-subset fonts (sign of multi-tool editing)

Most checks take FontInfo objects, so we don't need real PDFs.
"""

import pytest
from src.models import Flag
from src.modules.fonts import (
    FontInfo,
    extract_base_font_name,
    check_font_diversity,
    check_system_fonts,
    check_font_embedding,
    check_mixed_subset_fonts,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_font(
    name: str,
    base_name: str | None = None,
    is_subset: bool = False,
    is_embedded: bool = True,
) -> FontInfo:
    """Shortcut to create a FontInfo for testing."""
    if base_name is None:
        base_name = name
    return FontInfo(
        name=name,
        base_name=base_name,
        is_subset=is_subset,
        is_embedded=is_embedded,
        pages_used=[0],
        usage_count=1,
    )


# =============================================================================
# TEST extract_base_font_name
# =============================================================================

class TestExtractBaseFontName:
    """
    Subset fonts have a 6-letter uppercase prefix followed by '+'.
    Example: "BCDFGH+Helvetica" → ("Helvetica", True)
    """

    def test_subset_font(self):
        base, is_subset = extract_base_font_name("BCDFGH+Helvetica-Bold")
        assert base == "Helvetica-Bold"
        assert is_subset is True

    def test_non_subset_font(self):
        base, is_subset = extract_base_font_name("Arial")
        assert base == "Arial"
        assert is_subset is False

    def test_another_subset(self):
        base, is_subset = extract_base_font_name("ABCDEF+TimesNewRoman")
        assert base == "TimesNewRoman"
        assert is_subset is True

    def test_lowercase_not_subset(self):
        """Subset prefix must be uppercase — lowercase should not match."""
        base, is_subset = extract_base_font_name("abcdef+Arial")
        assert is_subset is False

    def test_short_prefix_not_subset(self):
        """Prefix must be exactly 6 letters."""
        base, is_subset = extract_base_font_name("ABC+Arial")
        assert is_subset is False

    def test_empty_string(self):
        base, is_subset = extract_base_font_name("")
        assert base == ""
        assert is_subset is False


# =============================================================================
# TEST check_font_diversity
# =============================================================================

class TestCheckFontDiversity:
    """
    Professional invoices use 1-3 font families.
    > 5 → medium, > 7 → high severity.
    """

    def test_few_fonts_no_flag(self):
        """3 fonts is normal for an invoice."""
        fonts = [
            make_font("Helvetica"),
            make_font("Helvetica-Bold"),
            make_font("Courier"),
        ]
        flags = check_font_diversity(fonts)
        assert len(flags) == 0

    def test_six_fonts_no_flag(self):
        """6 font families → no flag (threshold raised to >7)."""
        fonts = [
            make_font("Helvetica"),
            make_font("Arial"),
            make_font("Courier"),
            make_font("Verdana"),
            make_font("Georgia"),
            make_font("Tahoma"),
        ]
        flags = check_font_diversity(fonts)
        assert len(flags) == 0

    def test_eight_fonts_medium_flag(self):
        """8 font families → medium flag (threshold raised)."""
        fonts = [
            make_font("Helvetica"),
            make_font("Arial"),
            make_font("Courier"),
            make_font("Verdana"),
            make_font("Georgia"),
            make_font("Tahoma"),
            make_font("Palatino"),
            make_font("Futura"),
        ]
        flags = check_font_diversity(fonts)
        assert len(flags) == 1
        assert flags[0].severity == "medium"
        assert flags[0].code == "FONTS_HIGH_DIVERSITY"

    def test_eleven_fonts_high_flag(self):
        """11 font families → high flag."""
        fonts = [make_font(f"Font{i}") for i in range(11)]
        flags = check_font_diversity(fonts)
        assert len(flags) == 1
        assert flags[0].severity == "high"
        assert flags[0].code == "FONTS_EXCESSIVE_DIVERSITY"

    def test_bold_italic_same_family(self):
        """Bold/Italic variants of the same font count as one family."""
        fonts = [
            make_font("Helvetica"),
            make_font("Helvetica-Bold"),
            make_font("Helvetica-Italic"),
            make_font("Helvetica-BoldItalic"),
        ]
        flags = check_font_diversity(fonts)
        # All are "Helvetica" family → 1 family, no flag
        assert len(flags) == 0

    def test_empty_list_no_flag(self):
        assert check_font_diversity([]) == []


# =============================================================================
# TEST check_system_fonts
# =============================================================================

class TestCheckSystemFonts:
    """
    System fonts (Arial, Calibri, etc.) are suspicious in professional invoices.
    Professional companies use branded/custom fonts.
    """

    @pytest.mark.parametrize("font_name", [
        "Arial", "Calibri", "Times New Roman", "Comic Sans MS",
        "Verdana", "Tahoma", "Courier New",
    ])
    def test_system_fonts_flagged(self, font_name):
        fonts = [make_font(font_name, base_name=font_name)]
        flags = check_system_fonts(fonts)
        assert len(flags) == 1
        assert flags[0].severity == "low"
        assert flags[0].code == "FONTS_SYSTEM_FONTS"

    @pytest.mark.parametrize("font_name", [
        "Helvetica", "Helvetica-Bold", "Myriad-Pro", "Roboto",
    ])
    def test_professional_fonts_not_flagged(self, font_name):
        fonts = [make_font(font_name, base_name=font_name)]
        flags = check_system_fonts(fonts)
        assert len(flags) == 0

    def test_multiple_system_fonts_one_flag(self):
        """Multiple system fonts should produce a single flag listing them all."""
        fonts = [
            make_font("Arial", base_name="Arial"),
            make_font("Calibri", base_name="Calibri"),
        ]
        flags = check_system_fonts(fonts)
        assert len(flags) == 1  # One flag, not two

    def test_empty_list(self):
        assert check_system_fonts([]) == []


# =============================================================================
# TEST check_font_embedding
# =============================================================================

class TestCheckFontEmbedding:
    """
    Non-embedded fonts (except standard PDF fonts) are suspicious.
    """

    def test_all_embedded_no_flag(self):
        fonts = [
            make_font("CustomFont", is_embedded=True),
            make_font("AnotherFont", is_embedded=True),
        ]
        flags = check_font_embedding(fonts)
        assert len(flags) == 0

    def test_non_embedded_flagged(self):
        fonts = [make_font("MyCustomFont", is_embedded=False)]
        flags = check_font_embedding(fonts)
        assert len(flags) == 1
        assert flags[0].code == "FONTS_NOT_EMBEDDED"
        assert flags[0].severity == "low"

    def test_standard_fonts_not_flagged(self):
        """Standard PDF fonts (Helvetica, Times, Courier) don't need embedding."""
        fonts = [
            make_font("Helvetica", base_name="Helvetica", is_embedded=False),
            make_font("Times-Roman", base_name="Times-Roman", is_embedded=False),
            make_font("Courier", base_name="Courier", is_embedded=False),
        ]
        flags = check_font_embedding(fonts)
        assert len(flags) == 0


# =============================================================================
# TEST check_mixed_subset_fonts
# =============================================================================

class TestCheckMixedSubsetFonts:
    """
    Same font appearing as both subset (ABCDEF+Arial) and non-subset (Arial)
    suggests the document was edited with different tools.
    """

    def test_mixed_subset_flagged(self):
        """Same base name, one subset, one not → low flag (normal PDF behavior)."""
        fonts = [
            make_font("BCDFGH+Arial", base_name="Arial", is_subset=True),
            make_font("Arial", base_name="Arial", is_subset=False),
        ]
        # Both have base_name "Arial" but lowercase matching
        flags = check_mixed_subset_fonts(fonts)
        assert len(flags) == 1
        assert flags[0].code == "FONTS_MIXED_SUBSETS"
        assert flags[0].severity == "low"

    def test_all_subset_no_flag(self):
        """All subsets of the same font → normal, no flag."""
        fonts = [
            make_font("ABCDEF+Helvetica", base_name="Helvetica", is_subset=True),
            make_font("GHIJKL+Helvetica", base_name="Helvetica", is_subset=True),
        ]
        # Need to make sure base_name matches (lowercase)
        # Both are subset=True → no mixed issue
        flags = check_mixed_subset_fonts(fonts)
        assert len(flags) == 0

    def test_different_fonts_no_flag(self):
        """Different fonts, one subset one not → no flag (different fonts)."""
        fonts = [
            make_font("ABCDEF+Helvetica", base_name="Helvetica", is_subset=True),
            make_font("Arial", base_name="Arial", is_subset=False),
        ]
        flags = check_mixed_subset_fonts(fonts)
        assert len(flags) == 0

    def test_case_insensitive(self):
        """Base name matching should be case-insensitive."""
        fonts = [
            make_font("ABCDEF+arial", base_name="arial", is_subset=True),
            make_font("Arial", base_name="Arial", is_subset=False),
        ]
        flags = check_mixed_subset_fonts(fonts)
        assert len(flags) == 1
