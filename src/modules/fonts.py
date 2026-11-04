"""
Module D: Font Analysis

This module analyzes fonts used in the PDF to detect signs of editing.

What we check:
1. Font diversity - Too many different fonts suggests amateur editing
2. System fonts - Arial/Calibri in a supposedly professional invoice
3. Font consistency - Same text styled with different fonts
4. Subset fonts - Font names like ABCDEF+Arial indicate embedded subsets

Why fonts matter for fraud detection:
- Professional invoices use consistent, branded fonts
- When someone edits a PDF, they often use available system fonts
- Font mismatches reveal where text was added or modified
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass
import fitz  # PyMuPDF

from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData

logger = logging.getLogger(__name__)


# =============================================================================
# FONT DATA STRUCTURES
# =============================================================================

@dataclass
class FontInfo:
    """
    Information about a font used in the document.

    Attributes:
        name: Font name (e.g., "Arial", "BCDFGH+Helvetica")
        base_name: Font name without subset prefix (e.g., "Helvetica")
        is_subset: Whether font is embedded as subset (has prefix like ABCDEF+)
        is_embedded: Whether font is embedded in the PDF
        pages_used: List of page numbers where this font appears
        usage_count: How many times this font is used
    """
    name: str
    base_name: str
    is_subset: bool = False
    is_embedded: bool = False
    pages_used: list[int] = None
    usage_count: int = 0

    def __post_init__(self):
        if self.pages_used is None:
            self.pages_used = []


# =============================================================================
# FONT EXTRACTION
# =============================================================================

def extract_base_font_name(font_name: str) -> tuple[str, bool]:
    """
    Extract the base font name, removing subset prefix if present.

    Subset fonts have a prefix like "ABCDEF+" before the font name.
    This happens when only some characters of a font are embedded.

    Args:
        font_name: Full font name (e.g., "BCDFGH+Helvetica-Bold")

    Returns:
        Tuple of (base_name, is_subset)

    Example:
        >>> extract_base_font_name("BCDFGH+Helvetica-Bold")
        ("Helvetica-Bold", True)
        >>> extract_base_font_name("Arial")
        ("Arial", False)
    """
    # Subset pattern: 6 uppercase letters followed by +
    subset_pattern = r"^[A-Z]{6}\+"

    if re.match(subset_pattern, font_name):
        base_name = font_name[7:]  # Remove "ABCDEF+"
        return base_name, True
    else:
        return font_name, False


def extract_fonts_from_pdf(pdf_path: str) -> list[FontInfo]:
    """
    Extract all fonts used in a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of FontInfo objects for each unique font
    """
    fonts_dict = {}  # font_name -> FontInfo

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF: {e}")
        return []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get fonts used on this page
            # Returns list of tuples: (xref, ext, type, basefont, name, encoding, referencer)
            font_list = page.get_fonts(full=True)

            for font_data in font_list:
                xref, ext, font_type, basefont, name, encoding, referencer = font_data

                # Use basefont or name as the font identifier
                font_name = basefont or name or "Unknown"

                # Extract base name and check if subset
                base_name, is_subset = extract_base_font_name(font_name)

                # Check if font is embedded
                # Embedded fonts usually have ext != "" or are subsets
                is_embedded = bool(ext) or is_subset

                # Add or update font info
                if font_name not in fonts_dict:
                    fonts_dict[font_name] = FontInfo(
                        name=font_name,
                        base_name=base_name,
                        is_subset=is_subset,
                        is_embedded=is_embedded,
                        pages_used=[page_num],
                        usage_count=1,
                    )
                else:
                    if page_num not in fonts_dict[font_name].pages_used:
                        fonts_dict[font_name].pages_used.append(page_num)
                    fonts_dict[font_name].usage_count += 1
    finally:
        # Ensure the doc is closed even if an exception occurs mid-loop
        doc.close()
    return list(fonts_dict.values())


# =============================================================================
# FONT ANALYSIS CHECKS
# =============================================================================

# Common system fonts that are suspicious in professional invoices
# Professional documents usually use branded/custom fonts
SYSTEM_FONTS = [
    "arial",
    "calibri",
    "times new roman",
    "times",
    "comic sans",
    "courier new",
    "courier",
    "verdana",
    "tahoma",
    "trebuchet",
    "georgia",
    "palatino",
    "cambria",
    "consolas",
    "lucida",
    "segoe",
]

# Fonts typically used by professional tools (not suspicious)
PROFESSIONAL_FONTS = [
    "helvetica",  # Adobe standard
    "myriad",     # Adobe
    "minion",     # Adobe
    "frutiger",   # Professional
    "univers",    # Professional
    "futura",     # Professional
    "gotham",     # Professional
    "avenir",     # Professional
    "roboto",     # Google (modern)
    "open sans",  # Google (modern)
    "lato",       # Google (modern)
    "source sans", # Adobe open source
]


def check_font_diversity(fonts: list[FontInfo]) -> list[Flag]:
    """
    Check if the document uses too many different fonts.

    Professional invoices typically use 1-3 fonts (heading, body, maybe monospace).
    More than 5 different font families is suspicious.

    Args:
        fonts: List of fonts found in document

    Returns:
        List of flags for font diversity issues
    """
    flags = []

    # Count unique base font families (ignore Bold/Italic variants)
    font_families = set()
    for font in fonts:
        # Extract family name (remove -Bold, -Italic, etc.)
        family = re.split(r"[-,]", font.base_name)[0].strip()
        font_families.add(family.lower())

    num_families = len(font_families)

    # Complex invoices can legitimately use many fonts (headers, body, tables,
    # legal text, barcodes, price emphasis). Thresholds raised to reduce
    # false positives on real invoices.
    if num_families > 10:
        flags.append(Flag(
            severity="high",
            code="FONTS_EXCESSIVE_DIVERSITY",
            message=f"Document uses {num_families} different font families (suspicious for an invoice)",
            details={
                "font_count": num_families,
                "fonts": list(font_families),
            }
        ))
    elif num_families > 7:
        flags.append(Flag(
            severity="medium",
            code="FONTS_HIGH_DIVERSITY",
            message=f"Document uses {num_families} different font families",
            details={
                "font_count": num_families,
                "fonts": list(font_families),
            }
        ))

    return flags


def check_system_fonts(fonts: list[FontInfo]) -> list[Flag]:
    """
    Check for system fonts that are unusual in professional invoices.

    If a document claims to be from a large company (EDF, Amazon, etc.),
    it shouldn't use basic system fonts like Arial or Calibri.

    Args:
        fonts: List of fonts found in document

    Returns:
        List of flags for suspicious system font usage
    """
    flags = []

    system_fonts_found = []

    for font in fonts:
        base_lower = font.base_name.lower()

        # Check against system fonts
        for sys_font in SYSTEM_FONTS:
            if sys_font in base_lower:
                # Don't flag if it's also a professional font
                is_professional = any(
                    prof in base_lower for prof in PROFESSIONAL_FONTS
                )
                if not is_professional:
                    system_fonts_found.append(font.base_name)
                break

    if system_fonts_found:
        # Only flag as low severity - system fonts aren't always suspicious
        flags.append(Flag(
            severity="low",
            code="FONTS_SYSTEM_FONTS",
            message=f"Document uses common system fonts: {', '.join(set(system_fonts_found))}",
            details={
                "system_fonts": list(set(system_fonts_found)),
            }
        ))

    return flags


def check_font_embedding(fonts: list[FontInfo]) -> list[Flag]:
    """
    Check for font embedding issues.

    Professional PDFs usually embed their fonts (or use standard PDF fonts).
    Non-embedded fonts might display differently on different systems.

    Args:
        fonts: List of fonts found in document

    Returns:
        List of flags for embedding issues
    """
    flags = []

    # Standard PDF fonts that don't need embedding
    standard_fonts = [
        "helvetica", "times", "courier", "symbol", "zapfdingbats"
    ]

    non_embedded = []
    for font in fonts:
        if not font.is_embedded:
            base_lower = font.base_name.lower()
            # Skip standard PDF fonts
            if not any(std in base_lower for std in standard_fonts):
                non_embedded.append(font.base_name)

    if non_embedded:
        flags.append(Flag(
            severity="low",
            code="FONTS_NOT_EMBEDDED",
            message=f"Some fonts are not embedded: {', '.join(non_embedded)}",
            details={
                "non_embedded_fonts": non_embedded,
            }
        ))

    return flags


def check_mixed_subset_fonts(fonts: list[FontInfo]) -> list[Flag]:
    """
    Check for mixed subset and non-subset versions of the same font.

    If the same font appears both as a subset (ABCDEF+Arial) and non-subset (Arial),
    it might indicate the document was edited with different tools.

    Args:
        fonts: List of fonts found in document

    Returns:
        List of flags for mixed subset issues
    """
    flags = []

    # Group fonts by base name
    base_name_groups = {}
    for font in fonts:
        base = font.base_name.lower()
        if base not in base_name_groups:
            base_name_groups[base] = []
        base_name_groups[base].append(font)

    # Check for mixed subset/non-subset
    for base_name, font_group in base_name_groups.items():
        has_subset = any(f.is_subset for f in font_group)
        has_non_subset = any(not f.is_subset for f in font_group)

        if has_subset and has_non_subset:
            flags.append(Flag(
                severity="low",
                code="FONTS_MIXED_SUBSETS",
                message=f"Font '{base_name}' appears both as subset and non-subset (possible editing)",
                details={
                    "font_name": base_name,
                    "variants": [f.name for f in font_group],
                }
            ))

    return flags


def _is_glyph_fallback(text: str) -> bool:
    """True if this span is a lone symbol the body font could not render.

    A renderer that meets a character missing from the current font substitutes a
    symbol-capable one for that character alone. The result is a one-character
    span in a different family — indistinguishable, by family alone, from retyped
    text. It is not evidence of editing: it is how every PDF generator draws a
    bullet in a font that has no bullet glyph.

    Observed on the known-good i2d_free_fiber.pdf, where the three "suspicious"
    lines were "• Téléphone :", "• Courrier :" and "• Internet :" — in each case
    the only differing span was the leading "•" in DejaVuSans, because the body
    font NimbusSanL-Regu has no bullet. Three lines is also the threshold that
    escalates this flag to HIGH, so one systemic fallback pattern was worth
    1.5 nats against a legitimate invoice.

    Kept deliberately narrow — a SINGLE non-alphanumeric character. A retyped
    value is digits or letters and is never caught by this.

    Examples:
        >>> _is_glyph_fallback("•")
        True
        >>> _is_glyph_fallback("1350")   # a retyped amount
        False
    """
    stripped = text.strip()
    return len(stripped) == 1 and not stripped.isalnum()


# Font family = base name without style suffix (-Bold, -Italic, etc.) and
# without subset prefix (ABCDEF+). Common aliases are normalised
# (ArialMT = Arial, etc.) so that one family under two names is not a "change".
FAMILY_ALIASES = {
    "arialmt": "arial",
    "timesnewroman": "times",
    "times": "times",
    "couriernew": "courier",
    "couriermt": "courier",
    "helveticaneue": "helvetica",
}


# PostScript-name vendor suffixes, not part of the family: Monotype's "MT"
# and Microsoft's "PS"/"PSMT". The regular face carries them glued to the
# family ("CourierNewPSMT"), the bold one splits them around the style
# ("CourierNewPS-BoldMT"): without stripping, one family read as two.
_PS_SUFFIX_RE = re.compile(r"(?:psmt|ps|mt)$")


def _font_family(font_name: str) -> str:
    """Font family, ignoring style, subset prefix, vendor suffix and aliases.

    Examples:
        >>> _font_family("ABCDEF+Helvetica-Bold")
        'helvetica'
        >>> _font_family("ArialMT")
        'arial'
        >>> _font_family("CourierNewPSMT") == _font_family("CourierNewPS-BoldMT")
        True
        >>> _font_family("TimesNewRomanPS-BoldMT")
        'times'
        >>> _font_family("CIDFont+F1")      # auto-generated name, not a family
        '_cidfont'
    """
    base, _ = extract_base_font_name(font_name)            # drop "ABCDEF+"
    family = re.split(r"[-,]", base)[0].strip().lower()     # drop "-Bold", ",Italic"
    # Generic CID font names (CIDFont+F1, ...) are auto-generated and don't
    # represent real font families.
    if family.startswith("cidfont"):
        return "_cidfont"
    stripped = _PS_SUFFIX_RE.sub("", family).strip()
    if len(stripped) >= 4:                 # never shrink a short name to nothing
        family = stripped
    return FAMILY_ALIASES.get(family, family)


# Two spans share a baseline when their origins' y differ by less than this
# fraction of the font size. Measured on the real-tool corpus: a year retyped
# by an editor sits 0.04-0.37 pt off the original baseline (fonts 10-12 pt),
# while a superscript ("1er", "m²") sits 2-4 pt higher.
MIDWORD_BASELINE_TOL_FRAC = 0.12

# Maximum horizontal gap between the two halves of one "word", as a fraction
# of the font size. A normal space is ~0.25-0.30 em wide: anything below
# 0.10 em means the glyphs touch — the two spans are one word on screen.
MIDWORD_MAX_GAP_FRAC = 0.10


def _is_lone_accented_letter(text: str) -> bool:
    """True for a single non-ASCII letter ("é", "ç", "ü").

    A generator whose font lacks an accented glyph draws that one letter in a
    fallback font: "Factur" + "é" is then a mid-word font change with nothing
    edited. Same reasoning as _is_glyph_fallback, for letters.

    Examples:
        >>> _is_lone_accented_letter("é")
        True
        >>> _is_lone_accented_letter("4")      # a retyped digit is NOT excused
        False
    """
    stripped = text.strip()
    return len(stripped) == 1 and stripped.isalpha() and not stripped.isascii()


def _junction_splits_number(left: str, right: str) -> bool:
    """True when the font change cuts through a number or a date.

    Every font has the ten digits, so a digit drawn in another font is never
    a glyph fallback: it was put there by someone. That is why such a split
    weighs more than a letter split (see check_midline_font_changes).

    Examples:
        >>> _junction_splits_number("19/08/", "1998")   # retyped birth year
        True
        >>> _junction_splits_number("1 2", "50,00")
        True
        >>> _junction_splits_number("EASY", "MONNERET")  # letters: no
        False
    """
    # No stripping: a space at the junction means two words, not a split one
    # ("Total : " + "1998" is a label followed by a value).
    if not left or not right:
        return False
    if right[0].isdigit():
        return left[-1].isdigit() or left[-1] in "/.,-"
    # "17" + ".01": the cut falls just before the decimal separator.
    return _is_number_separator_junction(left, right)


def _is_number_separator_junction(left: str, right: str) -> bool:
    """True for "17" + ".01" / "260" + ",01": a digit, then a separator + digit.

    The most realistic partial retype changes only the cents, so the new
    span starts with the decimal separator. Seen on every real invoice
    retyped this way (ZUGFeRD corpus); the year-only synthetic retype never
    produced it.
    """
    return (len(right) >= 2 and right[0] in ".," and right[1].isdigit()
            and bool(left) and left[-1].isdigit())


def _continues_word(left: str, right: str) -> bool:
    """Does `right` continue the word `left` ends with (no break between them)?

    Yes when `right` starts with a letter or digit ("19/08/" + "1998"), or
    with a decimal separator between digits ("17" + ".01").
    """
    return right[0].isalnum() or _is_number_separator_junction(left, right)


def _midword_font_changes(text_dict: dict, page_label: int) -> list[dict]:
    """Font family changes INSIDE a word, across PyMuPDF blocks.

    Why a separate pass: when an editor retypes "1992" as "1998", the new
    digits are a new text object, which PyMuPDF puts in its own block. The
    per-line check only compares spans of one line of one block, so it never
    sees "Date de naissance : 19/08/" (block 3) next to "1998" (block 8).

    Why only INSIDE a word, not anywhere on the line: a form filled by
    software legitimately prints the label in one font and the value in
    another ("Nom : " + "Jean Dupont") — but always with a space between
    them. A font change with NO space, between two characters of the same
    token, is how a partial retype looks; honest generators don't do it.

    A pair of spans is reported when ALL of these hold:
      - they are in different blocks or lines (same line = per-line check's job);
      - same baseline (|Δ origin y| ≤ MIDWORD_BASELINE_TOL_FRAC × size);
      - the right one starts where the left one ends
        (0 - small overlap ≤ gap ≤ MIDWORD_MAX_GAP_FRAC × size);
      - no whitespace at the junction, and the right one starts with a letter
        or digit (the left may end with "/", "." etc.: "19/08/" + "1998");
      - different real families (no CID names, no lone fallback glyphs).

    Args:
        text_dict: `page.get_text("dict")` of one page.
        page_label: 1-based page number for the report.

    Returns:
        Entries shaped like the per-line check's `suspicious_lines` items.

    Example:
        "Date de naissance : 19/08/" (Helvetica-Bold, x1=196.3)
        + "1998" (Times-Roman, x0=196.3, baseline 0.04 pt off)
        -> [{"page": 1, "text": "Date de naissance : 19/08/1998", ...}]
    """
    spans = []
    for block in text_dict.get("blocks", []):
        for line_no, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                spans.append({"key": (block.get("number"), line_no), "span": span})

    # Sort by baseline so each span is only compared with the few spans whose
    # baseline is close — not with every span of the page (a dense page has
    # thousands; all pairs would be millions of comparisons).
    spans.sort(key=lambda s: s["span"]["origin"][1])
    max_size = max((s["span"].get("size", 0) for s in spans), default=0)
    window = MIDWORD_BASELINE_TOL_FRAC * max(max_size, 1.0)

    candidate_pairs = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if b["span"]["origin"][1] - a["span"]["origin"][1] > window:
                break                                     # sorted: the rest are lower
            candidate_pairs.append((a, b))
            candidate_pairs.append((b, a))                # either may be on the left

    found = []
    for left, right in candidate_pairs:
        ls, rs = left["span"], right["span"]
        if right["key"] == left["key"]:
            continue                                      # same line: other check
        size = max(ls.get("size", 0), rs.get("size", 0), 1.0)
        if abs(ls["origin"][1] - rs["origin"][1]) > MIDWORD_BASELINE_TOL_FRAC * size:
            continue                                      # not the same baseline
        gap = rs["bbox"][0] - ls["bbox"][2]
        if not (-0.5 <= gap <= MIDWORD_MAX_GAP_FRAC * size):
            continue                                  # not touching
        l_text, r_text = ls["text"], rs["text"]
        if l_text[-1].isspace() or not _continues_word(l_text, r_text):
            continue                                  # a space = two words
        if _is_glyph_fallback(l_text) or _is_glyph_fallback(r_text):
            continue
        if _is_lone_accented_letter(l_text) or _is_lone_accented_letter(r_text):
            continue                                  # "Factur" + "é": accent fallback
        l_fam, r_fam = _font_family(ls.get("font", "")), _font_family(rs.get("font", ""))
        if "_cidfont" in (l_fam, r_fam) or l_fam == r_fam:
            continue
        found.append({
            "page": page_label,
            "text": (l_text.strip() + r_text.strip())[:100],
            "fonts": sorted({ls.get("font", ""), rs.get("font", "")}),
            "families": sorted({l_fam, r_fam}),
            "splits_number": _junction_splits_number(l_text, r_text),
        })
    return found


def check_midline_font_changes(pdf_path: str) -> list[Flag]:
    """
    Detect font changes within the same line of text.

    When someone edits a PDF (e.g., changes a date or amount), they often
    use a slightly different font than the original. This function detects
    lines where the font family changes mid-line.

    How it works:
    1. Extract all text spans with their font and position
    2. Group spans by line (same y-coordinate within tolerance)
    3. For each line, check if the font family changes between spans
    4. Flag lines where a different font family appears (not just Bold/Italic)

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of flags for mid-line font switches

    Example:
        A line like "Date de naissance : 19/08/2001" where
        "Date de naissance : 19/08/" is in NimbusSans-Bold and
        "2001" is in NimbusSanL-Bol → flagged as suspicious
    """
    flags = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF for font analysis: {e}")
        return []

    suspicious_lines = []
    midword_hits = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue  # Skip image blocks

                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if len(spans) < 2:
                        continue  # Need at least 2 spans to compare

                    # Get families for all spans that have actual text
                    span_families = []
                    for span in spans:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        font = span.get("font", "")
                        family = _font_family(font)
                        span_families.append({
                            "text": text,
                            "font": font,
                            "family": family,
                        })

                    if len(span_families) < 2:
                        continue

                    # Check if font family changes within this line
                    # Exclude generic CID font names — they're not real families,
                    # and lone symbol glyphs — see _is_glyph_fallback.
                    families_on_line = set(
                        s["family"] for s in span_families
                        if s["family"] != "_cidfont"
                        and not _is_glyph_fallback(s["text"])
                    )

                    if len(families_on_line) > 1:
                        # Font family changes mid-line — suspicious!
                        # Reconstruct the line text for the message
                        line_text = " ".join(s["text"] for s in span_families)
                        fonts_used = list(set(s["font"] for s in span_families))

                        suspicious_lines.append({
                            "page": page_num + 1,
                            "text": line_text[:100],
                            "fonts": fonts_used,
                            "families": list(families_on_line),
                        })

            # Second pass: a font change in the middle of a word whose two
            # halves PyMuPDF put in different blocks (retyped text is a new
            # text object, so the per-line loop above never sees both halves).
            midword_hits.extend(_midword_font_changes(text_dict, page_num + 1))
    finally:
        # Ensure the doc is closed even if an exception occurs mid-loop
        doc.close()

    if suspicious_lines:
        # Severity depends on how many lines are affected
        if len(suspicious_lines) >= 3:
            severity = "high"
        else:
            severity = "medium"

        # Build a clear message showing the most suspicious line
        first = suspicious_lines[0]
        message = (
            f"Font changes mid-line on page {first['page']}: "
            f"'{first['text'][:60]}' uses {len(first['fonts'])} different fonts"
        )

        flags.append(Flag(
            severity=severity,
            code="FONTS_MIDLINE_CHANGE",
            message=message,
            details={
                "suspicious_lines": suspicious_lines[:5],  # Limit to 5
                "total_suspicious_lines": len(suspicious_lines),
                "explanation": "Different font families on the same line suggest "
                              "text was edited after the document was created.",
            }
        ))

    if midword_hits:
        # Its own code, and at least "high" from the first hit: a font change between
        # two characters of ONE word, across two text objects, is how a partial
        # retype looks — far more specific than "two fonts somewhere on a line"
        # (which a label/value layout produces honestly). Still SOFT: never a
        # HARD veto on its own.
        # "critical" when the cut goes through a number/date: digits exist in
        # every font, so no fallback excuse exists (_junction_splits_number).
        numeric = [h for h in midword_hits if h["splits_number"]]
        first = (numeric or midword_hits)[0]
        flags.append(Flag(
            severity="critical" if numeric else "high",
            code="FONTS_MIDWORD_CHANGE",
            message=(
                f"Font changes in the middle of a word on page {first['page']}: "
                f"'{first['text'][:60]}' mixes {' and '.join(first['families'])}"
            ),
            details={
                "suspicious_lines": midword_hits[:5],
                "total_suspicious_lines": len(midword_hits),
                "explanation": "Part of a word is drawn in another font, as a "
                               "separate text object: the typical trace of "
                               "characters retyped over the original.",
            }
        ))

    return flags


# =============================================================================
# SEVERITY POINTS
# =============================================================================

SEVERITY_POINTS = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "critical": 50,
}


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_fonts(pdf_data: PDFData) -> ModuleResult:
    """
    Analyze fonts used in the PDF for signs of manipulation.

    Args:
        pdf_data: Extracted PDF data

    Returns:
        ModuleResult with score, flags, and confidence
    """
    all_flags = []

    # Extract fonts
    fonts = extract_fonts_from_pdf(pdf_data.file_path)

    if not fonts:
        # No fonts found - might be image-only PDF
        return ModuleResult(
            module="fonts",
            flags=[],
            score=100,
            confidence=0.3,  # Low confidence - couldn't analyze
        )

    # Run checks
    all_flags.extend(check_font_diversity(fonts))
    all_flags.extend(check_system_fonts(fonts))
    all_flags.extend(check_font_embedding(fonts))
    all_flags.extend(check_mixed_subset_fonts(fonts))
    all_flags.extend(check_midline_font_changes(pdf_data.file_path))

    # Calculate score
    score = 100
    for flag in all_flags:
        score -= SEVERITY_POINTS[flag.severity]
    score = max(0, score)

    # Confidence based on number of fonts analyzed
    if len(fonts) >= 3:
        confidence = 0.9
    elif len(fonts) >= 1:
        confidence = 0.7
    else:
        confidence = 0.3

    return ModuleResult(
        module="fonts",
        flags=all_flags,
        score=score,
        confidence=confidence,
    )
