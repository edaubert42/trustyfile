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

                # Extract font family for each span
                # Font family = base name without style suffix (-Bold, -Italic, etc.)
                # and without subset prefix (ABCDEF+)
                # We also normalize common aliases (ArialMT = Arial, etc.)
                FAMILY_ALIASES = {
                    "arialmt": "arial",
                    "timesnewroman": "times",
                    "times": "times",
                    "couriernew": "courier",
                    "couriermt": "courier",
                    "helveticaneue": "helvetica",
                }

                def get_family(font_name: str) -> str:
                    """Extract font family, ignoring style, subset prefix, and aliases."""
                    # Remove subset prefix
                    base, _ = extract_base_font_name(font_name)
                    # Remove common style suffixes
                    family = re.split(r"[-,]", base)[0].strip().lower()
                    # Skip generic CID font names (CIDFont+F1, etc.)
                    # These are auto-generated and don't represent real font families
                    if family.startswith("cidfont"):
                        return "_cidfont"
                    # Normalize known aliases
                    return FAMILY_ALIASES.get(family, family)

                # Get families for all spans that have actual text
                span_families = []
                for span in spans:
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font = span.get("font", "")
                    family = get_family(font)
                    span_families.append({
                        "text": text,
                        "font": font,
                        "family": family,
                    })

                if len(span_families) < 2:
                    continue

                # Check if font family changes within this line
                # Exclude generic CID font names — they're not real families
                families_on_line = set(
                    s["family"] for s in span_families
                    if s["family"] != "_cidfont"
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
