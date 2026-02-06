"""
Rich Template-Based Summary Generator for TrustyFile.

Groups flags by user-facing themes (not modules) and produces a natural-feeling
two-part summary: a short verdict + a multi-sentence explanation.

All template-based — no LLM, free, instant, works offline.
"""

from src.models import AnalysisResult, AnalysisSummary, ModuleResult, Flag
from src.scoring import collect_all_flags


# =============================================================================
# THEME DEFINITIONS
# =============================================================================

# Each flag code is mapped to a user-facing theme.
# Themes group related findings so the summary reads naturally.
FLAG_THEME_MAP: dict[str, str] = {
    # origin: online converters, AI tools, suspicious software
    "META_AI_GENERATED": "origin",
    "META_ONLINE_CONVERTER": "origin",
    "META_SUSPICIOUS_PRODUCER": "origin",
    "META_NO_METADATA": "origin",
    "META_NO_PRODUCER": "origin",

    # tampering: incremental edits, paste artifacts, font mixing, deleted objects
    "STRUCT_INCREMENTAL_UPDATES": "tampering",
    "STRUCT_DELETED_OBJECTS": "tampering",
    "FONTS_EXCESSIVE_DIVERSITY": "tampering",
    "FONTS_HIGH_DIVERSITY": "tampering",
    "FONTS_MIXED_SUBSETS": "tampering",
    "FONTS_MIDLINE_CHANGE": "tampering",
    "IMAGES_PASTE_NOISE_ANOMALY": "tampering",

    # tampering: modification time gap, impossible dates in metadata
    "META_DOCUMENT_MODIFIED": "tampering",
    "META_FUTURE_CREATION_DATE": "tampering",
    "META_IMPOSSIBLE_DATES": "tampering",

    # dates: content-level date inconsistencies (anachronisms, future dates in text)
    "CONTENT_FAR_FUTURE_DATE": "dates",
    "CONTENT_VERY_OLD_DATE": "dates",
    "CONTENT_ANACHRONISM_SERVICE": "dates",
    "CONTENT_ANACHRONISM_DUE": "dates",
    "CONTENT_ANACHRONISM_ORDER": "dates",
    "CONTENT_FUTURE_INVOICE_DATE": "dates",

    # identity: invalid SIRET/SIREN/VAT, company mismatch
    "CONTENT_INVALID_SIRET": "identity",
    "CONTENT_INVALID_SIREN": "identity",
    "CONTENT_INVALID_VAT": "identity",
    "CONTENT_SIREN_VAT_MISMATCH": "identity",
    "CONTENT_MISSING_COMPANY_ID": "identity",
    "CONTENT_INCONSISTENT_REFERENCES": "identity",
    "CONTENT_REFERENCE_DATE_MISMATCH": "identity",
    "EXTERNAL_SIRET_NOT_FOUND": "identity",
    "EXTERNAL_SIREN_NOT_FOUND": "identity",
    "EXTERNAL_COMPANY_CLOSED": "identity",
    "EXTERNAL_COMPANY_NAME_MISMATCH": "identity",
    "EXTERNAL_VAT_INVALID": "identity",
    "EXTERNAL_SIRET_VERIFICATION_FAILED": "identity",
    "EXTERNAL_SIREN_VERIFICATION_FAILED": "identity",
    "EXTERNAL_VAT_VERIFICATION_FAILED": "identity",

    # visual: QR mismatches, watermarks
    "VISUAL_QR_URL_SHORTENER": "visual",
    "VISUAL_QR_SUSPICIOUS_TLD": "visual",
    "VISUAL_QR_DOMAIN_MISMATCH": "visual",
    "VISUAL_WATERMARK_SPECIMEN": "visual",
    "VISUAL_WATERMARK_COPY": "visual",
    "VISUAL_WATERMARK_DRAFT": "visual",
    "VISUAL_WATERMARK_DUPLICATE": "visual",
    "VISUAL_WATERMARK_VOID": "visual",
    "VISUAL_WATERMARK_CANCELLED": "visual",
    "VISUAL_WATERMARK_NOT_VALID": "visual",
    "VISUAL_WATERMARK_SAMPLE": "visual",
    "VISUAL_WATERMARK_TEST": "visual",
    "VISUAL_WATERMARK_CONFIDENTIAL": "visual",
    "VISUAL_CONVERTER_WATERMARK": "visual",

    # images: screenshots, image-only PDFs, compression issues
    "IMAGES_SCREENSHOT_DETECTED": "images",
    "IMAGES_RESOLUTION_MISMATCH": "images",
    "IMAGES_HEAVY_COMPRESSION": "images",
    "IMAGES_EXCESSIVE_COUNT": "images",
    "IMAGES_NO_IMAGES": "images",
    "IMAGES_IMAGE_ONLY_PDF": "images",
    "IMAGES_MOSTLY_IMAGE_PDF": "images",

    # security: JavaScript, embedded files, hidden annotations
    "STRUCT_JAVASCRIPT_DETECTED": "security",
    "STRUCT_HIDDEN_ANNOTATIONS": "security",
    "STRUCT_SUSPICIOUS_ANNOTATIONS": "security",
    "STRUCT_EMBEDDED_FILES": "security",
    "STRUCT_ACROFORM_DETECTED": "security",
    "STRUCT_XMP_EDITOR_MISMATCH": "tampering",

    # fonts (standalone, not tampering-level)
    "FONTS_SYSTEM_FONTS": "fonts",
    "FONTS_NOT_EMBEDDED": "fonts",

    # signature (positive — handled separately in positive signals)
    "STRUCT_SIGNATURE_TRUSTED": "signature",
    "STRUCT_SIGNATURE_TRUSTED_EXPIRED": "signature",
    "STRUCT_SIGNATURE_NOT_TRUSTED": "signature",
    "STRUCT_SIGNATURE_UNVERIFIABLE": "signature",
    "STRUCT_SIGNATURE_INVALID": "signature",
}


# =============================================================================
# SENTENCE TEMPLATES
# =============================================================================

# Each flag code maps to a function that takes a Flag and returns a sentence.
# Using lambdas for simple cases, regular functions for complex ones.
# If a flag code is missing here, we fall back to flag.message.

def _get_converter_name(flag: Flag) -> str:
    """Extract the converter/tool name from flag details."""
    if flag.details:
        return flag.details.get("detected_tool", flag.details.get("producer", "an unknown tool"))
    return "an unknown tool"


SENTENCE_TEMPLATES: dict[str, callable] = {
    # --- origin ---
    "META_AI_GENERATED": lambda f: (
        f"The document was generated by {_get_converter_name(f)}, an AI tool."
    ),
    "META_ONLINE_CONVERTER": lambda f: (
        f"The document was processed through {_get_converter_name(f)}, "
        f"an online converter commonly used to modify documents."
    ),
    "META_SUSPICIOUS_PRODUCER": lambda f: (
        f"The document was created with {_get_converter_name(f)}, "
        f"which is unusual for a professional invoice."
    ),
    "META_NO_METADATA": lambda f: (
        "The document metadata has been completely stripped, "
        "which may indicate an attempt to hide its origin."
    ),
    "META_NO_PRODUCER": lambda f: (
        "No software information is present in the document metadata."
    ),

    # --- tampering ---
    "STRUCT_INCREMENTAL_UPDATES": lambda f: (
        f"The PDF has been edited {f.details.get('edit_count', 'multiple')} "
        f"time(s) after its initial creation."
        if f.details else "The PDF has been edited after its initial creation."
    ),
    "STRUCT_DELETED_OBJECTS": lambda f: (
        "The PDF contains deleted objects (ghost data), suggesting content "
        "was removed but traces remain."
    ),
    "FONTS_EXCESSIVE_DIVERSITY": lambda f: (
        f"The document uses {f.details.get('font_count', 'many')} different fonts, "
        f"which is unusual for a professional invoice."
        if f.details else "The document uses an excessive number of fonts."
    ),
    "FONTS_HIGH_DIVERSITY": lambda f: (
        f"The document uses {f.details.get('font_count', 'several')} different fonts, "
        f"more than typical for a professional document."
        if f.details else "The document uses more fonts than typical."
    ),
    "FONTS_MIXED_SUBSETS": lambda f: (
        f"The font \"{f.details.get('font_name', 'unknown')}\" appears in both "
        f"subset and full forms, which can indicate text editing."
        if f.details else "A font appears in mixed forms, suggesting editing."
    ),
    "FONTS_MIDLINE_CHANGE": lambda f: (
        f"A font change was detected mid-line, suggesting text was edited after creation."
    ),
    "IMAGES_PASTE_NOISE_ANOMALY": lambda f: (
        "A region of the document shows abnormal noise patterns consistent "
        "with a copy-paste operation."
    ),

    # --- dates ---
    "META_FUTURE_CREATION_DATE": lambda f: (
        "The PDF creation date is set in the future, which is impossible "
        "for a legitimate document."
    ),
    "META_DOCUMENT_MODIFIED": lambda f: (
        f"The document was modified "
        f"{int(f.details['time_difference_hours'] // 24)} days after creation."
        if f.details and f.details.get('time_difference_hours', 0) >= 24
        else f"The document was modified "
        f"{round(f.details['time_difference_hours'], 1)}h after creation."
        if f.details and f.details.get('time_difference_hours')
        else "The document was modified after its initial creation."
    ),
    "META_IMPOSSIBLE_DATES": lambda f: (
        "The modification date is before the creation date, "
        "which indicates metadata tampering."
    ),
    "CONTENT_FAR_FUTURE_DATE": lambda f: (
        f"A date far in the future was found"
        f" ({f.details.get('date', '')})"
        f"{' near: ' + f.details.get('context', '') if f.details and f.details.get('context') else ''}."
        if f.details else "A date far in the future was found in the document."
    ),
    "CONTENT_VERY_OLD_DATE": lambda f: (
        "A suspiciously old date (before 2000) was found in the document."
    ),
    "CONTENT_ANACHRONISM_SERVICE": lambda f: (
        "The service date comes after the invoice date, which is inconsistent."
    ),
    "CONTENT_ANACHRONISM_DUE": lambda f: (
        "The due date is before the invoice date, which is inconsistent."
    ),
    "CONTENT_ANACHRONISM_ORDER": lambda f: (
        "The order date comes after the invoice date, which is inconsistent."
    ),
    "CONTENT_FUTURE_INVOICE_DATE": lambda f: (
        "The invoice date is in the future."
    ),

    # --- identity ---
    "CONTENT_INVALID_SIRET": lambda f: (
        f"The SIRET number ({f.details.get('siret', 'unknown')}) has an invalid checksum."
        if f.details else "A SIRET number with invalid checksum was found."
    ),
    "CONTENT_INVALID_SIREN": lambda f: (
        f"The SIREN number ({f.details.get('siren', 'unknown')}) has an invalid checksum."
        if f.details else "A SIREN number with invalid checksum was found."
    ),
    "CONTENT_INVALID_VAT": lambda f: (
        f"The VAT number ({f.details.get('vat', 'unknown')}) is invalid."
        if f.details else "An invalid VAT number was found."
    ),
    "CONTENT_SIREN_VAT_MISMATCH": lambda f: (
        "The SIREN number derived from the VAT number does not match "
        "the SIREN/SIRET found in the document."
    ),
    "CONTENT_MISSING_COMPANY_ID": lambda f: (
        "No SIRET, SIREN, or RCS number was found, which is required "
        "on French invoices."
    ),
    "CONTENT_INCONSISTENT_REFERENCES": lambda f: (
        f"The document contains {f.details.get('unique_references', 'multiple')} "
        f"different reference numbers, which is suspicious."
        if f.details else "Inconsistent reference numbers were found."
    ),
    "CONTENT_REFERENCE_DATE_MISMATCH": lambda f: (
        "The invoice reference number does not match the invoice date."
    ),
    "EXTERNAL_SIRET_NOT_FOUND": lambda f: (
        f"The SIRET number ({f.details.get('siret', 'unknown')}) was not found "
        f"in the official French registry."
        if f.details else "A SIRET number was not found in the official registry."
    ),
    "EXTERNAL_SIREN_NOT_FOUND": lambda f: (
        f"The SIREN number ({f.details.get('siren', 'unknown')}) was not found "
        f"in the official French registry."
        if f.details else "A SIREN number was not found in the official registry."
    ),
    "EXTERNAL_COMPANY_CLOSED": lambda f: (
        f"The company \"{f.details.get('company_name', 'unknown')}\" is registered "
        f"as closed/inactive."
        if f.details else "The company is registered as closed/inactive."
    ),
    "EXTERNAL_COMPANY_NAME_MISMATCH": lambda f: (
        f"The company name in the document does not match the official registry "
        f"(expected \"{f.details.get('name_in_registry', 'unknown')}\")."
        if f.details else "The company name does not match the official registry."
    ),
    "EXTERNAL_VAT_INVALID": lambda f: (
        f"The VAT number ({f.details.get('vat', 'unknown')}) is invalid "
        f"according to the EU VIES system."
        if f.details else "A VAT number is invalid according to EU VIES."
    ),

    # --- visual ---
    "VISUAL_QR_URL_SHORTENER": lambda f: (
        f"A QR code uses a URL shortener ({f.details.get('domain', 'unknown')}), "
        f"which can hide the real destination."
        if f.details else "A QR code uses a URL shortener."
    ),
    "VISUAL_QR_SUSPICIOUS_TLD": lambda f: (
        f"A QR code points to a suspicious domain "
        f"({f.details.get('domain', 'unknown')})."
        if f.details else "A QR code points to a suspicious domain."
    ),
    "VISUAL_QR_DOMAIN_MISMATCH": lambda f: (
        f"A QR code points to {f.details.get('qr_domain', 'an unexpected domain')}, "
        f"which doesn't match the expected domain(s)."
        if f.details else "A QR code domain doesn't match the expected domain."
    ),
    "VISUAL_CONVERTER_WATERMARK": lambda f: (
        f"A visible watermark from a converter/editor was detected "
        f"(\"{f.details.get('matched_text', '')}\")."
        if f.details and f.details.get('matched_text')
        else "A visible converter/editor watermark was detected."
    ),

    # images
    "IMAGES_SCREENSHOT_DETECTED": lambda f: (
        "An image with screen-like dimensions was detected, "
        "suggesting the document may be a screenshot."
    ),
    "IMAGES_RESOLUTION_MISMATCH": lambda f: (
        "Images in the document have inconsistent resolutions, "
        "which may indicate images from different sources."
    ),
    "IMAGES_HEAVY_COMPRESSION": lambda f: (
        "An image is heavily compressed, which can indicate multiple re-saves."
    ),
    "IMAGES_IMAGE_ONLY_PDF": lambda f: (
        "The document is image-only with no text layer, "
        "which is typical of screenshots or scanned modifications."
    ),
    "IMAGES_MOSTLY_IMAGE_PDF": lambda f: (
        "The document is mostly images with very little text."
    ),

    # security
    "STRUCT_JAVASCRIPT_DETECTED": lambda f: (
        "The PDF contains JavaScript code, which is suspicious "
        "and potentially dangerous in an invoice."
    ),
    "STRUCT_EMBEDDED_FILES": lambda f: (
        f"The PDF contains {f.details.get('file_count', '')} embedded file(s), "
        f"which could be malicious."
        if f.details else "The PDF contains embedded files."
    ),
    "STRUCT_HIDDEN_ANNOTATIONS": lambda f: (
        "Hidden annotations were found in the PDF."
    ),
    "STRUCT_SUSPICIOUS_ANNOTATIONS": lambda f: (
        "Suspicious annotation types were found in the PDF "
        "(file attachments, multimedia, etc.)."
    ),
    "STRUCT_XMP_EDITOR_MISMATCH": lambda f: (
        "The document was modified by a different tool than its creator."
    ),
}


# =============================================================================
# VERDICT TEMPLATES
# =============================================================================

# Verdicts are selected based on risk level.
# When tampering or dates are the dominant theme, we pick a more specific verdict.
VERDICTS: dict[str, dict[str, str]] = {
    "CRITICAL": {
        "default": "Do not trust this document.",
        "tampering": "Do not trust this document — it has been tampered with.",
        "dates": "Do not trust this document — dates indicate fabrication.",
        "identity": "Do not trust this document — identity information is fraudulent.",
        "security": "Do not trust this document — it contains dangerous content.",
    },
    "HIGH": {
        "default": "This document is likely fraudulent.",
        "tampering": "This document has been altered after creation.",
        "dates": "This document contains suspicious date inconsistencies.",
        "identity": "The originator of this document could not be verified.",
        "origin": "This document was created with suspicious tools.",
    },
    "MEDIUM": {
        "default": "This document has issues that need verification.",
        "tampering": "This document may have been edited.",
        "dates": "Dates in this document are inconsistent.",
        "origin": "The origin of this document raises questions.",
    },
    "LOW": {
        "default": "This document appears legitimate. No issues detected.",
    },
}


# =============================================================================
# POSITIVE SIGNAL TEMPLATES
# =============================================================================

# When a module has score >= 90 and no flags, we mention it as a positive signal.
# Maps module name -> positive sentence.
POSITIVE_SIGNALS: dict[str, str] = {
    "metadata": "The document metadata is consistent and clean.",
    "content": "All dates and references in the document are consistent.",
    "visual": "No suspicious visual elements were detected.",
    "fonts": "Font usage is consistent with a professional document.",
    "images": "Images show no signs of manipulation.",
    "structure": "The PDF structure is clean with no signs of editing.",
    "external": "Company identification numbers were verified successfully.",
}

# Special positive signal for trusted signatures (detected via flag code)
SIGNATURE_POSITIVE = "The document carries a valid EU-trusted digital signature."


# =============================================================================
# CORE LOGIC
# =============================================================================

def _group_flags_by_theme(flags: list[Flag]) -> dict[str, list[Flag]]:
    """
    Group flags by user-facing theme.

    Args:
        flags: All flags from all modules, sorted by severity.

    Returns:
        Dict mapping theme name to list of flags in that theme.
        Flags with unknown codes go into "other".
    """
    groups: dict[str, list[Flag]] = {}

    for flag in flags:
        theme = FLAG_THEME_MAP.get(flag.code, "other")
        if theme not in groups:
            groups[theme] = []
        groups[theme].append(flag)

    return groups


def _get_dominant_theme(grouped: dict[str, list[Flag]]) -> str | None:
    """
    Find the theme with the most severe flags.

    We score each theme by summing severity weights of its flags.
    The theme with the highest total wins.

    Args:
        grouped: Flags grouped by theme.

    Returns:
        The dominant theme name, or None if no flags.
    """
    severity_weight = {"critical": 50, "high": 30, "medium": 15, "low": 5}

    # Exclude themes that aren't used for verdict selection
    verdict_themes = {"origin", "tampering", "dates", "identity", "security"}

    best_theme = None
    best_score = 0

    for theme, flags in grouped.items():
        if theme not in verdict_themes:
            continue
        total = sum(severity_weight.get(f.severity, 0) for f in flags)
        if total > best_score:
            best_score = total
            best_theme = theme

    return best_theme


def _build_verdict(risk_level: str, dominant_theme: str | None) -> str:
    """
    Pick the most appropriate verdict based on risk level and dominant theme.

    Args:
        risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        dominant_theme: The most prominent theme, or None.

    Returns:
        A short verdict sentence.
    """
    level_verdicts = VERDICTS.get(risk_level, VERDICTS["MEDIUM"])

    # Try theme-specific verdict first, fall back to default
    if dominant_theme and dominant_theme in level_verdicts:
        return level_verdicts[dominant_theme]
    return level_verdicts["default"]


def _flag_to_sentence(flag: Flag) -> str:
    """
    Convert a single flag into a human-readable sentence.

    Uses the template if available, otherwise falls back to flag.message.

    Args:
        flag: A Flag object.

    Returns:
        A sentence describing the finding.
    """
    template_fn = SENTENCE_TEMPLATES.get(flag.code)
    if template_fn:
        try:
            return template_fn(flag)
        except (KeyError, TypeError, AttributeError):
            # If the template fails (missing details key, etc.), fall back
            pass
    # Fallback: use the flag's own message
    return flag.message


def _build_bullets(
    grouped: dict[str, list[Flag]],
    module_results: list[ModuleResult],
) -> list[str]:
    """
    Build a concise list of issues found.

    Only shows problems — no positive signals. Keeps it short:
    one bullet per theme, only the most severe flag.

    Args:
        grouped: Flags grouped by theme.
        module_results: All module results (unused, kept for API compat).

    Returns:
        List of sentences, each one becoming a bullet point in the UI.
        Empty list if no issues (the verdict already says "appears legitimate").
    """
    bullets: list[str] = []

    # Order themes by importance
    theme_order = [
        "tampering", "dates", "identity", "origin",
        "security", "visual", "images", "fonts", "other",
    ]

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    mentioned_codes: set[str] = set()

    for theme in theme_order:
        if theme not in grouped:
            continue

        flags = grouped[theme]

        # Skip signature theme (informational, not an issue)
        if theme == "signature":
            continue

        # Sort by severity, pick only the most severe flag per theme
        sorted_flags = sorted(flags, key=lambda f: severity_order.get(f.severity, 4))

        for flag in sorted_flags:
            if flag.code in mentioned_codes:
                continue
            # Skip low-severity flags — not worth a bullet
            if flag.severity == "low":
                continue

            sentence = _flag_to_sentence(flag)
            if sentence and sentence not in bullets:
                bullets.append(sentence)
                mentioned_codes.add(flag.code)
                if len([c for c in mentioned_codes if c in [f.code for f in flags]]) >= 2:
                    break  # Max 2 bullets per theme

    return bullets


def _get_positive_signals(
    grouped: dict[str, list[Flag]],
    module_results: list[ModuleResult],
) -> list[str]:
    """
    Generate positive sentences for modules that are clean.

    A module is "clean" if it has score >= 90 and no flags.
    We also check for trusted signatures as a special positive signal.

    Args:
        grouped: Flags grouped by theme.
        module_results: All module results.

    Returns:
        List of positive signal sentences.
    """
    positives: list[str] = []

    # Check for trusted digital signature (special case — it's a flag but positive)
    if "signature" in grouped:
        for flag in grouped["signature"]:
            if flag.code == "STRUCT_SIGNATURE_TRUSTED":
                positives.append(SIGNATURE_POSITIVE)
                break

    # Check clean modules
    # Only mention positive signals if there are also negative findings
    # (for a fully clean doc, the verdict already says "appears legitimate")
    has_negative_flags = any(
        theme not in ("signature", "fonts") and theme in grouped
        for theme in grouped
    )

    if not has_negative_flags:
        return positives

    # Pick up to 2 positive signals to keep it concise
    count = 0
    for module in module_results:
        if count >= 2:
            break
        if module.score >= 90 and len(module.flags) == 0:
            signal = POSITIVE_SIGNALS.get(module.module)
            if signal:
                positives.append("On the positive side, " + signal.lower()
                                 if count == 0 and positives == []
                                 else signal.capitalize() if not signal[0].isupper()
                                 else signal)
                count += 1

    return positives


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_rich_summary(result: AnalysisResult) -> AnalysisSummary:
    """
    Generate a rich, template-based summary of the analysis.

    This is the main entry point. It produces a verdict (short bold statement)
    and a list of bullet-point findings.

    Args:
        result: The complete analysis result from TrustyFileAnalyzer.

    Returns:
        AnalysisSummary with verdict and bullets.

    Example:
        >>> summary = generate_rich_summary(result)
        >>> print(summary.verdict)
        "Some aspects of this document require manual verification."
        >>> for b in summary.bullets:
        ...     print(f"  - {b}")
    """
    all_flags = collect_all_flags(result.modules)

    # Group flags by user-facing theme
    grouped = _group_flags_by_theme(all_flags)

    # Find the dominant theme for verdict selection
    dominant_theme = _get_dominant_theme(grouped)

    # Build verdict
    verdict = _build_verdict(result.risk_level, dominant_theme)

    # Build bullet list of findings
    bullets = _build_bullets(grouped, result.modules)

    # Clean document: no bullets needed, verdict says it all
    return AnalysisSummary(verdict=verdict, bullets=bullets)
