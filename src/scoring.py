"""
Scoring System - Combines all module results into a final trust score.

`create_analysis_result` (the entry point) builds an evidence ledger from
module FLAGS and aggregates it in log-odds space — see src/evidence.py for the
math. Each flag becomes an Evidence item (`flag_to_evidence`), optionally
down-weighted per document type (`apply_doc_type_suppression`), and the
ledger is combined by `aggregate_evidence` to produce the final score.

Risk levels:
- LOW (80-100): Document appears legitimate
- MEDIUM (50-79): Some concerns, manual verification recommended
- HIGH (20-49): Multiple red flags, likely manipulated
- CRITICAL (0-19): Strong evidence of fraud
"""

from dataclasses import replace
from typing import Literal
from src.models import ModuleResult, AnalysisResult, Flag, DocumentType


# =============================================================================
# PER-(DOCUMENT TYPE, FLAG CODE) SOFT-WEIGHT MULTIPLIERS
# =============================================================================

# Per-(document_type, flag_code) soft-weight multipliers. Only FP-prone,
# genre-specific codes appear here. Hard evidence and universally-suspicious
# codes are NEVER in this table and always pass at full weight — so an attacker
# cannot disable tamper detection by steering the (text-based) classifier.
DOC_TYPE_CODE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "scientific": {"FONTS_HIGH_DIVERSITY": 0.15, "FONTS_EXCESSIVE_DIVERSITY": 0.15,
                   "STRUCT_INCREMENTAL_UPDATES": 0.3, "IMAGES_SCREENSHOT_DETECTED": 0.4},
    "technical": {"STRUCT_JAVASCRIPT_DETECTED": 1.0,  # explicit: NOT suppressed even here
                  "STRUCT_ACROFORM_DETECTED": 0.3, "FONTS_HIGH_DIVERSITY": 0.4},
    "legal_notice": {"FONTS_HIGH_DIVERSITY": 0.4, "STRUCT_INCREMENTAL_UPDATES": 0.4},
    "identity": {"FONTS_HIGH_DIVERSITY": 0.3, "STRUCT_ACROFORM_DETECTED": 0.5},
}


def apply_doc_type_suppression(evidence, document_type: str):
    """Down-weight genre-specific FP-prone SOFT codes for a document type.
    Hard evidence is never touched. Returns Evidence, a down-weighted copy, or
    None (fully suppressed, multiplier 0)."""
    if evidence.kind == "hard":
        return evidence
    mult = DOC_TYPE_CODE_MULTIPLIERS.get(document_type, {}).get(evidence.code, 1.0)
    if mult >= 1.0:
        return evidence
    if mult <= 0.0:
        return None
    return replace(evidence, weight=evidence.weight * mult)


# =============================================================================
# MODULE WEIGHTS
# =============================================================================

# Weights determine how much each module contributes to the final score
# Higher weight = more influence on the final result
MODULE_WEIGHTS = {
    "metadata": 1.0,    # PDF metadata analysis
    "content": 1.2,     # Text content & dates (most important for invoices)
    "visual": 0.8,      # QR codes & watermarks
    "fonts": 0.9,       # Font consistency
    "images": 0.8,      # Embedded images
    "forensics": 0.9,   # Image forensics (ELA)
    "structure": 1.3,   # PDF structure (incremental updates, JS, etc.) - very reliable
    "arithmetic": 1.2,  # totals-block identity + e-invoice XML cross-check (deterministic)
    "external": 1.5,    # External verification (very reliable when available)
    "identity": 1.5,   # MRZ check digit validation (cryptographically reliable)
}

# Default weight for unknown modules
DEFAULT_WEIGHT = 1.0


# =============================================================================
# RISK LEVEL THRESHOLDS
# =============================================================================

def get_risk_level(score: int) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """
    Convert a numeric score to a risk level category.

    Args:
        score: Trust score (0-100)

    Returns:
        Risk level string

    Thresholds:
        80-100: LOW - Document appears legitimate
        50-79:  MEDIUM - Some concerns
        20-49:  HIGH - Multiple red flags
        0-19:   CRITICAL - Strong evidence of fraud
    """
    if score >= 80:
        return "LOW"
    elif score >= 50:
        return "MEDIUM"
    elif score >= 20:
        return "HIGH"
    else:
        return "CRITICAL"


def collect_all_flags(module_results: list[ModuleResult]) -> list[Flag]:
    """
    Collect all flags from all modules, sorted by severity.

    Args:
        module_results: List of results from all analysis modules

    Returns:
        List of all flags, sorted by severity (critical first)
    """
    all_flags = []

    for result in module_results:
        for flag in result.flags:
            # Add module name to flag for context
            all_flags.append(flag)

    # Sort by severity (critical > high > medium > low)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_flags.sort(key=lambda f: severity_order.get(f.severity, 4))

    return all_flags


def count_flags_by_severity(flags: list[Flag]) -> dict[str, int]:
    """
    Count flags by severity level.

    Args:
        flags: List of all flags

    Returns:
        Dict mapping severity to count
    """
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for flag in flags:
        if flag.severity in counts:
            counts[flag.severity] += 1

    return counts


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def _is_scanner_pipeline_critical(flag: Flag) -> bool:
    """True if this flag is the known scanner→viewer false positive.

    Authentic scanned identity documents are saved by the scanner, then opened
    minutes later by a viewer (Adobe, Preview…), which stamps a new ModDate and
    trips META_DOCUMENT_MODIFIED. A gap under 30 minutes is a plausible
    scan-then-view workflow, not tampering. (Keys on code + gap, not severity —
    the 2026-07 judge panel re-graded this code from a flat critical to
    medium/high, and the identity-doc exemption must survive that.)
    """
    if flag.code != "META_DOCUMENT_MODIFIED" or flag.severity == "low":
        return False
    hours = (flag.details or {}).get("time_difference_hours", 999)
    try:
        return float(hours) < 0.5  # under 30 minutes
    except (TypeError, ValueError):
        return False  # unparseable → treat as genuine tamper evidence


def create_analysis_result(
    file_hash: str,
    module_results: list[ModuleResult],
    analysis_time_ms: int = 0,
    document_type: DocumentType = "unknown",
) -> AnalysisResult:
    """
    Create the final analysis result from all module results.

    Builds an evidence ledger from the flags raised by each module and
    aggregates it in log-odds space (see src/evidence.py) instead of taking a
    confidence-weighted average of module scores. Modules score from their
    FLAGS, not from their numeric `score` — the numeric score is only used to
    detect an abstaining clean module (no flags AND score >= 100), which
    contributes no evidence rather than nudging trust upward.

    Args:
        file_hash: SHA256 hash of the analyzed file
        module_results: List of results from all analysis modules
        analysis_time_ms: Time taken for analysis in milliseconds
        document_type: Detected document type (used for evidence suppression)

    Returns:
        AnalysisResult with final score, risk level, band, score range,
        evidence ledger, document type, and module details.
    """
    from src.evidence import flag_to_evidence, aggregate_evidence

    # Fail-closed: an explicit extraction failure means the document could not be
    # read, so it is UNVERIFIABLE — never a trusting high score.
    analyzable = not any(
        f.code == "EXTRACTION_FAILED" for r in module_results for f in r.flags
    )

    # Known false positive on authentic SCANNED identity documents: downgrade the
    # scanner→viewer META_DOCUMENT_MODIFIED critical to "low" so it contributes
    # negligible evidence instead of a critical-weight tamper signal. (Preserved
    # from the pre-ledger engine; the in-place mutation is a known smell to revisit
    # when detail-aware suppression lands in a later phase.)
    if document_type == "identity":
        for r in module_results:
            for f in r.flags:
                if _is_scanner_pipeline_critical(f):
                    f.severity = "low"

    # Build the evidence ledger. A module that found nothing (no flags AND
    # score >= 100) ABSTAINS: it contributes no evidence rather than nudging
    # trust upward. Modules score from their FLAGS, not their numeric score.
    evidence = []
    for r in module_results:
        if not r.flags and r.score >= 100:
            continue
        for f in r.flags:
            ev = flag_to_evidence(f, r.module)
            ev = apply_doc_type_suppression(ev, document_type)
            if ev is not None:
                evidence.append(ev)

    agg = aggregate_evidence(evidence, analyzable=analyzable)

    return AnalysisResult(
        file_hash=file_hash,
        trust_score=agg.score,
        risk_level=agg.risk_level,
        document_type=document_type,
        modules=module_results,
        analysis_time_ms=analysis_time_ms,
        band=agg.band,
        score_low=agg.score_low,
        score_high=agg.score_high,
        evidence_ledger=agg.ledger,
    )


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary(result: AnalysisResult) -> str:
    """
    Generate a human-readable summary of the analysis.

    Delegates to the rich summary generator and returns the combined
    verdict + explanation as a single string for backward compatibility.

    Args:
        result: The complete analysis result

    Returns:
        Summary string suitable for display
    """
    from src.summary import generate_rich_summary

    rich = generate_rich_summary(result)
    return f"{rich.verdict} {' '.join(rich.bullets)}"
