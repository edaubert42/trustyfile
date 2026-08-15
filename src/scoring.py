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
    coverage_blind: bool = False,
    hard_detector_blinded: bool = False,
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
        coverage_blind: True when the document has no usable text layer (see analyzer._coverage_blind); GENUINE/REVIEW results collapse to UNVERIFIABLE.
        hard_detector_blinded: True when a module able to produce HARD tamper
            evidence crashed, so the strongest available proof was never looked
            for; denies the GENUINE band.

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
    # negligible evidence instead of a critical-weight tamper signal.
    #
    # Gated on the identity module having actually READ AN MRZ, not on the
    # document_type label alone. document_type is decided by keyword hits in
    # attacker-supplied text (src/document_type.py), and this downgrade sits
    # OUTSIDE DOC_TYPE_CODE_MULTIPLIERS — so before this gate, pasting a few ID
    # keywords into an invoice bought the exemption outright: the same ledger
    # scored 38/SUSPICIOUS as "invoice" and 91/GENUINE as "identity". That
    # falsified the invariant asserted at the top of this module. An MRZ that
    # parses is a structural property of a real identity document, not a word
    # an attacker can sprinkle into a body of text.
    #
    # The downgrade builds a NEW Flag rather than mutating in place: the caller's
    # module_results are returned verbatim in AnalysisResult.modules and rendered
    # by the per-module breakdown, so mutating here would permanently hide the
    # module's real finding from the audit trail the reader most needs on an
    # identity document — and would double-apply if the same objects were scored
    # twice.
    mrz_found = any(
        r.module == "identity" and (r.details or {}).get("mrz_found")
        for r in module_results
    )
    downgrade_scanner_fp = document_type == "identity" and mrz_found

    # Build the evidence ledger. A module that found nothing (no flags AND
    # score >= 100) ABSTAINS: it contributes no evidence rather than nudging
    # trust upward. Modules score from their FLAGS, not their numeric score.
    evidence = []
    for r in module_results:
        if not r.flags and r.score >= 100:
            continue
        for f in r.flags:
            # The coverage flag's entire scoring role is the band override
            # (coverage_blind param); it is a display/summary marker, not
            # tamper evidence. Keeping it out of the ledger stops its 0.2-nat
            # weight from nudging the band or the genuine-laundering threshold.
            if f.code == "COVERAGE_NO_TEXT_LAYER":
                continue
            # MODULE_ERROR is pipeline STATUS, not evidence about the document:
            # it says our analysis failed, not that the file is 0.2 nats more
            # tampered. Same reasoning as COVERAGE_NO_TEXT_LAYER above. The
            # real consequence of a crash is handled as a band cap
            # (hard_detector_blinded), not as a token tamper contribution.
            if f.code == "MODULE_ERROR":
                continue
            if downgrade_scanner_fp and _is_scanner_pipeline_critical(f):
                f = replace(f, severity="low")
            ev = flag_to_evidence(f, r.module)
            ev = apply_doc_type_suppression(ev, document_type)
            if ev is not None:
                evidence.append(ev)

    agg = aggregate_evidence(evidence, analyzable=analyzable,
                             coverage_blind=coverage_blind,
                             hard_detector_blinded=hard_detector_blinded)

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
