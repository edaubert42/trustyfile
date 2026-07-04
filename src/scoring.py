"""
Scoring System - Combines all module results into a final trust score.

`create_analysis_result` (the entry point) now builds an evidence ledger from
module FLAGS and aggregates it in log-odds space — see src/evidence.py for the
math. `calculate_final_score` (confidence-weighted average of module scores)
is RETAINED for backward compatibility / tests but no longer drives the trust
score.

Risk levels:
- LOW (80-100): Document appears legitimate
- MEDIUM (50-79): Some concerns, manual verification recommended
- HIGH (20-49): Multiple red flags, likely manipulated
- CRITICAL (0-19): Strong evidence of fraud
"""

from dataclasses import dataclass
from typing import Literal
from src.models import ModuleResult, AnalysisResult, Flag, DocumentType


def apply_doc_type_suppression(evidence, document_type):
    """Placeholder pass-through; real per-code suppression added in Task 7."""
    return evidence


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
    "external": 1.5,    # External verification (very reliable when available)
    "identity": 1.5,   # MRZ check digit validation (cryptographically reliable)
}

# Default weight for unknown modules
DEFAULT_WEIGHT = 1.0


# =============================================================================
# CONFIDENCE MULTIPLIERS BY DOCUMENT TYPE
# =============================================================================

# Some modules are calibrated for invoices and produce false positives on other
# document types. For example:
#   - fonts: a scientific paper legitimately uses many fonts (equations, figures)
#   - content: invoice-specific date/amount logic is irrelevant in a legal notice
#   - structure: papers often have many incremental revisions
#
# Instead of changing module scores (which would hide real findings), we reduce
# their effective confidence so they weigh less in the final average.
#
# A multiplier of 0.15 means the module still runs and reports flags,
# but its contribution to the final score is nearly silenced.
#
# Primary types (invoice, identity, certificate, attestation, tax_document)
# have no adjustment — all modules run at full confidence.

CONFIDENCE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "technical": {
        "structure": 0.20,  # JS in forms/interactive specs is normal; many revisions expected
        "content": 0.30,    # Invoice date/amount logic doesn't apply to API docs
        "fonts": 0.40,      # Mixed typography common in technical docs (code, diagrams)
    },
    "scientific": {
        "fonts": 0.15,      # Font diversity is normal in academic papers
        "structure": 0.25,  # Papers often have many incremental updates
        "content": 0.20,    # Invoice-specific date/amount checks don't apply
        "images": 0.50,     # Screenshots/figures are expected
    },
    "legal_notice": {
        "fonts": 0.30,      # Notices may use varied typography
        "structure": 0.30,  # Legal docs may have many revisions
        "content": 0.30,    # Notices may reference future-dated validity periods
    },
    # "unknown" deliberately gets NO multipliers (same as primary types).
    # Reason: an unclassified document could be a suspicious invoice.
    # Reducing confidence for unknown docs would let fraud slip through
    # when the classifier fails to identify the document type.
    # The UI still shows a warning banner for unknown docs.
    "unknown": {},
    # Primary document types: some have reduced confidence for modules that
    # don't apply well to their format.
    "invoice": {},
    "identity": {
        # Scanned ID documents legitimately have variable/OCR-extracted fonts.
        # Font diversity is normal for scanned government docs → reduce penalty weight.
        "fonts": 0.3,
        # Invoice-specific checks (SIRET, amounts, due dates) don't apply to IDs.
        # Reducing content module so it doesn't penalise lack of "facture" structure.
        "content": 0.4,
        # Scanned PDFs have minimal internal PDF structure (no incremental updates, no JS).
        # Structure module will see a "simple" PDF and have low signal → reduce weight.
        "structure": 0.5,
    },
    "certificate": {},
    "attestation": {},
    "tax_document": {},
}


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


# =============================================================================
# SCORE CALCULATION
# =============================================================================

def calculate_final_score(
    module_results: list[ModuleResult],
    document_type: DocumentType = "unknown",
) -> int:
    """
    Calculate the final trust score from all module results.

    The formula is a confidence-weighted average:

    final_score = Σ(score_i × weight_i × effective_confidence_i) / Σ(weight_i × effective_confidence_i)

    For non-invoice document types, some modules have their confidence
    multiplied by a factor < 1.0 so they contribute less to the final score.
    This prevents false positives from modules calibrated for financial documents
    when analyzing e.g. scientific papers or legal notices.

    Args:
        module_results: List of results from all analysis modules
        document_type: Detected document type (affects confidence multipliers)

    Returns:
        Final trust score (0-100)
    """
    if not module_results:
        return 100  # No analysis = assume innocent

    # Get the multiplier table for this document type (empty dict = all 1.0)
    multipliers = CONFIDENCE_MULTIPLIERS.get(document_type, {})

    weighted_sum = 0.0
    weight_total = 0.0

    for result in module_results:
        weight = MODULE_WEIGHTS.get(result.module, DEFAULT_WEIGHT)

        # Apply document-type confidence multiplier (default 1.0 = no change)
        type_multiplier = multipliers.get(result.module, 1.0)
        effective_weight = weight * result.confidence * type_multiplier

        weighted_sum += result.score * effective_weight
        weight_total += effective_weight

    if weight_total == 0:
        return 100  # No confidence in any module

    final_score = weighted_sum / weight_total
    return round(max(0, min(100, final_score)))


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
    """True if this CRITICAL is the known scanner→viewer false positive.

    Authentic scanned identity documents are saved by the scanner, then opened
    minutes later by a viewer (Adobe, Preview…), which stamps a new ModDate and
    trips META_DOCUMENT_MODIFIED as "critical". A gap under 30 minutes is a
    plausible scan-then-view workflow, not tampering.
    """
    if flag.severity != "critical" or flag.code != "META_DOCUMENT_MODIFIED":
        return False
    hours = (flag.details or {}).get("time_difference_hours", 999)
    try:
        return float(hours) < 0.5  # under 30 minutes
    except (TypeError, ValueError):
        return False  # unparseable → treat as a genuine critical


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
            ev = apply_doc_type_suppression(ev, document_type)  # Task 7 fills this in
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
