"""
Scoring System - Combines all module results into a final trust score.

The final score is a weighted average of all module scores,
adjusted by each module's confidence level.

Risk levels:
- LOW (80-100): Document appears legitimate
- MEDIUM (50-79): Some concerns, manual verification recommended
- HIGH (20-49): Multiple red flags, likely manipulated
- CRITICAL (0-19): Strong evidence of fraud
"""

from dataclasses import dataclass
from typing import Literal
from src.models import ModuleResult, AnalysisResult, Flag


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
    "structure": 1.3,   # PDF structure (incremental updates, JS, etc.) - very reliable
    "external": 1.5,    # External verification (very reliable when available)
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


# =============================================================================
# SCORE CALCULATION
# =============================================================================

def calculate_final_score(module_results: list[ModuleResult]) -> int:
    """
    Calculate the final trust score from all module results.

    The formula is a confidence-weighted average:

    final_score = Σ(score_i × weight_i × confidence_i) / Σ(weight_i × confidence_i)

    This means:
    - Modules with higher confidence have more influence
    - Modules with higher weight have more influence
    - If a module has 0 confidence, it doesn't affect the score

    Args:
        module_results: List of results from all analysis modules

    Returns:
        Final trust score (0-100)
    """
    if not module_results:
        return 100  # No analysis = assume innocent

    weighted_sum = 0.0
    weight_total = 0.0

    for result in module_results:
        weight = MODULE_WEIGHTS.get(result.module, DEFAULT_WEIGHT)
        effective_weight = weight * result.confidence

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

def create_analysis_result(
    file_hash: str,
    module_results: list[ModuleResult],
    analysis_time_ms: int = 0,
) -> AnalysisResult:
    """
    Create the final analysis result from all module results.

    Args:
        file_hash: SHA256 hash of the analyzed file
        module_results: List of results from all analysis modules
        analysis_time_ms: Time taken for analysis in milliseconds

    Returns:
        AnalysisResult with final score, risk level, and all details
    """
    # Calculate final score
    trust_score = calculate_final_score(module_results)

    # Determine risk level from score
    risk_level = get_risk_level(trust_score)

    # Override risk level if there are critical flags
    # A single critical flag means the document should NOT be trusted
    all_flags = collect_all_flags(module_results)
    flag_counts = count_flags_by_severity(all_flags)

    if flag_counts["critical"] >= 1:
        # Any critical flag = at least HIGH risk
        if risk_level == "LOW" or risk_level == "MEDIUM":
            risk_level = "HIGH"
        # Base cap at 40 for a single critical flag
        trust_score = min(trust_score, 40)

        # High flags further reduce the cap (-5 each)
        trust_score -= flag_counts["high"] * 5
        # Medium flags slightly reduce (-2 each)
        trust_score -= flag_counts["medium"] * 2
        trust_score = max(trust_score, 5)

    if flag_counts["critical"] >= 2:
        # Multiple critical flags = CRITICAL risk
        risk_level = "CRITICAL"
        trust_score = min(trust_score, 19)

    return AnalysisResult(
        file_hash=file_hash,
        trust_score=trust_score,
        risk_level=risk_level,
        modules=module_results,
        analysis_time_ms=analysis_time_ms,
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
