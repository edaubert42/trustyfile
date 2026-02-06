"""
Data structures used across all TrustyFile modules.

These dataclasses define the standard format for module results and flags.
Every analysis module returns a ModuleResult containing Flag objects.
"""

from dataclasses import dataclass, field
from typing import Literal


# Severity levels for flags, from least to most concerning
SeverityLevel = Literal["low", "medium", "high", "critical"]


@dataclass
class Flag:
    """
    Represents a single suspicious finding in a document.

    Attributes:
        severity: How serious is this finding?
            - "low": Minor issue, might be normal (e.g., common font used)
            - "medium": Worth noting (e.g., document edited with online tool)
            - "high": Suspicious (e.g., dates don't match)
            - "critical": Very likely fraud (e.g., future date on old invoice)
        code: A unique identifier for this type of flag.
            Format: MODULE_SPECIFIC_ISSUE (e.g., "META_ONLINE_CONVERTER")
            This helps with testing and filtering specific flag types.
        message: Human-readable description of the issue.
            Should be clear enough for non-technical users.
        details: Optional dict with additional context.
            Example: {"converter": "iLovePDF", "version": "2.0"}

    Example:
        >>> flag = Flag(
        ...     severity="medium",
        ...     code="META_ONLINE_CONVERTER",
        ...     message="Document was processed by iLovePDF",
        ...     details={"producer": "iLovePDF"}
        ... )
    """
    severity: SeverityLevel
    code: str
    message: str
    details: dict | None = None


@dataclass
class ModuleResult:
    """
    Result returned by each analysis module.

    Attributes:
        module: Name of the module (e.g., "metadata", "content", "fonts")
        flags: List of suspicious findings. Empty list = nothing found.
        score: Trust score from 0 to 100.
            - 0 = Very suspicious, likely fraudulent
            - 100 = Looks completely legitimate
            - Starts at 100, decreases based on flags found
        confidence: How confident is the module in its analysis (0.0 to 1.0).
            - 1.0 = Very confident (clear data, no ambiguity)
            - 0.5 = Moderate confidence (some data missing or unclear)
            - 0.0 = No confidence (couldn't analyze, data corrupted)
            This affects how much the module's score weighs in the final score.

    Example:
        >>> result = ModuleResult(
        ...     module="metadata",
        ...     flags=[Flag("medium", "META_ONLINE_CONVERTER", "...")],
        ...     score=75,
        ...     confidence=0.95
        ... )
    """
    module: str
    flags: list[Flag] = field(default_factory=list)
    score: int = 100  # Start at 100 (innocent until proven guilty)
    confidence: float = 1.0


@dataclass
class AnalysisResult:
    """
    Final result combining all module analyses.

    This is what the user sees - the overall trust score and detailed breakdown.

    Attributes:
        file_hash: SHA256 hash of the analyzed file (for caching/reference)
        trust_score: Final combined score (0-100)
        risk_level: Human-readable risk category
        modules: List of individual module results
        analysis_time_ms: How long the analysis took in milliseconds
    """
    file_hash: str
    trust_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    modules: list[ModuleResult] = field(default_factory=list)
    analysis_time_ms: int = 0


@dataclass
class AnalysisSummary:
    """
    Rich summary with a short verdict and a list of bullet findings.

    Attributes:
        verdict: Short bold statement about the document's trustworthiness.
            Example: "We suspect this document has been altered."
        bullets: List of finding sentences, each displayed as a bullet point.
            Includes negative findings grouped by theme, plus positive signals.
    """
    verdict: str
    bullets: list[str] = field(default_factory=list)
