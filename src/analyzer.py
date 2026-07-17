"""
TrustyFile Analyzer - Main orchestrator for document fraud detection.

This module coordinates all analysis modules and produces the final result.

Usage:
    from src.analyzer import TrustyFileAnalyzer

    analyzer = TrustyFileAnalyzer()
    result = analyzer.analyze("invoice.pdf")

    print(f"Trust Score: {result.trust_score}/100")
    print(f"Risk Level: {result.risk_level}")
"""

import time
import logging
from pathlib import Path
from typing import Optional

from src.models import AnalysisResult, ModuleResult, Flag
from src.extractors.pdf_extractor import extract_pdf_data, PDFData
from src.modules.metadata import analyze_metadata
from src.modules.content import analyze_content
from src.modules.visual import analyze_visual
from src.modules.fonts import analyze_fonts
from src.modules.images import analyze_images
from src.modules.structure import analyze_structure
from src.modules.overlay import analyze_overlay
from src.modules.invisible import analyze_invisible
from src.modules.arithmetic import analyze_arithmetic
from src.modules.external import analyze_external
from src.modules.forensics import analyze_forensics
from src.modules.identity import analyze_identity, is_likely_identity_doc
from src.modules.twod_doc import PYLIBDMTX_AVAILABLE, analyze_twod_doc
from src.scoring import create_analysis_result, generate_summary
from src.document_type import detect_document_type

logger = logging.getLogger(__name__)

# Coverage gate (spec docs/superpowers/specs/2026-07-17-coverage-gate-design.md):
# a document whose text layer averages fewer than this many non-whitespace
# characters per page is "blind" — text-based modules had nothing to analyze.
# Value deliberately matches images.py's image-only heuristic (aligned by
# intent, not shared: the two checks compute different things).
MIN_TEXT_CHARS_PER_PAGE = 50


def _coverage_blind(
    text_by_page: list[str],
    page_count: int,
    module_results: list[ModuleResult],
) -> tuple[bool, float]:
    """Decide whether the document is blind (no usable text layer).

    A blind document can be proven suspicious, never proven fine: the caller
    uses this to band otherwise-GENUINE results as UNVERIFIABLE. Identity
    documents where the identity module actually FOUND an MRZ are exempt —
    scanned IDs are expected to be image-only and get real OCR analysis
    (spec D4). An identity crash or an MRZ-less run stays blind: fail closed.

    Returns:
        (blind, avg_non_whitespace_chars_per_page)
    """
    non_ws = sum(len("".join(page.split())) for page in text_by_page)
    avg = non_ws / max(1, page_count)
    if avg >= MIN_TEXT_CHARS_PER_PAGE:
        return False, avg
    for r in module_results:
        if r.module == "identity" and (r.details or {}).get("mrz_found"):
            return False, avg
    return True, avg


class TrustyFileAnalyzer:
    """
    Main analyzer class for TrustyFile document fraud detection.

    This class orchestrates all analysis modules and produces a final
    trust score with detailed breakdown.

    Attributes:
        enable_external: Whether to run external API verification
        enable_qr_scan: Whether to scan for QR codes (slower)

    Example:
        >>> analyzer = TrustyFileAnalyzer()
        >>> result = analyzer.analyze("invoice.pdf")
        >>> print(f"Score: {result.trust_score}, Risk: {result.risk_level}")
    """

    def __init__(
        self,
        enable_external: bool = False,
        enable_qr_scan: bool = True,
        enable_iban_check: bool = False,
        iban_api_key: str | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            enable_external: Enable external API verification (requires internet)
            enable_qr_scan: Enable QR code scanning (slower but recommended)
            enable_iban_check: Enable IBAN verification via anyapi.io
            iban_api_key: API key for anyapi.io IBAN validation
        """
        self.enable_external = enable_external
        self.enable_qr_scan = enable_qr_scan
        self.enable_iban_check = enable_iban_check
        self.iban_api_key = iban_api_key

    def analyze(
        self,
        file_path: str,
        expected_domains: Optional[list[str]] = None,
    ) -> AnalysisResult:
        """
        Analyze a document for signs of fraud.

        Args:
            file_path: Path to the PDF file
            expected_domains: Optional list of expected sender domains
                             (for QR code validation)

        Returns:
            AnalysisResult with trust score and detailed breakdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid PDF
        """
        start_time = time.time()

        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        # Extract PDF data
        logger.info(f"Analyzing: {file_path}")
        pdf_data = extract_pdf_data(file_path)

        # Fail closed: unreadable input must not score as trustworthy.
        if not pdf_data.extraction_ok:
            logger.warning("Extraction failed — returning UNVERIFIABLE-equivalent result")
            unverifiable = ModuleResult(
                module="extraction",
                flags=[Flag(
                    severity="high",
                    code="EXTRACTION_FAILED",
                    message="Document could not be parsed as a valid PDF",
                    details={"error": pdf_data.extraction_error},
                )],
                score=0,
                confidence=1.0,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            return create_analysis_result(
                file_hash=pdf_data.file_hash,
                module_results=[unverifiable],
                analysis_time_ms=elapsed_ms,
                document_type="unknown",
            )

        # Detect document type before running modules
        # This determines which modules get reduced confidence (to avoid false positives)
        doc_type = detect_document_type(pdf_data)
        logger.info(f"Document type detected: {doc_type}")

        # Run all modules with per-module crash isolation. Adversarial PDFs
        # are this tool's threat model — one MuPDF error inside one module
        # must degrade THAT module, never abort the whole analysis (2026-07
        # audit: previously any module exception crashed analyze() outright).
        module_results = []
        crashed_modules: list[str] = []

        def _run(name: str, fn) -> None:
            """Run one analysis module; on crash, record a MODULE_ERROR result.

            The degraded result carries one LOW-severity flag — its negligible
            ledger weight comes from that LOW severity (0.2 nats), not from
            confidence: flag_to_evidence hardcodes confidence=1.0 for every
            flag, so confidence is never consulted by the ledger. The
            confidence=0.0 set on the ModuleResult below is only a signal to
            any non-ledger consumer that this module found nothing usable.
            """
            logger.debug("Running %s analysis...", name)
            try:
                module_results.append(fn())
            except Exception as exc:
                logger.warning("Module '%s' crashed on %s: %s",
                               name, file_path, exc, exc_info=True)
                crashed_modules.append(name)
                module_results.append(ModuleResult(
                    module=name,
                    flags=[Flag(
                        severity="low",
                        code="MODULE_ERROR",
                        message=(f"The '{name}' analysis failed on this document "
                                 "and was skipped"),
                        # Exception CLASS NAME only — the full str(exc) can
                        # carry internals (paths, library error text) that
                        # must not reach this untrusted-input tool's output
                        # channel. The full message stays server-side, in the
                        # logger.warning call above (exc_info=True).
                        details={"module": name, "error": type(exc).__name__},
                    )],
                    score=100,
                    confidence=0.0,
                ))

        _run("metadata", lambda: analyze_metadata(pdf_data))                      # Module A
        _run("content", lambda: analyze_content(pdf_data))                        # Module B
        _run("visual", lambda: analyze_visual(pdf_data,                           # Module C
                                              expected_domains=expected_domains,
                                              check_qr=self.enable_qr_scan))
        _run("fonts", lambda: analyze_fonts(pdf_data))                            # Module D
        _run("images", lambda: analyze_images(pdf_data))                          # Module F
        _run("structure", lambda: analyze_structure(pdf_data))                    # Module E
        _run("overlay", lambda: analyze_overlay(pdf_data))                        # cover-and-retype
        _run("invisible", lambda: analyze_invisible(pdf_data))                    # Tr-3 shadow values
        _run("arithmetic", lambda: analyze_arithmetic(pdf_data))                  # totals + e-invoice XML
        _run("forensics", lambda: analyze_forensics(file_path))                   # Module H (ELA)
        _run("twod_doc", lambda: analyze_twod_doc(pdf_data))                      # 2D-Doc cross-check

        # Module Identity: MRZ analysis (passports, ID cards).
        # Triggered if:
        #   - classifier already detected "identity" (doc has a text layer with ID keywords), OR
        #   - doc is "unknown" AND passes the quick pre-check (identity keywords OR face photo).
        #     Scanned IDs have no text → classifier returns "unknown" → pre-check saves the day.
        # The pre-check (keywords + face detection at 100 DPI) is much faster than OCR itself.
        try:
            _run_identity = doc_type == "identity" or (
                doc_type == "unknown" and is_likely_identity_doc(pdf_data)
            )
        except Exception as exc:
            logger.warning("Identity pre-check crashed: %s", exc)
            _run_identity = False
        if _run_identity:
            _run("identity", lambda: analyze_identity(pdf_data))
            # If the classifier missed it (returned "unknown" because the scan has no
            # text layer), override so scoring uses identity-specific multipliers and caps.
            if doc_type == "unknown":
                doc_type = "identity"
                logger.info("Document type overridden to 'identity' (face/keyword pre-check)")

        # Module G: External Verification (optional)
        if self.enable_external:
            _run("external", lambda: analyze_external(
                pdf_data,
                verify_iban=self.enable_iban_check,
                iban_api_key=self.iban_api_key,
            ))

        # Fail closed on systemic failure: when three or more modules crash,
        # the document defeated the analysis pipeline — the result cannot be
        # trusted in either direction, so it must be UNVERIFIABLE rather than
        # a high score computed from the few modules that survived.
        if len(crashed_modules) >= 3:
            logger.warning("Analysis degraded: %d modules crashed (%s) — UNVERIFIABLE",
                           len(crashed_modules), ", ".join(crashed_modules))
            module_results.append(ModuleResult(
                module="analysis",
                flags=[Flag(
                    severity="high",
                    code="EXTRACTION_FAILED",
                    message=(f"{len(crashed_modules)} analysis modules failed on this "
                             "document — result is unverifiable"),
                    details={"crashed_modules": crashed_modules},
                )],
                score=0,
                confidence=1.0,
            ))

        # Calculate analysis time
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Create final result (document_type used for confidence multipliers)
        result = create_analysis_result(
            file_hash=pdf_data.file_hash,
            module_results=module_results,
            analysis_time_ms=elapsed_ms,
            document_type=doc_type,
        )

        logger.info(f"Analysis complete: score={result.trust_score}, risk={result.risk_level}")
        return result

    def analyze_with_summary(
        self,
        file_path: str,
        expected_domains: Optional[list[str]] = None,
    ) -> tuple[AnalysisResult, str]:
        """
        Analyze a document and return both result and human-readable summary.

        Args:
            file_path: Path to the PDF file
            expected_domains: Optional list of expected sender domains

        Returns:
            Tuple of (AnalysisResult, summary_string)
        """
        result = self.analyze(file_path, expected_domains)
        summary = generate_summary(result)
        return result, summary


def quick_analyze(file_path: str) -> dict:
    """
    Quick analysis function for simple use cases.

    Returns a simple dict instead of full AnalysisResult.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dict with score, risk_level, and flag_count

    Example:
        >>> result = quick_analyze("invoice.pdf")
        >>> print(result)
        {'score': 85, 'risk_level': 'LOW', 'flag_count': 2}
    """
    analyzer = TrustyFileAnalyzer(enable_external=False)
    result = analyzer.analyze(file_path)

    total_flags = sum(len(m.flags) for m in result.modules)

    return {
        "score": result.trust_score,
        "risk_level": result.risk_level,
        "flag_count": total_flags,
        "analysis_time_ms": result.analysis_time_ms,
    }
