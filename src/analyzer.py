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

from src.models import AnalysisResult, ModuleResult
from src.extractors.pdf_extractor import extract_pdf_data, PDFData
from src.modules.metadata import analyze_metadata
from src.modules.content import analyze_content
from src.modules.visual import analyze_visual
from src.modules.fonts import analyze_fonts
from src.modules.images import analyze_images
from src.modules.structure import analyze_structure
from src.modules.external import analyze_external
from src.modules.forensics import analyze_forensics
from src.scoring import create_analysis_result, generate_summary

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize the analyzer.

        Args:
            enable_external: Enable external API verification (requires internet)
            enable_qr_scan: Enable QR code scanning (slower but recommended)
        """
        self.enable_external = enable_external
        self.enable_qr_scan = enable_qr_scan

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

        # Run all modules
        module_results = []

        # Module A: Metadata Analysis
        logger.debug("Running metadata analysis...")
        module_results.append(analyze_metadata(pdf_data))

        # Module B: Content Analysis
        logger.debug("Running content analysis...")
        module_results.append(analyze_content(pdf_data))

        # Module C: Visual Analysis
        logger.debug("Running visual analysis...")
        module_results.append(analyze_visual(
            pdf_data,
            expected_domains=expected_domains,
            check_qr=self.enable_qr_scan,
        ))

        # Module D: Font Analysis
        logger.debug("Running font analysis...")
        module_results.append(analyze_fonts(pdf_data))

        # Module F: Image Analysis
        logger.debug("Running image analysis...")
        module_results.append(analyze_images(pdf_data))

        # Module E: Structure Analysis
        logger.debug("Running structure analysis...")
        module_results.append(analyze_structure(pdf_data))

        # Module H: Forensic Analysis (ELA)
        logger.debug("Running forensic analysis...")
        module_results.append(analyze_forensics(file_path))

        # Module G: External Verification (optional)
        if self.enable_external:
            logger.debug("Running external verification...")
            module_results.append(analyze_external(pdf_data))

        # Calculate analysis time
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Create final result
        result = create_analysis_result(
            file_hash=pdf_data.file_hash,
            module_results=module_results,
            analysis_time_ms=elapsed_ms,
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
