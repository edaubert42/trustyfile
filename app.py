"""
TrustyFile - Document Fraud Detection

A Streamlit web application for analyzing invoices and documents
to detect signs of manipulation or fraud.

Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import fitz  # PyMuPDF

from src.analyzer import TrustyFileAnalyzer
from src.scoring import generate_summary, collect_all_flags, count_flags_by_severity


# =============================================================================
# PDF PREVIEW FUNCTIONS
# =============================================================================

def render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5) -> bytes:
    """
    Render a PDF page as a PNG image.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to render (0-indexed)
        zoom: Zoom factor for resolution (1.5 = 150% size)

    Returns:
        PNG image bytes
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Create a matrix for zoom
    mat = fitz.Matrix(zoom, zoom)

    # Render page to pixmap (image)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PNG bytes
    img_bytes = pix.tobytes("png")

    doc.close()
    return img_bytes


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TrustyFile - Document Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .score-low { background-color: #198754; border: 2px solid #146c43; color: white; }
    .score-medium { background-color: #fd7e14; border: 2px solid #dc6a10; color: white; }
    .score-high { background-color: #dc3545; border: 2px solid #b02a37; color: white; }
    .score-critical { background-color: #721c24; border: 2px solid #4a1119; color: white; }
    .flag-critical { color: #721c24; font-weight: bold; }
    .flag-high { color: #dc3545; }
    .flag-medium { color: #fd7e14; }
    .flag-low { color: #6c757d; }
    .file-hash {
        background-color: #f8f9fa;
        padding: 0.75rem 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-family: monospace;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## Settings")

    enable_external = st.checkbox(
        "Enable external verification",
        value=False,
        help="Verify SIRET/VAT numbers against official APIs (requires internet)"
    )

    enable_qr = st.checkbox(
        "Scan for QR codes",
        value=True,
        help="Detect and analyze QR codes in the document"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **TrustyFile** analyzes documents to detect signs of fraud or manipulation.

    Upload a PDF invoice and get a trust score with detailed breakdown.

    **Modules:**
    - Metadata analysis
    - Content & date verification
    - Visual elements (QR codes, watermarks)
    - Font consistency
    - Image analysis
    - External verification (optional)
    """)

    st.markdown("---")
    st.markdown("Made with Claude Code")


# =============================================================================
# MAIN CONTENT
# =============================================================================

col_spacer1, col_logo, col_spacer2 = st.columns([1, 2, 1])
with col_logo:
    st.image("static/full_logo_trustyfile.png", use_container_width=True)
st.markdown('<p class="sub-header">Document Fraud Detection - Upload an invoice to analyze</p>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload the invoice or document you want to analyze"
)

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Run analysis
        with st.spinner("Analyzing document..."):
            analyzer = TrustyFileAnalyzer(
                enable_external=enable_external,
                enable_qr_scan=enable_qr,
            )
            result = analyzer.analyze(tmp_path)
            summary = generate_summary(result)

        # Get page count for navigation
        page_count = get_pdf_page_count(tmp_path)

        # Display results with PDF preview
        st.markdown("---")

        # Two-column layout: Preview | Results
        preview_col, results_col = st.columns([1, 1])

        # =================================================================
        # LEFT COLUMN: PDF Preview
        # =================================================================
        with preview_col:
            # Page navigation if multiple pages
            if page_count > 1:
                st.selectbox(
                    "Page",
                    range(1, page_count + 1),
                    format_func=lambda x: f"Page {x} / {page_count}",
                    key="page_selector"
                )
                page_index = st.session_state.page_selector - 1
            else:
                page_index = 0

            # Render and display the page
            try:
                img_bytes = render_pdf_page(tmp_path, page_index, zoom=2.0)
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.error(f"Could not render PDF preview: {e}")

        # =================================================================
        # RIGHT COLUMN: Analysis Results
        # =================================================================
        with results_col:
            # Main score box (large and prominent)
            risk_class = f"score-{result.risk_level.lower()}"
            risk_emoji = {
                "LOW": "✅",
                "MEDIUM": "⚠️",
                "HIGH": "🔴",
                "CRITICAL": "🚨"
            }.get(result.risk_level, "?")

            st.markdown(f"""
            <div class="score-box {risk_class}">
                <h1 style="margin:0; font-size: 4rem; font-weight: bold;">{result.trust_score}/100</h1>
                <h2 style="margin:0.5rem 0 0 0;">{risk_emoji} {result.risk_level} RISK</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**{summary}**")

            # Flag counts
            all_flags = collect_all_flags(result.modules)
            counts = count_flags_by_severity(all_flags)

            st.markdown("### Issues Found")
            issues_cols = st.columns(4)
            with issues_cols[0]:
                if counts["critical"]:
                    st.markdown(f'🚨 **{counts["critical"]}** Critical')
            with issues_cols[1]:
                if counts["high"]:
                    st.markdown(f'🔴 **{counts["high"]}** High')
            with issues_cols[2]:
                if counts["medium"]:
                    st.markdown(f'⚠️ **{counts["medium"]}** Medium')
            with issues_cols[3]:
                if counts["low"]:
                    st.markdown(f'ℹ️ **{counts["low"]}** Low')

            if not any(counts.values()):
                st.markdown("✅ No issues found")

            st.markdown(f"**Analysis completed in {result.analysis_time_ms}ms**")

        # File hash (full width, below the two columns)
        st.markdown(f"""
        <div class="file-hash">
            <strong>SHA256:</strong> {result.file_hash}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## Module Breakdown")

        cols = st.columns(len(result.modules))
        for i, module in enumerate(result.modules):
            with cols[i]:
                # Module score color
                if module.score >= 80:
                    color = "#28a745"
                elif module.score >= 50:
                    color = "#ffc107"
                else:
                    color = "#dc3545"

                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 5px;">
                    <h4 style="margin: 0; text-transform: capitalize;">{module.module}</h4>
                    <h2 style="margin: 0.5rem 0; color: {color};">{module.score}</h2>
                    <small>Confidence: {module.confidence:.0%}</small>
                </div>
                """, unsafe_allow_html=True)

        # Detailed flags
        if all_flags:
            st.markdown("---")
            st.markdown("## Detailed Findings")

            for flag in all_flags:
                severity_icon = {
                    "critical": "🚨",
                    "high": "🔴",
                    "medium": "⚠️",
                    "low": "ℹ️"
                }.get(flag.severity, "•")

                with st.expander(f"{severity_icon} [{flag.severity.upper()}] {flag.message}"):
                    st.markdown(f"**Code:** `{flag.code}`")
                    if flag.details:
                        st.json(flag.details)

        # Raw data (collapsible)
        with st.expander("📊 Raw Analysis Data"):
            for module in result.modules:
                st.markdown(f"### {module.module.capitalize()}")
                st.json({
                    "score": module.score,
                    "confidence": module.confidence,
                    "flags": [
                        {
                            "severity": f.severity,
                            "code": f.code,
                            "message": f.message,
                        }
                        for f in module.flags
                    ]
                })

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)

else:
    # No file uploaded - show instructions
    st.markdown("""
    ### How it works

    1. **Upload** a PDF invoice or document
    2. **Wait** for the analysis to complete (~1-2 seconds)
    3. **Review** the trust score and detailed findings

    ### What we detect

    | Check | Description |
    |-------|-------------|
    | 🔧 **Metadata** | Online converters, AI tools, date manipulation |
    | 📅 **Dates** | Anachronisms, impossible dates, future invoices |
    | 📝 **References** | Inconsistent invoice numbers across document |
    | 🏢 **Legal Info** | Invalid SIRET/VAT numbers |
    | 🔤 **Fonts** | Excessive font diversity, editing traces |
    | 🖼️ **Images** | Screenshots, resolution mismatches |
    | 📱 **QR Codes** | Suspicious URLs, domain mismatches |
    | 💧 **Watermarks** | SPECIMEN, COPY, converter marks |

    ### Risk Levels

    - ✅ **LOW** (80-100): Document appears legitimate
    - ⚠️ **MEDIUM** (50-79): Some concerns, verify manually
    - 🔴 **HIGH** (20-49): Multiple red flags
    - 🚨 **CRITICAL** (0-19): Strong evidence of fraud
    """)
