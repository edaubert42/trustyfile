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
from src.scoring import collect_all_flags, count_flags_by_severity
from src.summary import generate_rich_summary
from src.extractors.pdf_extractor import extract_pdf_data


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
    page_title="TrustyFile - Document Fraud Detector",
    page_icon="static/logotf_small.png",
    layout="wide",
    initial_sidebar_state="collapsed",
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

</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

# Default settings (sidebar removed)
enable_external = False
enable_qr = True


# =============================================================================
# MAIN CONTENT
# =============================================================================

import base64
with open("static/logotf_small.png", "rb") as _logo_file:
    _logo_b64 = base64.b64encode(_logo_file.read()).decode()
with open("static/upload_icon_medium.png", "rb") as _upload_file:
    _upload_b64 = base64.b64encode(_upload_file.read()).decode()

st.markdown("""<style>
    .block-container { padding-top: 1.5rem !important; }
    /* Clean up the dropzone: no border, centered content */
    [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    /* Hide the label */
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    /* Hide the "Drag and drop file here" and limit text */
    [data-testid="stFileUploaderDropzone"] > div > span,
    [data-testid="stFileUploaderDropzone"] > div > small,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    /* Style the browse button */
    [data-testid="baseButton-secondary"] {
        padding: 0.6rem 2.5rem !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px;">
    <img src="data:image/png;base64,{_logo_b64}" style="height:52px;">
    <span style="font-size:1.9rem; font-weight:bold; color:white;">TrustyFile</span>
    <span style="font-size:1.05rem; color:#888; margin-left:4px;">Document Fraud Detector</span>
</div>
""", unsafe_allow_html=True)

# Uploader — use a container so we can control what shows around it
upload_container = st.container()
with upload_container:
    # Tagline + icon only shown on landing page (no file yet)
    # We render uploader inside a centered column regardless
    col_left, col_upload, col_right = st.columns([1, 2, 1])
    with col_upload:
        # Landing page content (hidden via CSS when file is uploaded)
        st.markdown(f"""
        <div id="landing-content">
            <p style="color:white; font-size:1.6rem; text-align:center; margin:3rem auto 2rem auto; max-width:600px;">
                <strong style="font-size:1.8rem;">Detect if a document is fraud instantly.</strong><br>
                <span style="font-weight:normal; font-size:1.1rem; color:white;">TrustyFile spots clever edits and hidden changes inside documents.</span>
            </p>
            <div style="text-align:center; margin-bottom:1.5rem;">
                <img src="data:image/png;base64,{_upload_b64}" style="height:160px; opacity:0.85;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["pdf"],
            label_visibility="collapsed",
        )
        st.markdown("""
        <div id="landing-disclaimer">
            <p style="color:#666; font-size:0.75rem; text-align:center; margin-top:0.8rem;">
                By uploading a file, you agree that your document will be analyzed locally.
                No data is sent to external servers.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Hide landing content when a file is uploaded
if uploaded_file is not None:
    st.markdown("""<style>
        #landing-content, #landing-disclaimer { display: none !important; }
    </style>""", unsafe_allow_html=True)

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
            summary = generate_rich_summary(result)

        # Get page count for navigation
        page_count = get_pdf_page_count(tmp_path)

        # Display results with PDF preview
        st.markdown("---")

        # Two-column layout: Preview | Results
        preview_col, results_col = st.columns([2, 3])

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

            # File properties block
            pdf_data = extract_pdf_data(tmp_path)
            meta = pdf_data.metadata

            # Check for signature info in structure module flags
            signature_info = None
            for module in result.modules:
                if module.module == "structure":
                    for flag in module.flags:
                        if flag.details and "signer" in flag.details:
                            signature_info = flag.details.get("signer", "Unknown")
                            break

            def format_date(dt):
                return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"

            label_style = "color:#888; font-size:0.8rem; width:120px; flex-shrink:0;"
            value_style = "color:#eee; font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
            row_style = "display:flex; align-items:center; padding:6px 0; border-bottom:1px solid #2a2a2a;"

            editor = meta.producer or meta.creator or '—'
            sig = signature_info or 'None'

            file_rows = f"""
            <div style="{row_style}">
                <span style="{label_style}">Name</span>
                <span style="{value_style}" title="{uploaded_file.name}">{uploaded_file.name}</span>
            </div>
            <div style="{row_style}">
                <span style="{label_style}">Creation date</span>
                <span style="{value_style}" title="{format_date(meta.creation_date)}">{format_date(meta.creation_date)}</span>
            </div>
            <div style="{row_style}">
                <span style="{label_style}">Modification date</span>
                <span style="{value_style}" title="{format_date(meta.mod_date)}">{format_date(meta.mod_date)}</span>
            </div>
            <div style="{row_style}">
                <span style="{label_style}">Editor</span>
                <span style="{value_style}" title="{editor}">{editor}</span>
            </div>
            <div style="{row_style}">
                <span style="{label_style}">Signed by</span>
                <span style="{value_style}" title="{sig}">{sig}</span>
            </div>
            <div style="display:flex; align-items:center; padding:6px 0;">
                <span style="{label_style}">SHA256</span>
                <span style="{value_style} font-size:0.75rem;" title="{result.file_hash}">{result.file_hash}</span>
            </div>
            """

            st.markdown(f"""
            <div style="margin-top:1.5rem;">
                <span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;
                    letter-spacing:0.5px;">File Properties</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">
                {file_rows}
            </div>
            """, unsafe_allow_html=True)

        # =================================================================
        # RIGHT COLUMN: Analysis Results
        # =================================================================
        with results_col:
            # Main score gauge (circular arc like HubSpot Website Grader)
            score = result.trust_score
            risk_level = result.risk_level

            # Color based on score
            if score >= 80:
                gauge_color = "#28a745"
            elif score >= 50:
                gauge_color = "#fd7e14"
            elif score >= 20:
                gauge_color = "#dc3545"
            else:
                gauge_color = "#721c24"

            # SVG circular gauge (full donut ring)
            # Uses a circle with stroke-dasharray to fill proportionally
            # The circle starts at the top (rotated -90°)
            radius = 80
            circumference = 2 * 3.14159 * radius  # full circle
            filled = circumference * (score / 100)
            size = 220
            center = size // 2

            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0 0 0;">
                <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
                    <!-- Background circle (gray track) -->
                    <circle cx="{center}" cy="{center}" r="{radius}"
                        fill="none" stroke="#2a2a2a" stroke-width="14"/>
                    <!-- Filled arc (score), rotated so it starts from the top -->
                    <circle cx="{center}" cy="{center}" r="{radius}"
                        fill="none" stroke="{gauge_color}" stroke-width="14"
                        stroke-linecap="round"
                        stroke-dasharray="{filled} {circumference}"
                        transform="rotate(90 {center} {center})"
                        style="transition: stroke-dasharray 1s ease;"/>
                    <!-- Score text -->
                    <text x="{center}" y="{center - 5}" text-anchor="middle"
                        font-size="48" font-weight="bold" fill="white">{score}</text>
                    <text x="{center}" y="{center + 22}" text-anchor="middle"
                        font-size="14" fill="{gauge_color}" font-weight="bold">
                        {risk_level} RISK</text>
                </svg>
            </div>
            """, unsafe_allow_html=True)

            # Module score bars (right below the gauge)
            bars_html = ""
            for module in result.modules:
                if module.score >= 80:
                    bar_color = "#28a745"
                elif module.score >= 50:
                    bar_color = "#fd7e14"
                elif module.score >= 20:
                    bar_color = "#dc3545"
                else:
                    bar_color = "#721c24"

                bars_html += f'''
                <div style="display:flex; align-items:center; margin-bottom:6px;">
                    <span style="width:90px; font-size:0.8rem; color:#eee;
                        text-transform:capitalize;">{module.module}</span>
                    <div style="flex:1; background:#2a2a2a; border-radius:4px;
                        height:10px; overflow:hidden;">
                        <div style="width:{module.score}%; height:100%;
                            background:{bar_color}; border-radius:4px;
                            transition:width 1s ease;"></div>
                    </div>
                    <span style="width:35px; text-align:right; font-size:0.8rem;
                        color:#eee; margin-left:8px; font-weight:bold;">{module.score}</span>
                </div>'''

            # Flag counts
            all_flags = collect_all_flags(result.modules)
            counts = count_flags_by_severity(all_flags)

            # Issues pills (right below the circle)
            pills_html = ""
            if counts["critical"]:
                pills_html += f'<span style="background:#721c24; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["critical"]} Critical</span> '
            if counts["high"]:
                pills_html += f'<span style="background:#dc3545; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["high"]} High</span> '
            if counts["medium"]:
                pills_html += f'<span style="background:#fd7e14; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["medium"]} Medium</span> '
            if counts["low"]:
                pills_html += f'<span style="background:#555; color:#ccc; padding:4px 10px; border-radius:12px; font-size:0.8rem;">{counts["low"]} Low</span> '

            if not any(counts.values()):
                pills_html = '<span style="background:#28a745; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">No issues found</span>'

            st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center; margin-top:0.3rem;">{pills_html}</div>'
                        f'<p style="text-align:center; font-size:0.7rem; color:#888; margin-top:0.4rem;">Analysis completed in {result.analysis_time_ms}ms</p>',
                        unsafe_allow_html=True)

            # Module score bars
            st.markdown(f'<div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">{bars_html}</div>',
                        unsafe_allow_html=True)

            # Summary block — verdict (bold) + bullet list of findings
            bullets_html = "".join(
                f'<li style="margin-bottom:0.4rem;">{bullet}</li>'
                for bullet in summary.bullets
            )
            st.markdown(f"""
            <div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">
                <span style="color:white; font-size:1.3rem; font-weight:bold; text-transform:uppercase;
                    letter-spacing:0.5px; display:block; text-align:center;">Our Analysis</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">
                <p style="color:white; font-size:1.03rem; font-weight:bold; margin:0 0 0.5rem 0;">{summary.verdict}</p>
                <ul style="color:#ccc; font-size:0.95rem; line-height:1.5; padding-left:0.8rem; margin:0;">
                    {bullets_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Detailed flags
        if all_flags:
            st.markdown("---")
            st.markdown("## Detailed Findings")

            for flag in all_flags:
                severity_icon = {
                    "critical": "●",
                    "high": "●",
                    "medium": "●",
                    "low": "○"
                }.get(flag.severity, "•")

                with st.expander(f"{severity_icon} [{flag.severity.upper()}] {flag.message}"):
                    st.markdown(f"**Code:** `{flag.code}`")
                    if flag.details:
                        st.json(flag.details)

        # Raw data (collapsible)
        with st.expander("Raw Analysis Data"):
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

# Footer
st.markdown("""
<div style="position:fixed; bottom:0; left:0; right:0; text-align:center;
    padding:0.8rem; background:#0e1117; border-top:1px solid #1a1a2e; z-index:999;">
    <span style="color:#555; font-size:0.75rem;">
        TrustyFile — Open-source document fraud detection · Made with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
