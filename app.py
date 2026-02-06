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
import requests

from src.analyzer import TrustyFileAnalyzer
from src.scoring import collect_all_flags, count_flags_by_severity, MODULE_WEIGHTS, DEFAULT_WEIGHT
from src.summary import generate_rich_summary
from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.structure import get_modification_history


# =============================================================================
# VIRUSTOTAL LOOKUP
# =============================================================================

def check_virustotal(sha256: str) -> dict | None:
    """
    Look up a file hash on VirusTotal.

    Uses the VirusTotal v3 API to check if this file has been scanned before.
    Returns a dict with the results, or None if the lookup failed.

    Args:
        sha256: The SHA256 hash of the file

    Returns:
        dict with keys:
            - "status": "clean" | "malicious" | "unknown"
            - "malicious": int (number of engines that flagged the file)
            - "total": int (total number of engines)
            - "link": str (URL to the VirusTotal report)
        or None if the API call failed (no key, network error, etc.)
    """
    try:
        api_key = st.secrets.get("VIRUSTOTAL_API_KEY", "")
        if not api_key:
            return None

        url = f"https://www.virustotal.com/api/v3/files/{sha256}"
        headers = {"x-apikey": api_key}
        resp = requests.get(url, headers=headers, timeout=10)

        link = f"https://www.virustotal.com/gui/file/{sha256}"

        if resp.status_code == 404:
            # File was never scanned on VirusTotal
            return {"status": "unknown", "malicious": 0, "total": 0, "link": link}

        if resp.status_code != 200:
            return None

        data = resp.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
        total = sum(stats.values())

        if malicious > 0:
            return {"status": "malicious", "malicious": malicious, "total": total, "link": link}
        else:
            return {"status": "clean", "malicious": 0, "total": total, "link": link}

    except Exception:
        return None


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

    /* Label with hover tooltip */
    .tf-label {
        position: relative;
        cursor: help;
        font-size: 0.8rem;
    }
    .tf-label .tf-tip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 50%;
        left: 100%;
        transform: translateY(-50%);
        margin-left: 8px;
        background: #333;
        color: #ccc;
        font-size: 0.7rem;
        padding: 4px 10px;
        border-radius: 6px;
        white-space: nowrap;
        transition: opacity 0.2s;
        pointer-events: none;
        z-index: 10;
    }
    .tf-label:hover .tf-tip {
        visibility: visible;
        opacity: 1;
    }

    /* Styled expanders — dark card look */
    [data-testid="stExpander"] {
        background: #1a1a2e;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary {
        padding: 10px 14px;
        font-size: 0.88rem;
        font-weight: bold;
        color: #eee;
    }
    [data-testid="stExpander"] summary:hover {
        color: white;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: 1px solid #2a2a2a;
        padding: 14px;
    }

</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

# Default settings (sidebar removed)
enable_external = True
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

# Hide landing content and minimize uploader when a file is uploaded
if uploaded_file is not None:
    st.markdown("""<style>
        #landing-content, #landing-disclaimer { display: none !important; }
        /* Hide the dropzone, keep only the filename bar */
        [data-testid="stFileUploaderDropzone"] {
            display: none !important;
        }
        [data-testid="stFileUploader"] {
            margin-bottom: -0.5rem !important;
        }
        /* Keep original top padding for logo */
        .block-container { padding-top: 1.5rem !important; }
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
            history = get_modification_history(tmp_path)

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

            producer = meta.producer or meta.creator or '—'
            sig = signature_info or 'None'

            # Detect editor from XMP toolkit (if different from producer)
            # The XMP toolkit reveals if a different software modified the PDF
            editor_name = None
            if history.get("diffs"):
                # Use the last diff's to_tool which contains the detection
                last_diff = history["diffs"][-1]
                to_tool = last_diff.get("to_tool", "")
                # If it contains "→ modified with", extract the editor name
                if "→ modified with" in to_tool:
                    editor_name = to_tool.split("→ modified with ")[-1]

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
                <span style="{label_style}">Producer</span>
                <span style="{value_style}" title="{producer}">{producer}</span>
            </div>
            """

            # Only show Editor line if a different software modified the PDF
            if editor_name:
                file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">Editor</span>
                <span style="{value_style}" title="{editor_name}">{editor_name}</span>
            </div>
            """

            file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">Signed by</span>
                <span style="{value_style}" title="{sig}">{sig}</span>
            </div>
            <div style="display:flex; align-items:center; padding:6px 0;">
                <span style="{label_style}">SHA256</span>
                <span style="{value_style} font-size:0.75rem;" title="{result.file_hash}">{result.file_hash}</span>
            </div>
            """

            # VirusTotal lookup — check the file hash against VT's database
            vt_result = check_virustotal(result.file_hash)
            if vt_result is not None:
                vt_link = vt_result["link"]
                if vt_result["status"] == "clean":
                    vt_pill = (
                        f'<a href="{vt_link}" target="_blank" style="text-decoration:none;">'
                        f'<span style="background:#28a745; color:white; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.78rem; font-weight:bold;">'
                        f'Verified — No threats</span></a>'
                    )
                elif vt_result["status"] == "malicious":
                    vt_pill = (
                        f'<a href="{vt_link}" target="_blank" style="text-decoration:none;">'
                        f'<span style="background:#dc3545; color:white; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.78rem; font-weight:bold;">'
                        f'{vt_result["malicious"]}/{vt_result["total"]} engines detected threats</span></a>'
                    )
                else:
                    # "unknown" — file not in VT database
                    vt_pill = (
                        f'<a href="{vt_link}" target="_blank" style="text-decoration:none;">'
                        f'<span style="background:#555; color:#ccc; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.78rem;">'
                        f'Not in VirusTotal database</span></a>'
                    )
                file_rows += f"""
            <div style="display:flex; align-items:center; padding:6px 0;">
                <span style="{label_style}">VirusTotal</span>
                {vt_pill}
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
                        {"TRUSTED" if score >= 95 else f"{risk_level} RISK"}</text>
                </svg>
            </div>
            """, unsafe_allow_html=True)

            # Grouped score bars (4 categories instead of 7 modules)
            # Map module names to their results for easy lookup
            module_map = {m.module: m for m in result.modules}

            # Define the 4 groups: (label, list of module names, tooltip)
            groups = [
                ("Modifications", ["structure"], "Detects if the PDF was edited after creation"),
                ("Consistency", ["content"], "Checks dates, amounts and references for contradictions"),
                ("Tampering", ["images", "fonts", "visual", "metadata"], "Detects visual edits, font swaps and image manipulation"),
                ("Authenticity", ["external"], "Verifies the originator's identity (SIRET, VAT)<br><i style='color:#888;'>Internet connection needed</i>"),
            ]

            bars_html = ""
            for label, member_names, tooltip in groups:
                # Collect modules that actually ran for this group
                members = [module_map[name] for name in member_names if name in module_map]

                if not members:
                    # No modules ran for this group (e.g., external disabled)
                    # Show a greyed-out bar with N/A
                    bars_html += f'''
                    <div style="display:flex; align-items:center; margin-bottom:6px;">
                        <span style="width:100px; flex-shrink:0;">
                            <span class="tf-label" style="color:#555;">{label}<span class="tf-tip">{tooltip}</span></span>
                        </span>
                        <div style="flex:1; background:#1a1a1a; border-radius:4px;
                            height:10px; overflow:hidden;">
                        </div>
                        <span style="width:35px; text-align:right; font-size:0.8rem;
                            color:#555; margin-left:8px;">N/A</span>
                    </div>'''
                    continue

                # Weighted average score across member modules
                # Uses same weight logic as global scoring
                weighted_sum = 0.0
                weight_total = 0.0
                for m in members:
                    w = MODULE_WEIGHTS.get(m.module, DEFAULT_WEIGHT) * m.confidence
                    weighted_sum += m.score * w
                    weight_total += w
                group_score = round(weighted_sum / weight_total) if weight_total > 0 else 100

                if group_score >= 80:
                    bar_color = "#28a745"
                elif group_score >= 50:
                    bar_color = "#fd7e14"
                elif group_score >= 20:
                    bar_color = "#dc3545"
                else:
                    bar_color = "#721c24"

                bars_html += f'''
                <div style="display:flex; align-items:center; margin-bottom:6px;">
                    <span style="width:100px; flex-shrink:0;">
                        <span class="tf-label" style="color:#eee;">{label}<span class="tf-tip">{tooltip}</span></span>
                    </span>
                    <div style="flex:1; background:#2a2a2a; border-radius:4px;
                        height:10px; overflow:hidden;">
                        <div style="width:{group_score}%; height:100%;
                            background:{bar_color}; border-radius:4px;
                            transition:width 1s ease;"></div>
                    </div>
                    <span style="width:35px; text-align:right; font-size:0.8rem;
                        color:#eee; margin-left:8px; font-weight:bold;">{group_score}</span>
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

            # Summary block — verdict + flag list with pills
            _sev_colors = {
                "critical": "#721c24",
                "high": "#dc3545",
                "medium": "#fd7e14",
                "low": "#555",
            }

            flags_list_html = ""
            if all_flags:
                for flag in all_flags:
                    fc = _sev_colors.get(flag.severity, "#555")
                    flags_list_html += (
                        f'<div style="display:flex; align-items:center; gap:8px; padding:5px 0; '
                        f'border-bottom:1px solid #2a2a2a;">'
                        f'<span style="background:{fc}; color:white; font-size:0.62rem; '
                        f'font-weight:bold; padding:2px 0; border-radius:8px; width:58px; '
                        f'text-align:center; display:inline-block; flex-shrink:0;">{flag.severity.upper()}</span>'
                        f'<span style="color:#ccc; font-size:0.85rem;">{flag.message}</span>'
                        f'</div>'
                    )
            else:
                flags_list_html = ""

            st.markdown(f"""
            <div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">
                <span style="color:white; font-size:1.3rem; font-weight:bold; text-transform:uppercase;
                    letter-spacing:0.5px; display:block; text-align:center;">Summary</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">
                <p style="color:white; font-size:1.03rem; font-weight:bold; margin:0 0 0.8rem 0;">{summary.verdict}</p>
                {flags_list_html}
            </div>
            """, unsafe_allow_html=True)

        # Modification history (only shown if PDF has multiple versions)
        if history["version_count"] > 1 and history.get("diffs"):
            import html as html_mod  # for escaping user content

            with results_col:
                row_style = "display:flex; align-items:flex-start; padding:6px 0; border-bottom:1px solid #2a2a2a;"
                label_style = "color:#888; font-size:0.8rem; width:90px; flex-shrink:0;"
                value_style = "color:#eee; font-size:0.8rem;"
                max_lines = 5  # Max diff lines to show per version

                history_rows = ""

                # Version 1: original creation
                # Get the software used for version 1 from the first diff's from_tool
                first_software = ""
                if history["diffs"]:
                    first_software = history["diffs"][0].get("from_tool", "")

                if pdf_data.metadata.creation_date:
                    software_html = f'<div style="color:#888; font-size:0.7rem;">Producer: {html_mod.escape(first_software)}</div>' if first_software else ""
                    history_rows += (
                        f'<div style="{row_style}">'
                        f'<span style="{label_style}">Version 1</span>'
                        f'<div style="{value_style}">'
                        f'<div>{pdf_data.metadata.creation_date.strftime("%Y-%m-%d %H:%M")} — Original document</div>'
                        f'{software_html}'
                        f'</div></div>'
                    )

                # Each subsequent version
                for diff in history["diffs"]:
                    version_num = diff["to_version"]

                    # Software used for this version
                    version_software = diff.get("to_tool", "")

                    # Time info
                    time_info = ""
                    if pdf_data.metadata.creation_date and pdf_data.metadata.mod_date:
                        if version_num == history["version_count"]:
                            time_info = pdf_data.metadata.mod_date.strftime("%Y-%m-%d %H:%M")
                            delta = pdf_data.metadata.mod_date - pdf_data.metadata.creation_date
                            hours = delta.total_seconds() / 3600
                            if hours >= 24:
                                time_info += f" — +{int(hours // 24)}d {int(hours % 24)}h after creation"
                            elif hours >= 1:
                                time_info += f" — +{int(hours)}h{int((hours % 1) * 60)}min after creation"
                            else:
                                time_info += f" — +{int(hours * 60)}min after creation"

                    # Build change lines (limited to max_lines)
                    changes_html = ""
                    line_count = 0
                    total_lines = 0
                    if diff["changes"]:
                        for change in diff["changes"]:
                            for line in change.get("removed", []):
                                if line.strip():
                                    total_lines += 1
                                    if line_count < max_lines:
                                        safe = html_mod.escape(line[:80])
                                        changes_html += f'<div style="color:#e06060; font-size:0.75rem;">- {safe}</div>'
                                        line_count += 1
                            for line in change.get("added", []):
                                if line.strip():
                                    total_lines += 1
                                    if line_count < max_lines:
                                        safe = html_mod.escape(line[:80])
                                        changes_html += f'<div style="color:#60c060; font-size:0.75rem;">+ {safe}</div>'
                                        line_count += 1

                        if total_lines > max_lines:
                            remaining = total_lines - max_lines
                            changes_html += f'<div style="color:#888; font-size:0.7rem; font-style:italic;">... and {remaining} more lines</div>'
                    else:
                        changes_html = '<div style="color:#888; font-size:0.75rem;">Non-text changes</div>'

                    software_html = f'<div style="color:#888; font-size:0.7rem;">Editor: {html_mod.escape(version_software)}</div>' if version_software else ""
                    history_rows += (
                        f'<div style="{row_style}">'
                        f'<span style="{label_style}">Version {version_num}</span>'
                        f'<div style="{value_style}">'
                        f'<div>{time_info if time_info else "Unknown time"}</div>'
                        f'{software_html}'
                        f'{changes_html}'
                        f'</div></div>'
                    )

                st.markdown(
                    f'<div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">'
                    f'<span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;'
                    f' letter-spacing:0.5px;">Modification History</span>'
                    f'<hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">'
                    f'{history_rows}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Detected issues
        if all_flags:
            import json as json_mod
            import html as html_mod_flags

            st.markdown("---")
            st.markdown("## Detected Issues")

            severity_colors = {
                "critical": "#721c24",
                "high": "#dc3545",
                "medium": "#fd7e14",
                "low": "#555",
            }

            # Build all issues as a single HTML block with <details>/<summary>
            # This lets us put colored pills on the clickable line
            issues_html = ""
            for flag in all_flags:
                color = severity_colors.get(flag.severity, "#555")
                safe_msg = html_mod_flags.escape(flag.message)
                safe_code = html_mod_flags.escape(flag.code)

                # Details content — show as readable key/value rows
                details_content = (
                    f'<div style="color:#666; font-size:0.75rem; margin-top:6px; margin-bottom:8px;">{safe_code}</div>'
                )
                if flag.details:
                    detail_row = "display:flex; padding:5px 0; border-bottom:1px solid #222;"
                    detail_key = "color:#888; font-size:0.8rem; width:140px; flex-shrink:0;"
                    detail_val = "color:#ddd; font-size:0.8rem; word-break:break-word;"

                    rows_html = ""
                    for key, val in flag.details.items():
                        safe_key = html_mod_flags.escape(str(key).replace("_", " ").capitalize())
                        # Format value: lists as comma-separated, dicts as JSON, rest as string
                        if isinstance(val, list):
                            safe_val = html_mod_flags.escape(", ".join(str(v) for v in val))
                        elif isinstance(val, dict):
                            safe_val = html_mod_flags.escape(json_mod.dumps(val, ensure_ascii=False, default=str))
                        else:
                            safe_val = html_mod_flags.escape(str(val))
                        rows_html += (
                            f'<div style="{detail_row}">'
                            f'<span style="{detail_key}">{safe_key}</span>'
                            f'<span style="{detail_val}">{safe_val}</span>'
                            f'</div>'
                        )
                    details_content += (
                        f'<div style="background:#111; border-radius:6px; padding:8px 14px; margin-top:4px;">'
                        f'{rows_html}</div>'
                    )

                issues_html += (
                    f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                    f'border-radius:8px; padding:0; margin-bottom:6px;">'
                    f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                    f'display:flex; align-items:center; gap:10px;">'
                    f'<span style="background:{color}; color:white; font-size:0.68rem; '
                    f'font-weight:bold; padding:2px 0; border-radius:10px; '
                    f'width:70px; text-align:center; flex-shrink:0; '
                    f'display:inline-block;">{flag.severity.upper()}</span>'
                    f'<span style="color:#eee; font-size:0.85rem;">{safe_msg}</span>'
                    f'</summary>'
                    f'<div style="padding:4px 14px 14px 14px; border-top:1px solid #2a2a2a;">'
                    f'{details_content}'
                    f'</div></details>'
                )

            st.markdown(issues_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## More Details")

        # Full modification history (collapsible)
        if history["version_count"] > 1 and history.get("diffs"):
            import html as html_mod

            # Reusable styles
            _card = ("background:#111; border:1px solid #2a2a2a; border-radius:8px; "
                     "padding:14px 18px; margin-bottom:10px;")
            _vlabel = "font-size:0.85rem; font-weight:bold; color:white; margin:0 0 8px 0;"
            _mrow = "display:flex; gap:6px; align-items:center; margin-bottom:4px;"
            _mkey = "color:#888; font-size:0.78rem; width:70px; flex-shrink:0;"
            _mval = "color:#ccc; font-size:0.78rem;"
            _diff = ("background:#0a0a0a; border-radius:6px; padding:10px 14px; "
                     "margin-top:10px; font-family:monospace; font-size:0.78rem; "
                     "line-height:1.6; overflow-x:auto;")
            _badge = ("display:inline-block; background:#2a2a2a; color:#aaa; "
                      "font-size:0.7rem; padding:2px 8px; border-radius:10px; "
                      "margin:2px 4px 2px 0;")
            _arrow = "text-align:center; color:#555; font-size:1.2rem; margin:4px 0;"

            # Build the full content as one HTML string
            hist_content = ""

            # Version 1: Original
            first_sw = history["diffs"][0].get("from_tool", "") if history["diffs"] else ""
            v1_meta = ""
            if pdf_data.metadata.creation_date:
                v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">Date</span>'
                            f'<span style="{_mval}">{pdf_data.metadata.creation_date.strftime("%Y-%m-%d %H:%M")}</span></div>')
            if first_sw:
                v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">Producer</span>'
                            f'<span style="{_mval}">{html_mod.escape(first_sw)}</span></div>')

            hist_content += (
                f'<div style="{_card}">'
                f'<p style="{_vlabel}">Version 1 '
                f'<span style="background:#28a745; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px; margin-left:6px;">ORIGINAL</span></p>'
                f'{v1_meta}</div>'
            )

            # Each subsequent version
            for diff in history["diffs"]:
                to_v = diff["to_version"]
                from_sw = diff.get("from_tool", "")
                to_sw = diff.get("to_tool", "")

                hist_content += f'<div style="{_arrow}">&#9660;</div>'

                v_meta = ""
                if to_v == history["version_count"] and pdf_data.metadata.mod_date and pdf_data.metadata.creation_date:
                    delta = pdf_data.metadata.mod_date - pdf_data.metadata.creation_date
                    hours = delta.total_seconds() / 3600
                    if hours >= 24:
                        delta_str = f"+{int(hours // 24)}d {int(hours % 24)}h"
                    elif hours >= 1:
                        delta_str = f"+{int(hours)}h{int((hours % 1) * 60)}min"
                    else:
                        delta_str = f"+{int(hours * 60)}min"
                    v_meta += (f'<div style="{_mrow}"><span style="{_mkey}">Date</span>'
                               f'<span style="{_mval}">{pdf_data.metadata.mod_date.strftime("%Y-%m-%d %H:%M")} '
                               f'<span style="color:#e06060;">({delta_str} after creation)</span></span></div>')

                if to_sw and to_sw != from_sw:
                    v_meta += (f'<div style="{_mrow}"><span style="{_mkey}">Editor</span>'
                               f'<span style="{_mval}">{html_mod.escape(to_sw)}</span></div>')

                diff_html = ""
                if diff["changes"]:
                    for change in diff["changes"]:
                        diff_html += f'<div style="color:#888; font-size:0.7rem; margin-bottom:4px;">Page {change["page"]}</div>'
                        for line in change.get("removed", []):
                            if line.strip():
                                diff_html += f'<div style="color:#e06060;">- {html_mod.escape(line)}</div>'
                        for line in change.get("added", []):
                            if line.strip():
                                diff_html += f'<div style="color:#60c060;">+ {html_mod.escape(line)}</div>'
                    diff_html = f'<div style="{_diff}">{diff_html}</div>'
                else:
                    diff_html = '<div style="color:#888; font-size:0.78rem; font-style:italic; margin-top:8px;">No text changes detected</div>'

                obj_html = ""
                obj_changes = diff.get("object_changes", [])
                if obj_changes:
                    by_type: dict[str, list] = {}
                    for oc in obj_changes:
                        by_type.setdefault(oc.get("type", "unknown"), []).append(oc)
                    badges = "".join(f'<span style="{_badge}">{t} ({len(items)})</span>' for t, items in by_type.items())
                    obj_html = f'<div style="margin-top:10px;"><span style="color:#888; font-size:0.7rem;">Modified objects: </span>{badges}</div>'

                hist_content += (
                    f'<div style="{_card}">'
                    f'<p style="{_vlabel}">Version {to_v} '
                    f'<span style="background:#dc3545; color:white; font-size:0.68rem; font-weight:bold; '
                    f'padding:2px 10px; border-radius:10px; margin-left:6px;">MODIFIED</span></p>'
                    f'{v_meta}{diff_html}{obj_html}</div>'
                )

            version_count = history["version_count"]
            st.markdown(
                f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">Full Modification History</span>'
                f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{version_count} versions</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                f'{hist_content}'
                f'</div></details>',
                unsafe_allow_html=True,
            )

        # Raw analysis data (collapsible)
        import json as json_mod_raw
        import html as html_mod_raw

        raw_content = ""
        for module in result.modules:
            # Module header with score pill
            mod_score = module.score
            if mod_score >= 80:
                sc_color = "#28a745"
            elif mod_score >= 50:
                sc_color = "#fd7e14"
            elif mod_score >= 20:
                sc_color = "#dc3545"
            else:
                sc_color = "#721c24"

            mod_name = module.module.capitalize()
            flag_count = len(module.flags)

            # Module card with full raw data
            mod_weight = MODULE_WEIGHTS.get(module.module, DEFAULT_WEIGHT)

            flags_html = ""
            if module.flags:
                for f in module.flags:
                    f_color = severity_colors.get(f.severity, "#555")
                    safe_code = html_mod_raw.escape(f.code)
                    safe_msg = html_mod_raw.escape(f.message)

                    # Flag details as key/value rows
                    details_html = ""
                    if f.details:
                        _drow = "display:flex; padding:3px 0; border-bottom:1px solid #151515;"
                        _dkey = "color:#777; font-size:0.73rem; width:130px; flex-shrink:0;"
                        _dval = "color:#bbb; font-size:0.73rem; word-break:break-word;"

                        detail_rows = ""
                        for dk, dv in f.details.items():
                            safe_dk = html_mod_raw.escape(str(dk).replace("_", " ").capitalize())
                            if isinstance(dv, list):
                                safe_dv = html_mod_raw.escape(", ".join(str(v) for v in dv))
                            elif isinstance(dv, dict):
                                safe_dv = html_mod_raw.escape(json_mod_raw.dumps(dv, ensure_ascii=False, default=str))
                            elif isinstance(dv, bool):
                                safe_dv = f'<span style="color:{"#60c060" if dv else "#e06060"};">{"true" if dv else "false"}</span>'
                            else:
                                safe_dv = html_mod_raw.escape(str(dv))
                            detail_rows += (
                                f'<div style="{_drow}">'
                                f'<span style="{_dkey}">{safe_dk}</span>'
                                f'<span style="{_dval}">{safe_dv}</span></div>'
                            )
                        details_html = (
                            f'<div style="background:#0a0a0a; border-radius:4px; '
                            f'padding:6px 12px; margin:6px 0 0 0;">{detail_rows}</div>'
                        )

                    flags_html += (
                        f'<div style="padding:8px 0; border-bottom:1px solid #1a1a1a;">'
                        f'<div style="display:flex; align-items:center; gap:8px;">'
                        f'<span style="background:{f_color}; color:white; font-size:0.62rem; '
                        f'font-weight:bold; padding:1px 0; border-radius:8px; width:58px; '
                        f'text-align:center; display:inline-block; flex-shrink:0;">{f.severity.upper()}</span>'
                        f'<span style="color:#ccc; font-size:0.78rem;">{safe_msg}</span>'
                        f'</div>'
                        f'<div style="color:#666; font-size:0.72rem; margin-top:3px; padding-left:66px;">{safe_code}</div>'
                        f'{f"<div style=padding-left:66px;>{details_html}</div>" if details_html else ""}'
                        f'</div>'
                    )
            else:
                flags_html = '<div style="color:#666; font-size:0.78rem; padding:4px 0;">No flags</div>'

            raw_content += (
                f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                f'padding:12px 16px; margin-bottom:8px;">'
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
                f'<span style="color:white; font-size:0.88rem; font-weight:bold;">{mod_name}</span>'
                f'<span style="background:{sc_color}; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{mod_score}/100</span>'
                f'</div>'
                f'<div style="display:flex; gap:16px; color:#888; font-size:0.75rem; margin-bottom:8px; '
                f'padding-bottom:8px; border-bottom:1px solid #1a1a1a;">'
                f'<span>confidence: {module.confidence:.0%}</span>'
                f'<span>weight: {mod_weight}</span>'
                f'<span>flags: {len(module.flags)}</span>'
                f'</div>'
                f'{flags_html}'
                f'</div>'
            )

        # Analysis Details — collapsible block with per-module raw data
        module_count = len(result.modules)
        st.markdown(
            f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
            f'border-radius:8px; padding:0; margin-bottom:8px;">'
            f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
            f'display:flex; align-items:center; gap:10px;">'
            f'<span style="color:#eee; font-size:0.85rem;">Analysis Details</span>'
            f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
            f'padding:2px 10px; border-radius:10px;">{module_count} modules</span>'
            f'</summary>'
            f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
            f'{raw_content}'
            f'</div></details>',
            unsafe_allow_html=True,
        )

        # More Details — Entities block (verified companies from external module)
        external_module = next((m for m in result.modules if m.module == "external"), None)
        if external_module and external_module.details.get("verified_companies"):
            companies = external_module.details["verified_companies"]
            import html as html_mod_ent

            entities_html = ""
            for comp in companies:
                name = html_mod_ent.escape(comp.get("name") or "Unknown")
                siren = comp.get("siren") or ""
                siret = comp.get("siret") or ""
                address = html_mod_ent.escape(comp.get("address") or "")
                status = comp.get("status", "")
                creation = comp.get("creation_date") or ""

                # Status pill
                if status == "active":
                    status_pill = ('<span style="background:#28a745; color:white; font-size:0.62rem; '
                                   'font-weight:bold; padding:2px 8px; border-radius:8px;">Active</span>')
                elif status == "closed":
                    status_pill = ('<span style="background:#dc3545; color:white; font-size:0.62rem; '
                                   'font-weight:bold; padding:2px 8px; border-radius:8px;">Closed</span>')
                else:
                    status_pill = ""

                _row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
                _key = "color:#888; font-size:0.75rem; width:80px; flex-shrink:0;"
                _val = "color:#ccc; font-size:0.75rem;"

                rows = ""
                if siren:
                    rows += f'<div style="{_row}"><span style="{_key}">SIREN</span><span style="{_val}">{siren}</span></div>'
                if siret:
                    rows += f'<div style="{_row}"><span style="{_key}">SIRET</span><span style="{_val}">{siret}</span></div>'
                if address and address != "None, None None":
                    rows += f'<div style="{_row}"><span style="{_key}">Address</span><span style="{_val}">{address}</span></div>'
                if creation:
                    rows += f'<div style="{_row}"><span style="{_key}">Created</span><span style="{_val}">{creation}</span></div>'

                entities_html += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
                    f'<span style="color:white; font-size:0.88rem; font-weight:bold;">{name}</span>'
                    f'{status_pill}'
                    f'</div>'
                    f'{rows}'
                    f'</div>'
                )

            entity_count = len(companies)
            st.markdown(
                f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">Entities</span>'
                f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{entity_count} entities</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                f'{entities_html}'
                f'</div></details>',
                unsafe_allow_html=True,
            )

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
