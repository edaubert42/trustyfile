"""
TrustyFile - Document Fraud Detection

A Streamlit web application for analyzing invoices and documents
to detect signs of manipulation or fraud.

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import re
from pathlib import Path


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
        import requests
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
    import fitz
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
    import fitz
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TrustyFile - Document Fraud Detector",
    page_icon="static/upload_icon_medium.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

# =============================================================================
# SIDEBAR
# =============================================================================

# Default settings (sidebar removed)
enable_external = True
enable_qr = True

# IBAN verification: API key from Streamlit secrets or environment variable
# In Streamlit Cloud: add IBAN_API_KEY in Settings > Secrets
# Locally: set IBAN_API_KEY environment variable or add to .streamlit/secrets.toml
try:
    iban_api_key = st.secrets.get("IBAN_API_KEY")
except Exception:
    iban_api_key = None
if not iban_api_key:
    iban_api_key = os.environ.get("IBAN_API_KEY")
enable_iban_check = bool(iban_api_key)


# =============================================================================
# MAIN CONTENT
# =============================================================================

import base64

@st.cache_data
def _load_icon_b64():
    with open("static/upload_icon_medium.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

_logo_b64 = _load_icon_b64()
_upload_b64 = _logo_b64  # same file, no need to read twice

# Inject meta tags + hide content until loaded via components.html
components.html("""
<script>
    // Access parent document (Streamlit main frame)
    const doc = window.parent.document;
    if (!doc.querySelector('meta[http-equiv="Content-Security-Policy"]')) {
        const meta = doc.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = 'upgrade-insecure-requests';
        doc.head.appendChild(meta);
    }
    if (!doc.querySelector('meta[name="viewport"]')) {
        const vp = doc.createElement('meta');
        vp.name = 'viewport';
        vp.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0';
        doc.head.appendChild(vp);
    }
    if (!doc.querySelector('script[data-website-id="50120522-6e98-4433-9238-81c6ba8358c1"]')) {
        const umami = doc.createElement('script');
        umami.defer = true;
        umami.src = 'https://cloud.umami.is/script.js';
        umami.dataset.websiteId = '50120522-6e98-4433-9238-81c6ba8358c1';
        doc.head.appendChild(umami);
    }

    // IMMEDIATELY hide content in parent document to prevent flash
    if (!doc.getElementById('tf-hide-flash')) {
        const style = doc.createElement('style');
        style.id = 'tf-hide-flash';
        style.textContent = 'section.stMain { opacity: 0 !important; }';
        doc.head.appendChild(style);
    }

    // Reveal content after Streamlit finishes rendering
    const reveal = () => {
        const s = doc.getElementById('tf-hide-flash');
        if (s) s.textContent = 'section.stMain { opacity: 1 !important; transition: opacity 0.15s ease-in; }';
    };
    if (window.parent.requestIdleCallback) {
        window.parent.requestIdleCallback(reveal, { timeout: 800 });
    } else {
        setTimeout(reveal, 400);
    }
</script>
""", height=0)

# All CSS in ONE st.markdown to avoid empty flex gap elements
st.markdown(f"""
<style>
    /* Hide the native Streamlit header entirely */
    header[data-testid="stHeader"],
    header.stAppHeader {{
        display: none !important;
    }}
    /* Custom header bar */
    .custom-header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;
        background: #0e1117;
        border-bottom: 1px solid #2a2a3e;
        display: flex;
        align-items: center;
        padding: 0 24px;
        z-index: 999;
    }}
    .custom-header a.home-link {{
        display: flex;
        align-items: center;
        text-decoration: none;
        cursor: pointer;
    }}
    .custom-header img {{
        height: 36px;
        width: 36px;
        object-fit: contain;
        margin-right: 6px;
    }}
    .custom-header .title {{
        font-size: 1.4rem;
        font-weight: bold;
        color: white;
        margin-right: 12px;
    }}
    .custom-header .subtitle {{
        font-size: 0.9rem;
        color: #888;
        font-weight: normal;
    }}

    #MainMenu {{display: none !important;}}
    footer {{display: none !important;}}
    .stAppDeployButton {{display: none !important;}}
    div[data-testid="stDecoration"] {{display: none !important;}}
    div[data-testid="stStatusWidget"] {{display: none !important;}}

    /* Single scroll context: only section.stMain scrolls */
    html, body, [data-testid="stAppViewContainer"], .main {{
        overflow: visible !important;
        height: auto !important;
        scroll-snap-type: none !important;
        scroll-padding-top: 0 !important;
        -webkit-overflow-scrolling: touch !important;
    }}
    section.stMain {{
        scroll-snap-type: none !important;
        scroll-padding-top: 0 !important;
        scroll-behavior: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }}

    /* Content starts right below the fixed header (56px) */
    .block-container {{padding-top: 64px !important; margin-top: 0 !important;}}
    .stMainBlockContainer {{padding-top: 64px !important; margin-top: 0 !important;}}
    .stAppViewBlockContainer {{padding-top: 64px !important; margin-top: 0 !important;}}
    section.stMain .block-container {{padding-top: 64px !important; margin-top: 0 !important; z-index: 1;}}
    section.stMain {{padding-top: 0 !important; margin-top: 0 !important;}}
    [data-testid="stAppViewContainer"] {{padding-top: 0 !important; margin-top: 0 !important;}}
    [data-testid="stSidebarHeader"] {{height: 0 !important;}}

    /* Tighten h2 section headers (Detected Issues, More Details) */
    h2 {{
        margin-top: -0.5rem !important;
        padding-top: 0 !important;
    }}

    /* Reduce gap between Preview title and image below */
    .preview-title {{
        margin-top: 1rem !important;
        margin-bottom: 0.7rem !important;
    }}
    .preview-title + div {{
        margin-top: 0 !important;
    }}

    /* Desktop: reverse column order so preview stays on left */
    @media (min-width: 769px) {{
        [data-testid="stColumns"] {{
            flex-direction: row-reverse !important;
        }}
    }}

    /* Prevent horizontal overflow without affecting vertical scroll */
    .block-container, [data-testid="stAppViewContainer"] {{
        overflow-x: clip !important;
    }}

    /* Mobile responsive adjustments */
    @media (max-width: 480px) {{
        .block-container,
        .stMainBlockContainer,
        .stAppViewBlockContainer {{
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
            max-width: 100vw !important;
            box-sizing: border-box !important;
        }}
        .block-container,
        .stMainBlockContainer,
        .stAppViewBlockContainer,
        section.stMain .block-container {{
            padding-top: 64px !important;
        }}
        #landing-content p {{
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            font-size: 2.5rem !important;
        }}
        #landing-tagline p {{
            font-size: 0.85rem !important;
            margin-top: 0.8rem !important;
        }}
        [data-testid="stFileUploader"] {{
            max-width: 65% !important;
            margin: 0 auto !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            max-width: 100% !important;
            min-height: 0 !important;
            padding: 0.5rem 0.8rem 0.3rem 0.8rem !important;
            box-sizing: border-box !important;
            margin: 0 auto !important;
            height: auto !important;
            max-height: fit-content !important;
        }}
        [data-testid="stFileUploader"] > div {{
            height: auto !important;
        }}
        /* Smaller upload icon on mobile */
        [data-testid="stFileUploaderDropzoneInstructions"]::before {{
            width: 80px !important;
            height: 80px !important;
            margin-bottom: 0.5rem !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"]::after {{
            font-size: 0.9rem !important;
        }}
        /* Shrink content that may overflow */
        [data-testid="stImage"],
        [data-testid="stExpander"],
        [data-testid="stVerticalBlock"] {{
            max-width: 100% !important;
            overflow-x: clip !important;
        }}
        img {{
            max-width: 100% !important;
            height: auto !important;
        }}
        /* On mobile: hide the entire preview column (PDF + file properties) */
        [data-testid="stColumns"] [data-testid="stColumn"]:last-child {{
            display: none !important;
        }}
    }}
    .score-box {{
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    .score-low {{ background-color: #198754; border: 2px solid #146c43; color: white; }}
    .score-medium {{ background-color: #ffb627; border: 2px solid #dc6a10; color: white; }}
    .score-high {{ background-color: #ff6b35; border: 2px solid #e55a25; color: white; }}
    .score-critical {{ background-color: #ef233c; border: 2px solid #4a1119; color: white; }}
    .flag-critical {{ color: #ef233c; font-weight: bold; }}
    .flag-high {{ color: #ff6b35; }}
    .flag-medium {{ color: #ffb627; }}
    .flag-low {{ color: #6c757d; }}

    /* Label with hover tooltip */
    .tf-label {{
        position: relative;
        cursor: help;
        font-size: 0.8rem;
    }}
    .tf-label .tf-tip {{
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 100%;
        left: 0;
        transform: none;
        margin-top: 4px;
        background: #333;
        color: #ccc;
        font-size: 0.7rem;
        padding: 4px 10px;
        border-radius: 6px;
        white-space: nowrap;
        transition: opacity 0.2s;
        pointer-events: none;
        z-index: 10;
    }}
    .tf-label:hover .tf-tip {{
        visibility: visible;
        opacity: 1;
    }}

    /* Styled expanders — dark card look */
    [data-testid="stExpander"] {{
        background: #1a1a2e;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        margin-bottom: 8px;
    }}
    [data-testid="stExpander"] summary {{
        padding: 10px 14px;
        font-size: 0.88rem;
        font-weight: bold;
        color: #eee;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: white;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        border-top: 1px solid #2a2a2a;
        padding: 14px;
    }}

    /* Landing page: style the dropzone as a dashed border zone */
    [data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed #444 !important;
        border-radius: 16px !important;
        background: transparent !important;
        padding: 2rem 2rem 1rem 2rem !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: border-color 0.3s, background 0.3s;
        max-width: 420px !important;
        min-height: 340px !important;
        margin: 0 auto !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: #888 !important;
        background: rgba(255,255,255,0.03) !important;
    }}
    /* Hide the label */
    [data-testid="stFileUploader"] label {{
        display: none !important;
    }}
    /* Hide ALL default dropzone content (text, icon, small print) */
    [data-testid="stFileUploaderDropzoneInstructions"] > div {{
        display: none !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] svg {{
        display: none !important;
    }}
    /* Restyle the native browse button as "Browse file" */
    [data-testid="stFileUploaderDropzone"] button {{
        order: 3 !important;
        border: 1.5px solid #666 !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.5rem !important;
        color: #bbb !important;
        background: transparent !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        margin-top: 0 !important;
        margin-bottom: 2rem !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        border-color: #999 !important;
        color: #eee !important;
    }}
    /* Hide the default button text and replace with "Browse file" */
    [data-testid="stFileUploaderDropzone"] button {{
        font-size: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button::after {{
        content: "Browse file";
        font-size: 0.9rem;
    }}
    /* Make instructions a flex column so ::before (icon) is on top, ::after (text) below */
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
    }}
    /* Custom upload icon */
    [data-testid="stFileUploaderDropzoneInstructions"]::before {{
        content: "";
        display: block;
        width: 140px;
        height: 140px;
        margin: 0 auto 1rem auto;
        background: url("data:image/png;base64,{_upload_b64}") center/contain no-repeat;
        opacity: 0.85;
        order: 1;
    }}
    /* "Drop your PDF here" + "or" text */
    [data-testid="stFileUploaderDropzoneInstructions"]::after {{
        content: "Drop your PDF here\\A or";
        display: block;
        white-space: pre;
        text-align: center;
        order: 2;
        color: #bbb;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        line-height: 2;
    }}
</style>
<div class="custom-header">
    <a class="home-link" href="/" target="_parent">
        <img src="data:image/png;base64,{_logo_b64}" alt="logo">
        <span class="title">TrustyFile</span>
    </a>
    <span class="subtitle">Document Fraud Detector</span>
</div>
""", unsafe_allow_html=True)

# File uploader — always rendered at top level (outside landing columns)
uploaded_file = st.file_uploader(
    "Choose file",
    type=["pdf"],
    label_visibility="collapsed",
)

# Landing page content — only rendered when no file is uploaded
if uploaded_file is None:
    col_left, col_upload, col_right = st.columns([1, 2, 1])
    with col_upload:
        st.markdown("""
        <div id="landing-content">
            <p style="color:white; font-size:2.2rem; text-align:center; margin:0 auto 1.5rem auto; max-width:600px; font-weight:bold; letter-spacing:0.5px;">
                Don't trust. Verify.
            </p>
        </div>
        <div id="landing-tagline">
            <p style="color:white; font-size:1.2rem; text-align:center; margin:1.2rem auto 0.5rem auto; max-width:500px; line-height:1.5;">
                TrustyFile spots clever edits and hidden changes inside documents.
            </p>
        </div>
        <div id="landing-disclaimer">
            <p style="color:#666; font-size:0.75rem; text-align:center; margin-top:0.8rem;">
                Your document is analyzed locally and never stored. Only the file hash and company identifiers
                may be sent to external APIs (VirusTotal, INSEE, EU DSS) for verification.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Hide landing content and minimize uploader when a file is uploaded
if uploaded_file is not None:
    st.markdown("""<style>
        /* Hide the dropzone, keep only the filename bar */
        [data-testid="stFileUploaderDropzone"] {
            display: none !important;
        }
        /* File uploader: no extra space, visible file icon */
        [data-testid="stFileUploader"] {
            margin-bottom: 0 !important;
            overflow: visible !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            overflow: visible !important;
        }
        [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        /* Tight gap between all elements */
        .block-container > [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
    </style>""", unsafe_allow_html=True)

if uploaded_file is not None:
    # Lazy imports: only load heavy analysis modules when a file is uploaded
    from src.analyzer import TrustyFileAnalyzer
    from src.scoring import collect_all_flags, count_flags_by_severity, MODULE_WEIGHTS, DEFAULT_WEIGHT
    from src.summary import generate_rich_summary
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.structure import get_modification_history
    from src.modules.twod_doc import scan_pdf_for_2d_doc, extract_verified_data, PYLIBDMTX_AVAILABLE

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
                enable_iban_check=enable_iban_check,
                iban_api_key=iban_api_key,
            )
            result = analyzer.analyze(tmp_path)
            summary = generate_rich_summary(result)

        # Scan for 2D-DOC barcodes (French government signed barcodes)
        twod_doc_results = []
        if PYLIBDMTX_AVAILABLE:
            with st.spinner("Scanning for 2D-DOC barcodes..."):
                twod_doc_results = scan_pdf_for_2d_doc(tmp_path)

        # Get page count for navigation
        page_count = get_pdf_page_count(tmp_path)

        # Display results with PDF preview
        st.markdown('<hr style="margin:0 0 0.3rem 0;">', unsafe_allow_html=True)

        # Two-column layout: Preview | Results
        results_col, preview_col = st.columns([3, 2])

        # =================================================================
        # LEFT COLUMN: PDF Preview
        # =================================================================
        with preview_col:
            st.markdown("""<div class="preview-title"><span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;
                letter-spacing:0.5px;">Preview</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:2px 0 0 0;"></div>""", unsafe_allow_html=True)
            # Page navigation if multiple pages
            if page_count > 1:
                st.selectbox(
                    "Page",
                    range(1, page_count + 1),
                    format_func=lambda x: f"Page {x} / {page_count}",
                    key="page_selector",
                    label_visibility="collapsed"
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

            # Extract stated date from first page only (letter/invoice date)
            # The establishment date is always on the first page
            from src.modules.content import extract_dates_from_text
            first_page_text = pdf_data.text_by_page[0] if pdf_data.text_by_page else ""
            first_page_dates = extract_dates_from_text(first_page_text)
            doc_dates = [d for d in first_page_dates if d.date_type in ("document", "invoice")]
            stated_date = doc_dates[0].date if doc_dates else None

            def red_pill(text, tooltip=""):
                title_attr = f' title="{tooltip}"' if tooltip else ""
                return (f'<span{title_attr} style="background:#ef233c; color:white; padding:2px 8px; '
                        f'border-radius:10px; font-size:0.7rem; font-weight:bold; '
                        f'margin-left:8px; white-space:nowrap; cursor:default;">{text}</span>')

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

            # Compute pills for date anomalies
            mod_pill = ""
            stated_pill = ""

            # Pill on modification date if gap > 1 day from creation
            if meta.creation_date and meta.mod_date:
                mod_gap = abs((meta.mod_date - meta.creation_date).total_seconds())
                if mod_gap > 86400:
                    days = int(mod_gap // 86400)
                    mod_pill = red_pill(f"+{days}d", f"Modified {days} days after creation")

            # Pill on stated date vs creation date if gap > 7 days
            if stated_date and meta.creation_date:
                stated_gap = abs((stated_date - meta.creation_date).days)
                if stated_gap > 7:
                    stated_pill = red_pill(f"+{stated_gap}d", f"Created {stated_gap} days after the date stated in the document")

            def label_with_tip(label, tip):
                return (f'<span class="tf-label" style="{label_style}">'
                        f'{label}<span class="tf-tip">{tip}</span></span>')

            stated_row = ""
            if stated_date:
                stated_row = f"""
            <div style="{row_style}">
                {label_with_tip("Stated date", "Date written in the document text (e.g. &quot;Paris, le 06/01/2026&quot;)")}
                <span style="{value_style} flex-shrink:0;">{stated_date.strftime("%Y-%m-%d")}</span>{stated_pill}
            </div>
            """

            file_rows = f"""
            <div style="{row_style}">
                <span style="{label_style}">Name</span>
                <span style="{value_style}" title="{uploaded_file.name}">{uploaded_file.name}</span>
            </div>
            {stated_row}
            <div style="{row_style}">
                {label_with_tip("Creation date", "Date the PDF file was originally generated by the software")}
                <span style="{value_style}" title="{format_date(meta.creation_date)}">{format_date(meta.creation_date)}</span>
            </div>
            <div style="{row_style}">
                {label_with_tip("Modification date", "Date the PDF was last saved or edited.<br>Should be identical to creation date for untouched documents")}
                <span style="{value_style} flex-shrink:0;" title="{format_date(meta.mod_date)}">{format_date(meta.mod_date)}</span>{mod_pill}
            </div>
            <div style="{row_style}">
                <span style="{label_style}">Producer</span>
                <span style="{value_style}" title="{producer}">{producer}</span>
            </div>
            """

            # Show Creator if different from Producer (often contains device/app info)
            creator = meta.creator or ""
            if creator and creator.lower() != (meta.producer or "").lower():
                file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">Creator</span>
                <span style="{value_style}" title="{creator}">{creator}</span>
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
                        f'<span style="background:#ef233c; color:white; padding:3px 10px; '
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
            <div style="margin-top:0.5rem;">
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
                gauge_color = "#ffb627"
            elif score >= 20:
                gauge_color = "#ef233c"
            else:
                gauge_color = "#ef233c"

            # SVG circular gauge (full donut ring)
            # Uses a circle with stroke-dasharray to fill proportionally
            # The circle starts at the top (rotated -90°)
            radius = 80
            circumference = 2 * 3.14159 * radius  # full circle
            filled = circumference * (score / 100)
            size = 220
            center = size // 2

            st.markdown(f"""
            <div style="text-align: center; padding: 0;">
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
                    bar_color = "#ffb627"
                elif group_score >= 20:
                    bar_color = "#ef233c"
                else:
                    bar_color = "#ef233c"

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
                pills_html += f'<span style="background:#ef233c; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["critical"]} Critical</span> '
            if counts["high"]:
                pills_html += f'<span style="background:#ff6b35; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["high"]} High</span> '
            if counts["medium"]:
                pills_html += f'<span style="background:#ffb627; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["medium"]} Medium</span> '
            if counts["low"]:
                pills_html += f'<span style="background:#555; color:#ccc; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["low"]} Low</span> '

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
                "critical": "#ef233c",
                "high": "#ff6b35",
                "medium": "#ffb627",
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
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:0 0 0.8rem 0;">
                {flags_list_html}
            </div>
            <div style="margin-bottom:0.5rem;"></div>
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
                            elif int(hours * 60) > 0:
                                time_info += f" — +{int(hours * 60)}min after creation"
                            else:
                                time_info += f" — +{int(delta.total_seconds())}s after creation"

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
                "critical": "#ef233c",
                "high": "#ff6b35",
                "medium": "#ffb627",
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
                    elif int(hours * 60) > 0:
                        delta_str = f"+{int(hours * 60)}min"
                    else:
                        delta_str = f"+{int(delta.total_seconds())}s"
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
                    f'<span style="background:#ef233c; color:white; font-size:0.68rem; font-weight:bold; '
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
                sc_color = "#ffb627"
            elif mod_score >= 20:
                sc_color = "#ef233c"
            else:
                sc_color = "#ef233c"

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

        # More Details — Entities block (verified companies + unverified identifiers)
        external_module = next((m for m in result.modules if m.module == "external"), None)
        verified_companies = []
        if external_module and external_module.details.get("verified_companies"):
            verified_companies = external_module.details["verified_companies"]

        # Extract identifiers from full document text
        from src.modules.content import extract_siret, extract_siren, extract_french_vat, extract_rcs
        import html as html_mod_ent

        full_text = "\n".join(pdf_data.text_by_page) if pdf_data.text_by_page else ""
        found_sirets = extract_siret(full_text)
        found_sirens = extract_siren(full_text)
        found_vats = extract_french_vat(full_text)
        found_rcs = extract_rcs(full_text)

        # Collect identifiers already covered by verified companies
        # Include SIRENs derived from verified SIRETs to avoid duplicates
        verified_sirets = {c.get("siret", "") for c in verified_companies}
        verified_sirens = {c.get("siren", "") for c in verified_companies}
        for vs in list(verified_sirets):
            if vs and len(vs) == 14:
                verified_sirens.add(vs[:9])

        # Build entities HTML
        entities_html = ""
        entity_count = 0

        # 1) Verified companies (with Active/Closed pills)
        for comp in verified_companies:
            name = html_mod_ent.escape(comp.get("name") or "Unknown")
            siren = comp.get("siren") or ""
            siret = comp.get("siret") or ""
            address = html_mod_ent.escape(comp.get("address") or "")
            status = comp.get("status", "")
            creation = comp.get("creation_date") or ""

            if status == "active":
                status_pill = ('<span style="background:#28a745; color:white; font-size:0.62rem; '
                               'font-weight:bold; padding:2px 8px; border-radius:8px;">Active</span>')
            elif status == "closed":
                status_pill = ('<span style="background:#ef233c; color:white; font-size:0.62rem; '
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
            entity_count += 1

        # 2) Unverified entities: group all identifiers by SIREN into one card per entity
        not_verified_pill = ('<span style="background:#555; color:#ccc; font-size:0.62rem; '
                             'font-weight:bold; padding:2px 8px; border-radius:8px;">Not verified</span>')
        _row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
        _key = "color:#888; font-size:0.75rem; width:80px; flex-shrink:0;"
        _val = "color:#ccc; font-size:0.75rem;"

        # Build one entity per unique SIREN, attaching all related identifiers
        # Key = SIREN (9 digits), Value = dict with all identifiers
        unverified_by_siren: dict[str, dict] = {}

        for siret, is_valid, context in found_sirets:
            siren_key = siret[:9]
            if siren_key in verified_sirens or siret in verified_sirets:
                continue
            if siren_key not in unverified_by_siren:
                unverified_by_siren[siren_key] = {"siren": siren_key, "siret": None, "vat": None, "rcs": None}
            unverified_by_siren[siren_key]["siret"] = siret

        for siren, is_valid, context in found_sirens:
            if siren in verified_sirens:
                continue
            if siren not in unverified_by_siren:
                unverified_by_siren[siren] = {"siren": siren, "siret": None, "vat": None, "rcs": None}

        for vat, is_valid, context in found_vats:
            # Extract SIREN from VAT: FR + 2 check digits + 9 digit SIREN
            vat_siren = vat[4:] if len(vat) == 13 else None
            if vat_siren and vat_siren in unverified_by_siren:
                unverified_by_siren[vat_siren]["vat"] = vat
            elif vat_siren and vat_siren not in verified_sirens:
                unverified_by_siren[vat_siren] = {"siren": vat_siren, "siret": None, "vat": vat, "rcs": None}

        for rcs_string, context in found_rcs:
            # Try to match RCS number (9 digits) to a known SIREN
            rcs_digits = re.sub(r"\D", "", rcs_string)
            rcs_siren = rcs_digits[:9] if len(rcs_digits) >= 9 else None
            if rcs_siren and rcs_siren in unverified_by_siren:
                unverified_by_siren[rcs_siren]["rcs"] = rcs_string
            elif rcs_siren and rcs_siren not in verified_sirens:
                unverified_by_siren[rcs_siren] = {"siren": rcs_siren, "siret": None, "vat": None, "rcs": rcs_string}

        # Try to extract company name and address from text for each entity
        def _extract_entity_name(text: str) -> str | None:
            """Find company name with legal form (SAS, SARL, etc.) in text."""
            legal_forms = r"(?:SAS|SARL|EURL|SASU|SCI|SNC|SCOP|SELARL|SEL|SELAFA)"
            # Pattern 1: "ACME SOLUTIONS SAS" — uppercase words followed by legal form
            # Must start with uppercase letter, not a digit (avoids matching addresses)
            patterns = [
                rf"([A-ZÀ-Ÿ][A-ZÀ-Ÿ\s&\-']+?)\s+({legal_forms})\b",
                rf"\b({legal_forms})\s+([A-ZÀ-Ÿ][A-ZÀ-Ÿ\s&\-']+?)(?:\s*[-–—,.\n]|\s+(?:au capital|RCS|SIRET|SIREN|TVA|\d))",
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    groups = m.groups()
                    # Reconstruct name: put legal form after company name
                    if re.match(legal_forms, groups[0]):
                        name = f"{groups[1].strip()} {groups[0]}"
                    else:
                        name = f"{groups[0].strip()} {groups[1]}"
                    name = name.strip()
                    # Skip if too short (likely a false match) or too long
                    if len(name) > 60:
                        name = name[:60].rsplit(" ", 1)[0]
                    if len(name) > 5:
                        return name
            return None

        def _extract_entity_address(text: str) -> str | None:
            """Find a French postal address in text (street + postal code + city)."""
            # Pattern: optional street line, then 5-digit postal code + city name
            # City: letters only (stops before keywords like SIRET, SIREN, TVA, etc.)
            m = re.search(
                r"(\d{1,4}[\s,]+(?:rue|avenue|boulevard|bd|av|place|chemin|impasse|allée|route|passage)"
                r"[^,\n]{3,50}[\s,]+)?(\d{5})\s+([A-ZÀ-Ÿa-zà-ÿ][a-zà-ÿA-ZÀ-Ÿ\s\-]*[a-zà-ÿA-ZÀ-Ÿ])",
                text, re.IGNORECASE
            )
            if m:
                street = (m.group(1) or "").strip().rstrip(",").strip()
                postal = m.group(2)
                # Clean city: stop at known keywords that might follow
                city = re.split(r"\s+(?:SIRET|SIREN|TVA|RCS|Tel|Fax|Cedex\b)", m.group(3), flags=re.IGNORECASE)[0].strip()
                if street:
                    return f"{street}, {postal} {city}"
                return f"{postal} {city}"
            return None

        entity_name = _extract_entity_name(full_text)
        entity_address = _extract_entity_address(full_text)

        for siren_key, idents in unverified_by_siren.items():
            # Entity label: use company name if found, otherwise "SIREN XXXXXXXXX"
            label = html_mod_ent.escape(entity_name) if entity_name else f"SIREN {siren_key}"

            rows = ""
            if idents["siret"]:
                rows += f'<div style="{_row}"><span style="{_key}">SIRET</span><span style="{_val}">{idents["siret"]}</span></div>'
            rows += f'<div style="{_row}"><span style="{_key}">SIREN</span><span style="{_val}">{siren_key}</span></div>'
            if idents["vat"]:
                rows += f'<div style="{_row}"><span style="{_key}">VAT</span><span style="{_val}">{idents["vat"]}</span></div>'
            if idents["rcs"]:
                rows += f'<div style="{_row}"><span style="{_key}">RCS</span><span style="{_val}">{idents["rcs"]}</span></div>'
            if entity_address:
                rows += f'<div style="{_row}"><span style="{_key}">Address</span><span style="{_val}">{html_mod_ent.escape(entity_address)}</span></div>'

            entities_html += (
                f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                f'padding:12px 16px; margin-bottom:8px;">'
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
                f'<span style="color:white; font-size:0.88rem; font-weight:bold;">{label}</span>'
                f'{not_verified_pill}'
                f'</div>'
                f'{rows}'
                f'</div>'
            )
            entity_count += 1

        # Show Entities panel if any entities were found
        if entity_count > 0:
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

        # Bank Information panel (IBAN verification results)
        verified_ibans = []
        if external_module and external_module.details.get("verified_ibans"):
            verified_ibans = external_module.details["verified_ibans"]

        # Also show IBANs found in text even without API verification
        from src.modules.content import extract_iban
        found_ibans = extract_iban(full_text)

        if verified_ibans or found_ibans:
            bank_html = ""
            bank_count = 0
            shown_ibans = set()

            _row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
            _key = "color:#888; font-size:0.75rem; width:80px; flex-shrink:0;"
            _val = "color:#ccc; font-size:0.75rem;"

            # 1) API-verified IBANs
            for iban_data in verified_ibans:
                iban_str = iban_data.get("iban", "")
                iban_clean = iban_str.replace(" ", "")
                shown_ibans.add(iban_clean)
                bank_count += 1

                is_valid = iban_data.get("valid", False)
                if is_valid:
                    status_pill = ('<span style="background:#28a745; color:white; font-size:0.62rem; '
                                   'font-weight:bold; padding:2px 8px; border-radius:8px;">Valid</span>')
                else:
                    status_pill = ('<span style="background:#ef233c; color:white; font-size:0.62rem; '
                                   'font-weight:bold; padding:2px 8px; border-radius:8px;">Invalid</span>')

                # Format IBAN with spaces for readability (groups of 4)
                iban_display = " ".join([iban_clean[i:i+4] for i in range(0, len(iban_clean), 4)])

                rows = f'<div style="{_row}"><span style="{_key}">IBAN</span><span style="{_val}">{iban_display}</span></div>'
                bic = iban_data.get("bic", "")
                if bic:
                    rows += f'<div style="{_row}"><span style="{_key}">BIC</span><span style="{_val}">{bic}</span></div>'
                bank_name = iban_data.get("bank_name") or ""
                if bank_name:
                    rows += f'<div style="{_row}"><span style="{_key}">Bank</span><span style="{_val}">{html_mod_ent.escape(bank_name)}</span></div>'
                branch_name = iban_data.get("branch_name") or ""
                if branch_name:
                    rows += f'<div style="{_row}"><span style="{_key}">Branch</span><span style="{_val}">{html_mod_ent.escape(branch_name)}</span></div>'
                country = iban_data.get("country_code", "")
                if country:
                    rows += f'<div style="{_row}"><span style="{_key}">Country</span><span style="{_val}">{country}</span></div>'

                import unicodedata
                title_raw = unicodedata.normalize("NFD", bank_name)
                title_raw = "".join(c for c in title_raw if unicodedata.category(c) != "Mn")
                title = html_mod_ent.escape(title_raw).upper() if bank_name else "BANK ACCOUNT"
                bank_html += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
                    f'<span style="color:white; font-size:0.88rem; font-weight:bold;">'
                    f'{title}</span>'
                    f'{status_pill}'
                    f'</div>'
                    f'{rows}'
                    f'</div>'
                )

            # 2) IBANs found in text but not API-verified
            not_verified_pill = ('<span style="background:#555; color:#ccc; font-size:0.62rem; '
                                 'font-weight:bold; padding:2px 8px; border-radius:8px;">Not verified</span>')
            for iban_str, is_checksum_valid, context in found_ibans:
                if iban_str in shown_ibans:
                    continue
                shown_ibans.add(iban_str)
                bank_count += 1

                iban_display = " ".join([iban_str[i:i+4] for i in range(0, len(iban_str), 4)])
                checksum_note = "Valid checksum" if is_checksum_valid else "Invalid checksum"

                rows = f'<div style="{_row}"><span style="{_key}">IBAN</span><span style="{_val}">{iban_display}</span></div>'
                rows += f'<div style="{_row}"><span style="{_key}">Country</span><span style="{_val}">{iban_str[:2]}</span></div>'
                rows += f'<div style="{_row}"><span style="{_key}">Check</span><span style="{_val}">{checksum_note}</span></div>'

                bank_html += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
                    f'<span style="color:white; font-size:0.88rem; font-weight:bold;">Bank account</span>'
                    f'{not_verified_pill}'
                    f'</div>'
                    f'{rows}'
                    f'</div>'
                )

            if bank_count > 0:
                st.markdown(
                    f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                    f'border-radius:8px; padding:0; margin-bottom:8px;">'
                    f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                    f'display:flex; align-items:center; gap:10px;">'
                    f'<span style="color:#eee; font-size:0.85rem;">Bank Information</span>'
                    f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                    f'padding:2px 10px; border-radius:10px;">{bank_count} account{"s" if bank_count > 1 else ""}</span>'
                    f'</summary>'
                    f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                    f'{bank_html}'
                    f'</div></details>',
                    unsafe_allow_html=True,
                )

        # 2D-DOC Verified Data panel
        if twod_doc_results:
            import html as html_mod_2d

            twod_content = ""
            for twod_doc_data in twod_doc_results:
                verified = extract_verified_data(twod_doc_data)

                _2d_row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
                _2d_key = "color:#888; font-size:0.75rem; width:160px; flex-shrink:0;"
                _2d_val = "color:#ccc; font-size:0.75rem;"
                _2d_val_bold = "color:white; font-size:0.75rem; font-weight:bold;"

                rows_2d = ""
                rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Document type</span>'
                            f'<span style="{_2d_val}">{html_mod_2d.escape(verified.document_type)} '
                            f'({html_mod_2d.escape(verified.document_type_name)})</span></div>')

                # Track which DIs we display so we can show remaining as raw
                shown_dis = set()

                # --- Identity fields (shared across document types) ---
                if verified.name:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Name</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.name)}</span></div>')
                    shown_dis.update({"10", "12", "13", "44", "6G", "60"})

                # --- Driving license fields (type AB) ---
                if verified.civility:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Civility</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.civility)}</span></div>')
                    shown_dis.add("6H")

                if verified.sex:
                    sex_label = {"M": "Male", "F": "Female"}.get(verified.sex, verified.sex)
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Sex</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(sex_label)}</span></div>')
                    shown_dis.add("68")

                if verified.nationality:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Nationality</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.nationality)}</span></div>')
                    shown_dis.update({"67", "6C"})

                if verified.birth_date:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Date of birth</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.birth_date)}</span></div>')
                    shown_dis.add("69")

                if verified.birth_place:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Place of birth</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.birth_place)}</span></div>')
                    shown_dis.add("6A")

                if verified.document_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Document number</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.document_number)}</span></div>')
                    shown_dis.add("65")

                if verified.permit_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Permit number</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.permit_number)}</span></div>')
                    shown_dis.add("AC")

                if verified.permit_categories:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Permit categories</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.permit_categories)}</span></div>')
                    shown_dis.add("E4")

                # --- Tax fields ---
                if verified.fiscal_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Tax ID</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.fiscal_number)}</span></div>')
                    shown_dis.add("47")

                if verified.address or verified.postal_code or verified.city:
                    addr_parts = []
                    if verified.address:
                        addr_parts.append(verified.address)
                    if verified.postal_code or verified.city:
                        addr_parts.append(f"{verified.postal_code or ''} {verified.city or ''}".strip())
                    addr = ", ".join(addr_parts)
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Address</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(addr)}</span></div>')
                    shown_dis.update({"22", "24", "25"})

                if verified.reference_income is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Reference income</span>'
                                f'<span style="{_2d_val_bold}">{verified.reference_income:,.0f} &euro;</span></div>')
                    shown_dis.add("41")

                if verified.tax_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Income tax</span>'
                                f'<span style="{_2d_val_bold}">{verified.tax_amount:,.0f} &euro;</span></div>')
                    shown_dis.add("4V")

                if verified.withheld_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Withholding tax</span>'
                                f'<span style="{_2d_val}">{verified.withheld_amount:,.0f} &euro;</span></div>')
                    shown_dis.add("4X")

                if verified.calculated_balance is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Calculated balance</span>'
                                f'<span style="{_2d_val_bold}">{verified.calculated_balance:,.0f} &euro;</span></div>')

                if verified.household_parts and verified.household_parts != 1.0:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Household parts</span>'
                                f'<span style="{_2d_val}">{verified.household_parts}</span></div>')
                    shown_dis.add("43")

                # --- Invoice fields ---
                if verified.invoice_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Invoice number</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.invoice_number)}</span></div>')

                if verified.invoice_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">Invoice amount</span>'
                                f'<span style="{_2d_val_bold}">{verified.invoice_amount:,.2f} &euro;</span></div>')

                # Show remaining unmapped fields as raw
                extra_fields = ""
                for field in twod_doc_data.fields:
                    if field.di in shown_dis:
                        continue
                    field_val = html_mod_2d.escape(str(field.value))
                    extra_fields += (f'<div style="{_2d_row}"><span style="{_2d_key}">Field {field.di}</span>'
                                     f'<span style="{_2d_val}">{field_val}</span></div>')

                if extra_fields:
                    rows_2d += (f'<div style="margin-top:8px; padding-top:6px; border-top:1px solid #2a2a2a;">'
                                f'<span style="color:#888; font-size:0.7rem; font-weight:bold;">Raw signed fields</span>'
                                f'</div>{extra_fields}')

                twod_content += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'{rows_2d}</div>'
                )

            st.markdown(
                f'<details open style="background:#0a2e1a; border:1px solid #28a745; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">2D-DOC</span>'
                f'<span style="background:#28a745; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">Verified</span>'
                f'<span style="color:#888; font-size:0.72rem;">Compare with visible PDF content</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #28a745;">'
                f'{twod_content}'
                f'</div></details>',
                unsafe_allow_html=True,
            )

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)

# Footer
st.markdown("""
<div style="position:fixed; bottom:0; left:0; right:0; text-align:center;
    padding:0.3rem; background:#0e1117; border-top:1px solid #1a1a2e; z-index:999;">
    <span style="color:#555; font-size:0.65rem;">
        TrustyFile - Document Fraud Detector
    </span>
</div>
""", unsafe_allow_html=True)
