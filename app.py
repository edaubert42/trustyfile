"""
TrustyFile - Document Fraud Detection

A Streamlit web application for analyzing invoices and documents
to detect signs of manipulation or fraud.

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import html
import json
import logging
import tempfile
from contextlib import ExitStack
from datetime import datetime
import os
import re
from pathlib import Path
from src.dates import days_between, seconds_between
from src.i18n import t, get_lang
from src.preview import render_pdf_page, collect_evidence_boxes, first_flagged_page

# The landing page lists the verification guides and the checker tools. Their
# slugs and titles live with the generator (scripts/build_guides.py), which
# also renders the crawler-facing copy of this landing page — one source, so
# the two pages list exactly the same links and the crawler never sees a link
# a human does not.
try:
    from scripts.build_guides import GUIDES as _GUIDES, TOOLS as _TOOLS
except Exception:  # a broken or missing generator must never take the app down
    _GUIDES = {"fr": [], "en": []}
    _TOOLS = {"fr": [], "en": []}

# Which checker page can re-run the check behind a flag. Matched on the flag
# code (CONTENT_INVALID_IBAN, EXTERNAL_SIRET_NOT_FOUND, ...) rather than on a
# hand-kept list, so a new identifier flag gets its link for free. VAT before
# SIREN so that CONTENT_SIREN_VAT_MISMATCH lands on the VAT checker, which
# shows the key the SIREN implies — exactly what that flag is about.
_TOOL_KIND_BY_CODE_TOKEN = (("IBAN", "iban"), ("SIRET", "siret"), ("VAT", "vat"), ("SIREN", "siren"))


def tool_url_for_flag(code: str, lang: str) -> str | None:
    """Site-relative URL of the checker page matching a flag code, if any.

    Example:
        >>> tool_url_for_flag("CONTENT_INVALID_IBAN", "fr")
        '/outils/verifier-un-iban'
        >>> tool_url_for_flag("META_ONLINE_CONVERTER", "fr") is None
        True
    """
    prefix = "/outils/" if lang == "fr" else "/en/tools/"
    for token, kind in _TOOL_KIND_BY_CODE_TOKEN:
        if token in code:
            for tool in _TOOLS.get(lang, []):
                if tool.kind == kind:
                    return prefix + tool.slug
    return None


def esc(s) -> str:
    """Escape a value for safe interpolation into unsafe_allow_html markdown.

    PDF metadata (Producer, Creator, editor, signer) and the uploaded
    filename are attacker-controlled: a malicious PDF can set these fields
    to arbitrary strings. Because the file-properties block is rendered via
    st.markdown(..., unsafe_allow_html=True), any unescaped value would be
    interpreted as raw HTML, enabling stored XSS (e.g. a Producer of
    "<img src=x onerror=alert(1)>"). Always wrap attacker-controlled values
    with this helper before interpolating them into HTML, in both element
    text and attribute (e.g. title="...") positions.
    """
    return html.escape(str(s if s is not None else ""), quote=True)


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

def get_pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    import fitz
    doc = fitz.open(pdf_path)
    try:
        count = len(doc)
    finally:
        doc.close()
    return count


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=t("page_title"),
    page_icon="static/favicon-48x48.png",
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

# Current language for JS injection
_current_lang = get_lang()

# Inject meta tags + hide content until loaded.
#
# st.html rather than st.components.v1.html, which Streamlit removes "after
# 2026-06-01". Note st.iframe is NOT the replacement the deprecation warning
# implies — it takes a src URL, not HTML. Two behaviour differences matter:
#   - st.html is not iframed, so `doc` is this document directly instead of
#     window.parent.document (which would be the same window here anyway, but
#     is misleading and breaks if Streamlit ever frames the app).
#   - JavaScript is dropped unless unsafe_allow_javascript=True. The payload is
#     a static literal built from our own translation strings, not user input.
st.html(f"""
<script>
// Everything is wrapped in an IIFE. st.components.v1.html gave each render its
// own iframe scope; st.html runs inline in the page's global scope, so a
// top-level `const` re-declares itself on every Streamlit rerun and throws
// "Identifier 'doc' has already been declared", aborting the rest of the script.
(function() {{
    // st.html renders inline, so this IS the Streamlit main document
    const doc = document;
    if (!doc.querySelector('meta[http-equiv="Content-Security-Policy"]')) {{
        const meta = doc.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = 'upgrade-insecure-requests';
        doc.head.appendChild(meta);
    }}
    if (!doc.querySelector('meta[name="viewport"]')) {{
        const vp = doc.createElement('meta');
        vp.name = 'viewport';
        vp.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0';
        doc.head.appendChild(vp);
    }}
    // SEO: meta description
    if (!doc.querySelector('meta[name="description"]')) {{
        const desc = doc.createElement('meta');
        desc.name = 'description';
        desc.content = '{t("meta_description")}';
        doc.head.appendChild(desc);
    }}
    // SEO: Open Graph tags (social media previews)
    const ogTags = {{
        'og:title': '{t("page_title")}',
        'og:description': '{t("og_description")}',
        'og:type': 'website',
        'og:url': 'https://www.trustyfile.com',
        // Served by Caddy from static/og-image-1200x630.png. Was hosted on
        // i.ibb.co: a third-party image host we do not control is a silent
        // single point of failure for every social preview.
        'og:image': 'https://www.trustyfile.com/og-image.png',
        'og:image:width': '1200',
        'og:image:height': '630',
        'og:image:alt': 'TrustyFile',
        'og:site_name': 'TrustyFile'
    }};
    for (const [prop, content] of Object.entries(ogTags)) {{
        if (!doc.querySelector(`meta[property="${{prop}}"]`)) {{
            const tag = doc.createElement('meta');
            tag.setAttribute('property', prop);
            tag.content = content;
            doc.head.appendChild(tag);
        }}
    }}
    // SEO: Twitter card tags
    const twTags = {{
        // summary_large_image: the 1200x630 card, not the small square thumb
        'twitter:card': 'summary_large_image',
        'twitter:title': '{t("page_title")}',
        'twitter:description': '{t("twitter_description")}',
        'twitter:image': 'https://www.trustyfile.com/og-image.png',
        // Not a Twitter tag, but injected the same way (meta[name=...]).
        // max-image-preview:large unlocks full-size thumbnails in Google Discover.
        'robots': 'index, follow, max-image-preview:large, max-snippet:-1'
    }};
    for (const [name, content] of Object.entries(twTags)) {{
        if (!doc.querySelector(`meta[name="${{name}}"]`)) {{
            const tag = doc.createElement('meta');
            tag.name = name;
            tag.content = content;
            doc.head.appendChild(tag);
        }}
    }}
    // SEO: canonical URL (avoid duplicate content)
    if (!doc.querySelector('link[rel="canonical"]')) {{
        const link = doc.createElement('link');
        link.rel = 'canonical';
        link.href = 'https://www.trustyfile.com';
        doc.head.appendChild(link);
    }}
    // Override Streamlit's default favicon with our own
    doc.querySelectorAll('link[rel="icon"], link[rel="shortcut icon"]').forEach(el => el.remove());
    const fav48 = doc.createElement('link');
    fav48.rel = 'icon';
    fav48.type = 'image/png';
    fav48.sizes = '48x48';
    fav48.href = '/favicon-48x48.png';
    doc.head.appendChild(fav48);
    const fav192 = doc.createElement('link');
    fav192.rel = 'icon';
    fav192.type = 'image/png';
    fav192.sizes = '192x192';
    fav192.href = '/favicon-192x192.png';
    doc.head.appendChild(fav192);
    const favIco = doc.createElement('link');
    favIco.rel = 'icon';
    favIco.type = 'image/x-icon';
    favIco.href = '/favicon.ico';
    doc.head.appendChild(favIco);
    // SEO: language attribute
    doc.documentElement.lang = '{t("html_lang")}';
    // SEO: JSON-LD structured data (rich snippets)
    if (!doc.querySelector('script[type="application/ld+json"]')) {{
        const ld = doc.createElement('script');
        ld.type = 'application/ld+json';
        ld.textContent = JSON.stringify({{
            "@context": "https://schema.org",
            "@type": "WebApplication",
            "name": "TrustyFile",
            "url": "https://www.trustyfile.com",
            "description": "{t("json_ld_description")}",
            "applicationCategory": "SecurityApplication",
            "operatingSystem": "Any",
            "inLanguage": "{_current_lang}",
            "offers": {{ "@type": "Offer", "price": "0", "priceCurrency": "EUR" }},
            "image": "https://www.trustyfile.com/og-image.png"
        }});
        doc.head.appendChild(ld);
    }}
    if (!doc.querySelector('script[data-website-id="50120522-6e98-4433-9238-81c6ba8358c1"]')) {{
        const umami = doc.createElement('script');
        umami.defer = true;
        umami.src = 'https://cloud.umami.is/script.js';
        umami.dataset.websiteId = '50120522-6e98-4433-9238-81c6ba8358c1';
        doc.head.appendChild(umami);
    }}

    // Language auto-detection lives server-side now, in src/i18n.py
    // (_detect_browser_lang, off the Accept-Language header). Doing it here in
    // JS was wrong: history.replaceState changed the URL without telling
    // Streamlit, so the page rendered in French with ?lang=en in the address
    // bar and only flipped language on the next rerun.

    // The boot/analysis splash that used to live here (an opacity toggle on
    // section.stMain) is now its own block below — see "LOADING SPLASH".
}})();
</script>
""", unsafe_allow_javascript=True)

# =============================================================================
# LOADING SPLASH
# =============================================================================
#
# A visitor should see the brand and a single spinner, never the machinery.
#
# Two moments used to leak that machinery:
#   1. Boot — Streamlit paints its widgets as each one arrives, so the header,
#      uploader, headline and step icons popped in one at a time. (The white
#      flash *before* this is handled earlier, in the served index.html — see
#      scripts/patch_streamlit_index.py.)
#   2. Analysis — the landing page tears down and the results build up in
#      pieces, so the page visibly churns and jumps while it reflows.
#
# Both are covered by one opaque overlay: raised at boot and again when a file
# is selected, lowered when the page underneath is genuinely ready.
#
# SAFETY: an overlay that fails to come down is a dead site, so every path is
# bounded by a hard timeout. If a readiness signal never arrives, the splash
# lifts anyway and the visitor sees a working page.
st.html(f"""
<style>
    #tf-splash {{
        position: fixed; inset: 0; z-index: 2147483600;
        background: #0e1117;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 1.5rem;
        opacity: 1; transition: opacity 0.25s ease-out;
    }}
    #tf-splash.tf-splash-hiding {{ opacity: 0; pointer-events: none; }}
    #tf-splash img {{ width: 84px; height: 84px; }}
    #tf-splash .tf-label {{
        color: #cfd6e4; font: 500 0.95rem/1.4 system-ui, sans-serif;
        text-align: center; max-width: 80vw; margin-top: -0.5rem;
    }}
    #tf-splash .tf-spin {{
        width: 28px; height: 28px; border-radius: 50%;
        border: 3px solid rgba(255,255,255,0.15);
        border-top-color: #4a90e2;
        animation: tf-spin 0.8s linear infinite;
    }}
    @keyframes tf-spin {{ to {{ transform: rotate(360deg); }} }}
    /* Streamlit's own spinners and their step-by-step status text are
       internal detail; the splash speaks for them. */
    [data-testid="stSpinner"] {{ display: none !important; }}
    /* The readiness marker is a signal, not content — keep it out of the
       layout so it cannot leave a gap at the bottom of the page. */
    [data-testid="stElementContainer"]:has(#tf-results-ready) {{ display: none !important; }}
    /* The rendered page preview arrives after everything else; fade it in
       instead of letting it pop. */
    [data-testid="stImage"] img {{ animation: tf-fade 0.3s ease-in; }}
    @keyframes tf-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
</style>
<script>
(function() {{
    if (window.__tfSplash) return;      // st.html re-runs on every rerun
    window.__tfSplash = true;

    const BOOT_MAX_MS = 8000;      // never hold the page longer than this
    const ANALYSIS_MAX_MS = 90000; // analysis is slower; still bounded
    const LOGO = "data:image/png;base64,{_logo_b64}";
    const ANALYSIS_LABEL = {json.dumps(t("analyzing"))};

    let splash = null;
    let timer = null;

    const lower = () => {{
        if (timer) {{ clearTimeout(timer); timer = null; }}
        if (!splash) return;
        const node = splash;
        splash = null;
        node.classList.add('tf-splash-hiding');
        setTimeout(() => node.remove(), 300);
    }};

    // label is optional: boot is short enough that a caption would just flash,
    // but an analysis can run for many seconds and a bare spinner leaves the
    // visitor wondering whether anything is happening.
    const raise = (maxMs, label) => {{
        if (splash) return;
        // Drop the previous run's readiness marker, otherwise the observer
        // below would see the stale one and uncover before the new page is
        // built — exactly the churn this is meant to hide.
        const stale = document.getElementById('tf-results-ready');
        if (stale) stale.remove();
        splash = document.createElement('div');
        splash.id = 'tf-splash';
        const logo = document.createElement('img');
        logo.src = LOGO;
        logo.alt = '';
        const spin = document.createElement('div');
        spin.className = 'tf-spin';
        splash.appendChild(logo);
        splash.appendChild(spin);
        if (label) {{
            const cap = document.createElement('div');
            cap.className = 'tf-label';
            cap.textContent = label;
            splash.appendChild(cap);
        }}
        document.body.appendChild(splash);
        // The safety net: whatever happens, this lifts.
        timer = setTimeout(lower, maxMs);
    }};

    // --- boot ----------------------------------------------------------
    // Ready = the uploader exists (the app has rendered), then two animation
    // frames so the browser has actually painted it before we uncover.
    raise(BOOT_MAX_MS);
    const bootPoll = setInterval(() => {{
        if (!document.querySelector('[data-testid="stFileUploader"]')) return;
        clearInterval(bootPoll);
        requestAnimationFrame(() => requestAnimationFrame(lower));
    }}, 100);
    setTimeout(() => clearInterval(bootPoll), BOOT_MAX_MS);

    // --- analysis ------------------------------------------------------
    // Cover the teardown of the landing page and the build-up of the results.
    // app.py emits #tf-results-ready once the results are fully rendered.
    document.addEventListener('change', (e) => {{
        if (!e.isTrusted) return;                       // ignore upload-guard replays
        const el = e.target;
        if (!el || el.type !== 'file') return;
        if (!el.files || el.files.length === 0) return;
        raise(ANALYSIS_MAX_MS, ANALYSIS_LABEL);
    }}, true);

    const observer = new MutationObserver(() => {{
        if (document.getElementById('tf-results-ready')) {{
            requestAnimationFrame(() => requestAnimationFrame(lower));
        }}
    }});
    observer.observe(document.body, {{childList: true, subtree: true}});

    // Let the upload guard drop the splash the moment it gives up. Without
    // this the guard stops retrying (and removes its banner) while the splash
    // sits for another 30s, leaving an opaque screen explaining nothing.
    window.__tfSplashLower = lower;
}})();
</script>
""", unsafe_allow_javascript=True)


def mark_page_ready() -> None:
    """Tell the loading splash that the page beneath it is fully rendered.

    Emits the marker the splash's MutationObserver waits for. Call it on every
    path that finishes a render — including the early exits that st.stop() out
    of the analysis — or the splash sits there until its safety timeout.
    """
    st.html('<div id="tf-results-ready" style="display:none"></div>')


# =============================================================================
# UPLOAD RESILIENCE
# =============================================================================
#
# Fixes the mobile "first upload silently does nothing" bug.
#
# What happens on a phone: tapping "browse" opens the system file picker,
# which backgrounds the page. A backgrounded tab has its JavaScript throttled,
# so a websocket still being established stalls and dies. Streamlit's frontend
# is then disconnected, and the file selection that arrives when the picker
# closes goes nowhere — the frontend never asks the server for an upload URL.
# Measured on a real Pixel: the connection stayed dead for ~3 minutes while
# the page looked perfectly normal, because app CSS hides the Streamlit header
# that would otherwise show "Connecting…".
#
# The file itself is fine — it sits in the <input>, already selected. All that
# is missing is a frontend able to act on it. So: re-dispatch the change event
# until Streamlit picks it up, and tell the user what is happening meanwhile.
#
# Re-dispatching is safe. If the upload already succeeded a chip is present and
# we stop; while disconnected the event is a no-op; the retry only "wins" on
# the run where the connection has come back.
st.html(f"""
<script>
(function() {{
    if (window.__tfUploadGuard) return;      // st.html re-runs on every rerun
    window.__tfUploadGuard = true;

    const RETRY_EVERY_MS = 2500;
    const GIVE_UP_AFTER_MS = 60000;
    const BANNER_AFTER_MS = 5000;   // don't nag during a normal fast upload
    const MESSAGE = {json.dumps(t("upload_reconnecting"))};

    // Streamlit 1.58 renders an accepted file as a chip; its presence is our
    // "the frontend has taken the file" signal.
    const accepted = () => !!document.querySelector(
        '[data-testid="stFileChipName"], [data-testid="stFileChip"]');

    let banner = null;
    const showBanner = () => {{
        if (banner || accepted()) return;
        banner = document.createElement('div');
        banner.id = 'tf-upload-banner';
        banner.textContent = MESSAGE;
        banner.style.cssText = [
            'position:fixed', 'left:50%', 'transform:translateX(-50%)',
            // Above the loading splash (2147483600): when a selection lands on
            // a dead connection the splash is up, and a bare splash with no
            // explanation is worse than no splash at all.
            'bottom:16px', 'z-index:2147483647', 'max-width:90vw',
            'background:rgba(74,144,226,0.95)', 'color:#fff',
            'padding:10px 16px', 'border-radius:8px',
            'font:600 0.85rem/1.3 system-ui,sans-serif', 'text-align:center',
            'box-shadow:0 4px 14px rgba(0,0,0,0.35)',
        ].join(';');
        document.body.appendChild(banner);
    }};
    const hideBanner = () => {{
        if (banner) {{ banner.remove(); banner = null; }}
    }};

    let activeRetry = null;

    document.addEventListener('change', (e) => {{
        // Ignore our own replayed events: they would re-enter this handler and
        // start a second retry loop, which starts a third, and so on.
        // Browser-generated events are trusted; dispatchEvent ones never are.
        if (!e.isTrusted) return;

        const input = e.target;
        if (!input || input.type !== 'file') return;
        if (!input.files || input.files.length === 0) return;

        // One loop at a time — a new selection supersedes the previous one.
        if (activeRetry) clearInterval(activeRetry);

        // Keep our own reference: Streamlit clears input.files once it has
        // taken the file, and a cleared input has nothing left to re-dispatch.
        const pending = Array.from(input.files);
        const startedAt = Date.now();

        const timer = activeRetry = setInterval(() => {{
            const waited = Date.now() - startedAt;

            if (accepted()) {{ clearInterval(timer); activeRetry = null; hideBanner(); return; }}
            if (waited > GIVE_UP_AFTER_MS) {{
                clearInterval(timer); activeRetry = null; hideBanner();
                // Uncover the page: retrying is over, so leaving the loading
                // splash up would just be an opaque screen with no explanation.
                if (window.__tfSplashLower) window.__tfSplashLower();
                return;
            }}
            if (waited > BANNER_AFTER_MS) showBanner();

            // The uploader can be re-mounted by a rerun, so re-resolve it
            // rather than holding the original element.
            const target = document.querySelector(
                '[data-testid="stFileUploader"] input[type=file]') || input;

            // Restore the selection if Streamlit reset the input, then let
            // React see a fresh change event.
            if ((!target.files || target.files.length === 0) &&
                typeof DataTransfer !== 'undefined') {{
                try {{
                    const dt = new DataTransfer();
                    pending.forEach((f) => dt.items.add(f));
                    target.files = dt.files;
                }} catch (err) {{ /* older WebKit: fall through and just retry */ }}
            }}
            target.dispatchEvent(new Event('change', {{bubbles: true}}));
        }}, RETRY_EVERY_MS);
    }}, true);
}})();
</script>
""", unsafe_allow_javascript=True)

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

    /* Responsive pill text: show short version on mobile, long on desktop */
    .pill-long {{ display: inline; }}
    .pill-short {{ display: none; }}

    /* Mobile responsive adjustments */
    @media (max-width: 480px) {{
        .pill-long {{ display: none !important; }}
        .pill-short {{ display: inline !important; }}
        /* More space between collapsible sections on mobile */
        details {{ margin-bottom: 8px !important; }}

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

    /* ---- Print / "Save as PDF" (roadmap #8, tier 1) ----
       The results page IS the report. This stylesheet turns the dark UI into
       a black-on-white document: hides the chrome (header, uploader, page
       selector, footer, the print button itself), stacks the two columns and
       remaps the inline dark-theme colours through [style*=...] attribute
       selectors — blunt, but it leaves every existing inline style alone. */
    .tf-print-only {{ display: none; }}
    @media print {{
        @page {{ margin: 12mm; }}
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] {{
            background: #fff !important;
            color: #000 !important;
        }}
        .tf-print-only {{ display: block; }}
        .custom-header, .tf-footer, #tf-print-wrap,
        [data-testid="stFileUploader"], [data-testid="stSelectbox"],
        [data-testid="stStatusWidget"], [data-testid="stDecoration"],
        [data-testid="stToolbar"], [data-testid="stBottom"],
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
        /* Results above the preview, instead of two narrow columns */
        [data-testid="stHorizontalBlock"] {{ display: block !important; }}
        [data-testid="stColumn"] {{ width: 100% !important; flex: none !important; }}
        [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] {{
            padding-top: 0 !important;
            max-width: none !important;
        }}
        /* Text. Everything rendered through st.markdown goes via React, which
           re-serialises inline styles (`color:#eee` in the source becomes
           `color: rgb(238, 238, 238)` in the DOM), so no rule can match the
           hex spelled in app.py. Blanket black text inside every Streamlit
           block, then give the coloured severity pills their white text back. */
        [data-testid="stMarkdown"] *, [data-testid="stHtml"] *,
        [data-testid="stExpander"] * {{
            color: #000 !important;
        }}
        [data-testid="stMarkdown"] [style*="color: white"][style*="background: rgb("],
        [data-testid="stMarkdown"] [style*="color: white"][style*="background-color: rgb("] {{
            color: #fff !important;
        }}
        /* Dark cards and bar tracks -> light paper cards. rgb() forms of
           #1a1a2e #111 #0e1117 #1a1a1a #2a2a2a #0a0a0a #1e1e1e #333 and the
           tinted entity cards #0a2e1a #0a1e3a #1a2a3a. */
        [style*="background: rgb(26, 26, 46)"], [style*="background: rgb(17, 17, 17)"],
        [style*="background: rgb(14, 17, 23)"], [style*="background: rgb(26, 26, 26)"],
        [style*="background: rgb(42, 42, 42)"], [style*="background: rgb(10, 10, 10)"],
        [style*="background: rgb(30, 30, 30)"], [style*="background: rgb(51, 51, 51)"],
        [style*="background: rgb(10, 46, 26)"], [style*="background: rgb(10, 30, 58)"],
        [style*="background: rgb(26, 42, 58)"] {{
            background: #f3f4f6 !important;
        }}
        [style*="border: 1px solid rgb(42, 42, 42)"], [style*="border-top: 1px solid rgb(42, 42, 42)"],
        [style*="border-bottom: 1px solid rgb(34, 34, 34)"] {{
            border-color: #d1d5db !important;
        }}
        hr {{ border-color: #d1d5db !important; }}
        [data-testid="stExpander"], [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
            background: #fff !important;
            border-color: #d1d5db !important;
        }}
        /* Gauge: the white score glyph and the dark track */
        svg text[fill="white"] {{ fill: #000 !important; }}
        svg circle[stroke="#2a2a2a"] {{ stroke: #e5e7eb !important; }}
        /* Keep severity pills and score bars in colour */
        * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        details, .tf-print-keep {{ break-inside: avoid; }}
        img {{ max-width: 100% !important; }}
        a {{ text-decoration: none; color: inherit; }}
    }}

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
    .tf-label:hover .tf-tip,
    .tf-label:focus .tf-tip,
    .tf-label:focus-within .tf-tip {{
        visibility: visible;
        opacity: 1;
    }}
    /* The labels carry tabindex="0" so a tap (or Tab) focuses them: hover
       does not exist on a touchscreen, and these tips are the only place
       the score categories and the date rows are explained. */
    .tf-label:focus {{
        outline: 1px dotted #666;
        outline-offset: 2px;
    }}
    @media (max-width: 600px) {{
        /* The tip is absolutely positioned inside a ~100px label, so plain
           `white-space: normal` would wrap it into a 72px-wide sliver.
           max-content lets it take the width it needs, capped to the viewport. */
        .tf-label .tf-tip {{
            white-space: normal;
            width: max-content;
            max-width: 70vw;
        }}
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
        /* Symmetric padding: with the asymmetric 2rem/1rem it used to have,
           justify-content:center still centred the content in the padding box,
           so the whole group sat visually high inside the dashed border. */
        padding: 2rem !important;
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
    /* The button lives inside a wrapper <span> that is the actual flex item
       of the dropzone — order must go on that span, not the nested button. */
    [data-testid="stFileUploaderDropzone"] > span {{
        order: 3 !important;
    }}
    /* "ou" / "or" as its own flex item between the instructions (order 0) and
       the button (order 3). It used to be a second line inside the
       instructions ::after, which glued it to "Déposez votre PDF ici" and left
       it far from the button it actually separates. Its own item lets the
       margins sit evenly on both sides. */
    [data-testid="stFileUploaderDropzone"]::after {{
        content: "{t("drop_or")}";
        order: 2 !important;
        display: block !important;
        /* Top margin is the smaller of the two: the title's line box already
           contributes descender space above, so equal margins read top-heavy. */
        margin: 0.6rem 0 0.9rem !important;
        color: #888 !important;
        font-size: 0.95rem !important;
        line-height: 1 !important;
    }}
    /* Restyle the native browse button as "Browse file" */
    [data-testid="stFileUploaderDropzone"] button {{
        border: 1.5px solid #666 !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.5rem !important;
        color: #bbb !important;
        background: transparent !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        /* No bottom margin: it sat inside the flex item, so centring counted it
           as content and pushed everything up, leaving dead space under the
           button. Spacing now comes from the dropzone's own padding. */
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        border-color: #999 !important;
        color: #eee !important;
    }}
    /* Hide the default button's whole icon+label wrapper ("Upload") and replace with "Browse file" */
    [data-testid="stFileUploaderDropzone"] button > div {{
        display: none !important;
    }}
    [data-testid="stFileUploaderDropzone"] button::after {{
        content: "{t("browse_file")}";
        font-size: 0.9rem;
    }}
    /* Make instructions a flex column so ::before (icon) is on top, ::after (text) below */
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
        /* Streamlit ships `flex: 1 1 0%` here, so this block grew to swallow all
           the dropzone's free space — the icon and title stayed pinned to the
           top while the gap piled up underneath them. Hug the content instead
           and let justify-content:center place the group as a whole. */
        flex: 0 0 auto !important;
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
    /* "Drop your PDF here" — a single line now; the "or" is a separate flex
       item on the dropzone itself (see above). */
    [data-testid="stFileUploaderDropzoneInstructions"]::after {{
        content: "{t("drop_pdf_here")}";
        display: block;
        text-align: center;
        order: 2;
        color: #bbb;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        line-height: 1.3;
    }}
    /* Short viewports (phone landscape, small/unmaximized windows): shrink the
       dropzone so the Browse button isn't clipped below the fold.

       Must come AFTER the landing-page dropzone rules above: both sets use
       !important with the same specificity, so source order decides. While
       this block sat earlier in the stylesheet its min-height and padding were
       silently overridden and the button stayed off-screen in landscape. */
    @media (max-height: 500px) {{
        [data-testid="stFileUploaderDropzone"] {{
            min-height: 0 !important;
            padding: 0.6rem 1rem !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"]::before {{
            width: 70px !important;
            height: 70px !important;
            margin-bottom: 0.3rem !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"]::after {{
            font-size: 0.85rem !important;
            line-height: 1.3 !important;
        }}
        /* Tighten the "or" gap too, or the group overflows a short viewport */
        [data-testid="stFileUploaderDropzone"]::after {{
            margin: 0.4rem 0 !important;
            font-size: 0.8rem !important;
        }}
    }}
</style>
<div class="custom-header">
    <a class="home-link" href="/" target="_parent">
        <img src="data:image/png;base64,{_logo_b64}" alt="logo">
        <span class="title">TrustyFile</span>
    </a>
    <span class="subtitle">{t("subtitle")}</span>
</div>
""", unsafe_allow_html=True)

# File uploader — always rendered at top level (outside landing columns)
uploaded_file = st.file_uploader(
    "Choose file",
    type=["pdf"],
    label_visibility="collapsed",
)


# "Try with a sample document". A first-time visitor rarely has a PDF at hand,
# and the landing page shows nothing of what a result looks like. The sample is
# a fictitious, deliberately tampered invoice shipped with the repo
# (static/demo/, built by scripts/make_demo_invoice.py — never a real
# document). It stands in for the upload object: the rest of this page only
# ever reads .name and .getvalue(), and a real upload always wins over it.
_DEMO_PDF = Path(__file__).resolve().parent / "static" / "demo" / "facture-exemple.pdf"


class _DemoUpload:
    """Duck-typed stand-in for Streamlit's UploadedFile, backed by the sample PDF."""

    name = "facture-exemple.pdf"

    def getvalue(self) -> bytes:
        return _DEMO_PDF.read_bytes()


if uploaded_file is not None:
    st.session_state.pop("demo_active", None)
elif st.session_state.get("demo_active") and _DEMO_PDF.exists():
    uploaded_file = _DemoUpload()

if uploaded_file is None:
    st.markdown(f"""
    <p style="color:#666; font-size:0.75rem; text-align:center; margin-top:-0.5rem;">
        {t("disclaimer_upload")}
    </p>
    """, unsafe_allow_html=True)
    if _DEMO_PDF.exists():
        _demo_l, _demo_c, _demo_r = st.columns([2, 1.6, 2])
        if _demo_c.button(t("demo_button"), key="demo_button", use_container_width=True):
            st.session_state["demo_active"] = True
            st.rerun()

# Landing page content — only rendered when no file is uploaded
if uploaded_file is None:
    col_left, col_upload, col_right = st.columns([1, 2, 1])
    with col_upload:
        st.markdown(f"""
        <div id="landing-content">
            <p style="color:white; font-size:2.2rem; text-align:center; margin:0 auto 0.8rem auto; max-width:600px; font-weight:bold; letter-spacing:0.5px;">
                {t("landing_headline")}
            </p>
        </div>
        <div id="landing-tagline">
            <p style="color:white; font-size:1.2rem; text-align:center; margin:1.2rem auto 0.5rem auto; max-width:500px; line-height:1.5;">
                {t("landing_tagline")}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Guide links. Until 2026-08-27 these existed only on the crawler-only
    # copy of this page; a human gets the same row now, so the two stay
    # equivalent (see scripts/build_guides.py render_landing).
    _guides_prefix = "/guides/" if _current_lang == "fr" else "/en/guides/"
    _guide_links = " · ".join(
        f'<a href="{_guides_prefix}{g.slug}" style="color:#888; text-decoration:none;">{html.escape(g.title)}</a>'
        for g in _GUIDES.get(_current_lang, [])
    )
    _landing_guides_html = (
        '<p style="color:#666; font-size:0.74rem; text-align:center; margin:1.8rem auto 0 auto; '
        'line-height:1.9; max-width:640px;">'
        f'{t("footer_guides")}{" :" if _current_lang == "fr" else ":"} {_guide_links}</p>'
    ) if _guide_links else ""
    # Checker tools (IBAN / SIRET / SIREN), same treatment: a visitor with a
    # single number to check should not have to upload a document to find them.
    _tools_prefix = "/outils/" if _current_lang == "fr" else "/en/tools/"
    _tool_links = " · ".join(
        f'<a href="{_tools_prefix}{tool.slug}" style="color:#888; text-decoration:none;">{html.escape(tool.title)}</a>'
        for tool in _TOOLS.get(_current_lang, [])
    )
    _landing_tools_html = (
        '<p style="color:#666; font-size:0.74rem; text-align:center; margin:0.4rem auto 0 auto; '
        'line-height:1.9; max-width:640px;">'
        f'{t("footer_tools")}{" :" if _current_lang == "fr" else ":"} {_tool_links}</p>'
    ) if _tool_links else ""

    # "How it works" steps
    st.markdown(f"""
    <style>
        .steps-row {{
            display: flex;
            align-items: stretch;
            justify-content: center;
            margin: 2.5rem auto 0 auto;
            gap: 0;
        }}
        .step-card {{
            width: 180px;
            flex-shrink: 0;
            text-align: center;
            padding: 1.2rem 1rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            transition: border-color 0.2s, background 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .step-card:hover {{
            background: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 255, 255, 0.15);
        }}
        .step-arrow {{
            color: rgba(255, 255, 255, 0.15);
            font-size: 1.3rem;
            padding: 0 0.6rem;
            flex-shrink: 0;
            display: flex;
            align-items: center;
        }}
        .step-number {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.10);
            color: #999;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }}
        .step-title {{
            color: #ddd;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}
        .step-desc {{
            color: #777;
            font-size: 0.76rem;
            line-height: 1.4;
        }}
        @media (max-width: 600px) {{
            .steps-row {{
                flex-direction: column;
                align-items: center;
                gap: 0;
            }}
            .step-card {{
                width: 220px;
            }}
            .step-arrow {{
                transform: rotate(90deg);
                padding: 0.3rem 0;
            }}
        }}
    </style>
    <div class="steps-row">
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-title">{t("step1_title")}</div>
            <div class="step-desc">{t("step1_desc")}</div>
        </div>
        <div class="step-arrow">&#x276F;</div>
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-title">{t("step2_title")}</div>
            <div class="step-desc">{t("step2_desc")}</div>
        </div>
        <div class="step-arrow">&#x276F;</div>
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-title">{t("step3_title")}</div>
            <div class="step-desc">{t("step3_desc")}</div>
        </div>
    </div>
    {_landing_guides_html}
    {_landing_tools_html}
    <p style="color:#555; font-size:0.70rem; text-align:center; margin-top:2rem;">
        {t("disclaimer_api")}
    </p>
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
        /* Streamlit 1.58 renders the accepted file as a "chip"; the older
           stFileUploaderFile testid this rule used to name no longer exists,
           so it had stopped applying to anything. */
        [data-testid="stFileUploader"] [data-testid="stFileChip"],
        [data-testid="stFileUploader"] [data-testid="stFileChipName"] {
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
    from src.scoring import collect_all_flags, count_flags_by_severity
    from src.summary import generate_rich_summary, get_flag_sentence
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.structure import get_modification_history
    from src.modules.twod_doc import scan_pdf_for_2d_doc, extract_verified_data, PYLIBDMTX_AVAILABLE
    import hashlib

    # Compute a hash of the file bytes to detect if it's a new file
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Check file size — reject documents over 10MB
    _file_size_mb = len(file_bytes) / (1024 * 1024)
    if _file_size_mb > 10:
        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:30vh; text-align:center; gap:1.5rem;">
            <div style="font-size:5rem; font-weight:bold; color:white;">Oops...</div>
            <div style="background:rgba(192,57,43,0.15); border:1px solid rgba(192,57,43,0.5); border-radius:8px; padding:1.2rem 2rem; max-width:420px;">
                <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e05252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <div style="font-size:1.1rem; font-weight:bold; color:#e05252;">{t("too_large_title")}</div>
                </div>
                <div style="color:#aaa; font-size:0.8rem; text-align:center;">{t("too_large_message").format(size=f"{_file_size_mb:.1f}")}</div>
            </div>
            <a href="." style="margin-top:0.5rem; padding:0.6rem 1.5rem; background:#4a90e2; color:white; border-radius:8px; text-decoration:none; font-weight:500;">
                {t("back_to_home")}
            </a>
        </div>
        """, unsafe_allow_html=True)
        mark_page_ready()
        st.stop()

    # Check page count before analysis — reject documents over 20 pages
    import fitz as _fitz
    with _fitz.open(stream=file_bytes, filetype="pdf") as _doc:
        _page_count = len(_doc)
    if _page_count > 20:
        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:30vh; text-align:center; gap:1.5rem;">
            <div style="font-size:5rem; font-weight:bold; color:white;">Oops...</div>
            <div style="background:rgba(192,57,43,0.15); border:1px solid rgba(192,57,43,0.5); border-radius:8px; padding:1.2rem 2rem; max-width:420px;">
                <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e05252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <div style="font-size:1.1rem; font-weight:bold; color:#e05252;">{t("too_many_pages_title")}</div>
                </div>
                <div style="color:#aaa; font-size:0.8rem; text-align:center;">{t("too_many_pages_message").format(count=_page_count)}</div>
            </div>
            <a href="." style="margin-top:0.5rem; padding:0.6rem 1.5rem; background:#4a90e2; color:white; border-radius:8px; text-decoration:none; font-weight:500;">
                {t("back_to_home")}
            </a>
        </div>
        """, unsafe_allow_html=True)
        mark_page_ready()
        st.stop()

    # Materialise the upload on disk for THIS script run only. Every reader
    # below (analyzer, preview renderer, 2D-Doc scan, revision history) takes
    # a path, while the bytes themselves stay in Streamlit's session memory,
    # so re-writing a few MB per rerun is cheap — and it is what lets the
    # disclaimer say the document is not retained: the file is unlinked in
    # the `finally` at the end of this block, and on analysis failure below.
    # (Until 2026-08-27 the path was cached in session_state and the file
    # outlived the analysis until the next upload, or forever if none came.)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Only re-run the analysis if the file changed (different hash than last time)
    if st.session_state.get("last_file_hash") != file_hash:
        try:
            # No st.spinner here: the loading splash covers this whole stretch
            # and shows t("analyzing") itself, so a second spinner underneath
            # it would only ever be hidden.
            analyzer = TrustyFileAnalyzer(
                enable_external=enable_external,
                enable_qr_scan=enable_qr,
                enable_iban_check=enable_iban_check,
                iban_api_key=iban_api_key,
            )
            result = analyzer.analyze(tmp_path)
            summary = generate_rich_summary(result, lang=get_lang())

            # Scan for 2D-DOC barcodes (French government signed barcodes).
            # Le 2D-Doc n'existe que sur les CNI (TD1) — pas sur les passeports (TD3).
            # On détecte le format MRZ depuis le module identity pour éviter un scan inutile.
            _identity_mod = next(
                (m for m in result.modules if m.module == "identity"), None
            )
            _mrz_lines = (_identity_mod.details or {}).get("mrz_lines") if _identity_mod else None
            # ICAO 9303 : première lettre 'P' = passeport. Le 2D-Doc n'existe que sur les CNI.
            _is_passport = bool(_mrz_lines and _mrz_lines[0].startswith("P"))

            twod_doc_results = []
            if PYLIBDMTX_AVAILABLE and not _is_passport:
                twod_doc_results = scan_pdf_for_2d_doc(tmp_path)

            # Cache everything in session_state so page navigation doesn't re-trigger analysis
            st.session_state["last_file_hash"] = file_hash
            st.session_state["cached_result"] = result
            # Wall-clock time of THIS analysis, for the printed report. Kept in
            # session_state so a later rerun (page change, language toggle)
            # does not restamp it.
            st.session_state["cached_analysed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["cached_summary"] = summary
            st.session_state["cached_twod_doc"] = twod_doc_results
            # A stale manual page selection from a PREVIOUS file must not
            # suppress this file's auto-nav-to-first-flagged-page default —
            # st.selectbox only honors `index=` when its session_state key is
            # absent, so a genuinely new file must clear it (2026-07-21 finding).
            st.session_state.pop("page_selector", None)

        except Exception as _cache_err:
            # If analysis fails: clear the cache, drop the temp file, and
            # re-raise so the outer handler catches it
            st.session_state.pop("last_file_hash", None)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise _cache_err
    else:
        # Same file: reuse cached results (page navigation, language switch, etc.)
        result = st.session_state["cached_result"]
        summary = st.session_state["cached_summary"]
        twod_doc_results = st.session_state["cached_twod_doc"]

    # Everything below renders inside one placeholder. If anything in the
    # ~1,500 lines of results rendering raises, the except clause empties the
    # placeholder — taking the half-drawn columns with it — and puts the error
    # card in its place. ExitStack enters the container exactly as a `with`
    # would, without indenting the whole block one level deeper.
    _page_slot = st.empty()
    _page_ctx = ExitStack()
    try:
        _page_ctx.enter_context(_page_slot.container())

        # Sample-document banner: say what the reader is looking at, and give
        # them the way back to analysing their own file (the dropzone is
        # hidden while a "file" is loaded).
        if isinstance(uploaded_file, _DemoUpload):
            _demo_text, _demo_btn = st.columns([4, 1])
            _demo_text.markdown(
                f'<div style="background:#1a1a2e; border:1px solid #2563eb; border-radius:8px; '
                f'padding:8px 14px; color:#cdd6f4; font-size:0.82rem;">{esc(t("demo_banner"))}</div>',
                unsafe_allow_html=True,
            )
            if _demo_btn.button(t("demo_exit"), key="demo_exit", use_container_width=True):
                st.session_state.pop("demo_active", None)
                st.session_state.pop("last_file_hash", None)
                st.rerun()

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
            st.markdown(f"""<div class="preview-title"><span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;
                letter-spacing:0.5px;">{t("preview")}</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:2px 0 0 0;"></div>""", unsafe_allow_html=True)
            # Auto-nav default: the first page carrying a highlighted finding
            # (spec 2026-07-21, D2). Clamped defensively — first_flagged_page
            # derives purely from flag data, page_count from the PDF itself;
            # they agree by construction, but the clamp costs nothing.
            default_page_1indexed = min(first_flagged_page(result) + 1, page_count)

            # Page navigation if multiple pages
            if page_count > 1:
                st.selectbox(
                    t("label_page"),
                    range(1, page_count + 1),
                    index=default_page_1indexed - 1,
                    format_func=lambda x: t("page_x_of_y").format(x=x, y=page_count),
                    key="page_selector",
                    label_visibility="collapsed"
                )
                page_index = st.session_state.page_selector - 1
            else:
                page_index = 0

            # Render and display the page, with tamper-region highlights
            try:
                boxes = collect_evidence_boxes(result, page_index)
                img_bytes = render_pdf_page(tmp_path, page_index, zoom=2.0, boxes=boxes)
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.error(t("could_not_render").format(error=e))

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
                mod_gap = seconds_between(meta.mod_date, meta.creation_date)
                if mod_gap > 86400:
                    days = int(mod_gap // 86400)
                    mod_pill = red_pill(
                        f'<span class="pill-short">{t("pill_days_short").format(days=days)}</span>'
                        f'<span class="pill-long">{t("pill_days_long").format(days=days)}</span>',
                        t("pill_mod_tooltip").format(days=days)
                    )

            # Pill on stated date vs creation date if gap > 7 days
            if stated_date and meta.creation_date:
                # stated_date comes from the page text (naive) and creation_date
                # from PDF metadata (UTC-aware); days_between normalises both.
                # See src/dates.py for why this keeps going wrong.
                stated_gap = days_between(stated_date, meta.creation_date)
                if stated_gap > 7:
                    stated_pill = red_pill(
                        f'<span class="pill-short">{t("pill_days_short").format(days=stated_gap)}</span>'
                        f'<span class="pill-long">{t("pill_days_long").format(days=stated_gap)}</span>',
                        t("pill_stated_tooltip").format(days=stated_gap)
                    )

            def label_with_tip(label, tip):
                return (f'<span class="tf-label" tabindex="0" style="{label_style}">'
                        f'{label}<span class="tf-tip">{tip}</span></span>')

            stated_row = ""
            if stated_date:
                stated_row = f"""
            <div style="{row_style}">
                {label_with_tip(t("label_stated_date"), t("tip_stated_date"))}
                <span style="{value_style} flex-shrink:0;">{stated_date.strftime("%Y-%m-%d")}</span>{stated_pill}
            </div>
            """

            file_rows = f"""
            <div style="{row_style}">
                <span style="{label_style}">{t("label_name")}</span>
                <span style="{value_style}" title="{esc(uploaded_file.name)}">{esc(uploaded_file.name)}</span>
            </div>
            {stated_row}
            <div style="{row_style}">
                {label_with_tip(t("label_creation_date"), t("tip_creation_date"))}
                <span style="{value_style}" title="{esc(format_date(meta.creation_date))}">{esc(format_date(meta.creation_date))}</span>
            </div>
            <div style="{row_style}">
                {label_with_tip(t("label_modification_date"), t("tip_modification_date"))}
                <span style="{value_style} flex-shrink:0;" title="{esc(format_date(meta.mod_date))}">{esc(format_date(meta.mod_date))}</span>{mod_pill}
            </div>
            <div style="{row_style}">
                <span style="{label_style}">{t("label_producer")}</span>
                <span style="{value_style}" title="{esc(producer)}">{esc(producer)}</span>
            </div>
            """

            # Show Creator if different from Producer (often contains device/app info)
            creator = meta.creator or ""
            if creator and creator.lower() != (meta.producer or "").lower():
                file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">{t("label_creator")}</span>
                <span style="{value_style}" title="{esc(creator)}">{esc(creator)}</span>
            </div>
            """

            # Only show Editor line if a different software modified the PDF
            if editor_name:
                file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">{t("label_editor")}</span>
                <span style="{value_style}" title="{esc(editor_name)}">{esc(editor_name)}</span>
            </div>
            """

            file_rows += f"""
            <div style="{row_style}">
                <span style="{label_style}">{t("label_signed_by")}</span>
                <span style="{value_style}" title="{esc(sig)}">{esc(sig)}</span>
            </div>
            <div style="display:flex; align-items:center; padding:6px 0;">
                <span style="{label_style}">{t("label_sha256")}</span>
                <span style="{value_style} font-size:0.75rem;" title="{esc(result.file_hash)}">{esc(result.file_hash)}</span>
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
                        f'{t("vt_no_threats")}</span></a>'
                    )
                elif vt_result["status"] == "malicious":
                    vt_pill = (
                        f'<a href="{vt_link}" target="_blank" style="text-decoration:none;">'
                        f'<span style="background:#ef233c; color:white; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.78rem; font-weight:bold;">'
                        f'{t("vt_threats").format(malicious=vt_result["malicious"], total=vt_result["total"])}</span></a>'
                    )
                else:
                    # "unknown" — file not in VT database
                    vt_pill = (
                        f'<a href="{vt_link}" target="_blank" style="text-decoration:none;">'
                        f'<span style="background:#555; color:#ccc; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.78rem;">'
                        f'{t("vt_unknown")}</span></a>'
                    )
                file_rows += f"""
            <div style="display:flex; align-items:center; padding:6px 0;">
                <span style="{label_style}">{t("label_virustotal")}</span>
                {vt_pill}
            </div>
            """

            st.markdown(f"""
            <div style="margin-top:0.5rem;">
                <span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;
                    letter-spacing:0.5px;">{t("file_properties")}</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">
                {file_rows}
            </div>
            """, unsafe_allow_html=True)

        # =================================================================
        # RIGHT COLUMN: Analysis Results
        # =================================================================
        with results_col:
            # Print / "Save as PDF" (roadmap #8, tier 1). The report header is
            # print-only: on screen the same facts sit in the file-properties
            # block further down, but a printed page has no URL bar and no
            # filename bar to say what it is about, so they come first there.
            # The script opens every <details> before printing (CSS cannot)
            # and restores the reader's state afterwards.
            _analysed_at = st.session_state.get("cached_analysed_at", "")
            st.html(f"""
            <div class="tf-print-only tf-print-keep" style="border-bottom:2px solid #000; padding-bottom:6px; margin-bottom:14px;">
                <div style="font-size:1.1rem; font-weight:bold;">{t("print_report_title")}</div>
                <div style="font-size:0.8rem;">{t("print_file")} {esc(uploaded_file.name)}</div>
                <div style="font-size:0.8rem;">SHA-256 <code>{esc(result.file_hash)}</code></div>
                <div style="font-size:0.8rem;">{t("print_date")} {_analysed_at} &middot; {t("print_generated_by")} https://www.trustyfile.com</div>
            </div>
            <div id="tf-print-wrap" style="display:flex; justify-content:flex-end; margin:0 0 0.2rem 0;">
                <button id="tf-print" type="button" style="background:transparent; color:#888; border:1px solid #2a2a3e; border-radius:6px; padding:4px 10px; font-size:0.75rem; cursor:pointer;">
                    {t("print_report")}
                </button>
            </div>
            <script>
            (function() {{
                const btn = document.getElementById('tf-print');
                if (btn && !btn.dataset.bound) {{
                    btn.dataset.bound = '1';
                    btn.addEventListener('click', function() {{ window.print(); }});
                }}
                if (!window.__tfPrintHooks) {{
                    window.__tfPrintHooks = true;
                    window.addEventListener('beforeprint', function() {{
                        document.querySelectorAll('details:not([open])').forEach(function(d) {{
                            d.dataset.tfAutoOpen = '1';
                            d.open = true;
                        }});
                    }});
                    window.addEventListener('afterprint', function() {{
                        document.querySelectorAll('details[data-tf-auto-open]').forEach(function(d) {{
                            d.open = false;
                            delete d.dataset.tfAutoOpen;
                        }});
                    }});
                }}
            }})();
            </script>
            """, unsafe_allow_javascript=True)

            # Main score gauge (circular arc like HubSpot Website Grader)
            score = result.trust_score
            risk_level = result.risk_level

            # Color based on score. UNVERIFIABLE carries score 0 internally
            # (the coverage gate fails closed), but a giant bold "0" reads as
            # "maximally fraudulent" to a user whose scan simply had no text
            # layer. Show a neutral glyph and a full grey ring instead: "not
            # measured", not "empty tank".
            gauge_number = str(score)
            if risk_level == "UNVERIFIABLE":
                gauge_color = "#6c757d"   # neutral grey: could not verify (not fraud)
                gauge_number = t("score_not_available")
            elif score >= 80:
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
            if risk_level == "UNVERIFIABLE":
                filled = circumference          # full grey ring: nothing measured
            else:
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
                        font-size="48" font-weight="bold" fill="white">{gauge_number}</text>
                    <text x="{center}" y="{center + 22}" text-anchor="middle"
                        font-size="14" fill="{gauge_color}" font-weight="bold">
                        {t("trusted") if score >= 95 else t(f"risk_{risk_level}")}</text>
                </svg>
                <p style="color:#777; font-size:0.72rem; line-height:1.4; max-width:340px; margin:0.2rem auto 0.4rem auto;">
                    {t("score_caveat")}
                </p>
                <p style="color:#c9d1d9; font-size:0.78rem; line-height:1.45; max-width:340px; margin:0 auto 0.9rem auto;">
                    <strong>{t("next_step_label")}</strong> {t(f"next_step_{result.band}")}
                </p>
            </div>
            """, unsafe_allow_html=True)

            _dt = result.document_type

            # Grouped score bars (4 categories instead of 7 modules)
            # Map module names to their results for easy lookup
            module_map = {m.module: m for m in result.modules}

            # Define the 4 groups: (label, list of module names, tooltip)
            groups = [
                (t("group_modifications"), ["structure", "overlay"], t("tip_modifications")),
                (t("group_consistency"), ["content", "arithmetic"], t("tip_consistency")),
                (t("group_tampering"), ["images", "fonts", "visual", "metadata", "forensics"], t("tip_tampering")),
                (t("group_authenticity"), ["external", "identity", "twod_doc"], t("tip_authenticity")),
            ]

            # Evidence-ledger contributions (audit item 23): the bars show the
            # ACTUAL score pressure per group — the same group-capped sums the
            # aggregation used — not the old (deleted) weighted-average formula.
            from src.evidence import module_contributions
            contrib = module_contributions(result.evidence_ledger)

            def _group_net(names: list[str]) -> float:
                return sum(contrib.get(n, {}).get("tamper", 0.0)
                           - contrib.get(n, {}).get("genuine", 0.0)
                           for n in names)

            total_pressure = sum(max(0.0, _group_net(names))
                                 for _, names, _ in groups) or 0.0

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
                            <span class="tf-label" tabindex="0" style="color:#555;">{label}<span class="tf-tip">{tooltip}</span></span>
                        </span>
                        <div style="flex:1; background:#1a1a1a; border-radius:4px;
                            height:10px; overflow:hidden;">
                        </div>
                        <span style="width:35px; text-align:right; font-size:0.8rem;
                            color:#555; margin-left:8px;">N/A</span>
                    </div>'''
                    continue

                # Net evidence pressure for this group (positive = pushed the
                # score DOWN; negative = genuine evidence lifted it).
                net = _group_net(member_names)
                hard = any(contrib.get(n, {}).get("hard_tamper") for n in member_names)

                if hard:
                    width, bar_color, value_label = 100, "#ef233c", "veto"
                elif net > 0:
                    width = round(100 * net / total_pressure) if total_pressure else 0
                    bar_color = "#ef233c" if net >= 1.5 else "#ffb627"
                    value_label = f"+{net:.1f}"
                elif net < 0:
                    width, bar_color, value_label = 25, "#28a745", f"{net:.1f}"
                else:
                    width, bar_color, value_label = 0, "#28a745", "0"

                bars_html += f'''
                <div style="display:flex; align-items:center; margin-bottom:6px;">
                    <span style="width:100px; flex-shrink:0;">
                        <span class="tf-label" tabindex="0" style="color:#eee;">{label}<span class="tf-tip">{tooltip}</span></span>
                    </span>
                    <div style="flex:1; background:#2a2a2a; border-radius:4px;
                        height:10px; overflow:hidden;">
                        <div style="width:{width}%; height:100%;
                            background:{bar_color}; border-radius:4px;
                            transition:width 1s ease;"></div>
                    </div>
                    <span style="width:35px; text-align:right; font-size:0.8rem;
                        color:#eee; margin-left:8px; font-weight:bold;">{value_label}</span>
                </div>'''

            # Flag counts
            all_flags = collect_all_flags(result.modules)
            counts = count_flags_by_severity(all_flags)

            # Issues pills (right below the circle)
            pills_html = ""
            if counts["critical"]:
                pills_html += f'<span style="background:#ef233c; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["critical"]} {t("severity_critical")}</span> '
            if counts["high"]:
                pills_html += f'<span style="background:#ff6b35; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["high"]} {t("severity_high")}</span> '
            if counts["medium"]:
                pills_html += f'<span style="background:#ffb627; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["medium"]} {t("severity_medium")}</span> '
            if counts["low"]:
                pills_html += f'<span style="background:#555; color:#ccc; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{counts["low"]} {t("severity_low")}</span> '

            if not any(counts.values()):
                pills_html = f'<span style="background:#28a745; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{t("no_issues")}</span>'

            _dt_label = t(f"doctype_{_dt}")
            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center; margin-top:0.3rem;">{pills_html}</div>'
                f'<p style="text-align:center; font-size:0.7rem; color:#888; margin-top:0.4rem;">{t("analysis_completed").format(ms=result.analysis_time_ms)}</p>'
                f'<div style="text-align:center; margin-top:8px;">'
                f'<span style="background:#1e1e1e; border:1px solid #333; border-radius:20px;'
                f' padding:4px 14px; font-size:0.8rem; color:#aaa;">'
                f'{t("doctype_label")}&nbsp;:&nbsp;'
                f'<strong style="color:#e0e0e0;">{_dt_label}</strong>'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            # Module score bars
            st.markdown(f'<div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">{bars_html}</div>',
                        unsafe_allow_html=True)

            # Summary block — verdict + flag list + document type
            _sev_colors = {
                "critical": "#ef233c",
                "high": "#ff6b35",
                "medium": "#ffb627",
                "low": "#555",
            }

            import html as _html_esc
            flags_list_html = ""
            if all_flags:
                for flag in all_flags:
                    fc = _sev_colors.get(flag.severity, "#555")
                    safe_sentence = _html_esc.escape(get_flag_sentence(flag, get_lang()))
                    flags_list_html += (
                        f'<div style="display:flex; align-items:center; gap:8px; padding:5px 0; '
                        f'border-bottom:1px solid #2a2a2a;">'
                        f'<span style="background:{fc}; color:white; font-size:0.62rem; '
                        f'font-weight:bold; padding:2px 0; border-radius:8px; width:58px; '
                        f'text-align:center; display:inline-block; flex-shrink:0;">{t(f"severity_{flag.severity}")}</span>'
                        f'<span style="color:#ccc; font-size:0.85rem;">{safe_sentence}</span>'
                        f'</div>'
                    )

            _secondary_types = {"scientific", "legal_notice", "technical", "unknown"}
            _banner_html = ""
            if _dt in _secondary_types:
                _banner_html = (
                    f'<div style="padding:8px 14px; margin:8px 0 0.8rem 0;'
                    f' background:#1a2a3a; border-left:3px solid #4a9ade; border-radius:4px;">'
                    f'<span style="font-size:0.75rem; color:#ffffff;">{t(f"doctype_banner_{_dt}")}</span>'
                    f'</div>'
                )

            _summary_html = (
                '<div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">'
                f'<span style="color:white; font-size:1.3rem; font-weight:bold; text-transform:uppercase;'
                f' letter-spacing:0.5px; display:block; text-align:center;">{t("summary")}</span>'
                '<hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">'
                f'<p style="color:white; font-size:1.03rem; font-weight:bold; margin:0 0 0.8rem 0;">{summary.verdict}</p>'
                f'{_banner_html}'
                '<hr style="border:none; border-top:1px solid #2a2a2a; margin:0 0 0.8rem 0;">'
                f'{flags_list_html}'
                '</div>'
                '<div style="margin-bottom:0.5rem;"></div>'
            )
            st.markdown(_summary_html, unsafe_allow_html=True)


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
                    software_html = f'<div style="color:#888; font-size:0.7rem;">{t("label_producer")}: {html_mod.escape(first_software)}</div>' if first_software else ""
                    history_rows += (
                        f'<div style="{row_style}">'
                        f'<span style="{label_style}">{t("version_n").format(n=1)}</span>'
                        f'<div style="{value_style}">'
                        f'<div>{pdf_data.metadata.creation_date.strftime("%Y-%m-%d %H:%M")} — {t("original_document")}</div>'
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
                            _ac = t("after_creation")
                            if hours >= 24:
                                time_info += f" — +{int(hours // 24)}d {int(hours % 24)}h {_ac}"
                            elif hours >= 1:
                                time_info += f" — +{int(hours)}h{int((hours % 1) * 60)}min {_ac}"
                            elif int(hours * 60) > 0:
                                time_info += f" — +{int(hours * 60)}min {_ac}"
                            else:
                                time_info += f" — +{int(delta.total_seconds())}s {_ac}"

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
                            changes_html += f'<div style="color:#888; font-size:0.7rem; font-style:italic;">{t("and_n_more_lines").format(n=remaining)}</div>'
                    else:
                        changes_html = f'<div style="color:#888; font-size:0.75rem;">{t("non_text_changes")}</div>'

                    software_html = f'<div style="color:#888; font-size:0.7rem;">{t("label_editor")}: {html_mod.escape(version_software)}</div>' if version_software else ""
                    history_rows += (
                        f'<div style="{row_style}">'
                        f'<span style="{label_style}">{t("version_n").format(n=version_num)}</span>'
                        f'<div style="{value_style}">'
                        f'<div>{time_info if time_info else t("unknown_time")}</div>'
                        f'{software_html}'
                        f'{changes_html}'
                        f'</div></div>'
                    )

                st.markdown(
                    f'<div style="margin-top:1.5rem; max-width:440px; margin-left:auto; margin-right:auto;">'
                    f'<span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;'
                    f' letter-spacing:0.5px;">{t("full_modification_history")}</span>'
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
            st.markdown(f"## {t('detected_issues')}")

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
                safe_msg = html_mod_flags.escape(get_flag_sentence(flag, get_lang()))
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

                # Identifier flags (IBAN / SIRET / SIREN / VAT) link to the free
                # checker page: the reader can re-type the number from the
                # document — or a corrected one — and see the check digit
                # explained, without uploading anything.
                tool_url = tool_url_for_flag(flag.code, get_lang())
                if tool_url:
                    details_content += (
                        f'<div style="margin-top:8px;"><a href="{tool_url}" target="_blank" rel="noopener" '
                        f'style="color:#8ab4f8; font-size:0.8rem; text-decoration:none;">'
                        f'{html_mod_flags.escape(t("flag_check_tool"))} &rarr;</a></div>'
                    )

                issues_html += (
                    f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                    f'border-radius:8px; padding:0; margin-bottom:8px;">'
                    f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                    f'display:flex; align-items:center; gap:10px;">'
                    f'<span style="background:{color}; color:white; font-size:0.68rem; '
                    f'font-weight:bold; padding:2px 0; border-radius:10px; '
                    f'width:70px; text-align:center; flex-shrink:0; '
                    f'display:inline-block;">{t(f"severity_{flag.severity}")}</span>'
                    f'<span style="color:#eee; font-size:0.85rem;">{safe_msg}</span>'
                    f'</summary>'
                    f'<div style="padding:4px 14px 14px 14px; border-top:1px solid #2a2a2a;">'
                    f'{details_content}'
                    f'</div></details>'
                )

            st.markdown(issues_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"## {t('more_details')}")

        more_details_html = ""

        # Full modification history (collapsible)
        if history["version_count"] > 1 and history.get("diffs"):
            import html as html_mod

            # Reusable styles
            _card = ("background:#111; border:1px solid #2a2a2a; border-radius:8px; "
                     "padding:14px 18px; margin-bottom:8px;")
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
                v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_date")}</span>'
                            f'<span style="{_mval}">{pdf_data.metadata.creation_date.strftime("%Y-%m-%d %H:%M")}</span></div>')
            if first_sw:
                v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_producer")}</span>'
                            f'<span style="{_mval}">{html_mod.escape(first_sw)}</span></div>')

            hist_content += (
                f'<div style="{_card}">'
                f'<p style="{_vlabel}">{t("version_n").format(n=1)} '
                f'<span style="background:#28a745; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px; margin-left:6px;">{t("original")}</span></p>'
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
                    v_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_date")}</span>'
                               f'<span style="{_mval}">{pdf_data.metadata.mod_date.strftime("%Y-%m-%d %H:%M")} '
                               f'<span style="color:#e06060;">({delta_str} {t("after_creation")})</span></span></div>')

                if to_sw and to_sw != from_sw:
                    v_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_editor")}</span>'
                               f'<span style="{_mval}">{html_mod.escape(to_sw)}</span></div>')

                diff_html = ""
                if diff["changes"]:
                    for change in diff["changes"]:
                        diff_html += f'<div style="color:#888; font-size:0.7rem; margin-bottom:4px;">{t("label_page")} {change["page"]}</div>'
                        for line in change.get("removed", []):
                            if line.strip():
                                diff_html += f'<div style="color:#e06060;">- {html_mod.escape(line)}</div>'
                        for line in change.get("added", []):
                            if line.strip():
                                diff_html += f'<div style="color:#60c060;">+ {html_mod.escape(line)}</div>'
                    diff_html = f'<div style="{_diff}">{diff_html}</div>'
                else:
                    diff_html = f'<div style="color:#888; font-size:0.78rem; font-style:italic; margin-top:8px;">{t("no_text_changes")}</div>'

                obj_html = ""
                obj_changes = diff.get("object_changes", [])
                if obj_changes:
                    by_type: dict[str, list] = {}
                    for oc in obj_changes:
                        by_type.setdefault(oc.get("type", "unknown"), []).append(oc)
                    badges = "".join(f'<span style="{_badge}">{obj_t} ({len(items)})</span>' for obj_t, items in by_type.items())
                    obj_html = f'<div style="margin-top:10px;"><span style="color:#888; font-size:0.7rem;">{t("modified_objects")}</span>{badges}</div>'

                hist_content += (
                    f'<div style="{_card}">'
                    f'<p style="{_vlabel}">{t("version_n").format(n=to_v)} '
                    f'<span style="background:#ef233c; color:white; font-size:0.68rem; font-weight:bold; '
                    f'padding:2px 10px; border-radius:10px; margin-left:6px;">{t("modified")}</span></p>'
                    f'{v_meta}{diff_html}{obj_html}</div>'
                )

            version_count = history["version_count"]
            more_details_html += (
                f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">{t("detailed_modification_history")}</span>'
                f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{t("n_versions").format(n=version_count)}</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                f'{hist_content}'
                f'</div></details>'
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
            flags_html = ""
            if module.flags:
                for f in module.flags:
                    f_color = severity_colors.get(f.severity, "#555")
                    safe_code = html_mod_raw.escape(f.code)
                    safe_msg = html_mod_raw.escape(get_flag_sentence(f, get_lang()))

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
                        f'text-align:center; display:inline-block; flex-shrink:0;">{t(f"severity_{f.severity}")}</span>'
                        f'<span style="color:#ccc; font-size:0.78rem;">{safe_msg}</span>'
                        f'</div>'
                        f'<div style="color:#666; font-size:0.72rem; margin-top:3px; padding-left:66px;">{safe_code}</div>'
                        f'{f"<div style=padding-left:66px;>{details_html}</div>" if details_html else ""}'
                        f'</div>'
                    )
            else:
                flags_html = f'<div style="color:#666; font-size:0.78rem; padding:4px 0;">{t("no_flags")}</div>'

            raw_content += (
                f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                f'padding:12px 16px; margin-bottom:8px;">'
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
                f'<span style="color:white; font-size:0.88rem; font-weight:bold;">{mod_name}</span>'
                f'<span style="background:{sc_color}; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{mod_score}/100</span>'
                f'</div>'
                # module.confidence (0.3-1.0, e.g. metadata 0.3 when a PDF has no
                # metadata) is deliberately not shown here: the evidence ledger
                # never reads it (flag_to_evidence pins confidence to 1.0), so a
                # "Fiabilité : 30 %" next to the module score had no bearing on
                # the verdict and read as a document-reliability figure it is not.
                f'<div style="display:flex; gap:16px; color:#888; font-size:0.75rem; margin-bottom:8px; '
                f'padding-bottom:8px; border-bottom:1px solid #1a1a1a;">'
                f'<span>{t("label_flags")}: {len(module.flags)}</span>'
                f'</div>'
                f'{flags_html}'
                f'</div>'
            )

        # Analysis Details — collapsible block with per-module raw data
        module_count = len(result.modules)
        more_details_html += (
            f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
            f'border-radius:8px; padding:0; margin-bottom:8px;">'
            f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
            f'display:flex; align-items:center; gap:10px;">'
            f'<span style="color:#eee; font-size:0.85rem;">{t("analysis_details")}</span>'
            f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
            f'padding:2px 10px; border-radius:10px;">{t("n_modules").format(n=module_count)}</span>'
            f'</summary>'
            f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
            f'{raw_content}'
            f'</div></details>'
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
                status_pill = (f'<span style="background:#28a745; color:white; font-size:0.62rem; '
                               f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_active")}</span>')
            elif status == "closed":
                status_pill = (f'<span style="background:#ef233c; color:white; font-size:0.62rem; '
                               f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_closed")}</span>')
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
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_address")}</span><span style="{_val}">{address}</span></div>'
            if creation:
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_creation_date")}</span><span style="{_val}">{creation}</span></div>'

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
        not_verified_pill = (f'<span style="background:#555; color:#ccc; font-size:0.62rem; '
                             f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_not_verified")}</span>')
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
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_address")}</span><span style="{_val}">{html_mod_ent.escape(entity_address)}</span></div>'

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
            more_details_html += (
                f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">{t("entities")}</span>'
                f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{t("n_entities").format(n=entity_count)}</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                f'{entities_html}'
                f'</div></details>'
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
                    status_pill = (f'<span style="background:#28a745; color:white; font-size:0.62rem; '
                                   f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_valid")}</span>')
                else:
                    status_pill = (f'<span style="background:#ef233c; color:white; font-size:0.62rem; '
                                   f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_invalid")}</span>')

                # Format IBAN with spaces for readability (groups of 4)
                iban_display = " ".join([iban_clean[i:i+4] for i in range(0, len(iban_clean), 4)])

                rows = f'<div style="{_row}"><span style="{_key}">IBAN</span><span style="{_val}">{iban_display}</span></div>'
                bic = iban_data.get("bic", "")
                if bic:
                    rows += f'<div style="{_row}"><span style="{_key}">BIC</span><span style="{_val}">{bic}</span></div>'
                bank_name = iban_data.get("bank_name") or ""
                if bank_name:
                    rows += f'<div style="{_row}"><span style="{_key}">{t("label_bank")}</span><span style="{_val}">{html_mod_ent.escape(bank_name)}</span></div>'
                branch_name = iban_data.get("branch_name") or ""
                if branch_name:
                    rows += f'<div style="{_row}"><span style="{_key}">{t("label_branch")}</span><span style="{_val}">{html_mod_ent.escape(branch_name)}</span></div>'
                country = iban_data.get("country_code", "")
                if country:
                    rows += f'<div style="{_row}"><span style="{_key}">{t("label_country")}</span><span style="{_val}">{country}</span></div>'

                import unicodedata
                title_raw = unicodedata.normalize("NFD", bank_name)
                title_raw = "".join(c for c in title_raw if unicodedata.category(c) != "Mn")
                title = html_mod_ent.escape(title_raw).upper() if bank_name else t("bank_account").upper()
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
            not_verified_pill = (f'<span style="background:#555; color:#ccc; font-size:0.62rem; '
                                 f'font-weight:bold; padding:2px 8px; border-radius:8px;">{t("status_not_verified")}</span>')
            for iban_str, is_checksum_valid, context in found_ibans:
                if iban_str in shown_ibans:
                    continue
                shown_ibans.add(iban_str)
                bank_count += 1

                iban_display = " ".join([iban_str[i:i+4] for i in range(0, len(iban_str), 4)])
                checksum_note = t("valid_checksum") if is_checksum_valid else t("invalid_checksum")

                rows = f'<div style="{_row}"><span style="{_key}">IBAN</span><span style="{_val}">{iban_display}</span></div>'
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_country")}</span><span style="{_val}">{iban_str[:2]}</span></div>'
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_check")}</span><span style="{_val}">{checksum_note}</span></div>'

                bank_html += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
                    f'<span style="color:white; font-size:0.88rem; font-weight:bold;">{t("bank_account")}</span>'
                    f'{not_verified_pill}'
                    f'</div>'
                    f'{rows}'
                    f'</div>'
                )

            if bank_count > 0:
                more_details_html += (
                    f'<details style="background:#1a1a2e; border:1px solid #2a2a2a; '
                    f'border-radius:8px; padding:0; margin-bottom:8px;">'
                    f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                    f'display:flex; align-items:center; gap:10px;">'
                    f'<span style="color:#eee; font-size:0.85rem;">{t("bank_information")}</span>'
                    f'<span style="background:#2563eb; color:white; font-size:0.68rem; font-weight:bold; '
                    f'padding:2px 10px; border-radius:10px;">{t("n_accounts_plural").format(n=bank_count) if bank_count > 1 else t("n_accounts").format(n=bank_count)}</span>'
                    f'</summary>'
                    f'<div style="padding:14px; border-top:1px solid #2a2a2a;">'
                    f'{bank_html}'
                    f'</div></details>'
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
                rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_doc_type")}</span>'
                            f'<span style="{_2d_val}">{html_mod_2d.escape(verified.document_type)} '
                            f'({html_mod_2d.escape(verified.document_type_name)})</span></div>')

                # Track which DIs we display so we can show remaining as raw
                shown_dis = set()

                # --- Identity fields (shared across document types) ---
                if verified.name:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_name")}</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.name)}</span></div>')
                    shown_dis.update({"10", "12", "13", "44", "6G", "60"})

                # --- Driving license fields (type AB) ---
                if verified.civility:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_civility")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.civility)}</span></div>')
                    shown_dis.add("6H")

                if verified.sex:
                    sex_label = {"M": t("twod_sex_male"), "F": t("twod_sex_female")}.get(verified.sex, verified.sex)
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_sex")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(sex_label)}</span></div>')
                    shown_dis.add("68")

                if verified.nationality:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_nationality")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.nationality)}</span></div>')
                    shown_dis.update({"67", "6C"})

                if verified.birth_date:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_birth_date")}</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.birth_date)}</span></div>')
                    shown_dis.add("69")

                if verified.birth_place:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_birth_place")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.birth_place)}</span></div>')
                    shown_dis.add("6A")

                if verified.document_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_document_number")}</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.document_number)}</span></div>')
                    shown_dis.add("65")

                if verified.permit_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_permit_number")}</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(verified.permit_number)}</span></div>')
                    shown_dis.add("AC")

                if verified.permit_categories:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_permit_categories")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.permit_categories)}</span></div>')
                    shown_dis.add("E4")

                # --- Tax fields ---
                if verified.fiscal_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_tax_id")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.fiscal_number)}</span></div>')
                    shown_dis.add("47")

                if verified.address or verified.postal_code or verified.city:
                    addr_parts = []
                    if verified.address:
                        addr_parts.append(verified.address)
                    if verified.postal_code or verified.city:
                        addr_parts.append(f"{verified.postal_code or ''} {verified.city or ''}".strip())
                    addr = ", ".join(addr_parts)
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_address")}</span>'
                                f'<span style="{_2d_val_bold}">{html_mod_2d.escape(addr)}</span></div>')
                    shown_dis.update({"22", "24", "25"})

                if verified.reference_income is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_reference_income")}</span>'
                                f'<span style="{_2d_val_bold}">{verified.reference_income:,.0f} &euro;</span></div>')
                    shown_dis.add("41")

                if verified.tax_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_income_tax")}</span>'
                                f'<span style="{_2d_val_bold}">{verified.tax_amount:,.0f} &euro;</span></div>')
                    shown_dis.add("4V")

                if verified.withheld_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_withholding_tax")}</span>'
                                f'<span style="{_2d_val}">{verified.withheld_amount:,.0f} &euro;</span></div>')
                    shown_dis.add("4X")

                if verified.calculated_balance is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_calculated_balance")}</span>'
                                f'<span style="{_2d_val_bold}">{verified.calculated_balance:,.0f} &euro;</span></div>')

                if verified.household_parts and verified.household_parts != 1.0:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_household_parts")}</span>'
                                f'<span style="{_2d_val}">{verified.household_parts}</span></div>')
                    shown_dis.add("43")

                # --- Invoice fields ---
                if verified.invoice_number:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_invoice_number")}</span>'
                                f'<span style="{_2d_val}">{html_mod_2d.escape(verified.invoice_number)}</span></div>')

                if verified.invoice_amount is not None:
                    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_invoice_amount")}</span>'
                                f'<span style="{_2d_val_bold}">{verified.invoice_amount:,.2f} &euro;</span></div>')

                # Show remaining unmapped fields as raw
                extra_fields = ""
                for field in twod_doc_data.fields:
                    if field.di in shown_dis:
                        continue
                    field_val = html_mod_2d.escape(str(field.value))
                    extra_fields += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_field").format(di=field.di)}</span>'
                                     f'<span style="{_2d_val}">{field_val}</span></div>')

                if extra_fields:
                    rows_2d += (f'<div style="margin-top:8px; padding-top:6px; border-top:1px solid #2a2a2a;">'
                                f'<span style="color:#888; font-size:0.7rem; font-weight:bold;">{t("twod_raw_fields")}</span>'
                                f'</div>{extra_fields}')

                twod_content += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'{rows_2d}</div>'
                )

            more_details_html += (
                f'<details open style="background:#0a2e1a; border:1px solid #28a745; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">{t("twod_doc")}</span>'
                f'<span style="background:#28a745; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{t("twod_verified")}</span>'
                f'<span style="color:#888; font-size:0.72rem;">{t("twod_compare")}</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #28a745;">'
                f'{twod_content}'
                f'</div></details>'
            )

        # MRZ Zone panel (identity documents)
        identity_mod = next((m for m in result.modules if m.module == "identity"), None)
        if identity_mod and identity_mod.details.get("mrz_found"):
            import html as html_mrz
            from datetime import date as _d

            det = identity_mod.details
            mrz_format = det.get("mrz_format", "")
            # Label du type de document selon le format MRZ
            if mrz_format == "TD3":
                doc_type_label = t("mrz_type_passport")
            elif mrz_format == "TD1":
                doc_type_label = t("mrz_type_id_card")
            else:
                doc_type_label = mrz_format

            # --- Badge selon les flags du module identity ---
            identity_flags = identity_mod.flags
            has_high = any(f.severity in ("high", "critical") for f in identity_flags)
            has_medium = any(f.severity == "medium" for f in identity_flags)
            if has_high:
                badge_bg = "#ef233c"
                badge_label = t("mrz_badge_suspect")
            elif has_medium:
                badge_bg = "#f59e0b"
                badge_label = t("mrz_badge_warning")
            else:
                badge_bg = "#28a745"
                badge_label = t("mrz_badge_valid")

            _mrz_row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
            _mrz_key = "color:#888; font-size:0.75rem; width:160px; flex-shrink:0;"
            _mrz_val = "color:#ccc; font-size:0.75rem;"
            _mrz_val_bold = "color:white; font-size:0.75rem; font-weight:bold;"
            _mrz_ok = "color:#28a745; margin-right:4px;"
            _mrz_warn = "color:#f59e0b; margin-right:4px;"
            _mrz_bad = "color:#ef233c; margin-right:4px;"

            # --- Champs principaux MRZ ---
            mrz_rows = ""

            # Nom
            if det.get("name"):
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_name")}</span>'
                    f'<span style="{_mrz_val_bold}">{html_mrz.escape(det["name"])}</span>'
                    f'</div>'
                )

            # Date de naissance + âge
            if det.get("date_of_birth"):
                try:
                    dob = _d.fromisoformat(det["date_of_birth"])
                    age = (_d.today() - dob).days // 365
                    dob_str = f"{dob.strftime('%d/%m/%Y')} ({age} {t('mrz_years')})"
                except Exception:
                    dob_str = det["date_of_birth"]
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_dob")}</span>'
                    f'<span style="{_mrz_val_bold}">{html_mrz.escape(dob_str)}</span>'
                    f'</div>'
                )

            # Nationalité
            if det.get("nationality"):
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_nationality")}</span>'
                    f'<span style="{_mrz_val}">{html_mrz.escape(det["nationality"])}</span>'
                    f'</div>'
                )

            # Sexe
            if det.get("sex"):
                sex_map = {"M": t("mrz_sex_male"), "F": t("mrz_sex_female"), "<": t("mrz_sex_unspecified")}
                sex_label = sex_map.get(det["sex"], det["sex"])
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_sex")}</span>'
                    f'<span style="{_mrz_val}">{html_mrz.escape(sex_label)}</span>'
                    f'</div>'
                )

            # Numéro de document
            if det.get("doc_number"):
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_doc_number")}</span>'
                    f'<span style="{_mrz_val_bold}">{html_mrz.escape(det["doc_number"])}</span>'
                    f'</div>'
                )

            # Date de délivrance (extraite par OCR pleine page — non encodée dans la MRZ)
            if det.get("issuance_date"):
                try:
                    from datetime import date as _d_iso
                    issuance_display = _d_iso.fromisoformat(det["issuance_date"]).strftime('%d/%m/%Y')
                except Exception:
                    issuance_display = det["issuance_date"]
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_issuance")}</span>'
                    f'<span style="{_mrz_val}">{html_mrz.escape(issuance_display)}</span>'
                    f'</div>'
                )

            # Date d'expiration + badge VALIDE/EXPIRÉ
            if det.get("expiry_date"):
                try:
                    expiry = _d.fromisoformat(det["expiry_date"])
                    expiry_str = expiry.strftime('%d/%m/%Y')
                    if expiry >= _d.today():
                        exp_badge = (
                            f'<span style="background:#28a745; color:white; font-size:0.65rem; '
                            f'font-weight:bold; padding:1px 8px; border-radius:8px; margin-left:6px;">'
                            f'{t("mrz_valid")}</span>'
                        )
                    else:
                        exp_badge = (
                            f'<span style="background:#ef233c; color:white; font-size:0.65rem; '
                            f'font-weight:bold; padding:1px 8px; border-radius:8px; margin-left:6px;">'
                            f'{t("mrz_expired")}</span>'
                        )
                except Exception:
                    expiry_str = det["expiry_date"]
                    exp_badge = ""
                mrz_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_expiry")}</span>'
                    f'<span style="{_mrz_val_bold}">{html_mrz.escape(expiry_str)}{exp_badge}</span>'
                    f'</div>'
                )

            # --- Section cohérence MRZ ---
            coherence = det.get("mrz_coherence", {})
            coherence_rows = ""
            if coherence:
                # Type document
                dt_valid = coherence.get("doc_type_valid", True)
                dt_prefix = coherence.get("doc_type_prefix", "")
                dt_icon = f'<span style="{_mrz_ok}">&#10003;</span>' if dt_valid else f'<span style="{_mrz_warn}">&#9651;</span>'
                dt_text = t("mrz_type_passport") if mrz_format == "TD3" else t("mrz_type_id_card")
                coherence_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_coherence_doc_type")}</span>'
                    f'<span style="{_mrz_val}">{dt_icon}{html_mrz.escape(dt_text)} ({html_mrz.escape(dt_prefix)})</span>'
                    f'</div>'
                )

                # Code émetteur
                issuer = coherence.get("issuer", "")
                issuer_valid = coherence.get("issuer_format_valid", True)
                is_icon = f'<span style="{_mrz_ok}">&#10003;</span>' if issuer_valid else f'<span style="{_mrz_warn}">&#9651;</span>'
                if issuer:
                    coherence_rows += (
                        f'<div style="{_mrz_row}">'
                        f'<span style="{_mrz_key}">{t("mrz_coherence_issuer")}</span>'
                        f'<span style="{_mrz_val}">{is_icon}{html_mrz.escape(issuer.replace("<", ""))} '
                        f'({t("mrz_format_valid") if issuer_valid else t("mrz_format_invalid")})</span>'
                        f'</div>'
                    )

                # Période de validité
                validity_years = coherence.get("validity_period_years")
                validity_ok = coherence.get("validity_plausible", True)
                if validity_years is not None:
                    vp_icon = f'<span style="{_mrz_ok}">&#10003;</span>' if validity_ok else f'<span style="{_mrz_warn}">&#9651;</span>'
                    expected_range = "5–10 " + t("mrz_years") if mrz_format == "TD3" else "10–15 " + t("mrz_years")
                    coherence_rows += (
                        f'<div style="{_mrz_row}">'
                        f'<span style="{_mrz_key}">{t("mrz_coherence_validity")}</span>'
                        f'<span style="{_mrz_val}">{vp_icon}{validity_years} {t("mrz_years")} '
                        f'({t("mrz_expected")}: {expected_range})</span>'
                        f'</div>'
                    )

            # --- Section authenticité du scan ---
            photo = det.get("photo_analysis", {})
            photo_rows = ""
            if photo.get("ela_run"):
                ela_ok = not photo.get("is_ela_suspicious", False)
                ela_ratio_pct = photo.get("suspicious_ratio", 0) * 100
                ela_icon = f'<span style="{_mrz_ok}">&#10003;</span>' if ela_ok else f'<span style="{_mrz_warn}">&#9651;</span>'
                ela_text = (
                    t("mrz_ela_ok").format(pct=f"{ela_ratio_pct:.0f}")
                    if ela_ok
                    else t("mrz_ela_suspect").format(pct=f"{ela_ratio_pct:.0f}")
                )
                photo_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">ELA</span>'
                    f'<span style="{_mrz_val}">{ela_icon}{html_mrz.escape(ela_text)}</span>'
                    f'</div>'
                )

                noise_ok = photo.get("noise_uniform", True)
                noise_cv = photo.get("noise_cv", 0)
                noise_icon = f'<span style="{_mrz_ok}">&#10003;</span>' if noise_ok else f'<span style="{_mrz_warn}">&#9651;</span>'
                noise_text = (
                    t("mrz_noise_ok")
                    if noise_ok
                    else t("mrz_noise_suspect").format(cv=f"{noise_cv:.2f}")
                )
                photo_rows += (
                    f'<div style="{_mrz_row}">'
                    f'<span style="{_mrz_key}">{t("mrz_noise")}</span>'
                    f'<span style="{_mrz_val}">{noise_icon}{html_mrz.escape(noise_text)}</span>'
                    f'</div>'
                )

            # --- Assembler le panneau complet ---
            mrz_inner = (
                f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                f'padding:12px 16px; margin-bottom:8px;">'
                f'{mrz_rows}'
                f'</div>'
            )

            if coherence_rows:
                mrz_inner += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px;">'
                    f'<p style="color:#888; font-size:0.72rem; font-weight:bold; margin:0 0 6px 0;">'
                    f'{t("mrz_coherence_title")}</p>'
                    f'{coherence_rows}'
                    f'</div>'
                )

            if photo_rows:
                mrz_inner += (
                    f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:0;">'
                    f'<p style="color:#888; font-size:0.72rem; font-weight:bold; margin:0 0 6px 0;">'
                    f'{t("mrz_scan_auth_title")}</p>'
                    f'{photo_rows}'
                    f'</div>'
                )

            more_details_html += (
                f'<details open style="background:#0a1e3a; border:1px solid #2563eb; '
                f'border-radius:8px; padding:0; margin-bottom:8px;">'
                f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
                f'display:flex; align-items:center; gap:10px;">'
                f'<span style="color:#eee; font-size:0.85rem;">{t("mrz_zone_title")}</span>'
                f'<span style="background:{badge_bg}; color:white; font-size:0.68rem; font-weight:bold; '
                f'padding:2px 10px; border-radius:10px;">{badge_label}</span>'
                f'<span style="color:#888; font-size:0.72rem;">{html_mrz.escape(doc_type_label)}</span>'
                f'</summary>'
                f'<div style="padding:14px; border-top:1px solid #2563eb;">'
                f'{mrz_inner}'
                f'</div></details>'
            )

        # Render all More Details sections in a single st.markdown
        if more_details_html:
            st.markdown(more_details_html, unsafe_allow_html=True)

    except Exception:
        # The analyzer isolates each detection module, but nothing guarded the
        # ~1,500 lines of rendering above: one odd flag detail or a preview
        # failure showed the user a raw Python traceback. Log the full trace
        # server-side and show the same card as the too-large / too-many-pages
        # paths. Streamlit's own control-flow exceptions (st.stop, rerun)
        # derive from BaseException, so this clause leaves them alone.
        logging.getLogger(__name__).exception("results rendering failed")
        _page_ctx.close()
        _page_slot.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:30vh; text-align:center; gap:1.5rem;">
            <div style="font-size:5rem; font-weight:bold; color:white;">Oops...</div>
            <div style="background:rgba(192,57,43,0.15); border:1px solid rgba(192,57,43,0.5); border-radius:8px; padding:1.2rem 2rem; max-width:420px;">
                <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e05252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <div style="font-size:1.1rem; font-weight:bold; color:#e05252;">{t("render_error_title")}</div>
                </div>
                <div style="color:#aaa; font-size:0.8rem; text-align:center;">{t("render_error_message")}</div>
            </div>
            <a href="." style="margin-top:0.5rem; padding:0.6rem 1.5rem; background:#4a90e2; color:white; border-radius:8px; text-decoration:none; font-weight:500;">
                {t("back_to_home")}
            </a>
        </div>
        """, unsafe_allow_html=True)
    finally:
        _page_ctx.close()
        # The temp file lives for this script run only (see where it is
        # written); the next rerun re-materialises it from the upload bytes.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Footer
st.markdown(f"""
<style>
/* The centred contact link is absolutely flanked by the Guides link and the
   language toggle, so on a narrow phone the three collide. Below 420px drop
   the "Questions or feedback?" prefix and keep just the address, which is the
   part that is actually actionable. */
@media (max-width: 420px) {{
    .tf-footer-label {{ display: none; }}
}}
</style>
<div class="tf-footer" style="position:fixed; bottom:0; left:0; right:0; display:flex; align-items:center; justify-content:center;
    padding:0.3rem 1rem; background:#0e1117; border-top:1px solid #1a1a2e; z-index:999;">
    <a href="{'/guides/' if _current_lang == 'fr' else '/en/guides/'}" style="position:absolute; left:1rem;
        color:#888; text-decoration:none; font-size:0.78rem; padding:2px 4px;">{t("footer_guides")}</a>
    <a href="mailto:contact@trustyfile.com" style="color:#888; font-size:0.78rem; text-decoration:none;"><span class="tf-footer-label">{t("footer_contact")} </span><span style="text-decoration:underline;">contact@trustyfile.com</span></a>
    <a href="?lang={'en' if _current_lang == 'fr' else 'fr'}" style="position:absolute; right:1rem;
        color:#888; text-decoration:none; font-size:0.78rem; padding:2px 4px;">{'EN' if _current_lang == 'fr' else 'FR'}</a>
</div>
""", unsafe_allow_html=True)

mark_page_ready()
