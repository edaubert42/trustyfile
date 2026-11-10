"""
TrustyFile - Document Fraud Detection

A Streamlit web application for analyzing invoices and documents
to detect signs of manipulation or fraud.

Run with: streamlit run app.py
"""

import streamlit as st
import json
import logging
import tempfile
from contextlib import ExitStack
from datetime import datetime
import os
from pathlib import Path
from src.i18n import t, get_lang
from src.preview import (
    render_pdf_page, collect_evidence_boxes, collect_numbered_boxes,
    first_flagged_page,
)
from src.web.viewmodel import build_document_view
from src.web.views import (
    actions, details, errors, findings, landing, properties, verdict,
)
from src.web.views import batch as batch_views
from src.web.views.common import esc
# The landing page lists the verification guides and the checker tools; the
# same lists feed the crawler-facing copy of the page (one source).
from src.web.views.links import GUIDES as _GUIDES, TOOLS as _TOOLS
from src.web.virustotal import check_virustotal

def _virustotal_lookup(sha256: str) -> dict | None:
    """VirusTotal answer for a file hash, with the API key from Streamlit secrets."""
    try:
        api_key = st.secrets.get("VIRUSTOTAL_API_KEY", "")
    except Exception:
        api_key = ""
    return check_virustotal(sha256, api_key)



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

    /* JSON report download (roadmap #8, tier 2): a native st.download_button
       restyled to sit beside the HTML print button as a quiet, same-size pair. */
    .st-key-tf_report_row [data-testid="stDownloadButton"] button {{
        background: transparent; color: #888; border: 1px solid #2a2a3e;
        border-radius: 6px; padding: 7px 14px; font-size: 0.85rem; min-height: 0;
        line-height: 1.2;
    }}
    /* Report actions: same 440 px column as the findings. The main action
       ("analyse another document") takes a full row; the two quiet exports
       share the row below, half each. Left to wrap freely, the three buttons
       broke into 2 + 1 with the third one floating alone. */
    .st-key-tf_report_row {{
        max-width: 440px; margin: 1.4rem auto 0 auto; gap: 10px !important;
    }}
    .st-key-tf_report_row > * {{ flex: 1 1 0 !important; min-width: 0 !important; width: auto !important; }}
    .st-key-tf_report_row > *:first-child {{ flex: 1 1 100% !important; }}
    .st-key-tf_report_row #tf-new-analysis {{ display: block !important; text-align: center; }}
    .st-key-tf_report_row #tf-print-wrap, .st-key-tf_report_row #tf-print,
    .st-key-tf_report_row [data-testid="stDownloadButton"],
    .st-key-tf_report_row [data-testid="stDownloadButton"] button {{ width: 100% !important; }}
    .st-key-tf_report_row [data-testid="stDownloadButton"] button:hover {{
        color: #ccc; border-color: #4a90e2;
    }}
    /* Feedback comment form: same centred 440 px column as the findings above,
       instead of the full width of the results column. */
    /* No gap between the question and its privacy line: they read as one. */
    .st-key-tf_feedback {{ gap: 0 !important; }}
    .st-key-tf_feedback [data-testid="stForm"] {{
        max-width: 440px; margin: 0 auto;
    }}
    @media print {{
        @page {{ margin: 12mm; }}
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] {{
            background: #fff !important;
            color: #000 !important;
        }}
        .tf-print-only {{ display: block; }}
        .custom-header, .tf-footer, #tf-print-wrap, #tf-new-analysis, .st-key-tf_feedback, [data-testid="stDownloadButton"],
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

# File uploader — always rendered at top level (outside landing columns).
# Several files at once are accepted (roadmap #14): one file goes straight to
# the results page as before; two or more open the batch page further down, a
# table with one row per document, each row opening the full results page.
MAX_BATCH_FILES = 10
uploaded_files = st.file_uploader(
    "Choose file",
    # Photos too: a single JPEG/PNG opens the 2D-Doc photo scan (a landlord
    # photographing a paper tax notice). On a phone this button then offers
    # the camera directly, at full resolution.
    type=["pdf", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
    accept_multiple_files=True,
) or []
_batch_truncated = len(uploaded_files) > MAX_BATCH_FILES
uploaded_files = uploaded_files[:MAX_BATCH_FILES]
# `batch_files` is non-empty only in batch mode. `uploaded_file` stays what
# the rest of this script has always rendered — ONE document: the lone
# upload, the row the reader opened from the batch table, or None.
batch_files = uploaded_files if len(uploaded_files) > 1 else []
if batch_files:
    _open_id = st.session_state.get("batch_open")
    uploaded_file = next((f for f in batch_files if f.file_id == _open_id), None)
else:
    uploaded_file = uploaded_files[0] if uploaded_files else None
    st.session_state.pop("batch_open", None)


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


if uploaded_files:
    st.session_state.pop("demo_active", None)
elif st.session_state.get("demo_active") and _DEMO_PDF.exists():
    uploaded_file = _DemoUpload()

# The landing page shows when there is nothing to render: no single document
# and no batch.
_landing = uploaded_file is None and not batch_files
if _landing:
    st.markdown(landing.upload_disclaimer_html(_current_lang), unsafe_allow_html=True)
    if _DEMO_PDF.exists():
        _demo_l, _demo_c, _demo_r = st.columns([2, 1.6, 2])
        if _demo_c.button(t("demo_button"), key="demo_button", use_container_width=True):
            st.session_state["demo_active"] = True
            st.rerun()

# Landing page content — only rendered when no file is uploaded
if _landing:
    col_left, col_upload, col_right = st.columns([1, 2, 1])
    with col_upload:
        st.markdown(landing.headline_html(_current_lang), unsafe_allow_html=True)

    # "How it works" cards, then the guide and tool links. A human gets the
    # same link rows as the crawler-facing copy of this page (see
    # scripts/build_guides.py render_landing), so the two stay equivalent.
    st.markdown(
        landing.steps_html(_GUIDES.get(_current_lang, []), _TOOLS.get(_current_lang, []), _current_lang),
        unsafe_allow_html=True,
    )

# Hide landing content and minimize uploader when a file is uploaded
if uploaded_file is not None or batch_files:
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
    if not (batch_files and uploaded_file is None):
        # Single-result page: every top-level child but the results container
        # is invisible (scripts, styles, the hidden uploader), yet each still
        # costs one flex gap — ~50 px of blank band under the logo. The page
        # content lives in one container, which keeps its own inner spacing.
        st.markdown("""<style>
            .block-container > [data-testid="stVerticalBlock"] { gap: 0 !important; }
        </style>""", unsafe_allow_html=True)

# =============================================================================
# BATCH PAGE (roadmap #14) — several files, one row each
# =============================================================================
if batch_files and uploaded_file is None:
    import hashlib
    from src.analyzer import TrustyFileAnalyzer
    from src.batch import analyse_document
    from src.modules.twod_doc import scan_pdf_for_2d_doc, PYLIBDMTX_AVAILABLE
    from src.report import build_batch_report, batch_report_filename, report_json

    # One analysis per (content, name, language), cached in session_state so
    # a rerun — opening a row and coming back — costs nothing. The language is
    # part of the key because the summary is generated in one language; a
    # toggle re-analyses rather than showing yesterday's French under an
    # English heading. Entries for files removed from the uploader are dropped.
    _batch_cache = st.session_state.setdefault("batch_cache", {})
    _batch_keys = []                      # (upload, content md5, cache key)
    for _f in batch_files:
        _md5 = hashlib.md5(_f.getvalue()).hexdigest()
        _batch_keys.append((_f, _md5, f"{_md5}:{_f.name}:{get_lang()}"))
    _todo = [(_f, _key) for _f, _md5, _key in _batch_keys if _key not in _batch_cache]
    if _todo:
        _bar = st.progress(0.0)
        for _i, (_f, _key) in enumerate(_todo):
            _bar.progress(_i / len(_todo), text=t("batch_analysing").format(i=_i + 1, n=len(_todo), name=_f.name))
            _batch_cache[_key] = analyse_document(
                _f.name,
                _f.getvalue(),
                lang=get_lang(),
                analyzer_factory=lambda: TrustyFileAnalyzer(
                    enable_external=enable_external,
                    enable_qr_scan=enable_qr,
                    enable_iban_check=enable_iban_check,
                    iban_api_key=iban_api_key,
                ),
                scan_2d_doc=scan_pdf_for_2d_doc if PYLIBDMTX_AVAILABLE else None,
            )
        _bar.empty()
    _current_keys = {_key for _, _, _key in _batch_keys}
    for _stale in [k for k in _batch_cache if k not in _current_keys]:
        del _batch_cache[_stale]
    _batch_rows = [(_f, _md5, _batch_cache[_key]) for _f, _md5, _key in _batch_keys]

    st.markdown('<hr style="margin:0 0 0.3rem 0;">', unsafe_allow_html=True)
    st.markdown(batch_views.title_html(len(_batch_rows), _current_lang), unsafe_allow_html=True)
    if _batch_truncated:
        st.markdown(batch_views.too_many_html(MAX_BATCH_FILES, _current_lang), unsafe_allow_html=True)

    # Table as columns rather than HTML: each row needs a real Streamlit
    # button, and a widget cannot live inside an st.markdown table.
    _head = st.columns(batch_views.COLUMN_WIDTHS, vertical_alignment="center")
    for _col, _key in zip(_head, batch_views.COLUMN_KEYS):
        _col.markdown(batch_views.column_heading_html(_key, _current_lang), unsafe_allow_html=True)

    for _f, _md5, _item in _batch_rows:
        _c_name, _c_band, _c_score, _c_finding, _c_open = st.columns(
            batch_views.COLUMN_WIDTHS, vertical_alignment="center")
        _c_name.markdown(batch_views.file_name_html(_f.name), unsafe_allow_html=True)
        _band_html, _score_html, _finding_html = batch_views.row_cells_html(_item, _current_lang)
        _c_band.markdown(_band_html, unsafe_allow_html=True)
        _c_score.markdown(_score_html, unsafe_allow_html=True)
        _c_finding.markdown(_finding_html, unsafe_allow_html=True)
        if _item.ok and _c_open.button(t("batch_open"), key=f"batch_open_{_f.file_id}", use_container_width=True):
            # Seed the single-document cache with this row's analysis: the
            # results page keys its cache on the content md5, so it renders
            # straight away instead of analysing the file a second time.
            st.session_state["batch_open"] = _f.file_id
            st.session_state["last_file_hash"] = _md5
            st.session_state["cached_result"] = _item.result
            st.session_state["cached_summary"] = _item.summary
            st.session_state["cached_twod_doc"] = _item.twod_doc
            # Without this the results page would show the signature
            # badges of whichever document was analysed before.
            st.session_state["cached_twod_doc_signatures"] = _item.twod_doc_signatures
            st.session_state["cached_analysed_at"] = _item.analysed_at
            st.session_state.pop("page_selector", None)
            st.rerun()

    # The whole batch as one JSON file: every row, analysed or not.
    _batch_when = datetime.now()
    with st.container(key="tf_report_row"):
        _b_space, _b_dl = st.columns([3, 1.4], vertical_alignment="center")
        _b_dl.download_button(
            t("batch_download"),
            data=report_json(build_batch_report(
                [_item for _, _, _item in _batch_rows],
                lang=get_lang(),
                analysed_at=_batch_when.strftime("%Y-%m-%d %H:%M"),
            )),
            file_name=batch_report_filename(_batch_when),
            mime="application/json",
            key="batch_download",
            on_click="ignore",
            use_container_width=True,
        )

if uploaded_file is not None:
    # Lazy imports: only load heavy analysis modules when a file is uploaded
    from src.analyzer import TrustyFileAnalyzer
    from src.summary import generate_rich_summary
    from src.report import build_report, report_json, report_filename
    from src.modules.twod_doc import scan_pdf_for_2d_doc, PYLIBDMTX_AVAILABLE
    from src.pipeline import scan_2d_doc
    from src.twod_doc_view import twod_pdf_panel_html
    import hashlib

    # Compute a hash of the file bytes to detect if it's a new file
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Check file size — reject documents over 10MB
    _file_size_mb = len(file_bytes) / (1024 * 1024)
    if _file_size_mb > 10:
        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(errors.error_card_html(
            t("too_large_title"),
            t("too_large_message").format(size=f"{_file_size_mb:.1f}"),
            _current_lang,
        ), unsafe_allow_html=True)
        mark_page_ready()
        st.stop()

    # A photo (JPEG/PNG, recognised by its bytes, not its name) never enters
    # the PDF pipeline: it only gets the 2D-Doc photo scan, and never a trust
    # score. The 10 MB check above applies to it too.
    from src.intake import looks_like_photo
    if looks_like_photo(file_bytes):
        from src.modules.twod_doc_photo import scan_photo, photo_verdict
        from src.twod_doc_view import photo_result_html

        # Cache by content: Streamlit reruns the script on every click or
        # language switch, and the scan must not run again each time.
        _photo_key = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("photo_scan_key") != _photo_key:
            st.session_state["photo_scan_result"] = scan_photo(file_bytes)
            st.session_state["photo_scan_key"] = _photo_key
        _photo_result = st.session_state["photo_scan_result"]
        _photo_level, _photo_msg = photo_verdict(_photo_result)
        # No personal data in the log: verdict key, attempt name, timing.
        logging.getLogger(__name__).info("Photo scan: %s (%s) in %.2fs", _photo_msg,
                    _photo_result.attempt or "-", _photo_result.elapsed_s)

        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(photo_result_html(_photo_result, _photo_level, _photo_msg),
                    unsafe_allow_html=True)
        mark_page_ready()
        st.stop()

    # Check page count before analysis — reject documents over 20 pages
    import fitz as _fitz
    try:
        with _fitz.open(stream=file_bytes, filetype="pdf") as _doc:
            _page_count = len(_doc)
    except Exception:
        # Not a PDF at all (a text file renamed .pdf, a truncated download):
        # PyMuPDF refuses to open it. Say so with the usual card instead of
        # letting the exception reach the reader as a raw error box.
        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(errors.error_card_html(
            t("not_a_pdf_title"), t("batch_problem_not_a_pdf"), _current_lang,
        ), unsafe_allow_html=True)
        mark_page_ready()
        st.stop()
    if _page_count > 20:
        st.markdown("<hr style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        st.markdown(errors.error_card_html(
            t("too_many_pages_title"),
            t("too_many_pages_message").format(count=_page_count),
            _current_lang,
        ), unsafe_allow_html=True)
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

            # Scan for 2D-DOC barcodes (French government signed barcodes)
            # and check each code's signature for the panel badge. Passports
            # are skipped (no 2D-Doc on a TD3) — see src/pipeline.py, shared
            # with the batch page. Done once, then cached.
            twod_doc_results, twod_doc_signatures = scan_2d_doc(
                tmp_path, result,
                scan_pdf_for_2d_doc if PYLIBDMTX_AVAILABLE else None,
            )

            # Cache everything in session_state so page navigation doesn't re-trigger analysis
            st.session_state["last_file_hash"] = file_hash
            st.session_state["cached_result"] = result
            # Wall-clock time of THIS analysis, for the printed report. Kept in
            # session_state so a later rerun (page change, language toggle)
            # does not restamp it.
            st.session_state["cached_analysed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["cached_summary"] = summary
            st.session_state["cached_twod_doc"] = twod_doc_results
            st.session_state["cached_twod_doc_signatures"] = twod_doc_signatures
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
        twod_doc_signatures = st.session_state.get("cached_twod_doc_signatures", [])

    # Everything below renders inside one placeholder. If anything in the
    # ~1,500 lines of results rendering raises, the except clause empties the
    # placeholder — taking the half-drawn columns with it — and puts the error
    # card in its place. ExitStack enters the container exactly as a `with`
    # would, without indenting the whole block one level deeper.
    _page_slot = st.empty()
    _page_ctx = ExitStack()
    try:
        _page_ctx.enter_context(_page_slot.container())

        # Opened from the batch table: one click back to the lot.
        if batch_files and st.button(t("batch_back"), key="batch_back", type="tertiary"):
            st.session_state.pop("batch_open", None)
            st.rerun()

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

        # Everything the page shows besides the score (page count, findings
        # with their region on the page, metadata, revision history,
        # VirusTotal answer) is computed ONCE per document and kept for the
        # reruns that follow (page change, language switch, thumbs up...).
        # See src/web/viewmodel.py.
        _view_key = (file_hash, uploaded_file.name)
        if st.session_state.get("cached_view_key") != _view_key:
            st.session_state["cached_view"] = build_document_view(
                tmp_path, result, uploaded_file.name,
                virustotal_lookup=_virustotal_lookup,
            )
            st.session_state["cached_view_key"] = _view_key
        view = st.session_state["cached_view"]
        page_count = view.page_count
        all_flags = view.all_flags
        # The reasons shown next to the verdict: the preview draws their
        # numbers, the results list prints them.
        _reasons = view.reasons

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
                numbered = collect_numbered_boxes(_reasons, page_index)
                # Located text findings are not in `result`: highlight them
                # too, so a badge never floats next to plain text.
                boxes += [bbox for bbox, _n in numbered if bbox not in boxes]
                img_bytes = render_pdf_page(tmp_path, page_index, zoom=2.0,
                                            boxes=boxes, numbered=numbered)
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.error(t("could_not_render").format(error=e))

            # File properties block
            st.markdown(properties.file_properties_html(view, _current_lang), unsafe_allow_html=True)

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
            st.html(verdict.print_header_html(
                uploaded_file.name, result.file_hash, _analysed_at, _current_lang))

            # Score gauge, verdict and advice
            st.markdown(verdict.gauge_html(result, summary, _current_lang), unsafe_allow_html=True)

            # Findings, modification timeline, document type. Built here but
            # rendered AFTER the feedback block below: the question sits
            # right under the verdict, above "Summary", where nobody can
            # miss it.
            _summary_html = findings.summary_html(result, view, _current_lang)

            # --- Feedback: was this result useful? (thumbs up / down) -------
            # Stored by src/feedback.py: always a log line, plus a SQLite row
            # when TRUSTYFILE_FEEDBACK_DB is set (a Railway Volume). Never the
            # document. Widget keys carry the file hash so a new upload starts
            # with a blank vote instead of inheriting the previous file's.
            from src.feedback import FeedbackRecord, add_comment, save_feedback
            _fb_suffix = result.file_hash[:16]
            _fb_vote_key = f"fb_vote_{_fb_suffix}"   # st.feedback value: 1 = up, 0 = down
            _fb_id_key = f"fb_id_{_fb_suffix}"       # database row of this vote
            _fb_sent_key = f"fb_sent_{_fb_suffix}"   # comment already sent

            def _record_vote() -> None:
                """on_change of the thumbs: store the vote, or update it if the
                reader changes their mind (one opinion = one row)."""
                vote = st.session_state.get(_fb_vote_key)
                if vote is None:  # same thumb clicked again: selection cleared, keep the stored vote
                    return
                st.session_state[_fb_id_key] = save_feedback(
                    FeedbackRecord(
                        rating="up" if vote == 1 else "down",
                        band=result.band,
                        score=None if result.band == "UNVERIFIABLE" else result.trust_score,
                        document_type=result.document_type,
                        flag_codes=[f.code for f in all_flags],
                        lang=get_lang(),
                    ),
                    feedback_id=st.session_state.get(_fb_id_key),
                )

            with st.container(key="tf_feedback"):
                with st.container(horizontal=True, horizontal_alignment="center",
                                  vertical_alignment="center", gap="small"):
                    st.html(actions.feedback_question_html(_current_lang), width="content")
                    st.feedback("thumbs", key=_fb_vote_key, on_change=_record_vote)
                _vote = st.session_state.get(_fb_vote_key)
                if _vote == 0 and not st.session_state.get(_fb_sent_key):
                    with st.form(key=f"fb_form_{_fb_suffix}", border=False):
                        _comment = st.text_area(
                            t("feedback_comment_label"),
                            placeholder=t("feedback_comment_placeholder"),
                            max_chars=1000,
                            height=80,
                        )
                        if st.form_submit_button(t("feedback_send")):
                            add_comment(st.session_state.get(_fb_id_key), _comment)
                            st.session_state[_fb_sent_key] = True
                            st.rerun()
                if _vote is not None and (_vote == 1 or st.session_state.get(_fb_sent_key)):
                    st.html(actions.feedback_thanks_html(_current_lang))
                st.html(actions.feedback_privacy_html(_current_lang))

            st.markdown(_summary_html, unsafe_allow_html=True)

            # Report actions at the END of the column: exporting is the last
            # thing a reader does, so it no longer sits above the verdict. One
            # horizontal container that wraps on a phone, instead of three
            # columns whose empty spacer stacked into a blank row on mobile.
            # Print button + JSON download (roadmap #8, tier 2) + a link back
            # to the upload page (the dropzone is hidden on the results page). The download is the
            # same serializer the API endpoint returns (src/report.py), built
            # from the cached result + summary so a rerun costs one json.dumps.
            # `on_click="ignore"` keeps a click from rerunning the whole page.
            # The <script> stays in the same st.html as its button: it binds by
            # id when the element mounts, so they must mount together.
            with st.container(key="tf_report_row", horizontal=True,
                              horizontal_alignment="center", vertical_alignment="center", gap="small"):
                # width="content": each action is as wide as its label, so the
                # container can centre the three as one group (and wrap them
                # on a phone) instead of stretching each into a third.
                st.html(actions.new_analysis_link_html(_current_lang), width="content")
                st.html(actions.print_button_html(_current_lang),
                        unsafe_allow_javascript=True, width="content")
                _report_text = report_json(build_report(
                    result,
                    filename=uploaded_file.name,
                    analysed_at=_analysed_at,
                    lang=get_lang(),
                    summary=summary,
                ))
                st.download_button(
                    t("download_report"),
                    data=_report_text,
                    file_name=report_filename(result),
                    mime="application/json",
                    key="download_report",
                    on_click="ignore",
                    width="content",
                )


        # One fold for everything below the verdict (progressive disclosure):
        # the score caveat, the evidence bars, every finding, then one panel
        # per kind of evidence. Built by src/web/views/details.py.
        _more_details_html = (
            details.history_panel_html(view, _current_lang)
            + details.modules_panel_html(result, _current_lang)
            + details.entities_panel_html(result, view.full_text, _current_lang)
            + details.bank_panel_html(result, view.full_text, _current_lang)
            # Badge and colour follow the signature status
            # (src/twod_doc_view.py:pdf_signature_badge).
            + twod_pdf_panel_html(twod_doc_results, twod_doc_signatures, _current_lang)
            + details.mrz_panel_html(result, _current_lang)
        )
        st.markdown(
            details.all_details_html(
                findings.evidence_bars_html(result, _current_lang),
                details.issues_html(all_flags, result.document_type, _current_lang),
                _more_details_html,
                _current_lang,
            ),
            unsafe_allow_html=True,
        )

    except Exception:
        # The analyzer isolates each detection module, but nothing guarded the
        # ~1,500 lines of rendering above: one odd flag detail or a preview
        # failure showed the user a raw Python traceback. Log the full trace
        # server-side and show the same card as the too-large / too-many-pages
        # paths. Streamlit's own control-flow exceptions (st.stop, rerun)
        # derive from BaseException, so this clause leaves them alone.
        logging.getLogger(__name__).exception("results rendering failed")
        _page_ctx.close()
        _page_slot.markdown(errors.error_card_html(
            t("render_error_title"), t("render_error_message"), _current_lang,
        ), unsafe_allow_html=True)
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
