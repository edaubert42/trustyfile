"""Top of the results column: print header and the score gauge."""

from __future__ import annotations

from src.i18n import translator
from src.models import AnalysisResult, AnalysisSummary
from src.summary import next_step_key
from src.web.views.common import esc


def print_header_html(file_name: str, file_hash: str, analysed_at: str,
                      lang: str) -> str:
    """
    Report header that only shows on paper ("Save as PDF").

    On screen the same facts sit in the file properties block; a printed page
    has no URL bar and no file name bar, so they come first there.
    """
    t = translator(lang)
    return f"""
            <div class="tf-print-only tf-print-keep" style="border-bottom:2px solid #000; padding-bottom:6px; margin-bottom:14px;">
                <div style="font-size:1.1rem; font-weight:bold;">{t("print_report_title")}</div>
                <div style="font-size:0.8rem;">{t("print_file")} {esc(file_name)}</div>
                <div style="font-size:0.8rem;">SHA-256 <code>{esc(file_hash)}</code></div>
                <div style="font-size:0.8rem;">{t("print_date")} {analysed_at} &middot; {t("print_generated_by")} https://www.trustyfile.com</div>
            </div>
            """


def gauge_html(result: AnalysisResult, summary: AnalysisSummary,
               lang: str) -> str:
    """
    The circular score gauge with the verdict and the advice under it.

    Args:
        result: The analysis (score, band, document type).
        summary: Its generated summary (the verdict sentence).
        lang: "fr" or "en".

    Returns:
        The block as an HTML string (an inline SVG and two or three lines).
    """
    t = translator(lang)
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

    # Two lines under the gauge, no more: the verdict, then what to do.
    # The full score caveat moved to the "see all details" block. One
    # short reminder stays, and only on a GOOD result: that is where
    # a reader could over-trust the verdict ("green = authentic"). On
    # a bad result the next step already says to check by hand.
    _caveat_html = ""
    if risk_level != "UNVERIFIABLE" and score >= 80:
        _caveat_html = (
            '<p style="color:#888; font-size:0.75rem; line-height:1.4; max-width:420px; '
            f'margin:0 auto 0.4rem auto;">{esc(t("score_caveat_short"))}</p>'
        )

    # Headline under the gauge. On a SUSPICIOUS result it is the advice
    # itself ("Do not rely on this document..."): the verdict sentence
    # names ONE theme (e.g. "identity information is fraudulent") and
    # can disagree with the first reason listed just below (e.g. an
    # overwritten amount). The reasons already say what is wrong, so
    # the headline only says what to do. Other bands keep
    # verdict + advice. (The JSON report still carries the verdict.)
    #
    # On a GENUINE result the caveat replaces the advice: "confirm with the
    # issuer if it matters" only repeated what the caveat already implies.
    _advice = t(next_step_key(result.band, result.document_type))
    if result.band == "SUSPICIOUS":
        _headline = _advice
        _advice_html = ""
    elif result.band == "GENUINE":
        _headline = summary.verdict
        _advice_html = ""
    else:
        _headline = summary.verdict
        _advice_html = (
            '<p style="color:#c9d1d9; font-size:0.88rem; line-height:1.5; max-width:420px; '
            f'margin:0 auto 0.5rem auto;">{_advice}</p>'
        )

    return f"""
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
                <p style="color:white; font-size:1.15rem; font-weight:bold; line-height:1.35; max-width:420px; margin:0.1rem auto 0.6rem auto;">
                    {_headline}
                </p>
                {_advice_html}
                {_caveat_html}
            </div>
            """
