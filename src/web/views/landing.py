"""Landing page: headline, "how it works" steps, guide and tool links."""

from __future__ import annotations

import html

from src.i18n import translator


def upload_disclaimer_html(lang: str) -> str:
    """Small grey line under the drop zone ("your document is not kept")."""
    t = translator(lang)
    return f"""
    <p style="color:#666; font-size:0.75rem; text-align:center; margin-top:-0.5rem;">
        {t("disclaimer_upload")}
    </p>
    """


def headline_html(lang: str) -> str:
    """Headline and tagline above the drop zone."""
    t = translator(lang)
    return f"""
        <div id="landing-content">
            <h1 style="color:white; font-size:2.2rem; line-height:1.6; padding:0; text-align:center; margin:0 auto 0.8rem auto; max-width:600px; font-weight:bold; letter-spacing:0.5px;">
                {t("landing_headline")}
            </h1>
        </div>
        <div id="landing-tagline">
            <p style="color:white; font-size:1.2rem; text-align:center; margin:1.2rem auto 0.5rem auto; max-width:500px; line-height:1.5;">
                {t("landing_tagline")}
            </p>
        </div>
        """


def steps_html(guides: list, tools: list, lang: str) -> str:
    """
    The three "how it works" cards, then the guide and tool links.

    Args:
        guides: Guide pages of this language (objects with .slug and .title).
        tools: Checker tools of this language (same shape).
        lang: "fr" or "en".

    Returns:
        The block as an HTML string (its own <style> included).
    """
    t = translator(lang)
    _guides_prefix = "/guides/" if lang == "fr" else "/en/guides/"
    _guide_links = " · ".join(
        f'<a href="{_guides_prefix}{g.slug}" style="color:#888; text-decoration:none;">{html.escape(g.title)}</a>'
        for g in guides
    )
    _landing_guides_html = (
        '<p style="color:#666; font-size:0.74rem; text-align:center; margin:1.8rem auto 0 auto; '
        'line-height:1.9; max-width:640px;">'
        f'{t("footer_guides")}{" :" if lang == "fr" else ":"} {_guide_links}</p>'
    ) if _guide_links else ""
    # Checker tools (IBAN / SIRET / SIREN), same treatment: a visitor with a
    # single number to check should not have to upload a document to find them.
    _tools_prefix = "/outils/" if lang == "fr" else "/en/tools/"
    _tool_links = " · ".join(
        f'<a href="{_tools_prefix}{tool.slug}" style="color:#888; text-decoration:none;">{html.escape(tool.title)}</a>'
        for tool in tools
    )
    _landing_tools_html = (
        '<p style="color:#666; font-size:0.74rem; text-align:center; margin:0.4rem auto 0 auto; '
        'line-height:1.9; max-width:640px;">'
        f'{t("footer_tools")}{" :" if lang == "fr" else ":"} {_tool_links}</p>'
    ) if _tool_links else ""

    # "How it works" steps
    return f"""
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
    """
