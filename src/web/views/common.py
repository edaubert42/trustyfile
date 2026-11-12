"""Small pieces shared by several views: escaping, section titles, colours."""

from __future__ import annotations

import html

# Section heading used across the results page ("SUMMARY", "DETECTED ISSUES"
# ...). A template: call SECTION_TITLE.format(title).
SECTION_TITLE = ('<span style="color:white; font-size:0.95rem; font-weight:bold; '
                 'text-transform:uppercase; letter-spacing:0.5px;">{}</span>'
                 '<hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 8px 0;">')

# One colour per finding severity (pills and dots).
SEVERITY_COLORS = {
    "critical": "#ef233c",
    "high": "#ff6b35",
    "medium": "#ffb627",
    "low": "#555",
}


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


# structure.py writes a re-saved revision's tool as "Word → modified with
# iLovePDF" (viewmodel._editor_name parses that exact text). Only the page
# translates it, so the French history no longer shows English.
_MODIFIED_WITH = " → modified with "


def tool_label(tool: str, t) -> str:
    """A revision's tool, in the page language.

    Example (fr):
        "Word → modified with iLovePDF" -> "Word → enregistré avec iLovePDF"
    """
    return tool.replace(_MODIFIED_WITH, f" → {t('history_saved_with')} ")
