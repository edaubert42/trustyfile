"""The "Oops..." card shown when a document cannot be displayed."""

from __future__ import annotations

from src.i18n import translator


def error_card_html(title: str, message: str, lang: str) -> str:
    """
    Full-width error card with a link back to the home page.

    Used for a file that is too large, has too many pages, or whose results
    page failed to render.

    Args:
        title: Short red heading, already translated.
        message: One sentence of explanation, already translated.
        lang: "fr" or "en", for the "back to home" link.

    Returns:
        The card as an HTML string.
    """
    t = translator(lang)
    return f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:30vh; text-align:center; gap:1.5rem;">
            <div style="font-size:5rem; font-weight:bold; color:white;">Oops...</div>
            <div style="background:rgba(192,57,43,0.15); border:1px solid rgba(192,57,43,0.5); border-radius:8px; padding:1.2rem 2rem; max-width:420px;">
                <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e05252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <div style="font-size:1.1rem; font-weight:bold; color:#e05252;">{title}</div>
                </div>
                <div style="color:#aaa; font-size:0.8rem; text-align:center;">{message}</div>
            </div>
            <a href="." style="margin-top:0.5rem; padding:0.6rem 1.5rem; background:#4a90e2; color:white; border-radius:8px; text-decoration:none; font-weight:500;">
                {t("back_to_home")}
            </a>
        </div>
        """
