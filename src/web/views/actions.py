"""Feedback question and report actions at the end of the results column."""

from __future__ import annotations

from src.i18n import translator


def feedback_question_html(lang: str) -> str:
    """"Was this result useful?", next to the thumbs."""
    t = translator(lang)
    return f'<span style="color:#c9d1d9; font-size:0.9rem;">{t("feedback_question")}</span>'


def feedback_thanks_html(lang: str) -> str:
    """Green "thank you" line shown once a vote is recorded."""
    t = translator(lang)
    return (f'<p style="text-align:center; color:#28a745; font-size:0.85rem; margin:0;">'
            f'{t("feedback_thanks")}</p>')


def feedback_privacy_html(lang: str) -> str:
    """Small grey line: what a vote stores (never the document)."""
    t = translator(lang)
    return (f'<p style="text-align:center; color:#666; font-size:0.7rem; margin:0.2rem 0 0 0;">'
            f'{t("feedback_privacy")}</p>')


def new_analysis_link_html(lang: str) -> str:
    """Blue "analyse another document" link back to the upload page."""
    t = translator(lang)
    return f"""
                <a id="tf-new-analysis" href="/?lang={lang}" target="_parent" style="display:inline-block; background:#2563eb; color:white; text-decoration:none; border:1px solid #2563eb; border-radius:6px; padding:7px 14px; font-size:0.85rem; font-weight:600; line-height:1.2;">
                    {t("analyze_another")}
                </a>
                """


def print_button_html(lang: str) -> str:
    """
    "Print" button and its script.

    The script opens every <details> before printing (CSS cannot) and
    restores the reader's state afterwards. It stays in the same HTML block
    as the button: it binds by id when the element mounts.
    """
    t = translator(lang)
    return f"""
                <div id="tf-print-wrap" style="display:flex; justify-content:center;">
                    <button id="tf-print" type="button" style="background:transparent; color:#888; border:1px solid #2a2a3e; border-radius:6px; padding:7px 14px; font-size:0.85rem; line-height:1.2; cursor:pointer;">
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
                """
