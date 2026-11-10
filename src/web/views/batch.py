"""Batch page: one table row per uploaded document."""

from __future__ import annotations

from src.batch import BatchItem
from src.i18n import translator
from src.web.views.common import esc

# Relative widths of the table columns: file, verdict, score, finding, button.
COLUMN_WIDTHS = [3, 1.6, 0.8, 4, 1]
# Translation keys of the column headings (the button column has none).
COLUMN_KEYS = ("batch_col_file", "batch_col_verdict", "batch_col_score", "batch_col_finding")


def title_html(count: int, lang: str | None = None) -> str:
    """Page title ("3 documents analysed") and the one-line introduction."""
    t = translator(lang)
    return (
        f'<div style="color:white; font-size:1.3rem; font-weight:bold;">'
        f'{esc(t("batch_title").format(n=count))}</div>'
        f'<div style="color:#888; font-size:0.8rem; margin-bottom:0.6rem;">{esc(t("batch_intro"))}</div>'
    )


def too_many_html(max_files: int, lang: str | None = None) -> str:
    """Warning shown when more files were dropped than the batch accepts."""
    t = translator(lang)
    return (
        f'<div style="color:#ffb627; font-size:0.8rem; margin-bottom:0.4rem;">'
        f'{esc(t("batch_too_many").format(max=max_files))}</div>'
    )


def column_heading_html(key: str, lang: str | None = None) -> str:
    """One column heading of the table."""
    t = translator(lang)
    return (
        f'<div style="color:#888; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;">'
        f'{esc(t(key))}</div>'
    )


def pill_html(color: str, label: str) -> str:
    """Rounded coloured label, e.g. the verdict of a row."""
    return (
        f'<span style="background:{color}; color:white; font-size:0.68rem; font-weight:bold; '
        f'padding:2px 10px; border-radius:10px; white-space:nowrap;">{esc(label)}</span>'
    )


def file_name_html(name: str) -> str:
    """File name cell, cut with an ellipsis when too long (full name on hover)."""
    return (
        f'<div style="color:white; font-size:0.85rem; overflow:hidden; text-overflow:ellipsis; '
        f'white-space:nowrap;" title="{esc(name)}">{esc(name)}</div>'
    )


def row_cells_html(item: BatchItem, lang: str | None = None) -> tuple[str, str, str]:
    """
    The verdict, score and finding cells of one row.

    Args:
        item: The document's batch record, analysed or refused.
        lang: "fr" or "en".

    Returns:
        (verdict pill, score, finding sentence), three HTML strings. A refused
        document gets a grey "refused" pill, a dash and the reason.
    """
    t = translator(lang)
    if not item.ok:
        return (
            pill_html("#6c757d", t("batch_refused")),
            '<span style="color:#6c757d;">—</span>',
            f'<div style="color:#888; font-size:0.8rem;">{esc(t(f"batch_problem_{item.problem}"))}</div>',
        )
    result = item.result
    # Same thresholds and colours as the gauge on the results page.
    if result.risk_level == "UNVERIFIABLE":
        color, score_text = "#6c757d", t("score_not_available")
    elif result.trust_score >= 80:
        color, score_text = "#28a745", str(result.trust_score)
    elif result.trust_score >= 50:
        color, score_text = "#ffb627", str(result.trust_score)
    else:
        color, score_text = "#ef233c", str(result.trust_score)
    return (
        pill_html(color, t(f"risk_{result.risk_level}")),
        f'<span style="color:{color}; font-weight:bold; font-size:1rem;">{score_text}</span>',
        f'<div style="color:#cdd6f4; font-size:0.8rem;">{esc(item.summary.verdict)}</div>',
    )
