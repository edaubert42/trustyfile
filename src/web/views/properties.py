""""File properties" block under the preview: name, dates, producer, hash."""

from __future__ import annotations

from src.dates import days_between, seconds_between
from src.i18n import translator
from src.web.viewmodel import DocumentView
from src.web.views.common import esc


def file_properties_html(view: DocumentView, lang: str | None = None) -> str:
    """
    The file properties table, with red pills on suspicious date gaps.

    Args:
        view: The per-document data computed once after the analysis.
        lang: "fr" or "en".

    Returns:
        The block as an HTML string.
    """
    t = translator(lang)
    meta = view.metadata
    signature_info = view.signature_info
    stated_date = view.stated_date
    editor_name = view.editor_name
    file_name = view.file_name
    file_hash = view.file_hash

    def format_date(dt):
        return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"

    def red_pill(text, tooltip=""):
        title_attr = f' title="{tooltip}"' if tooltip else ""
        return (f'<span{title_attr} style="background:#ef233c; color:white; padding:2px 8px; '
                f'border-radius:10px; font-size:0.7rem; font-weight:bold; '
                f'margin-left:8px; white-space:nowrap; cursor:default;">{text}</span>')

    label_style = "color:#888; font-size:0.8rem; width:120px; flex-shrink:0;"
    value_style = "color:#eee; font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
    row_style = "display:flex; align-items:center; padding:6px 0; border-bottom:1px solid #2a2a2a;"

    producer = meta.producer or meta.creator or '—'
    sig = signature_info or "—"  # same placeholder as the empty dates above

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
                <span style="{value_style}" title="{esc(file_name)}">{esc(file_name)}</span>
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
                <span style="{value_style} font-size:0.75rem;" title="{esc(file_hash)}">{esc(file_hash)}</span>
            </div>
            """

    # VirusTotal: the file hash looked up in VT's database (None = no answer).
    vt_result = view.virustotal
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

    return f"""
            <div style="margin-top:0.5rem;">
                <span style="color:white; font-size:0.95rem; font-weight:bold; text-transform:uppercase;
                    letter-spacing:0.5px;">{t("file_properties")}</span>
                <hr style="border:none; border-top:1px solid #2a2a2a; margin:6px 0 10px 0;">
                {file_rows}
            </div>
            """
