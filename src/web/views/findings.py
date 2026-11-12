"""Summary block of the results column: findings, history timeline, bars."""

from __future__ import annotations

import html
import math

from src.evidence import (
    BAND_GENUINE_MIN, BAND_REVIEW_MIN, GENUINE_CODES, PRIOR_LOG_ODDS, module_contributions,
)
from src.i18n import translator
from src.models import AnalysisResult
from src.summary import get_flag_sentence
from src.web.viewmodel import DocumentView
from src.web.views.common import SECTION_TITLE, SEVERITY_COLORS, esc


def _nats_to_reach(score: int) -> float:
    """Tamper evidence (nats) that alone brings a clean document down to `score`.

    score = 100 * (1 - sigmoid(PRIOR_LOG_ODDS + nats)), solved for nats.

    Examples:
        >>> round(_nats_to_reach(50), 2)   # 4.0 with PRIOR_LOG_ODDS = -4.0
        4.0
        >>> round(_nats_to_reach(80), 2)
        2.61
    """
    tamper_share = 1 - score / 100
    return math.log(tamper_share / (1 - tamper_share)) - PRIOR_LOG_ODDS


BAR_FULL_NATS = _nats_to_reach(BAND_REVIEW_MIN)    # full bar: down to 50 alone
BAR_RED_NATS = _nats_to_reach(BAND_GENUINE_MIN)    # red: out of GENUINE alone


def evidence_bars_html(result: AnalysisResult, lang: str) -> str:
    """
    Four horizontal bars: how much each family of checks weighed on the score.

    Args:
        result: The analysis (modules and evidence ledger).
        lang: "fr" or "en".

    Returns:
        The bars as an HTML string.
    """
    t = translator(lang)
    # Grouped score bars (4 categories instead of 7 modules)
    # Map module names to their results for easy lookup
    module_map = {m.module: m for m in result.modules}

    # Define the 4 groups: (label, list of module names, tooltip)
    groups = [
        (t("group_modifications"), ["structure", "overlay"], t("tip_modifications")),
        (t("group_consistency"), ["content", "arithmetic", "payslip"], t("tip_consistency")),
        (t("group_tampering"), ["images", "fonts", "visual", "metadata", "forensics"], t("tip_tampering")),
        (t("group_authenticity"), ["external", "identity", "twod_doc"], t("tip_authenticity")),
    ]

    # Evidence-ledger contributions (audit item 23): the bars show the
    # ACTUAL score pressure per group — the same group-capped sums the
    # aggregation used — not the old (deleted) weighted-average formula.
    contrib = module_contributions(result.evidence_ledger)

    def _group_net(names: list[str]) -> float:
        return sum(contrib.get(n, {}).get("tamper", 0.0)
                   - contrib.get(n, {}).get("genuine", 0.0)
                   for n in names)

    # Codes that forced the band, per module (proven modification, virtual
    # printer). Their weight in nats says nothing: the band was imposed.
    forced_band_by_module: dict[str, str] = {}
    for m in result.modules:
        for f in m.flags:
            band = result.forced_codes.get(f.code)
            if band and forced_band_by_module.get(m.module) != "SUSPICIOUS":
                forced_band_by_module[m.module] = band

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
        forced = {forced_band_by_module.get(n) for n in member_names}

        # No number next to the bar: the values are log-odds "nats",
        # an internal unit that means nothing to a reader ("+3.7").
        # Bar length and colour carry the comparison on their own.
        #
        # ABSOLUTE scale (2026-11-12): a full bar is the evidence that alone
        # brings the score down to 50 (BAR_FULL_NATS), red from the amount
        # that alone takes a document out of GENUINE. The old bars showed
        # each group's SHARE of the total, so the only weak hint of a
        # GENUINE 92 document filled its bar.
        if hard or "SUSPICIOUS" in forced:
            width, bar_color, value_label = 100, "#ef233c", ""
        elif net > 0 or "REVIEW" in forced:
            width = min(100, round(100 * max(net, 0.0) / BAR_FULL_NATS))
            if "REVIEW" in forced:
                width = max(width, round(100 * BAR_RED_NATS / BAR_FULL_NATS))
            bar_color = "#ef233c" if net >= BAR_RED_NATS else "#ffb627"
            value_label = ""
        elif net < 0:
            width, bar_color, value_label = 25, "#28a745", ""
        else:
            width, bar_color, value_label = 0, "#28a745", ""

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
    return bars_html


def summary_html(result: AnalysisResult, view: DocumentView, lang: str) -> str:
    """
    The "Summary" block: main findings, modification timeline, document type.

    Args:
        result: The analysis.
        view: The per-document data computed once after the analysis.
        lang: "fr" or "en".

    Returns:
        The block as an HTML string.
    """
    t = translator(lang)
    all_flags = view.all_flags
    _reasons = view.reasons
    _reason_numbers = view.reason_numbers
    history = view.history
    _dt = result.document_type

    _dt_label = t(f"doctype_{_dt}")

    # Summary block — verdict + flag list + document type
    _sev_colors = SEVERITY_COLORS

    flags_list_html = ""
    # Verdict first: only the few most severe findings sit next to
    # the verdict (src/summary.py:top_reasons picks them). The full
    # list, with severities and raw details, is in the "see all
    # details" block at the bottom of the page.

    def _finding_row(flag, number: int | None) -> str:
        """One line of the findings list, as HTML.

                Left: a numbered red disc when the finding points at a region
                of the document — the SAME number drawn on the preview
                (src/preview.py:_draw_number_badge) — otherwise a plain dot
                whose colour ranks the finding (green = reassuring signal).
                Right: the finding as one sentence.
                """
        safe_sentence = html.escape(get_flag_sentence(flag, lang))
        if number is not None:
            marker = (
                '<span style="background:#ef233c; color:white; width:20px; height:20px; '
                'border-radius:50%; flex-shrink:0; font-size:0.72rem; font-weight:bold; '
                f'display:inline-flex; align-items:center; justify-content:center;">{number}</span>'
            )
        else:
            dot_color = ("#28a745" if flag.code in GENUINE_CODES
                         else _sev_colors.get(flag.severity, "#555"))
            marker = (
                f'<span style="background:{dot_color}; width:9px; height:9px; border-radius:50%; '
                'flex-shrink:0; margin:6px 5.5px 0 5.5px;"></span>'
            )
        return (
            '<div style="display:flex; align-items:flex-start; gap:10px; padding:7px 0; '
            'border-bottom:1px solid #2a2a2a;">'
            f'{marker}'
            f'<span style="color:#ddd; font-size:0.9rem; line-height:1.4;">{safe_sentence}</span>'
            '</div>'
        )

    for flag, _number in zip(_reasons, _reason_numbers):
        flags_list_html += _finding_row(flag, _number)

    # "Show more": the other findings unfold right here, in the same
    # list, instead of sending the reader to the bottom of the page.
    # The link says how many there are, so a short list never reads
    # as "that is all there is". `is` comparison: two different flags
    # can be equal field by field, identity is what "already shown"
    # means. (Opened automatically when the page is printed.)
    _other_flags = [f for f in all_flags if not any(f is r for r in _reasons)]
    if _other_flags:
        _more_key = "more_findings_one" if len(_other_flags) == 1 else "more_findings_many"
        flags_list_html += (
            '<details>'
            '<summary class="tf-disclosure tf-disclosure-small">'
            f'<span class="tf-when-closed">{esc(t(_more_key).format(n=len(_other_flags)))}</span>'
            f'<span class="tf-when-open">{esc(t("hide_findings"))}</span></summary>'
            + "".join(_finding_row(f, None) for f in _other_flags) +
            '</details>'
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

    _section_title = SECTION_TITLE
    # Modification history, drawn as a vertical timeline: one dot per
    # step in the life of the file (grey = creation, red = a later
    # save), joined by a line, with the date on the right and each
    # change written "old -> new". Comes after the reasons and the
    # feedback question. Only rendered when the PDF has several
    # versions with a diff to show.
    _history_html = ""
    if history["version_count"] > 1 and history.get("diffs"):
        from src.web.views.history import fill_covered_values, pair_changes

        # new text -> text it covers, from the cover-and-retype
        # findings: lets an "added" line show the value it replaced.
        _covered_by = {
            f.details["replacement"]: f.details["covered"]
            for f in all_flags
            if f.details and f.details.get("replacement") and f.details.get("covered")
        }

        max_changes = 3  # changes shown per step; the rest is counted
        _date_format = "%d/%m/%Y %H:%M" if lang == "fr" else "%Y-%m-%d %H:%M"
        _created_at = view.metadata.creation_date
        _modified_at = view.metadata.mod_date

        def _timeline_step(title: str, when: str, body_html: str,
                           dot_color: str, is_last: bool) -> str:
            """One step of the timeline, as HTML.

                    Left: the dot, and under it the line down to the next
                    step (no line under the last one). Right: the title with
                    the date pushed to the far right, then the body.

                    Example:
                        _timeline_step("Modification", "12/03/2026 14:05",
                                       "<div>...</div>", "#ef233c", True)
                    """
            line_html = ("" if is_last else
                         '<div style="width:1px; flex:1; background:#3a3a4e; margin-top:3px;"></div>')
            when_html = (f'<span style="color:#888; font-size:0.75rem; white-space:nowrap;">{when}</span>'
                         if when else "")
            return (
                '<div style="display:flex; gap:10px;">'
                '<div style="display:flex; flex-direction:column; align-items:center;">'
                f'<div style="width:10px; height:10px; border-radius:50%; background:{dot_color};'
                ' margin-top:5px; flex-shrink:0;"></div>'
                f'{line_html}</div>'
                f'<div style="flex:1; min-width:0; padding-bottom:{"0" if is_last else "12px"};">'
                '<div style="display:flex; justify-content:space-between; align-items:baseline; gap:8px;">'
                f'<span style="color:#eee; font-size:0.85rem;">{title}</span>{when_html}</div>'
                f'{body_html}</div></div>'
            )

        # Step 1: creation. The tool that made the file is the
        # "from_tool" of the first diff.
        first_software = history["diffs"][0].get("from_tool", "")
        _tool_style = "color:#888; font-size:0.75rem;"
        steps_html = _timeline_step(
            t("history_step_created"),
            _created_at.strftime(_date_format) if _created_at else "",
            (f'<div style="{_tool_style}">{html.escape(first_software)}</div>'
             if first_software else ""),
            "#888",
            is_last=False,
        )

        # One step per later save.
        _modification_count = len(history["diffs"])
        max_visible_modifications = 2  # later saves are folded
        modification_steps: list[str] = []
        for position, diff in enumerate(history["diffs"], start=1):
            is_last = position == _modification_count
            body_html = ""

            version_software = diff.get("to_tool", "")
            if version_software:
                body_html += f'<div style="{_tool_style}">{html.escape(version_software)}</div>'

            # A PDF stores two dates only (creation, last
            # modification), so only the LAST save can be dated; it
            # also gets "+3min after creation".
            when = ""
            if is_last and _modified_at:
                when = _modified_at.strftime(_date_format)
                if _created_at:
                    delta = _modified_at - _created_at
                    hours = delta.total_seconds() / 3600
                    _ac = t("after_creation")
                    if hours >= 24:
                        _day = "j" if lang == "fr" else "d"  # jours / days
                        elapsed = f"+{int(hours // 24)}{_day} {int(hours % 24)}h"
                    elif hours >= 1:
                        elapsed = f"+{int(hours)}h{int((hours % 1) * 60)}min"
                    elif int(hours * 60) > 0:
                        elapsed = f"+{int(hours * 60)}min"
                    else:
                        elapsed = f"+{int(delta.total_seconds())}s"
                    body_html += f'<div style="{_tool_style}">{elapsed} {_ac}</div>'

            # Every change of this save, as (old, new) pairs.
            pairs = []
            for change in diff["changes"] or []:
                pairs += pair_changes(change.get("removed", []), change.get("added", []))
            pairs = fill_covered_values(pairs, _covered_by)

            for old, new in pairs[:max_changes]:
                old_html = (f'<span style="color:#e06060; text-decoration:line-through;">'
                            f'{html.escape(old[:80])}</span>' if old else "")
                new_html = (f'<span style="color:#60c060; font-weight:600;">'
                            f'{html.escape(new[:80])}</span>' if new else "")
                # The arrow only makes sense between two values.
                arrow_html = ('<span style="color:#888; margin:0 6px;">&rarr;</span>'
                              if old and new else "")
                body_html += (f'<div style="font-size:0.82rem; margin-top:4px;">'
                              f'{old_html}{arrow_html}{new_html}</div>')
            if len(pairs) > max_changes:
                body_html += (f'<div style="color:#888; font-size:0.75rem; font-style:italic; margin-top:2px;">'
                              f'{t("and_n_more_lines").format(n=len(pairs) - max_changes)}</div>')
            if not pairs:
                body_html += f'<div style="{_tool_style}">{t("non_text_changes")}</div>'

            title = (t("history_step_modified") if _modification_count == 1
                     else t("history_step_modified_n").format(n=position))
            modification_steps.append(
                _timeline_step(title, when, body_html, "#ef233c", is_last))

        # Preview: a file saved many times would push everything else
        # down the page. The first saves stay visible, the rest waits
        # behind a "show N more" line (opened automatically on print).
        visible_steps = modification_steps[:max_visible_modifications]
        hidden_steps = modification_steps[max_visible_modifications:]
        steps_html += "".join(visible_steps)
        if hidden_steps:
            steps_html += (
                '<details>'
                '<summary style="cursor:pointer; color:#8ab4f8; font-size:0.82rem; padding:2px 0 8px 0;">'
                f'{t("history_show_more").format(n=len(hidden_steps))}</summary>'
                + "".join(hidden_steps) +
                '</details>'
            )

        _history_html = (
            '<div style="margin:2rem auto 0 auto; max-width:440px;">'
            + _section_title.format(t("full_modification_history"))
            + steps_html +
            '</div>'
        )

    # Findings first (the verdict itself now sits under the gauge),
    # then where the evidence comes from, then one discreet line of
    # context. Same section-title style as "Preview" on the right.
    _no_findings_html = (f'<p style="color:#28a745; font-size:0.9rem; margin:0;">{t("no_issues")}</p>'
                         if not all_flags else "")
    _summary_html = (
        '<div style="margin-top:1rem; max-width:440px; margin-left:auto; margin-right:auto;">'
        + _section_title.format(t("summary")) +
        f'{_banner_html}'
        f'{flags_list_html}{_no_findings_html}'
        '</div>'
        # History after the reasons: the reader first learns WHAT is
        # wrong, then sees the proof of how the file changed. The
        # "where the evidence comes from" bars moved into the detail
        # fold: the reasons and the numbered badges already say it.
        + _history_html +
        f'<p style="text-align:center; font-size:0.72rem; color:#888; margin:1.2rem 0 0 0;">'
        f'{t("doctype_label")}&nbsp;: <strong style="color:#bbb;">{_dt_label}</strong>'
        f' &middot; {t("analysis_completed").format(ms=result.analysis_time_ms)}</p>'
    )
    return _summary_html
