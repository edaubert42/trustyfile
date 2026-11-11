""""See all details" fold: every finding, and one panel per kind of evidence."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from datetime import date as _d

from src.i18n import translator
from src.models import AnalysisResult
from src.modules.content import (
    extract_french_vat, extract_iban, extract_rcs, extract_siren, extract_siret,
)
from src.summary import get_flag_sentence
from src.web.viewmodel import DocumentView
from src.web.views.common import SECTION_TITLE, SEVERITY_COLORS, esc
from src.web.views.links import guide_url_for_flag, tool_url_for_flag


def issues_html(all_flags: list, document_type: str, lang: str) -> str:
    """
    Every finding as a foldable line: severity pill, sentence, raw details.

    Args:
        all_flags: All findings of the document.
        document_type: Its detected genre, to pick the right guide link.
        lang: "fr" or "en".

    Returns:
        HTML string, empty when the document has no finding.
    """
    t = translator(lang)
    issues_html = ""
    if all_flags:

        severity_colors = SEVERITY_COLORS

        # Build all issues as a single HTML block with <details>/<summary>
        # This lets us put colored pills on the clickable line
        for flag in all_flags:
            color = severity_colors.get(flag.severity, "#555")
            safe_msg = html.escape(get_flag_sentence(flag, lang))
            safe_code = html.escape(flag.code)

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
                    safe_key = html.escape(str(key).replace("_", " ").capitalize())
                    # Format value: lists as comma-separated, dicts as JSON, rest as string
                    if isinstance(val, list):
                        safe_val = html.escape(", ".join(str(v) for v in val))
                    elif isinstance(val, dict):
                        safe_val = html.escape(json.dumps(val, ensure_ascii=False, default=str))
                    else:
                        safe_val = html.escape(str(val))
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
            tool_url = tool_url_for_flag(flag.code, lang)
            guide_url = guide_url_for_flag(flag.code, lang, document_type)
            if tool_url:
                details_content += (
                    f'<div style="margin-top:8px;"><a href="{tool_url}" target="_blank" rel="noopener" '
                    f'style="color:#8ab4f8; font-size:0.8rem; text-decoration:none;">'
                    f'{html.escape(t("flag_check_tool"))} &rarr;</a></div>'
                )
            elif guide_url:
                # One action per flag: identifier flags get the checker (which
                # itself links the guides), every other flag gets the guide
                # that explains what was found and what to do next.
                details_content += (
                    f'<div style="margin-top:8px;"><a href="{guide_url}" target="_blank" rel="noopener" '
                    f'style="color:#8ab4f8; font-size:0.8rem; text-decoration:none;">'
                    f'{html.escape(t("flag_read_guide"))} &rarr;</a></div>'
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
    return issues_html


def history_panel_html(view: DocumentView, lang: str) -> str:
    """
    Detailed modification history: one card per saved version, with its diff.

    Returns:
        HTML string, empty when the PDF has a single version.
    """
    t = translator(lang)
    history = view.history
    more_details_html = ""
    if history["version_count"] > 1 and history.get("diffs"):

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
        if view.metadata.creation_date:
            v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_date")}</span>'
                        f'<span style="{_mval}">{view.metadata.creation_date.strftime("%Y-%m-%d %H:%M")}</span></div>')
        if first_sw:
            v1_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_producer")}</span>'
                        f'<span style="{_mval}">{html.escape(first_sw)}</span></div>')

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
            if to_v == history["version_count"] and view.metadata.mod_date and view.metadata.creation_date:
                delta = view.metadata.mod_date - view.metadata.creation_date
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
                           f'<span style="{_mval}">{view.metadata.mod_date.strftime("%Y-%m-%d %H:%M")} '
                           f'<span style="color:#e06060;">({delta_str} {t("after_creation")})</span></span></div>')

            if to_sw and to_sw != from_sw:
                v_meta += (f'<div style="{_mrow}"><span style="{_mkey}">{t("label_editor")}</span>'
                           f'<span style="{_mval}">{html.escape(to_sw)}</span></div>')

            diff_html = ""
            if diff["changes"]:
                for change in diff["changes"]:
                    diff_html += f'<div style="color:#888; font-size:0.7rem; margin-bottom:4px;">{t("label_page")} {change["page"]}</div>'
                    for line in change.get("removed", []):
                        if line.strip():
                            diff_html += f'<div style="color:#e06060;">- {html.escape(line)}</div>'
                    for line in change.get("added", []):
                        if line.strip():
                            diff_html += f'<div style="color:#60c060;">+ {html.escape(line)}</div>'
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
    return more_details_html


def modules_panel_html(result: AnalysisResult, lang: str) -> str:
    """Per-module raw data: score, findings and their details."""
    t = translator(lang)
    more_details_html = ""
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

        # Module card with full raw data
        flags_html = ""
        if module.flags:
            for f in module.flags:
                f_color = SEVERITY_COLORS.get(f.severity, "#555")
                safe_code = html.escape(f.code)
                safe_msg = html.escape(get_flag_sentence(f, lang))

                # Flag details as key/value rows
                details_html = ""
                if f.details:
                    _drow = "display:flex; padding:3px 0; border-bottom:1px solid #151515;"
                    _dkey = "color:#777; font-size:0.73rem; width:130px; flex-shrink:0;"
                    _dval = "color:#bbb; font-size:0.73rem; word-break:break-word;"

                    detail_rows = ""
                    for dk, dv in f.details.items():
                        safe_dk = html.escape(str(dk).replace("_", " ").capitalize())
                        if isinstance(dv, list):
                            safe_dv = html.escape(", ".join(str(v) for v in dv))
                        elif isinstance(dv, dict):
                            safe_dv = html.escape(json.dumps(dv, ensure_ascii=False, default=str))
                        elif isinstance(dv, bool):
                            safe_dv = f'<span style="color:{"#60c060" if dv else "#e06060"};">{"true" if dv else "false"}</span>'
                        else:
                            safe_dv = html.escape(str(dv))
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
    return more_details_html


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


def entities_panel_html(result: AnalysisResult, full_text: str, lang: str) -> str:
    """
    Companies named in the document: verified ones, then bare identifiers.

    Args:
        result: The analysis (the external module holds verified companies).
        full_text: The document's text, all pages.
        lang: "fr" or "en".

    Returns:
        HTML string, empty when no company or identifier was found.
    """
    t = translator(lang)
    more_details_html = ""
    # More Details — Entities block (verified companies + unverified identifiers)
    external_module = next((m for m in result.modules if m.module == "external"), None)
    verified_companies = []
    if external_module and external_module.details.get("verified_companies"):
        verified_companies = external_module.details["verified_companies"]

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
        name = html.escape(comp.get("name") or "Unknown")
        siren = comp.get("siren") or ""
        siret = comp.get("siret") or ""
        address = html.escape(comp.get("address") or "")
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

    entity_name = _extract_entity_name(full_text)
    entity_address = _extract_entity_address(full_text)

    for siren_key, idents in unverified_by_siren.items():
        # Entity label: use company name if found, otherwise "SIREN XXXXXXXXX"
        label = html.escape(entity_name) if entity_name else f"SIREN {siren_key}"

        rows = ""
        if idents["siret"]:
            rows += f'<div style="{_row}"><span style="{_key}">SIRET</span><span style="{_val}">{idents["siret"]}</span></div>'
        rows += f'<div style="{_row}"><span style="{_key}">SIREN</span><span style="{_val}">{siren_key}</span></div>'
        if idents["vat"]:
            rows += f'<div style="{_row}"><span style="{_key}">VAT</span><span style="{_val}">{idents["vat"]}</span></div>'
        if idents["rcs"]:
            rows += f'<div style="{_row}"><span style="{_key}">RCS</span><span style="{_val}">{idents["rcs"]}</span></div>'
        if entity_address:
            rows += f'<div style="{_row}"><span style="{_key}">{t("label_address")}</span><span style="{_val}">{html.escape(entity_address)}</span></div>'

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
    return more_details_html


def bank_panel_html(result: AnalysisResult, full_text: str, lang: str) -> str:
    """
    Bank accounts: IBANs verified by the API, then IBANs only read in the text.

    Returns:
        HTML string, empty when the document has no IBAN.
    """
    t = translator(lang)
    more_details_html = ""
    external_module = next((m for m in result.modules if m.module == "external"), None)
    # Bank Information panel (IBAN verification results)
    verified_ibans = []
    if external_module and external_module.details.get("verified_ibans"):
        verified_ibans = external_module.details["verified_ibans"]

    # Also show IBANs found in text even without API verification
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
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_bank")}</span><span style="{_val}">{html.escape(bank_name)}</span></div>'
            branch_name = iban_data.get("branch_name") or ""
            if branch_name:
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_branch")}</span><span style="{_val}">{html.escape(branch_name)}</span></div>'
            country = iban_data.get("country_code", "")
            if country:
                rows += f'<div style="{_row}"><span style="{_key}">{t("label_country")}</span><span style="{_val}">{country}</span></div>'

            title_raw = unicodedata.normalize("NFD", bank_name)
            title_raw = "".join(c for c in title_raw if unicodedata.category(c) != "Mn")
            title = html.escape(title_raw).upper() if bank_name else t("bank_account").upper()
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
    return more_details_html


def mrz_panel_html(result: AnalysisResult, lang: str) -> str:
    """
    Machine-readable zone of an identity document: fields and coherence checks.

    Returns:
        HTML string, empty when the document has no MRZ.
    """
    t = translator(lang)
    more_details_html = ""
    # MRZ Zone panel (identity documents)
    identity_mod = next((m for m in result.modules if m.module == "identity"), None)
    if identity_mod and identity_mod.details.get("mrz_found"):

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

        # --- Champs principaux MRZ ---
        mrz_rows = ""

        # Nom
        if det.get("name"):
            mrz_rows += (
                f'<div style="{_mrz_row}">'
                f'<span style="{_mrz_key}">{t("mrz_name")}</span>'
                f'<span style="{_mrz_val_bold}">{html.escape(det["name"])}</span>'
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
                f'<span style="{_mrz_val_bold}">{html.escape(dob_str)}</span>'
                f'</div>'
            )

        # Nationalité
        if det.get("nationality"):
            mrz_rows += (
                f'<div style="{_mrz_row}">'
                f'<span style="{_mrz_key}">{t("mrz_nationality")}</span>'
                f'<span style="{_mrz_val}">{html.escape(det["nationality"])}</span>'
                f'</div>'
            )

        # Sexe
        if det.get("sex"):
            sex_map = {"M": t("mrz_sex_male"), "F": t("mrz_sex_female"), "<": t("mrz_sex_unspecified")}
            sex_label = sex_map.get(det["sex"], det["sex"])
            mrz_rows += (
                f'<div style="{_mrz_row}">'
                f'<span style="{_mrz_key}">{t("mrz_sex")}</span>'
                f'<span style="{_mrz_val}">{html.escape(sex_label)}</span>'
                f'</div>'
            )

        # Numéro de document
        if det.get("doc_number"):
            mrz_rows += (
                f'<div style="{_mrz_row}">'
                f'<span style="{_mrz_key}">{t("mrz_doc_number")}</span>'
                f'<span style="{_mrz_val_bold}">{html.escape(det["doc_number"])}</span>'
                f'</div>'
            )

        # Date de délivrance (extraite par OCR pleine page — non encodée dans la MRZ)
        if det.get("issuance_date"):
            try:
                issuance_display = _d.fromisoformat(det["issuance_date"]).strftime('%d/%m/%Y')
            except Exception:
                issuance_display = det["issuance_date"]
            mrz_rows += (
                f'<div style="{_mrz_row}">'
                f'<span style="{_mrz_key}">{t("mrz_issuance")}</span>'
                f'<span style="{_mrz_val}">{html.escape(issuance_display)}</span>'
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
                f'<span style="{_mrz_val_bold}">{html.escape(expiry_str)}{exp_badge}</span>'
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
                f'<span style="{_mrz_val}">{dt_icon}{html.escape(dt_text)} ({html.escape(dt_prefix)})</span>'
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
                    f'<span style="{_mrz_val}">{is_icon}{html.escape(issuer.replace("<", ""))} '
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
                f'<span style="{_mrz_val}">{ela_icon}{html.escape(ela_text)}</span>'
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
                f'<span style="{_mrz_val}">{noise_icon}{html.escape(noise_text)}</span>'
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
            f'<span style="color:#888; font-size:0.72rem;">{html.escape(doc_type_label)}</span>'
            f'</summary>'
            f'<div style="padding:14px; border-top:1px solid #2563eb;">'
            f'{mrz_inner}'
            f'</div></details>'
        )
    return more_details_html


def _detail_section(title: str, intro: str, body_html: str) -> str:
    """One titled section of the fold: heading, one grey line, then the body.

    Returns an empty string when there is no body, so a document without
    (say) bank details does not show an empty heading.
    """
    if not body_html:
        return ""
    return (
        '<section style="margin-top:1.6rem;">'
        + SECTION_TITLE.format(esc(title))
        + f'<p style="color:#888; font-size:0.8rem; line-height:1.5; margin:0 0 0.7rem 0;">{esc(intro)}</p>'
        + body_html
        + '</section>'
    )


def all_details_html(bars_html: str, issues: str, file_panels: str,
                     technical_panels: str, lang: str) -> str:
    """
    The single fold holding everything below the verdict.

    The page opens on the verdict and the preview only; a reader who wants
    the evidence opens this block. It answers four questions in order,
    each a titled section with one line of explanation:

        Why this score?          the four bars
        Every point found        each point, foldable to its raw data
        What the file contains   history, companies, bank, 2D-Doc, MRZ
        Going further            the result of each check

    The caveat about the score closes the block, as a note.

    Args:
        bars_html: evidence_bars_html() of the document.
        issues: issues_html() of the document (may be empty).
        file_panels: What was read in the file, panels concatenated
            (may be empty).
        technical_panels: The per-check panel (may be empty).
        lang: "fr" or "en".

    Returns:
        The fold as an HTML string.
    """
    t = translator(lang)
    body = (
        _detail_section(t("detail_why_title"), t("detail_why_intro"),
                        f'<div style="max-width:440px;">{bars_html}</div>')
        + _detail_section(t("detail_points_title"), t("detail_points_intro"), issues)
        + _detail_section(t("detail_file_title"), t("detail_file_intro"), file_panels)
        + _detail_section(t("detail_tech_title"), t("detail_tech_intro"), technical_panels)
        + f'<p style="color:#888; font-size:0.8rem; line-height:1.5; margin:1.6rem 0 0 0;">'
          f'{esc(t("score_caveat"))}</p>'
    )
    return (
        '<details class="tf-all-details" style="margin-top:1.5rem;">'
        '<summary class="tf-disclosure tf-disclosure-large">'
        f'<span class="tf-when-closed">{esc(t("show_all_details"))}</span>'
        f'<span class="tf-when-open">{esc(t("hide_all_details"))}</span></summary>'
        f'{body}'
        '</details>'
    )
