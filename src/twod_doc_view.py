"""
HTML for 2D-Doc data, shared by the PDF results page and the photo scan view.

The rows of the PDF results page and of the photo scan live here so the
photo view (a 2D-Doc read from a phone photo) shows exactly the same fields,
labels and formatting without a second copy of ~120 lines.

Everything is escaped: barcode fields are attacker-controlled text.
"""

from __future__ import annotations

import html

from src.i18n import translator
from src.modules.twod_doc import TwoDocData, extract_verified_data


def twod_rows_html(twod_doc_data: TwoDocData, lang: str) -> str:
    """Rows (label / value) for one decoded 2D-Doc, as an HTML fragment.

    Known fields first (type, identity, address, tax, invoice), then every
    remaining field raw under "other fields", so nothing signed is hidden.

    Args:
        twod_doc_data: A parsed 2D-Doc (parse_twod_doc output).
        lang: "fr" or "en".

    Returns:
        HTML <div> rows, without the surrounding card.
    """
    t = translator(lang)
    verified = extract_verified_data(twod_doc_data)

    _2d_row = "display:flex; padding:4px 0; border-bottom:1px solid #151515;"
    _2d_key = "color:#888; font-size:0.75rem; width:160px; flex-shrink:0;"
    _2d_val = "color:#ccc; font-size:0.75rem;"
    _2d_val_bold = "color:white; font-size:0.75rem; font-weight:bold;"

    rows_2d = ""
    rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_doc_type")}</span>'
                f'<span style="{_2d_val}">{html.escape(verified.document_type)} '
                f'({html.escape(verified.document_type_name)})</span></div>')

    # Track which DIs we display so we can show remaining as raw
    shown_dis = set()

    # --- Identity fields (shared across document types) ---
    if verified.name:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_name")}</span>'
                    f'<span style="{_2d_val_bold}">{html.escape(verified.name)}</span></div>')
        shown_dis.update({"10", "12", "13", "44", "6G", "60"})

    # --- Driving license fields (type AB) ---
    if verified.civility:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_civility")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.civility)}</span></div>')
        shown_dis.add("6H")

    if verified.sex:
        sex_label = {"M": t("twod_sex_male"), "F": t("twod_sex_female")}.get(verified.sex, verified.sex)
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_sex")}</span>'
                    f'<span style="{_2d_val}">{html.escape(sex_label)}</span></div>')
        shown_dis.add("68")

    if verified.nationality:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_nationality")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.nationality)}</span></div>')
        shown_dis.update({"67", "6C"})

    if verified.birth_date:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_birth_date")}</span>'
                    f'<span style="{_2d_val_bold}">{html.escape(verified.birth_date)}</span></div>')
        shown_dis.add("69")

    if verified.birth_place:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_birth_place")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.birth_place)}</span></div>')
        shown_dis.add("6A")

    if verified.document_number:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_document_number")}</span>'
                    f'<span style="{_2d_val_bold}">{html.escape(verified.document_number)}</span></div>')
        shown_dis.add("65")

    if verified.permit_number:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_permit_number")}</span>'
                    f'<span style="{_2d_val_bold}">{html.escape(verified.permit_number)}</span></div>')
        shown_dis.add("AC")

    if verified.permit_categories:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_permit_categories")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.permit_categories)}</span></div>')
        shown_dis.add("E4")

    # --- Tax fields ---
    if verified.fiscal_number:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_tax_id")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.fiscal_number)}</span></div>')
        shown_dis.add("47")

    if verified.address or verified.postal_code or verified.city:
        addr_parts = []
        if verified.address:
            addr_parts.append(verified.address)
        if verified.postal_code or verified.city:
            addr_parts.append(f"{verified.postal_code or ''} {verified.city or ''}".strip())
        addr = ", ".join(addr_parts)
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_address")}</span>'
                    f'<span style="{_2d_val_bold}">{html.escape(addr)}</span></div>')
        shown_dis.update({"22", "24", "25"})

    if verified.reference_income is not None:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_reference_income")}</span>'
                    f'<span style="{_2d_val_bold}">{verified.reference_income:,.0f} &euro;</span></div>')
        shown_dis.add("41")

    if verified.tax_amount is not None:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_income_tax")}</span>'
                    f'<span style="{_2d_val_bold}">{verified.tax_amount:,.0f} &euro;</span></div>')
        shown_dis.add("4V")

    if verified.withheld_amount is not None:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_withholding_tax")}</span>'
                    f'<span style="{_2d_val}">{verified.withheld_amount:,.0f} &euro;</span></div>')
        shown_dis.add("4X")

    if verified.calculated_balance is not None:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_calculated_balance")}</span>'
                    f'<span style="{_2d_val_bold}">{verified.calculated_balance:,.0f} &euro;</span></div>')

    if verified.household_parts and verified.household_parts != 1.0:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_household_parts")}</span>'
                    f'<span style="{_2d_val}">{verified.household_parts}</span></div>')
        shown_dis.add("43")

    # --- Invoice fields ---
    if verified.invoice_number:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_invoice_number")}</span>'
                    f'<span style="{_2d_val}">{html.escape(verified.invoice_number)}</span></div>')

    if verified.invoice_amount is not None:
        rows_2d += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_invoice_amount")}</span>'
                    f'<span style="{_2d_val_bold}">{verified.invoice_amount:,.2f} &euro;</span></div>')

    # Show remaining unmapped fields as raw
    extra_fields = ""
    for field in twod_doc_data.fields:
        if field.di in shown_dis:
            continue
        field_val = html.escape(str(field.value))
        extra_fields += (f'<div style="{_2d_row}"><span style="{_2d_key}">{t("twod_field").format(di=field.di)}</span>'
                         f'<span style="{_2d_val}">{field_val}</span></div>')

    if extra_fields:
        rows_2d += (f'<div style="margin-top:8px; padding-top:6px; border-top:1px solid #2a2a2a;">'
                    f'<span style="color:#888; font-size:0.7rem; font-weight:bold;">{t("twod_raw_fields")}</span>'
                    f'</div>{extra_fields}')

    return rows_2d


# Colours of the photo verdict banner, by level (see twod_doc_photo.photo_verdict).
# Same palette as the rest of the results page: green #28a745 is the PDF
# 2D-Doc panel's, red #e05252 the error cards', gray-blue for "no conclusion".
_LEVEL_STYLE = {
    "ok":      {"border": "#28a745", "bg": "rgba(40,167,69,0.12)",  "fg": "#5fd47c", "icon": "✓"},
    "alert":   {"border": "#e05252", "bg": "rgba(224,82,82,0.12)",  "fg": "#ff7b7b", "icon": "!"},
    "neutral": {"border": "#5b6b82", "bg": "rgba(91,107,130,0.12)", "fg": "#b8c4d6", "icon": "?"},
}


def photo_result_html(result, level: str, key: str, lang: str) -> str:
    """The photo scan view: verdict banner, then the signed data, then advice.

    Args:
        result: PhotoScanResult from twod_doc_photo.scan_photo.
        level, key: photo_verdict(result) — the banner colour and the
            translation key of its message (title is key + "_title").
        lang: "fr" or "en".

    Returns:
        One HTML fragment (rendered with st.markdown(unsafe_allow_html=True)).
        The fields of the code are shown only when it was decoded; the
        "compare with the paper" warning only under a valid signature, a
        "not guaranteed" caption otherwise.
    """
    t = translator(lang)
    style = _LEVEL_STYLE[level]
    parts = [
        f'<div id="photo-result" style="max-width:640px; margin:1.5rem auto; padding:0 1rem;">',
        f'<div style="color:#888; font-size:0.8rem; text-transform:uppercase; '
        f'letter-spacing:0.08em; margin-bottom:0.6rem;">{html.escape(t("photo_scan_title"))}</div>',
        # Verdict banner
        f'<div data-level="{level}" style="display:flex; gap:0.9rem; align-items:flex-start; '
        f'background:{style["bg"]}; border:1px solid {style["border"]}; border-radius:10px; '
        f'padding:1rem 1.2rem;">',
        f'<div style="flex-shrink:0; width:2rem; height:2rem; border-radius:50%; '
        f'background:{style["border"]}; color:white; font-weight:bold; display:flex; '
        f'align-items:center; justify-content:center;">{style["icon"]}</div>',
        f'<div><div style="color:{style["fg"]}; font-size:1.1rem; font-weight:bold; '
        f'margin-bottom:0.3rem;">{html.escape(t(key + "_title"))}</div>',
        f'<div style="color:#ccc; font-size:0.85rem; line-height:1.45;">{html.escape(t(key))}</div>',
        '</div></div>',
    ]

    if result.status == "decoded" and result.twod_doc is not None:
        try:
            rows = twod_rows_html(result.twod_doc, lang)
        except Exception:  # a malformed signed field must not break the page
            rows = (f'<div style="color:#888; font-size:0.75rem;">'
                    f'{html.escape(t("photo_fields_error"))}</div>')
        # Under a valid signature: "authentic data, now check it is THIS
        # applicant's". Under anything else the same rows are only what the
        # code claims, and must not read as vouched for.
        caption = "photo_compare_warning" if level == "ok" else "photo_unsigned_fields"
        parts += [
            f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
            f'padding:12px 16px; margin-top:1rem;">{rows}</div>',
            f'<div style="color:#999; font-size:0.78rem; line-height:1.45; margin-top:0.8rem;">'
            f'{html.escape(t(caption))}</div>',
        ]

    parts += [
        '<div style="text-align:center; margin-top:1.5rem;">'
        '<a href="." target="_self" style="padding:0.6rem 1.5rem; background:#4a90e2; color:white; '
        f'border-radius:8px; text-decoration:none; font-weight:500;">{html.escape(t("back_to_home"))}</a>'
        '</div>',
        '</div>',
    ]
    return "".join(parts)


def pdf_signature_badge(statuses: list[str]) -> tuple[str, str]:
    """Badge of the PDF 2D-Doc panel, from the signature status of each code.

    Pure function, the single place that decides what the PDF panel claims.
    Before 2026-11-05 the panel always said "Verified" in green, even for a
    code whose signature had been checked and found INVALID: the green badge
    contradicted the critical flag shown a few lines above it.

    Same posture as twod_doc_photo.photo_verdict:
    - green ("ok") only if EVERY code on the page has a valid signature —
      one unchecked code must not ride on another one's proof;
    - red ("alert") as soon as one code is "invalid" (certificate resolved,
      maths failed). Cautious wording ("does not match"), because certificate
      resolution is candidate matching (see twod_doc._signature_flags);
    - neutral ("neutral") for everything else: test certificate, issuer
      missing from our snapshot, no signature... We could not check, which
      proves nothing either way.

    Args:
        statuses: SignatureCheck.status of each decoded code, any order.

    Returns:
        (level, translation key). level is a key of _LEVEL_STYLE.

    Example:
        >>> pdf_signature_badge(["valid"])
        ('ok', 'twod_sig_valid')
        >>> pdf_signature_badge(["valid", "invalid"])
        ('alert', 'twod_sig_invalid')
        >>> pdf_signature_badge(["cert_not_found"])
        ('neutral', 'twod_sig_unverifiable')
    """
    if not statuses:
        # No status at all (crypto check crashed): no claim.
        return "neutral", "twod_sig_unverifiable"
    if "invalid" in statuses:
        return "alert", "twod_sig_invalid"
    if all(status == "valid" for status in statuses):
        return "ok", "twod_sig_valid"
    if "test_cert" in statuses:
        return "neutral", "twod_sig_test_cert"
    return "neutral", "twod_sig_unverifiable"


def twod_pdf_panel_html(twod_docs: list[TwoDocData], statuses: list[str],
                        lang: str) -> str:
    """The 2D-Doc panel of the PDF results page (one card per decoded code).

    The panel colour and its badge follow pdf_signature_badge: a forged code
    no longer sits in a green "Verified" frame.

    Args:
        twod_docs: parsed 2D-Docs found on the page (scan_pdf_for_2d_doc).
        statuses: signature status of each of them (same order).
        lang: "fr" or "en".

    Returns:
        A <details> HTML fragment, or "" when there is no code.
    """
    t = translator(lang)
    if not twod_docs:
        return ""

    level, key = pdf_signature_badge(statuses)
    style = _LEVEL_STYLE[level]

    cards = ""
    for twod_doc_data in twod_docs:
        try:
            rows = twod_rows_html(twod_doc_data, lang)
        except Exception:  # a malformed signed field must not break the page
            rows = (f'<div style="color:#888; font-size:0.75rem;">'
                    f'{html.escape(t("photo_fields_error"))}</div>')
        cards += (
            f'<div style="background:#111; border:1px solid #222; border-radius:8px; '
            f'padding:12px 16px; margin-bottom:8px;">{rows}</div>'
        )

    return (
        f'<details open data-level="{level}" style="background:{style["bg"]}; '
        f'border:1px solid {style["border"]}; border-radius:8px; padding:0; margin-bottom:8px;">'
        f'<summary style="cursor:pointer; padding:10px 14px; list-style:none; '
        f'display:flex; flex-wrap:wrap; align-items:center; gap:10px;">'
        f'<span style="color:#eee; font-size:0.85rem;">{html.escape(t("twod_doc"))}</span>'
        f'<span style="background:{style["border"]}; color:white; font-size:0.68rem; '
        f'font-weight:bold; padding:2px 10px; border-radius:10px;">{html.escape(t(key))}</span>'
        f'<span style="color:#888; font-size:0.72rem;">{html.escape(t("twod_compare"))}</span>'
        f'</summary>'
        f'<div style="padding:14px; border-top:1px solid {style["border"]};">{cards}</div>'
        f'</details>'
    )
