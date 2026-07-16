"""
Rich Template-Based Summary Generator for TrustyFile.

Groups flags by user-facing themes (not modules) and produces a natural-feeling
two-part summary: a short verdict + a multi-sentence explanation.

All template-based — no LLM, free, instant, works offline.
"""

from src.models import AnalysisResult, AnalysisSummary, ModuleResult, Flag
from src.scoring import collect_all_flags


# =============================================================================
# THEME DEFINITIONS
# =============================================================================

# Each flag code is mapped to a user-facing theme.
# Themes group related findings so the summary reads naturally.
FLAG_THEME_MAP: dict[str, str] = {
    # origin: online converters, AI tools, suspicious software
    "META_AI_GENERATED": "origin",
    "META_ONLINE_CONVERTER": "origin",
    "META_SUSPICIOUS_PRODUCER": "origin",
    "META_NO_METADATA": "origin",
    "META_NO_PRODUCER": "origin",
    "META_IMAGE_EDITOR": "origin",
    "META_IMAGE_CONVERSION": "origin",

    # tampering: incremental edits, paste artifacts, font mixing, deleted objects
    "STRUCT_INCREMENTAL_UPDATES": "tampering",
    "STRUCT_DELETED_OBJECTS": "tampering",
    "STRUCT_HIDDEN_MODIFICATION": "tampering",
    "FONTS_EXCESSIVE_DIVERSITY": "tampering",
    "FONTS_HIGH_DIVERSITY": "tampering",
    "FONTS_MIXED_SUBSETS": "tampering",
    "FONTS_MIDLINE_CHANGE": "tampering",
    "IMAGES_PASTE_NOISE_ANOMALY": "tampering",

    # tampering: opaque-box cover-and-retype / edit-trace overlays (Phase 2)
    "OVERLAY_COVER_AND_RETYPE": "tampering",
    "OVERLAY_COVER_AND_RETYPE_CONFIRMED": "tampering",
    "OVERLAY_OPAQUE_FILL_OVER_TEXT": "tampering",
    "OVERLAY_CONFLICTING_TEXT": "tampering",
    "OVERLAY_REVISION_VALUE_CHANGED": "tampering",
    "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED": "tampering",
    "OVERLAY_IMAGE_OVER_VALUE": "tampering",
    "OVERLAY_IMAGE_OVER_TEXT": "tampering",

    # tampering: arithmetic inconsistencies (e-invoice numbers that don't add up)
    "ARITH_XML_INTERNAL_INCONSISTENT": "tampering",
    "ARITH_XML_VISIBLE_MISMATCH": "tampering",
    "ARITH_IBAN_MISMATCH": "tampering",
    "ARITH_XML_CONSISTENT": "tampering",       # genuine-direction; bullets skip it

    # tampering: signed 2D-Doc barcode vs visible text consistency
    "TWOD_DOC_FIELD_MISMATCH": "tampering",
    "TWOD_DOC_CONSISTENT": "tampering",        # genuine-direction; bullets skip it

    # tampering: modification time gap, impossible dates in metadata
    "META_DOCUMENT_MODIFIED": "tampering",
    "META_DOCUMENT_REPAIRED": "tampering",
    "META_FUTURE_CREATION_DATE": "tampering",
    "META_IMPOSSIBLE_DATES": "tampering",

    # tampering: Error Level Analysis (image re-compression forensics)
    "FORENSICS_ELA_MAJOR_EDIT": "tampering",
    "FORENSICS_ELA_MINOR_EDIT": "tampering",

    # dates: content-level date inconsistencies (anachronisms, future dates in text)
    "CONTENT_FAR_FUTURE_DATE": "dates",
    "CONTENT_VERY_OLD_DATE": "dates",
    "CONTENT_ANACHRONISM_SERVICE": "dates",
    "CONTENT_ANACHRONISM_DUE": "dates",
    "CONTENT_ANACHRONISM_ORDER": "dates",
    "CONTENT_FUTURE_DOCUMENT_DATE": "dates",
    "CONTENT_DATE_CREATION_MISMATCH": "dates",
    "CONTENT_EXPIRED_DOCUMENT": "dates",

    # identity: invalid SIRET/SIREN/VAT, company mismatch
    "CONTENT_INVALID_SIRET": "identity",
    "CONTENT_INVALID_IBAN": "identity",
    "CONTENT_INVALID_SIREN": "identity",
    "CONTENT_INVALID_VAT": "identity",
    "CONTENT_SIREN_VAT_MISMATCH": "identity",
    "CONTENT_MISSING_COMPANY_ID": "identity",
    "CONTENT_INCONSISTENT_REFERENCES": "identity",
    "CONTENT_REFERENCE_DATE_MISMATCH": "identity",
    "EXTERNAL_SIRET_NOT_FOUND": "identity",
    "EXTERNAL_SIREN_NOT_FOUND": "identity",
    "EXTERNAL_COMPANY_CLOSED": "identity",
    "EXTERNAL_COMPANY_NAME_MISMATCH": "identity",
    "EXTERNAL_VAT_INVALID": "identity",
    "EXTERNAL_SIRET_VERIFICATION_FAILED": "identity",
    "EXTERNAL_SIREN_VERIFICATION_FAILED": "identity",
    "EXTERNAL_VAT_VERIFICATION_FAILED": "identity",

    # identity (MRZ family — machine-readable zone of ID documents)
    "MRZ_DOC_NUMBER_CHECK_FAIL": "identity",
    "MRZ_DOB_CHECK_FAIL": "identity",
    "MRZ_DOB_IMPOSSIBLE": "identity",
    "MRZ_EXPIRY_CHECK_FAIL": "identity",
    "MRZ_DOCUMENT_EXPIRED": "identity",
    "MRZ_OPTIONAL_CHECK_FAIL": "identity",
    "MRZ_FINAL_CHECK_FAIL": "identity",
    "MRZ_INVALID_SEX": "identity",
    "MRZ_DOC_TYPE_UNEXPECTED": "identity",
    "MRZ_PASSPORT_TYPE_UNUSUAL": "identity",
    "MRZ_ISSUER_FORMAT_INVALID": "identity",
    "MRZ_NAME_FORMAT_INVALID": "identity",
    "MRZ_NATIONALITY_FORMAT_INVALID": "identity",
    "MRZ_DOC_NUMBER_FORMAT_INVALID": "identity",
    "MRZ_CNI_TYPE_UNEXPECTED": "identity",
    "MRZ_CNI_OPTIONAL2_NOT_EMPTY": "identity",
    "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH": "identity",
    "MRZ_US_VALIDITY_PERIOD_UNEXPECTED": "identity",
    "MRZ_VALIDITY_PERIOD_UNUSUAL": "identity",

    # identity verification (external registries / account numbers)
    "EXTERNAL_IBAN_INVALID": "identity",
    "EXTERNAL_IBAN_VERIFICATION_FAILED": "identity",

    # visual: QR mismatches, watermarks
    "VISUAL_QR_URL_SHORTENER": "visual",
    "VISUAL_QR_SUSPICIOUS_TLD": "visual",
    "VISUAL_QR_DOMAIN_MISMATCH": "visual",
    "VISUAL_CONVERTER_WATERMARK": "visual",

    # images: screenshots, image-only PDFs, compression issues
    "IMAGES_SCREENSHOT_DETECTED": "images",
    "IMAGES_RESOLUTION_MISMATCH": "images",
    "IMAGES_HEAVY_COMPRESSION": "images",
    "IMAGES_EXCESSIVE_COUNT": "images",
    "IMAGES_NO_IMAGES": "images",
    "IMAGES_IMAGE_ONLY_PDF": "images",
    "IMAGES_MOSTLY_IMAGE_PDF": "images",
    "IMAGES_EXIF_DATE_AFTER_DOC": "images",
    "IMAGES_EXIF_EDITOR_SOFTWARE": "images",

    # security: JavaScript, embedded files, hidden annotations
    "STRUCT_JAVASCRIPT_DETECTED": "security",
    "STRUCT_HIDDEN_ANNOTATIONS": "security",
    "STRUCT_SUSPICIOUS_ANNOTATIONS": "security",
    "STRUCT_EMBEDDED_FILES": "security",
    "STRUCT_ACROFORM_DETECTED": "security",
    "STRUCT_XMP_EDITOR_MISMATCH": "tampering",

    # fonts (standalone, not tampering-level)
    "FONTS_SYSTEM_FONTS": "fonts",
    "FONTS_NOT_EMBEDDED": "fonts",

    # signature — positive codes (TRUSTED*, UNVERIFIED) are low severity and
    # filtered out of bullets by the low-severity skip; negative ones
    # (NOT_TRUSTED, UNVERIFIABLE, INVALID) surface as bullets like any finding.
    "STRUCT_SIGNATURE_TRUSTED": "signature",
    "STRUCT_SIGNATURE_TRUSTED_EXPIRED": "signature",
    "STRUCT_SIGNATURE_UNVERIFIED": "signature",
    "STRUCT_SIGNATURE_NOT_TRUSTED": "signature",
    "STRUCT_SIGNATURE_UNVERIFIABLE": "signature",
    "STRUCT_SIGNATURE_INVALID": "signature",

    # other (pipeline-status codes — not a finding about the document itself)
    "EXTRACTION_FAILED": "other",
    "MODULE_ERROR": "other",
}


# =============================================================================
# SENTENCE TEMPLATES
# =============================================================================

# Each flag code maps to a function that takes a Flag and returns a sentence.
# Using lambdas for simple cases, regular functions for complex ones.
# If a flag code is missing here, we fall back to flag.message.

def _get_converter_name(flag: Flag) -> str:
    """Extract the converter/tool name from flag details."""
    if flag.details:
        return flag.details.get("detected_tool", flag.details.get("producer", "an unknown tool"))
    return "an unknown tool"


SENTENCE_TEMPLATES: dict[str, callable] = {
    # --- origin ---
    "META_AI_GENERATED": lambda f: (
        f"The document was generated by {_get_converter_name(f)}, an AI tool."
    ),
    "META_ONLINE_CONVERTER": lambda f: (
        f"The document was processed through {_get_converter_name(f)}, "
        f"an online converter commonly used to modify documents."
    ),
    "META_SUSPICIOUS_PRODUCER": lambda f: (
        f"The document was created with {_get_converter_name(f)}, "
        f"which is unusual for a professional invoice."
    ),
    "META_NO_METADATA": lambda f: (
        "The document metadata has been completely stripped, "
        "which may indicate an attempt to hide its origin."
    ),
    "META_NO_PRODUCER": lambda f: (
        "No software information is present in the document metadata."
    ),
    "META_IMAGE_EDITOR": lambda f: (
        "The document was produced with an image editor "
        f"({f.details.get('producer') or f.details.get('creator') or 'unknown tool'}) "
        "rather than an office/invoicing tool."
        if f.details else
        "The document was produced with an image editor rather than an "
        "office/invoicing tool."),
    "META_IMAGE_CONVERSION": lambda f: (
        "The document was converted from an image, which erases the original "
        "text layer and its history."),

    # --- tampering ---
    "STRUCT_INCREMENTAL_UPDATES": lambda f: (
        f"The PDF has been edited {f.details.get('edit_count', 'multiple')} "
        f"time(s) after its initial creation."
        if f.details else "The PDF has been edited after its initial creation."
    ),
    "STRUCT_DELETED_OBJECTS": lambda f: (
        "The PDF contains deleted objects (ghost data), suggesting content "
        "was removed but traces remain."
    ),
    "STRUCT_HIDDEN_MODIFICATION": lambda f: (
        "The file contains a modification saved WITHOUT the standard "
        "incremental-update markers — an attempt to edit without leaving a "
        "trace."),
    "FONTS_EXCESSIVE_DIVERSITY": lambda f: (
        f"The document uses {f.details.get('font_count', 'many')} different fonts, "
        f"which is unusual for a professional invoice."
        if f.details else "The document uses an excessive number of fonts."
    ),
    "FONTS_HIGH_DIVERSITY": lambda f: (
        f"The document uses {f.details.get('font_count', 'several')} different fonts, "
        f"more than typical for a professional document."
        if f.details else "The document uses more fonts than typical."
    ),
    "FONTS_MIXED_SUBSETS": lambda f: (
        f"The font \"{f.details.get('font_name', 'unknown')}\" appears in both "
        f"subset and full forms, which can indicate text editing."
        if f.details else "A font appears in mixed forms, suggesting editing."
    ),
    "FONTS_MIDLINE_CHANGE": lambda f: (
        f"A font change was detected mid-line, suggesting text was edited after creation."
    ),
    "IMAGES_PASTE_NOISE_ANOMALY": lambda f: (
        "A region of the document shows abnormal noise patterns consistent "
        "with a copy-paste operation."
    ),
    "OVERLAY_COVER_AND_RETYPE": lambda f: (
        f"A value appears to have been covered and replaced: "
        f"\"{f.details.get('covered', '')}\" was hidden and "
        f"\"{f.details.get('replacement', '')}\" placed over it."
        if f.details and f.details.get('covered') is not None
        else "A value appears to have been covered by a box and retyped over it."
    ),
    "OVERLAY_COVER_AND_RETYPE_CONFIRMED": lambda f: (
        f"A value was covered and replaced: "
        f"\"{f.details.get('covered', '')}\" was hidden and "
        f"\"{f.details.get('replacement', '')}\" placed over it. This is "
        f"corroborated by the document's Producer/Creator, which shows it "
        f"was saved by an interactive editing tool."
        if f.details and f.details.get('covered') is not None
        else ("A value was covered by a box and retyped over it, corroborated "
              "by the document's Producer/Creator showing an interactive "
              "editing tool was used.")
    ),
    "OVERLAY_OPAQUE_FILL_OVER_TEXT": lambda f: (
        f"An opaque box covers still-present text "
        f"(\"{f.details.get('hidden', '')}\"), which remains recoverable underneath."
        if f.details and f.details.get('hidden')
        else "An opaque box covers text that is still recoverable underneath it."
    ),
    "OVERLAY_CONFLICTING_TEXT": lambda f: (
        "Two conflicting values overlap at the same position, "
        "suggesting the original value was retyped."
    ),
    "OVERLAY_REVISION_VALUE_CHANGED": lambda f: (
        f"A value changed between two saved revisions of this document: "
        f"\"{f.details.get('old_value', '')}\" became "
        f"\"{f.details.get('new_value', '')}\"."
        if f.details and f.details.get('old_value') is not None
        else "A value changed between two saved revisions of this document."
    ),
    "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED": lambda f: (
        f"A value changed between two saved revisions of this document: "
        f"\"{f.details.get('old_value', '')}\" became "
        f"\"{f.details.get('new_value', '')}\". This is corroborated by the "
        f"revision that made the change having been saved by an interactive "
        f"editing tool."
        if f.details and f.details.get('old_value') is not None
        else ("A value changed between two saved revisions of this document, "
              "corroborated by the changing revision having been saved by an "
              "interactive editing tool.")
    ),
    "OVERLAY_IMAGE_OVER_VALUE": lambda f: (
        f"An image is placed over a still-present value "
        f"(\"{f.details.get('covered', '')}\"), which remains recoverable "
        f"underneath it."
        if f.details and f.details.get('covered')
        else "An image is placed over a value that is still recoverable underneath it."
    ),
    "OVERLAY_IMAGE_OVER_TEXT": lambda f: (
        f"An image is placed over still-present text "
        f"(\"{f.details.get('covered', '')}\"), which remains recoverable "
        f"underneath it."
        if f.details and f.details.get('covered')
        else "An image is placed over text that is still recoverable underneath it."
    ),
    "ARITH_IBAN_MISMATCH": lambda f: (
        f"The bank account shown on the page (IBAN "
        f"{', '.join(f.details.get('visible_ibans', []))}) does not match the "
        f"payment account in the document's embedded e-invoice data "
        f"({', '.join(f.details.get('xml_ibans', []))}) — the signature of a "
        f"payment-redirection edit."
        if f.details else ("The visible bank account (IBAN) does not match the "
                           "embedded e-invoice payment data.")
    ),
    "ARITH_XML_INTERNAL_INCONSISTENT": lambda f: (
        f"The embedded e-invoice data does not add up: basis "
        f"{f.details.get('basis', 0):,.2f} + tax {f.details.get('tax', 0):,.2f} "
        f"does not equal the grand total {f.details.get('grand', 0):,.2f}."
        if f.details else "The embedded e-invoice totals do not add up."
    ),
    "ARITH_XML_VISIBLE_MISMATCH": lambda f: (
        f"The total printed on the page ({f.details.get('visible', 0):,.2f}) does "
        f"not match any total in the document's embedded e-invoice data. The "
        f"visible and machine-readable amounts disagree."
        if f.details else ("The printed total does not match the embedded "
                            "e-invoice data.")
    ),
    "TWOD_DOC_FIELD_MISMATCH": lambda f: (
        f"The signed 2D-Doc barcode says {f.details.get('field', 'a field')} "
        f"= \"{f.details.get('expected', '')}\", but the visible text shows "
        "something else — the printed value may have been edited."
        if f.details else
        "A field in the signed 2D-Doc barcode does not match the visible text."),
    "TWOD_DOC_CONSISTENT": lambda f: (
        "All fields of the signed 2D-Doc barcode match the visible text."),
    "FORENSICS_ELA_MAJOR_EDIT": lambda f: (
        "Error-level analysis shows a region whose compression history differs "
        "sharply from the rest of the page — consistent with a pasted or "
        "edited area."),
    "FORENSICS_ELA_MINOR_EDIT": lambda f: (
        "Error-level analysis shows a small region with a different "
        "compression history — possibly an edited area."),

    # --- dates ---
    "META_FUTURE_CREATION_DATE": lambda f: (
        "The PDF creation date is set in the future, which is impossible "
        "for a legitimate document."
    ),
    "META_DOCUMENT_MODIFIED": lambda f: (
        f"The document was modified "
        f"{int(f.details['time_difference_hours'] // 24)} day(s) after creation."
        if f.details and f.details.get('time_difference_hours', 0) >= 24
        else f"The document was modified "
        f"{round(f.details['time_difference_hours'], 1)}h after creation."
        if f.details and f.details.get('time_difference_hours', 0) >= 1
        else f"The document was modified "
        f"{int(f.details['time_difference_hours'] * 60)} minute(s) after creation."
        if f.details and f.details.get('time_difference_hours')
        else "The document was modified after its initial creation."
    ),
    "META_IMPOSSIBLE_DATES": lambda f: (
        "The modification date is before the creation date, "
        "which indicates metadata tampering."
    ),
    "CONTENT_FAR_FUTURE_DATE": lambda f: (
        f"A date far in the future was found"
        f" ({f.details.get('date', '')})"
        f"{' near: ' + f.details.get('context', '') if f.details and f.details.get('context') else ''}."
        if f.details else "A date far in the future was found in the document."
    ),
    "CONTENT_VERY_OLD_DATE": lambda f: (
        "A suspiciously old date (before 2000) was found in the document."
    ),
    "CONTENT_ANACHRONISM_SERVICE": lambda f: (
        "The service date comes after the invoice date, which is inconsistent."
    ),
    "CONTENT_ANACHRONISM_DUE": lambda f: (
        "The due date is before the invoice date, which is inconsistent."
    ),
    "CONTENT_ANACHRONISM_ORDER": lambda f: (
        "The order date comes after the invoice date, which is inconsistent."
    ),
    "CONTENT_FUTURE_DOCUMENT_DATE": lambda f: (
        "The document date is in the future."
    ),
    "CONTENT_DATE_CREATION_MISMATCH": lambda f: (
        "A date mentioned in the document contradicts the file's own creation "
        "timestamp."),
    "CONTENT_EXPIRED_DOCUMENT": lambda f: (
        "The document refers to a validity period that has already expired."),

    # --- identity ---
    "CONTENT_INVALID_SIRET": lambda f: (
        f"The SIRET number ({f.details.get('siret', 'unknown')}) has an invalid checksum."
        if f.details else "A SIRET number with invalid checksum was found."
    ),
    "CONTENT_INVALID_SIREN": lambda f: (
        f"The SIREN number ({f.details.get('siren', 'unknown')}) has an invalid checksum."
        if f.details else "A SIREN number with invalid checksum was found."
    ),
    "CONTENT_INVALID_VAT": lambda f: (
        f"The VAT number ({f.details.get('vat', 'unknown')}) is invalid."
        if f.details else "An invalid VAT number was found."
    ),
    "CONTENT_SIREN_VAT_MISMATCH": lambda f: (
        "The SIREN number derived from the VAT number does not match "
        "the SIREN/SIRET found in the document."
    ),
    "CONTENT_MISSING_COMPANY_ID": lambda f: (
        "No SIRET, SIREN, or RCS number was found, which is required "
        "on French invoices."
    ),
    "CONTENT_INCONSISTENT_REFERENCES": lambda f: (
        f"The document contains {f.details.get('unique_references', 'multiple')} "
        f"different reference numbers, which is suspicious."
        if f.details else "Inconsistent reference numbers were found."
    ),
    "CONTENT_REFERENCE_DATE_MISMATCH": lambda f: (
        "The invoice reference number does not match the invoice date."
    ),
    "EXTERNAL_SIRET_NOT_FOUND": lambda f: (
        f"The SIRET number ({f.details.get('siret', 'unknown')}) was not found "
        f"in the official French registry."
        if f.details else "A SIRET number was not found in the official registry."
    ),
    "EXTERNAL_SIREN_NOT_FOUND": lambda f: (
        f"The SIREN number ({f.details.get('siren', 'unknown')}) was not found "
        f"in the official French registry."
        if f.details else "A SIREN number was not found in the official registry."
    ),
    "EXTERNAL_COMPANY_CLOSED": lambda f: (
        f"The company \"{f.details.get('company_name', 'unknown')}\" is registered "
        f"as closed/inactive."
        if f.details else "The company is registered as closed/inactive."
    ),
    "EXTERNAL_COMPANY_NAME_MISMATCH": lambda f: (
        f"The company name in the document does not match the official registry "
        f"(expected \"{f.details.get('name_in_registry', 'unknown')}\")."
        if f.details else "The company name does not match the official registry."
    ),
    "EXTERNAL_VAT_INVALID": lambda f: (
        f"The VAT number ({f.details.get('vat', 'unknown')}) is invalid "
        f"according to the EU VIES system."
        if f.details else "A VAT number is invalid according to EU VIES."
    ),
    "EXTERNAL_IBAN_INVALID": lambda f: (
        "A bank account number (IBAN) printed on the document fails its "
        "checksum — it cannot be a real account."),
    "EXTERNAL_IBAN_VERIFICATION_FAILED": lambda f: (
        "The bank account number (IBAN) could not be verified."),

    # --- identity (MRZ family — identity.py messages are hardcoded French;
    # these EN templates are what English users see) ---
    "MRZ_DOC_NUMBER_CHECK_FAIL": lambda f: (
        "The ID document number fails its MRZ check digit — genuine documents "
        "cannot fail this."),
    "MRZ_DOB_CHECK_FAIL": lambda f: (
        "The date of birth fails its MRZ check digit."),
    "MRZ_DOB_IMPOSSIBLE": lambda f: (
        "The date of birth encoded in the MRZ is impossible."),
    "MRZ_EXPIRY_CHECK_FAIL": lambda f: (
        "The expiry date fails its MRZ check digit."),
    "MRZ_DOCUMENT_EXPIRED": lambda f: (
        "The identity document is expired."),
    "MRZ_OPTIONAL_CHECK_FAIL": lambda f: (
        "The MRZ optional-data field fails its check digit."),
    "MRZ_FINAL_CHECK_FAIL": lambda f: (
        "The MRZ global check digit fails — the machine-readable zone has "
        "been altered or mistyped."),
    "MRZ_INVALID_SEX": lambda f: (
        "The MRZ sex field holds an invalid value."),
    "MRZ_DOC_TYPE_UNEXPECTED": lambda f: (
        "The MRZ document-type code does not match this kind of document."),
    "MRZ_PASSPORT_TYPE_UNUSUAL": lambda f: (
        "The passport MRZ type code is unusual."),
    "MRZ_ISSUER_FORMAT_INVALID": lambda f: (
        "The MRZ issuing-country code is malformed."),
    "MRZ_NAME_FORMAT_INVALID": lambda f: (
        "The MRZ name field is malformed."),
    "MRZ_NATIONALITY_FORMAT_INVALID": lambda f: (
        "The MRZ nationality code is malformed."),
    "MRZ_DOC_NUMBER_FORMAT_INVALID": lambda f: (
        "The MRZ document number is malformed."),
    "MRZ_CNI_TYPE_UNEXPECTED": lambda f: (
        "The French ID card MRZ type code is unexpected."),
    "MRZ_CNI_OPTIONAL2_NOT_EMPTY": lambda f: (
        "A French ID card MRZ field that must be empty is not."),
    "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH": lambda f: (
        "The US passport-card issuer field contradicts the document type."),
    "MRZ_US_VALIDITY_PERIOD_UNEXPECTED": lambda f: (
        "The US document's validity period is not the standard one."),
    "MRZ_VALIDITY_PERIOD_UNUSUAL": lambda f: (
        "The document's validity period is unusual for its type."),

    # --- visual ---
    "VISUAL_QR_URL_SHORTENER": lambda f: (
        f"A QR code uses a URL shortener ({f.details.get('domain', 'unknown')}), "
        f"which can hide the real destination."
        if f.details else "A QR code uses a URL shortener."
    ),
    "VISUAL_QR_SUSPICIOUS_TLD": lambda f: (
        f"A QR code points to a suspicious domain "
        f"({f.details.get('domain', 'unknown')})."
        if f.details else "A QR code points to a suspicious domain."
    ),
    "VISUAL_QR_DOMAIN_MISMATCH": lambda f: (
        f"A QR code points to {f.details.get('qr_domain', 'an unexpected domain')}, "
        f"which doesn't match the expected domain(s)."
        if f.details else "A QR code domain doesn't match the expected domain."
    ),
    "VISUAL_CONVERTER_WATERMARK": lambda f: (
        f"A visible watermark from a converter/editor was detected "
        f"(\"{f.details.get('matched_text', '')}\")."
        if f.details and f.details.get('matched_text')
        else "A visible converter/editor watermark was detected."
    ),

    # images
    "IMAGES_SCREENSHOT_DETECTED": lambda f: (
        "An image with screen-like dimensions was detected, "
        "suggesting the document may be a screenshot."
    ),
    "IMAGES_RESOLUTION_MISMATCH": lambda f: (
        "Images in the document have inconsistent resolutions, "
        "which may indicate images from different sources."
    ),
    "IMAGES_HEAVY_COMPRESSION": lambda f: (
        "An image is heavily compressed, which can indicate multiple re-saves."
    ),
    "IMAGES_IMAGE_ONLY_PDF": lambda f: (
        "The document is image-only with no text layer, "
        "which is typical of screenshots or scanned modifications."
    ),
    "IMAGES_MOSTLY_IMAGE_PDF": lambda f: (
        "The document is mostly images with very little text."
    ),
    "IMAGES_NO_IMAGES": lambda f: (
        "The document contains no images (no logo or letterhead), which is "
        "unusual for a professional invoice."),
    "IMAGES_EXIF_DATE_AFTER_DOC": lambda f: (
        "An embedded image carries an EXIF timestamp LATER than the document's "
        "creation date — it was made after the document supposedly existed."),
    "IMAGES_EXIF_EDITOR_SOFTWARE": lambda f: (
        "An embedded image was saved by an image editor (per its EXIF data). "
        "Common for logos; meaningful only alongside other findings."),

    # security
    "STRUCT_JAVASCRIPT_DETECTED": lambda f: (
        "The PDF contains JavaScript code, which is suspicious "
        "and potentially dangerous in an invoice."
    ),
    "STRUCT_EMBEDDED_FILES": lambda f: (
        f"The PDF contains {f.details.get('file_count', '')} embedded file(s), "
        f"which could be malicious."
        if f.details else "The PDF contains embedded files."
    ),
    "STRUCT_HIDDEN_ANNOTATIONS": lambda f: (
        "Hidden annotations were found in the PDF."
    ),
    "STRUCT_SUSPICIOUS_ANNOTATIONS": lambda f: (
        "Suspicious annotation types were found in the PDF "
        "(file attachments, multimedia, etc.)."
    ),
    "STRUCT_XMP_EDITOR_MISMATCH": lambda f: (
        "The document was modified by a different tool than its creator."
    ),
    # --- signature (negative findings only; positive codes never bullet) ---
    "STRUCT_SIGNATURE_INVALID": lambda f: (
        "The document's digital signature failed cryptographic verification — "
        "the content was likely modified after signing."
    ),
    "STRUCT_SIGNATURE_NOT_TRUSTED": lambda f: (
        "The document is signed, but the certificate is not on the EU trusted list."
    ),
    "STRUCT_SIGNATURE_UNVERIFIABLE": lambda f: (
        "The document has a signature in a format that could not be verified."
    ),

    # --- other (pipeline-status codes) ---
    "EXTRACTION_FAILED": lambda f: (
        "The file could not be read as a valid PDF, so it cannot be verified."),
    "MODULE_ERROR": lambda f: (
        f"One analysis ({f.details.get('module', '?')}) failed on this "
        "document and was skipped."
        if f.details else "One analysis failed on this document and was skipped."),
}


SENTENCE_TEMPLATES_FR: dict[str, callable] = {
    # --- origin ---
    "META_AI_GENERATED": lambda f: (
        f"Le document a été généré par {_get_converter_name(f)}, un outil IA."
    ),
    "META_ONLINE_CONVERTER": lambda f: (
        f"Le document a été traité via {_get_converter_name(f)}, "
        f"un convertisseur en ligne couramment utilisé pour modifier des documents."
    ),
    "META_SUSPICIOUS_PRODUCER": lambda f: (
        f"Le document a été créé avec {_get_converter_name(f)}, "
        f"ce qui est inhabituel pour une facture professionnelle."
    ),
    "META_NO_METADATA": lambda f: (
        "Les métadonnées du document ont été entièrement supprimées, "
        "ce qui peut indiquer une tentative de masquer son origine."
    ),
    "META_NO_PRODUCER": lambda f: (
        "Aucune information logicielle n'est présente dans les métadonnées du document."
    ),
    "META_IMAGE_EDITOR": lambda f: (
        "Le document a été produit avec un éditeur d'images "
        f"({f.details.get('producer') or f.details.get('creator') or 'outil inconnu'}) "
        "plutôt qu'un outil bureautique ou de facturation."
        if f.details else
        "Le document a été produit avec un éditeur d'images plutôt qu'un "
        "outil bureautique ou de facturation."),
    "META_IMAGE_CONVERSION": lambda f: (
        "Le document a été converti depuis une image, ce qui efface la couche "
        "texte d'origine et son historique."),

    # --- tampering ---
    "STRUCT_INCREMENTAL_UPDATES": lambda f: (
        f"Le PDF a été modifié {f.details.get('edit_count', 'plusieurs')} "
        f"fois après sa création initiale."
        if f.details else "Le PDF a été modifié après sa création initiale."
    ),
    "STRUCT_DELETED_OBJECTS": lambda f: (
        "Le PDF contient des objets supprimés (données fantômes), "
        "suggérant que du contenu a été retiré mais des traces subsistent."
    ),
    "STRUCT_HIDDEN_MODIFICATION": lambda f: (
        "Le fichier contient une modification enregistrée SANS les marqueurs "
        "standard de mise à jour incrémentale — une tentative d'édition sans "
        "laisser de trace."),
    "FONTS_EXCESSIVE_DIVERSITY": lambda f: (
        f"Le document utilise {f.details.get('font_count', 'beaucoup de')} polices différentes, "
        f"ce qui est inhabituel pour une facture professionnelle."
        if f.details else "Le document utilise un nombre excessif de polices."
    ),
    "FONTS_HIGH_DIVERSITY": lambda f: (
        f"Le document utilise {f.details.get('font_count', 'plusieurs')} polices différentes, "
        f"plus que la normale pour un document professionnel."
        if f.details else "Le document utilise plus de polices que la normale."
    ),
    "FONTS_MIXED_SUBSETS": lambda f: (
        f"La police \"{f.details.get('font_name', 'inconnue')}\" apparaît en formes "
        f"sous-ensemble et complète, ce qui peut indiquer une modification de texte."
        if f.details else "Une police apparaît sous formes mixtes, suggérant une modification."
    ),
    "FONTS_MIDLINE_CHANGE": lambda f: (
        "Un changement de police a été détecté en milieu de ligne, "
        "suggérant que le texte a été modifié après sa création."
    ),
    "IMAGES_PASTE_NOISE_ANOMALY": lambda f: (
        "Une zone du document présente des patterns de bruit anormaux "
        "cohérents avec une opération de copier-coller."
    ),
    "OVERLAY_COVER_AND_RETYPE": lambda f: (
        f"Une valeur semble avoir été masquée puis remplacée : "
        f"« {f.details.get('covered', '')} » a été cachée et "
        f"« {f.details.get('replacement', '')} » placée par-dessus."
        if f.details and f.details.get('covered') is not None
        else "Une valeur semble avoir été masquée par un rectangle puis ressaisie par-dessus."
    ),
    "OVERLAY_COVER_AND_RETYPE_CONFIRMED": lambda f: (
        f"Une valeur a été masquée puis remplacée : "
        f"« {f.details.get('covered', '')} » a été cachée et "
        f"« {f.details.get('replacement', '')} » placée par-dessus. Ceci est "
        f"corroboré par les métadonnées Producer/Creator du document, qui "
        f"indiquent qu'il a été enregistré par un outil d'édition interactif."
        if f.details and f.details.get('covered') is not None
        else ("Une valeur a été masquée par un rectangle puis ressaisie "
              "par-dessus, corroboré par les métadonnées Producer/Creator du "
              "document indiquant l'usage d'un outil d'édition interactif.")
    ),
    "OVERLAY_OPAQUE_FILL_OVER_TEXT": lambda f: (
        f"Un rectangle opaque recouvre du texte toujours présent "
        f"(« {f.details.get('hidden', '')} »), qui reste récupérable en dessous."
        if f.details and f.details.get('hidden')
        else "Un rectangle opaque recouvre du texte encore récupérable en dessous."
    ),
    "OVERLAY_CONFLICTING_TEXT": lambda f: (
        "Deux valeurs contradictoires se superposent au même endroit, "
        "suggérant que la valeur d'origine a été ressaisie."
    ),
    "OVERLAY_REVISION_VALUE_CHANGED": lambda f: (
        f"Une valeur a changé entre deux versions enregistrées de ce document : "
        f"« {f.details.get('old_value', '')} » est devenue "
        f"« {f.details.get('new_value', '')} »."
        if f.details and f.details.get('old_value') is not None
        else "Une valeur a changé entre deux versions enregistrées de ce document."
    ),
    "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED": lambda f: (
        f"Une valeur a changé entre deux versions enregistrées de ce document : "
        f"« {f.details.get('old_value', '')} » est devenue "
        f"« {f.details.get('new_value', '')} ». Ceci est corroboré par le fait "
        f"que la version ayant introduit ce changement a été enregistrée par "
        f"un outil d'édition interactif."
        if f.details and f.details.get('old_value') is not None
        else ("Une valeur a changé entre deux versions enregistrées de ce "
              "document, corroboré par le fait que la version ayant introduit "
              "ce changement a été enregistrée par un outil d'édition "
              "interactif.")
    ),
    "OVERLAY_IMAGE_OVER_VALUE": lambda f: (
        f"Une image est placée sur une valeur toujours présente "
        f"(« {f.details.get('covered', '')} »), qui reste récupérable en dessous."
        if f.details and f.details.get('covered')
        else "Une image est placée sur une valeur encore récupérable en dessous."
    ),
    "OVERLAY_IMAGE_OVER_TEXT": lambda f: (
        f"Une image est placée sur du texte toujours présent "
        f"(« {f.details.get('covered', '')} »), qui reste récupérable en dessous."
        if f.details and f.details.get('covered')
        else "Une image est placée sur du texte encore récupérable en dessous."
    ),
    "ARITH_IBAN_MISMATCH": lambda f: (
        f"Le compte bancaire indiqué sur la page (IBAN "
        f"{', '.join(f.details.get('visible_ibans', []))}) ne correspond pas au "
        f"compte de paiement des données de facture électronique intégrées "
        f"({', '.join(f.details.get('xml_ibans', []))}) — signature d'une "
        f"redirection de paiement."
        if f.details else ("Le compte bancaire (IBAN) visible ne correspond pas "
                           "aux données de paiement de la facture électronique intégrée.")
    ),
    "ARITH_XML_INTERNAL_INCONSISTENT": lambda f: (
        f"Les données de la facture électronique intégrée ne sont pas cohérentes : "
        f"base {f.details.get('basis', 0):,.2f} + taxe {f.details.get('tax', 0):,.2f} "
        f"ne correspond pas au total {f.details.get('grand', 0):,.2f}."
        if f.details else "Les totaux de la facture électronique intégrée ne sont pas cohérents."
    ),
    "ARITH_XML_VISIBLE_MISMATCH": lambda f: (
        f"Le total imprimé sur la page ({f.details.get('visible', 0):,.2f}) ne "
        f"correspond à aucun total des données de facture électronique intégrées. "
        f"Les montants visibles et lisibles par machine divergent."
        if f.details else ("Le total imprimé ne correspond pas aux données de "
                           "facture électronique intégrées.")
    ),
    "TWOD_DOC_FIELD_MISMATCH": lambda f: (
        f"Le code 2D-Doc signé indique {f.details.get('field', 'un champ')} "
        f"= « {f.details.get('expected', '')} », mais le texte visible montre "
        "autre chose — la valeur imprimée a pu être modifiée."
        if f.details else
        "Un champ du code 2D-Doc signé ne correspond pas au texte visible."),
    "TWOD_DOC_CONSISTENT": lambda f: (
        "Tous les champs du code 2D-Doc signé correspondent au texte visible."),
    "FORENSICS_ELA_MAJOR_EDIT": lambda f: (
        "L'analyse des niveaux d'erreur révèle une zone dont l'historique de "
        "compression diffère nettement du reste de la page — compatible avec "
        "un collage ou une retouche."),
    "FORENSICS_ELA_MINOR_EDIT": lambda f: (
        "L'analyse des niveaux d'erreur révèle une petite zone à l'historique "
        "de compression différent — possible retouche."),

    # --- dates ---
    "META_FUTURE_CREATION_DATE": lambda f: (
        "La date de création du PDF est définie dans le futur, "
        "ce qui est impossible pour un document légitime."
    ),
    "META_DOCUMENT_MODIFIED": lambda f: (
        f"Le document a été modifié "
        f"{int(f.details['time_difference_hours'] // 24)} jour(s) après sa création."
        if f.details and f.details.get('time_difference_hours', 0) >= 24
        else f"Le document a été modifié "
        f"{round(f.details['time_difference_hours'], 1)}h après sa création."
        if f.details and f.details.get('time_difference_hours', 0) >= 1
        else f"Le document a été modifié "
        f"{int(f.details['time_difference_hours'] * 60)} minute(s) après sa création."
        if f.details and f.details.get('time_difference_hours')
        else "Le document a été modifié après sa création initiale."
    ),
    "META_IMPOSSIBLE_DATES": lambda f: (
        "La date de modification est antérieure à la date de création, "
        "ce qui indique une anomalie dans les métadonnées."
    ),
    "CONTENT_FAR_FUTURE_DATE": lambda f: (
        f"Une date lointaine dans le futur a été trouvée"
        f" ({f.details.get('date', '')})"
        f"{' près de : ' + f.details.get('context', '') if f.details and f.details.get('context') else ''}."
        if f.details else "Une date lointaine dans le futur a été trouvée dans le document."
    ),
    "CONTENT_VERY_OLD_DATE": lambda f: (
        "Une date suspicieusement ancienne (avant 2000) a été trouvée dans le document."
    ),
    "CONTENT_ANACHRONISM_SERVICE": lambda f: (
        "La date de prestation est postérieure à la date de facture, ce qui est incohérent."
    ),
    "CONTENT_ANACHRONISM_DUE": lambda f: (
        "La date d'échéance est antérieure à la date de facture, ce qui est incohérent."
    ),
    "CONTENT_ANACHRONISM_ORDER": lambda f: (
        "La date de commande est postérieure à la date de facture, ce qui est incohérent."
    ),
    "CONTENT_FUTURE_DOCUMENT_DATE": lambda f: (
        "La date du document est dans le futur."
    ),
    "CONTENT_DATE_CREATION_MISMATCH": lambda f: (
        "Une date mentionnée dans le document contredit la date de création "
        "du fichier."),
    "CONTENT_EXPIRED_DOCUMENT": lambda f: (
        "Le document fait référence à une période de validité déjà expirée."),

    # --- identity ---
    "CONTENT_INVALID_SIRET": lambda f: (
        f"Le numéro SIRET ({f.details.get('siret', 'inconnu')}) a un checksum invalide."
        if f.details else "Un numéro SIRET avec un checksum invalide a été trouvé."
    ),
    "CONTENT_INVALID_SIREN": lambda f: (
        f"Le numéro SIREN ({f.details.get('siren', 'inconnu')}) a un checksum invalide."
        if f.details else "Un numéro SIREN avec un checksum invalide a été trouvé."
    ),
    "CONTENT_INVALID_VAT": lambda f: (
        f"Le numéro de TVA ({f.details.get('vat', 'inconnu')}) est invalide."
        if f.details else "Un numéro de TVA invalide a été trouvé."
    ),
    "CONTENT_SIREN_VAT_MISMATCH": lambda f: (
        "Le numéro SIREN dérivé du numéro de TVA ne correspond pas "
        "au SIREN/SIRET trouvé dans le document."
    ),
    "CONTENT_MISSING_COMPANY_ID": lambda f: (
        "Aucun numéro SIRET, SIREN ou RCS n'a été trouvé, "
        "ce qui est obligatoire sur les factures françaises."
    ),
    "CONTENT_INCONSISTENT_REFERENCES": lambda f: (
        f"Le document contient {f.details.get('unique_references', 'plusieurs')} "
        f"numéros de référence différents, ce qui est suspect."
        if f.details else "Des numéros de référence incohérents ont été trouvés."
    ),
    "CONTENT_REFERENCE_DATE_MISMATCH": lambda f: (
        "Le numéro de référence de la facture ne correspond pas à la date de facture."
    ),
    "EXTERNAL_SIRET_NOT_FOUND": lambda f: (
        f"Le numéro SIRET ({f.details.get('siret', 'inconnu')}) n'a pas été trouvé "
        f"dans le registre officiel français."
        if f.details else "Un numéro SIRET n'a pas été trouvé dans le registre officiel."
    ),
    "EXTERNAL_SIREN_NOT_FOUND": lambda f: (
        f"Le numéro SIREN ({f.details.get('siren', 'inconnu')}) n'a pas été trouvé "
        f"dans le registre officiel français."
        if f.details else "Un numéro SIREN n'a pas été trouvé dans le registre officiel."
    ),
    "EXTERNAL_COMPANY_CLOSED": lambda f: (
        f"L'entreprise \"{f.details.get('company_name', 'inconnue')}\" est enregistrée "
        f"comme fermée/inactive."
        if f.details else "L'entreprise est enregistrée comme fermée/inactive."
    ),
    "EXTERNAL_COMPANY_NAME_MISMATCH": lambda f: (
        f"Le nom de l'entreprise dans le document ne correspond pas au registre officiel "
        f"(attendu : \"{f.details.get('name_in_registry', 'inconnu')}\")."
        if f.details else "Le nom de l'entreprise ne correspond pas au registre officiel."
    ),
    "EXTERNAL_VAT_INVALID": lambda f: (
        f"Le numéro de TVA ({f.details.get('vat', 'inconnu')}) est invalide "
        f"selon le système EU VIES."
        if f.details else "Un numéro de TVA est invalide selon EU VIES."
    ),
    "EXTERNAL_IBAN_INVALID": lambda f: (
        "Un IBAN imprimé sur le document échoue à sa somme de contrôle — ce "
        "ne peut pas être un compte réel."),
    "EXTERNAL_IBAN_VERIFICATION_FAILED": lambda f: (
        "L'IBAN n'a pas pu être vérifié."),

    # --- identity (famille MRZ — formulations dédiées pour la parité EN/FR ;
    # le texte anglais correspondant existe car identity.py ne construit ses
    # messages qu'en français) ---
    "MRZ_DOC_NUMBER_CHECK_FAIL": lambda f: (
        "Le numéro de document échoue à son chiffre de contrôle MRZ — un "
        "document authentique ne peut pas échouer à ce test."),
    "MRZ_DOB_CHECK_FAIL": lambda f: (
        "La date de naissance échoue à son chiffre de contrôle MRZ."),
    "MRZ_DOB_IMPOSSIBLE": lambda f: (
        "La date de naissance encodée dans la MRZ est impossible."),
    "MRZ_EXPIRY_CHECK_FAIL": lambda f: (
        "La date d'expiration échoue à son chiffre de contrôle MRZ."),
    "MRZ_DOCUMENT_EXPIRED": lambda f: (
        "La pièce d'identité est expirée."),
    "MRZ_OPTIONAL_CHECK_FAIL": lambda f: (
        "Le champ optionnel de la MRZ échoue à son chiffre de contrôle."),
    "MRZ_FINAL_CHECK_FAIL": lambda f: (
        "Le chiffre de contrôle global de la MRZ échoue — la zone de lecture "
        "automatique a été altérée ou mal saisie."),
    "MRZ_INVALID_SEX": lambda f: (
        "Le champ sexe de la MRZ contient une valeur invalide."),
    "MRZ_DOC_TYPE_UNEXPECTED": lambda f: (
        "Le code de type de document MRZ ne correspond pas à ce type de pièce."),
    "MRZ_PASSPORT_TYPE_UNUSUAL": lambda f: (
        "Le code de type MRZ du passeport est inhabituel."),
    "MRZ_ISSUER_FORMAT_INVALID": lambda f: (
        "Le code pays émetteur de la MRZ est mal formé."),
    "MRZ_NAME_FORMAT_INVALID": lambda f: (
        "Le champ nom de la MRZ est mal formé."),
    "MRZ_NATIONALITY_FORMAT_INVALID": lambda f: (
        "Le code nationalité de la MRZ est mal formé."),
    "MRZ_DOC_NUMBER_FORMAT_INVALID": lambda f: (
        "Le numéro de document de la MRZ est mal formé."),
    "MRZ_CNI_TYPE_UNEXPECTED": lambda f: (
        "Le code de type MRZ de la carte d'identité française est inattendu."),
    "MRZ_CNI_OPTIONAL2_NOT_EMPTY": lambda f: (
        "Un champ MRZ de la carte d'identité française qui doit rester vide "
        "ne l'est pas."),
    "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH": lambda f: (
        "Le champ émetteur de la passport card américaine contredit le type "
        "de document."),
    "MRZ_US_VALIDITY_PERIOD_UNEXPECTED": lambda f: (
        "La période de validité du document américain n'est pas la période "
        "standard."),
    "MRZ_VALIDITY_PERIOD_UNUSUAL": lambda f: (
        "La période de validité du document est inhabituelle pour son type."),

    # --- visual ---
    "VISUAL_QR_URL_SHORTENER": lambda f: (
        f"Un QR code utilise un raccourcisseur d'URL ({f.details.get('domain', 'inconnu')}), "
        f"ce qui peut masquer la vraie destination."
        if f.details else "Un QR code utilise un raccourcisseur d'URL."
    ),
    "VISUAL_QR_SUSPICIOUS_TLD": lambda f: (
        f"Un QR code pointe vers un domaine suspect "
        f"({f.details.get('domain', 'inconnu')})."
        if f.details else "Un QR code pointe vers un domaine suspect."
    ),
    "VISUAL_QR_DOMAIN_MISMATCH": lambda f: (
        f"Un QR code pointe vers {f.details.get('qr_domain', 'un domaine inattendu')}, "
        f"qui ne correspond pas aux domaines attendus."
        if f.details else "Un domaine de QR code ne correspond pas au domaine attendu."
    ),
    "VISUAL_CONVERTER_WATERMARK": lambda f: (
        f"Un filigrane de convertisseur/éditeur a été détecté "
        f"(\"{f.details.get('matched_text', '')}\")."
        if f.details and f.details.get('matched_text')
        else "Un filigrane de convertisseur/éditeur a été détecté."
    ),

    # --- images ---
    "IMAGES_SCREENSHOT_DETECTED": lambda f: (
        "Une image aux dimensions typiques d'un écran a été détectée, "
        "suggérant que le document est peut-être une capture d'écran."
    ),
    "IMAGES_RESOLUTION_MISMATCH": lambda f: (
        "Les images du document ont des résolutions incohérentes, "
        "ce qui peut indiquer des images provenant de sources différentes."
    ),
    "IMAGES_HEAVY_COMPRESSION": lambda f: (
        "Une image est fortement compressée, ce qui peut indiquer de multiples re-sauvegardes."
    ),
    "IMAGES_IMAGE_ONLY_PDF": lambda f: (
        "Le document est composé uniquement d'images sans couche de texte, "
        "ce qui est typique des captures d'écran ou des modifications scannées."
    ),
    "IMAGES_MOSTLY_IMAGE_PDF": lambda f: (
        "Le document est principalement composé d'images avec très peu de texte."
    ),
    "IMAGES_NO_IMAGES": lambda f: (
        "Le document ne contient aucune image (pas de logo ni d'en-tête) — "
        "inhabituel pour une facture professionnelle."
    ),
    "IMAGES_EXIF_DATE_AFTER_DOC": lambda f: (
        "Une image intégrée porte une date EXIF POSTÉRIEURE à la date de "
        "création du document — elle a été créée après le document lui-même."),
    "IMAGES_EXIF_EDITOR_SOFTWARE": lambda f: (
        "Une image intégrée a été enregistrée par un éditeur d'images (selon "
        "ses données EXIF). Courant pour un logo ; significatif seulement "
        "avec d'autres indices."),

    # --- security ---
    "STRUCT_JAVASCRIPT_DETECTED": lambda f: (
        "Le PDF contient du code JavaScript, ce qui est suspect "
        "et potentiellement dangereux dans une facture."
    ),
    "STRUCT_EMBEDDED_FILES": lambda f: (
        f"Le PDF contient {f.details.get('file_count', '')} fichier(s) embarqué(s), "
        f"qui pourraient être malveillants."
        if f.details else "Le PDF contient des fichiers embarqués."
    ),
    "STRUCT_HIDDEN_ANNOTATIONS": lambda f: (
        "Des annotations cachées ont été trouvées dans le PDF."
    ),
    "STRUCT_SUSPICIOUS_ANNOTATIONS": lambda f: (
        "Des types d'annotations suspects ont été trouvés dans le PDF "
        "(pièces jointes, multimédia, etc.)."
    ),
    "STRUCT_XMP_EDITOR_MISMATCH": lambda f: (
        "Le document a été modifié par un outil différent de son créateur."
    ),
    # --- signature (constats négatifs uniquement ; les codes positifs ne
    # produisent jamais de puce) ---
    "STRUCT_SIGNATURE_INVALID": lambda f: (
        "La signature numérique du document a échoué à la vérification "
        "cryptographique — le contenu a probablement été modifié après signature."
    ),
    "STRUCT_SIGNATURE_NOT_TRUSTED": lambda f: (
        "Le document est signé, mais le certificat ne figure pas sur la liste "
        "de confiance de l'UE."
    ),
    "STRUCT_SIGNATURE_UNVERIFIABLE": lambda f: (
        "Le document porte une signature dans un format qui n'a pas pu être vérifié."
    ),

    # --- other (codes de statut du pipeline) ---
    "EXTRACTION_FAILED": lambda f: (
        "Le fichier n'a pas pu être lu comme un PDF valide et ne peut donc "
        "pas être vérifié."),
    "MODULE_ERROR": lambda f: (
        f"Une analyse ({f.details.get('module', '?')}) a échoué sur ce "
        "document et a été ignorée."
        if f.details else "Une analyse a échoué sur ce document et a été ignorée."),
}


# =============================================================================
# VERDICT TEMPLATES
# =============================================================================

# Verdicts are selected based on risk level.
# When tampering or dates are the dominant theme, we pick a more specific verdict.
VERDICTS: dict[str, dict[str, str]] = {
    "CRITICAL": {
        "default": "Do not trust this document.",
        "tampering": "This document shows clear signs of modification.",
        "dates": "Dates in this document indicate modifications after its creation.",
        "identity": "Do not trust this document — identity information is fraudulent.",
        "security": "Do not trust this document — it contains dangerous content.",
    },
    "HIGH": {
        "default": "This document is likely fraudulent.",
        "tampering": "This document has been altered after creation.",
        "dates": "This document contains suspicious date inconsistencies.",
        "identity": "The originator of this document could not be verified.",
        "origin": "This document was created with suspicious tools.",
    },
    "MEDIUM": {
        "default": "This document has issues that need verification.",
        "tampering": "This document may have been edited.",
        "dates": "Dates in this document are inconsistent.",
        "origin": "The origin of this document raises questions.",
    },
    "LOW": {
        "default": "This document appears legitimate. No issues detected.",
    },
    "UNVERIFIABLE": {
        "default": "This document could not be read or verified. Treat it with caution and request a valid original.",
    },
}

VERDICTS_FR: dict[str, dict[str, str]] = {
    "CRITICAL": {
        "default": "Ne faites pas confiance à ce document.",
        "tampering": "Ce document présente des signes évidents de modification.",
        "dates": "Les dates de ce document indiquent une ou des modifications ultérieures à la date de création.",
        "identity": "Les informations d'identité de ce document sont frauduleuses.",
        "security": "Ce document contient du contenu dangereux.",
    },
    "HIGH": {
        "default": "Ce document est probablement frauduleux.",
        "tampering": "Ce document a été modifié après sa création.",
        "dates": "Ce document contient des incohérences de dates suspectes.",
        "identity": "L'émetteur de ce document n'a pas pu être vérifié.",
        "origin": "Ce document a été créé avec des outils suspects.",
    },
    "MEDIUM": {
        "default": "Ce document présente des anomalies qui nécessitent une vérification.",
        "tampering": "Ce document a peut-être été modifié.",
        "dates": "Les dates de ce document sont incohérentes.",
        "origin": "L'origine de ce document soulève des questions.",
    },
    "LOW": {
        "default": "Ce document semble légitime. Aucun problème détecté.",
    },
    "UNVERIFIABLE": {
        "default": "Ce document n'a pas pu être lu ou vérifié. Traitez-le avec prudence et demandez un original valide.",
    },
}


# =============================================================================
# POSITIVE SIGNAL TEMPLATES
# =============================================================================

# When a module has score >= 90 and no flags, we mention it as a positive signal.
# Maps module name -> positive sentence.
POSITIVE_SIGNALS: dict[str, str] = {
    "metadata": "The document metadata is consistent and clean.",
    "content": "All dates and references in the document are consistent.",
    "visual": "No suspicious visual elements were detected.",
    "fonts": "Font usage is consistent with a professional document.",
    "images": "Images show no signs of manipulation.",
    "structure": "The PDF structure is clean with no signs of editing.",
    "external": "Company identification numbers were verified successfully.",
}

# Special positive signal for trusted signatures (detected via flag code)
SIGNATURE_POSITIVE = "The document carries a valid EU-trusted digital signature."

POSITIVE_SIGNALS_FR: dict[str, str] = {
    "metadata": "Les métadonnées du document sont cohérentes et propres.",
    "content": "Toutes les dates et références du document sont cohérentes.",
    "visual": "Aucun élément visuel suspect n'a été détecté.",
    "fonts": "L'utilisation des polices est cohérente avec un document professionnel.",
    "images": "Les images ne présentent aucun signe de manipulation.",
    "structure": "La structure PDF est propre, sans signe de modification.",
    "external": "Les numéros d'identification de l'entreprise ont été vérifiés avec succès.",
}

SIGNATURE_POSITIVE_FR = "Le document porte une signature numérique valide reconnue par l'UE."


# =============================================================================
# CORE LOGIC
# =============================================================================

def _group_flags_by_theme(flags: list[Flag]) -> dict[str, list[Flag]]:
    """
    Group flags by user-facing theme.

    Args:
        flags: All flags from all modules, sorted by severity.

    Returns:
        Dict mapping theme name to list of flags in that theme.
        Flags with unknown codes go into "other".
    """
    groups: dict[str, list[Flag]] = {}

    for flag in flags:
        theme = FLAG_THEME_MAP.get(flag.code, "other")
        if theme not in groups:
            groups[theme] = []
        groups[theme].append(flag)

    return groups


def _get_dominant_theme(grouped: dict[str, list[Flag]]) -> str | None:
    """
    Find the theme with the most severe flags.

    We score each theme by summing severity weights of its flags.
    The theme with the highest total wins.

    Args:
        grouped: Flags grouped by theme.

    Returns:
        The dominant theme name, or None if no flags.
    """
    severity_weight = {"critical": 50, "high": 30, "medium": 15, "low": 5}

    # Exclude themes that aren't used for verdict selection
    verdict_themes = {"origin", "tampering", "dates", "identity", "security"}

    best_theme = None
    best_score = 0

    for theme, flags in grouped.items():
        if theme not in verdict_themes:
            continue
        total = sum(severity_weight.get(f.severity, 0) for f in flags)
        if total > best_score:
            best_score = total
            best_theme = theme

    return best_theme


def _build_verdict(risk_level: str, dominant_theme: str | None, lang: str = "en") -> str:
    """
    Pick the most appropriate verdict based on risk level and dominant theme.

    Args:
        risk_level: "LOW", "MEDIUM", "HIGH", "CRITICAL", or "UNVERIFIABLE"
        dominant_theme: The most prominent theme, or None.
        lang: Language code ("en" or "fr").

    Returns:
        A short verdict sentence.
    """
    verdicts = VERDICTS_FR if lang == "fr" else VERDICTS
    level_verdicts = verdicts.get(risk_level, verdicts["MEDIUM"])

    # Try theme-specific verdict first, fall back to default
    if dominant_theme and dominant_theme in level_verdicts:
        return level_verdicts[dominant_theme]
    return level_verdicts["default"]


def _flag_to_sentence(flag: Flag, lang: str = "en") -> str:
    """
    Convert a single flag into a human-readable sentence.

    Uses the template if available, otherwise falls back to flag.message.

    Args:
        flag: A Flag object.
        lang: Language code ("en" or "fr").

    Returns:
        A sentence describing the finding.
    """
    templates = SENTENCE_TEMPLATES_FR if lang == "fr" else SENTENCE_TEMPLATES
    template_fn = templates.get(flag.code)
    if template_fn:
        try:
            return template_fn(flag)
        except (KeyError, TypeError, AttributeError):
            # If the template fails (missing details key, etc.), fall back
            pass
    # Fallback: use the flag's own message
    return flag.message


def _build_bullets(
    grouped: dict[str, list[Flag]],
    module_results: list[ModuleResult],
    lang: str = "en",
) -> list[str]:
    """
    Build a concise list of issues found.

    Only shows problems — no positive signals. Keeps it short:
    one bullet per theme, only the most severe flag.

    Args:
        grouped: Flags grouped by theme.
        module_results: All module results (unused, kept for API compat).

    Returns:
        List of sentences, each one becoming a bullet point in the UI.
        Empty list if no issues (the verdict already says "appears legitimate").
    """
    bullets: list[str] = []

    # Order themes by importance. "signature" IS included: a broken/invalid
    # signature is a real finding the user must see — only the low-severity
    # informational codes (TRUSTED*, UNVERIFIED) are filtered below. (An
    # earlier version silently hid ALL signature findings — audit bug.)
    theme_order = [
        "tampering", "dates", "identity", "origin",
        "security", "signature", "visual", "images", "fonts", "other",
    ]

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    mentioned_codes: set[str] = set()

    for theme in theme_order:
        if theme not in grouped:
            continue

        flags = grouped[theme]

        # Sort by severity, pick only the most severe flag per theme
        sorted_flags = sorted(flags, key=lambda f: severity_order.get(f.severity, 4))

        for flag in sorted_flags:
            if flag.code in mentioned_codes:
                continue
            # Skip low-severity flags — not worth a bullet
            if flag.severity == "low":
                continue
            # Skip GENUINE-direction flags — positive evidence is presented in
            # the positive-signals section, never as a finding bullet.
            from src.evidence import GENUINE_CODES
            if flag.code in GENUINE_CODES:
                continue

            sentence = _flag_to_sentence(flag, lang)
            if sentence and sentence not in bullets:
                bullets.append(sentence)
                mentioned_codes.add(flag.code)
                if len([c for c in mentioned_codes if c in [f.code for f in flags]]) >= 2:
                    break  # Max 2 bullets per theme

    return bullets


def _get_positive_signals(
    grouped: dict[str, list[Flag]],
    module_results: list[ModuleResult],
    lang: str = "en",
) -> list[str]:
    """
    Generate positive sentences for modules that are clean.

    A module is "clean" if it has score >= 90 and no flags.
    We also check for trusted signatures as a special positive signal.

    Args:
        grouped: Flags grouped by theme.
        module_results: All module results.

    Returns:
        List of positive signal sentences.
    """
    positives: list[str] = []

    sig_positive = SIGNATURE_POSITIVE_FR if lang == "fr" else SIGNATURE_POSITIVE
    pos_signals = POSITIVE_SIGNALS_FR if lang == "fr" else POSITIVE_SIGNALS

    # Check for trusted digital signature (special case — it's a flag but positive)
    if "signature" in grouped:
        for flag in grouped["signature"]:
            if flag.code == "STRUCT_SIGNATURE_TRUSTED":
                positives.append(sig_positive)
                break

    # Machine-verified e-invoice consistency (flag with genuine direction)
    for flags_in_theme in grouped.values():
        if any(f.code == "ARITH_XML_CONSISTENT" for f in flags_in_theme):
            positives.append(
                "Les données de facture électronique intégrées correspondent au "
                "document visible (totaux/IBAN vérifiés)."
                if lang == "fr" else
                "The embedded e-invoice data matches the visible document "
                "(totals/IBAN verified)."
            )
            break

    # Machine-verified 2D-Doc consistency (flag with genuine direction)
    for flags_in_theme in grouped.values():
        if any(f.code == "TWOD_DOC_CONSISTENT" for f in flags_in_theme):
            positives.append(
                "Les champs du code 2D-Doc signé correspondent au document visible."
                if lang == "fr" else
                "The signed 2D-Doc barcode fields match the visible document."
            )
            break

    # Check clean modules
    # Only mention positive signals if there are also negative findings
    # (for a fully clean doc, the verdict already says "appears legitimate")
    has_negative_flags = any(
        theme not in ("signature", "fonts") and theme in grouped
        for theme in grouped
    )

    if not has_negative_flags:
        return positives

    # Pick up to 2 positive signals to keep it concise
    count = 0
    intro = "En revanche, " if lang == "fr" else "On the positive side, "
    for module in module_results:
        if count >= 2:
            break
        if module.score >= 90 and len(module.flags) == 0:
            signal = pos_signals.get(module.module)
            if signal:
                positives.append(intro + signal[0].lower() + signal[1:]
                                 if count == 0 and positives == []
                                 else signal.capitalize() if not signal[0].isupper()
                                 else signal)
                count += 1

    return positives


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_rich_summary(result: AnalysisResult, lang: str = "en") -> AnalysisSummary:
    """
    Generate a rich, template-based summary of the analysis.

    This is the main entry point. It produces a verdict (short bold statement)
    and a list of bullet-point findings.

    Args:
        result: The complete analysis result from TrustyFileAnalyzer.
        lang: Language code ("en" or "fr"). Defaults to "en".

    Returns:
        AnalysisSummary with verdict and bullets.

    Example:
        >>> summary = generate_rich_summary(result, lang="fr")
        >>> print(summary.verdict)
        "Ce document semble légitime. Aucun problème détecté."
    """
    all_flags = collect_all_flags(result.modules)

    # Group flags by user-facing theme
    grouped = _group_flags_by_theme(all_flags)

    # Find the dominant theme for verdict selection
    dominant_theme = _get_dominant_theme(grouped)

    # Build verdict
    verdict = _build_verdict(result.risk_level, dominant_theme, lang)

    # Build bullet list of findings
    bullets = _build_bullets(grouped, result.modules, lang)

    # Clean document: no bullets needed, verdict says it all
    return AnalysisSummary(verdict=verdict, bullets=bullets)


def get_flag_sentence(flag: Flag, lang: str = "en") -> str:
    """
    Return a translated human-readable sentence for a single flag.

    Public wrapper around _flag_to_sentence(), intended for use by the UI.

    Args:
        flag: A Flag object from any analysis module.
        lang: Language code ("en" or "fr"). Defaults to "en".

    Returns:
        A translated sentence, or flag.message as fallback.
    """
    return _flag_to_sentence(flag, lang)
