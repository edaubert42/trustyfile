"""
Scoring System - Combines all module results into a final trust score.

`create_analysis_result` (the entry point) builds an evidence ledger from
module FLAGS and aggregates it in log-odds space — see src/evidence.py for the
math. Each flag becomes an Evidence item (`flag_to_evidence`), optionally
down-weighted per document type (`apply_doc_type_suppression`), and the
ledger is combined by `aggregate_evidence` to produce the final score.

Risk levels:
- LOW (80-100): Document appears legitimate
- MEDIUM (50-79): Some concerns, manual verification recommended
- HIGH (20-49): Multiple red flags, likely manipulated
- CRITICAL (0-19): Strong evidence of fraud
"""

from dataclasses import replace
from src.models import ModuleResult, AnalysisResult, Flag, DocumentType


# =============================================================================
# PER-(DOCUMENT TYPE, FLAG CODE) SOFT-WEIGHT MULTIPLIERS
# =============================================================================

# Per-(document_type, flag_code) soft-weight multipliers. Only FP-prone,
# genre-specific codes appear here. Hard evidence and universally-suspicious
# codes are NEVER in this table and always pass at full weight — so an attacker
# cannot disable tamper detection by steering the (text-based) classifier.
DOC_TYPE_CODE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "scientific": {"FONTS_HIGH_DIVERSITY": 0.15, "FONTS_EXCESSIVE_DIVERSITY": 0.15,
                   "STRUCT_INCREMENTAL_UPDATES": 0.3, "IMAGES_SCREENSHOT_DETECTED": 0.4},
    "technical": {"STRUCT_JAVASCRIPT_DETECTED": 1.0,  # explicit: NOT suppressed even here
                  "STRUCT_ACROFORM_DETECTED": 0.3, "FONTS_HIGH_DIVERSITY": 0.4},
    "legal_notice": {"FONTS_HIGH_DIVERSITY": 0.4, "STRUCT_INCREMENTAL_UPDATES": 0.4},
    "identity": {"FONTS_HIGH_DIVERSITY": 0.3, "STRUCT_ACROFORM_DETECTED": 0.5},
}


# Flags that are shown to the reader but carry NO weight in the ledger, because
# they do not claim anything about the document's integrity. Two kinds:
#
#   Pipeline status — says our analysis fell short, not that the file is more
#   tampered. The real consequence is handled by a band override
#   (coverage_blind / hard_detector_blinded), never by a token tamper nudge.
#
#   Document descriptions — ordinary properties of ordinary files. Charging
#   tamper nats for them is a standing tax on every clean document. Each was
#   measured across the local corpus before being added here; see the per-code
#   notes. A code belongs here only if it does NOT discriminate good from bad.
# All of these still SHOW in the per-module breakdown — they are informative to a
# reader. They simply stop charging tamper nats.
#
# Excluding the three description codes changed nothing measurable on the 88-doc
# labeled corpus (false positives, false negatives and ground-truth band agreement
# all identical; mean legitimate score 95.5 -> 95.8). The reason to do it is the
# gate interaction the corpus cannot show, because it holds no signed samples:
# every low flag adds 0.2 nats to net_tamper, and net_tamper gates BOTH the
# soft-genuine strip (1.5) and the hard-positive grant (0.7). Measured: a document
# with a genuinely verified ANTS 2D-Doc signature scores 100, but add four low
# description-flags and it drops to 96 — the cryptographic proof is switched off
# because the file has no Producer field, uses Arial and carries no images. None of
# those facts say anything about integrity.
NON_EVIDENCE_CODES: frozenset[str] = frozenset({
    # Pipeline status
    "COVERAGE_NO_TEXT_LAYER",  # scoring role is the coverage_blind band override
    "MODULE_ERROR",            # a module crashed; see hard_detector_blinded
    # A registry lookup that did not happen (API down, rate-limited, timeout)
    # says nothing about the document. Scored, these made the trust score of a
    # genuine invoice depend on api.gouv.fr's uptime. Still shown to the reader.
    "EXTERNAL_SIRET_VERIFICATION_FAILED",
    "EXTERNAL_SIREN_VERIFICATION_FAILED",
    "EXTERNAL_VAT_VERIFICATION_FAILED",
    "EXTERNAL_IBAN_VERIFICATION_FAILED",
    # Document descriptions
    "META_NO_PRODUCER",   # 9/30 known-good vs 0/58 known-bad — anti-correlated
    "IMAGES_NO_IMAGES",   # images.py's own comment: "many legitimate invoices are text-only"
    # Weakly correlated (24/58 bad vs 6/30 good) but never decisive: at 0.2 nats a
    # low flag cannot move a band by itself, so its only real effect was pushing
    # net_tamper across the gates above. Kept visible, not scored.
    "FONTS_SYSTEM_FONTS",
    # Scoring role is the REVIEW ceiling on issued documents only
    # (UNVERIFIED_ORIGIN_CODES): a CV or a notice printed to PDF loses nothing.
    "META_VIRTUAL_PRINTER",
})


# =============================================================================
# PROVEN MODIFICATION -> SUSPICIOUS band (user decisions 2026-11-12, see
# docs/superpowers/plans/2026-11-12-scoring-provenance-rules.md)
# =============================================================================

# Proof that visible content changed after the file was first written. True
# for any document genre: a revision diff does not depend on what the text says.
MODIFICATION_PROOF_CODES: frozenset[str] = frozenset({
    "OVERLAY_REVISION_TEXT_CHANGED",   # text differs between two revisions
    "OVERLAY_REVISION_VALUE_CHANGED",  # an amount/date/ID differs between revisions
    "FONTS_MIDWORD_CHANGE",            # two font families inside ONE word
                                       # (no legit hit, local corpus 2026-11-12)
    # Not STRUCT_HIDDEN_MODIFICATION: legit Factur-X toolkits (Symtrax) append
    # the XML without touching ModDate — measured on the ZUGFeRD corpus.
})

# Proof that the file is not what the issuer's software produced: it went
# through a consumer editor, or was reopened and saved later. Only meaningful
# for documents an issuer generates once and sends (invoices, official papers);
# a CV or an article is legitimately edited by its own author.
PROVENANCE_BREAK_CODES: frozenset[str] = frozenset({
    "META_ONLINE_CONVERTER",       # iLovePDF, PDF Candy, Sejda...
    "META_IMAGE_EDITOR",           # Photoshop, GIMP... in the Producer chain
    "META_DOCUMENT_MODIFIED",      # ModDate > CreationDate (high = gap over 30 min)
    "STRUCT_XMP_EDITOR_MISMATCH",  # XMP says Acrobat edited it, Producer says otherwise
    "META_NO_CREATION_DATE",       # creation date dropped by a rebuilding tool (Smallpdf)
    "STRUCT_MARKUP_ANNOTATIONS",   # ink, text box, highlight added by a reader
    "OVERLAY_DUPLICATED_TEXT",     # a line pasted again on top of itself
    "OVERLAY_FREEHAND_DRAWING",    # a pen stroke flattened into the page
    "META_MALFORMED_DATE",         # a metadata date that is not a date (hand-edited)
})

# The provenance codes that a legitimate re-save pipeline also produces
# (signing, attaching Factur-X XML, a library that skips the field).
_PIPELINE_EXCUSABLE_CODES: frozenset[str] = frozenset({
    "META_DOCUMENT_MODIFIED", "STRUCT_XMP_EDITOR_MISMATCH", "META_NO_CREATION_DATE",
})

# Genres an issuer generates once and hands over untouched.
ISSUED_DOCUMENT_TYPES: frozenset[str] = frozenset({
    "invoice", "identity", "certificate", "attestation", "tax_document", "payslip",
})

# Signing a PDF appends a revision and rewrites ModDate: on a signed document,
# a date gap is the signature itself. INVALID is not an excuse.
_SIGNATURE_PRESENT_CODES: frozenset[str] = frozenset({
    "STRUCT_SIGNATURE_TRUSTED", "STRUCT_SIGNATURE_TRUSTED_EXPIRED",
    "STRUCT_SIGNATURE_NOT_TRUSTED", "STRUCT_SIGNATURE_UNVERIFIED",
    "STRUCT_SIGNATURE_UNVERIFIABLE",
})


# The file's origin cannot be verified: an issued document re-printed through a
# virtual printer may be the issuer's own export or a screen edit printed again.
# Nothing proves tampering, so it is capped at REVIEW (never GENUINE), not forced
# into SUSPICIOUS. Measured 2026-11-12: 2 legit invoices use one (both already
# REVIEW for other reasons), and no suspicious corpus doc needs it.
UNVERIFIED_ORIGIN_CODES: frozenset[str] = frozenset({"META_VIRTUAL_PRINTER"})


# The e-invoice XML disagrees with the visible page: the visible layer was
# edited after the XML was written. Cancels the e-invoice date excuse below.
_EINVOICE_MISMATCH_CODES: frozenset[str] = frozenset({
    "ARITH_XML_VISIBLE_MISMATCH", "ARITH_IBAN_MISMATCH",
})


def proven_modification_codes(flags: list[Flag], document_type: str,
                              coverage_blind: bool = False,
                              einvoice_xml: bool = False) -> list[str]:
    """Return the flag codes proving the file is not the issuer's original.

    A non-empty result forces the SUSPICIOUS band (see
    evidence.PROVEN_MODIFICATION_CEILING). Low-severity flags never count:
    a module grades a finding "low" exactly when it judged it harmless (a
    re-save that changed nothing, a scanner-to-viewer pipeline).

    Provenance codes only count on issued documents, or on scans (blind
    documents carry no text, so their genre is always "unknown"). For the
    _PIPELINE_EXCUSABLE_CODES, "high" is required: the medium date-gap grade
    means under 30 minutes, the normal scan-then-save or generate-then-view
    pipeline.

    Two legitimate pipelines re-save a finished PDF, so they excuse the
    _PIPELINE_EXCUSABLE_CODES (never a converter):
      - signing it (a signature appends a revision);
      - attaching a Factur-X/ZUGFeRD XML (einvoice_xml=True). Measured: 40 of
        126 ZUGFeRD reference invoices carry a ModDate days after creation.
        The excuse is void when the XML contradicts the visible page — an
        edited visible layer no longer matches the XML written before it.

    Examples:
        >>> f = Flag(severity="high", code="META_ONLINE_CONVERTER", message="")
        >>> proven_modification_codes([f], "invoice")
        ['META_ONLINE_CONVERTER']
        >>> proven_modification_codes([f], "scientific")
        []
    """
    codes = {f.code for f in flags}
    signed = bool(codes & _SIGNATURE_PRESENT_CODES)
    einvoice_ok = einvoice_xml and not (codes & _EINVOICE_MISMATCH_CODES)
    issued = document_type in ISSUED_DOCUMENT_TYPES or coverage_blind

    found = []
    for f in flags:
        if f.severity == "low":
            continue
        if f.code in MODIFICATION_PROOF_CODES:
            found.append(f.code)
        elif f.code in PROVENANCE_BREAK_CODES and issued:
            excusable = f.code in _PIPELINE_EXCUSABLE_CODES
            if excusable and (signed or einvoice_ok or f.severity == "medium"):
                continue
            found.append(f.code)
    return sorted(set(found))


def apply_doc_type_suppression(evidence, document_type: str):
    """Down-weight genre-specific FP-prone SOFT codes for a document type.
    Hard evidence is never touched. Returns Evidence, a down-weighted copy, or
    None (fully suppressed, multiplier 0)."""
    if evidence.kind == "hard":
        return evidence
    mult = DOC_TYPE_CODE_MULTIPLIERS.get(document_type, {}).get(evidence.code, 1.0)
    if mult >= 1.0:
        return evidence
    if mult <= 0.0:
        return None
    return replace(evidence, weight=evidence.weight * mult)


def collect_all_flags(module_results: list[ModuleResult]) -> list[Flag]:
    """
    Collect all flags from all modules, sorted by severity.

    Args:
        module_results: List of results from all analysis modules

    Returns:
        List of all flags, sorted by severity (critical first)
    """
    all_flags = []

    for result in module_results:
        for flag in result.flags:
            # Add module name to flag for context
            all_flags.append(flag)

    # Sort by severity (critical > high > medium > low)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_flags.sort(key=lambda f: severity_order.get(f.severity, 4))

    return all_flags


def _is_scanner_pipeline_critical(flag: Flag) -> bool:
    """True if this flag is the known scanner→viewer false positive.

    Authentic scanned identity documents are saved by the scanner, then opened
    minutes later by a viewer (Adobe, Preview…), which stamps a new ModDate and
    trips META_DOCUMENT_MODIFIED. A gap under 30 minutes is a plausible
    scan-then-view workflow, not tampering. (Keys on code + gap, not severity —
    the 2026-07 judge panel re-graded this code from a flat critical to
    medium/high, and the identity-doc exemption must survive that.)
    """
    if flag.code != "META_DOCUMENT_MODIFIED" or flag.severity == "low":
        return False
    hours = (flag.details or {}).get("time_difference_hours", 999)
    try:
        return float(hours) < 0.5  # under 30 minutes
    except (TypeError, ValueError):
        return False  # unparseable → treat as genuine tamper evidence


def create_analysis_result(
    file_hash: str,
    module_results: list[ModuleResult],
    analysis_time_ms: int = 0,
    document_type: DocumentType = "unknown",
    coverage_blind: bool = False,
    hard_detector_blinded: bool = False,
) -> AnalysisResult:
    """
    Create the final analysis result from all module results.

    Builds an evidence ledger from the flags raised by each module and
    aggregates it in log-odds space (see src/evidence.py) instead of taking a
    confidence-weighted average of module scores. Modules score from their
    FLAGS, not from their numeric `score` — the numeric score is only used to
    detect an abstaining clean module (no flags AND score >= 100), which
    contributes no evidence rather than nudging trust upward.

    Args:
        file_hash: SHA256 hash of the analyzed file
        module_results: List of results from all analysis modules
        analysis_time_ms: Time taken for analysis in milliseconds
        document_type: Detected document type (used for evidence suppression)
        coverage_blind: True when the document has no usable text layer (see analyzer._coverage_blind); GENUINE/REVIEW results collapse to UNVERIFIABLE.
        hard_detector_blinded: True when a module able to produce HARD tamper
            evidence crashed, so the strongest available proof was never looked
            for; denies the GENUINE band.

    Returns:
        AnalysisResult with final score, risk level, band, score range,
        evidence ledger, document type, and module details.
    """
    from src.evidence import flag_to_evidence, aggregate_evidence

    # Fail-closed: an explicit extraction failure means the document could not be
    # read, so it is UNVERIFIABLE — never a trusting high score.
    analyzable = not any(
        f.code == "EXTRACTION_FAILED" for r in module_results for f in r.flags
    )

    # Known false positive on authentic SCANNED identity documents: downgrade the
    # scanner→viewer META_DOCUMENT_MODIFIED critical to "low" so it contributes
    # negligible evidence instead of a critical-weight tamper signal.
    #
    # Gated on the identity module having actually READ AN MRZ, not on the
    # document_type label alone. document_type is decided by keyword hits in
    # attacker-supplied text (src/document_type.py), and this downgrade sits
    # OUTSIDE DOC_TYPE_CODE_MULTIPLIERS — so before this gate, pasting a few ID
    # keywords into an invoice bought the exemption outright: the same ledger
    # scored 38/SUSPICIOUS as "invoice" and 91/GENUINE as "identity". That
    # falsified the invariant asserted at the top of this module. An MRZ that
    # parses is a structural property of a real identity document, not a word
    # an attacker can sprinkle into a body of text.
    #
    # The downgrade builds a NEW Flag rather than mutating in place: the caller's
    # module_results are returned verbatim in AnalysisResult.modules and rendered
    # by the per-module breakdown, so mutating here would permanently hide the
    # module's real finding from the audit trail the reader most needs on an
    # identity document — and would double-apply if the same objects were scored
    # twice.
    mrz_found = any(
        r.module == "identity" and (r.details or {}).get("mrz_found")
        for r in module_results
    )
    downgrade_scanner_fp = document_type == "identity" and mrz_found

    # Build the evidence ledger. A module that found nothing (no flags AND
    # score >= 100) ABSTAINS: it contributes no evidence rather than nudging
    # trust upward. Modules score from their FLAGS, not their numeric score.
    evidence = []
    scored_flags = []  # flags as scored (after the scanner downgrade)
    for r in module_results:
        if not r.flags and r.score >= 100:
            continue
        for f in r.flags:
            if downgrade_scanner_fp and _is_scanner_pipeline_critical(f):
                f = replace(f, severity="low")
            scored_flags.append(f)
            if f.code in NON_EVIDENCE_CODES:
                continue
            ev = flag_to_evidence(f, r.module)
            ev = apply_doc_type_suppression(ev, document_type)
            if ev is not None:
                evidence.append(ev)

    proven = proven_modification_codes(
        scored_flags, document_type, coverage_blind,
        einvoice_xml=any((r.details or {}).get("einvoice_xml") for r in module_results))
    unverified = sorted({f.code for f in scored_flags if f.code in UNVERIFIED_ORIGIN_CODES}
                        if document_type in ISSUED_DOCUMENT_TYPES else set())
    agg = aggregate_evidence(evidence, analyzable=analyzable,
                             coverage_blind=coverage_blind,
                             hard_detector_blinded=hard_detector_blinded,
                             proven_modification=bool(proven),
                             origin_unverified=bool(unverified))
    forced_codes = {code: "REVIEW" for code in unverified}
    forced_codes.update({code: "SUSPICIOUS" for code in proven})

    return AnalysisResult(
        file_hash=file_hash,
        trust_score=agg.score,
        risk_level=agg.risk_level,
        document_type=document_type,
        modules=module_results,
        analysis_time_ms=analysis_time_ms,
        band=agg.band,
        score_low=agg.score_low,
        score_high=agg.score_high,
        evidence_ledger=agg.ledger,
        forced_codes=forced_codes,
    )
