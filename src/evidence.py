"""
Evidence ledger primitives for TrustyFile scoring (Phase 1, spec §3.1-3.4).

Replaces the confidence-weighted average with additive log-odds: each piece of
Evidence carries a weight of evidence (log-likelihood-ratio, in nats) pushing
toward "tampered" (direction="tamper") or "genuine" (direction="genuine").
"hard" evidence is cryptographic/arithmetic certainty and vetoes/floors the
score; "soft" evidence is heuristic and is combined in log-odds space.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal
from src.models import Flag

# --- Tunable defaults (single source of truth; retuned by the calibration harness) ---
SEVERITY_WEIGHTS: dict[str, float] = {"low": 0.2, "medium": 0.7, "high": 1.5, "critical": 3.0}
PRIOR_LOG_ODDS: float = -4.0

# Codes that are cryptographic/arithmetic CERTAINTIES → hard evidence.
# (Grows as later phases add signature/2D-Doc/Factur-X/MRZ-checksum codes.)
HARD_NEGATIVE_CODES: frozenset[str] = frozenset({
    "EXTRACTION_FAILED",           # Phase 0: unreadable → not trustworthy
    # NOTE: identifier checks are NOT hard. A SINGLE-id checksum failure
    # (SIRET/SIREN/VAT) fails on reference/template invoices, foreign companies,
    # and OCR slips — calibrated to SOFT high. Real FPs:
    # facturx_invoice_EN16931.pdf, mustang_Facture_F20180027.pdf (both legit).
    # CONTENT_SIREN_VAT_MISMATCH was demoted to SOFT high too (2026-07 audit):
    # page-wide pooling hard-capped legit two-party/reverse-charge invoices at
    # 15; even the same-line pairing that replaced it is a layout heuristic,
    # not an arithmetic certainty about ONE party's identifiers.
    "OVERLAY_COVER_AND_RETYPE_CONFIRMED",           # Phase 2 Task 4: geometry + editor Producer (C2)
    "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED",     # Phase 2 Task 5: revision diff + editor Producer (C2)
})
HARD_POSITIVE_CODES: frozenset[str] = frozenset(set())  # future: cryptographically verified signatures (pyHanko)
# Soft positive signals — evidence FOR the document. Only signals an attacker
# cannot cheaply manufacture belong here (an empty signature FIELD, for
# instance, is attacker-controllable and must never be genuine-direction).
GENUINE_CODES: frozenset[str] = frozenset({
    "STRUCT_SIGNATURE_TRUSTED",          # EU-trusted, validated signature
    "STRUCT_SIGNATURE_TRUSTED_EXPIRED",  # QC signature, cert since expired
    "ARITH_XML_CONSISTENT",              # visible layer matches embedded e-invoice XML (totals/IBAN)
    "TWOD_DOC_CONSISTENT",               # visible layer matches the signed 2D-Doc barcode fields
})


@dataclass
class Evidence:
    code: str
    weight: float                              # nats; magnitude of evidence
    kind: Literal["hard", "soft"]
    direction: Literal["tamper", "genuine"]
    group: str                                 # correlation group (usually module)
    message: str
    confidence: float = 1.0                    # how sure THIS check is (0..1)
    details: dict | None = None


def flag_to_evidence(flag: Flag, module: str) -> Evidence:
    """Adapt a legacy Flag to Evidence, consulting the hard/positive registries."""
    if flag.code in HARD_NEGATIVE_CODES:
        kind, direction = "hard", "tamper"
    elif flag.code in HARD_POSITIVE_CODES:
        kind, direction = "hard", "genuine"
    elif flag.code in GENUINE_CODES:
        kind, direction = "soft", "genuine"
    else:
        kind, direction = "soft", "tamper"
    return Evidence(
        code=flag.code,
        weight=SEVERITY_WEIGHTS.get(flag.severity, SEVERITY_WEIGHTS["low"]),
        kind=kind,
        direction=direction,
        group=module,
        message=flag.message,
        confidence=1.0,
        details=flag.details,
    )


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _group_capped_soft_sum(items: list["Evidence"]) -> float:
    """Diminishing-returns cap within a correlation group: the largest item at
    full weight, the rest at half weight. Prevents correlated weak flags (e.g.
    producer + font + structure all reflecting one edit) from stacking linearly."""
    if not items:
        return 0.0
    weights = sorted((abs(i.weight) * i.confidence for i in items), reverse=True)
    return weights[0] + 0.5 * sum(weights[1:])


def _soft_buckets(evidence: list["Evidence"]) -> dict[tuple[str, str], list["Evidence"]]:
    """Soft evidence split by (group, direction) — the group-capping unit."""
    buckets: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    for e in evidence:
        if e.kind != "soft":
            continue
        buckets[(e.group, e.direction)].append(e)
    return buckets


def aggregate_soft(evidence: list["Evidence"], prior: float = PRIOR_LOG_ODDS) -> float:
    """Posterior logit = prior + Σ group-capped soft weights (tamper +, genuine −)."""
    logit = prior
    # Split by (group, direction) so tamper and genuine contributions cap separately.
    for (group, direction), items in _soft_buckets(evidence).items():
        capped = _group_capped_soft_sum(items)
        logit += capped if direction == "tamper" else -capped
    return logit


HARD_NEGATIVE_CEILING = 15
HARD_POSITIVE_FLOOR = 85
BAND_GENUINE_MIN = 80
BAND_REVIEW_MIN = 50
BAND_SUSPICIOUS_CRITICAL_MAX = 20  # below this, risk is CRITICAL


@dataclass
class AggregateResult:
    score: int
    band: str          # GENUINE | REVIEW | SUSPICIOUS | UNVERIFIABLE
    risk_level: str    # LOW | MEDIUM | HIGH | CRITICAL | UNVERIFIABLE
    score_low: int
    score_high: int
    ledger: list  # list[Evidence]


def _band_and_risk(score: int) -> tuple[str, str]:
    if score >= BAND_GENUINE_MIN:
        return "GENUINE", "LOW"
    if score >= BAND_REVIEW_MIN:
        return "REVIEW", "MEDIUM"
    if score >= BAND_SUSPICIOUS_CRITICAL_MAX:
        return "SUSPICIOUS", "HIGH"
    return "SUSPICIOUS", "CRITICAL"


def _band_width(evidence: list["Evidence"]) -> int:
    soft = [e for e in evidence if e.kind == "soft"]
    mean_conf = (sum(e.confidence for e in soft) / len(soft)) if soft else 1.0
    dirs = {e.direction for e in soft}
    conflict = 1 if {"tamper", "genuine"} <= dirs else 0
    return round(8 * (1 - mean_conf) + 4 * conflict)


def aggregate_evidence(evidence: list["Evidence"], analyzable: bool = True,
                       prior: float = PRIOR_LOG_ODDS) -> AggregateResult:
    if not analyzable:
        return AggregateResult(score=0, band="UNVERIFIABLE", risk_level="UNVERIFIABLE",
                               score_low=0, score_high=0, ledger=list(evidence))

    # Genuine evidence must not LAUNDER tamper evidence. Consistency signals
    # (matching e-invoice XML, trusted signature) say "the layers agree" —
    # but when meaningful tamper evidence exists, the document's integrity is
    # already in question, so agreement proves nothing. Without this, a
    # fraudster who leaves the XML intact while patching the visible layer
    # (or its metadata) earns a trust boost that lifts a tampered document
    # back into GENUINE (observed on the tamper corpus: 73/REVIEW → 85).
    #
    # The threshold is the NET group-capped tamper contribution, not any
    # single flag: four stacked mediums are as disqualifying as one high
    # (adversarial review: 4 mediums = 77/REVIEW was laundered to 87/GENUINE
    # under a per-flag test). Genuine evidence still lifts clean documents
    # and offsets small incidental noise — its purpose.
    #
    # Only SOFT genuine evidence is dropped: a future HARD positive
    # (cryptographically verified signature) is a deliberate floor decision,
    # not something soft heuristics may silently strip.
    has_hard_tamper = any(e.kind == "hard" and e.direction == "tamper" for e in evidence)
    net_tamper = sum(
        _group_capped_soft_sum(items)
        for (_, direction), items in _soft_buckets(evidence).items()
        if direction == "tamper"
    )
    if has_hard_tamper or net_tamper >= SEVERITY_WEIGHTS["high"]:
        evidence = [e for e in evidence
                    if not (e.kind == "soft" and e.direction == "genuine")]

    logit = aggregate_soft(evidence, prior=prior)
    score = round(100 * (1 - sigmoid(logit)))
    score = max(0, min(100, score))

    has_hard_neg = any(e.kind == "hard" and e.direction == "tamper" for e in evidence)
    has_hard_pos = any(e.kind == "hard" and e.direction == "genuine" for e in evidence)
    if has_hard_neg:
        score = min(score, HARD_NEGATIVE_CEILING)      # tampering veto wins
    elif has_hard_pos:
        score = max(score, HARD_POSITIVE_FLOOR)

    band, risk = _band_and_risk(score)
    half = _band_width(evidence)
    return AggregateResult(
        score=score, band=band, risk_level=risk,
        score_low=max(0, score - half), score_high=min(100, score + half),
        ledger=list(evidence),
    )
