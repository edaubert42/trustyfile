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
    "CONTENT_SIREN_VAT_MISMATCH",  # deterministic FR VAT-key ↔ SIREN contradiction
    "CONTENT_INVALID_SIRET",       # Luhn checksum failure on text-layer SIRET
    "CONTENT_INVALID_SIREN",
    "CONTENT_INVALID_VAT",
})
HARD_POSITIVE_CODES: frozenset[str] = frozenset(set())  # populated in Phase 2/3 (signatures, Factur-X, 2D-Doc)
GENUINE_CODES: frozenset[str] = frozenset(set())        # soft positive signals, populated later


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


def aggregate_soft(evidence: list["Evidence"], prior: float = PRIOR_LOG_ODDS) -> float:
    """Posterior logit = prior + Σ group-capped soft weights (tamper +, genuine −)."""
    logit = prior
    # Split by (group, direction) so tamper and genuine contributions cap separately.
    buckets: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    for e in evidence:
        if e.kind != "soft":
            continue
        buckets[(e.group, e.direction)].append(e)
    for (group, direction), items in buckets.items():
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
