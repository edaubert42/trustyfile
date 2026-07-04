"""
Evidence ledger primitives for TrustyFile scoring (Phase 1, spec §3.1-3.4).

Replaces the confidence-weighted average with additive log-odds: each piece of
Evidence carries a weight of evidence (log-likelihood-ratio, in nats) pushing
toward "tampered" (direction="tamper") or "genuine" (direction="genuine").
"hard" evidence is cryptographic/arithmetic certainty and vetoes/floors the
score; "soft" evidence is heuristic and is combined in log-odds space.
"""

from dataclasses import dataclass
from typing import Literal
from src.models import Flag

# --- Tunable defaults (single source of truth; retuned by the calibration harness) ---
SEVERITY_WEIGHTS: dict[str, float] = {"low": 0.2, "medium": 0.7, "high": 1.5, "critical": 3.0}
PRIOR_LOG_ODDS: float = -4.0


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
    """Adapt a legacy Flag to soft, tamper-direction Evidence (compatibility shim).

    Hard-evidence and genuine-direction codes are reclassified in a later task
    (see the hard/positive registry); by default a Flag is soft tamper evidence
    weighted by its severity.
    """
    return Evidence(
        code=flag.code,
        weight=SEVERITY_WEIGHTS.get(flag.severity, SEVERITY_WEIGHTS["low"]),
        kind="soft",
        direction="tamper",
        group=module,
        message=flag.message,
        confidence=1.0,
        details=flag.details,
    )
