# Phase 1 — Scoring Engine (Evidence Ledger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confidence-weighted **average** in `scoring.py` with an **additive log-odds evidence ledger** that separates hard (cryptographic/arithmetic) from soft (heuristic) evidence, so one strong finding can no longer be diluted by clean modules — the single biggest detection-quality gain in the redesign.

**Architecture:** A new `src/evidence.py` holds the `Evidence` dataclass, a severity→weight shim, and a pure `aggregate_evidence()` function (prior log-odds + Σ soft weights with per-group capping, then hard-negative clamp / hard-positive floor, mapped to a 0-100 score, a band, and a range). `scoring.create_analysis_result()` becomes a thin orchestrator: it converts each module's existing `Flag`s to `Evidence` via the shim, runs the aggregator, and populates the existing `trust_score`/`risk_level` (UI stays working) plus new `band`/`score_low`/`score_high`/`evidence_ledger` fields on `AnalysisResult`. Modules are **not** rewritten in Phase 1 — they keep emitting `Flag`s; the shim adapts them. The gameable document-type cap-skip is replaced by per-flag-code suppression.

**Tech Stack:** Python 3.11+ (dev venv `.venv/`, 3.12). Pure-pip. `math` (stdlib) for the log-odds math is sufficient for the engine; `numpy`/`scipy`/`scikit-learn` are permitted by the spec but Phase 1 needs only `math`. PyMuPDF etc. unchanged.

## Global Constraints

- **Pure-pip only.** Phase 1 adds no dependency beyond stdlib `math` (numpy/scipy/sklearn allowed by spec §6 but not required here).
- **The document never leaves the machine.** (Unchanged; the engine does no I/O.)
- **Fail closed.** No-evidence-analyzable / all-modules-abstain / EXTRACTION_FAILED must yield an explicit `UNVERIFIABLE` outcome, never a trusting high score. (Spec §3.3.)
- **Preserve the UI contract.** `AnalysisResult.trust_score` (int 0-100) and `AnalysisResult.risk_level` (`"LOW"|"MEDIUM"|"HIGH"|"CRITICAL"`) MUST remain present and populated — `app.py` reads them. `Flag` and `ModuleResult` keep their current fields. New fields are additive.
- **Git enabled.** Work on branch `phase1-scoring-engine` off `main`. Each task ends with a full-suite `pytest -q` run **and** a commit; commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Test runner:** `.venv/bin/python -m pytest`. Starting suite state on `main`: **367 passed, 0 failed**. Phase 1 rewrites the scoring tests (see Task 6) — the end state is all-green with the new engine's tests.
- **Calibratable defaults, stated explicitly.** All numeric constants below are Phase-1 defaults grounded in spec §6; they live in one module-level table so the calibration harness (Task 9) can retune them. They are NOT magic numbers scattered in code.

### Tunable defaults (single source of truth — `src/evidence.py`)

- **Soft severity weights** (nats, toward *tamper*): `low=0.2, medium=0.7, high=1.5, critical=3.0`.
- **Prior log-odds** of tampering: `-4.0` (a no-evidence document scores ≈ 98 → GENUINE).
- **Score mapping:** `p_tamper = sigmoid(logit)`, `score = round(100 * (1 - p_tamper))`.
- **Bands** (from final score): `GENUINE ≥ 80` (risk LOW), `REVIEW 50-79` (risk MEDIUM), `SUSPICIOUS 20-49` (risk HIGH), `SUSPICIOUS < 20` (risk CRITICAL). `UNVERIFIABLE` is a separate state, not a score band.
- **Hard-negative clamp:** any hard-negative Evidence clamps `score ≤ 15` (band SUSPICIOUS/CRITICAL).
- **Hard-positive floor:** any hard-positive Evidence raises `score ≥ 85` (band GENUINE) — **unless** a hard-negative is also present, in which case the hard-negative wins (clamp applies).
- **Correlation-group cap:** within a `group`, summed soft weight is capped at `max_item_weight + 0.5 * (sum_of_remaining_item_weights)` (diminishing returns — correlated weak flags can't stack linearly).
- **Band width (uncertainty):** `± round(8 * (1 - mean_confidence) + 4 * conflict)` where `conflict = 1` if both tamper-direction and genuine-direction soft evidence are present, else 0. Clamped so the range stays within [0, 100].

---

### Task 1: `Evidence` model + severity→weight shim

**Files:**
- Create: `src/evidence.py`
- Modify: `src/models.py` (re-export `Evidence` for convenience; add new `AnalysisResult` fields in Task 5 — not here)
- Test: `tests/test_evidence.py` (create)

**Interfaces:**
- Produces:
  - `@dataclass Evidence(code:str, weight:float, kind:Literal["hard","soft"], direction:Literal["tamper","genuine"], group:str, message:str, confidence:float=1.0, details:dict|None=None)`
  - `SEVERITY_WEIGHTS: dict[str,float]` = `{"low":0.2,"medium":0.7,"high":1.5,"critical":3.0}`
  - `PRIOR_LOG_ODDS: float = -4.0`
  - `flag_to_evidence(flag: Flag, module: str) -> Evidence` — maps a `Flag` to soft, tamper-direction Evidence by default, `weight = SEVERITY_WEIGHTS[flag.severity]`, `group = module`, carrying code/message/details; confidence 1.0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evidence.py`:

```python
from src.models import Flag
from src.evidence import Evidence, flag_to_evidence, SEVERITY_WEIGHTS


def test_flag_to_evidence_defaults():
    f = Flag(severity="high", code="STRUCT_JS", message="js found", details={"x": 1})
    e = flag_to_evidence(f, module="structure")
    assert isinstance(e, Evidence)
    assert e.kind == "soft"
    assert e.direction == "tamper"
    assert e.weight == SEVERITY_WEIGHTS["high"] == 1.5
    assert e.group == "structure"
    assert e.code == "STRUCT_JS"
    assert e.message == "js found"
    assert e.details == {"x": 1}


def test_severity_weight_ordering():
    assert (SEVERITY_WEIGHTS["low"] < SEVERITY_WEIGHTS["medium"]
            < SEVERITY_WEIGHTS["high"] < SEVERITY_WEIGHTS["critical"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -v`
Expected: FAIL — `src.evidence` does not exist.

- [ ] **Step 3: Create `src/evidence.py`**

```python
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
```

- [ ] **Step 4: Re-export from models for convenience**

In `src/models.py`, at the end of the file, add:

```python
# Evidence lives in src/evidence.py to avoid a circular import (it imports Flag);
# re-exported here so callers can `from src.models import Evidence`.
def _import_evidence():
    from src.evidence import Evidence  # noqa: F401
    return Evidence
```

(Do NOT `from src.evidence import Evidence` at module top of models.py — evidence.py imports from models.py, which would be circular. Callers import `Evidence` from `src.evidence` directly. This helper is optional; if it complicates, omit it and always import from `src.evidence`.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → Expected: 0 failed (367 + 2 new = 369 passed).
```bash
git add -A && git commit -m "feat(scoring): Evidence dataclass + severity→weight shim (Phase 1 Task 1)"
```

---

### Task 2: Additive log-odds soft aggregator

**Files:**
- Modify: `src/evidence.py` (add `sigmoid`, `_group_capped_soft_sum`, `aggregate_soft`)
- Test: `tests/test_evidence.py` (append)

**Interfaces:**
- Consumes: `Evidence`, `PRIOR_LOG_ODDS`.
- Produces:
  - `sigmoid(x: float) -> float`
  - `aggregate_soft(evidence: list[Evidence], prior: float = PRIOR_LOG_ODDS) -> float` — returns posterior **logit** (float) from prior + group-capped soft weights (tamper positive, genuine negative). Ignores hard evidence (handled in Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evidence.py`:

```python
from src.evidence import sigmoid, aggregate_soft, PRIOR_LOG_ODDS, Evidence


def _soft(code, weight, direction="tamper", group="g"):
    return Evidence(code=code, weight=weight, kind="soft",
                    direction=direction, group=group, message=code)


def test_no_evidence_returns_prior():
    assert aggregate_soft([]) == PRIOR_LOG_ODDS


def test_one_strong_signal_not_diluted_by_many_clean():
    # The core failure of averaging: a strong tamper signal must survive
    # even when it is the ONLY module that fired (others are clean = no evidence).
    strong = [_soft("BAD", 3.0)]  # one critical-weight finding
    logit = aggregate_soft(strong)
    assert logit > PRIOR_LOG_ODDS + 2.5   # the +3.0 tamper weight is preserved


def test_genuine_evidence_lowers_logit():
    logit = aggregate_soft([_soft("GOOD", 1.5, direction="genuine")])
    assert logit < PRIOR_LOG_ODDS


def test_sigmoid_monotonic():
    assert sigmoid(-10) < sigmoid(0) < sigmoid(10)
    assert 0.0 < sigmoid(0) < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -k "aggregate_soft or sigmoid or one_strong or genuine_evidence or no_evidence_returns" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement the aggregator**

Add to `src/evidence.py`:

```python
import math
from collections import defaultdict


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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): additive log-odds soft aggregator with group capping (Phase 1 Task 2)"
```

---

### Task 3: Aggregate result — hard veto/clamp, positive floor, score, band, range

**Files:**
- Modify: `src/evidence.py` (add `AggregateResult`, band constants, `aggregate_evidence`)
- Test: `tests/test_evidence.py` (append)

**Interfaces:**
- Consumes: `aggregate_soft`, `sigmoid`, `Evidence`.
- Produces:
  - constants `HARD_NEGATIVE_CEILING=15`, `HARD_POSITIVE_FLOOR=85`, band thresholds.
  - `@dataclass AggregateResult(score:int, band:str, risk_level:str, score_low:int, score_high:int, ledger:list[Evidence])` where `band ∈ {"GENUINE","REVIEW","SUSPICIOUS","UNVERIFIABLE"}` and `risk_level ∈ {"LOW","MEDIUM","HIGH","CRITICAL","UNVERIFIABLE"}`.
  - `aggregate_evidence(evidence:list[Evidence], analyzable:bool=True, prior:float=PRIOR_LOG_ODDS) -> AggregateResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evidence.py`:

```python
from src.evidence import aggregate_evidence, Evidence


def _ev(code, weight, kind="soft", direction="tamper", group="g", conf=1.0):
    return Evidence(code=code, weight=weight, kind=kind, direction=direction,
                    group=group, message=code, confidence=conf)


def test_clean_document_is_genuine():
    r = aggregate_evidence([])
    assert r.band == "GENUINE"
    assert r.risk_level == "LOW"
    assert r.score >= 80


def test_hard_negative_clamps_regardless_of_clean_modules():
    # One hard-negative among otherwise-clean evidence must clamp low.
    r = aggregate_evidence([_ev("REVISION_AMOUNT_CHANGED", 3.0, kind="hard")])
    assert r.score <= 15
    assert r.band == "SUSPICIOUS"
    assert r.risk_level == "CRITICAL"


def test_hard_positive_floors_up():
    r = aggregate_evidence([_ev("SIG_TRUSTED_FULL", 3.0, kind="hard", direction="genuine")])
    assert r.score >= 85
    assert r.band == "GENUINE"


def test_hard_negative_beats_hard_positive():
    r = aggregate_evidence([
        _ev("SIG_TRUSTED_FULL", 3.0, kind="hard", direction="genuine"),
        _ev("REVISION_AMOUNT_CHANGED", 3.0, kind="hard", direction="tamper"),
    ])
    assert r.score <= 15  # tampering wins


def test_unverifiable_when_not_analyzable():
    r = aggregate_evidence([], analyzable=False)
    assert r.band == "UNVERIFIABLE"
    assert r.risk_level == "UNVERIFIABLE"


def test_soft_pileup_does_not_reach_hard_clamp():
    # Several soft highs are serious but must NOT auto-equal a hard-negative clamp.
    r = aggregate_evidence([_ev(f"H{i}", 1.5, group=f"m{i}") for i in range(4)])
    assert r.score > 15  # soft evidence lands in SUSPICIOUS/REVIEW, not hard-clamped
    assert r.band in ("SUSPICIOUS", "REVIEW")


def test_band_width_grows_with_low_confidence():
    high_conf = aggregate_evidence([_ev("X", 1.5, conf=1.0)])
    low_conf = aggregate_evidence([_ev("X", 1.5, conf=0.3)])
    assert (low_conf.score_high - low_conf.score_low) >= (high_conf.score_high - high_conf.score_low)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -k "aggregate_evidence or clean_document or hard_negative or hard_positive or unverifiable or soft_pileup or band_width" -v`
Expected: FAIL — `aggregate_evidence`/`AggregateResult` not defined.

- [ ] **Step 3: Implement**

Add to `src/evidence.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -v`
Expected: PASS (all Task 1-3 tests).

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): aggregate_evidence — hard veto/floor, bands, score range (Phase 1 Task 3)"
```

---

### Task 4: Hard / positive evidence registry (reclassify key flag codes)

**Files:**
- Modify: `src/evidence.py` (add `HARD_NEGATIVE_CODES`, `HARD_POSITIVE_CODES`, `GENUINE_CODES`; upgrade `flag_to_evidence` to consult them)
- Test: `tests/test_evidence.py` (append)

**Interfaces:**
- Modifies: `flag_to_evidence(flag, module)` now sets `kind`/`direction` from the registries (default remains soft/tamper).
- Produces: three `frozenset[str]` registries.

Rationale: Phase 0/later modules already emit some codes that are cryptographic/arithmetic certainties or positive signals. Map the ones that exist today; later phases add their codes to these sets.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evidence.py`:

```python
from src.models import Flag
from src.evidence import flag_to_evidence


def test_extraction_failed_is_hard_negative():
    e = flag_to_evidence(Flag("high", "EXTRACTION_FAILED", "x"), "extraction")
    assert e.kind == "hard" and e.direction == "tamper"


def test_siren_vat_mismatch_is_hard_negative():
    # deterministic arithmetic contradiction between two fields
    e = flag_to_evidence(Flag("critical", "CONTENT_SIREN_VAT_MISMATCH", "x"), "content")
    assert e.kind == "hard" and e.direction == "tamper"


def test_generic_flag_stays_soft_tamper():
    e = flag_to_evidence(Flag("medium", "META_SUSPICIOUS_PRODUCER", "x"), "metadata")
    assert e.kind == "soft" and e.direction == "tamper"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -k "hard_negative or stays_soft" -v`
Expected: FAIL (EXTRACTION_FAILED / SIREN_VAT currently classified soft).

- [ ] **Step 3: Add registries and consult them**

Add to `src/evidence.py` (near the top constants):

```python
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
```

Replace `flag_to_evidence` with:

```python
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
```

Note: the earlier `test_flag_to_evidence_defaults` uses code `STRUCT_JS` (not in any registry) → still soft/tamper, so it stays green.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): hard/positive evidence code registries (Phase 1 Task 4)"
```

---

### Task 5: New `AnalysisResult` fields (band, range, ledger)

**Files:**
- Modify: `src/models.py` (`AnalysisResult` dataclass ~99-117)
- Test: `tests/test_models.py` (create)

**Interfaces:**
- Produces: `AnalysisResult` gains `band: str = "GENUINE"`, `score_low: int = 0`, `score_high: int = 100`, `evidence_ledger: list = field(default_factory=list)`. Existing fields (`file_hash`, `trust_score`, `risk_level`, `document_type`, `modules`, `analysis_time_ms`) unchanged and keep their defaults/positions so existing constructor calls still work.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
from src.models import AnalysisResult


def test_analysis_result_has_new_fields_with_defaults():
    r = AnalysisResult(file_hash="h", trust_score=90, risk_level="LOW")
    assert r.band == "GENUINE"
    assert r.score_low == 0 and r.score_high == 100
    assert r.evidence_ledger == []


def test_existing_fields_intact():
    r = AnalysisResult(file_hash="h", trust_score=42, risk_level="HIGH",
                       document_type="invoice", analysis_time_ms=5)
    assert r.trust_score == 42 and r.risk_level == "HIGH"
    assert r.document_type == "invoice" and r.analysis_time_ms == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — `band`/`score_low` etc. don't exist.

- [ ] **Step 3: Add the fields**

In `src/models.py`, in the `AnalysisResult` dataclass, add after `analysis_time_ms: int = 0`:

```python
    band: str = "GENUINE"            # GENUINE | REVIEW | SUSPICIOUS | UNVERIFIABLE
    score_low: int = 0               # lower bound of the score band
    score_high: int = 100            # upper bound of the score band
    evidence_ledger: list = field(default_factory=list)  # list[Evidence]
```

(`field` is already imported in models.py.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): AnalysisResult band/range/ledger fields (Phase 1 Task 5)"
```

---

### Task 6: Rewire `create_analysis_result` onto the ledger (+ migrate scoring tests)

**Files:**
- Modify: `src/scoring.py` (`create_analysis_result` ~233-332)
- Test: `tests/test_scoring.py` (migrate — this is the big one)

**Interfaces:**
- Consumes: `flag_to_evidence`, `aggregate_evidence`, `AggregateResult`.
- Modifies: `create_analysis_result(file_hash, module_results, analysis_time_ms=0, document_type="unknown") -> AnalysisResult` — signature unchanged; internals now build Evidence and aggregate. Populates `trust_score` (=`AggregateResult.score`), `risk_level`, `band`, `score_low`, `score_high`, `evidence_ledger`.
- `calculate_final_score`, `get_risk_level`, `collect_all_flags`, `count_flags_by_severity` are RETAINED (still imported by tests/other code) but no longer drive the trust score.

- [ ] **Step 1: Rewrite `create_analysis_result`**

Replace the body of `create_analysis_result` in `src/scoring.py` (keep the signature and docstring intro) with:

```python
    from src.evidence import flag_to_evidence, aggregate_evidence
    from src.models import AnalysisResult

    # An explicit extraction failure means the document is unanalyzable → UNVERIFIABLE.
    analyzable = not any(
        f.code == "EXTRACTION_FAILED" for r in module_results for f in r.flags
    )

    # A module that found nothing (score 100, no flags) ABSTAINS — it contributes
    # no evidence rather than nudging trust upward.
    evidence = []
    for r in module_results:
        if not r.flags and r.score >= 100:
            continue  # abstain
        for f in r.flags:
            ev = flag_to_evidence(f, r.module)
            # Per-flag-code suppression by document type (replaces cap-skip).
            ev = apply_doc_type_suppression(ev, document_type)
            if ev is not None:
                evidence.append(ev)

    agg = aggregate_evidence(evidence, analyzable=analyzable)

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
    )
```

(`apply_doc_type_suppression` is defined in Task 7; for this task, add a temporary pass-through at the top of scoring.py so Task 6 is testable independently:)

```python
def apply_doc_type_suppression(evidence, document_type):
    """Placeholder pass-through; real per-code suppression added in Task 7."""
    return evidence
```

- [ ] **Step 2: Migrate the scoring tests**

The existing `tests/test_scoring.py` asserts the OLD weighted-average + caps behavior, which no longer holds. Rewrite the affected classes to the new engine. Keep `TestGetRiskLevel` (thresholds unchanged) and the `collect_all_flags`/`count_flags_by_severity` tests (those functions are retained). Replace the average-math and cap tests with ledger-behavior tests. Add this test class to `tests/test_scoring.py` (and DELETE the old `TestCalculateFinalScore` average-assertion class and the old critical-cap classes that assert `<= 40`/`<= 19` average math):

```python
class TestEvidenceLedgerScoring:
    def test_clean_modules_score_genuine(self):
        results = [make_result("metadata", 100, 1.0), make_result("content", 100, 1.0)]
        r = create_analysis_result("h", results)
        assert r.band == "GENUINE" and r.risk_level == "LOW" and r.trust_score >= 80

    def test_one_module_with_findings_not_diluted(self):
        # structure found a serious problem; everything else clean.
        results = [
            make_result("metadata", 100, 1.0),
            make_result("content", 100, 1.0),
            make_result("fonts", 100, 1.0),
            make_result("structure", 0, 1.0, flags=[make_flag("high", "STRUCT_JS")]),
        ]
        r = create_analysis_result("h", results)
        # Old averaging gave ~93 (LOW). The ledger must NOT read GENUINE here.
        assert r.band in ("REVIEW", "SUSPICIOUS")
        assert r.trust_score < 80

    def test_hard_negative_flag_clamps(self):
        results = [make_result("content", 0, 1.0,
                   flags=[make_flag("critical", "CONTENT_SIREN_VAT_MISMATCH")])]
        r = create_analysis_result("h", results)
        assert r.trust_score <= 15 and r.risk_level == "CRITICAL"

    def test_extraction_failed_is_unverifiable(self):
        results = [make_result("extraction", 0, 1.0,
                   flags=[make_flag("high", "EXTRACTION_FAILED")])]
        r = create_analysis_result("h", results)
        assert r.band == "UNVERIFIABLE" and r.risk_level == "UNVERIFIABLE"

    def test_abstaining_modules_do_not_raise_trust(self):
        # A single medium finding with many clean (abstaining) modules stays flagged.
        results = [make_result(f"m{i}", 100, 1.0) for i in range(8)]
        results.append(make_result("visual", 85, 1.0, flags=[make_flag("medium", "VISUAL_X")]))
        r = create_analysis_result("h", results)
        assert r.trust_score < 90  # the medium finding is visible, not averaged away

    def test_result_carries_ledger_and_band_range(self):
        results = [make_result("structure", 0, 1.0, flags=[make_flag("high", "STRUCT_JS")])]
        r = create_analysis_result("h", results)
        assert len(r.evidence_ledger) == 1
        assert r.score_low <= r.trust_score <= r.score_high
```

Also update any assertion in the retained classes that referenced the old average (e.g. an "empty → 100" test: with the ledger, empty modules → GENUINE, score ≈ 98; change the assertion to `>= 80` / band GENUINE).

- [ ] **Step 3: Run scoring tests**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: PASS (migrated tests). Iterate until green — if a retained test still asserts old-average numbers, update it to the ledger behavior (this is expected migration, not a code bug).

- [ ] **Step 4: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): drive trust_score from evidence ledger; migrate scoring tests (Phase 1 Task 6)"
```

---

### Task 7: Per-flag-code document-type suppression (replace the gameable cap-skip)

**Files:**
- Modify: `src/scoring.py` (implement `apply_doc_type_suppression`; remove the old `CONFIDENCE_MULTIPLIERS` whole-module silencing and `_skip_caps_types` cap-skip from the now-unused average path)
- Test: `tests/test_scoring.py` (append)

**Interfaces:**
- Modifies: `apply_doc_type_suppression(evidence: Evidence, document_type: str) -> Evidence | None` — returns the Evidence unchanged, a down-weighted copy, or `None` (fully suppressed) based on a per-(doc_type, code) table. **Never** suppresses hard evidence or universally-suspicious codes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scoring.py`:

```python
class TestDocTypeSuppression:
    def test_font_diversity_suppressed_for_scientific(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        e = Evidence("FONTS_HIGH_DIVERSITY", 1.5, "soft", "tamper", "fonts", "m")
        out = apply_doc_type_suppression(e, "scientific")
        assert out is None or out.weight < 1.5  # suppressed/down-weighted

    def test_hard_evidence_never_suppressed(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        e = Evidence("CONTENT_SIREN_VAT_MISMATCH", 3.0, "hard", "tamper", "content", "m")
        out = apply_doc_type_suppression(e, "scientific")
        assert out is not None and out.kind == "hard" and out.weight == 3.0

    def test_universal_code_never_suppressed_by_classifier(self):
        from src.scoring import apply_doc_type_suppression
        from src.evidence import Evidence
        # A tampering code an attacker cannot dodge by reclassifying the doc.
        e = Evidence("STRUCT_JAVASCRIPT_DETECTED", 3.0, "soft", "tamper", "structure", "m")
        out = apply_doc_type_suppression(e, "technical")
        assert out is not None and out.weight == 3.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_scoring.py::TestDocTypeSuppression -v`
Expected: FAIL — current `apply_doc_type_suppression` is a pass-through.

- [ ] **Step 3: Implement suppression**

Replace the placeholder `apply_doc_type_suppression` in `src/scoring.py` with:

```python
from dataclasses import replace

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
```

- [ ] **Step 4: Remove the dead cap-skip / whole-module multipliers**

In `src/scoring.py`, the old `CONFIDENCE_MULTIPLIERS` table, `_skip_caps_types`, and the critical-cap block inside `create_analysis_result` are no longer used (Task 6 replaced the body). Delete `CONFIDENCE_MULTIPLIERS` and `_skip_caps_types` and any now-unreferenced helper, keeping `MODULE_WEIGHTS`/`calculate_final_score`/`get_risk_level`/`collect_all_flags`/`count_flags_by_severity` only if still imported elsewhere. Grep first: `grep -rn "CONFIDENCE_MULTIPLIERS\|_skip_caps_types" src/ tests/ app.py` — remove only if the sole references are the definitions themselves.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v` → PASS.

- [ ] **Step 6: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(scoring): per-flag-code doc-type suppression replaces gameable cap-skip (Phase 1 Task 7)"
```

---

### Task 8: Analyzer integration check (end-to-end on the corpus)

**Files:**
- Modify: `src/analyzer.py` (only if the EXTRACTION_FAILED guard from Phase 0 needs to set `band`/ledger — verify it flows through `create_analysis_result`)
- Test: `tests/test_analyzer_ledger.py` (create)

**Interfaces:** none new — this task verifies the wired system end-to-end.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyzer_ledger.py`:

```python
from scripts.build_corpus import build_corpus
from src.analyzer import TrustyFileAnalyzer


def test_corpus_scores_through_ledger(tmp_path):
    paths = build_corpus(str(tmp_path))
    a = TrustyFileAnalyzer(enable_external=False)

    clean = a.analyze(paths["legitimate"][0])
    assert clean.band == "GENUINE"
    assert clean.evidence_ledger is not None
    assert clean.score_low <= clean.trust_score <= clean.score_high

    # The metadata-date-tampered forgery carries findings → not GENUINE.
    forged = a.analyze([p for p in paths["suspicious"] if "metadata" in p][0])
    assert forged.band in ("REVIEW", "SUSPICIOUS")
    assert forged.trust_score < clean.trust_score
```

- [ ] **Step 2: Run to verify failure/behavior**

Run: `.venv/bin/python -m pytest tests/test_analyzer_ledger.py -v`
Expected: initially may FAIL if the analyzer's EXTRACTION_FAILED early-return (Phase 0) constructs `AnalysisResult` without `band` — confirm it now gets `band` from `create_analysis_result` (it already calls it, so band flows through). If the early-return path builds AnalysisResult directly, ensure it sets `band="UNVERIFIABLE"`.

- [ ] **Step 3: Reconcile the analyzer**

Confirm `src/analyzer.py`'s Phase-0 EXTRACTION_FAILED guard routes through `create_analysis_result` (it does — Task 2 of Phase 0 used it). If it sets fields directly anywhere, ensure `band="UNVERIFIABLE"`. No other change expected — modules still return `Flag`s, `create_analysis_result` does the ledger work.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_ledger.py -v` → PASS.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "test(scoring): end-to-end ledger scoring on the forged corpus (Phase 1 Task 8)"
```

---

### Task 9: Calibration harness — evidence-weight & band reporting

**Files:**
- Modify: `scripts/eval_harness.py` (`run_evaluation` — also collect band distribution + per-code evidence weights)
- Test: `tests/test_eval_harness.py` (append)

**Interfaces:**
- Modifies: `run_evaluation` report dict gains `"band_counts": {...}` and `"code_weights": {code: weight}` (the effective log-LR weight each code currently carries, from `SEVERITY_WEIGHTS`/registries), so calibration can compare assumed weights against measured FP/FN once labels exist.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_harness.py`:

```python
def test_report_includes_band_counts_and_weights(tmp_path):
    from scripts.eval_harness import run_evaluation
    (tmp_path / "legitimate").mkdir(); (tmp_path / "suspicious").mkdir()
    report = run_evaluation(str(tmp_path))
    assert "band_counts" in report
    assert isinstance(report["band_counts"], dict)
    assert "code_weights" in report
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py::test_report_includes_band_counts_and_weights -v`
Expected: FAIL — keys absent.

- [ ] **Step 3: Extend `run_evaluation`**

In `scripts/eval_harness.py`, in the report dict initialization add `"band_counts": {}` and `"code_weights": {}`; in the per-document loop, tally `result.band` into a `Counter`; after the loop, populate `code_weights` from the evidence registries:

```python
    from collections import Counter
    band_counter = Counter()
    # ... inside the per-document loop, after `result = analyzer.analyze(...)`:
    band_counter[result.band] += 1
    # ... after the loop:
    report["band_counts"] = dict(band_counter)
    from src.evidence import SEVERITY_WEIGHTS, HARD_NEGATIVE_CODES
    report["code_weights"] = {
        **{c: "HARD_NEG" for c in HARD_NEGATIVE_CODES},
        "_soft_severity_weights": dict(SEVERITY_WEIGHTS),
    }
```

(Integrate with the existing structure; keep the empty-corpus graceful path working — `band_counts` stays `{}`.)

- [ ] **Step 4: Run tests + smoke**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v` → PASS.
Run: `.venv/bin/python -m scripts.build_corpus samples && .venv/bin/python -m scripts.eval_harness samples` → prints band_counts (4 docs) and code_weights, no crash.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(harness): report band distribution + evidence weights for calibration (Phase 1 Task 9)"
```

---

## Self-Review

**Spec coverage (§3.1-3.4, §5, §6):**
- Additive log-odds replacing the average → Tasks 2, 6 ✓
- Hard vs soft separation, veto/clamp, positive floor → Tasks 3, 4 ✓
- Correlation-group capping (double-count guard) → Task 2 ✓
- Fail-closed / abstain / UNVERIFIABLE → Tasks 3, 6 ✓
- Three-band GENUINE/REVIEW/SUSPICIOUS + score range + ledger → Tasks 3, 5, 6 ✓
- Per-flag-code doc-type suppression replacing gameable cap-skip → Task 7 ✓
- Two-sided (tamper/genuine) evidence → Tasks 1-4 (direction field; positive registries seeded, populated in Phase 2/3) ✓
- Positive trust signals in the score → Tasks 3-4 (mechanism built; specific hard-positive codes like signature/Factur-X/2D-Doc are emitted by Phase 2/3 modules and added to `HARD_POSITIVE_CODES` then) — **noted as cross-phase dependency, not a Phase-1 gap.**
- Calibration harness weight inputs → Task 9 ✓
- UI ledger DISPLAY (surfacing the band/range/ledger in app.py) is **explicitly deferred** to a follow-up (spec §UI changes); Phase 1 keeps `trust_score`/`risk_level` so the existing UI is unaffected.

**Placeholder scan:** every code step has complete runnable code. The Task 6 `apply_doc_type_suppression` pass-through is an explicit, labeled temporary that Task 7 replaces (not a placeholder — it's testable as-is).

**Type consistency:** `Evidence` fields, `AggregateResult` fields, `flag_to_evidence(flag, module)`, `aggregate_soft(evidence, prior)`, `aggregate_evidence(evidence, analyzable, prior)`, `apply_doc_type_suppression(evidence, document_type) -> Evidence|None`, and the new `AnalysisResult` fields are consistent across Tasks 1-9. `create_analysis_result` signature is unchanged (UI/analyzer callers unaffected).

**Scope flag:** this is one cohesive subsystem (the scoring engine) — kept as a single plan. The UI ledger rendering and the population of hard-positive codes are correctly deferred to later phases, so Phase 1 is independently testable and leaves the app fully working.
