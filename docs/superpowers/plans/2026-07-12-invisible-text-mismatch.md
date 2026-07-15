# Invisible-Text Mismatch Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect PDF text rendered invisibly (Tr mode 3) that conflicts with, or hides from, the visible page — the value-shadowing and whitespace-injection attacks described in `docs/superpowers/specs/2026-07-12-invisible-text-mismatch-design.md`.

**Architecture:** A new standalone module `src/modules/invisible.py` (module name `invisible`) reads `page.get_texttrace()`, gates out OCR-sandwich/scan pages, and compares invisible value-bearing spans against painted spans. Span-comparison helpers shared with the overlay module move verbatim into a new `src/modules/span_conflict.py`. Severity mirrors the overlay C2 pattern (SOFT high → HARD critical `_CONFIRMED` with interactive-editor Producer).

**Tech Stack:** Python 3.11, PyMuPDF (`fitz`), pytest. No new dependencies.

## Global Constraints

- Read the spec first: `docs/superpowers/specs/2026-07-12-invisible-text-mismatch-design.md`. Its D1–D4 sign-offs are binding.
- Implement on a NEW branch `phase7-invisible-text` cut from `main`, inside an isolated worktree (superpowers:using-git-worktrees) — the `phase6-robustness` checkout is owned by another session. Never commit to `phase6-robustness`.
- Python invocation is ALWAYS `.venv/bin/python -m pytest …` (plain `python`/`pytest` do not exist on this machine).
- Never crash on malformed PDFs: every fitz touch is try/excepted; failures degrade `confidence`, never raise (CLAUDE.md).
- The corpus band gate (`tests/test_corpus_bands.py`) must stay green after every task.
- Type hints everywhere; Google-format docstrings in English; `logging`, never `print`.
- Flag codes (exact): `INVISIBLE_TEXT_VALUE_CONFLICT` (high), `INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED` (critical), `INVISIBLE_TEXT_INJECTION` (medium).
- Constants (exact values from the spec): `INVISIBLE_DOMINANCE_MAX = 0.50`, row overlap ≥ `0.5` of the smaller span height, size match within `SIZE_MATCH_FRAC` (0.10, reused), IBAN shape `[A-Z]{2}\d{2}[A-Z0-9]{10,30}`.

---

### Task 1: Extract `span_conflict.py` from `overlay.py` (mechanical move)

**Files:**
- Create: `src/modules/span_conflict.py`
- Modify: `src/modules/overlay.py` (delete moved definitions; import them back; rename two private call sites)
- Test: existing `tests/test_overlay*.py` (no new tests — the move must be behavior-invisible)

**Interfaces:**
- Consumes: nothing new.
- Produces (later tasks import these from `src.modules.span_conflict`):
  `span_text(span: dict) -> str`, `norm(s: str) -> str`,
  `char_class(s: str) -> str`, `amount_token(text: str) -> str | None`,
  `is_placeholder(text: str) -> bool`,
  `is_horizontal(span_dir, page_angle_deg: float) -> bool`,
  `pair_is_conflict(a: dict, b: dict, page_angle: float) -> bool`,
  `page_angle(page) -> float`,
  `INTERACTIVE_EDITOR_PRODUCERS: tuple[str, ...]`,
  `SIZE_MATCH_FRAC: float`, `HORIZONTAL_TOL_DEG: float`.

- [ ] **Step 1: Record the green baseline**

Run: `.venv/bin/python -m pytest tests/test_overlay.py tests/test_overlay_acceptance.py tests/test_overlay_embedded_image.py tests/test_overlay_embedded_span.py tests/test_corpus_bands.py -q`
Expected: all PASS (note the exact pass count — it must be identical after the move).

- [ ] **Step 2: Create `src/modules/span_conflict.py`**

Header to write, followed by the moved definitions **cut verbatim** from `overlay.py` (current line refs given; copy bodies exactly, do not edit logic):

```python
"""
Span-comparison helpers shared by the overlay and invisible modules.

Extracted verbatim from overlay.py (phase 7, invisible-text detector design
D4). These answer one question from several angles: do two text spans hold
conflicting versions of the SAME field? No behavior change vs. overlay.py —
the overlay test suite is the regression proof.
"""
import re
import unicodedata

# --- constants (from overlay.py:28-44) ---
HORIZONTAL_TOL_DEG = 4.0
SIZE_MATCH_FRAC = 0.10
# Move the HUMAN DECISION comment block above this tuple verbatim from
# overlay.py:26-38 (LibreOffice rationale) — it records a user sign-off.
INTERACTIVE_EDITOR_PRODUCERS = (
    "acrobat", "foxit", "pdf-xchange", "pdfxchange", "libreoffice",
    "nitro", "sejda", "pdfescape", "master pdf",
)
```

Then move, in this order, each with its attached comments/docstrings:
- `span_text` (overlay.py:84) and `_has_combining_mark` (94)
- `_CURRENCY_STRIP` comment+constant (~96-104) and `char_class` (106)
- `norm` (119)
- `_AMOUNT_RE` comment+regex (~128) and `amount_token` (141)
- `is_horizontal` (151) — keep its local `import math`
- `_PLACEHOLDER_TOKENS` (195), `_PLACEHOLDER_ZEROS_RE` (208), `is_placeholder` (211)
- `_page_angle` (274) → **rename to `page_angle`** (public now)
- `_pair_is_conflict` (374) → **rename to `pair_is_conflict`**

- [ ] **Step 3: Rewire `overlay.py`**

Delete the moved definitions from `overlay.py` and add below its existing imports:

```python
from src.modules.span_conflict import (
    HORIZONTAL_TOL_DEG, SIZE_MATCH_FRAC, INTERACTIVE_EDITOR_PRODUCERS,
    span_text, char_class, norm, amount_token, is_horizontal,
    is_placeholder, page_angle as _page_angle,
    pair_is_conflict as _pair_is_conflict,
)
```

The `as _page_angle` / `as _pair_is_conflict` aliases keep every existing
overlay call site (`overlay.py:441,559,1203,…`) compiling untouched. Remove
`import unicodedata` from overlay.py if now unused (`grep unicodedata`).

- [ ] **Step 4: Verify behavior is unchanged**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: same pass count as Step 1, zero failures.

- [ ] **Step 5: Commit**

```bash
git add src/modules/span_conflict.py src/modules/overlay.py
git commit -m "refactor(overlay): extract shared span-conflict helpers to span_conflict.py"
```

---

### Task 2: `invisible.py` skeleton + digital-native gate

**Files:**
- Create: `src/modules/invisible.py`
- Test: `tests/test_invisible.py`

**Interfaces:**
- Consumes: `span_conflict` helpers (Task 1 signatures).
- Produces: `analyze_invisible(pdf_data) -> ModuleResult` (module name `"invisible"`); internal `_split_spans(page) -> tuple[list, list]` and `_is_digital_native(painted: list, invisible: list) -> bool` used by Tasks 3–6.

- [ ] **Step 1: Write the failing tests (fixture builder + 3 gate scenarios)**

Create `tests/test_invisible.py`:

```python
"""Tests for the invisible-text mismatch detector (spec 2026-07-12)."""
import fitz
import pytest

from src.modules.invisible import analyze_invisible


class _PdfData:
    """Minimal stand-in for the extractor's pdf_data (duck-typed)."""
    def __init__(self, path: str):
        self.file_path = str(path)
        self.extraction_ok = True


def _pdf(tmp_path, name, painted=(), invisible=(), producer=None):
    """Build a one-page PDF. painted/invisible: (x, y, text, fontsize)."""
    doc = fitz.open()
    page = doc.new_page()
    for x, y, text, size in painted:
        page.insert_text(fitz.Point(x, y), text, fontname="helv", fontsize=size)
    for x, y, text, size in invisible:
        page.insert_text(fitz.Point(x, y), text, fontname="helv",
                         fontsize=size, render_mode=3)
    if producer:
        doc.set_metadata({"producer": producer})
    path = str(tmp_path / name)
    doc.save(path)
    doc.close()
    return path


def _codes(result):
    return [f.code for f in result.flags]


def test_clean_digital_invoice_no_flags(tmp_path):
    p = _pdf(tmp_path, "clean.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == [] and res.module == "invisible"


def test_pure_ocr_sandwich_no_flags(tmp_path):
    # Scanned page: invisible OCR layer only, zero painted text.
    p = _pdf(tmp_path, "sandwich.pdf",
             invisible=[(72, 100, "Facture INV-2026-001", 11),
                        (72, 140, "Total: 1 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == []


def test_invisible_dominant_hybrid_page_skipped(tmp_path):
    # A stamp of painted text on an otherwise invisible-layer page:
    # invisible area >= 50% of text area -> page skipped by the gate.
    p = _pdf(tmp_path, "hybrid.pdf",
             painted=[(72, 60, "PAYE", 10)],
             invisible=[(72, 100, "Facture originale numero 12345", 11),
                        (72, 140, "Total: 1 025,00", 12),
                        (72, 180, "TVA: 205,00", 12),
                        (72, 220, "Client: Dupont SARL", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == []


def test_never_raises_on_garbage(tmp_path):
    p = tmp_path / "garbage.pdf"
    p.write_bytes(b"%PDF-1.7 not really a pdf")
    res = analyze_invisible(_PdfData(str(p)))
    assert res.module == "invisible" and res.confidence <= 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.modules.invisible'`.

- [ ] **Step 3: Write the minimal module**

Create `src/modules/invisible.py`:

```python
"""
Invisible-text mismatch detector (Module: invisible) — phase 7.

Detects Tr-mode-3 ("paint nothing") text that conflicts with, or hides from,
the visible page. See docs/superpowers/specs/2026-07-12-invisible-text-
mismatch-design.md. Invisible text is NOT suspicious per se (legit OCR
sandwich) — only conflicts/injections on digitally-authored pages are flagged.
"""
import logging
import re

import fitz

from src.models import Flag, ModuleResult
from src.modules.span_conflict import (
    span_text, amount_token, is_placeholder, pair_is_conflict, page_angle,
    INTERACTIVE_EDITOR_PRODUCERS,
)

logger = logging.getLogger(__name__)

SEVERITY_POINTS = {"low": 5, "medium": 15, "high": 30, "critical": 50}

# A page is analysed only when painted text exists AND invisible text is a
# minority of its text area — pure OCR sandwiches and hybrid scans are the
# own-OCR-reconciliation module's territory, not this one's (spec, gate).
INVISIBLE_DOMINANCE_MAX = 0.50
ROW_OVERLAP_MIN = 0.5

# IBAN shape after whitespace strip + uppercase (spec).
_IBAN_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}")

# Producers whose invisible layers may legitimately touch mixed pages
# (re-OCR tools). Damper: injection suppressed, conflict demoted (Task 6).
OCR_TOOL_PRODUCERS = ("ocrmypdf", "tesseract", "abbyy", "readiris",
                      "finereader", "naps2")


def _split_spans(page) -> tuple[list, list]:
    """Partition texttrace spans into (painted, invisible).

    Painted = fill/fill+stroke (types 0, 2); invisible = Tr 3 (type 3).
    Other types (stroke-only, clip variants) are ignored, matching overlay's
    convention. Implementation note from the spec: verify how get_texttrace()
    reports Tr 4-7; if they surface as type 3 they are covered for free.
    """
    painted, invisible = [], []
    for s in page.get_texttrace():
        t = s.get("type")
        if t in (0, 2):
            painted.append(s)
        elif t == 3:
            invisible.append(s)
    return painted, invisible


def _area(spans: list) -> float:
    total = 0.0
    for s in spans:
        r = fitz.Rect(s["bbox"])
        total += max(0.0, r.width) * max(0.0, r.height)
    return total


def _is_digital_native(painted: list, invisible: list) -> bool:
    """Gate: painted text exists and invisible text is an area minority."""
    if not painted:
        return False
    if not invisible:
        return True
    total = _area(painted) + _area(invisible)
    return total > 0 and (_area(invisible) / total) < INVISIBLE_DOMINANCE_MAX


def analyze_invisible(pdf_data) -> ModuleResult:
    """Flag invisible text that conflicts with or hides from the visible page.

    Never raises: malformed/encrypted/textless PDFs yield a partial
    ModuleResult with reduced confidence instead of an exception (CLAUDE.md).
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="invisible", flags=[], score=100,
                            confidence=0.5)

    flags: list[Flag] = []
    confidence = 1.0

    doc = None
    try:
        doc = fitz.open(pdf_data.file_path)
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                painted, invisible = _split_spans(page)
                if not _is_digital_native(painted, invisible):
                    continue
                # Detection signals land here in later tasks.
            except Exception as exc:  # one bad page must not sink the module
                logger.warning("invisible: page %d failed: %s", page_num, exc)
                confidence = min(confidence, 0.5)
    except Exception as exc:
        logger.warning("invisible: analysis failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = 0.5
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    return ModuleResult(module="invisible", flags=flags,
                        score=max(0, score), confidence=confidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/invisible.py tests/test_invisible.py
git commit -m "feat(invisible): module skeleton with digital-native page gate"
```

---

### Task 3: `INVISIBLE_TEXT_VALUE_CONFLICT` (high, SOFT)

**Files:**
- Modify: `src/modules/invisible.py`
- Test: `tests/test_invisible.py`

**Interfaces:**
- Consumes: `_split_spans`, `_is_digital_native` (Task 2), `pair_is_conflict`/`page_angle`/`amount_token`/`is_placeholder`/`span_text` (Task 1).
- Produces: `_value_token(text: str) -> str | None`; `_find_conflicts(painted: list, invisible: list, angle: float) -> list[dict]` (each dict: `{"hidden": str, "visible": str, "bbox": tuple}`); the emitted Flag carries `details={"count": int, "instances": [{"page": int, "hidden": str, "visible": str, "bbox": tuple}]}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_invisible.py`:

```python
def test_conflicting_invisible_amount_flagged(tmp_path):
    # Visible 1 025,00 with an invisible 2 025,00 on the same row/size.
    # Label and amount are SEPARATE spans (how generators typically emit
    # them): pair_is_conflict requires matching char classes, so the painted
    # comparand must be the bare numeric span, not "Total: 1 025,00".
    p = _pdf(tmp_path, "conflict.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total:", 12),
                      (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT"]
    inst = res.flags[0].details["instances"][0]
    assert "2" in inst["hidden"] and inst["page"] == 0


def test_invisible_placeholder_not_flagged(tmp_path):
    p = _pdf(tmp_path, "placeholder.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "0,00", 12)])
    assert analyze_invisible(_PdfData(p)).flags == []


def test_size_mismatch_not_a_conflict(tmp_path):
    # >10% size difference -> different fields, not a shadowed value.
    p = _pdf(tmp_path, "sizes.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 7)])
    assert analyze_invisible(_PdfData(p)).flags == []


def test_identical_invisible_text_not_a_conflict(tmp_path):
    # Same normalized value (a legit duplicated layer) -> no flag.
    p = _pdf(tmp_path, "same.pdf",
             painted=[(72, 140, "1 025,00", 12)],
             invisible=[(72, 140, "1025,00", 12)])
    assert analyze_invisible(_PdfData(p)).flags == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: `test_conflicting_invisible_amount_flagged` FAILS (no flags emitted); the three negatives PASS vacuously (that is expected — they pin behavior against regressions introduced in this task).

- [ ] **Step 3: Implement the conflict signal**

Add to `src/modules/invisible.py` (below `_is_digital_native`):

```python
def _value_token(text: str) -> str | None:
    """A monetary amount or IBAN-shaped token in `text`, or None."""
    tok = amount_token(text)
    if tok:
        return tok
    m = _IBAN_RE.search("".join(text.split()).upper())
    return m.group() if m else None


def _rows_overlap(b1: fitz.Rect, b2: fitz.Rect) -> bool:
    """Vertical overlap >= ROW_OVERLAP_MIN of the smaller span height."""
    overlap = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    smaller = min(b1.height, b2.height)
    return smaller > 0 and (overlap / smaller) >= ROW_OVERLAP_MIN


def _near_horizontally(inv: fitz.Rect, pt: fitz.Rect, painted_span: dict) -> bool:
    """Horizontal overlap, or a gap under the painted span's glyph advance."""
    gap = max(pt.x0 - inv.x1, inv.x0 - pt.x1)
    if gap <= 0:
        return True
    nchars = max(1, len(painted_span.get("chars", [])))
    advance = pt.width / nchars
    return gap < advance


def _find_conflicts(painted: list, invisible: list, angle: float) -> list[dict]:
    """Invisible value tokens that disagree with painted text on their row."""
    hits = []
    for inv in invisible:
        hidden = span_text(inv)
        if not _value_token(hidden) or is_placeholder(hidden):
            continue
        ibox = fitz.Rect(inv["bbox"])
        for p in painted:
            pbox = fitz.Rect(p["bbox"])
            if not _rows_overlap(ibox, pbox):
                continue
            if not _near_horizontally(ibox, pbox, p):
                continue
            if pair_is_conflict(inv, p, angle):
                hits.append({"hidden": hidden.strip(),
                             "visible": span_text(p).strip(),
                             "bbox": tuple(inv["bbox"])})
                break                      # one conflict per invisible span
    return hits
```

In `analyze_invisible`, replace the `# Detection signals land here` comment with instance collection, and emit the aggregated flag after the page loop:

```python
                conflict_instances: list[dict]  # declared BEFORE the try block
```

Concretely: before `doc = None` add `conflict_instances: list[dict] = []`; inside the page loop after the gate add:

```python
                angle = page_angle(page)
                for hit in _find_conflicts(painted, invisible, angle):
                    conflict_instances.append({"page": page_num, **hit})
```

and after the `finally` block, before the score computation:

```python
    if conflict_instances:
        n = len(conflict_instances)
        flags.append(Flag(
            severity="high",
            code="INVISIBLE_TEXT_VALUE_CONFLICT",
            message=(f"{n} invisible text value(s) disagree with the visible "
                     f"page (e.g. hidden '{conflict_instances[0]['hidden']}' "
                     f"vs visible '{conflict_instances[0]['visible']}')"),
            details={"count": n, "instances": conflict_instances},
        ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/invisible.py tests/test_invisible.py
git commit -m "feat(invisible): flag invisible values conflicting with visible text"
```

---

### Task 4: `_CONFIRMED` escalation (critical, HARD)

**Files:**
- Modify: `src/modules/invisible.py`, `src/evidence.py:23-35`
- Test: `tests/test_invisible.py`

**Interfaces:**
- Consumes: `INTERACTIVE_EDITOR_PRODUCERS` (Task 1), conflict machinery (Task 3).
- Produces: `_producer_haystack(pdf_data) -> str` (lower-cased `producer + creator`); the CONFIRMED code registered in `evidence.HARD_NEGATIVE_CODES`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_invisible.py`:

```python
def test_conflict_with_editor_producer_is_confirmed(tmp_path):
    p = _pdf(tmp_path, "confirmed.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)],
             producer="Adobe Acrobat Pro DC 24.0")
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED"]
    assert res.flags[0].severity == "critical"


def test_confirmed_code_is_hard_negative():
    from src.evidence import HARD_NEGATIVE_CODES
    assert "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED" in HARD_NEGATIVE_CODES
    # The SOFT base code must NOT be hard.
    assert "INVISIBLE_TEXT_VALUE_CONFLICT" not in HARD_NEGATIVE_CODES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: both new tests FAIL (base code emitted instead of CONFIRMED; registry miss).

- [ ] **Step 3: Implement escalation + registry entry**

Add to `src/modules/invisible.py`:

```python
def _producer_haystack(pdf_data) -> str:
    """Lower-cased Producer+Creator, '' on any failure (never raises)."""
    try:
        doc = fitz.open(pdf_data.file_path)
        try:
            meta = doc.metadata or {}
            return (f"{meta.get('producer') or ''} "
                    f"{meta.get('creator') or ''}").casefold()
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("invisible: metadata read failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        return ""
```

In `analyze_invisible`, before `doc = None` add:

```python
    haystack = _producer_haystack(pdf_data)
    producer_is_editor = any(p in haystack for p in INTERACTIVE_EDITOR_PRODUCERS)
```

and replace the flag-emission block from Task 3 with the code-swap (exactly one of the two codes ever fires — overlay convention):

```python
    if conflict_instances:
        n = len(conflict_instances)
        if producer_is_editor:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED", "critical"
        else:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT", "high"
        flags.append(Flag(
            severity=severity,
            code=code,
            message=(f"{n} invisible text value(s) disagree with the visible "
                     f"page (e.g. hidden '{conflict_instances[0]['hidden']}' "
                     f"vs visible '{conflict_instances[0]['visible']}')"),
            details={"count": n, "instances": conflict_instances,
                     "producer_is_editor": producer_is_editor},
        ))
```

In `src/evidence.py`, add one line inside the `HARD_NEGATIVE_CODES` frozenset literal (after the `OVERLAY_*_CONFIRMED` entries):

```python
    "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED",      # Phase 7: Tr-3 shadow value + editor Producer (C2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invisible.py tests/test_scoring.py -v` (scoring tests guard the evidence registry; if `tests/test_scoring.py` does not exist, run `.venv/bin/python -m pytest tests/ -q`)
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/invisible.py src/evidence.py tests/test_invisible.py
git commit -m "feat(invisible): HARD _CONFIRMED escalation with editor-producer corroborator"
```

---

### Task 5: `INVISIBLE_TEXT_INJECTION` (medium, SOFT)

**Files:**
- Modify: `src/modules/invisible.py`
- Test: `tests/test_invisible.py`

**Interfaces:**
- Consumes: `_value_token`, `_rows_overlap` (Task 3).
- Produces: `_find_injections(painted: list, invisible: list) -> list[dict]` (each dict: `{"hidden": str, "bbox": tuple}`); Flag details shaped like Task 3's.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_invisible.py`:

```python
def test_invisible_iban_in_whitespace_flagged(tmp_path):
    # Three painted lines keep the invisible IBAN well under the 50%
    # dominance gate (area ratio has margin, not a knife's edge).
    p = _pdf(tmp_path, "inject.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12),
                      (72, 180, "TVA (20%): 205,00", 12)],
             invisible=[(72, 400, "FR7630006000011234567890189", 10)])
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_INJECTION"]
    assert res.flags[0].severity == "medium"


def test_invisible_prose_in_whitespace_not_flagged(tmp_path):
    # Non-value invisible text (e.g. accessibility artifacts) stays silent.
    p = _pdf(tmp_path, "prose.pdf",
             painted=[(72, 140, "Total: 1 025,00", 12)],
             invisible=[(72, 400, "page un sur deux", 10)])
    assert analyze_invisible(_PdfData(p)).flags == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: `test_invisible_iban_in_whitespace_flagged` FAILS (no flag); the prose negative passes vacuously.

- [ ] **Step 3: Implement the injection signal**

Add to `src/modules/invisible.py`:

```python
def _find_injections(painted: list, invisible: list) -> list[dict]:
    """Invisible value-bearing spans on rows with NO painted text at all."""
    hits = []
    for inv in invisible:
        hidden = span_text(inv)
        if not _value_token(hidden) or is_placeholder(hidden):
            continue
        ibox = fitz.Rect(inv["bbox"])
        if any(_rows_overlap(ibox, fitz.Rect(p["bbox"])) for p in painted):
            continue                      # row has visible text -> Task 3's job
        hits.append({"hidden": hidden.strip(), "bbox": tuple(inv["bbox"])})
    return hits
```

In `analyze_invisible`: add `injection_instances: list[dict] = []` next to `conflict_instances`; inside the page loop after the conflict scan add:

```python
                for hit in _find_injections(painted, invisible):
                    injection_instances.append({"page": page_num, **hit})
```

and after the conflict flag emission add:

```python
    if injection_instances:
        n = len(injection_instances)
        flags.append(Flag(
            severity="medium",
            code="INVISIBLE_TEXT_INJECTION",
            message=(f"{n} invisible value-bearing text(s) hidden in blank "
                     f"areas (e.g. '{injection_instances[0]['hidden']}')"),
            details={"count": n, "instances": injection_instances},
        ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/invisible.py tests/test_invisible.py
git commit -m "feat(invisible): flag invisible values injected into whitespace"
```

---

### Task 6: OCR-producer damper

**Files:**
- Modify: `src/modules/invisible.py`
- Test: `tests/test_invisible.py`

**Interfaces:**
- Consumes: `OCR_TOOL_PRODUCERS` (Task 2 constant), `_producer_haystack` (Task 4).
- Produces: final flag-emission behavior — with an OCR producer: injection suppressed entirely, conflict demoted to `medium` (still SOFT; CONFIRMED is structurally unreachable because OCR tools are absent from `INTERACTIVE_EDITOR_PRODUCERS`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_invisible.py`:

```python
def test_ocr_producer_demotes_conflict_to_medium(tmp_path):
    p = _pdf(tmp_path, "ocr_conflict.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)],
             producer="ocrmypdf 16.4.0 / Tesseract OCR")
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT"]
    assert res.flags[0].severity == "medium"


def test_ocr_producer_suppresses_injection(tmp_path):
    p = _pdf(tmp_path, "ocr_inject.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12),
                      (72, 180, "TVA (20%): 205,00", 12)],
             invisible=[(72, 400, "FR7630006000011234567890189", 10)],
             producer="ABBYY FineReader 15")
    assert analyze_invisible(_PdfData(p)).flags == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: both FAIL (conflict stays high; injection still emitted).

- [ ] **Step 3: Implement the damper**

In `analyze_invisible`, next to `producer_is_editor` add:

```python
    producer_is_ocr = any(p in haystack for p in OCR_TOOL_PRODUCERS)
```

Adjust the conflict emission's severity selection:

```python
        if producer_is_editor:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED", "critical"
        elif producer_is_ocr:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT", "medium"
        else:
            code, severity = "INVISIBLE_TEXT_VALUE_CONFLICT", "high"
```

and guard the injection emission:

```python
    if injection_instances and not producer_is_ocr:
```

Also record the damper in the conflict flag's details: add `"producer_is_ocr": producer_is_ocr` to the details dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: 14 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/invisible.py tests/test_invisible.py
git commit -m "feat(invisible): OCR-producer damper (suppress injection, demote conflict)"
```

---

### Task 7: Analyzer registration, module weight, corpus assertion

**Files:**
- Modify: `src/analyzer.py:29,182` (import + `_run` registration), `src/scoring.py:60-71` (`MODULE_WEIGHTS`)
- Test: `tests/test_invisible.py` (corpus assertion), full suite

**Interfaces:**
- Consumes: `analyze_invisible` (Task 2).
- Produces: the module runs in production under crash isolation; UI weight entry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_invisible.py`:

```python
import glob
import os

_LEGIT = sorted(glob.glob("samples/legitimate/**/*.pdf", recursive=True))


@pytest.mark.skipif(not _LEGIT, reason="local corpus not present")
@pytest.mark.parametrize("path", _LEGIT, ids=os.path.basename)
def test_no_invisible_flags_on_legit_corpus(path):
    """Zero-FP hard gate: no legitimate sample may emit any INVISIBLE_* code."""
    res = analyze_invisible(_PdfData(path))
    assert res.flags == [], f"{path} -> {[f.code for f in res.flags]}"


def test_module_registered_in_analyzer():
    import inspect
    import src.analyzer as analyzer
    src_text = inspect.getsource(analyzer)
    assert "analyze_invisible" in src_text and '_run("invisible"' in src_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invisible.py -v`
Expected: `test_module_registered_in_analyzer` FAILS; corpus tests PASS (or SKIP if no local corpus) — a corpus failure here means a real FP: STOP and report it to the user, do not tune thresholds unilaterally (F1/F2 sign-off rule).

- [ ] **Step 3: Register the module**

In `src/analyzer.py` next to the overlay import (line 29):

```python
from src.modules.invisible import analyze_invisible
```

and next to the overlay registration (line 182):

```python
        _run("invisible", lambda: analyze_invisible(pdf_data))                    # Tr-3 shadow values
```

In `src/scoring.py` `MODULE_WEIGHTS` (UI bars only — scoring is evidence-ledger based):

```python
    "invisible": 0.9,   # Tr-3 invisible-text mismatch (vector-level, deterministic)
```

- [ ] **Step 4: Run the FULL suite + corpus band gate**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: everything green, including `tests/test_corpus_bands.py`. Any corpus band shift is a stop-and-report event.

- [ ] **Step 5: Commit**

```bash
git add src/analyzer.py src/scoring.py tests/test_invisible.py
git commit -m "feat(invisible): register module in analyzer with UI weight"
```

---

## Post-implementation

- Adversarial review before merge (project convention), then log residual
  risks in `docs/superpowers/2026-07-10-audit-backlog.md` — at minimum the
  spec's accepted ones: Tr 4–7 encoding (verify during Task 2 and record the
  finding), size-mismatch evasion, scanned-page invisible text out of scope.
- Additional residual risk found at plan time: when a generator paints
  label+value as ONE span ("Total: 1 025,00"), its char class is "mixed" and
  `pair_is_conflict` rejects the pair against a numeric hidden span — Flag 1
  misses that shape. An amount-token fallback pass (mirroring overlay's
  embedded-span path) would close it; that is a spec change needing user
  sign-off, so log it rather than implement it.
- Do NOT merge into `phase6-robustness`; this branch targets `main` via PR
  after phase 6 lands.
