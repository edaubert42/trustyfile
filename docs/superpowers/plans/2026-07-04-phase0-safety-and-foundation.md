# Phase 0 — Safety & Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the live security/leak/crash holes, make PDF date handling timezone-correct, make extraction fail *closed* with an enriched single parse pass, green the 5 failing tests, and stand up a calibration harness — so the detection-engine redesign (Phases 1-5) builds on a safe, tested base.

**Architecture:** Small, isolated fixes to existing modules plus two new files (an enriched-extraction helper stays inside `pdf_extractor.py`; a new `scripts/eval_harness.py`). No behavioural redesign of scoring yet — that is Phase 1. Each task is independently testable via pytest.

**Tech Stack:** Python 3.11+ (dev venv is 3.12 at `.venv/`), PyMuPDF (`fitz`), pytest. Pure-pip only. No new dependencies in Phase 0.

## Global Constraints

- **The document never leaves the machine.** No network call may transmit the uploaded file or its extracted identifiers by default. (Copied from spec §1.)
- **Pure-pip, lightweight dependencies only.** Phase 0 adds **no** new dependency.
- **Git enabled (local only, no remote).** Work happens on branch `phase0-safety-and-foundation` off the baseline commit. Every task ends with a **full-suite regression run** (`pytest -q`) **and** a commit of that task's changes, giving an independent per-task rollback point. Do not add a remote or push.
- **Fail closed, never fail open.** Unanalyzable input must never yield a high trust score; it must surface an explicit extraction flag. (Spec §3.3.)
- **Run tests with the project venv:** `.venv/bin/python -m pytest …` (system python is 3.9 and lacks deps).
- **Baseline:** before starting, `.venv/bin/python -m pytest -q` reports **344 passed, 5 failed**. Phase 0 ends at **0 failed** with all new tests passing.

---

### Task 1: Timezone-aware PDF date parsing

**Why:** `parse_pdf_date` discards the timezone offset (`pdf_extractor.py:143-148`), so a creation date stored `+02'00'` and a mod date stored `Z` for the *same instant* read as a 2-hour "modification" → phantom `META_DOCUMENT_MODIFIED` critical. It also zero-pads short-but-valid dates (`D:2024` → invalid month `00` → silent `None`). Fix parsing to be timezone-aware and preserve the raw string; update the two sites that compare the parsed datetimes so no naive/aware `TypeError` is introduced.

**Files:**
- Modify: `src/extractors/pdf_extractor.py` (`parse_pdf_date` ~118-155; `PDFMetadata` ~48-55; `extract_metadata` ~177-186)
- Modify: `src/modules/metadata.py` (`check_dates` ~286-352)
- Modify: `src/modules/content.py` (`check_date_vs_creation` — compares a text date to `metadata.creation_date`)
- Test: `tests/test_pdf_extractor_dates.py` (create), `tests/test_metadata.py` (existing date tests must still pass)

**Interfaces:**
- Produces: `parse_pdf_date(s: str | None) -> datetime | None` now returns a **timezone-aware** datetime (UTC). New helper `_now_utc() -> datetime` in `pdf_extractor.py`. New `PDFMetadata` fields `creation_date_raw: str | None`, `mod_date_raw: str | None`.
- Consumes (downstream): `metadata.check_dates` and `content.check_date_vs_creation` now assume `creation_date`/`mod_date` are aware.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pdf_extractor_dates.py`:

```python
from datetime import timezone
from src.extractors.pdf_extractor import parse_pdf_date


def test_offset_normalized_to_utc():
    # 14:30:52 at +02:00 == 12:30:52 UTC
    dt = parse_pdf_date("D:20240115143052+02'00'")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0  # normalized to UTC
    assert (dt.hour, dt.minute) == (12, 30)


def test_same_instant_different_zones_are_equal():
    # Creation stored local +02:00, mod stored UTC 'Z' for the SAME instant
    creation = parse_pdf_date("D:20240115143052+02'00'")
    mod = parse_pdf_date("D:20240115123052Z")
    assert creation == mod  # no phantom modification


def test_short_form_year_only():
    # 'D:2024' is a spec-valid date meaning 2024-01-01
    dt = parse_pdf_date("D:2024")
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == (2024, 1, 1)


def test_malformed_returns_none():
    assert parse_pdf_date("D:garbage") is None
    assert parse_pdf_date(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pdf_extractor_dates.py -v`
Expected: FAIL (`test_offset_normalized_to_utc` — current code returns naive datetime with offset ignored; `test_short_form_year_only` — current zero-pad produces month 00 → None).

- [ ] **Step 3: Rewrite `parse_pdf_date` and add `_now_utc`**

In `src/extractors/pdf_extractor.py`, replace the `parse_pdf_date` function (currently ~118-155) with:

```python
from datetime import datetime, timezone, timedelta


def _now_utc() -> datetime:
    """Timezone-aware 'now' in UTC — compare only against aware datetimes."""
    return datetime.now(timezone.utc)


def parse_pdf_date(date_string: str | None) -> datetime | None:
    """
    Parse a PDF date string into a timezone-aware UTC datetime.

    PDF dates look like "D:20240115143052+01'00'" (D: prefix, YYYYMMDDHHmmSS,
    then an optional timezone: 'Z', or +/-HH'mm'). Shorter forms are valid and
    default the missing components (per ISO 32000): "D:2024" == 2024-01-01 00:00:00.

    Returns None only when the string is truly unparseable (itself a tamper signal
    the caller should surface — see extract_metadata's MALFORMED_DATE handling).
    """
    if not date_string:
        return None

    s = date_string.strip()
    if s.startswith("D:"):
        s = s[2:]

    # Split off timezone: 'Z', or +HH'mm' / -HH'mm' (apostrophes optional).
    tz = timezone.utc
    tzmatch = None
    import re
    m = re.search(r"(Z|[+\-]\d{2}'?\d{2}'?)$", s)
    if m:
        tzraw = m.group(1)
        s = s[: m.start()]
        if tzraw != "Z":
            sign = 1 if tzraw[0] == "+" else -1
            digits = tzraw[1:].replace("'", "")
            hh, mm = int(digits[:2]), int(digits[2:4])
            tz = timezone(sign * timedelta(hours=hh, minutes=mm))

    # Fill missing components with their spec defaults: year required, then
    # month=01, day=01, hour=00, min=00, sec=00.
    if len(s) < 4 or not s[:4].isdigit():
        logger.warning(f"Could not parse PDF date '{date_string}'")
        return None
    defaults = ["0001", "01", "01", "00", "00", "00"]  # year has no default
    fields = [s[0:4], s[4:6], s[6:8], s[8:10], s[10:12], s[12:14]]
    try:
        year = int(fields[0])
        month = int(fields[1]) if fields[1] else 1
        day = int(fields[2]) if fields[2] else 1
        hour = int(fields[3]) if fields[3] else 0
        minute = int(fields[4]) if fields[4] else 0
        second = int(fields[5]) if fields[5] else 0
        dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse PDF date '{date_string}'")
        return None
```

(The `defaults` list is documentation of the spec defaults; the code applies them inline via the `if fields[i]` guards.)

- [ ] **Step 4: Preserve raw date strings on `PDFMetadata`**

In `src/extractors/pdf_extractor.py`, add two fields to the `PDFMetadata` dataclass (after `mod_date`, ~49):

```python
    creation_date_raw: str | None = None
    mod_date_raw: str | None = None
```

In `extract_metadata` (~177), populate them from the raw dict:

```python
    metadata = PDFMetadata(
        creation_date=parse_pdf_date(raw.get("creationDate")),
        mod_date=parse_pdf_date(raw.get("modDate")),
        creation_date_raw=raw.get("creationDate"),
        mod_date_raw=raw.get("modDate"),
        producer=raw.get("producer"),
        creator=raw.get("creator"),
        author=raw.get("author"),
        title=raw.get("title"),
        subject=raw.get("subject"),
        keywords=raw.get("keywords"),
    )
```

- [ ] **Step 5: Fix aware/naive comparison sites**

In `src/modules/metadata.py` `check_dates` (~286-287), replace the naive `now`:

```python
    from datetime import timezone
    flags = []
    now = datetime.now(timezone.utc)
```

Both `creation_date` and `mod_date` here come from `parse_pdf_date` (now aware), so `creation_date > now + timedelta(days=1)` and `mod_date - creation_date` are aware-aware — safe. If either is passed in naive by a test, coerce at the top of `check_dates`:

```python
    def _aware(d):
        if d is not None and d.tzinfo is None:
            return d.replace(tzinfo=timezone.utc)
        return d
    creation_date = _aware(creation_date)
    mod_date = _aware(mod_date)
```

In `src/modules/content.py` `check_date_vs_creation`, the text-extracted date is naive while `metadata.creation_date` is now aware. At the point of comparison, coerce the metadata date to naive-UTC to compare like-with-like (text dates carry no zone):

```python
    if creation_date is not None and creation_date.tzinfo is not None:
        creation_date = creation_date.astimezone(timezone.utc).replace(tzinfo=None)
```

(Add `from datetime import timezone` to the imports if not present.)

- [ ] **Step 6: Run the new tests and the metadata suite**

Run: `.venv/bin/python -m pytest tests/test_pdf_extractor_dates.py tests/test_metadata.py -v`
Expected: new date tests PASS; existing metadata date tests still PASS (they build `datetime` values without tz; the `_aware` coercion keeps them working).

- [ ] **Step 7: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no NEW failures vs baseline (still the pre-existing 5 failures from other files, which later tasks fix).

---

### Task 2: Enriched, fail-closed extraction + container-anomaly flags

**Why:** Six modules each re-`fitz.open()` the file; `extract_pdf_data` raises on malformed PDFs (violating the project's own "never crash" rule), and `is_repaired`/`is_encrypted`/`is_pdf` — cheap high-value tamper/evasion signals — are discarded. Enrich `PDFData` once, fail closed, and surface container anomalies. (Spec §3.5, §4.1 container census.)

**Files:**
- Modify: `src/extractors/pdf_extractor.py` (`PDFData` dataclass ~58-84; `extract_pdf_data` ~215-262)
- Modify: `src/analyzer.py` (`analyze` ~97-190 — guard extraction, add size/page guard)
- Test: `tests/test_extraction_safety.py` (create)

**Interfaces:**
- Produces: `PDFData` gains `is_repaired: bool`, `is_encrypted: bool`, `is_pdf: bool`, `pdf_version: str | None`, `file_size_bytes: int`, `extraction_ok: bool`, `extraction_error: str | None`. `extract_pdf_data` **never raises** on a malformed PDF — it returns a `PDFData` with `extraction_ok=False` and empty text.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extraction_safety.py`:

```python
import pytest
from src.extractors.pdf_extractor import extract_pdf_data, PDFData


def test_garbage_file_does_not_raise(tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    data = extract_pdf_data(str(bad))
    assert isinstance(data, PDFData)
    assert data.extraction_ok is False
    assert data.extraction_error  # non-empty message


def test_valid_pdf_sets_flags(tmp_path):
    import fitz
    p = tmp_path / "ok.pdf"
    doc = fitz.open()
    doc.insert_page(0, "Hello invoice")
    doc.save(str(p))
    doc.close()
    data = extract_pdf_data(str(p))
    assert data.extraction_ok is True
    assert data.is_pdf is True
    assert data.is_encrypted is False
    assert data.file_size_bytes > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_extraction_safety.py -v`
Expected: FAIL — `extract_pdf_data` currently raises on garbage and `PDFData` has no `extraction_ok`.

- [ ] **Step 3: Extend the `PDFData` dataclass**

In `src/extractors/pdf_extractor.py`, add fields to `PDFData` (after `text_by_page`, ~80):

```python
    is_repaired: bool = False
    is_encrypted: bool = False
    is_pdf: bool = True
    pdf_version: str | None = None
    file_size_bytes: int = 0
    extraction_ok: bool = True
    extraction_error: str | None = None
```

- [ ] **Step 4: Make `extract_pdf_data` fail closed and populate flags**

Replace the body of `extract_pdf_data` (~239-262) with:

```python
    file_path = Path(file_path)
    file_size = file_path.stat().st_size if file_path.exists() else 0

    # Hash first — works even if the PDF is corrupt.
    file_hash = calculate_file_hash(file_path)

    try:
        with fitz.open(file_path) as doc:
            is_encrypted = bool(getattr(doc, "is_encrypted", False))
            # Encrypted docs need a password before text/metadata are readable.
            if is_encrypted:
                try:
                    doc.authenticate("")  # try empty password
                except Exception:
                    pass
            metadata, raw_metadata = extract_metadata(doc)
            text_by_page = extract_text(doc)
            return PDFData(
                file_path=str(file_path),
                file_hash=file_hash,
                page_count=len(doc),
                metadata=metadata,
                raw_metadata=raw_metadata,
                text_by_page=text_by_page,
                is_repaired=bool(getattr(doc, "is_repaired", False)),
                is_encrypted=is_encrypted,
                is_pdf=bool(getattr(doc, "is_pdf", True)),
                pdf_version=raw_metadata.get("format"),
                file_size_bytes=file_size,
                extraction_ok=True,
            )
    except Exception as e:  # malformed / non-PDF / unreadable
        logger.warning(f"Extraction failed for {file_path}: {e}")
        return PDFData(
            file_path=str(file_path),
            file_hash=file_hash,
            page_count=0,
            metadata=PDFMetadata(),
            raw_metadata={},
            text_by_page=[],
            is_pdf=False,
            file_size_bytes=file_size,
            extraction_ok=False,
            extraction_error=str(e),
        )
```

- [ ] **Step 5: Add a size/page guard in the analyzer**

In `src/analyzer.py` `analyze`, after `pdf_data = extract_pdf_data(file_path)` (~109), add:

```python
        # Fail closed: unreadable input must not score as trustworthy.
        if not pdf_data.extraction_ok:
            logger.warning("Extraction failed — returning UNVERIFIABLE-equivalent result")
            unverifiable = ModuleResult(
                module="extraction",
                flags=[Flag(
                    severity="high",
                    code="EXTRACTION_FAILED",
                    message="Document could not be parsed as a valid PDF",
                    details={"error": pdf_data.extraction_error},
                )],
                score=0,
                confidence=1.0,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            return create_analysis_result(
                file_hash=pdf_data.file_hash,
                module_results=[unverifiable],
                analysis_time_ms=elapsed_ms,
                document_type="unknown",
            )
```

Add the import at the top of `analyzer.py` if not present: `from src.models import AnalysisResult, ModuleResult, Flag`.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_safety.py -v`
Expected: PASS.

- [ ] **Step 7: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures (still the pre-existing 5).

---

### Task 3: Guard the scoring crash

**Why:** `_is_scanner_pipeline_critical` does `float(hours)` on an attacker-controlled `details` value (`scoring.py:287`); a non-numeric value raises `ValueError` and kills the entire analysis.

**Files:**
- Modify: `src/scoring.py` (`_is_scanner_pipeline_critical`, ~280-287)
- Test: `tests/test_scoring.py` (append)

**Interfaces:** none changed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scoring.py`:

```python
def test_scanner_pipeline_handles_non_numeric_hours():
    from src.scoring import create_analysis_result
    from src.models import ModuleResult, Flag
    bad = ModuleResult(
        module="structure",
        flags=[Flag(
            severity="critical",
            code="META_DOCUMENT_MODIFIED",
            message="x",
            details={"time_difference_hours": "not-a-number"},
        )],
        score=0, confidence=1.0,
    )
    # Must not raise, even for an identity document (the branch that reads hours).
    result = create_analysis_result("hash", [bad], document_type="identity")
    assert result.trust_score is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_scoring.py::test_scanner_pipeline_handles_non_numeric_hours -v`
Expected: FAIL with `ValueError: could not convert string to float`.

- [ ] **Step 3: Guard the conversion**

In `src/scoring.py`, replace the last line of `_is_scanner_pipeline_critical` (~286-287):

```python
            hours = (f.details or {}).get("time_difference_hours", 999)
            try:
                return float(hours) < 0.5  # under 30 minutes
            except (TypeError, ValueError):
                return False  # unparseable → treat as a genuine critical, not exempt
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/test_scoring.py::test_scanner_pipeline_handles_non_numeric_hours -v`
Expected: PASS.

- [ ] **Step 5: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures.

---

### Task 4: Remove the EU DSS document upload (constraint violation)

**Why:** `verify_signature_dss` base64-encodes the whole PDF and POSTs it to `ec.europa.eu` (`structure.py:1048-1101`), in the default path, twice per analysis. This violates the hard "document never leaves the machine" constraint. Phase 0 replaces it with a **local, offline** stub that reports only *presence* of a signature (no verification, no upload); real offline validation via pyHanko lands in Phase 2.

**Files:**
- Modify: `src/modules/structure.py` (`verify_signature_dss` ~1048-1200; callers at ~1256-1259 and ~1762-1764)
- Test: `tests/test_structure_no_upload.py` (create)

**Interfaces:**
- Produces: `verify_signature_dss(pdf_path: str) -> dict` still returns a dict with keys `has_signature`, `is_valid`, `is_trusted`, `signer`, `signature_level`, but **never makes a network request**. `is_trusted` is always `False` in Phase 0 (nothing is verified yet).

- [ ] **Step 1: Write the failing test**

Create `tests/test_structure_no_upload.py`:

```python
import src.modules.structure as structure


def test_verify_signature_makes_no_network_call(monkeypatch, tmp_path):
    import fitz
    p = tmp_path / "doc.pdf"
    doc = fitz.open(); doc.insert_page(0, "hello"); doc.save(str(p)); doc.close()

    called = {"hit": False}

    def boom(*a, **k):
        called["hit"] = True
        raise AssertionError("network call attempted — document must never leave the machine")

    # Any attempt to use requests must fail the test.
    import requests
    monkeypatch.setattr(requests, "post", boom)

    info = structure.verify_signature_dss(str(p))
    assert called["hit"] is False
    assert info["is_trusted"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_structure_no_upload.py -v`
Expected: FAIL — current `verify_signature_dss` calls `requests.post`.

- [ ] **Step 3: Replace `verify_signature_dss` with a local stub**

In `src/modules/structure.py`, replace the entire `verify_signature_dss` function (from `def verify_signature_dss` ~1048 through its final `return`/`except`, up to but not including the next top-level `def`) with:

```python
def verify_signature_dss(pdf_path: str) -> dict:
    """
    Report presence of a digital signature WITHOUT uploading the document.

    Phase 0: the previous implementation POSTed the whole PDF to the EU DSS
    demo API, violating the 'document never leaves the machine' constraint.
    Real offline cryptographic validation (ByteRange, chain, post-signing diff)
    arrives in Phase 2 via pyHanko. Until then we only detect presence locally
    and never assert trust.
    """
    present = False
    try:
        present = has_digital_signature(pdf_path)
    except Exception as e:
        logger.error(f"Signature presence check failed: {e}")
    return {
        "has_signature": present,
        "is_valid": False,
        "is_trusted": False,
        "is_fully_valid": False,
        "signer": None,
        "signature_level": "",
        "note": "offline — signature present but not cryptographically verified",
    }
```

- [ ] **Step 4: Confirm callers still work**

The two callers (`check_incremental_updates` ~1256-1259 and `analyze_structure` ~1762-1764) read `.get("is_trusted")` / `.get("is_fully_valid")` / `.get("signer")` — all still present in the returned dict (now falsy). No caller change needed; verify by reading those lines.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_structure_no_upload.py tests/test_structure.py -v`
Expected: new test PASS; existing structure tests still PASS.

- [ ] **Step 6: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures.

---

### Task 5: Remove the anyapi.io IBAN upload (PII leak)

**Why:** the IBAN loop sends full IBANs to the commercial `anyapi.io` service (`external.py:771-816`) and ignores the local checksum already computed by `content.validate_iban_checksum`. Phase 0 replaces the network call with **local mod-97 validation** so no banking data leaves the machine. (Richer local bank-registry resolution is Phase 4.)

**Files:**
- Modify: `src/modules/external.py` (`verify_iban_api` and the IBAN block ~771-816)
- Test: `tests/test_external_iban_local.py` (create)

**Interfaces:**
- Produces: `verify_iban_api(iban: str, api_key: str | None = None) -> tuple[dict, str | None]` now validates **locally** (mod-97) and never makes a network request. Returns `({"valid": bool, "iban": str, "countryCode": str, "bic": None}, None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_external_iban_local.py`:

```python
import src.modules.external as external


def test_iban_verification_is_local(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network call attempted for IBAN")))
    # Valid French IBAN (mod-97 == 1)
    result, err = external.verify_iban_api("FR7630006000011234567890189", "unused-key")
    assert err is None
    assert result["valid"] is True
    # Invalid IBAN
    bad, err2 = external.verify_iban_api("FR7630006000011234567890188", "unused-key")
    assert bad["valid"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_external_iban_local.py -v`
Expected: FAIL — current `verify_iban_api` calls `requests.get` against anyapi.io.

- [ ] **Step 3: Replace `verify_iban_api` with local mod-97**

In `src/modules/external.py`, replace the entire `verify_iban_api` function body with a local implementation:

```python
def verify_iban_api(iban: str, api_key: str | None = None) -> tuple[dict, str | None]:
    """
    Validate an IBAN LOCALLY (ISO 7064 mod-97). No network call — Phase 0
    removes the anyapi.io upload that leaked banking data off the machine.
    `api_key` is accepted for signature compatibility and ignored.
    """
    cleaned = "".join(iban.split()).upper()
    try:
        # Move first 4 chars to the end, convert letters A=10..Z=35, mod 97 == 1.
        rearranged = cleaned[4:] + cleaned[:4]
        digits = "".join(str(int(c, 36)) if c.isalpha() else c for c in rearranged)
        valid = int(digits) % 97 == 1
    except (ValueError, TypeError):
        valid = False
    return (
        {"valid": valid, "iban": cleaned, "countryCode": cleaned[:2], "bic": None},
        None,
    )
```

(This mirrors `content.validate_iban_checksum`; Phase 3 will consolidate onto `python-stdnum`.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_external_iban_local.py -v`
Expected: PASS.

- [ ] **Step 5: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures.

---

### Task 6: Escape attacker-controlled strings in the UI (stored XSS)

**Why:** PDF metadata (Producer, Creator, editor, signer) and the filename are interpolated into `st.markdown(..., unsafe_allow_html=True)` **unescaped** (`app.py:1054-1102`). A Producer of `<img src=x onerror=alert(1)>` executes in the viewer's browser. Escape every attacker-controlled value, in both element text and `title="..."` attributes.

**Files:**
- Modify: `app.py` (imports ~10-16; file-properties block ~1002-1102)
- Test: `tests/test_app_escaping.py` (create) — tests the pure helper, not Streamlit.

**Interfaces:**
- Produces: module-level helper `esc(s) -> str` in `app.py` wrapping `html.escape(str(s or ""), quote=True)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_escaping.py`:

```python
def test_esc_neutralizes_html():
    import app
    out = app.esc('<img src=x onerror="alert(1)">')
    assert "<img" not in out
    assert "&lt;img" in out
    assert '"' not in out or "&quot;" in out


def test_esc_handles_none():
    import app
    assert app.esc(None) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_escaping.py -v`
Expected: FAIL — `app.esc` does not exist.

- [ ] **Step 3: Add the helper**

In `app.py`, add near the top imports (after line ~16):

```python
import html


def esc(s) -> str:
    """Escape a value for safe interpolation into unsafe_allow_html markdown."""
    return html.escape(str(s if s is not None else ""), quote=True)
```

- [ ] **Step 4: Wrap every attacker-controlled interpolation**

In the file-properties block, escape the values. Change these lines (~1057-1100). Filename row:

```python
                <span style="{value_style}" title="{esc(uploaded_file.name)}">{esc(uploaded_file.name)}</span>
```

Producer row (~1070):

```python
                <span style="{value_style}" title="{esc(producer)}">{esc(producer)}</span>
```

Creator row (~1080):

```python
                <span style="{value_style}" title="{esc(creator)}">{esc(creator)}</span>
```

Editor row (~1089):

```python
                <span style="{value_style}" title="{esc(editor_name)}">{esc(editor_name)}</span>
```

Signer row (~1096):

```python
                <span style="{value_style}" title="{esc(sig)}">{esc(sig)}</span>
```

Leave `result.file_hash` (hex, safe) and `format_date(...)` (formatted date, safe) as-is, but wrapping them in `esc()` is harmless and consistent — apply if trivial.

- [ ] **Step 5: Run the escaping test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_app_escaping.py -v`
Expected: PASS.
Then: `.venv/bin/python -c "import app"` — Expected: no error (module imports cleanly).

- [ ] **Step 6: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures.

---

### Task 7: Green failing test — canva producer classification

**Why:** `test_highly_suspicious_flagged_high[canva]` fails because `canva` is in both `HIGHLY_SUSPICIOUS_PRODUCERS` (expected `META_ONLINE_CONVERTER`) and the in-function `IMAGE_EDITORS` list (which returns first with `META_IMAGE_EDITOR`). This is a *stale test*: treating Canva as an image editor is the intended, more-accurate classification. Fix by removing `canva` from `HIGHLY_SUSPICIOUS_PRODUCERS` (and `SUSPICIOUS_PRODUCERS`) so the taxonomy is unambiguous — Canva is an image/design editor, detected by `META_IMAGE_EDITOR`.

**Files:**
- Modify: `src/modules/metadata.py` (`SUSPICIOUS_PRODUCERS` ~30-60, `HIGHLY_SUSPICIOUS_PRODUCERS` ~63-70)
- Test: `tests/test_metadata.py` (add an explicit canva → image-editor assertion)

**Interfaces:** `HIGHLY_SUSPICIOUS_PRODUCERS` and `SUSPICIOUS_PRODUCERS` no longer contain `"canva"`.

- [ ] **Step 1: Write the intended-behaviour test**

Append to `tests/test_metadata.py`:

```python
def test_canva_classified_as_image_editor():
    from src.modules.metadata import check_producer
    flags = check_producer("Canva", None)
    assert any(f.code == "META_IMAGE_EDITOR" for f in flags)
    assert not any(f.code == "META_ONLINE_CONVERTER" for f in flags)
```

- [ ] **Step 2: Run both tests to see current state**

Run: `.venv/bin/python -m pytest "tests/test_metadata.py::TestCheckProducerConverters::test_highly_suspicious_flagged_high[canva]" tests/test_metadata.py::test_canva_classified_as_image_editor -v`
Expected: parametrized `[canva]` FAILS (asserts `META_ONLINE_CONVERTER`); new test PASSES (canva already hits image-editor path). This confirms the taxonomy conflict.

- [ ] **Step 3: Remove `canva` from the converter lists**

In `src/modules/metadata.py`, delete the `"canva",` entry from `SUSPICIOUS_PRODUCERS` (the "Design tools" comment + line ~42-43) and from `HIGHLY_SUSPICIOUS_PRODUCERS` (~69). Canva remains in the in-function `IMAGE_EDITORS` list (`check_producer` ~182), so it is still flagged — as `META_IMAGE_EDITOR`.

- [ ] **Step 4: Run the parametrized suite**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -v`
Expected: PASS — `HIGHLY_SUSPICIOUS_PRODUCERS` no longer includes canva, so the parametrized test never runs the canva case; the new explicit test passes.

- [ ] **Step 5: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: failures now 4 (canva fixed).

---

### Task 8: Green failing tests — screenshot tolerance & compression threshold (stale tests)

**Why:** `test_within_tolerance` expects a 30px delta to match a screen resolution, but the tolerance was deliberately tightened from 50→20px to kill an A4-at-90PPI false positive (`images.py:152-156`). `test_heavy_compression_flagged` expects a 60:1 ratio to flag, but the threshold was deliberately raised from 60→80:1 (`images.py:273`). Both are *stale tests contradicting intentional FP-reduction changes* — update the tests to the current intended thresholds.

**Files:**
- Modify: `tests/test_images.py` (`test_within_tolerance` ~93-99, `test_heavy_compression_flagged` ~180-188)

**Interfaces:** none — test-only change.

- [ ] **Step 1: Update `test_within_tolerance` to the 20px contract**

In `tests/test_images.py`, replace the body of `test_within_tolerance` with a within-20px example plus a beyond-20px negative case:

```python
    def test_within_tolerance(self):
        """Within the tightened 20px tolerance should match; beyond it should not."""
        # 1920x1080 + 10px each way → still a screenshot
        images = [make_image(width=1930, height=1075)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 1

    def test_beyond_tolerance_not_flagged(self):
        """A4 at 90 PPI (744x1052) must NOT match iPad (768x1024) — the FP that drove 50→20px."""
        images = [make_image(width=744, height=1052)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 0
```

- [ ] **Step 2: Update `test_heavy_compression_flagged` to the 80:1 threshold**

Replace its image construction so the ratio exceeds 80:1 (1000×1000×3 = 3,000,000 bytes; 25,000 bytes → 120:1):

```python
    def test_heavy_compression_flagged(self):
        """
        1000x1000 RGB = 3,000,000 bytes uncompressed.
        File 25,000 bytes → ratio 120:1 → exceeds the 80:1 threshold → flagged.
        """
        images = [make_image(
            width=1000, height=1000,
            filter="DCTDecode",
            size_bytes=25000,
        )]
        flags = check_heavy_compression(images)
        assert len(flags) == 1
```

- [ ] **Step 3: Run the images suite**

Run: `.venv/bin/python -m pytest tests/test_images.py -v`
Expected: PASS (both updated tests + the new negative case).

- [ ] **Step 4: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: failures now 2 (images fixed).

---

### Task 9: Green failing test — implement `CONTENT_ANACHRONISM_SERVICE`

**Why:** `test_service_after_invoice_flagged` expects a service-date-after-invoice-date anachronism flag. The module docstring promises it but only due/order checks exist (`content.py:658-713`). This is a *missing check the test correctly demands* and it aligns with spec §4.2. Implement it (service date should not be materially after the invoice date).

**Files:**
- Modify: `src/modules/content.py` (`check_date_logic` ~658-713)
- Test: `tests/test_content.py` (`test_service_after_invoice_flagged` already exists ~410-417)

**Interfaces:** `check_date_logic` now also emits `Flag(code="CONTENT_ANACHRONISM_SERVICE", severity="high")`.

- [ ] **Step 1: Confirm the existing test and its expectation**

Run: `.venv/bin/python -m pytest tests/test_content.py::TestCheckDateLogic::test_service_after_invoice_flagged -v`
Expected: FAIL (`assert 0 == 1`) — the flag is never produced.

- [ ] **Step 2: Add the service anachronism check**

In `src/modules/content.py` `check_date_logic`, add a `service_dates` lookup next to the others (~673-675):

```python
    service_dates = [d for d in dates if d.date_type == "service"]
```

and add this block just before `return flags` (~712):

```python
    # Check: Service date vs Invoice date
    # A service/delivery date materially AFTER the invoice date is an anachronism
    # (you cannot invoice for a service not yet rendered). 1-day tolerance.
    if invoice_dates and service_dates:
        invoice_date = invoice_dates[0].date
        for sd in service_dates:
            if sd.date > invoice_date + timedelta(days=1):
                flags.append(Flag(
                    severity="high",
                    code="CONTENT_ANACHRONISM_SERVICE",
                    message=(
                        f"Service date ({sd.date.strftime('%Y-%m-%d')}) is after "
                        f"invoice date ({invoice_date.strftime('%Y-%m-%d')})"
                    ),
                    details={
                        "invoice_date": invoice_date.isoformat(),
                        "service_date": sd.date.isoformat(),
                    },
                ))
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_content.py::TestCheckDateLogic -v`
Expected: PASS.

- [ ] **Step 4: Regression checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: failures now 1.

---

### Task 10: Green failing test — implement `CONTENT_MISSING_COMPANY_ID`

**Why:** `test_missing_company_id_flagged` expects a medium flag when a French invoice carries no SIRET/SIREN/VAT at all — legally impossible for a real French company. `check_legal_mentions` only flags *present-but-invalid* values, never absence (`content.py:1640-1732`). Implement the absence check (spec §4.2). Guard it so it only fires when the text looks invoice-like (avoids flagging non-invoice documents).

**Files:**
- Modify: `src/modules/content.py` (`check_legal_mentions` ~1640-1732)
- Test: `tests/test_content.py` (`test_missing_company_id_flagged` exists ~723-728; check its sibling positive test if any)

**Interfaces:** `check_legal_mentions` now emits `Flag(code="CONTENT_MISSING_COMPANY_ID", severity="medium")` when invoice-like text has no SIRET/SIREN/VAT.

- [ ] **Step 1: Read the test to confirm the trigger**

The test input is `"Facture n° 2024-001\nMontant: 100€\nTVA: 20%"` and expects exactly one `CONTENT_MISSING_COMPANY_ID`. Confirm there is no companion test asserting the flag is ABSENT for a document that legitimately has an ID — if there is, the guard below must not break it.

Run: `.venv/bin/python -m pytest tests/test_content.py::TestCheckLegalMentions -v`
Expected: `test_missing_company_id_flagged` FAILS; any "valid mentions" test PASSES.

- [ ] **Step 2: Add the absence check**

In `src/modules/content.py` `check_legal_mentions`, insert before `return flags` (~1731):

```python
    # Flag ABSENCE of any legal identifier on invoice-like documents.
    # French invoices legally require a SIRET/SIREN; none present is suspicious.
    # Gate on invoice-like text so we don't flag letters/attestations.
    text_lower = text.lower()
    # French-only indicators: this check flags FRENCH invoices missing FRENCH
    # legal IDs (SIRET/SIREN/VAT). Do NOT include the English "invoice" — a
    # foreign/English invoice legitimately has no SIRET and must not be flagged
    # (guards test_non_french_document_not_flagged).
    looks_like_invoice = any(
        kw in text_lower for kw in ("facture", "tva", "montant", "total ttc")
    )
    has_any_id = bool(sirets or sirens or vats)
    if looks_like_invoice and not has_any_id:
        flags.append(Flag(
            severity="medium",
            code="CONTENT_MISSING_COMPANY_ID",
            message="French invoice has no SIRET, SIREN or VAT number (legally required)",
            details={"explanation": "Registered French companies must print their SIRET/SIREN on invoices."},
        ))
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_content.py::TestCheckLegalMentions -v`
Expected: PASS (missing-ID test passes; any valid-mentions test still passes because `has_any_id` is true there).

- [ ] **Step 4: Full-suite green checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: **0 failed** — all 5 original failures resolved and all new Phase 0 tests passing.

---

### Task 11: Calibration / evaluation harness skeleton

**Why:** spec §6 requires a labelled-corpus harness to measure per-flag FP/FN rates and (later) fit scoring weights — replacing today's hand-guessed thresholds. No corpus exists yet (user provides later), so build the runnable skeleton that degrades gracefully on an empty corpus and prints a per-flag-code report.

**Files:**
- Create: `scripts/eval_harness.py`
- Create: `samples/README.md` (documents the expected corpus layout; the `samples/` dir is gitignored per CLAUDE.md and stays local)
- Test: `tests/test_eval_harness.py` (create)

**Interfaces:**
- Produces: `run_evaluation(corpus_dir: str) -> dict` returning `{"total": int, "by_label": {...}, "flag_code_counts": {...}, "false_positives": int, "false_negatives": int, "notes": [...]}`. Convention: `corpus_dir/legitimate/*.pdf` are genuine, `corpus_dir/suspicious/*.pdf` are tampered.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_harness.py`:

```python
from scripts.eval_harness import run_evaluation


def test_empty_corpus_is_graceful(tmp_path):
    (tmp_path / "legitimate").mkdir()
    (tmp_path / "suspicious").mkdir()
    report = run_evaluation(str(tmp_path))
    assert report["total"] == 0
    assert report["false_positives"] == 0
    assert report["false_negatives"] == 0
    assert isinstance(report["flag_code_counts"], dict)


def test_missing_corpus_dir_notes_it(tmp_path):
    report = run_evaluation(str(tmp_path / "does_not_exist"))
    assert report["total"] == 0
    assert any("no corpus" in n.lower() for n in report["notes"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: FAIL — `scripts.eval_harness` does not exist.

- [ ] **Step 3: Create the harness**

Create `scripts/eval_harness.py`:

```python
"""
Calibration / evaluation harness for TrustyFile (spec §6).

Runs the analyzer over a labelled corpus and reports per-flag-code counts and
FP/FN rates. Corpus layout:
    <corpus_dir>/legitimate/*.pdf   → expected GENUINE (score >= 50)
    <corpus_dir>/suspicious/*.pdf   → expected SUSPICIOUS (score < 50)

No corpus is bundled (evaluation-only licences + org no-leak rule). Point it at
a local directory the user provides. Usage:
    .venv/bin/python -m scripts.eval_harness /path/to/corpus
"""
import sys
from collections import Counter
from pathlib import Path


def run_evaluation(corpus_dir: str) -> dict:
    from src.analyzer import TrustyFileAnalyzer

    root = Path(corpus_dir)
    notes: list[str] = []
    report = {
        "total": 0,
        "by_label": {"legitimate": 0, "suspicious": 0},
        "flag_code_counts": {},
        "false_positives": 0,   # legitimate doc scored < 50
        "false_negatives": 0,   # suspicious doc scored >= 50
        "notes": notes,
    }

    if not root.exists():
        notes.append(f"No corpus at {root} — nothing to evaluate.")
        return report

    analyzer = TrustyFileAnalyzer(enable_external=False)
    flag_counter: Counter = Counter()

    for label, threshold_ok in (("legitimate", lambda s: s >= 50),
                                ("suspicious", lambda s: s < 50)):
        subdir = root / label
        if not subdir.exists():
            notes.append(f"Missing subdir: {subdir}")
            continue
        for pdf in sorted(subdir.glob("*.pdf")):
            try:
                result = analyzer.analyze(str(pdf))
            except Exception as e:  # harness must survive one bad file
                notes.append(f"Analyze failed for {pdf.name}: {e}")
                continue
            report["total"] += 1
            report["by_label"][label] += 1
            for m in result.modules:
                for f in m.flags:
                    flag_counter[f.code] += 1
            if label == "legitimate" and not threshold_ok(result.trust_score):
                report["false_positives"] += 1
            if label == "suspicious" and not threshold_ok(result.trust_score):
                report["false_negatives"] += 1

    report["flag_code_counts"] = dict(flag_counter)
    return report


def _print_report(report: dict) -> None:
    print(f"Documents evaluated: {report['total']}")
    print(f"  legitimate: {report['by_label']['legitimate']}  "
          f"suspicious: {report['by_label']['suspicious']}")
    print(f"False positives (legit flagged): {report['false_positives']}")
    print(f"False negatives (fraud missed):  {report['false_negatives']}")
    if report["flag_code_counts"]:
        print("Flag code frequencies:")
        for code, n in sorted(report["flag_code_counts"].items(),
                              key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {code}")
    for note in report["notes"]:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "samples"
    _print_report(run_evaluation(corpus))
```

- [ ] **Step 4: Create `scripts/__init__.py` if needed and the samples README**

Ensure `scripts/` is importable — create an empty `scripts/__init__.py` if the package import fails.

Create `samples/README.md`:

```markdown
# Calibration corpus (local only — gitignored)

Place labelled PDFs here to calibrate and regression-test the detector:

    samples/legitimate/*.pdf   # known-genuine documents (expect score >= 50)
    samples/suspicious/*.pdf   # known-tampered documents (expect score < 50)

Run:  `.venv/bin/python -m scripts.eval_harness samples`

## Sourcing

- **legitimate/** — public sample documents are fine to download and seed here:
  - Factur-X / ZUGFeRD sample invoices (e.g. the `akretion/factur-x` and
    ZUGFeRD reference sample sets) — also exercise the Phase 2/3 XML cross-check
  - Blank French CERFA / tax / URSSAF forms from service-public.fr / impots.gouv.fr
  - Public utility / telecom / marketplace sample invoices (EDF, Orange, Amazon PDFs)
  - Sample digitally-signed PDFs (for the Phase 2 pyHanko path)
- **suspicious/** — user-provided tampered documents (arrive later). Do NOT
  commit or upload any document here — sensitive and licence-restricted, and
  public evaluation datasets (Find it!, DocTamper) are research/eval-only.
```

- [ ] **Step 5: Run the harness tests**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: PASS.

- [ ] **Step 6: Smoke-run the harness on the empty samples dir**

Run: `.venv/bin/python -m scripts.eval_harness samples`
Expected: prints `Documents evaluated: 0` and a `NOTE:` line — no crash.

- [ ] **Step 7: Final Phase-0 checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: **all pass, 0 failed.**

---

### Task 12: Seed the calibration corpus (legit base + forged tampered variants)

**Why:** the harness (Task 11) needs real documents to measure FP/FN. Per the user: legitimate PDFs can be sourced online, and the suspicious half can be *forged* from a clean base (authorized — we are testing our own detector; matches the CLAUDE.md "Test Document Ideas"). Build a generator that emits one clean legit invoice and several forged variants, each mapping to a distinct fraud technique, and a test asserting the forged ones score below the legit one (an end-to-end detector sanity check + baseline).

**Files:**
- Create: `scripts/build_corpus.py`
- Create/populate: `samples/legitimate/*.pdf`, `samples/suspicious/*.pdf` (gitignored)
- Test: `tests/test_corpus_baseline.py` (create)

**Interfaces:**
- Consumes: `scripts.generate_demo_pdf` patterns; `src.analyzer.TrustyFileAnalyzer`.
- Produces: `build_corpus(out_dir: str) -> dict` returning `{"legitimate": [paths], "suspicious": [paths]}`.

- [ ] **Step 1: Write the failing baseline test**

Create `tests/test_corpus_baseline.py`:

```python
from scripts.build_corpus import build_corpus
from src.analyzer import TrustyFileAnalyzer


def test_forged_variants_score_below_clean(tmp_path):
    paths = build_corpus(str(tmp_path))
    analyzer = TrustyFileAnalyzer(enable_external=False)

    legit_scores = [analyzer.analyze(p).trust_score for p in paths["legitimate"]]
    forged_scores = [analyzer.analyze(p).trust_score for p in paths["suspicious"]]

    assert legit_scores and forged_scores
    # The clean invoice should score higher than the average forged variant.
    assert (sum(legit_scores) / len(legit_scores)) > (sum(forged_scores) / len(forged_scores))
    # At least one forged variant must be clearly flagged.
    assert min(forged_scores) < 50
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_corpus_baseline.py -v`
Expected: FAIL — `scripts.build_corpus` does not exist.

- [ ] **Step 3: Create the corpus builder**

Create `scripts/build_corpus.py`. It generates one clean invoice (a de-fanged copy of `generate_demo_pdf` — coherent dates, a legitimate producer, consistent fonts, no watermark) and forges tampered variants from it:

```python
"""
Build a calibration corpus: one clean legit invoice + forged tampered variants.

Legit samples may also be sourced online (public Factur-X / utility invoices —
see samples/README.md); this builder guarantees a reproducible local baseline so
the harness and CI always have data. Forging our own tampered documents to test
our own detector is intended (CLAUDE.md "Test Document Ideas").
"""
from datetime import datetime, timedelta
from pathlib import Path

import fitz


def _clean_invoice(path: str) -> None:
    """A coherent, legitimate-looking invoice: valid dates, normal producer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS", fontname="helv", fontsize=18)
    page.insert_text(fitz.Point(40, 85),
                     "SIRET: 552 081 317 00016  |  TVA: FR03552081317",
                     fontname="helv", fontsize=8)
    page.insert_text(fitz.Point(40, 140), "FACTURE  N° FA-2025-00847", fontname="helv", fontsize=14)
    page.insert_text(fitz.Point(350, 120), "Date de facturation: 15/01/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(350, 135), "Date d'échéance: 14/02/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(40, 300), "Prestation de conseil", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 360), "Total HT:", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 360), "1 000,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 380), "TVA (20%):", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 380), "200,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(40, 460), "IBAN: FR7630006000011234567890189",
                     fontname="helv", fontsize=9)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "ReportLab PDF Library",
        "creator": "ACME Billing System",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": creation.strftime("D:%Y%m%d%H%M%S"),  # equal → no modification
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _forge_metadata_dates(src: str, dst: str) -> None:
    """Tamper: ModDate 45 days after CreationDate + online-converter producer."""
    doc = fitz.open(src)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "iLovePDF",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": (creation + timedelta(days=45)).strftime("D:%Y%m%d%H%M%S"),
    })
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    doc.close()


def _forge_amount_overlay(src: str, dst: str) -> None:
    """Tamper: white-box over the total, new (higher) amount drawn on top."""
    doc = fitz.open(src)
    page = doc[0]
    # Cover "1 200,00 €" and draw "9 200,00 €" over it.
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)  # full save; also try incremental variant below
    doc.close()


def _forge_incremental_edit(src: str, dst: str) -> None:
    """Tamper: append an incremental update that overlays a new amount."""
    import shutil
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    doc = fitz.open(dst)
    page = doc[0]
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)  # keeps prior revision
    doc.close()


def build_corpus(out_dir: str) -> dict:
    out = Path(out_dir)
    legit = out / "legitimate" / "clean_invoice.pdf"
    _clean_invoice(str(legit))

    forged = []
    f1 = out / "suspicious" / "forged_metadata_dates.pdf"
    _forge_metadata_dates(str(legit), str(f1)); forged.append(str(f1))
    f2 = out / "suspicious" / "forged_amount_overlay.pdf"
    _forge_amount_overlay(str(legit), str(f2)); forged.append(str(f2))
    f3 = out / "suspicious" / "forged_incremental_edit.pdf"
    _forge_incremental_edit(str(legit), str(f3)); forged.append(str(f3))

    return {"legitimate": [str(legit)], "suspicious": forged}


if __name__ == "__main__":
    import sys
    result = build_corpus(sys.argv[1] if len(sys.argv) > 1 else "samples")
    print("Legitimate:", *result["legitimate"], sep="\n  ")
    print("Suspicious:", *result["suspicious"], sep="\n  ")
```

- [ ] **Step 4: Run the baseline test**

Run: `.venv/bin/python -m pytest tests/test_corpus_baseline.py -v`
Expected: PASS — forged variants average below the clean invoice and at least one scores < 50. If the test fails because the clean invoice itself scores low, that is a *finding* (a current false positive) — record it as a note and, if it is one of the known FP factories in spec §2, leave the assertion but xfail with a reason so it is revisited in the relevant later phase. Do not weaken the detector to pass.

- [ ] **Step 5: Generate the persistent corpus and run the harness**

Run: `.venv/bin/python -m scripts.build_corpus samples`
Then: `.venv/bin/python -m scripts.eval_harness samples`
Expected: harness reports `Documents evaluated: 4`, with the forged variants contributing flag codes (e.g. `META_DOCUMENT_MODIFIED`, `META_ONLINE_CONVERTER`, `STRUCT_INCREMENTAL_UPDATES`).

- [ ] **Step 6: Final Phase-0 checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 0 failed.

---

## Self-Review

**Spec coverage (Phase 0 items from `2026-07-04-detection-improvements-design.md` §7, §3.3, §3.5, §9 "Phase 0"):**
- XSS escaping → Task 6 ✓
- Remove EU DSS document upload → Task 4 ✓
- Remove anyapi.io IBAN call → Task 5 ✓
- Guard `float(hours)` crash → Task 3 ✓
- Fail-closed on malformed/encrypted PDFs → Task 2 ✓ (scoring-band UNVERIFIABLE label deferred to Phase 1, as scoped; Task 2 delivers fail-closed extraction + `EXTRACTION_FAILED` flag + score 0)
- Timezone-aware date parsing → Task 1 ✓
- Enriched single parse pass → Task 2 ✓ (fields added + populated; migrating the 6 re-opening modules onto them happens in their respective later phases, as noted in spec §3.5)
- Green the 5 failing tests → Tasks 7 (canva), 8 (screenshot, compression), 9 (service anachronism), 10 (missing company id) ✓
- Calibration harness skeleton → Task 11 ✓
- Seed calibration corpus (legit base + forged tampered variants) → Task 12 ✓ (added per user; legit half also sourceable online per `samples/README.md`)

**Placeholder scan:** every code step contains complete, runnable code; no TBD/TODO/"handle edge cases". ✓

**Type consistency:** `verify_signature_dss` return dict keys used by callers (`is_trusted`, `is_fully_valid`, `signer`, `signature_level`) all present (Task 4). `verify_iban_api` signature `(iban, api_key=None) -> (dict, str|None)` matches its caller in `external.py` (Task 5). `PDFData` new fields consumed only by Task 2's analyzer guard. `esc()` defined once (Task 6). ✓

**Scoping notes carried forward to Phase 1:** the full three-band GENUINE/REVIEW/SUSPICIOUS + UNVERIFIABLE output and the evidence-ledger model are Phase 1; Phase 0 only makes extraction fail closed and adds the container/extraction flags they will consume.
