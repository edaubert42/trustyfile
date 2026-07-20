# MRZ Checksum Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `analyze_identity` runs the full ICAO MRZ checksums (`validate_td3`/`validate_td1`) when a clean embedded MRZ sits on a digital-native page, and keeps today's OCR + structure-only path for real scans.

**Architecture:** Text-layer-first with a digital-native gate. `extract_mrz_lines` (currently dead) gains a page index; a new `_is_digital_native_page` helper decides, via `page.get_image_info()`, whether the MRZ page carries a full-page raster image (= a scan → OCR path) or is digitally typeset (= run deterministic checksums). Checksum-failure flags flow as SOFT/tamper evidence — no evidence-registry change. Spec: `docs/superpowers/specs/2026-07-20-mrz-checksum-wiring-design.md`.

**Tech Stack:** Python 3.12, PyMuPDF (fitz), pytest. No new dependencies.

## Global Constraints

- Test runner is ALWAYS `.venv/bin/python -m pytest` (plain `python`/`pytest` do not exist on this machine).
- Digital-native gate: a page is digital-native ⇔ it has NO raster image whose bbox area ≥ `FULL_PAGE_IMAGE_AREA_RATIO` (= 0.6) of the page area. `_is_digital_native_page` fails closed (returns False → OCR path) on any error or out-of-range page index.
- Checksum failures stay SOFT: the 5 codes (`MRZ_DOC_NUMBER_CHECK_FAIL`, `MRZ_DOB_CHECK_FAIL`, `MRZ_EXPIRY_CHECK_FAIL`, `MRZ_FINAL_CHECK_FAIL` = critical; `MRZ_OPTIONAL_CHECK_FAIL` = high) are NOT added to any registry in `src/evidence.py`. A realistic single-digit tamper trips a field check AND the composite (2 criticals → ≈38/SUSPICIOUS); a lone failure → ≈73/REVIEW.
- `details["mrz_found"] = True` MUST be set whenever an MRZ is found on either path (build-1 coverage-gate exemption depends on it). `details["mrz_source"]` records `"text_layer"` or `"ocr"`.
- The OCR path (`_validate_mrz_structure_only` + `ocr_extract_mrz`) is behavior-unchanged — only relocated below the text-layer attempt.
- `fitz` is imported locally inside functions in this module (not at top); new code follows that pattern and uses `with fitz.open(...) as doc:` for auto-close.
- No summary.py changes: the 5 checksum codes are already themed + EN/FR-templated (verified 2026-07-20: 3 occurrences each). Adding no new code strings keeps `test_summary_drift.py` green.
- Canonical ICAO 9303 specimens (verified 2026-07-20 to validate cleanly against `validate_td3`/`validate_td1`, aside from an expected `MRZ_DOCUMENT_EXPIRED` since both expire 2012):
  - TD3: `P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<` / `L898902C36UTO7408122F1204159ZE184226B<<<<<10`
  - TD1: `I<UTOD231458907<<<<<<<<<<<<<<<` / `7408122F1204159UTO<<<<<<<<<<<6` / `ERIKSSON<<ANNA<MARIA<<<<<<<<<<`
- Every MRZ string contains `<<<<<`, which is in `_IDENTITY_KEYWORDS`, so any MRZ fixture auto-routes through `analyze_identity` via `is_likely_identity_doc`.
- Commits on branch `phase8-mrz-checksums`; end every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Page-indexed `extract_mrz_lines` + `_is_digital_native_page` gate

**Files:**
- Modify: `src/modules/identity.py` (`extract_mrz_lines` signature + returns; add constant + `_is_digital_native_page`)
- Test: `tests/test_identity_mrz_checksums.py` (create)

**Interfaces:**
- Produces: `extract_mrz_lines(pdf_data) -> tuple[list[str], int] | None` (was `list[str] | None`); `FULL_PAGE_IMAGE_AREA_RATIO: float = 0.6`; `_is_digital_native_page(file_path: str, page_index: int) -> bool`. Task 2 consumes all three.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_identity_mrz_checksums.py`:

```python
"""MRZ checksum wiring (spec 2026-07-20): deterministic ICAO checksums on a
digital-native MRZ; OCR structure-only path for scans."""
import fitz
import pytest

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.identity import (
    extract_mrz_lines, _is_digital_native_page, FULL_PAGE_IMAGE_AREA_RATIO,
)

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
       "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def _write_mrz_pdf(path, lines, full_page_image=False):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if full_page_image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
        pix.clear_with(240)
        page.insert_image(page.rect, pixmap=pix)
    y = 760
    for ln in lines:
        page.insert_text(fitz.Point(40, y), ln, fontname="cour", fontsize=10)
        y += 14
    doc.save(str(path))
    doc.close()


def test_extract_mrz_lines_returns_lines_and_page(tmp_path):
    pdf = tmp_path / "mrz.pdf"
    _write_mrz_pdf(pdf, TD3)
    result = extract_mrz_lines(extract_pdf_data(str(pdf)))
    assert result is not None
    lines, page_index = result
    assert lines == TD3
    assert page_index == 0


def test_extract_mrz_lines_none_when_absent(tmp_path):
    pdf = tmp_path / "plain.pdf"
    _write_mrz_pdf(pdf, ["Just an ordinary invoice line, no MRZ here at all."])
    assert extract_mrz_lines(extract_pdf_data(str(pdf))) is None


def test_digital_native_true_for_text_only_page(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_mrz_pdf(pdf, TD3, full_page_image=False)
    assert _is_digital_native_page(str(pdf), 0) is True


def test_digital_native_false_for_full_page_image(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_mrz_pdf(pdf, TD3, full_page_image=True)
    assert _is_digital_native_page(str(pdf), 0) is False


def test_digital_native_false_on_out_of_range_page(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_mrz_pdf(pdf, TD3)
    assert _is_digital_native_page(str(pdf), 5) is False       # fail closed
    assert _is_digital_native_page("/no/such/file.pdf", 0) is False


def test_full_page_ratio_constant():
    assert FULL_PAGE_IMAGE_AREA_RATIO == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identity_mrz_checksums.py -v`
Expected: ImportError (`_is_digital_native_page` / `FULL_PAGE_IMAGE_AREA_RATIO` not defined), or the page-index test fails because `extract_mrz_lines` still returns a bare list.

- [ ] **Step 3: Change `extract_mrz_lines` to return the page index**

In `src/modules/identity.py`, change the signature line:

```python
def extract_mrz_lines(pdf_data: PDFData) -> tuple[list[str], int] | None:
```

Change the TD3 return (inside the `for page_num` loop):

```python
                logger.debug(f"MRZ TD3 détectée page {page_num + 1}, ligne {i + 1}")
                return [line_a, line_b], page_num
```

Change the TD1 return:

```python
                logger.debug(f"MRZ TD1 détectée page {page_num + 1}, ligne {i + 1}")
                return [line_a, line_b, line_c], page_num
```

Update the docstring `Returns:` line and example to reflect the tuple:

```python
    Returns:
        Tuple (lines, page_index) — 2 lines (TD3) or 3 lines (TD1) plus the
        0-based page they were found on — or None if nothing matched.
```
(and in the docstring example, change `>>> lines` to `>>> lines, page = extract_mrz_lines(pdf_data)`.)

- [ ] **Step 4: Add the constant and the digital-native gate**

In `src/modules/identity.py`, immediately above `def extract_mrz_lines`, add:

```python
FULL_PAGE_IMAGE_AREA_RATIO = 0.6  # image bbox / page area at/above which a page reads as a scan


def _is_digital_native_page(file_path: str, page_index: int) -> bool:
    """True if the page has NO full-page raster image — i.e. it was digitally
    typeset, not scanned or photographed.

    Used to decide whether an embedded MRZ text layer is trustworthy for
    deterministic ICAO checksums: a scan's OCR text layer is not (OCR confuses
    0/O, 5/S…), but a digitally-typed MRZ is exact. Fails closed (returns False
    → caller uses the OCR path) on any error or out-of-range page.
    """
    import fitz
    try:
        with fitz.open(file_path) as doc:
            if page_index < 0 or page_index >= doc.page_count:
                return False
            page = doc[page_index]
            page_area = abs(page.rect.width * page.rect.height)
            if page_area <= 0:
                return False
            for info in page.get_image_info():
                bbox = fitz.Rect(info["bbox"])
                if abs(bbox.width * bbox.height) >= FULL_PAGE_IMAGE_AREA_RATIO * page_area:
                    return False
        return True
    except Exception:
        return False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identity_mrz_checksums.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/modules/identity.py tests/test_identity_mrz_checksums.py
git commit -m "feat(identity): page-indexed extract_mrz_lines + digital-native gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Rewire `analyze_identity` to text-layer-first + fix critical score branch

**Files:**
- Modify: `src/modules/identity.py` (`analyze_identity` head + score loop + return)
- Test: `tests/test_identity_mrz_checksums.py` (append)

**Interfaces:**
- Consumes: `extract_mrz_lines` (tuple form), `_is_digital_native_page` (Task 1); `validate_td3`, `validate_td1`, `_validate_mrz_structure_only`, `_check_mrz_coherence`, `ocr_extract_mrz` (existing).
- Produces: `analyze_identity` sets `details["mrz_source"]` ∈ {"text_layer","ocr"}; runs checksums on the text-layer/digital-native path; keeps structure-only on the OCR path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identity_mrz_checksums.py`:

```python
from src.modules.identity import analyze_identity
from src.analyzer import TrustyFileAnalyzer

TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<",
       "7408122F1204159UTO<<<<<<<<<<<6",
       "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]


def _codes(module_result):
    return [f.code for f in module_result.flags]


def test_valid_td3_textlayer_runs_checksums_no_fail(tmp_path):
    pdf = tmp_path / "passport.pdf"
    _write_mrz_pdf(pdf, TD3)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details["mrz_found"] is True
    assert r.details["mrz_source"] == "text_layer"
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]
    assert r.details.get("date_of_birth") == "1974-08-12"  # parsed from a clean MRZ


def test_tampered_td3_textlayer_fails_checksums(tmp_path):
    tampered = TD3[1][:13] + "5" + TD3[1][14:]   # flip one DOB digit, no recompute
    pdf = tmp_path / "tampered.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered])
    r = analyze_identity(extract_pdf_data(str(pdf)))
    codes = _codes(r)
    assert "MRZ_DOB_CHECK_FAIL" in codes
    assert "MRZ_FINAL_CHECK_FAIL" in codes


def test_valid_td1_textlayer_runs_checksums_no_fail(tmp_path):
    pdf = tmp_path / "idcard.pdf"
    _write_mrz_pdf(pdf, TD1)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details["mrz_source"] == "text_layer"
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]


def test_mrz_over_full_page_image_does_not_run_textlayer_checksums(tmp_path):
    # A scan (full-page image) with an OCR-style MRZ text layer must NOT take
    # the deterministic checksum path — even a tampered MRZ here stays off it.
    tampered = TD3[1][:13] + "5" + TD3[1][14:]
    pdf = tmp_path / "scan.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered], full_page_image=True)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details.get("mrz_source") != "text_layer"
    # structure-only path emits no checksum codes
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]


def test_tampered_td3_end_to_end_is_suspicious(tmp_path):
    # Full analyzer: '<<<<<' routes to identity; tampered digital-native MRZ
    # fails 2 checksums (field + composite) -> SUSPICIOUS.
    tampered = TD3[1][:13] + "5" + TD3[1][14:]
    pdf = tmp_path / "tampered_e2e.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered])
    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(pdf))
    assert result.document_type == "identity"
    assert result.band == "SUSPICIOUS", f"{result.band}/{result.trust_score}"


def test_valid_td3_not_coverage_blind(tmp_path):
    # Regression tie to build 1: a digital-native ID is not banded UNVERIFIABLE.
    pdf = tmp_path / "valid_e2e.pdf"
    _write_mrz_pdf(pdf, TD3)
    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(pdf))
    assert result.band != "UNVERIFIABLE"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identity_mrz_checksums.py -v -k "textlayer or tampered or coverage_blind or td1"`
Expected: the `mrz_source`/checksum tests fail — `analyze_identity` still always OCRs, so `details` has no `mrz_source` and the digital-native TD3 never gets checksum validation.

- [ ] **Step 3: Rewire the head of `analyze_identity`**

In `src/modules/identity.py`, replace this block (from the `details` declaration through the `_validate_mrz_structure_only` call):

```python
    details: dict = {"mrz_found": False}

    # Toujours OCR — les IDs sont toujours des scans
    logger.info("Module identity : lancement OCR pour détection MRZ...")
    mrz_lines = None
    mrz_page = 0
    try:
        from src.extractors.ocr import ocr_extract_mrz
        mrz_result = ocr_extract_mrz(pdf_data.file_path, pdf_data.page_count)
        if mrz_result is not None:
            mrz_lines, mrz_page = mrz_result
    except Exception as e:
        logger.warning(f"Module identity : OCR échoué ({e})")

    if mrz_lines is None:
        logger.info("Module identity : aucune MRZ détectée par OCR")
        return ModuleResult(
            module="identity",
            flags=[],
            score=100,
            confidence=0.4,
            details={
                "mrz_found": False,
                "note": "Aucune MRZ détectée. Le document est peut-être une photo partielle ou la résolution est insuffisante.",
            },
        )

    # MRZ trouvée — validation structurelle uniquement (OCR → checksums non fiables)
    details["mrz_found"] = True
    details["mrz_format"] = "TD3" if len(mrz_lines) == 2 else "TD1"
    details["mrz_lines"] = mrz_lines
    details["mrz_page"] = mrz_page
    details["note"] = "MRZ détectée par OCR — validation structurelle (dates, sexe). Checksums non vérifiables sur scan."

    logger.info(f"Module identity : MRZ {details['mrz_format']} détectée (page {mrz_page + 1})")

    flags, mrz_fields = _validate_mrz_structure_only(mrz_lines)
    details.update(mrz_fields)
```

with:

```python
    details: dict = {"mrz_found": False}

    mrz_lines = None
    mrz_page = 0
    mrz_source = None
    use_checksums = False

    # --- Path 1: clean embedded MRZ on a digital-native page -----------------
    # A digitally-typeset page (no full-page scan image behind the text) yields
    # a deterministic MRZ: ICAO checksums are trustworthy and a failure is real
    # tamper evidence. This is also the shape of a fabricated ID PDF — the case
    # the checksum path exists to catch.
    text_layer = extract_mrz_lines(pdf_data)
    if text_layer is not None:
        candidate_lines, candidate_page = text_layer
        if _is_digital_native_page(pdf_data.file_path, candidate_page):
            mrz_lines, mrz_page = candidate_lines, candidate_page
            mrz_source = "text_layer"
            use_checksums = True
            logger.info("Module identity : MRZ couche-texte, page digital-native "
                        "→ validation checksums ICAO")

    # --- Path 2: OCR fallback (real scan/photo, or MRZ over a scan image) -----
    if mrz_lines is None:
        logger.info("Module identity : lancement OCR pour détection MRZ...")
        try:
            from src.extractors.ocr import ocr_extract_mrz
            mrz_result = ocr_extract_mrz(pdf_data.file_path, pdf_data.page_count)
            if mrz_result is not None:
                mrz_lines, mrz_page = mrz_result
                mrz_source = "ocr"
        except Exception as e:
            logger.warning(f"Module identity : OCR échoué ({e})")

    if mrz_lines is None:
        logger.info("Module identity : aucune MRZ détectée")
        return ModuleResult(
            module="identity",
            flags=[],
            score=100,
            confidence=0.4,
            details={
                "mrz_found": False,
                "note": "Aucune MRZ détectée. Le document est peut-être une photo partielle ou la résolution est insuffisante.",
            },
        )

    # MRZ trouvée
    details["mrz_found"] = True
    details["mrz_source"] = mrz_source
    details["mrz_format"] = "TD3" if len(mrz_lines) == 2 else "TD1"
    details["mrz_lines"] = mrz_lines
    details["mrz_page"] = mrz_page

    if use_checksums:
        # Digital-native MRZ → full ICAO checksum validation (deterministic).
        details["note"] = ("MRZ en couche texte (document digital-native) — "
                           "validation complète des checksums ICAO 9303.")
        if len(mrz_lines) == 2:
            flags, mrz_fields = validate_td3(mrz_lines[0], mrz_lines[1])
        else:
            flags, mrz_fields = validate_td1(mrz_lines[0], mrz_lines[1], mrz_lines[2])
        confidence = 0.9
    else:
        # OCR path → structure-only (OCR errors make raw checksums unreliable).
        details["note"] = "MRZ détectée par OCR — validation structurelle (dates, sexe). Checksums non vérifiables sur scan."
        flags, mrz_fields = _validate_mrz_structure_only(mrz_lines)
        confidence = 0.65
    details.update(mrz_fields)

    logger.info(f"Module identity : MRZ {details['mrz_format']} détectée "
                f"(page {mrz_page + 1}, source {mrz_source})")
```

- [ ] **Step 4: Add the `critical` branch to the module score loop**

The checksum path introduces critical flags into this module for the first time; the score loop never handled `critical`. In `src/modules/identity.py`, change:

```python
    score = 100
    for flag in flags:
        if flag.severity == "high":
            score -= 25
        elif flag.severity == "medium":
            score -= 10
        elif flag.severity == "low":
            score -= 5
    score = max(0, score)
```

to:

```python
    score = 100
    for flag in flags:
        if flag.severity == "critical":
            score -= 50
        elif flag.severity == "high":
            score -= 25
        elif flag.severity == "medium":
            score -= 10
        elif flag.severity == "low":
            score -= 5
    score = max(0, score)
```

(Note: the ledger scores from flags, not this number — but the per-module score is shown in the UI and must not read 100 alongside critical flags.)

- [ ] **Step 5: Make the return confidence path-dependent**

Change:

```python
    return ModuleResult(
        module="identity",
        flags=flags,
        score=score,
        # Confiance modérée : MRZ trouvée mais pas de checksum possible
        confidence=0.65,
        details=details,
    )
```

to:

```python
    return ModuleResult(
        module="identity",
        flags=flags,
        score=score,
        # 0.9 when checksums validated the MRZ (deterministic), else 0.65 (OCR).
        confidence=confidence,
        details=details,
    )
```

- [ ] **Step 6: Run the new tests + existing identity tests**

Run: `.venv/bin/python -m pytest tests/test_identity_mrz_checksums.py tests/test_identity_us.py tests/test_identity_facecheck_graceful.py -v`
Expected: all pass. The two existing identity tests exercise the OCR/structure path and must be unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/modules/identity.py tests/test_identity_mrz_checksums.py
git commit -m "feat(identity): text-layer-first MRZ with ICAO checksums on digital-native docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Full-suite exit gate

**Files:**
- No source changes expected. Fixes only if the suite surfaces a regression.

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures, 1 xfailed (pre-existing `forged_image_patch_over_text`). Includes `test_summary_drift.py` (proves no new unmapped codes) and both existing identity tests unchanged.

- [ ] **Step 2: Run the corpus band gate (local corpus present)**

Run: `.venv/bin/python -m pytest tests/test_corpus_bands.py tests/test_coverage_gate.py -q`
Expected: all pass — no legit corpus band moved; the coverage gate is unaffected (identity docs with a found MRZ stay exempt).

- [ ] **Step 3: Manual end-to-end spot check**

```bash
.venv/bin/python - <<'EOF'
import tempfile, fitz
from src.analyzer import TrustyFileAnalyzer
TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
       "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
def build(path, l2):
    d = fitz.open(); p = d.new_page(width=595, height=842); y = 760
    for ln in [TD3[0], l2]:
        p.insert_text(fitz.Point(40, y), ln, fontname="cour", fontsize=10); y += 14
    d.save(path); d.close()
tmp = tempfile.mkdtemp()
build(f"{tmp}/ok.pdf", TD3[1])
build(f"{tmp}/bad.pdf", TD3[1][:13] + "5" + TD3[1][14:])
a = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False)
for name in ("ok", "bad"):
    r = a.analyze(f"{tmp}/{name}.pdf")
    print(name, r.document_type, r.band, r.trust_score,
          [f.code for m in r.modules if m.module=="identity" for f in m.flags])
EOF
```
Expected: `ok` → identity, GENUINE (only MRZ_DOCUMENT_EXPIRED); `bad` → identity, SUSPICIOUS with `MRZ_DOB_CHECK_FAIL` + `MRZ_FINAL_CHECK_FAIL`.

- [ ] **Step 4: Commit any stragglers and stop**

If steps 1-2 required fixes, commit them. Do NOT merge: the branch waits for the final whole-branch review.
