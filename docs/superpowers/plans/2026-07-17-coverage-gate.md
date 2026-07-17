# Coverage Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Documents with no usable text layer (scans/photos) band UNVERIFIABLE instead of ~96/GENUINE, unless tamper evidence already made them SUSPICIOUS or an MRZ was found.

**Architecture:** A pure gate function in `analyzer.py` measures average non-whitespace chars/page and checks the identity module's `mrz_found` marker; when blind, the analyzer appends a synthetic `coverage` ModuleResult and threads `coverage_blind=True` through `create_analysis_result` into `aggregate_evidence`, which overrides GENUINE/REVIEW bands to UNVERIFIABLE (SUSPICIOUS passes through). Summary layer gets the new flag's theme/templates and a scan-specific verdict. Spec: `docs/superpowers/specs/2026-07-17-coverage-gate-design.md`.

**Tech Stack:** Python 3.12, PyMuPDF (fitz), pytest. No new dependencies.

## Global Constraints

- Threshold: `MIN_TEXT_CHARS_PER_PAGE = 50` — blind ⇔ average **strictly below** 50 non-whitespace chars/page (spec D2).
- Flag code: `COVERAGE_NO_TEXT_LAYER`, severity `low`, module `"coverage"`. It goes in NO registry (not `HARD_NEGATIVE_CODES`, not `GENUINE_CODES`) — it is not tamper evidence.
- Override only bands `GENUINE` and `REVIEW` → `UNVERIFIABLE` (score 0, range [0,0]). `SUSPICIOUS` always passes through (spec D3).
- Exemption: identity ModuleResult with `details["mrz_found"] == True` (spec D4). Identity crash / no MRZ stays blind.
- EN verdict (verbatim): "This appears to be a scanned or photographed document — its content could not be machine-verified. Request the digital original."
- FR verdict (verbatim): "Ce document semble être un scan ou une photo — son contenu n'a pas pu être vérifié automatiquement. Demandez l'original numérique."
- Test runner is ALWAYS `.venv/bin/python -m pytest` (plain `python`/`pytest` do not exist on this machine).
- Exit bar: full suite green including `tests/test_summary_drift.py`, `tests/test_corpus_baseline.py`, and (corpus present locally) `tests/test_corpus_bands.py` — no legit corpus doc may change band.
- app.py needs NO change: the evidence-bars bucket map iterates fixed module lists with `contrib.get(...)`, so the new `coverage` group is ignored gracefully (verified 2026-07-17, app.py:1237-1248); the UNVERIFIABLE gauge branch already exists (app.py:1174).
- Commits on branch `phase8-coverage-gate`; end every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Gate function `_coverage_blind`

**Files:**
- Modify: `src/analyzer.py` (add constant + function after the imports/logger block, around line 40)
- Test: `tests/test_coverage_gate.py` (create)

**Interfaces:**
- Produces: `MIN_TEXT_CHARS_PER_PAGE: int = 50` and `_coverage_blind(text_by_page: list[str], page_count: int, module_results: list[ModuleResult]) -> tuple[bool, float]` in `src.analyzer`. Task 3 calls it; tests import both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coverage_gate.py`:

```python
"""Coverage gate (spec 2026-07-17): blind documents fail closed.

Empirical baseline this feature fixes: an image-only PDF (0 text chars)
scored 96/GENUINE/LOW with only META_NO_PRODUCER + IMAGES_IMAGE_ONLY_PDF.
"""
import fitz
import pytest

from src.analyzer import MIN_TEXT_CHARS_PER_PAGE, _coverage_blind
from src.models import ModuleResult


def test_threshold_is_strictly_below_50():
    assert MIN_TEXT_CHARS_PER_PAGE == 50
    blind_49, avg_49 = _coverage_blind(["x" * 49], 1, [])
    not_blind_50, avg_50 = _coverage_blind(["x" * 50], 1, [])
    assert blind_49 is True and avg_49 == 49.0
    assert not_blind_50 is False and avg_50 == 50.0


def test_average_is_over_page_count_and_ignores_whitespace():
    # 100 non-ws chars spread over 2 pages -> avg 50 -> not blind
    assert _coverage_blind(["x" * 100, ""], 2, [])[0] is False
    # 99 non-ws chars over 2 pages -> avg 49.5 -> blind
    assert _coverage_blind(["x" * 99, ""], 2, [])[0] is True
    # whitespace never counts
    assert _coverage_blind(["  \n\t " * 100], 1, [])[0] is True
    # zero pages must not divide by zero
    assert _coverage_blind([], 0, [])[0] is True


def test_identity_mrz_exemption():
    blind_doc = ["x" * 5]
    with_mrz = [ModuleResult(module="identity", details={"mrz_found": True})]
    without_mrz = [ModuleResult(module="identity", details={"mrz_found": False})]
    crashed = [ModuleResult(module="identity", details={})]
    other_module = [ModuleResult(module="images", details={"mrz_found": True})]

    assert _coverage_blind(blind_doc, 1, with_mrz)[0] is False
    assert _coverage_blind(blind_doc, 1, without_mrz)[0] is True   # fail closed
    assert _coverage_blind(blind_doc, 1, crashed)[0] is True       # fail closed
    assert _coverage_blind(blind_doc, 1, other_module)[0] is True  # only identity counts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v`
Expected: FAIL / ERROR with `ImportError: cannot import name 'MIN_TEXT_CHARS_PER_PAGE'`

- [ ] **Step 3: Implement the gate function**

In `src/analyzer.py`, directly after `logger = logging.getLogger(__name__)` (line 39), add:

```python
# Coverage gate (spec docs/superpowers/specs/2026-07-17-coverage-gate-design.md):
# a document whose text layer averages fewer than this many non-whitespace
# characters per page is "blind" — text-based modules had nothing to analyze.
# Value deliberately matches images.py's image-only heuristic (aligned by
# intent, not shared: the two checks compute different things).
MIN_TEXT_CHARS_PER_PAGE = 50


def _coverage_blind(
    text_by_page: list[str],
    page_count: int,
    module_results: list[ModuleResult],
) -> tuple[bool, float]:
    """Decide whether the document is blind (no usable text layer).

    A blind document can be proven suspicious, never proven fine: the caller
    uses this to band otherwise-GENUINE results as UNVERIFIABLE. Identity
    documents where the identity module actually FOUND an MRZ are exempt —
    scanned IDs are expected to be image-only and get real OCR analysis
    (spec D4). An identity crash or an MRZ-less run stays blind: fail closed.

    Returns:
        (blind, avg_non_whitespace_chars_per_page)
    """
    non_ws = sum(len("".join(page.split())) for page in text_by_page)
    avg = non_ws / max(1, page_count)
    if avg >= MIN_TEXT_CHARS_PER_PAGE:
        return False, avg
    for r in module_results:
        if r.module == "identity" and (r.details or {}).get("mrz_found"):
            return False, avg
    return True, avg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_coverage_gate.py src/analyzer.py
git commit -m "feat(coverage): blindness gate function with MRZ exemption

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `aggregate_evidence` band override

**Files:**
- Modify: `src/evidence.py:156-206` (`aggregate_evidence`)
- Modify: `src/scoring.py:119-190` (`create_analysis_result` threads the parameter)
- Test: `tests/test_coverage_gate.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `aggregate_evidence(evidence, analyzable=True, prior=PRIOR_LOG_ODDS, coverage_blind=False)` and `create_analysis_result(file_hash, module_results, analysis_time_ms=0, document_type="unknown", coverage_blind=False)`. Task 3 passes `coverage_blind` from the analyzer.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coverage_gate.py`:

```python
from src.evidence import Evidence, aggregate_evidence


def _ev(code: str, weight: float, group: str) -> Evidence:
    return Evidence(code=code, weight=weight, kind="soft",
                    direction="tamper", group=group, message="")


def test_blind_empty_ledger_becomes_unverifiable():
    # Empty ledger scores 98/GENUINE today; blind must override it.
    agg = aggregate_evidence([], coverage_blind=True)
    assert agg.band == "UNVERIFIABLE"
    assert agg.risk_level == "UNVERIFIABLE"
    assert agg.score == 0 and agg.score_low == 0 and agg.score_high == 0


def test_blind_review_band_becomes_unverifiable():
    # One critical (3.0 nats): logit -4+3=-1 -> score 73 -> REVIEW -> override.
    agg = aggregate_evidence([_ev("X_CRIT", 3.0, "images")], coverage_blind=True)
    assert agg.band == "UNVERIFIABLE"


def test_blind_preserves_suspicious():
    # Two criticals in different groups: logit -4+3+3=2 -> score 12 -> SUSPICIOUS.
    ledger = [_ev("X_CRIT", 3.0, "images"), _ev("Y_CRIT", 3.0, "forensics")]
    agg = aggregate_evidence(ledger, coverage_blind=True)
    assert agg.band == "SUSPICIOUS"
    assert agg.score == aggregate_evidence(ledger, coverage_blind=False).score
    assert agg.ledger  # ledger is preserved for display


def test_extraction_failure_precedence_unchanged():
    agg = aggregate_evidence([], analyzable=False, coverage_blind=True)
    assert agg.band == "UNVERIFIABLE" and agg.score == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v -k "blind or precedence"`
Expected: FAIL with `TypeError: aggregate_evidence() got an unexpected keyword argument 'coverage_blind'`

- [ ] **Step 3: Implement the override**

In `src/evidence.py`, change the `aggregate_evidence` signature (line 156):

```python
def aggregate_evidence(evidence: list["Evidence"], analyzable: bool = True,
                       prior: float = PRIOR_LOG_ODDS,
                       coverage_blind: bool = False) -> AggregateResult:
```

Then, immediately after `band, risk = _band_and_risk(score)` (line 200), insert:

```python
    # Coverage gate (spec 2026-07-17, D1/D3): a blind document — no usable
    # text layer, so most modules never saw anything — must not read as a
    # clean bill of health. GENUINE/REVIEW collapse to UNVERIFIABLE; a
    # SUSPICIOUS band passes through untouched, because image-side tamper
    # evidence stands on its own ("a blind document can be proven
    # suspicious, never proven fine").
    if coverage_blind and band in ("GENUINE", "REVIEW"):
        return AggregateResult(score=0, band="UNVERIFIABLE",
                               risk_level="UNVERIFIABLE",
                               score_low=0, score_high=0,
                               ledger=list(evidence))
```

In `src/scoring.py`, change the `create_analysis_result` signature (line 119):

```python
def create_analysis_result(
    file_hash: str,
    module_results: list[ModuleResult],
    analysis_time_ms: int = 0,
    document_type: DocumentType = "unknown",
    coverage_blind: bool = False,
) -> AnalysisResult:
```

and change its aggregation call (line 177):

```python
    agg = aggregate_evidence(evidence, analyzable=analyzable,
                             coverage_blind=coverage_blind)
```

Also add one line to the docstring's Args section: `coverage_blind: True when the document has no usable text layer (see analyzer._coverage_blind); GENUINE/REVIEW results collapse to UNVERIFIABLE.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py tests/test_evidence.py tests/test_scoring.py -v`
Expected: all pass (existing evidence/scoring tests prove the default `False` path is unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/evidence.py src/scoring.py tests/test_coverage_gate.py
git commit -m "feat(coverage): UNVERIFIABLE band override for blind documents

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Analyzer wiring + summary theme/templates + integration tests

(Theme map and code emission MUST land in the same commit: `test_summary_drift.py` fails if `COVERAGE_NO_TEXT_LAYER` is emitted without a `FLAG_THEME_MAP` entry, and fails the orphan check if the entry exists without an emitter. Note: `TEMPLATE_REQUIRED_PREFIXES` in that test does NOT need a `COVERAGE_` entry — that list only forces templates for codes whose module messages are hardcoded French; our flag message is English AND we add EN/FR templates anyway.)

**Files:**
- Modify: `src/analyzer.py` (inside `analyze()`, after the ≥3-crash block ending line 244, before `elapsed_ms`)
- Modify: `src/summary.py:168-171` (theme map "other" block), `:617-624` (EN templates), `:1068-1076` (FR templates)
- Test: `tests/test_coverage_gate.py` (append)

**Interfaces:**
- Consumes: `_coverage_blind`, `MIN_TEXT_CHARS_PER_PAGE` (Task 1); `create_analysis_result(coverage_blind=)` (Task 2).
- Produces: analyzer emits module `"coverage"` / flag `COVERAGE_NO_TEXT_LAYER` with `details={"avg_chars_per_page": float, "threshold": int}`. Tasks 4–6 rely on that exact code string.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_coverage_gate.py`:

```python
from src.analyzer import TrustyFileAnalyzer

INVOICE_TEXT = (
    "FACTURE N° 2026-0142\n"
    "ACME Services SARL — 12 rue de la Paix, 75002 Paris\n"
    "Total HT : 1 000,00 EUR — TVA 20% : 200,00 EUR\n"
    "Total TTC : 1 200,00 EUR\n"
)


def _write_text_pdf(path) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    doc.save(str(path))
    doc.close()


def _write_image_only_pdf(path) -> None:
    src = fitz.open()
    page = src.new_page()
    page.insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    pix = page.get_pixmap(dpi=150)
    src.close()
    doc = fitz.open()
    out = doc.new_page(width=595, height=842)
    out.insert_image(out.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def _analyze(path):
    return TrustyFileAnalyzer(
        enable_external=False, enable_qr_scan=False).analyze(str(path))


def test_image_only_scan_bands_unverifiable(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "UNVERIFIABLE"
    assert result.risk_level == "UNVERIFIABLE"
    assert result.trust_score == 0
    codes = [f.code for m in result.modules for f in m.flags]
    assert "COVERAGE_NO_TEXT_LAYER" in codes


def test_text_document_is_unaffected(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_text_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "GENUINE"
    codes = [f.code for m in result.modules for f in m.flags]
    assert "COVERAGE_NO_TEXT_LAYER" not in codes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v -k "bands_unverifiable or unaffected"`
Expected: `test_image_only_scan_bands_unverifiable` FAILS (band is "GENUINE", score 96 — the bug this feature fixes); `test_text_document_is_unaffected` PASSES already.

- [ ] **Step 3: Wire the analyzer**

In `src/analyzer.py`, inside `analyze()`, AFTER the `if len(crashed_modules) >= 3:` block (ends line 244) and BEFORE `elapsed_ms = ...`, insert:

```python
        # Coverage gate (spec 2026-07-17): blind documents fail closed.
        # Runs after all modules so the identity MRZ exemption can consult
        # the identity module's actual outcome.
        coverage_blind, avg_chars = _coverage_blind(
            pdf_data.text_by_page, pdf_data.page_count, module_results)
        if coverage_blind:
            module_results.append(ModuleResult(
                module="coverage",
                flags=[Flag(
                    severity="low",  # display plumbing, not tamper evidence
                    code="COVERAGE_NO_TEXT_LAYER",
                    message=("Document has no machine-readable text layer "
                             "(scan or photo); content checks could not run"),
                    details={"avg_chars_per_page": round(avg_chars, 1),
                             "threshold": MIN_TEXT_CHARS_PER_PAGE},
                )],
                score=0,
                confidence=1.0,
            ))
```

Then change the `create_analysis_result` call below it (line 250) to pass the gate through:

```python
        result = create_analysis_result(
            file_hash=pdf_data.file_hash,
            module_results=module_results,
            analysis_time_ms=elapsed_ms,
            document_type=doc_type,
            coverage_blind=coverage_blind,
        )
```

(The extraction-failure early return at line 129 keeps its existing call — an unreadable file is already UNVERIFIABLE.)

- [ ] **Step 4: Add theme + sentence templates**

In `src/summary.py`, extend the "other" block of `FLAG_THEME_MAP` (after line 170, `"MODULE_ERROR": "other",`):

```python
    "COVERAGE_NO_TEXT_LAYER": "other",
```

In `SENTENCE_TEMPLATES`, after the `"MODULE_ERROR"` entry (line 623):

```python
    "COVERAGE_NO_TEXT_LAYER": lambda f: (
        "The document has no machine-readable text layer (scan or photo), "
        "so content checks could not run."),
```

In `SENTENCE_TEMPLATES_FR`, after the FR `"MODULE_ERROR"` entry (line 1075):

```python
    "COVERAGE_NO_TEXT_LAYER": lambda f: (
        "Le document ne contient aucune couche de texte exploitable (scan ou "
        "photo) ; les vérifications de contenu n'ont pas pu être effectuées."),
```

- [ ] **Step 5: Run integration + drift tests**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py tests/test_summary_drift.py -v`
Expected: all pass (drift gate discovers the new code AND both templates)

- [ ] **Step 6: Commit**

```bash
git add src/analyzer.py src/summary.py tests/test_coverage_gate.py
git commit -m "feat(coverage): wire blind gate into analyzer with themed flag

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Scan-specific UNVERIFIABLE verdict

**Files:**
- Modify: `src/summary.py` (constants near `VERDICTS_FR`'s end, line ~1141; branch in `generate_rich_summary`, line ~1466)
- Test: `tests/test_coverage_gate.py` (append)

**Interfaces:**
- Consumes: `COVERAGE_NO_TEXT_LAYER` emission (Task 3).
- Produces: module-level constants `COVERAGE_VERDICT`, `COVERAGE_VERDICT_FR` in `src.summary`; `generate_rich_summary` returns them for blind UNVERIFIABLE results.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coverage_gate.py`:

```python
from src.summary import generate_rich_summary


def test_blind_doc_gets_scan_specific_verdict(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_image_only_pdf(pdf)
    result = _analyze(pdf)
    assert "scanned or photographed" in generate_rich_summary(result, "en").verdict
    assert "scan ou une photo" in generate_rich_summary(result, "fr").verdict


def test_unreadable_doc_keeps_generic_unverifiable_verdict(tmp_path):
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    result = _analyze(bad)
    assert result.band == "UNVERIFIABLE"
    verdict = generate_rich_summary(result, "en").verdict
    assert "could not be read or verified" in verdict
    assert "scanned or photographed" not in verdict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v -k "verdict"`
Expected: `test_blind_doc_gets_scan_specific_verdict` FAILS (generic verdict returned); the garbage-pdf test PASSES already.

- [ ] **Step 3: Implement the verdict special-case**

In `src/summary.py`, after the closing brace of `VERDICTS_FR` (line 1141), add:

```python
# Coverage-specific UNVERIFIABLE verdicts (spec 2026-07-17). Selected by
# explicit code check in generate_rich_summary: the low-severity coverage
# flag can never win severity-weighted theme dominance, so the VERDICTS
# theme mechanism cannot route to it.
COVERAGE_VERDICT = (
    "This appears to be a scanned or photographed document — its content "
    "could not be machine-verified. Request the digital original."
)
COVERAGE_VERDICT_FR = (
    "Ce document semble être un scan ou une photo — son contenu n'a pas pu "
    "être vérifié automatiquement. Demandez l'original numérique."
)
```

In `generate_rich_summary` (line 1466), replace:

```python
    # Build verdict
    verdict = _build_verdict(result.risk_level, dominant_theme, lang)
```

with:

```python
    # Build verdict. Blind documents (coverage gate) get a scan-specific
    # UNVERIFIABLE verdict via explicit code check — see COVERAGE_VERDICT.
    if result.risk_level == "UNVERIFIABLE" and any(
            f.code == "COVERAGE_NO_TEXT_LAYER" for f in all_flags):
        verdict = COVERAGE_VERDICT_FR if lang == "fr" else COVERAGE_VERDICT
    else:
        verdict = _build_verdict(result.risk_level, dominant_theme, lang)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py tests/test_summary_verdict.py -v`
Expected: all pass (existing verdict tests prove non-blind paths unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/summary.py tests/test_coverage_gate.py
git commit -m "feat(coverage): scan-specific UNVERIFIABLE verdict (EN/FR)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Encrypted-file message rider

**Files:**
- Modify: `src/analyzer.py:118-126` (extraction-failure flag details)
- Modify: `src/summary.py:618-619` (EN `EXTRACTION_FAILED` template) and `:1069-1071` (FR)
- Test: `tests/test_coverage_gate.py` (append)

**Interfaces:**
- Consumes: `pdf_data.is_encrypted` (populated even on failed extraction — verified in the 2026-07-17 repro).
- Produces: `EXTRACTION_FAILED` flags carry `details["encrypted"]: bool`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_coverage_gate.py`:

```python
from src.summary import get_flag_sentence


def _write_encrypted_pdf(path) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), INVOICE_TEXT, fontsize=11)
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="secret-user-pw", owner_pw="secret-owner-pw")
    doc.close()


def test_encrypted_file_message_mentions_password(tmp_path):
    pdf = tmp_path / "locked.pdf"
    _write_encrypted_pdf(pdf)
    result = _analyze(pdf)
    assert result.band == "UNVERIFIABLE"  # already true today
    flag = next(f for m in result.modules for f in m.flags
                if f.code == "EXTRACTION_FAILED")
    assert flag.details.get("encrypted") is True
    assert "password" in get_flag_sentence(flag, "en").lower()
    assert "mot de passe" in get_flag_sentence(flag, "fr").lower()


def test_garbage_file_message_stays_generic(tmp_path):
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    result = _analyze(bad)
    flag = next(f for m in result.modules for f in m.flags
                if f.code == "EXTRACTION_FAILED")
    assert not flag.details.get("encrypted")
    assert "valid PDF" in get_flag_sentence(flag, "en")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v -k "password or generic"`
Expected: `test_encrypted_file_message_mentions_password` FAILS on the `details.get("encrypted")` assert; the garbage test PASSES already.

- [ ] **Step 3: Implement the rider**

In `src/analyzer.py`, in the extraction-failure block, change the flag's details line (line 123):

```python
                    details={"error": pdf_data.extraction_error,
                             "encrypted": pdf_data.is_encrypted},
```

In `src/summary.py`, replace the EN `EXTRACTION_FAILED` template (lines 618-619):

```python
    "EXTRACTION_FAILED": lambda f: (
        "The file is password-protected and could not be read, so it "
        "cannot be verified."
        if (f.details or {}).get("encrypted")
        else "The file could not be read as a valid PDF, so it cannot "
             "be verified."),
```

and the FR one (lines 1069-1071):

```python
    "EXTRACTION_FAILED": lambda f: (
        "Le fichier est protégé par mot de passe et n'a pas pu être lu ; "
        "il ne peut donc pas être vérifié."
        if (f.details or {}).get("encrypted")
        else "Le fichier n'a pas pu être lu comme un PDF valide et ne peut "
             "donc pas être vérifié."),
```

(The ≥3-crash `EXTRACTION_FAILED` flag in analyzer.py sets no `encrypted` key, so it renders the generic sentence — correct.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/analyzer.py src/summary.py tests/test_coverage_gate.py
git commit -m "fix(summary): encrypted files say password-protected, not invalid PDF

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Corpus — realistic OCR sandwich + unverifiable trap category

Two corpus consequences of the gate, found during planning:
(a) `_legit_ocr_sandwich` (scripts/build_corpus.py:236) carries ONE invisible span (~9 chars) — unrealistically sparse; under the gate it would band UNVERIFIABLE inside the LEGITIMATE list and silently distort `test_corpus_baseline`'s mean. Real OCR sandwiches carry full-page text; make the trap realistic.
(b) The trap set gains a new category `unverifiable/` so the new verdict class is corpus-enforced.

**Files:**
- Modify: `scripts/build_corpus.py:236-243` (`_legit_ocr_sandwich`), `:345-393` (`build_corpus`)
- Modify: `docs/superpowers/specs/2026-07-17-coverage-gate-design.md` (Tests section: correct the "all 30 legit corpus docs carry text layers" claim — the synthetic OCR sandwich was the counterexample; note the fix)
- Test: `tests/test_coverage_gate.py` (append)

**Interfaces:**
- Consumes: analyzer behavior from Tasks 1–3.
- Produces: `build_corpus(out_dir)` returns `{"legitimate": [...], "suspicious": [...], "unverifiable": [...]}`. Existing consumers (`tests/test_corpus_baseline.py` reads only the first two keys; `scripts/eval_harness.py` reads directories, ignores unknown ones) are unaffected — verify with `grep -rn "build_corpus(" tests/ scripts/`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coverage_gate.py`:

```python
from pathlib import Path

from scripts.build_corpus import build_corpus

CORPUS_UNVERIFIABLE = sorted(
    str(p) for p in
    (Path(__file__).resolve().parent.parent / "samples" / "unverifiable").glob("*.pdf")
)


def test_no_legit_corpus_doc_is_blind(tmp_path):
    """Permanent invariant: every synthetic LEGITIMATE trap doc must carry a
    realistic text layer. If this fails after adding a new trap doc, make the
    DOC realistic (real documents have text) — never weaken the gate."""
    paths = build_corpus(str(tmp_path))
    analyzer = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False)
    blind = [p for p in paths["legitimate"]
             if analyzer.analyze(p).band == "UNVERIFIABLE"]
    assert not blind, f"legit trap docs banded UNVERIFIABLE: {blind}"


def test_corpus_unverifiable_category(tmp_path):
    paths = build_corpus(str(tmp_path))
    assert paths["unverifiable"], "build_corpus must produce the unverifiable category"
    analyzer = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False)
    for p in paths["unverifiable"]:
        assert analyzer.analyze(p).band == "UNVERIFIABLE", p


@pytest.mark.skipif(not CORPUS_UNVERIFIABLE,
                    reason="local corpus not present (samples/ is gitignored)")
@pytest.mark.parametrize("pdf", CORPUS_UNVERIFIABLE)
def test_on_disk_unverifiable_corpus(pdf):
    result = TrustyFileAnalyzer(
        enable_external=False, enable_qr_scan=False).analyze(pdf)
    assert result.band == "UNVERIFIABLE", pdf
```

- [ ] **Step 2: Run tests to verify current state**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -v -k "corpus or blind"`
Expected: `test_no_legit_corpus_doc_is_blind` FAILS (legit_ocr_sandwich.pdf bands UNVERIFIABLE — the sparse-trap problem); `test_corpus_unverifiable_category` FAILS with `KeyError: 'unverifiable'`; the on-disk test SKIPS (no samples/unverifiable/ yet).

- [ ] **Step 3: Make the OCR sandwich realistic and add the unverifiable builder**

In `scripts/build_corpus.py`, replace `_legit_ocr_sandwich` (lines 236-243):

```python
def _legit_ocr_sandwich(dst: str) -> None:
    """Scan with a REALISTIC full-page invisible OCR layer (render_mode=3).
    Real OCR tools emit the whole page's text, not one span — the full layer
    both keeps this doc above the coverage gate's 50-chars/page threshold
    (it is machine-readable, so it IS verifiable) and stays >=50% invisible
    text so invisible.py's dominance gate correctly skips it."""
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
    pix.clear_with(255)
    page.insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)
    # The SIRET (valid Luhn, same one the project docs use in examples) keeps
    # the French-invoice-shaped text from newly tripping
    # CONTENT_MISSING_COMPANY_ID — this trap doc tests the invisible-text
    # path, not the missing-ID path.
    ocr_text = (
        "FACTURE N° SCAN-2026-018\n"
        "Fournisseur : ACME Services SARL\n"
        "SIRET : 552 081 317 66522\n"
        "12 rue de la Paix, 75002 Paris\n"
        "Client : Société Exemple SAS\n"
        "Prestation de conseil — juin 2026\n"
        "Total HT : 1 000,00 €\n"
        "TVA 20% : 200,00 €\n"
        "Total TTC : 1 200,00 €\n"
    )
    page.insert_textbox(fitz.Rect(50, 60, 545, 500), ocr_text,
                        fontname="helv", fontsize=12, render_mode=3)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()
```

Add a new builder directly below it:

```python
def _unverifiable_image_only_scan(dst: str) -> None:
    """Blind document: a 'scanned' invoice that is pure raster, no text layer.
    Expected band: UNVERIFIABLE (coverage gate). Before the gate this scored
    96/GENUINE — the fail-open hole the gate closes."""
    src = fitz.open(); page = src.new_page(width=595, height=842)
    page.insert_text(fitz.Point(60, 100),
                     "FACTURE N° 2026-0142 — Total TTC : 1 200,00 EUR",
                     fontname="helv", fontsize=12)
    pix = page.get_pixmap(dpi=150); src.close()
    doc = fitz.open(); out_page = doc.new_page(width=595, height=842)
    out_page.insert_image(out_page.rect, pixmap=pix)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()
```

In `build_corpus` (before the `return` at line 393), add:

```python
    uv = out / "unverifiable" / "image_only_scan.pdf"
    _unverifiable_image_only_scan(str(uv))

    return {"legitimate": legitimate, "suspicious": forged,
            "unverifiable": [str(uv)]}
```

(delete the old `return {"legitimate": legitimate, "suspicious": forged}` line.)

- [ ] **Step 4: Run the new tests and the corpus baselines**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py tests/test_corpus_baseline.py -v`
Expected: all pass; on-disk test still SKIPS.

- [ ] **Step 5: Regenerate the local on-disk corpus and run the corpus gates**

```bash
.venv/bin/python -c "from scripts.build_corpus import build_corpus; build_corpus('samples')"
```

Run: `.venv/bin/python -m pytest tests/test_corpus_bands.py tests/test_coverage_gate.py -v`
Expected: all pass — every `samples/legitimate/real/*.pdf` still GENUINE, every suspicious doc still non-GENUINE (one documented xfail), `samples/unverifiable/image_only_scan.pdf` bands UNVERIFIABLE.

- [ ] **Step 6: Amend the spec's corpus-impact claim**

In `docs/superpowers/specs/2026-07-17-coverage-gate-design.md`, in the Tests section, replace the sentence

`Corpus gates: all 30 legit corpus docs carry text layers → no band may move;`

with

`Corpus gates: all real legit corpus docs carry text layers; the synthetic
legit_ocr_sandwich trap carried only ~9 invisible chars (found during
planning) and was made realistic (full-page invisible OCR layer) so it stays
above the threshold — no legit band may move;`

- [ ] **Step 7: Commit**

```bash
git add scripts/build_corpus.py tests/test_coverage_gate.py docs/superpowers/specs/2026-07-17-coverage-gate-design.md
git commit -m "test(coverage): realistic OCR-sandwich trap + unverifiable corpus category

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Full-suite exit gate

**Files:**
- No source changes expected. Fixes only if the suite surfaces regressions.

**Interfaces:**
- Consumes: everything above.
- Produces: a green branch ready for adversarial review.

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures, 1 xfailed (forged_image_patch_over_text — pre-existing, documented). Test count ≥ 800 with the local corpus present.

- [ ] **Step 2: Spot-check the UI path manually**

```bash
.venv/bin/python - <<'EOF'
from scripts.build_corpus import build_corpus
from src.analyzer import TrustyFileAnalyzer
from src.summary import generate_rich_summary
import tempfile
paths = build_corpus(tempfile.mkdtemp())
r = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(paths["unverifiable"][0])
print(r.band, r.trust_score, generate_rich_summary(r, "fr").verdict)
EOF
```

Expected output: `UNVERIFIABLE 0 Ce document semble être un scan ou une photo — son contenu n'a pas pu être vérifié automatiquement. Demandez l'original numérique.`

- [ ] **Step 3: Commit any stragglers and stop**

If steps 1–2 required fixes, commit them with a descriptive message. Do NOT merge: the branch waits for adversarial review per the phase workflow.
