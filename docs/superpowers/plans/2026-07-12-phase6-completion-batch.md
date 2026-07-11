# Phase 6 Completion Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining audit-backlog items (17, 14, 21, 20, 24, 25, 23, 22, 26) on branch `phase6-robustness`: UBL e-invoice verification, 2D-Doc scoring, EXIF checks, author fingerprinting, and the P4 consistency cleanup.

**Architecture:** Every detection lands as flags consumed by the evidence ledger (`src/evidence.py`) — hard/soft × tamper/genuine, group-capped, anti-laundering-guarded. New checks are SOFT; positive signals join `GENUINE_CODES`. UI/summary sync happens after all new codes exist.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), PyMuPDF (fitz), defusedxml, Pillow, pylibdmtx (+ brew libdmtx), pytest.

**Spec:** `docs/superpowers/specs/2026-07-12-phase6-completion-batch-design.md` (user sign-offs D1–D5 inside).

## Global Constraints

- Interpreter: `.venv/bin/python` — plain `python` does not exist on this machine.
- Full suite must be green after every task: `.venv/bin/python -m pytest tests/ -q` (baseline: 671 passed, 1 xfailed).
- The corpus band gate (`tests/test_corpus_bands.py`) is THE zero-FP guarantee: every legit corpus doc stays GENUINE; band changes must be argued in the commit message, never silently loosened.
- New detection flags are SOFT (never added to `HARD_NEGATIVE_CODES`). Severity ceilings per spec D1/D5: 2D-Doc mismatch `high` (comparison confidence ≥ 0.95) / `medium` (below); EXIF/author checks max `medium`.
- "Silence, not suspicion": a module with nothing to check returns `ModuleResult(module=..., flags=[], score=100, confidence=0.5)` (0.3 when it looked and found nothing checkable).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Code/docstrings in English; summary templates need EN + FR pairs.
- Never parse untrusted XML with a non-defused parser (`defusedxml.ElementTree.fromstring` only).

---

### Task 1: UBL support in arithmetic (item 17)

**Files:**
- Modify: `src/modules/arithmetic.py` (extend `xml_totals`, `_xml_ibans`, `_extract_einvoice_xml`)
- Test: `tests/test_arithmetic.py` (append)
- Test: `tests/test_corpus_bands.py` (no change expected — verify only)

**Interfaces:**
- Consumes: `_local(tag)`, `_xml_fromstring`, `validate_iban_checksum`, `TOLERANCE` — all already in `arithmetic.py`.
- Produces: `xml_totals(xml_bytes)` and `_xml_ibans(xml_bytes)` now understand UBL; signatures unchanged (bytes in, same dict/set out). `_extract_einvoice_xml` additionally recognizes any `.xml` attachment whose root is a CII/UBL invoice. No new public names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arithmetic.py`:

```python
# =============================================================================
# UBL (OASIS / Peppol) — the second EN 16931 syntax (audit item 17)
# =============================================================================

UBL_NS = 'xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"'
CAC = 'xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"'
CBC = 'xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"'


def _ubl_xml(payable="120.00", tax_excl="100.00", tax_incl="120.00",
             tax="20.00", iban="FR7630006000011234567890189",
             root="Invoice") -> bytes:
    """Minimal UBL invoice with document-level totals, tax and payee IBAN."""
    ns = UBL_NS.replace("Invoice-2", f"{root}-2")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<{root} {ns} {CAC} {CBC}>
  <cbc:ID>UBL-TEST-1</cbc:ID>
  <cac:PaymentMeans>
    <cac:PayeeFinancialAccount>
      <cbc:ID>{iban}</cbc:ID>
    </cac:PayeeFinancialAccount>
  </cac:PaymentMeans>
  <cac:TaxTotal>
    <cbc:TaxAmount currencyID="EUR">{tax}</cbc:TaxAmount>
  </cac:TaxTotal>
  <cac:InvoiceLine>
    <cac:TaxTotal>
      <cbc:TaxAmount currencyID="EUR">999.99</cbc:TaxAmount>
    </cac:TaxTotal>
  </cac:InvoiceLine>
  <cac:LegalMonetaryTotal>
    <cbc:TaxExclusiveAmount currencyID="EUR">{tax_excl}</cbc:TaxExclusiveAmount>
    <cbc:TaxInclusiveAmount currencyID="EUR">{tax_incl}</cbc:TaxInclusiveAmount>
    <cbc:PayableAmount currencyID="EUR">{payable}</cbc:PayableAmount>
  </cac:LegalMonetaryTotal>
</{root}>""".encode()


class TestUblTotals:
    def test_ubl_totals_parsed(self):
        """UBL LegalMonetaryTotal maps into the same dict as CII."""
        totals = xml_totals(_ubl_xml())
        assert totals is not None
        assert totals["basis"] == 100.00
        assert totals["tax"] == 20.00
        assert totals["grand"] == 120.00
        assert totals["due"] == 120.00
        assert 120.00 in totals["totals"] and 100.00 in totals["totals"]

    def test_ubl_line_level_taxtotal_ignored(self):
        """The InvoiceLine TaxTotal (999.99) must NOT pollute the document tax."""
        totals = xml_totals(_ubl_xml())
        assert totals["tax"] == 20.00

    def test_ubl_internal_inconsistency_flagged(self):
        """100 + 20 != 130 -> ARITH_XML_INTERNAL_INCONSISTENT."""
        totals = xml_totals(_ubl_xml(tax_incl="130.00", payable="130.00"))
        flag = internal_check(totals)
        assert flag is not None and flag.code == "ARITH_XML_INTERNAL_INCONSISTENT"

    def test_ubl_internal_consistency_clears(self):
        assert internal_check(xml_totals(_ubl_xml())) is None

    def test_ubl_creditnote_root_recognized(self):
        totals = xml_totals(_ubl_xml(root="CreditNote"))
        assert totals is not None and totals["grand"] == 120.00

    def test_ubl_without_taxinclusive_skips(self):
        """No grand total -> None (nothing to check), same as CII."""
        xml = _ubl_xml().replace(
            b"<cbc:TaxInclusiveAmount currencyID=\"EUR\">120.00</cbc:TaxInclusiveAmount>", b"")
        assert xml_totals(xml) is None


class TestUblIbans:
    def test_ubl_payee_iban_extracted(self):
        assert _xml_ibans(_ubl_xml()) == {"FR7630006000011234567890189"}

    def test_ubl_invalid_iban_filtered(self):
        """A non-mod-97 ID under PayeeFinancialAccount is dropped."""
        assert _xml_ibans(_ubl_xml(iban="FR7630006000011234567890000")) == set()

    def test_ubl_iban_mismatch_flag(self):
        """Visible IBAN disjoint from UBL payee IBAN -> ARITH_IBAN_MISMATCH."""
        text = "Payez sur IBAN DE89370400440532013000 merci"
        flag = check_iban(text, _ubl_xml())
        assert flag is not None and flag.code == "ARITH_IBAN_MISMATCH"

    def test_ubl_iban_overlap_clears(self):
        text = "IBAN FR76 3000 6000 0112 3456 7890 189"
        assert check_iban(text, _ubl_xml()) is None
```

Also add these imports at the top of the test file **only if not already imported**: `from src.modules.arithmetic import xml_totals, internal_check, check_iban, _xml_ibans`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_arithmetic.py -q -k "Ubl"`
Expected: FAIL — `xml_totals(_ubl_xml())` returns `None` (no `GrandTotalAmount` in UBL), IBAN set empty.

- [ ] **Step 3: Implement UBL normalization in `src/modules/arithmetic.py`**

3a. Add a shared root parser and UBL constants right after `_local()`:

```python
# UBL (OASIS) — the OTHER EN 16931 syntax (Peppol). Recognized by ROOT element
# local name; CII documents root at CrossIndustryInvoice instead.
_UBL_ROOTS = ("Invoice", "CreditNote")
_CII_ROOT = "CrossIndustryInvoice"


def _xml_root(xml_bytes: bytes | None):
    """Defused-parsed root element, or None on empty/malformed/blocked input.

    Single parse point shared by xml_totals and _xml_ibans so the
    CII-vs-UBL dispatch happens on the actual document structure, never on
    the attachment's filename. Never raises.
    """
    if not xml_bytes:
        return None
    try:
        return _xml_fromstring(xml_bytes)
    except Exception:
        # Malformed XML, or a blocked XXE/entity-bomb payload → skip the check
        # (no crash, no flag) — same policy as the CII path.
        return None
```

3b. Add the UBL totals reader after `xml_totals`:

```python
def _ubl_totals(root) -> dict | None:
    """UBL LegalMonetaryTotal → the same {basis, tax, grand, due, totals} dict
    the CII path produces, so every downstream check works unchanged.

    Mapping (EN 16931 business terms):
      TaxExclusiveAmount → basis   (BT-109)
      TaxInclusiveAmount → grand   (BT-112, required — None without it)
      PayableAmount      → due     (BT-115)
      TaxTotal/TaxAmount → tax     (BT-110; document level = DIRECT child of
                                    the root — line-level TaxTotal lives under
                                    InvoiceLine and must not pollute the sum)
    """
    lmt = next((el for el in root.iter()
                if _local(el.tag) == "LegalMonetaryTotal"), None)
    if lmt is None:
        return None

    out: dict = {}
    for key, tag in (("basis", "TaxExclusiveAmount"),
                     ("grand", "TaxInclusiveAmount"),
                     ("due", "PayableAmount")):
        child = next((c for c in lmt if _local(c.tag) == tag), None)
        if child is not None and child.text:
            try:
                out[key] = float(child.text.strip())
            except ValueError:
                pass

    # Document-level tax: TaxTotal elements that are DIRECT children of the
    # root. UBL allows a second TaxTotal in the accounting currency (BT-111);
    # summing currencies would corrupt the identity, so when several document
    # TaxTotals exist we keep only amounts in the LegalMonetaryTotal currency.
    lmt_currency = next((c.get("currencyID") for c in lmt if c.get("currencyID")), None)
    tax_sum, seen = 0.0, False
    for tt in root:
        if _local(tt.tag) != "TaxTotal":
            continue
        ta = next((c for c in tt if _local(c.tag) == "TaxAmount"), None)
        if ta is None or not ta.text:
            continue
        if lmt_currency and ta.get("currencyID") and ta.get("currencyID") != lmt_currency:
            continue  # accounting-currency duplicate (BT-111), not the doc tax
        try:
            tax_sum += float(ta.text.strip())
            seen = True
        except ValueError:
            pass
    if seen:
        out["tax"] = tax_sum

    if "grand" not in out:
        return None  # without a grand total there is nothing to check
    out.setdefault("due", out["grand"])
    out["totals"] = {round(out[k], 2) for k in ("grand", "due", "basis") if k in out}
    return out
```

3c. Rewrite the body of `xml_totals` to dispatch (keep its docstring, extend it with one line: "Understands both EN 16931 syntaxes: CII (Factur-X/ZUGFeRD/XRechnung) and UBL (Peppol)."):

```python
    root = _xml_root(xml_bytes)
    if root is None:
        return None

    if _local(root.tag) in _UBL_ROOTS:
        return _ubl_totals(root)

    summation = next(
        (el for el in root.iter()
         if any(_local(c.tag) == _GRAND_TAG for c in el)),
        None,
    )
    if summation is None:
        return None

    out: dict = {}
    for key, tag in _XML_FIELDS.items():
        child = next((c for c in summation if _local(c.tag) == tag), None)
        if child is not None and child.text:
            try:
                out[key] = float(child.text.strip())
            except ValueError:
                pass
    if "grand" not in out:
        return None  # without a grand total there is nothing to check
    out.setdefault("due", out["grand"])
    out["totals"] = {round(out[k], 2) for k in ("grand", "due", "basis") if k in out}
    return out
```

(The old `try/except _xml_fromstring` block at the top of `xml_totals` is replaced by `_xml_root`.)

3d. In `_xml_ibans`, replace the parse block and add the UBL branch. The full new body:

```python
    root = _xml_root(xml_bytes)
    if root is None:
        return set()

    out: set[str] = set()
    if _local(root.tag) in _UBL_ROOTS:
        # UBL: cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID — the account
        # the invoice asks to be PAID TO (same semantics as the CII
        # PayeePartyCreditorFinancialAccount below).
        for acct in root.iter():
            if _local(acct.tag) != "PayeeFinancialAccount":
                continue
            for el in acct:
                if _local(el.tag) == "ID" and el.text:
                    cand = "".join(el.text.split()).upper()
                    if 15 <= len(cand) <= 34 and validate_iban_checksum(cand):
                        out.add(cand)
        return out

    for acct in root.iter():
        if _local(acct.tag) != "PayeePartyCreditorFinancialAccount":
            continue
        for el in acct.iter():
            if _local(el.tag) == "IBANID" and el.text:
                cand = "".join(el.text.split()).upper()
                if 15 <= len(cand) <= 34 and validate_iban_checksum(cand):
                    out.add(cand)
    return out
```

3e. Extend `_extract_einvoice_xml` so a generically-named XML attachment is recognized by its ROOT (spec: "not by filename alone"). Replace the loop body:

```python
        for i in range(doc.embfile_count()):
            try:
                name = doc.embfile_info(i).get("filename", "")
            except Exception:
                continue
            if _is_einvoice_attachment(name):
                return doc.embfile_get(i)
            # Fallback: a generically-named .xml attachment still counts if its
            # ROOT element is an e-invoice (CII or UBL) — recognition by
            # structure, not filename.
            if name.lower().endswith(".xml"):
                try:
                    data = doc.embfile_get(i)
                except Exception:
                    continue
                root = _xml_root(data)
                if root is not None and _local(root.tag) in (_CII_ROOT, *_UBL_ROOTS):
                    return data
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_arithmetic.py -q`
Expected: PASS (all, including pre-existing CII tests — the dispatch must not disturb them).

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green. `test_corpus_bands.py` in particular: `i2d_AzureInterior.pdf` must remain GENUINE (its UBL attachment now gets parsed; if its printed total matches the XML it may now earn `ARITH_XML_CONSISTENT` — score can only rise).

- [ ] **Step 5: Manual corpus verification**

Run:
```bash
.venv/bin/python -c "
from src.analyzer import TrustyFileAnalyzer
r = TrustyFileAnalyzer(enable_external=False).analyze('samples/legitimate/real/i2d_AzureInterior.pdf')
arith = [m for m in r.modules if m.module == 'arithmetic'][0]
print('score:', r.trust_score, r.band)
print('arithmetic flags:', [(f.severity, f.code) for f in arith.flags])
"
```
Expected: band GENUINE; arithmetic now reports either `ARITH_XML_CONSISTENT` or (if the visible total genuinely disagrees with the XML) a mismatch flag — inspect and, if a mismatch fires on this LEGIT doc, treat it as an FP: debug the UBL total mapping before proceeding (the doc is the calibration reference).

- [ ] **Step 6: Commit**

```bash
git add src/modules/arithmetic.py tests/test_arithmetic.py
git commit -m "feat(arithmetic): UBL (Peppol) e-invoice support via totals/IBAN normalization

Item 17: UBL LegalMonetaryTotal + document-level TaxTotal + PayeeFinancialAccount
map into the same internal dict as CII, so all four ARITH_* codes (incl. the
ARITH_XML_CONSISTENT genuine path) work unchanged. Generic .xml attachments are
recognized by root element, not filename.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: 2D-Doc scoring module (item 14, decisions D1/D2)

**Files:**
- Modify: `src/modules/twod_doc.py` (append `_flags_from_verification` + `analyze_twod_doc`)
- Modify: `src/evidence.py:40-44` (add `TWOD_DOC_CONSISTENT` to `GENUINE_CODES`)
- Modify: `src/analyzer.py` (import + `_run` registration)
- Test: `tests/test_twod_doc_module.py` (new file)

**Interfaces:**
- Consumes: `scan_pdf_for_2d_doc(pdf_path) -> list[TwoDocData]`, `verify_2d_doc_against_pdf(twod_doc, pdf_text) -> TwoDocVerificationResult`, `parse_twod_doc(raw) -> TwoDocData | None`, `PYLIBDMTX_AVAILABLE`, `GS`, `US` — all already in `twod_doc.py`; `Flag`, `ModuleResult` from `src.models`.
- Produces: `analyze_twod_doc(pdf_data) -> ModuleResult` (module name `"twod_doc"`); pure helper `_flags_from_verification(result: TwoDocVerificationResult) -> list[Flag]`; flag codes `TWOD_DOC_FIELD_MISMATCH` and `TWOD_DOC_CONSISTENT` (Task 9 maps them; Task 8 shows the module in the "authenticity" bar group).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_twod_doc_module.py`:

```python
"""
Module-level tests for analyze_twod_doc (audit item 14).

Uses parse_twod_doc on synthetic raw payloads (the same construction as the
module's __main__ harness) — no DataMatrix decoding, so these run without
libdmtx. The decode-dependent E2E lives in test_twod_doc_e2e.py.
"""
from src.models import ModuleResult
from src.modules.twod_doc import (
    GS,
    US,
    _flags_from_verification,
    analyze_twod_doc,
    parse_twod_doc,
    verify_2d_doc_against_pdf,
)

# Header: DC + version 03 + FR00 + 0001 + emit 2345 + sig 2345 + type 01 + perimeter 01
_HEADER = "DC03FR000001234523450101"


def _make_twod_doc(fields: dict[str, str]):
    """Build a parsed TwoDocData from {DI: value} using the real parser."""
    message = ""
    for di, value in fields.items():
        message += di + value
        if di != "24":                      # DI 24 (postal code) is fixed-length
            message += GS
    raw = _HEADER + message.rstrip(GS) + US + "FAKESIG"
    td = parse_twod_doc(raw)
    assert td is not None, "synthetic 2D-Doc failed to parse — fix the fixture"
    return td


MATCHING_TEXT = """AVIS D'IMPOT
JEAN DUPONT
123 RUE DE PARIS
75001 PARIS
Facture FAC-2024-001
"""


class TestFlagsFromVerification:
    def test_all_match_yields_consistent(self):
        td = _make_twod_doc({"10": "JEAN DUPONT", "25": "PARIS", "18": "FAC-2024-001"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        assert [f.code for f in flags] == ["TWOD_DOC_CONSISTENT"]
        assert flags[0].severity == "low"

    def test_single_field_no_consistent(self):
        """<2 comparisons must NOT earn the genuine flag (spec D2)."""
        td = _make_twod_doc({"25": "PARIS"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        assert _flags_from_verification(res) == []

    def test_fuzzy_field_mismatch_is_medium(self):
        """Name (confidence < 0.95) mismatch -> medium (spec D1)."""
        td = _make_twod_doc({"10": "MARIE MARTIN", "25": "PARIS", "18": "FAC-2024-001"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        mismatches = [f for f in flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].severity == "medium"
        assert mismatches[0].details["di"] == "10"
        assert "TWOD_DOC_CONSISTENT" not in [f.code for f in flags]

    def test_exact_field_mismatch_is_high(self):
        """Invoice number (confidence 1.0) mismatch -> high (spec D1)."""
        td = _make_twod_doc({"10": "JEAN DUPONT", "18": "FAC-9999-999"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        flags = _flags_from_verification(res)
        mismatches = [f for f in flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].severity == "high"

    def test_postal_code_mismatch_is_high(self):
        td = _make_twod_doc({"24": "69001", "25": "PARIS", "10": "JEAN DUPONT"})
        res = verify_2d_doc_against_pdf(td, MATCHING_TEXT)
        highs = [f for f in _flags_from_verification(res)
                 if f.code == "TWOD_DOC_FIELD_MISMATCH" and f.severity == "high"]
        assert len(highs) == 1


class TestAnalyzeTwodDoc:
    def test_extraction_failed_neutral(self):
        class FakePDFData:
            extraction_ok = False
            file_path = "/nonexistent.pdf"
            text_by_page = []
        r = analyze_twod_doc(FakePDFData())
        assert isinstance(r, ModuleResult)
        assert r.module == "twod_doc" and r.flags == [] and r.score == 100

    def test_no_barcode_neutral(self, tmp_path):
        """A plain text-only PDF has no DataMatrix: neutral, never a penalty."""
        import fitz
        p = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Facture 100,00 EUR")
        doc.save(str(p)); doc.close()

        class FakePDFData:
            extraction_ok = True
            file_path = str(p)
            text_by_page = ["Facture 100,00 EUR"]
        r = analyze_twod_doc(FakePDFData())
        assert r.flags == [] and r.score == 100
        assert r.confidence <= 0.5


class TestEvidenceRegistration:
    def test_consistent_is_genuine_direction(self):
        from src.evidence import GENUINE_CODES, flag_to_evidence
        from src.models import Flag
        assert "TWOD_DOC_CONSISTENT" in GENUINE_CODES
        ev = flag_to_evidence(
            Flag(severity="low", code="TWOD_DOC_CONSISTENT", message="x"),
            "twod_doc")
        assert ev.direction == "genuine" and ev.kind == "soft"

    def test_mismatch_is_soft_tamper(self):
        from src.evidence import HARD_NEGATIVE_CODES, flag_to_evidence
        from src.models import Flag
        assert "TWOD_DOC_FIELD_MISMATCH" not in HARD_NEGATIVE_CODES
        ev = flag_to_evidence(
            Flag(severity="high", code="TWOD_DOC_FIELD_MISMATCH", message="x"),
            "twod_doc")
        assert ev.direction == "tamper" and ev.kind == "soft"


class TestAnalyzerRegistration:
    def test_module_runs_in_pipeline(self, tmp_path):
        """The analyzer must include a twod_doc ModuleResult."""
        import fitz
        p = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "hello facture invoice total ttc tva 20")
        doc.save(str(p)); doc.close()
        from src.analyzer import TrustyFileAnalyzer
        result = TrustyFileAnalyzer(enable_external=False).analyze(str(p))
        assert "twod_doc" in [m.module for m in result.modules]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_twod_doc_module.py -q`
Expected: FAIL — `ImportError: cannot import name '_flags_from_verification'`.

- [ ] **Step 3: Implement in `src/modules/twod_doc.py`**

Append after `verify_pdf_with_2d_doc` (before the `__main__` block). Note the imports: `Flag` and `ModuleResult` — add `from src.models import Flag, ModuleResult` next to the module's other imports if not present.

```python
# =============================================================================
# SCORED MODULE WRAPPER (audit item 14)
# =============================================================================
# Severity policy (spec D1, 2026-07-12): a field mismatch is SOFT evidence —
# text matching misses on layout/accents/hyphenation, so it is never a hard
# veto. Tiered off the comparison confidence already encoded per field:
# >= 0.95 (amounts, RFR, fiscal/invoice numbers, postal code, calculated
# balance) -> high; below (name, address, city, income estimate) -> medium.
# Severities are PROVISIONAL until a real 2D-Doc sample joins the corpus.
_EXACT_CONFIDENCE = 0.95

# Positive signal (spec D2): a fully-matching 2D-Doc with >= 2 compared fields
# is soft GENUINE evidence (TWOD_DOC_CONSISTENT in evidence.GENUINE_CODES) —
# same policy as ARITH_XML_CONSISTENT; the anti-laundering guard strips it
# whenever meaningful tamper evidence exists.
_MIN_COMPARISONS_FOR_CONSISTENT = 2


def _flags_from_verification(result: TwoDocVerificationResult) -> list[Flag]:
    """Convert one 2D-Doc verification into scoring flags.

    Pure function (no I/O): the unit-testable core of analyze_twod_doc.

    Example:
        >>> res = verify_2d_doc_against_pdf(td, pdf_text)
        >>> [f.code for f in _flags_from_verification(res)]
        ['TWOD_DOC_CONSISTENT']
    """
    flags: list[Flag] = []
    for c in result.comparisons:
        if c.match:
            continue
        severity = "high" if c.confidence >= _EXACT_CONFIDENCE else "medium"
        flags.append(Flag(
            severity=severity,
            code="TWOD_DOC_FIELD_MISMATCH",
            message=(f"The document's signed 2D-Doc barcode contains "
                     f"{c.field_name} = \"{c.twod_doc_value}\", but that value "
                     f"was not found in the visible text. The visible layer "
                     f"may have been edited (the barcode data cannot be)."),
            details={"di": c.di, "field": c.field_name,
                     "expected": c.twod_doc_value,
                     "comparison_confidence": c.confidence},
        ))

    if (not flags and len(result.comparisons) >= _MIN_COMPARISONS_FOR_CONSISTENT
            and result.overall_match):
        flags.append(Flag(
            severity="low",   # weight comes from GENUINE_CODES direction
            code="TWOD_DOC_CONSISTENT",
            message=(f"All {len(result.comparisons)} fields of the document's "
                     "signed 2D-Doc barcode match the visible text — "
                     "machine-verified consistency evidence."),
            details={"fields_compared": len(result.comparisons)},
        ))
    return flags


def analyze_twod_doc(pdf_data) -> ModuleResult:
    """Scored module: cross-check 2D-Doc barcodes against the visible text.

    Neutral ("silence, not suspicion") whenever there is nothing to check:
    extraction failed, pylibdmtx missing, or no 2D-Doc present — most
    documents carry none, and absence is never evidence. A DataMatrix that
    fails 2D-Doc parsing is also NOT flagged (shipping labels are DataMatrix
    too; scan_pdf_for_2d_doc already drops non-2D-Doc payloads).

    Never raises: any scan failure degrades to a neutral low-confidence
    result (the analyzer's crash isolation is the backstop, not the plan).
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="twod_doc", flags=[], score=100, confidence=0.5)
    if not PYLIBDMTX_AVAILABLE:
        return ModuleResult(module="twod_doc", flags=[], score=100, confidence=0.5)

    try:
        twod_docs = scan_pdf_for_2d_doc(pdf_data.file_path)
    except Exception as exc:
        logger.warning("twod_doc: scan failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        return ModuleResult(module="twod_doc", flags=[], score=100, confidence=0.5)

    if not twod_docs:
        # Scanned and found nothing checkable — same convention as arithmetic.
        return ModuleResult(module="twod_doc", flags=[], score=100, confidence=0.3)

    text = "\n".join(getattr(pdf_data, "text_by_page", []) or [])
    flags: list[Flag] = []
    for td in twod_docs:
        try:
            flags.extend(_flags_from_verification(verify_2d_doc_against_pdf(td, text)))
        except Exception as exc:
            logger.warning("twod_doc: verification failed: %s", exc)

    score = 100
    for f in flags:
        if f.code == "TWOD_DOC_CONSISTENT":
            continue  # positive signal — never reduces the module score
        score -= {"low": 5, "medium": 15, "high": 30, "critical": 50}.get(f.severity, 0)
    return ModuleResult(module="twod_doc", flags=flags, score=max(0, score),
                        confidence=1.0)
```

- [ ] **Step 4: Register the genuine code in `src/evidence.py`**

In `GENUINE_CODES` (line ~40), add one entry:

```python
GENUINE_CODES: frozenset[str] = frozenset({
    "STRUCT_SIGNATURE_TRUSTED",          # EU-trusted, validated signature
    "STRUCT_SIGNATURE_TRUSTED_EXPIRED",  # QC signature, cert since expired
    "ARITH_XML_CONSISTENT",              # visible layer matches embedded e-invoice XML (totals/IBAN)
    "TWOD_DOC_CONSISTENT",               # visible layer matches the signed 2D-Doc barcode fields
})
```

- [ ] **Step 5: Register the module in `src/analyzer.py`**

Change the import at line 34 from `from src.modules.twod_doc import PYLIBDMTX_AVAILABLE` to:

```python
from src.modules.twod_doc import PYLIBDMTX_AVAILABLE, analyze_twod_doc
```

Add after the `_run("forensics", ...)` line:

```python
        _run("twod_doc", lambda: analyze_twod_doc(pdf_data))                     # 2D-Doc cross-check
```

- [ ] **Step 6: Run the tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_twod_doc_module.py -q`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green — the corpus has no 2D-Doc documents, so the corpus gate must be untouched. If any corpus doc changes band, STOP and investigate (would mean a false DataMatrix hit).

- [ ] **Step 7: Commit**

```bash
git add src/modules/twod_doc.py src/evidence.py src/analyzer.py tests/test_twod_doc_module.py
git commit -m "feat(twod_doc): score the 2D-Doc cross-check (TWOD_DOC_FIELD_MISMATCH / _CONSISTENT)

Item 14, decisions D1/D2: soft tiered mismatches (high for exact fields via
comparison confidence >= 0.95, medium for fuzzy), full match with >=2 fields
is soft genuine evidence under the anti-laundering guard. Neutral when
pylibdmtx or the barcode is absent. Severities provisional until a real
2D-Doc sample joins the corpus.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: libdmtx install + synthetic 2D-Doc E2E (decision D3)

**Files:**
- Test: `tests/test_twod_doc_e2e.py` (new file)
- Modify: none (environment + test only)

**Interfaces:**
- Consumes: `analyze_twod_doc`, `parse_twod_doc`, `PYLIBDMTX_AVAILABLE` (Task 2); `pylibdmtx.pylibdmtx.encode`.
- Produces: nothing new — an executable proof that decode→parse→verify→flags works end-to-end.

- [ ] **Step 1: Install the system library**

Run: `brew install libdmtx`
(This is the one environment mutation in the plan — decision D3. It may prompt for user permission.)

Verify: `.venv/bin/python -c "from src.modules.twod_doc import PYLIBDMTX_AVAILABLE; print(PYLIBDMTX_AVAILABLE)"`
Expected: `True`. If still `False`, check `python -c "from pylibdmtx import pylibdmtx"` for the load error (an arm64 Homebrew path issue would surface here — `brew --prefix libdmtx` must be on the dylib search path; if needed, document the fix in the commit).

Note: `packages.txt` already ships `libdmtx0b` for Streamlit Cloud — nothing to change there.

- [ ] **Step 2: Write the E2E test**

Create `tests/test_twod_doc_e2e.py`:

```python
"""
End-to-end 2D-Doc test: ENCODE a synthetic 2D-Doc payload into a real
DataMatrix image, embed it in a PDF, and run the full scored module on it.

Requires the libdmtx system library (brew install libdmtx / apt libdmtx0b);
skips cleanly when absent so CI without the dep stays green.
"""
import io

import pytest

from src.modules.twod_doc import GS, US, PYLIBDMTX_AVAILABLE, analyze_twod_doc

pytestmark = pytest.mark.skipif(
    not PYLIBDMTX_AVAILABLE, reason="libdmtx system library not installed")

_HEADER = "DC03FR000001234523450101"
_MESSAGE = (
    "10" + "JEAN DUPONT" + GS +
    "22" + "123 RUE DE PARIS" + GS +
    "24" + "75001" +
    "25" + "PARIS" + GS +
    "18" + "FAC-2024-001"
)
_PAYLOAD = _HEADER + _MESSAGE + US + "FAKESIGNATURE123"


def _pdf_with_datamatrix(tmp_path, visible_text: str) -> str:
    """A one-page PDF: DataMatrix top-left, visible_text below it."""
    import fitz
    from PIL import Image
    from pylibdmtx.pylibdmtx import encode

    enc = encode(_PAYLOAD.encode("latin-1"))
    img = Image.frombytes("RGB", (enc.width, enc.height), enc.pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    p = tmp_path / "twod.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(36, 36, 156, 156), stream=buf.getvalue())
    y = 200
    for line in visible_text.splitlines():
        page.insert_text((36, y), line, fontsize=11)
        y += 16
    doc.save(str(p))
    doc.close()
    return str(p)


class _PDFData:
    extraction_ok = True

    def __init__(self, path, text):
        self.file_path = path
        self.text_by_page = [text]


def test_matching_document_earns_consistent(tmp_path):
    text = "JEAN DUPONT\n123 RUE DE PARIS\n75001 PARIS\nFacture FAC-2024-001"
    path = _pdf_with_datamatrix(tmp_path, text)
    r = analyze_twod_doc(_PDFData(path, text))
    assert [f.code for f in r.flags] == ["TWOD_DOC_CONSISTENT"], (
        f"decode->verify pipeline broken; flags={[(f.severity, f.code) for f in r.flags]}")


def test_edited_invoice_number_flagged_high(tmp_path):
    text = "JEAN DUPONT\n123 RUE DE PARIS\n75001 PARIS\nFacture FAC-9999-999"
    path = _pdf_with_datamatrix(tmp_path, text)
    r = analyze_twod_doc(_PDFData(path, text))
    mismatches = [f for f in r.flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
    assert mismatches, "edited value must be caught by the signed barcode"
    assert any(f.severity == "high" and f.details["di"] == "18" for f in mismatches)
    assert "TWOD_DOC_CONSISTENT" not in [f.code for f in r.flags]
```

- [ ] **Step 3: Run the E2E tests**

Run: `.venv/bin/python -m pytest tests/test_twod_doc_e2e.py -v`
Expected: 2 PASS (not skipped — Step 1 installed the lib). If `scan_pdf_for_2d_doc` misses the barcode at default DPI, raise the inserted image size (Rect to 216×216) rather than changing module defaults.

- [ ] **Step 4: Full suite + commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

```bash
git add tests/test_twod_doc_e2e.py
git commit -m "test(twod_doc): synthetic DataMatrix end-to-end gate (encode->scan->verify->flags)

Decision D3: libdmtx installed locally; test skips cleanly where the system
lib is absent. Proves the full decode pipeline on both a matching and an
edited document.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: EXIF checks in images (item 21, decision D5)

**Files:**
- Modify: `src/modules/images.py` (add `_read_exif`, `check_exif_metadata`; wire into `analyze_images`)
- Test: `tests/test_images.py` (append)

**Interfaces:**
- Consumes: `extract_images_from_pdf(pdf_path) -> list[ImageInfo]` (has `.xref`), `pdf_data.metadata.creation_date` (tz-aware UTC `datetime | None`), Pillow.
- Produces: `_read_exif(image_bytes) -> tuple[datetime | None, str | None]` (naive EXIF datetime, Software string); `check_exif_metadata(pdf_path, doc_creation_date) -> list[Flag]`; codes `IMAGES_EXIF_DATE_AFTER_DOC` (medium), `IMAGES_EXIF_EDITOR_SOFTWARE` (low). At most ONE flag of each code per document.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_images.py`:

```python
# =============================================================================
# EXIF checks (audit item 21): date-after-doc (medium) + editor software (low)
# =============================================================================
import io
from datetime import datetime, timezone

from PIL import Image

from src.modules.images import _read_exif, check_exif_metadata


def _jpeg_bytes(exif_datetime: str | None = None, software: str | None = None) -> bytes:
    img = Image.new("RGB", (40, 40), "white")
    exif = Image.Exif()
    if exif_datetime:
        exif[306] = exif_datetime            # DateTime
    if software:
        exif[305] = software                 # Software
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _pdf_with_jpeg(tmp_path, jpeg: bytes) -> str:
    import fitz
    p = tmp_path / "img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(36, 36, 136, 136), stream=jpeg)
    doc.save(str(p)); doc.close()
    return str(p)


DOC_DATE = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


class TestReadExif:
    def test_reads_datetime_and_software(self):
        dt, sw = _read_exif(_jpeg_bytes("2026:03:01 10:00:00", "Adobe Photoshop 25.0"))
        assert dt == datetime(2026, 3, 1, 10, 0, 0)
        assert sw == "Adobe Photoshop 25.0"

    def test_no_exif(self):
        assert _read_exif(_jpeg_bytes()) == (None, None)

    def test_corrupt_bytes_silent(self):
        assert _read_exif(b"\xff\xd8not-really-a-jpeg") == (None, None)


class TestExifChecks:
    def test_image_created_after_doc_flagged_medium(self, tmp_path):
        path = _pdf_with_jpeg(tmp_path, _jpeg_bytes("2026:03:01 10:00:00"))
        flags = check_exif_metadata(path, DOC_DATE)
        assert [f.code for f in flags] == ["IMAGES_EXIF_DATE_AFTER_DOC"]
        assert flags[0].severity == "medium"

    def test_image_created_before_doc_clears(self, tmp_path):
        """Old logo in a new invoice: perfectly normal (spec: NOT flagged)."""
        path = _pdf_with_jpeg(tmp_path, _jpeg_bytes("2024:06:01 10:00:00"))
        assert check_exif_metadata(path, DOC_DATE) == []

    def test_within_24h_slack_clears(self, tmp_path):
        """Timezone slop: same-day EXIF stamps never fire (24h margin)."""
        path = _pdf_with_jpeg(tmp_path, _jpeg_bytes("2026:01:16 06:00:00"))
        assert check_exif_metadata(path, DOC_DATE) == []

    def test_editor_software_flagged_low(self, tmp_path):
        path = _pdf_with_jpeg(tmp_path, _jpeg_bytes(software="Adobe Photoshop 25.0"))
        flags = check_exif_metadata(path, DOC_DATE)
        assert [f.code for f in flags] == ["IMAGES_EXIF_EDITOR_SOFTWARE"]
        assert flags[0].severity == "low"

    def test_missing_doc_date_skips_date_check(self, tmp_path):
        path = _pdf_with_jpeg(tmp_path, _jpeg_bytes("2026:03:01 10:00:00"))
        assert check_exif_metadata(path, None) == []

    def test_one_flag_per_code_max(self, tmp_path):
        """Two offending images -> still one flag (details carry the list)."""
        import fitz
        p = tmp_path / "img2.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(fitz.Rect(36, 36, 136, 136),
                          stream=_jpeg_bytes("2026:03:01 10:00:00"))
        page.insert_image(fitz.Rect(150, 36, 250, 136),
                          stream=_jpeg_bytes("2026:04:01 10:00:00"))
        doc.save(str(p)); doc.close()
        flags = check_exif_metadata(str(p), DOC_DATE)
        assert len(flags) == 1
        assert len(flags[0].details["images"]) == 2
```

**Contingency:** if `insert_image` turns out to strip EXIF (APP1) on re-encode, these PDF-level tests will fail with empty flags. In that case keep the `_read_exif` tests, and rewrite the `check_exif_metadata` tests to monkeypatch `fitz.Document.extract_image` to return `{"image": jpeg, "ext": "jpeg"}`. Note which route was taken in the commit message. (PyMuPDF passes baseline JPEG streams through unmodified, so the direct route is expected to work.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_images.py -q -k "Exif or exif"`
Expected: FAIL — `ImportError: cannot import name '_read_exif'`.

- [ ] **Step 3: Implement in `src/modules/images.py`**

Add near the other imports: `import io`, `from datetime import datetime, timedelta, timezone`, and `from PIL import Image as PILImage`.

Add after `extract_images_from_pdf`:

```python
# =============================================================================
# EXIF METADATA (audit item 21)
# =============================================================================
# Only two conservative signals (spec D5 — F2 precedent: these can never force
# REVIEW alone):
#   IMAGES_EXIF_DATE_AFTER_DOC  (medium)  image claims to have been captured/
#       modified AFTER the PDF says it was created (+24h timezone slack) —
#       an asset from the future relative to its own document.
#   IMAGES_EXIF_EDITOR_SOFTWARE (low)     EXIF Software names an image editor.
#       Informational only: logos are legitimately edited.
# Deliberately NOT flagged: EXIF dates BEFORE the document (old logos are
# normal) and camera/device fields (screenshot detection already exists).

_EXIF_TAG_DATETIME = 306          # "DateTime" (file modification)
_EXIF_TAG_SOFTWARE = 305
_EXIF_IFD_EXIF = 0x8769
_EXIF_TAG_DATETIME_ORIGINAL = 36867

_EXIF_EDITOR_SOFTWARE = (
    "photoshop", "gimp", "paint.net", "paintshop", "pixelmator",
    "affinity photo", "corel", "photopea", "ms paint", "krita",
)

# Timezone slack: EXIF datetimes are timezone-naive, so anything inside one
# day of the document date is treated as "same time".
_EXIF_DATE_SLACK = timedelta(hours=24)


def _read_exif(image_bytes: bytes) -> tuple["datetime | None", "str | None"]:
    """Best EXIF datetime (naive) + Software string from raw image bytes.

    Prefers DateTimeOriginal (capture time) over DateTime (last save).
    Never raises: any parse problem returns (None, None) — EXIF is
    attacker-controllable input.

    Example:
        >>> _read_exif(jpeg_with_exif)      # doctest: +SKIP
        (datetime(2026, 3, 1, 10, 0), 'Adobe Photoshop 25.0')
    """
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None, None
        software = exif.get(_EXIF_TAG_SOFTWARE)
        software = str(software).strip() if software else None

        raw_dates = [exif.get(_EXIF_TAG_DATETIME)]
        try:
            raw_dates.append(exif.get_ifd(_EXIF_IFD_EXIF).get(_EXIF_TAG_DATETIME_ORIGINAL))
        except Exception:
            pass
        best: datetime | None = None
        for raw in raw_dates:
            if not raw:
                continue
            try:
                dt = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
            if best is None or dt > best:
                best = dt
        return best, software
    except Exception:
        return None, None


def check_exif_metadata(pdf_path: str, doc_creation_date) -> list[Flag]:
    """EXIF checks over every embedded raster image (one flag per code max).

    Args:
        pdf_path: The PDF to inspect.
        doc_creation_date: PDFMetadata.creation_date (tz-aware UTC) or None —
            when None the date check is skipped entirely.
    """
    late_images: list[dict] = []
    editor_images: list[dict] = []

    doc = None
    try:
        doc = fitz.open(pdf_path)
        seen: set[int] = set()
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                if not base or base.get("ext") not in ("jpeg", "jpg", "jpx", "tiff", "png"):
                    continue
                exif_dt, software = _read_exif(base.get("image", b""))

                if exif_dt is not None and doc_creation_date is not None:
                    exif_utc = exif_dt.replace(tzinfo=timezone.utc)
                    if exif_utc > doc_creation_date + _EXIF_DATE_SLACK:
                        late_images.append({"xref": xref,
                                            "exif_datetime": str(exif_dt)})
                if software and any(e in software.lower()
                                    for e in _EXIF_EDITOR_SOFTWARE):
                    editor_images.append({"xref": xref, "software": software})
    except Exception as e:
        logger.warning(f"EXIF check failed: {e}")
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    flags: list[Flag] = []
    if late_images:
        flags.append(Flag(
            severity="medium",
            code="IMAGES_EXIF_DATE_AFTER_DOC",
            message=(f"{len(late_images)} embedded image(s) carry an EXIF "
                     "timestamp LATER than the document's own creation date — "
                     "an image added after the document supposedly existed."),
            details={"images": late_images,
                     "doc_creation_date": str(doc_creation_date)},
        ))
    if editor_images:
        flags.append(Flag(
            severity="low",
            code="IMAGES_EXIF_EDITOR_SOFTWARE",
            message=(f"{len(editor_images)} embedded image(s) were saved by an "
                     "image editor according to their EXIF Software field. "
                     "Common for logos; relevant only alongside other signals."),
            details={"images": editor_images},
        ))
    return flags
```

Wire into `analyze_images`, after the `check_paste_artifacts` block:

```python
    # EXIF checks (audit item 21) — never let a Pillow failure sink the module
    try:
        all_flags.extend(check_exif_metadata(pdf_data.file_path,
                                             pdf_data.metadata.creation_date))
    except Exception as e:
        logger.warning(f"EXIF analysis failed: {e}")
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_images.py -q`
Expected: PASS (new and pre-existing).

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green. Watch `test_corpus_bands.py`: legit corpus docs with Photoshop-made logos may now carry a LOW flag — low flags cannot move a GENUINE band (0.2 nats), but confirm no band changed. If `IMAGES_EXIF_DATE_AFTER_DOC` fires on any LEGIT corpus doc, treat as FP: inspect the doc; if the image is a legitimately-refreshed asset (e.g. re-downloaded logo), widen nothing — instead report to the user before proceeding (it is a real calibration decision).

- [ ] **Step 5: Commit**

```bash
git add src/modules/images.py tests/test_images.py
git commit -m "feat(images): EXIF checks — image dated after document (medium), editor software (low)

Item 21: reads EXIF via Pillow from already-extracted image streams.
Deliberately silent on EXIF dates BEFORE the doc date (old logos are normal)
and on parse failures. One flag per code per document.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Author metadata joins the fingerprint scans (item 20)

**Files:**
- Modify: `src/modules/metadata.py` (`check_producer` signature + call site in `analyze_metadata`)
- Test: `tests/test_metadata.py` (append)

**Interfaces:**
- Consumes: `PDFMetadata.author` (already extracted, `pdf_extractor.py:57`).
- Produces: `check_producer(producer, creator, author=None)` — third optional param, default keeps every existing caller working. NO new codes: the existing META_ONLINE_CONVERTER / META_SUSPICIOUS_PRODUCER / META_IMAGE_EDITOR / META_IMAGE_CONVERSION / META_AI_GENERATED simply also match on the author field, with `details["author"]` populated.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metadata.py` (reuse the file's existing import style for `check_producer`):

```python
# =============================================================================
# Author field fingerprinting (audit item 20): the SAME suspicious-software
# lists now also scan PDFMetadata.author. No new codes, no new severities.
# =============================================================================

class TestAuthorFingerprinting:
    def test_converter_in_author_flagged(self):
        flags = check_producer(None, None, author="iLovePDF")
        assert any(f.code == "META_ONLINE_CONVERTER" for f in flags)
        assert any(f.details.get("author") == "iLovePDF" for f in flags)

    def test_ai_generator_in_author_flagged(self):
        flags = check_producer("Adobe PDF Library 15.0", None, author="ChatGPT")
        assert any(f.code == "META_AI_GENERATED" for f in flags)

    def test_person_name_author_clean(self):
        """A normal human author must never trip the fingerprints."""
        flags = check_producer("Adobe PDF Library 15.0", "Microsoft Word",
                               author="Marie Dupont")
        assert flags == []

    def test_none_author_backward_compatible(self):
        """Existing two-argument call sites keep working unchanged."""
        assert check_producer("Adobe PDF Library 15.0", "Microsoft Word") == []
```

If `test_metadata.py` asserts exact `details` dicts anywhere for `check_producer`, those assertions will now see an extra `"author": None` key — update them in this task (they are testing the same behavior).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -q -k Author`
Expected: FAIL — `check_producer() got an unexpected keyword argument 'author'`.

- [ ] **Step 3: Implement in `src/modules/metadata.py`**

In `check_producer` (line ~151):

1. Signature: `def check_producer(producer: str | None, creator: str | None, author: str | None = None) -> list[Flag]:`
2. Docstring: add an `author` arg line: `author: PDF Author field — scanned with the same fingerprint lists (audit item 20: converters/editors/AI tools sometimes stamp it instead of Producer).`
3. Extend the combined haystack (line ~168-172):

```python
    producer_lower = (producer or "").lower()
    creator_lower = (creator or "").lower()
    author_lower = (author or "").lower()
    combined = f"{producer_lower} {creator_lower} {author_lower}"
```

4. Every `details={...}` dict inside this function currently carrying `"producer"`/`"creator"` gains `"author": author` (there are five — lines ~188, ~207, ~225, ~241, ~257).

In `analyze_metadata` (line ~441), pass the field:

```python
    all_flags.extend(check_producer(
        pdf_data.metadata.producer,
        pdf_data.metadata.creator,
        pdf_data.metadata.author,
    ))
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -q`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, corpus bands unchanged (corpus docs have human/empty authors — if one flags, inspect: a real "iLovePDF"-authored corpus doc would already flag via Producer today).

- [ ] **Step 5: Commit**

```bash
git add src/modules/metadata.py tests/test_metadata.py
git commit -m "feat(metadata): scan the author field with the existing software fingerprints

Item 20: PDFMetadata.author was extracted but never consumed. Converters/
editors/AI tools sometimes stamp Author instead of Producer. Same codes,
same severities, no person-name heuristics (FP minefield, spec section 2).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: document_type tie -> unknown (item 24)

**Files:**
- Modify: `src/document_type.py`
- Test: `tests/test_analyzer_ledger.py` or new `tests/test_document_type.py` (create the dedicated file)

**Interfaces:**
- Consumes/Produces: `detect_document_type(pdf_data) -> DocumentType` — signature unchanged; NEW behavior: a tie for best score returns `"unknown"` (FP-safest: doc-type confidence multipliers only apply to unambiguous classifications).

- [ ] **Step 1: Write the failing test**

Create `tests/test_document_type.py`:

```python
"""Tests for the keyword document-type classifier (audit item 24)."""
from src.document_type import detect_document_type


class FakePDFData:
    def __init__(self, text: str):
        self.text_by_page = [text]


def test_clear_winner_classifies():
    text = "facture total ttc tva iban montant dû"
    assert detect_document_type(FakePDFData(text)) == "invoice"


def test_tie_returns_unknown():
    """3 invoice keywords vs 3 certificate keywords -> ambiguous -> unknown.

    Previously max() silently took the first dict key (audit item 24: the
    docstring promised tie->unknown; now the code delivers it).
    """
    text = ("facture total ttc tva "                       # 3 invoice hits
            "certificat certifie que homologation")        # 3 certificate hits
    assert detect_document_type(FakePDFData(text)) == "unknown"


def test_below_min_keywords_unknown():
    assert detect_document_type(FakePDFData("facture tva")) == "unknown"


def test_empty_text_unknown():
    assert detect_document_type(FakePDFData("")) == "unknown"
```

- [ ] **Step 2: Run tests to verify the tie test fails**

Run: `.venv/bin/python -m pytest tests/test_document_type.py -v`
Expected: `test_tie_returns_unknown` FAILS (returns `"invoice"`, first key); others pass.

- [ ] **Step 3: Implement**

In `src/document_type.py`, replace the winner-selection block (lines ~156-166):

```python
    # Find the winner. A TIE between two types means the document is ambiguous
    # — classify as "unknown" rather than letting dict order pick arbitrarily
    # (doc-type confidence suppression must only apply to CLEAR classifications).
    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]

    logger.debug(f"Document type scores: {scores}")
    logger.debug(f"Best match: {winners} (score={best_score})")

    # Require at least 3 keyword matches to classify.
    # 2 is too low — generic words like "introduction" + "results" can appear
    # in any document and falsely trigger the "scientific" category.
    if best_score < 3 or len(winners) > 1:
        return "unknown"

    return winners[0]  # type: ignore[return-value]
```

Fix the module-header comment (line ~31-32) from "Tie → unknown. Minimum score to classify: 2 keywords must match." to "Tie → unknown. Minimum score to classify: 3 keywords must match." and the `detect_document_type` docstring line "If the winner has fewer than 3 matches, returns \"unknown\"." gains: "A tie for the best score also returns \"unknown\"."

- [ ] **Step 4: Run the tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_document_type.py tests/ -q`
Expected: all green. Corpus gate especially: a corpus doc flipping to "unknown" changes its confidence multipliers — any band change must be investigated (a real invoice that ties with "technical" would be a classifier weakness worth reporting, not silently accepting).

- [ ] **Step 5: Commit**

```bash
git add src/document_type.py tests/test_document_type.py
git commit -m "fix(document_type): tie between types now returns unknown (docstring honesty)

Item 24: code did max() first-key tie-win while the docstring promised
tie->unknown; min-3-keywords stays (docs updated). Ambiguous docs must not
receive doc-type-specific evidence suppression.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: requirements.txt prune (item 25)

**Files:**
- Modify: `requirements.txt`

**Interfaces:** none (dependency list only). Verified 2026-07-12: `pdfplumber`, `pikepdf`, `python-magic` (`import magic`), `scikit-image` (`import skimage`) have ZERO imports across `src/`, `app.py`, `tests/`.

- [ ] **Step 1: Re-verify zero imports (cheap guard against drift since the audit)**

Run:
```bash
grep -rn "import pdfplumber\|from pdfplumber\|import pikepdf\|from pikepdf\|import magic\|from magic\|import skimage\|from skimage" src/ app.py tests/ || echo "CLEAN"
```
Expected: `CLEAN`. If any hit appears, keep that package and drop it from this task.

- [ ] **Step 2: Edit `requirements.txt`**

Remove exactly these four lines (and no others):

```
pdfplumber>=0.11.0       # Structured text extraction
pikepdf>=8.0.0           # PDF structure analysis (Module E)
scikit-image>=0.22.0     # Advanced image analysis (ELA, noise analysis)
python-magic>=0.4.27     # File type detection
```

Also update the section comment above the (now shorter) "PDF Processing" block if it references removed tools, and delete the now-orphaned "# hashlib is built-in, no external package needed" line ONLY if it sits directly under python-magic (keep the file tidy but minimal-diff).

- [ ] **Step 3: Prove the app still boots with a clean environment resolution**

Run: `.venv/bin/python -c "import src.analyzer; import app" 2>&1 | tail -2`
Expected: no ImportError (Streamlit import warnings are fine).
Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): drop pdfplumber, pikepdf, python-magic, scikit-image (never imported)

Item 25: zero imports across src/, app.py, tests/ (re-verified in this
commit). Lightens the Streamlit Cloud build.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Truthful UI bars — `module_contributions` + dead-weight removal (item 23)

**Files:**
- Modify: `src/evidence.py` (add `module_contributions`)
- Modify: `app.py:830` (import), `app.py:~1224-1262` (groups + bars), `app.py:1682` + `app.py:1745` (raw-card weight)
- Modify: `src/scoring.py` (delete `MODULE_WEIGHTS`, `DEFAULT_WEIGHT`, `get_risk_level`)
- Test: `tests/test_evidence.py` (append)

**Interfaces:**
- Consumes: `Evidence`, `_soft_buckets`, `_group_capped_soft_sum` (evidence.py internals); `result.evidence_ledger` (`AnalysisResult`, already populated by `create_analysis_result`).
- Produces: `module_contributions(ledger: list[Evidence]) -> dict[str, dict]` returning per-group `{"tamper": float, "genuine": float, "hard_tamper": bool, "hard_genuine": bool}` — the EXACT group-capped numbers `aggregate_evidence` sums, so the UI explains the score with the formula that produces it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evidence.py`:

```python
# =============================================================================
# module_contributions (audit item 23): the per-module numbers the UI shows
# must be the same group-capped sums aggregate_evidence actually uses.
# =============================================================================
from src.evidence import Evidence, module_contributions


def _ev(code="X_FLAG", weight=1.5, kind="soft", direction="tamper",
        group="metadata", confidence=1.0):
    return Evidence(code=code, weight=weight, kind=kind, direction=direction,
                    group=group, message="m", confidence=confidence)


class TestModuleContributions:
    def test_empty_ledger(self):
        assert module_contributions([]) == {}

    def test_single_soft_tamper(self):
        out = module_contributions([_ev(weight=1.5)])
        assert out["metadata"]["tamper"] == 1.5
        assert out["metadata"]["genuine"] == 0.0
        assert out["metadata"]["hard_tamper"] is False

    def test_group_capping_matches_aggregation(self):
        """Two highs in one group: full + half — identical to _group_capped_soft_sum."""
        out = module_contributions([_ev(weight=1.5), _ev(weight=1.5)])
        assert out["metadata"]["tamper"] == 1.5 + 0.75

    def test_confidence_scales(self):
        out = module_contributions([_ev(weight=1.5, confidence=0.5)])
        assert out["metadata"]["tamper"] == 0.75

    def test_genuine_direction_separate(self):
        out = module_contributions([
            _ev(weight=1.5, group="arithmetic"),
            _ev(code="ARITH_XML_CONSISTENT", weight=0.7, direction="genuine",
                group="arithmetic"),
        ])
        assert out["arithmetic"]["tamper"] == 1.5
        assert out["arithmetic"]["genuine"] == 0.7

    def test_hard_evidence_marked(self):
        out = module_contributions([_ev(kind="hard", group="overlay")])
        assert out["overlay"]["hard_tamper"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q -k Contributions`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement `module_contributions` in `src/evidence.py`**

Append at the end of the file:

```python
def module_contributions(ledger: list["Evidence"]) -> dict[str, dict]:
    """Per-group evidence contributions, for DISPLAY (audit item 23).

    Returns, for each correlation group (usually the module name), the same
    group-capped soft sums `aggregate_evidence` adds to the prior — so a UI
    built on this explains the score with the formula that actually produced
    it (the pre-ledger MODULE_WEIGHTS average no longer exists).

    Returns:
        {group: {"tamper": float, "genuine": float,
                 "hard_tamper": bool, "hard_genuine": bool}}
    """
    out: dict[str, dict] = {}

    def _slot(group: str) -> dict:
        return out.setdefault(group, {"tamper": 0.0, "genuine": 0.0,
                                      "hard_tamper": False, "hard_genuine": False})

    for (group, direction), items in _soft_buckets(ledger).items():
        _slot(group)[direction] = _group_capped_soft_sum(items)
    for e in ledger:
        if e.kind == "hard":
            _slot(e.group)["hard_tamper" if e.direction == "tamper"
                           else "hard_genuine"] = True
    return out
```

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q` — Expected: PASS.

- [ ] **Step 4: Rewire the app bars (`app.py` ~1224-1262)**

4a. Extend the groups so EVERY module is represented (overlay/arithmetic/forensics/identity/twod_doc are currently invisible in the bars):

```python
            # Define the 4 groups: (label, list of module names, tooltip)
            groups = [
                (t("group_modifications"), ["structure", "overlay"], t("tip_modifications")),
                (t("group_consistency"), ["content", "arithmetic"], t("tip_consistency")),
                (t("group_tampering"), ["images", "fonts", "visual", "metadata", "forensics"], t("tip_tampering")),
                (t("group_authenticity"), ["external", "identity", "twod_doc"], t("tip_authenticity")),
            ]
```

4b. Right after the `groups` definition, compute the real contributions once:

```python
            # Evidence-ledger contributions (audit item 23): the bars show the
            # ACTUAL score pressure per group — the same group-capped sums the
            # aggregation used — not the dead MODULE_WEIGHTS formula.
            from src.evidence import module_contributions
            contrib = module_contributions(result.evidence_ledger)

            def _group_net(names: list[str]) -> float:
                return sum(contrib.get(n, {}).get("tamper", 0.0)
                           - contrib.get(n, {}).get("genuine", 0.0)
                           for n in names)

            total_pressure = sum(max(0.0, _group_net(names))
                                 for _, names, _ in groups) or 0.0
```

4c. Replace the weighted-average block inside the loop (the `weighted_sum/weight_total/group_score` lines AND the `bar_color` chain AND the bar HTML that interpolates `group_score`) with:

```python
                # Net evidence pressure for this group (positive = pushed the
                # score DOWN; negative = genuine evidence lifted it).
                net = _group_net(member_names)
                hard = any(contrib.get(n, {}).get("hard_tamper") for n in member_names)

                if hard:
                    width, bar_color, value_label = 100, "#ef233c", "veto"
                elif net > 0:
                    width = round(100 * net / total_pressure) if total_pressure else 0
                    bar_color = "#ef233c" if net >= 1.5 else "#ffb627"
                    value_label = f"+{net:.1f}"
                elif net < 0:
                    width, bar_color, value_label = 25, "#28a745", f"{net:.1f}"
                else:
                    width, bar_color, value_label = 0, "#28a745", "0"

                bars_html += f'''
                <div style="display:flex; align-items:center; margin-bottom:6px;">
                    <span style="width:100px; flex-shrink:0;">
                        <span class="tf-label" style="color:#eee;">{label}<span class="tf-tip">{tooltip}</span></span>
                    </span>
                    <div style="flex:1; background:#2a2a2a; border-radius:4px;
                        height:10px; overflow:hidden;">
                        <div style="width:{width}%; height:100%;
                            background:{bar_color}; border-radius:4px;
                            transition:width 1s ease;"></div>
                    </div>
                    <span style="width:35px; text-align:right; font-size:0.8rem;
                        color:#eee; margin-left:8px; font-weight:bold;">{value_label}</span>
                </div>'''
```

(The `if not members:` N/A branch above it stays exactly as-is.)

4d. Fix the import at `app.py:830`:

```python
    from src.scoring import collect_all_flags, count_flags_by_severity
```

4e. Raw-module cards: delete line 1682 (`mod_weight = MODULE_WEIGHTS.get(module.module, DEFAULT_WEIGHT)`) and remove the fragment `f'<span>{t("label_weight")}: {mod_weight}</span>'` from the card HTML near line 1745 (keep neighboring spans intact). Verify: `grep -n "mod_weight\|MODULE_WEIGHTS\|DEFAULT_WEIGHT" app.py` → no hits.

- [ ] **Step 5: Delete the dead scoring code**

In `src/scoring.py` delete: the `MODULE_WEIGHTS` dict + its comment block, `DEFAULT_WEIGHT`, and the whole `get_risk_level` function. Then verify nothing else breaks:

```bash
grep -rn "MODULE_WEIGHTS\|DEFAULT_WEIGHT\|get_risk_level" src/ app.py tests/ || echo CLEAN
.venv/bin/python -c "import src.scoring; import app" && echo IMPORTS-OK
```
Expected: `CLEAN` then `IMPORTS-OK`. If `Literal` (or other imports) become unused in scoring.py, remove them.

- [ ] **Step 6: Full suite + visual sanity**

Run: `.venv/bin/python -m pytest tests/ -q` — Expected: all green.
Then run one real document through the UI path without a browser:

```bash
.venv/bin/python -c "
from src.analyzer import TrustyFileAnalyzer
from src.evidence import module_contributions
r = TrustyFileAnalyzer(enable_external=False).analyze('samples/legitimate/real/i2d_AzureInterior.pdf')
print('ledger groups:', module_contributions(r.evidence_ledger))
"
```
Expected: a dict keyed by module names with small floats — eyeball that genuine evidence (arithmetic) appears in the `genuine` slot.

- [ ] **Step 7: Commit**

```bash
git add src/evidence.py src/scoring.py app.py tests/test_evidence.py
git commit -m "fix(ui): score bars driven by real evidence contributions; drop dead MODULE_WEIGHTS

Item 23: the bars showed a weighted-average formula the ledger engine no
longer computes. module_contributions() exposes the exact group-capped sums
aggregation uses; overlay/arithmetic/forensics/identity/twod_doc join the
bar groups (previously invisible). MODULE_WEIGHTS, DEFAULT_WEIGHT and
get_risk_level deleted (zero consumers).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: summary.py sync + drift test (item 22)

**Files:**
- Modify: `src/summary.py` (`FLAG_THEME_MAP`, `SENTENCE_TEMPLATES`, `SENTENCE_TEMPLATES_FR`, `_get_positive_signals`)
- Test: `tests/test_summary_drift.py` (new file)

**Interfaces:**
- Consumes: every flag code emitted under `src/` (authoritative gap list computed 2026-07-12, below).
- Produces: complete theme coverage; EN/FR template-key parity; a drift test that makes unmapped codes a test failure forever.

**The 2026-07-12 gap list (recompute in Step 1 — Tasks 2/4 added codes since):**
- Missing themes: `ARITH_XML_CONSISTENT`, `CONTENT_DATE_CREATION_MISMATCH`, `CONTENT_EXPIRED_DOCUMENT`, `CONTENT_FUTURE_DOCUMENT_DATE`, `EXTERNAL_IBAN_INVALID`, `EXTERNAL_IBAN_VERIFICATION_FAILED`, `EXTRACTION_FAILED`, `FORENSICS_ELA_MAJOR_EDIT`, `FORENSICS_ELA_MINOR_EDIT`, `META_IMAGE_CONVERSION`, `META_IMAGE_EDITOR`, `MODULE_ERROR`, `STRUCT_HIDDEN_MODIFICATION`, 20 × `MRZ_*`, plus Task 2/4's `TWOD_DOC_FIELD_MISMATCH`, `TWOD_DOC_CONSISTENT`, `IMAGES_EXIF_DATE_AFTER_DOC`, `IMAGES_EXIF_EDITOR_SOFTWARE`.
- Orphans to delete: `CONTENT_FUTURE_INVOICE_DATE` (rename → `CONTENT_FUTURE_DOCUMENT_DATE`), all 10 `VISUAL_WATERMARK_*` theme entries/templates. KEEP `CONTENT_INVALID_SIREN/SIRET/VAT` (emitted via `_flag_invalid_id(...)` in content.py — string args, invisible to a naive `code="` grep).
- FR parity hole: `IMAGES_NO_IMAGES` has EN template only.
- `_build_bullets`' signature-skip is ALREADY fixed (comment cites the audit) — verify, don't re-fix.

- [ ] **Step 1: Write the drift test (it is the failing test for this task)**

Create `tests/test_summary_drift.py`:

```python
"""
Summary drift gate (audit item 22): every flag code emitted anywhere under
src/ must have a user-facing theme, and the EN/FR template dicts must cover
identical code sets. This test IS the durable fix — the audit found a
critical flag (CONTENT_FUTURE_DOCUMENT_DATE) invisible to themes because a
rename never reached summary.py; that class of bug now fails CI.
"""
import glob
import re

from src.summary import FLAG_THEME_MAP, SENTENCE_TEMPLATES, SENTENCE_TEMPLATES_FR

# Any quoted ALL-CAPS token with >= 1 underscore in module source. Broad on
# purpose: codes are emitted as code="X", code = "X" kwargs/assignments AND
# as bare string arguments (content.py's _flag_invalid_id). Non-code strings
# that happen to match go in NOT_FLAG_CODES with a justification.
_CODE_RE = re.compile(r'"([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)"')

NOT_FLAG_CODES = {
    "INCOME_CHECK",   # twod_doc.py: a ComparisonMatch DI label, not a Flag code
}

# Codes without a sentence template fall back to their (English)
# Flag.message — acceptable. But these prefixes MUST have templates: their
# module messages are hardcoded French (identity.py MRZ family), so the
# fallback would show French to EN users. (Every code needs a THEME — no
# exceptions.)
TEMPLATE_REQUIRED_PREFIXES = ("MRZ_",)


def _emitted_codes() -> set[str]:
    codes: set[str] = set()
    for path in glob.glob("src/**/*.py", recursive=True):
        if path.endswith("summary.py"):
            continue  # summary.py QUOTES every code — it is the map, not an emitter
        for m in _CODE_RE.finditer(open(path, encoding="utf-8").read()):
            codes.add(m.group(1))
    return codes - NOT_FLAG_CODES


def test_every_emitted_code_has_a_theme():
    missing = sorted(_emitted_codes() - set(FLAG_THEME_MAP))
    assert not missing, f"codes with no FLAG_THEME_MAP entry: {missing}"


def test_en_fr_template_parity():
    only_en = sorted(set(SENTENCE_TEMPLATES) - set(SENTENCE_TEMPLATES_FR))
    only_fr = sorted(set(SENTENCE_TEMPLATES_FR) - set(SENTENCE_TEMPLATES))
    assert not only_en and not only_fr, (
        f"template dicts diverge — EN-only: {only_en}, FR-only: {only_fr}")


def test_required_templates_present():
    emitted = _emitted_codes()
    required = {c for c in emitted if c.startswith(TEMPLATE_REQUIRED_PREFIXES)}
    missing = sorted(required - set(SENTENCE_TEMPLATES))
    assert not missing, f"French-messaged codes lacking EN templates: {missing}"


def test_no_orphan_map_entries():
    emitted = _emitted_codes()
    orphan_themes = sorted(set(FLAG_THEME_MAP) - emitted)
    orphan_templates = sorted(set(SENTENCE_TEMPLATES) - emitted)
    assert not orphan_themes, f"FLAG_THEME_MAP entries never emitted: {orphan_themes}"
    assert not orphan_templates, f"templates never emitted: {orphan_templates}"
```

- [ ] **Step 2: Run it — the failures ARE the work list**

Run: `.venv/bin/python -m pytest tests/test_summary_drift.py -v`
Expected: all four FAIL, printing exactly which codes to add/delete. Treat those lists as authoritative over this plan's static list. If a legitimately-non-code string surfaces (e.g. a new constant), add it to `NOT_FLAG_CODES` with a one-line justification comment.

- [ ] **Step 3: Fix `FLAG_THEME_MAP`**

In `src/summary.py`:
- Rename key `"CONTENT_FUTURE_INVOICE_DATE"` → `"CONTENT_FUTURE_DOCUMENT_DATE"` (same theme, `"dates"`).
- Delete the 10 `VISUAL_WATERMARK_*` entries.
- Add (grouped with kin, comment style matching the file):

```python
    # dates
    "CONTENT_DATE_CREATION_MISMATCH": "dates",
    "CONTENT_EXPIRED_DOCUMENT": "dates",
    # origin
    "META_IMAGE_EDITOR": "origin",
    "META_IMAGE_CONVERSION": "origin",
    # tampering
    "FORENSICS_ELA_MAJOR_EDIT": "tampering",
    "FORENSICS_ELA_MINOR_EDIT": "tampering",
    "STRUCT_HIDDEN_MODIFICATION": "tampering",
    "ARITH_XML_CONSISTENT": "tampering",       # genuine-direction; bullets skip it
    "TWOD_DOC_FIELD_MISMATCH": "tampering",
    "TWOD_DOC_CONSISTENT": "tampering",        # genuine-direction; bullets skip it
    # images
    "IMAGES_EXIF_DATE_AFTER_DOC": "images",
    "IMAGES_EXIF_EDITOR_SOFTWARE": "images",
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
    # other (pipeline-status codes)
    "EXTRACTION_FAILED": "other",
    "MODULE_ERROR": "other",
```

(Place the two EXTERNAL_* entries next to the file's existing `EXTERNAL_*` block and reuse ITS theme if it differs from "identity" — co-location beats this plan's guess. If the drift test lists an MRZ code not shown above, add it with theme "identity".)

- [ ] **Step 4: Add the EN templates**

In `SENTENCE_TEMPLATES`, rename key `"CONTENT_FUTURE_INVOICE_DATE"` → `"CONTENT_FUTURE_DOCUMENT_DATE"` (reword "invoice" → "document" inside its string), delete `VISUAL_WATERMARK_*` entries if present, then add:

```python
    "CONTENT_DATE_CREATION_MISMATCH": lambda f: (
        "A date mentioned in the document contradicts the file's own creation "
        "timestamp."),
    "CONTENT_EXPIRED_DOCUMENT": lambda f: (
        "The document refers to a validity period that has already expired."),
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
    "FORENSICS_ELA_MAJOR_EDIT": lambda f: (
        "Error-level analysis shows a region whose compression history differs "
        "sharply from the rest of the page — consistent with a pasted or "
        "edited area."),
    "FORENSICS_ELA_MINOR_EDIT": lambda f: (
        "Error-level analysis shows a small region with a different "
        "compression history — possibly an edited area."),
    "STRUCT_HIDDEN_MODIFICATION": lambda f: (
        "The file contains a modification saved WITHOUT the standard "
        "incremental-update markers — an attempt to edit without leaving a "
        "trace."),
    "TWOD_DOC_FIELD_MISMATCH": lambda f: (
        f"The signed 2D-Doc barcode says {f.details.get('field', 'a field')} "
        f"= \"{f.details.get('expected', '')}\", but the visible text shows "
        "something else — the printed value may have been edited."
        if f.details else
        "A field in the signed 2D-Doc barcode does not match the visible text."),
    "TWOD_DOC_CONSISTENT": lambda f: (
        "All fields of the signed 2D-Doc barcode match the visible text."),
    "IMAGES_EXIF_DATE_AFTER_DOC": lambda f: (
        "An embedded image carries an EXIF timestamp LATER than the document's "
        "creation date — it was made after the document supposedly existed."),
    "IMAGES_EXIF_EDITOR_SOFTWARE": lambda f: (
        "An embedded image was saved by an image editor (per its EXIF data). "
        "Common for logos; meaningful only alongside other findings."),
    "EXTERNAL_IBAN_INVALID": lambda f: (
        "A bank account number (IBAN) printed on the document fails its "
        "checksum — it cannot be a real account."),
    "EXTERNAL_IBAN_VERIFICATION_FAILED": lambda f: (
        "The bank account number (IBAN) could not be verified."),
    "EXTRACTION_FAILED": lambda f: (
        "The file could not be read as a valid PDF, so it cannot be verified."),
    "MODULE_ERROR": lambda f: (
        f"One analysis ({f.details.get('module', '?')}) failed on this "
        "document and was skipped."
        if f.details else "One analysis failed on this document and was skipped."),
    # MRZ family (identity.py messages are hardcoded French — these EN
    # templates are what English users see)
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
```

- [ ] **Step 5: Add the FR templates**

In `SENTENCE_TEMPLATES_FR`: same key operations (rename, delete orphans), plus the missing `IMAGES_NO_IMAGES` parity entry, then the FR pairs:

```python
    "IMAGES_NO_IMAGES": lambda f: (
        "Le document ne contient aucune image (pas de logo ni d'en-tête) — "
        "inhabituel pour une facture professionnelle."),
    "CONTENT_DATE_CREATION_MISMATCH": lambda f: (
        "Une date mentionnée dans le document contredit la date de création "
        "du fichier."),
    "CONTENT_EXPIRED_DOCUMENT": lambda f: (
        "Le document fait référence à une période de validité déjà expirée."),
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
    "FORENSICS_ELA_MAJOR_EDIT": lambda f: (
        "L'analyse des niveaux d'erreur révèle une zone dont l'historique de "
        "compression diffère nettement du reste de la page — compatible avec "
        "un collage ou une retouche."),
    "FORENSICS_ELA_MINOR_EDIT": lambda f: (
        "L'analyse des niveaux d'erreur révèle une petite zone à l'historique "
        "de compression différent — possible retouche."),
    "STRUCT_HIDDEN_MODIFICATION": lambda f: (
        "Le fichier contient une modification enregistrée SANS les marqueurs "
        "standard de mise à jour incrémentale — une tentative d'édition sans "
        "laisser de trace."),
    "TWOD_DOC_FIELD_MISMATCH": lambda f: (
        f"Le code 2D-Doc signé indique {f.details.get('field', 'un champ')} "
        f"= « {f.details.get('expected', '')} », mais le texte visible montre "
        "autre chose — la valeur imprimée a pu être modifiée."
        if f.details else
        "Un champ du code 2D-Doc signé ne correspond pas au texte visible."),
    "TWOD_DOC_CONSISTENT": lambda f: (
        "Tous les champs du code 2D-Doc signé correspondent au texte visible."),
    "IMAGES_EXIF_DATE_AFTER_DOC": lambda f: (
        "Une image intégrée porte une date EXIF POSTÉRIEURE à la date de "
        "création du document — elle a été créée après le document lui-même."),
    "IMAGES_EXIF_EDITOR_SOFTWARE": lambda f: (
        "Une image intégrée a été enregistrée par un éditeur d'images (selon "
        "ses données EXIF). Courant pour un logo ; significatif seulement "
        "avec d'autres indices."),
    "EXTERNAL_IBAN_INVALID": lambda f: (
        "Un IBAN imprimé sur le document échoue à sa somme de contrôle — ce "
        "ne peut pas être un compte réel."),
    "EXTERNAL_IBAN_VERIFICATION_FAILED": lambda f: (
        "L'IBAN n'a pas pu être vérifié."),
    "EXTRACTION_FAILED": lambda f: (
        "Le fichier n'a pas pu être lu comme un PDF valide et ne peut donc "
        "pas être vérifié."),
    "MODULE_ERROR": lambda f: (
        f"Une analyse ({f.details.get('module', '?')}) a échoué sur ce "
        "document et a été ignorée."
        if f.details else "Une analyse a échoué sur ce document et a été ignorée."),
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
```

- [ ] **Step 6: 2D-Doc positive signal in `_get_positive_signals`**

After the existing `ARITH_XML_CONSISTENT` block (~line 1071-1080), add the symmetric branch:

```python
    # Machine-verified 2D-Doc consistency (flag with genuine direction)
    for flags_in_theme in grouped.values():
        if any(f.code == "TWOD_DOC_CONSISTENT" for f in flags_in_theme):
            positives.append(
                "Les champs du code 2D-Doc signé correspondent au document visible."
                if lang == "fr" else
                "The signed 2D-Doc barcode fields match the visible document."
            )
            break
```

- [ ] **Step 7: Run the drift test until green, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_summary_drift.py -v`
Expected: 4 PASS. Iterate on whatever the failure lists print (they are authoritative).
Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (summary behavior tests may reference renamed keys — update any that assert `CONTENT_FUTURE_INVOICE_DATE`).

- [ ] **Step 8: Commit**

```bash
git add src/summary.py tests/test_summary_drift.py
git commit -m "fix(summary): full code-to-theme sync + EN/FR parity + drift gate

Item 22: CONTENT_FUTURE_DOCUMENT_DATE rename reaches the maps (critical flag
was invisible to themes); ~35 codes themed (MRZ family, ELA, EXIF, 2D-Doc,
pipeline-status); MRZ family gets EN+FR templates (module messages are
hardcoded French); VISUAL_WATERMARK_* orphans removed; 2D-Doc positive
signal displayed. test_summary_drift.py makes silent drift a CI failure.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: README refresh + backlog closure notes (items 26, 19/D4)

**Files:**
- Modify: `README.md` (scoring section + module list + system deps)
- Modify: `docs/superpowers/2026-07-10-audit-backlog.md` (closure marks + residuals)
- Modify: `CLAUDE.md` (tick the Phase 4 2D-Doc checkbox)

**Interfaces:** documentation only.

- [ ] **Step 1: README — replace the scoring section**

Read `README.md`; replace its stale weighted-average scoring section with:

```markdown
## Scoring: the evidence ledger

Every module emits flags; each flag becomes a piece of **evidence** —
`soft` or `hard`, pushing toward **tamper** or **genuine** — combined in
log-odds space (see `src/evidence.py`):

- **Soft evidence** (heuristics) is summed per module with diminishing
  returns: the strongest finding counts fully, the rest at half weight, so
  correlated symptoms of one edit don't stack linearly.
- **Genuine evidence** (a trusted signature, e-invoice XML that matches the
  page, a matching signed 2D-Doc) lifts the score — but is discarded
  whenever meaningful tamper evidence exists (anti-laundering guard):
  agreement proves nothing about a document already in question.
- **Hard evidence** is certainty, not heuristics: a hard tamper finding
  caps the score at 15; a hard genuine finding floors it at 85.

The score maps to bands: **GENUINE** (≥ 80), **REVIEW** (50–79),
**SUSPICIOUS** (< 50), or **UNVERIFIABLE** (unreadable input).
```

- [ ] **Step 2: README — module list (11 analysis modules)**

Replace the module list with:

```markdown
| Module | What it checks |
|---|---|
| metadata | Producer/Creator/Author fingerprints, date sanity, repair marks |
| content | Dates, references, French company ids, IBAN checksums, anachronisms |
| visual | Converter/SPECIMEN watermarks (QR decode: next feature) |
| fonts | Font diversity, subset mixing, mid-line changes |
| images | Screenshots, resolution/compression, paste artifacts, EXIF dates/software |
| structure | Incremental updates, signatures (EU DSS), JavaScript, embedded files |
| overlay | Cover-and-retype boxes, conflicting text, cross-revision value changes |
| arithmetic | e-invoice XML vs visible totals & IBAN — CII (Factur-X) **and** UBL (Peppol) |
| forensics | Error-level analysis (ELA) of page renders |
| identity | MRZ check digits on ID documents (conditional) |
| twod_doc | French 2D-Doc barcodes vs visible text (needs libdmtx) |
```

Add to the install/deploy notes: `brew install tesseract libdmtx` (macOS) / `packages.txt` already lists `tesseract-ocr`, `libdmtx0b` for Streamlit Cloud.

- [ ] **Step 3: Backlog updates**

In `docs/superpowers/2026-07-10-audit-backlog.md`:
- Item 14: prefix with `✅DONE (2026-07-12)`.
- Item 17: prefix with `✅DONE (2026-07-12)`.
- Item 19: prefix with `✅CLOSED-AS-COVERED (2026-07-12, D4)`.
- Items 20, 21, 22, 23, 24, 25, 26: prefix with `✅DONE (2026-07-12)`.
- Append under "Post-review residual risks":

```markdown
- (2026-07-12, D4) Prose-only revision changes (e.g. a payee NAME swapped
  between saved revisions, no digits involved) are unscored by design —
  overlay Signal 4 covers digit-bearing tokens only. Revisit with
  calibration samples.
- (2026-07-12, D1) TWOD_DOC_* severities are provisional: the corpus holds
  no real 2D-Doc document. Add a real avis d'impôt sample, then recalibrate.
```

In `CLAUDE.md`, Phase 4: change `- [ ] 2D-Doc verification: Decode French government 2D barcodes (requires libdmtx)` to `- [x]`.

- [ ] **Step 4: Final verification (whole batch)**

```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: green, now > 700 tests. Then the corpus gates explicitly:

```bash
.venv/bin/python -m pytest tests/test_corpus_bands.py tests/test_twod_doc_e2e.py -v
```
Expected: every legit doc GENUINE, tamper gate unchanged, E2E not skipped.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/2026-07-10-audit-backlog.md CLAUDE.md
git commit -m "docs: README evidence-ledger scoring + 11-module table; backlog closure (items 14-26)

Item 26 + D4 residuals logged. Phase 6 completion batch done.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## After the plan

House convention: run the whole-branch adversarial review
(`reviewing-branch-with-workflow` skill) before merging `phase6-robustness`
into `main`; log any accepted residuals in the backlog.
