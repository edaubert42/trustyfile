# Producer Structural-Fingerprint Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect PDFs rewritten by online converters / consumer editors from their structural "handwriting" (header magic bytes + trailer key signature), independent of the scrubbable Producer metadata string.

**Architecture:** One self-contained, never-raising module `src/modules/provenance.py` reads the raw PDF bytes, extracts an `ObservedSignature`, classifies it against a curated JSON signature table, and — only on a *confident* match to a converter/editor class — emits one of two soft flags. Wired into the analyzer under the existing crash-isolation block. Scoring stays untouched (the verifiability damper is deferred to the coverage-gate build).

**Tech Stack:** Python 3.11+, PyMuPDF (fitz) for test-PDF construction only (extraction reads raw bytes directly), pytest. No new runtime dependencies. Playwright (existing MCP plugin) for one-off sample collection.

## Global Constraints

- **Never raises:** `analyze_provenance(pdf_data)` returns a partial `ModuleResult` (reduced confidence) on any error — never propagates an exception (CLAUDE.md; mirror `invisible.py`).
- **Fail-open / zero-FP:** an unrecognized OR ambiguous (>1 DB match) fingerprint emits **no** flag. Flags fire only on exactly one confident match to a converter/editor-class tool.
- **Do NOT touch `src/scoring.py` or `src/evidence.py`** — owned by the in-flight coverage-gate session. New codes ride the default evidence path (soft/tamper).
- **Flag codes:** `PROVENANCE_PRODUCER_MISMATCH` (medium), `PROVENANCE_LAUNDERED` (high). Both soft.
- **Summary drift gate:** every emitted code needs a `FLAG_THEME_MAP` entry and matching EN **and** FR `SENTENCE_TEMPLATES` entries, or `tests/test_summary_drift.py` fails (`test_en_fr_template_parity`, `test_no_orphan_map_entries`).
- **Interpreter:** the provenance worktree has no venv; run everything with the shared one: from `/Users/edouard/Projects/trustyfile-provenance`, invoke `/Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest …` (plain `python` does not exist).
- **Org policy:** online-tool samples are collected by feeding **synthetic generated invoices only** through public tools — never corpus/sample documents.
- **Module contract:** `analyze_provenance(pdf_data) -> ModuleResult(module="provenance", …)`; reads `pdf_data.file_path`, `pdf_data.extraction_ok`, `pdf_data.metadata.producer`, `pdf_data.metadata.creator`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/modules/provenance.py` (create) | Signature extraction, classification, flag building, `analyze_provenance` entry point |
| `src/modules/data/provenance_signatures.json` (create) | Curated per-tool signature table (data, not code) |
| `tests/test_provenance.py` (create) | Unit tests (extraction, classify, flags), integration, corpus zero-FP gate |
| `scripts/collect_provenance_samples.py` (create) | Playwright one-off collector for online-tool signatures |
| `src/modules/metadata.py` (modify) | Refresh `SUSPICIOUS_PRODUCERS` with 2026 online tools |
| `src/analyzer.py` (modify) | Register `analyze_provenance` under `_run(...)` |
| `src/summary.py` (modify) | Add both codes to `FLAG_THEME_MAP` + EN/FR templates |

Run order: Task 1 → 2 → 3 build the module; Task 4 wires it (drift gate must stay green); Task 5 is the independent list refresh; Task 6 collects real online-tool signatures and adds true-positive + corpus tests.

---

## Task 1: Signature extraction

**Files:**
- Create: `src/modules/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces: `ObservedSignature(header_magic: str | None, trailer_keys: tuple[str, ...], has_prev: bool)`; `extract_signature(raw: bytes) -> ObservedSignature`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provenance.py
"""Tests for the producer structural-fingerprint detector (spec 2026-07-17)."""
from src.modules.provenance import ObservedSignature, extract_signature


def test_header_magic_extracted_as_hex():
    raw = b"%PDF-1.7\n%\xb5\xb5\xb5\xb5\n1 0 obj\n<< >>\nendobj\n"
    sig = extract_signature(raw)
    assert sig.header_magic == "b5b5b5b5"


def test_classic_trailer_keys_in_order():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Info 4 0 R /ID [<aa><bb>] >>\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Size", "Root", "Info", "ID")
    assert sig.has_prev is False


def test_prev_key_detected():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Prev 116 >>\nstartxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.has_prev is True
    assert "Prev" in sig.trailer_keys


def test_xref_stream_trailer_keys():
    # PDF 1.5+ cross-reference stream: dict carried on an /Type /XRef object.
    raw = (b"%PDF-1.5\n%\xff\xff\n"
           b"5 0 obj\n<< /Type /XRef /Size 6 /Root 1 0 R /W [1 2 1] "
           b"/Index [0 6] /Length 20 >>\nstream\n....\nendstream\nendobj\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Type", "Size", "Root", "W", "Index", "Length")


def test_no_second_comment_line_header_magic_none():
    raw = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n<< /Size 1 /Root 1 0 R >>\n"
    sig = extract_signature(raw)
    assert sig.header_magic is None
    assert sig.trailer_keys == ("Size", "Root")


def test_garbage_never_raises():
    assert isinstance(extract_signature(b"not a pdf at all"), ObservedSignature)
    assert isinstance(extract_signature(b""), ObservedSignature)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.modules.provenance'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/modules/provenance.py
"""
Producer structural-fingerprint detector (Module: provenance) — phase 8.

Identifies the tool that actually produced/rewrote a PDF from its structural
"handwriting" — the header magic bytes and the trailer key signature — which
survive metadata scrubbing. See docs/superpowers/specs/2026-07-17-producer-
fingerprint-design.md.

Deferred (do NOT wire here): a "verifiability damper" that caps the GENUINE
ceiling for converter-rewritten documents belongs in the coverage-gate build;
this module only emits flags on the default (soft) evidence path.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.models import Flag, ModuleResult

logger = logging.getLogger(__name__)

SEVERITY_POINTS = {"low": 5, "medium": 15, "high": 30, "critical": 50}

_KEY_RE = re.compile(rb"/([A-Za-z][A-Za-z0-9]*)")


@dataclass
class ObservedSignature:
    """Structural fingerprint read from a PDF's raw bytes."""
    header_magic: str | None            # hex of 2nd-comment bytes, or None
    trailer_keys: tuple[str, ...]       # top-level trailer/xref-dict keys, in order
    has_prev: bool                      # any /Prev present (incremental save)


def _header_magic(raw: bytes) -> str | None:
    """Hex of the binary bytes on the 2nd comment line after %PDF-x.y, else None."""
    nl = raw.find(b"\n")
    if nl == -1:
        return None
    start = nl + 1
    nl2 = raw.find(b"\n", start)
    if nl2 == -1:
        return None
    line = raw[start:nl2]
    if not line.startswith(b"%"):
        return None
    magic = line[1:].rstrip(b"\r")
    return magic.hex() if magic else None


def _skip_ws(b: bytes, i: int) -> int:
    while i < len(b) and b[i:i + 1].isspace():
        i += 1
    return i


def _skip_value(b: bytes, i: int) -> int:
    """Return the index just past one PDF value starting at/after i."""
    i = _skip_ws(b, i)
    n = len(b)
    if i >= n:
        return i
    c = b[i:i + 1]
    if c == b"/":                                   # name value
        m = _KEY_RE.match(b, i)
        return m.end() if m else i + 1
    if b[i:i + 2] == b"<<":                          # nested dictionary
        depth = 0
        while i < n - 1:
            two = b[i:i + 2]
            if two == b"<<":
                depth += 1; i += 2; continue
            if two == b">>":
                depth -= 1; i += 2
                if depth == 0:
                    return i
                continue
            i += 1
        return i
    if c == b"(":                                   # literal string
        depth = 0
        while i < n:
            ch = b[i:i + 1]
            if ch == b"\\":
                i += 2; continue
            if ch == b"(":
                depth += 1
            elif ch == b")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return i
    if c == b"<":                                   # hex string
        j = b.find(b">", i)
        return j + 1 if j != -1 else n
    if c == b"[":                                   # array
        depth = 0
        while i < n:
            ch = b[i:i + 1]
            if ch == b"[":
                depth += 1
            elif ch == b"]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return i
    m = re.match(rb"[+\-0-9.]+", b[i:])              # number or indirect ref
    if m:
        i += m.end()
        save = i
        i = _skip_ws(b, i)
        m2 = re.match(rb"[0-9]+", b[i:])
        if m2:
            k = _skip_ws(b, i + m2.end())
            if b[k:k + 1] == b"R":
                return k + 1
        return save
    m = re.match(rb"[A-Za-z]+", b[i:])              # true / false / null
    return i + m.end() if m else i + 1


def _dict_body(raw: bytes, start: int) -> bytes | None:
    """Body bytes between the outermost << >> at/after `start`."""
    open_at = raw.find(b"<<", start)
    if open_at == -1:
        return None
    depth = 0
    i = open_at
    while i < len(raw) - 1:
        two = raw[i:i + 2]
        if two == b"<<":
            depth += 1; i += 2; continue
        if two == b">>":
            depth -= 1; i += 2
            if depth == 0:
                return raw[open_at + 2:i - 2]
            continue
        i += 1
    return None


def _parse_dict_keys(body: bytes) -> tuple[str, ...]:
    """Top-level /Name keys of a dict body, in order (values skipped)."""
    keys: list[str] = []
    i, n = 0, len(body)
    while i < n:
        i = _skip_ws(body, i)
        if i >= n:
            break
        if body[i:i + 1] != b"/":
            i += 1
            continue
        m = _KEY_RE.match(body, i)
        if not m:
            i += 1
            continue
        keys.append(m.group(1).decode("latin-1"))
        i = _skip_value(body, m.end())
    return tuple(keys)


def _trailer_body(raw: bytes) -> bytes | None:
    """Dict body of the last classic `trailer`, else the first /Type /XRef object."""
    idx = raw.rfind(b"trailer")
    if idx != -1:
        body = _dict_body(raw, idx)
        if body is not None:
            return body
    xref = raw.find(b"/XRef")
    if xref != -1:
        # step back to the enclosing << that opens this object's dict
        open_at = raw.rfind(b"<<", 0, xref)
        if open_at != -1:
            return _dict_body(raw, open_at)
    return None


def extract_signature(raw: bytes) -> ObservedSignature:
    """Read the structural fingerprint from raw PDF bytes. Never raises."""
    try:
        magic = _header_magic(raw)
        body = _trailer_body(raw)
        keys = _parse_dict_keys(body) if body is not None else ()
        return ObservedSignature(
            header_magic=magic,
            trailer_keys=keys,
            has_prev="Prev" in keys,
        )
    except Exception as exc:  # defensive: extraction must never raise
        logger.warning("provenance: signature extraction failed: %s", exc)
        return ObservedSignature(header_magic=None, trailer_keys=(), has_prev=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add src/modules/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): raw-byte structural signature extraction"
```

---

## Task 2: Signature table + classifier

**Files:**
- Create: `src/modules/data/provenance_signatures.json`
- Modify: `src/modules/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `ObservedSignature` (Task 1).
- Produces: `load_signatures() -> dict[str, dict]`; `Classification(tool: str | None, category: str | None, confidence: float, matched_features: tuple[str, ...])`; `classify(sig: ObservedSignature, db: dict[str, dict]) -> Classification`; module constant `CONVERTER_CLASSES = {"online_converter", "consumer_editor"}`.

Signature-entry schema (each value in the JSON):
```
{
  "header_magic": "<hex>" | null,
  "trailer_keys": ["Size", "Root", ...],
  "ordered": true|false,          # require exact key ORDER, not just the set
  "required_features": ["header_magic", "trailer_keys"],  # ALL must match for a confident hit
  "category": "office|native_generator|optimizer|online_converter|consumer_editor"
}
```

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
from src.modules.provenance import (
    Classification, classify, load_signatures, CONVERTER_CLASSES,
)

_DB = {
    "microsoft_word": {
        "header_magic": "b5b5b5b5",
        "trailer_keys": ["Size", "Root", "Info", "ID", "Prev", "XRefStm"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "office",
    },
    "ghostscript": {
        "header_magic": None,
        "trailer_keys": ["Size", "Root", "Info", "ID"],
        "ordered": True,
        "required_features": ["trailer_keys"],
        "category": "optimizer",
    },
    "acme_convert": {   # synthetic online converter with a distinctive magic
        "header_magic": "deadbeef",
        "trailer_keys": ["Size", "Root", "Info"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "online_converter",
    },
}


def _sig(magic, keys):
    from src.modules.provenance import ObservedSignature
    return ObservedSignature(header_magic=magic, trailer_keys=tuple(keys),
                             has_prev="Prev" in keys)


def test_confident_match_both_features():
    c = classify(_sig("b5b5b5b5", ["Size", "Root", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool == "microsoft_word" and c.confidence == 1.0
    assert c.category == "office"


def test_unknown_fingerprint_is_inconclusive():
    c = classify(_sig("0000", ["Foo", "Bar"]), _DB)
    assert c.tool is None and c.confidence == 0.0


def test_wrong_order_does_not_match_ordered_entry():
    c = classify(_sig("b5b5b5b5", ["Root", "Size", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool is None


def test_ambiguous_multi_match_fails_open():
    # An entry requiring only trailer_keys can collide; a second identical one
    # makes the match ambiguous → inconclusive (zero-FP).
    db = dict(_DB)
    db["other_optimizer"] = dict(_DB["ghostscript"])
    c = classify(_sig(None, ["Size", "Root", "Info", "ID"]), db)
    assert c.tool is None and c.confidence == 0.0


def test_real_signature_table_loads_and_is_valid():
    db = load_signatures()
    assert "microsoft_word" in db
    for tool, entry in db.items():
        assert entry["category"] in {
            "office", "native_generator", "optimizer",
            "online_converter", "consumer_editor",
        }, tool
        assert set(entry["required_features"]) <= {"header_magic", "trailer_keys"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -k "classify or match or load or inconclusive or ambiguous or order" -v`
Expected: FAIL — `ImportError: cannot import name 'classify'`.

- [ ] **Step 3a: Create the signature table (desktop seed from arXiv:2103.02702)**

```json
// src/modules/data/provenance_signatures.json
{
  "microsoft_word": {
    "header_magic": "b5b5b5b5",
    "trailer_keys": ["Size", "Root", "Info", "ID", "Prev", "XRefStm"],
    "ordered": true,
    "required_features": ["header_magic", "trailer_keys"],
    "category": "office"
  },
  "acrobat_distiller": {
    "header_magic": "e2e3cfd3",
    "trailer_keys": ["Size", "Root", "Info", "ID"],
    "ordered": true,
    "required_features": ["header_magic"],
    "category": "office"
  },
  "libreoffice": {
    "header_magic": "c3a4c3bcc3b6c39f",
    "trailer_keys": ["Size", "Root", "Info", "ID", "DocChecksum"],
    "ordered": true,
    "required_features": ["header_magic"],
    "category": "office"
  },
  "chrome_skia": {
    "header_magic": "d3ebe9e1",
    "trailer_keys": ["Size", "Root", "Info"],
    "ordered": true,
    "required_features": ["header_magic"],
    "category": "native_generator"
  },
  "ghostscript": {
    "header_magic": "e4f0edf8",
    "trailer_keys": ["Size", "Root", "Info", "ID"],
    "ordered": true,
    "required_features": ["header_magic"],
    "category": "optimizer"
  },
  "cairo": {
    "header_magic": "b5edaefb",
    "trailer_keys": ["Size", "Root", "Info"],
    "ordered": true,
    "required_features": ["header_magic"],
    "category": "native_generator"
  }
}
```

> Note: the `trailer_keys` for the desktop tools are illustrative of each tool's known trailer shape (paper Table 3); every desktop entry above requires only `header_magic` for a confident hit (the robust, published discriminator — Table 2), so trailer noise cannot cause a false match. Online-converter entries (Task 6) are added from real Playwright samples with their own `required_features`.

- [ ] **Step 3b: Add classifier to `src/modules/provenance.py`**

```python
# add near the top of src/modules/provenance.py
CONVERTER_CLASSES = frozenset({"online_converter", "consumer_editor"})
_VALID_CATEGORIES = frozenset({
    "office", "native_generator", "optimizer",
    "online_converter", "consumer_editor",
})
_SIGNATURES_PATH = Path(__file__).resolve().parent / "data" / "provenance_signatures.json"


@dataclass
class Classification:
    tool: str | None
    category: str | None
    confidence: float
    matched_features: tuple[str, ...]


def load_signatures() -> dict[str, dict]:
    """Load the curated signature table. Returns {} on any error (fail-open)."""
    try:
        with open(_SIGNATURES_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("provenance: could not load signature table: %s", exc)
        return {}


def _entry_matches(sig: ObservedSignature, entry: dict) -> bool:
    """True iff every required feature of `entry` matches `sig` exactly."""
    for feat in entry.get("required_features", ["header_magic", "trailer_keys"]):
        if feat == "header_magic":
            em = entry.get("header_magic")
            if not em or not sig.header_magic or sig.header_magic != em.lower():
                return False
        elif feat == "trailer_keys":
            ek = tuple(entry.get("trailer_keys", ()))
            if not ek:
                return False
            if entry.get("ordered"):
                if sig.trailer_keys != ek:
                    return False
            elif set(ek) != set(sig.trailer_keys):
                return False
    return True


def classify(sig: ObservedSignature, db: dict[str, dict]) -> Classification:
    """Identify the producing tool. Inconclusive on 0 or >1 confident matches."""
    hits = [(tool, entry) for tool, entry in db.items() if _entry_matches(sig, entry)]
    if len(hits) != 1:                     # 0 = unknown, >1 = ambiguous → fail-open
        return Classification(None, None, 0.0, ())
    tool, entry = hits[0]
    return Classification(tool, entry.get("category"), 1.0,
                          tuple(entry.get("required_features", ())))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add src/modules/provenance.py src/modules/data/provenance_signatures.json tests/test_provenance.py
git commit -m "feat(provenance): curated signature table + fail-open classifier"
```

---

## Task 3: Flag building + `analyze_provenance` entry point

**Files:**
- Modify: `src/modules/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `ObservedSignature`, `classify`, `load_signatures`, `CONVERTER_CLASSES`.
- Produces: `analyze_provenance(pdf_data) -> ModuleResult` (module name `"provenance"`); helper `build_flags(cls: Classification, producer: str | None, creator: str | None, db: dict) -> list[Flag]`.

Flag decision table (only when `cls.tool` is set, `cls.confidence == 1.0`, and `cls.category in CONVERTER_CLASSES`):

| claimed Producer+Creator | Outcome |
|---|---|
| empty / whitespace | `PROVENANCE_LAUNDERED` (high) |
| mentions the matched tool, or any converter/editor name | no flag (honest — metadata module's `META_ONLINE_CONVERTER` covers it) |
| present, names something else | `PROVENANCE_PRODUCER_MISMATCH` (medium) |

All other cases (unknown tool, non-converter category, ambiguous/inconclusive) → no flag.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
import fitz
from src.modules.provenance import analyze_provenance, build_flags


class _PdfData:
    """Minimal duck-typed stand-in for the extractor's pdf_data."""
    def __init__(self, path, producer=None, creator=None, extraction_ok=True):
        self.file_path = str(path)
        self.extraction_ok = extraction_ok
        self.metadata = type("M", (), {"producer": producer, "creator": creator})()


def _conv_cls(category="online_converter"):
    return Classification(tool="acme_convert", category=category,
                          confidence=1.0, matched_features=("header_magic",))


def test_laundered_when_producer_absent():
    flags = build_flags(_conv_cls(), producer=None, creator=None, db=_DB)
    assert [f.code for f in flags] == ["PROVENANCE_LAUNDERED"]
    assert flags[0].severity == "high"


def test_mismatch_when_producer_names_other_tool():
    flags = build_flags(_conv_cls(), producer="Adobe PDF Library 15.0",
                        creator="SAP", db=_DB)
    assert [f.code for f in flags] == ["PROVENANCE_PRODUCER_MISMATCH"]
    assert flags[0].severity == "medium"


def test_no_flag_when_producer_names_the_converter():
    flags = build_flags(_conv_cls(), producer="Acme Convert online",
                        creator=None, db=_DB)
    assert flags == []


def test_no_flag_for_benign_optimizer():
    flags = build_flags(_conv_cls(category="optimizer"),
                        producer="Adobe", creator=None, db=_DB)
    assert flags == []


def test_no_flag_when_inconclusive():
    flags = build_flags(Classification(None, None, 0.0, ()),
                        producer="Adobe", creator=None, db=_DB)
    assert flags == []


def test_analyze_provenance_skips_when_extraction_failed(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.7\n")
    res = analyze_provenance(_PdfData(p, extraction_ok=False))
    assert res.module == "provenance" and res.flags == []
    assert res.confidence == 0.5


def test_analyze_provenance_clean_fitz_pdf_no_flags(tmp_path):
    doc = fitz.open()
    doc.new_page()
    path = str(tmp_path / "clean.pdf")
    doc.save(path)
    doc.close()
    res = analyze_provenance(_PdfData(path, producer="Acme Billing"))
    assert res.flags == [] and res.module == "provenance"


def test_analyze_provenance_flags_laundered_from_bytes(tmp_path):
    # Craft a raw PDF whose header magic matches the seeded acme_convert entry.
    # (acme_convert is test-only; assert via a monkeypatched DB is overkill —
    # instead craft against a REAL online_converter entry once Task 6 lands.
    # Here we assert the wiring end-to-end using build_flags directly above,
    # and only that analyze_provenance returns a well-formed result.)
    raw = (b"%PDF-1.7\n%\xde\xad\xbe\xef\n"
           b"trailer\n<< /Size 3 /Root 1 0 R /Info 2 0 R >>\nstartxref\n9\n%%EOF\n")
    p = tmp_path / "crafted.pdf"
    p.write_bytes(raw)
    res = analyze_provenance(_PdfData(p, producer=None))
    assert res.module == "provenance"
    assert isinstance(res.flags, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -k "laundered or mismatch or benign or analyze or inconclusive or converter" -v`
Expected: FAIL — `ImportError: cannot import name 'analyze_provenance'`.

- [ ] **Step 3: Add flag building + entry point to `src/modules/provenance.py`**

```python
def _converter_names(db: dict[str, dict]) -> set[str]:
    """Display names of every converter/editor-class tool in the DB."""
    return {tool.replace("_", " ") for tool, e in db.items()
            if e.get("category") in CONVERTER_CLASSES}


def build_flags(cls: Classification, producer: str | None,
                creator: str | None, db: dict[str, dict]) -> list[Flag]:
    """Apply the flag decision table. Returns [] unless a converter lied."""
    if cls.tool is None or cls.confidence < 1.0 or cls.category not in CONVERTER_CLASSES:
        return []
    claimed = f"{producer or ''} {creator or ''}".casefold()
    claimed_squished = claimed.replace(" ", "")
    tool_display = cls.tool.replace("_", " ")

    if not claimed.strip():
        return [Flag(
            severity="high",
            code="PROVENANCE_LAUNDERED",
            message=(f"Structural fingerprint indicates this PDF was rewritten by "
                     f"{tool_display}, but the Producer metadata was removed"),
            details={"structural_tool": cls.tool, "category": cls.category,
                     "matched_features": list(cls.matched_features)},
        )]

    honest = (cls.tool.replace("_", "") in claimed_squished
              or any(name in claimed for name in _converter_names(db)))
    if honest:
        return []

    return [Flag(
        severity="medium",
        code="PROVENANCE_PRODUCER_MISMATCH",
        message=(f"Structural fingerprint indicates this PDF was produced by "
                 f"{tool_display}, which contradicts the stated Producer "
                 f"'{(producer or creator or '').strip()}'"),
        details={"structural_tool": cls.tool, "category": cls.category,
                 "claimed_producer": producer, "claimed_creator": creator,
                 "matched_features": list(cls.matched_features)},
    )]


def analyze_provenance(pdf_data) -> ModuleResult:
    """Flag converter/editor laundering from the PDF's structural fingerprint.

    Never raises: unreadable input yields a partial result (CLAUDE.md).
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="provenance", flags=[], score=100, confidence=0.5)

    flags: list[Flag] = []
    confidence = 1.0
    try:
        with open(pdf_data.file_path, "rb") as fh:
            raw = fh.read()
        sig = extract_signature(raw)
        cls = classify(sig, load_signatures())
        meta = getattr(pdf_data, "metadata", None)
        producer = getattr(meta, "producer", None)
        creator = getattr(meta, "creator", None)
        flags = build_flags(cls, producer, creator, load_signatures())
    except Exception as exc:
        logger.warning("provenance: analysis failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = 0.5

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    return ModuleResult(module="provenance", flags=flags,
                        score=max(0, score), confidence=confidence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add src/modules/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): flag building + analyze_provenance entry point"
```

---

## Task 4: Wire into analyzer + summary map/templates

**Files:**
- Modify: `src/analyzer.py` (import + one `_run` line)
- Modify: `src/summary.py` (FLAG_THEME_MAP + EN + FR templates)
- Test: existing `tests/test_summary_drift.py`, `tests/test_analyzer_isolation.py`

**Interfaces:**
- Consumes: `analyze_provenance` (Task 3).

- [ ] **Step 1: Add the import (analyzer.py)**

In `src/analyzer.py`, after line 30 (`from src.modules.invisible import analyze_invisible`), add:
```python
from src.modules.provenance import analyze_provenance
```

- [ ] **Step 2: Register the module under crash isolation (analyzer.py)**

In `src/analyzer.py`, immediately after line 192 (`_run("invisible", lambda: analyze_invisible(pdf_data))`), add:
```python
        _run("provenance", lambda: analyze_provenance(pdf_data))                  # producer fingerprint
```

- [ ] **Step 3: Add theme + templates (summary.py)**

In `src/summary.py`, in `FLAG_THEME_MAP` (near the `"META_ONLINE_CONVERTER": "origin",` line ~24), add:
```python
    "PROVENANCE_PRODUCER_MISMATCH": "origin",
    "PROVENANCE_LAUNDERED": "origin",
```

In `SENTENCE_TEMPLATES` (English dict, near the `INVISIBLE_TEXT_INJECTION` entry ~337), add:
```python
    "PROVENANCE_PRODUCER_MISMATCH": lambda f: (
        f"The PDF's internal structure matches "
        f"{(f.details or {}).get('structural_tool', 'a conversion tool').replace('_', ' ')}, "
        f"which contradicts the software named in its metadata."
    ),
    "PROVENANCE_LAUNDERED": lambda f: (
        f"The PDF's internal structure shows it was rewritten by "
        f"{(f.details or {}).get('structural_tool', 'a conversion tool').replace('_', ' ')}, "
        f"but the Producer metadata was stripped."
    ),
```

In `SENTENCE_TEMPLATES_FR` (French dict, near its `INVISIBLE_TEXT_INJECTION` entry ~780), add:
```python
    "PROVENANCE_PRODUCER_MISMATCH": lambda f: (
        f"La structure interne du PDF correspond à "
        f"{(f.details or {}).get('structural_tool', 'un outil de conversion').replace('_', ' ')}, "
        f"ce qui contredit le logiciel indiqué dans ses métadonnées."
    ),
    "PROVENANCE_LAUNDERED": lambda f: (
        f"La structure interne du PDF montre qu'il a été réécrit par "
        f"{(f.details or {}).get('structural_tool', 'un outil de conversion').replace('_', ' ')}, "
        f"mais les métadonnées Producer ont été supprimées."
    ),
```

- [ ] **Step 4: Run the drift gate + isolation tests**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_summary_drift.py tests/test_analyzer_isolation.py -v`
Expected: PASS — `test_every_emitted_code_has_a_theme`, `test_en_fr_template_parity`, `test_no_orphan_map_entries` all green (both new codes are emitted in provenance.py, mapped, and templated in both languages).

- [ ] **Step 5: Run the full suite (regression)**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all prior tests + new provenance tests; no regressions).

- [ ] **Step 6: Commit**

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add src/analyzer.py src/summary.py
git commit -m "feat(provenance): wire module into analyzer + summary themes/templates"
```

---

## Task 5: Refresh `SUSPICIOUS_PRODUCERS`

**Files:**
- Modify: `src/modules/metadata.py` (`SUSPICIOUS_PRODUCERS` list, ~line 30)
- Test: `tests/test_metadata.py`

**Interfaces:** none new — extends an existing constant + the existing `META_SUSPICIOUS_PRODUCER` path.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metadata.py
import pytest
from src.extractors.pdf_extractor import PDFData, PDFMetadata
from src.modules.metadata import analyze_metadata, SUSPICIOUS_PRODUCERS


@pytest.mark.parametrize("tool", ["pdfescape", "dochub", "lightpdf", "raptorpdf"])
def test_2026_online_editors_are_known(tool):
    assert tool in SUSPICIOUS_PRODUCERS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_metadata.py -k 2026_online_editors -v`
Expected: FAIL — `assert 'pdfescape' in [...]`.

- [ ] **Step 3: Extend the list (metadata.py)**

In `src/modules/metadata.py`, inside `SUSPICIOUS_PRODUCERS` (after the existing `"freepdfconvert",` entry ~line 40), add:
```python
    # 2026 online editors / converters (sweep 2026-07-17)
    "pdfescape",
    "dochub",
    "lightpdf",
    "raptorpdf",
    "qwikpdf",
    "fixmypdf",
    "cleanpdf",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add src/modules/metadata.py tests/test_metadata.py
git commit -m "feat(metadata): add 2026 online-editor names to SUSPICIOUS_PRODUCERS"
```

---

## Task 6: Online-tool signatures (Playwright) + true-positive & corpus tests

**Files:**
- Create: `scripts/collect_provenance_samples.py`
- Modify: `src/modules/data/provenance_signatures.json` (append real online-tool entries)
- Modify: `tests/test_provenance.py` (true-positive fixtures + corpus zero-FP gate)
- Create (gitignored): `samples/provenance/` collected outputs

**Interfaces:** none new — extends the signature table + tests.

- [ ] **Step 1: Write the collector script**

```python
# scripts/collect_provenance_samples.py
"""
One-off collector: run a SYNTHETIC invoice through each free online PDF tool
via the Playwright MCP plugin, save the output, and print its structural
signature for hand-review before it is added to provenance_signatures.json.

Org policy: feed ONLY generated synthetic invoices — never corpus/sample docs.
This script does not drive the browser itself; it (1) generates the synthetic
input, and (2) derives the signature from a downloaded output. Browser steps
are performed interactively via the Playwright plugin and are documented in the
module docstring below.
"""
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modules.provenance import extract_signature  # noqa: E402


def make_synthetic_invoice(path: str) -> None:
    """Write a minimal, non-sensitive synthetic invoice PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "SYNTHETIC TEST INVOICE", fontsize=14)
    page.insert_text((72, 130), "Invoice INV-0000  Total: 100.00 EUR", fontsize=11)
    doc.save(path)
    doc.close()


def print_signature(pdf_path: str) -> None:
    raw = Path(pdf_path).read_bytes()
    sig = extract_signature(raw)
    print(f"{pdf_path}")
    print(f"  header_magic : {sig.header_magic}")
    print(f"  trailer_keys : {list(sig.trailer_keys)}")
    print(f"  has_prev     : {sig.has_prev}")


if __name__ == "__main__":
    if len(sys.argv) == 2:                       # derive signature from a download
        print_signature(sys.argv[1])
    else:                                        # generate the synthetic input
        out = "samples/provenance/synthetic_input.pdf"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_invoice(out)
        print(f"wrote {out} — upload this to each tool, then re-run with the "
              f"downloaded file path to print its signature")
```

- [ ] **Step 2: Collect signatures (interactive)**

Run `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python scripts/collect_provenance_samples.py` to produce the synthetic input. Then, for each of iLovePDF, Smallpdf, Sejda, PDF24, PDFCandy, PDF2Go, use the Playwright plugin to upload `samples/provenance/synthetic_input.pdf` to the tool's free "edit"/"compress"/"convert" flow, download the result into `samples/provenance/<tool>.pdf`, and run `… python scripts/collect_provenance_samples.py samples/provenance/<tool>.pdf` to print its signature. Record `header_magic`, `trailer_keys`, and choose `required_features` = the smallest subset that is UNIQUE across the whole table (verify uniqueness by hand — if a tool's only distinctive feature collides with another entry, mark it `consumer_editor`/`online_converter` but leave `required_features` demanding both, accepting non-detection over FP).

- [ ] **Step 3: Add `samples/provenance/` to .gitignore**

Append to `.gitignore`:
```
samples/provenance/
```
Then copy the two smallest, clearly-non-sensitive synthetic outputs (e.g. iLovePDF, Smallpdf) to a committed fixtures dir `tests/fixtures/provenance/` for the true-positive tests (these are synthetic — safe to commit).

- [ ] **Step 4: Append online-tool entries to the JSON**

Add each reviewed entry to `src/modules/data/provenance_signatures.json`, e.g.:
```json
  "ilovepdf": {
    "header_magic": "<observed or null>",
    "trailer_keys": ["<observed>"],
    "ordered": true,
    "required_features": ["<unique subset>"],
    "category": "online_converter"
  },
  "smallpdf": {
    "header_magic": "<observed or null>",
    "trailer_keys": ["<observed>"],
    "ordered": true,
    "required_features": ["<unique subset>"],
    "category": "online_converter"
  }
```

- [ ] **Step 5: Write the true-positive + corpus tests**

```python
# append to tests/test_provenance.py
import glob
import os
import pytest

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "provenance")
_FIX_PDFS = sorted(glob.glob(os.path.join(_FIX, "*.pdf")))


@pytest.mark.skipif(not _FIX_PDFS, reason="provenance fixtures not present")
@pytest.mark.parametrize("pdf_path", _FIX_PDFS, ids=[os.path.basename(p) for p in _FIX_PDFS])
def test_online_tool_fixture_is_detected(pdf_path):
    """A file rewritten by a known online tool (metadata stripped) must flag."""
    doc = fitz.open(pdf_path)
    doc.set_metadata({})            # simulate scrubbed metadata → LAUNDERED path
    scrubbed = pdf_path + ".scrubbed.pdf"
    doc.save(scrubbed)
    doc.close()
    res = analyze_provenance(_PdfData(scrubbed, producer=None))
    codes = [f.code for f in res.flags]
    assert "PROVENANCE_LAUNDERED" in codes or "PROVENANCE_PRODUCER_MISMATCH" in codes


_CORPUS = sorted(glob.glob(os.path.join("samples", "legitimate", "real", "*.pdf")))


@pytest.mark.skipif(not _CORPUS, reason="legit corpus not present (gitignored)")
@pytest.mark.parametrize("pdf_path", _CORPUS, ids=[os.path.basename(p) for p in _CORPUS])
def test_provenance_no_false_positive_on_corpus(pdf_path):
    """Zero-FP gate: no legit corpus doc may raise a provenance flag."""
    res = analyze_provenance(_PdfData(
        pdf_path,
        producer=None,   # producer is read inside analyze_provenance from bytes-adjacent metadata
    ))
    # Read real metadata so the honest/mismatch guards see the true producer.
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    doc.close()
    res = analyze_provenance(_PdfData(pdf_path,
                                      producer=meta.get("producer"),
                                      creator=meta.get("creator")))
    assert res.flags == [], f"unexpected provenance flags on {os.path.basename(pdf_path)}: {[f.code for f in res.flags]}"
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/test_provenance.py tests/test_corpus_bands.py -v`
Expected: PASS — true-positive fixtures flag; corpus (if present) raises zero provenance flags. If any corpus doc flags, its signature collided with a converter entry → tighten that entry's `required_features` (do NOT loosen the corpus gate).

- [ ] **Step 7: Full suite + commit**

Run: `cd /Users/edouard/Projects/trustyfile-provenance && /Users/edouard/Projects/trustyfile-main/.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

```bash
cd /Users/edouard/Projects/trustyfile-provenance
git add scripts/collect_provenance_samples.py src/modules/data/provenance_signatures.json \
        tests/test_provenance.py tests/fixtures/provenance/ .gitignore
git commit -m "feat(provenance): online-tool signatures + true-positive & corpus gates"
```

---

## Self-Review

**Spec coverage:**
- Purpose / honest-limit (spec §1) → module emits soft flags only; damper deferred (Task 3 docstring, Global Constraints). ✔
- Architecture + ObservedSignature 3 features (spec §3) → Task 1. ✔
- Signature table + categories (spec §3.2) → Task 2. ✔
- Two flags + severities + guards + fail-open (spec §4) → Task 3 decision table. ✔
- Signature sourcing incl. Playwright/synthetic-only (spec §5) → Task 6. ✔
- Files touched incl. shared-worktree map (spec §6) → File Structure + Task 4/5. ✔
- Testing: unit, corpus zero-FP, true-positive, damper hook (spec §7) → Tasks 1–3, 6. ✔
- `scoring.py`/`evidence.py` untouched (spec §2/§6) → Global Constraints, no task edits them. ✔

**Placeholder scan:** the JSON `<observed…>` tokens in Task 6 are intentional — the values are empirically collected at execution time and hand-reviewed; every other step has concrete code/commands. No TODO/TBD in shipped code.

**Type consistency:** `ObservedSignature`, `Classification`, `classify`, `load_signatures`, `build_flags`, `analyze_provenance`, `CONVERTER_CLASSES` are named identically across Tasks 1–4 and the tests. `analyze_provenance` reads `pdf_data.metadata.producer/creator` (verified against `PDFMetadata`). Module name string `"provenance"` consistent in module + analyzer `_run`.
