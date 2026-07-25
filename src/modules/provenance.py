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

CONVERTER_CLASSES = frozenset({"online_converter", "consumer_editor"})
_VALID_CATEGORIES = frozenset({
    "office", "native_generator", "optimizer",
    "online_converter", "consumer_editor",
})
_SIGNATURES_PATH = Path(__file__).resolve().parent / "data" / "provenance_signatures.json"


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


@dataclass
class Classification:
    """Result of matching an ObservedSignature against the signature table."""
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
    try:
        hits = [(tool, entry) for tool, entry in db.items() if _entry_matches(sig, entry)]
        if len(hits) != 1:                     # 0 = unknown, >1 = ambiguous → fail-open
            return Classification(None, None, 0.0, ())
        tool, entry = hits[0]
        return Classification(tool, entry.get("category"), 1.0,
                              tuple(entry.get("required_features", ())))
    except Exception as exc:  # defensive: classification must never raise
        logger.warning("provenance: classification failed: %s", exc)
        return Classification(None, None, 0.0, ())


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
