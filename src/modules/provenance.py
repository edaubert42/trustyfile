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
