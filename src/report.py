"""
Machine-readable analysis report (roadmap #8, tier 2).

`build_report` turns one AnalysisResult into a plain dict that `json.dumps`
accepts, mirroring what the results page shows: band, score and range,
per-module scores, every flag with the sentence the reader saw, and the FR/EN
verdict. It is the file the results page offers for download and the payload
the API endpoint (roadmap #13) returns, so the shape is versioned
(SCHEMA_VERSION) and pinned by tests/test_report.py instead of being inferred
from the UI.

Deliberately NOT in the report:
- ``ModuleResult.confidence`` — the evidence ledger pins confidence to 1.0,
  so the field is decorative; the results page dropped it for that reason.
- The evidence ledger — internal scoring weights. Freezing them into a public
  schema would tie every retuning to a schema bump (owner decision, 2026-09-05).
- ``ModuleResult.details`` — working data the modules keep for each other
  (MRZ lines, verified companies, text excerpts), not findings.

Why the walker (`_json_safe`): ``Flag.details`` is a free-form dict filled by
thirteen modules. It holds ``tuple(fitz.Rect)`` boxes, numpy scalars and
arrays, sometimes bytes or sets — all of which make ``json.dumps`` raise
``TypeError``. A report must exist for every result the app can display, so
unknown values degrade to text rather than fail the download (same posture as
"never crash on a malformed PDF").

Example:
    >>> report = build_report(result, filename="invoice.pdf",
    ...                       analysed_at=datetime(2026, 9, 5, 10, 30), lang="fr")
    >>> report["band"], report["trust_score"], report["file"]["name"]
    ('SUSPICIOUS', 42, 'invoice.pdf')
    >>> report_json(report)[:22]
    '{\\n  "schema_version": 1'
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np

from src.models import AnalysisResult, AnalysisSummary, Flag, ModuleResult
from src.summary import generate_rich_summary, get_flag_sentence

# Bump when a field is renamed, removed or changes meaning. Adding a field is
# backwards compatible and does not need a bump.
SCHEMA_VERSION = 1

GENERATOR = {"name": "TrustyFile", "url": "https://www.trustyfile.com"}


def _json_safe(value: Any) -> Any:
    """
    Return `value` rebuilt out of types `json.dumps` accepts.

    Plain JSON values pass through. Everything else is converted rather than
    rejected, because the report is produced from module output the report
    layer does not control:

    - tuple / set / frozenset / any other iterable → list (sets sorted, so the
      output is deterministic)
    - numpy scalar → the Python number it wraps; numpy array → nested lists
    - NaN / ±Infinity → None (json.dumps would write ``NaN``, which is not JSON)
    - bytes → hex string
    - datetime / date → ISO 8601 text; Enum → its value
    - dataclass → dict (then walked)
    - dict keys → str (JSON object keys must be strings)
    - anything else → ``str(value)``

    Example:
        >>> _json_safe({"bbox": (1, 2), "n": np.float64(0.5), "raw": b"\\xff"})
        {'bbox': [1, 2], 'n': 0.5, 'raw': 'ff'}
    """
    # numpy first: np.float64 IS a float subclass and would otherwise pass
    # through the float branch untouched (json.dumps accepts it, but the
    # contract is plain Python types); np.int64 / np.bool_ are not int/bool
    # subclasses at all. `.item()` unwraps any numpy scalar to its Python twin.
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    # bool is a subclass of int; both are fine as-is.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(v) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Other iterables become lists too. `iter()` rather than hasattr(__iter__):
    # fitz.Rect is iterable through the old sequence protocol (__getitem__ +
    # __len__) and has no __iter__ at all.
    try:
        iterator = iter(value)
    except TypeError:
        return str(value)
    try:
        return [_json_safe(v) for v in iterator]
    except Exception:  # an object whose iteration fails is just opaque
        return str(value)


def _flag_to_dict(flag: Flag, lang: str) -> dict[str, Any]:
    """One flag as the reader saw it, plus the stable code and raw message."""
    return {
        "severity": flag.severity,
        "code": flag.code,
        # The localised sentence shown in the results page; `message` is the
        # module's English text and stays for consumers that key off it.
        "sentence": get_flag_sentence(flag, lang),
        "message": flag.message,
        "details": _json_safe(flag.details) if flag.details is not None else None,
    }


def _module_to_dict(module: ModuleResult, lang: str) -> dict[str, Any]:
    """Name, the NN/100 pill shown on screen, and the module's flags."""
    return {
        "name": module.module,
        "score": module.score,
        "flags": [_flag_to_dict(f, lang) for f in module.flags],
    }


def build_report(
    result: AnalysisResult,
    *,
    filename: str,
    analysed_at: datetime | str,
    lang: str = "en",
    summary: AnalysisSummary | None = None,
) -> dict[str, Any]:
    """
    Build the versioned report dict for one analysed document.

    Args:
        result: The analysis to report on.
        filename: The uploaded file's name (the result only knows the hash).
        analysed_at: When the analysis ran — a datetime (written as ISO 8601)
            or the preformatted string app.py keeps in session_state.
        lang: "fr" or "en"; drives the flag sentences and the summary.
        summary: The AnalysisSummary the app already computed. Computed here
            from `result` when not given.

    Returns:
        A dict containing only JSON-encodable values. For an UNVERIFIABLE
        band `trust_score` and `score_range` are None: internally the coverage
        gate stores 0, but a consumer would read 0 as "certainly forged", so
        the JSON says the same thing the on-screen "—" does.

    Example:
        >>> build_report(result, filename="a.pdf", analysed_at="2026-09-05 10:30")["file"]
        {'name': 'a.pdf', 'sha256': 'ab12…'}
    """
    if summary is None:
        summary = generate_rich_summary(result, lang=lang)

    unverifiable = result.band == "UNVERIFIABLE"

    report = {
        "schema_version": SCHEMA_VERSION,
        "generator": dict(GENERATOR),
        "file": {"name": filename, "sha256": result.file_hash},
        "analysed_at": (
            analysed_at.isoformat(timespec="seconds")
            if isinstance(analysed_at, datetime)
            else analysed_at
        ),
        "language": lang,
        "band": result.band,
        "risk_level": result.risk_level,
        "trust_score": None if unverifiable else result.trust_score,
        "score_range": (
            None if unverifiable else {"low": result.score_low, "high": result.score_high}
        ),
        "document_type": result.document_type,
        "analysis_time_ms": result.analysis_time_ms,
        "summary": {"verdict": summary.verdict, "bullets": list(summary.bullets)},
        "modules": [_module_to_dict(m, lang) for m in result.modules],
    }
    # One last pass over the whole tree: the fields above are plain already,
    # but this is what guarantees the contract "report_json never raises".
    return _json_safe(report)


def report_json(report: dict[str, Any]) -> str:
    """
    Serialise a report dict to the text that is downloaded.

    Indented so a human can read it; `ensure_ascii=False` keeps accents as
    accents rather than ``\\u00e9``; `allow_nan=False` turns any NaN that
    escaped `_json_safe` into an exception here instead of an unparsable file.
    Ends with a newline like any text file.
    """
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def report_filename(result: AnalysisResult) -> str:
    """
    ``trustyfile-<first 12 hex chars of the hash>.json``.

    The prefix ties the file to the document without exposing the full hash in
    a file name; two reports of the same document get the same name.
    """
    prefix = result.file_hash[:12]
    return f"trustyfile-{prefix}.json" if prefix else "trustyfile-report.json"
