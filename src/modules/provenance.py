"""
Producer structural-fingerprint detector (Module: provenance) — phase 8.

Identifies the tool that actually produced/rewrote a PDF from its structural
"handwriting" — the header magic bytes and the trailer key signature — which
survive metadata scrubbing. See docs/superpowers/specs/2026-07-17-producer-
fingerprint-design.md.

Two families of flags:
  - converters (build_flags): an online converter's structure under another
    tool's name, or with the Producer removed;
  - rewrite engines (build_rewrite_flags, 2026-11-13): PyMuPDF's or qpdf's
    structure, the one pass that wipes an edit's file traces.

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
_NUM_RE = re.compile(rb"[+\-0-9.]+")
_DIGITS_RE = re.compile(rb"[0-9]+")
_ALPHA_RE = re.compile(rb"[A-Za-z]+")

CONVERTER_CLASSES = frozenset({"online_converter", "consumer_editor"})
_VALID_CATEGORIES = frozenset({
    "office", "native_generator", "optimizer",
    "online_converter", "consumer_editor",
})
_SIGNATURES_PATH = Path(__file__).resolve().parent / "data" / "provenance_signatures.json"

# -----------------------------------------------------------------------------
# Rewrite engines (2026-11-13)
# -----------------------------------------------------------------------------
# Libraries that write a whole PDF again in one pass. They never put their own
# name in the Producer field: the copy keeps the Producer of the file it came
# from. So when the bytes are theirs and the Producer names something else, the
# file was written a second time after the named software produced it. That
# one pass is also how a forger wipes an edit's traces (date gap, XMP packet,
# earlier revisions), see tests/test_corpus_rewrite.py.
#
# The rule reads the bytes, never the Producer's good faith: a file that says
# "PyMuPDF" or names nothing is treated like one under an unknown name. The
# Producer is free text, so an exemption for "honest" names would be a
# one-line way around the rule.
#
# These entries match on the header bytes alone (the converter flags above
# need the trailer too). Measured on the local corpus, 2026-11-13: of 176
# legitimate files, 0 start with PyMuPDF's bytes and 1 with qpdf's (a notice
# exported from Chrome, not an issued document); after a rewrite, 12 of 12
# forged copies do.
#
# Structural tool (key of the signature table) -> name shown to the reader.
REWRITE_ENGINES: dict[str, str] = {"pymupdf": "PyMuPDF", "qpdf": "qpdf"}

# Engines whose rewrite of a KNOWN tool's file counts as a provenance break
# (PROVENANCE_FILE_REWRITTEN, SUSPICIOUS on an issued document). qpdf is left
# out on purpose: it is a routine last step in legitimate pipelines (web
# optimisation, password removal, OCRmyPDF), so it only caps the band at
# REVIEW (PROVENANCE_WRITER_UNVERIFIED).
_BREAKING_REWRITE_ENGINES = frozenset({"pymupdf"})

# Software that writes PDF files with its own writer, never through MuPDF or
# qpdf. A Producer naming one of these on a file laid out by a rewrite engine
# is a contradiction, not a guess. A name outside this list ("Acme Billing
# 2.1") proves nothing: an application built on PyMuPDF may sign its files
# with its own name, and those cannot be told apart from a rewritten copy.
KNOWN_WRITERS: tuple[str, ...] = (
    # Adobe, Microsoft, office suites
    "adobe", "acrobat", "distiller", "microsoft", "libreoffice", "openoffice",
    # Browsers and operating systems
    "skia", "chromium", "quartz", "qt", "wkhtmltopdf", "cairo", "pdfium",
    # Server-side libraries behind billing software
    "itext", "itextsharp", "reportlab", "fop", "pdfbox", "jasperreports",
    "weasyprint", "tcpdf", "fpdf", "dompdf", "mpdf", "pdfkit", "pdflib",
    "pdf-lib", "aspose", "pdfsharp", "migradoc", "openpdf", "pypdf",
    "abcpdf", "windev", "typst", "prince",
    # TeX
    "pdftex", "xetex", "luatex", "latex",
    # Converters and editors (they have their own flags; listed so that the
    # rewrite is seen too)
    "ghostscript", "pdftools", "foxit", "nitro", "canva", "ilovepdf",
    "smallpdf", "sejda",
    # E-invoice toolkits and business suites
    "intarsys", "mustangproject", "sap", "sage", "oracle", "quickbooks", "xero",
)
# One regex, whole words only: "sage" must not match inside "message".
_KNOWN_WRITER_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(w) for w in KNOWN_WRITERS) + r")(?![a-z])"
)


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
    m = _NUM_RE.match(b, i)                          # number or indirect ref
    if m:
        # match(string, pos) on a compiled pattern searches in place from pos
        # -- unlike re.match(pattern, b[i:]), it never copies the remaining
        # buffer, so repeated numeric values can't turn parsing quadratic on
        # a crafted large trailer (adversarial-input hardening).
        i = m.end()
        save = i
        i = _skip_ws(b, i)
        m2 = _DIGITS_RE.match(b, i)
        if m2:
            k = _skip_ws(b, m2.end())
            if b[k:k + 1] == b"R":
                return k + 1
        return save
    m = _ALPHA_RE.match(b, i)                        # true / false / null
    return m.end() if m else i + 1


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


def _known_aliases(cls: Classification, db: dict[str, dict]) -> set[str]:
    """Casefolded producer_aliases declared on the matched entry, if any.

    Some table entries represent a shared structural signature covering
    several real-world tools (e.g. a common backend library) rather than one
    tool with one self-reported name — the entry's key alone (e.g.
    "mupdf_online_converter") never appears in ANY of those tools' own
    Producer strings, so without this a genuinely honest document from one
    of them would be misjudged as a lie. Declared per-entry, not globally,
    so an alias never vouches for a DIFFERENT entry's match.
    """
    entry = db.get(cls.tool, {})
    return {a.casefold() for a in entry.get("producer_aliases", [])}


def build_flags(cls: Classification, producer: str | None,
                creator: str | None, db: dict[str, dict]) -> list[Flag]:
    """Apply the flag decision table. Returns [] unless a converter lied."""
    if cls.tool is None or cls.confidence < 1.0 or cls.category not in CONVERTER_CLASSES:
        return []
    # Spec requires a confident match on BOTH header_magic and trailer_keys
    # before either converter flag may fire — enforced here in code, not
    # left to depend on every future signature-table entry happening to
    # require both features itself. A single-feature match (weaker
    # evidence) must never emit a flag, no matter what the table says.
    if not ({"header_magic", "trailer_keys"} <= set(cls.matched_features)):
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
              or any(name in claimed for name in _converter_names(db))
              or any(alias in claimed for alias in _known_aliases(cls, db)))
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


def _named_known_writer(claimed: str) -> str | None:
    """Return the known PDF writer named in a Producer/Creator string, if any.

    Args:
        claimed: Producer and Creator joined, already casefolded.

    Examples:
        >>> _named_known_writer("adobe pdf library 19.8 acrobat pdfmaker 19")
        'adobe'
        >>> _named_known_writer("qt 4.8.7 wkhtmltopdf 0.12.6")
        'qt'
        >>> _named_known_writer("acme billing 2.1") is None
        True
        >>> _named_known_writer("message router") is None   # "sage" inside a word
        True
    """
    match = _KNOWN_WRITER_RE.search(claimed)
    return match.group(1) if match else None


def build_rewrite_flags(cls: Classification, producer: str | None,
                        creator: str | None) -> list[Flag]:
    """Flag a file whose bytes were laid out by a rewrite engine.

    Decision table, for a file written by a REWRITE_ENGINES tool (PyMuPDF or
    qpdf). Every such file gets exactly one flag:

      PyMuPDF + Producer/Creator names a KNOWN_WRITERS tool
          -> PROVENANCE_FILE_REWRITTEN (high). That tool did not write these
             bytes: the file was written again afterwards.
      anything else
          -> PROVENANCE_WRITER_UNVERIFIED (low). The origin cannot be checked.
             details["reason"] says why:
               "rewritten"     qpdf under a known tool's name: a certain
                               rewrite, by a tool legitimate pipelines also use
               "unknown_name"  a name that cannot be checked against the bytes
                               (an app built on the library, or a rewrite)
               "no_name"       no Producer and no Creator at all

    The band consequences live in src/scoring.py (PROVENANCE_BREAK_CODES and
    UNVERIFIED_ORIGIN_CODES); this function only states what the file shows.

    Args:
        cls: Result of classify() for the file.
        producer: The /Producer metadata string, or None.
        creator: The /Creator metadata string, or None.

    Returns:
        A list holding zero or one Flag.

    Examples:
        >>> cls = Classification("pymupdf", "native_generator", 1.0, ("header_magic",))
        >>> [f.code for f in build_rewrite_flags(cls, "Adobe PDF Library 19.8", None)]
        ['PROVENANCE_FILE_REWRITTEN']
        >>> flag = build_rewrite_flags(cls, "Acme Billing 2.1", None)[0]
        >>> flag.code, flag.details["reason"]
        ('PROVENANCE_WRITER_UNVERIFIED', 'unknown_name')
        >>> word = Classification("microsoft_word", "office", 1.0, ("header_magic",))
        >>> build_rewrite_flags(word, "Microsoft Word", None)
        []
    """
    if cls.tool not in REWRITE_ENGINES or cls.confidence < 1.0:
        return []
    engine = REWRITE_ENGINES[cls.tool]
    claimed = f"{producer or ''} {creator or ''}".casefold().strip()
    stated = (producer or creator or "").strip()   # shown to the reader

    known = _named_known_writer(claimed)
    details = {"structural_tool": cls.tool, "engine": engine,
               "claimed_producer": producer, "claimed_creator": creator,
               "matched_features": list(cls.matched_features)}

    if known and cls.tool in _BREAKING_REWRITE_ENGINES:
        return [Flag(
            severity="high",
            code="PROVENANCE_FILE_REWRITTEN",
            message=(f"The metadata names '{stated}', but the file was written "
                     f"by {engine}: it was written again after it was produced"),
            details={**details, "known_writer": known},
        )]

    if known:
        reason = "rewritten"
    elif claimed:
        reason = "unknown_name"
    else:
        reason = "no_name"
    names = f"names '{stated}'" if stated else "names no software"
    return [Flag(
        severity="low",
        code="PROVENANCE_WRITER_UNVERIFIED",
        message=(f"The file was written by {engine}, a general-purpose PDF "
                 f"library; its metadata {names}, so its origin cannot be verified"),
        details={**details, "reason": reason},
    )]


def analyze_provenance(pdf_data) -> ModuleResult:
    """Flag converter laundering and silent rewrites from the structural fingerprint.

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
        db = load_signatures()
        cls = classify(sig, db)
        meta = getattr(pdf_data, "metadata", None)
        producer = getattr(meta, "producer", None)
        creator = getattr(meta, "creator", None)
        flags = build_flags(cls, producer, creator, db)
        # A different family of tools (categories never overlap), so at
        # most one of the two calls returns a flag.
        flags += build_rewrite_flags(cls, producer, creator)
    except Exception as exc:
        logger.warning("provenance: analysis failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = 0.5

    score = 100
    for f in flags:
        score -= SEVERITY_POINTS.get(f.severity, 0)
    return ModuleResult(module="provenance", flags=flags,
                        score=max(0, score), confidence=confidence)
