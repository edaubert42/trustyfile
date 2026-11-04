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
# Tools capable of the cover-a-value-and-retype edit. Presence of one of these
# is the C2 corroborator that upgrades a geometry-only finding (SOFT/REVIEW) to a
# HARD tamper verdict. HUMAN DECISION (whole-branch review): LibreOffice STAYS on
# this list even though it is also an authoring tool — LibreOffice Draw is a real
# fraud vector (open a finished PDF, box a value, retype). This does NOT endanger
# legit LibreOffice-authored invoices: the HARD path still requires the
# cover-and-retype GEOMETRY to be present first, which a from-scratch invoice
# never has, so the geometry gate filters legit authoring before provenance is
# ever consulted.
INTERACTIVE_EDITOR_PRODUCERS = (
    "acrobat", "foxit", "pdf-xchange", "pdfxchange",
    "nitro", "sejda", "pdfescape", "master pdf",
)


def is_interactive_editor(haystack: str) -> bool:
    """Was this file last saved by a tool that EDITS existing PDFs? (C2)

    `haystack` is the Producer and Creator joined ("LibreOffice 24.2 Writer").
    This is the corroborator that lets a geometry finding become HARD, so it
    must name editing tools, not authoring tools.

    LibreOffice is the subtle one: Writer, Calc and Impress CREATE documents
    and export them to PDF — a genuine invoice typed in Writer says
    "LibreOffice ... Writer". Only Draw opens an existing PDF to modify it.
    Listing plain "libreoffice" made every Writer export look "edited", so a
    single heuristic geometry hit on such a file became a HARD accusation.

    Examples:
        >>> is_interactive_editor("adobe acrobat pro dc 23.1")
        True
        >>> is_interactive_editor("libreoffice 24.2 draw")
        True
        >>> is_interactive_editor("libreoffice 24.2 writer")   # authoring
        False
    """
    h = haystack.casefold()
    if any(p in h for p in INTERACTIVE_EDITOR_PRODUCERS):
        return True
    return "libreoffice" in h and "draw" in h


def span_text(span: dict) -> str:
    out = []
    for c in span.get("chars", []):
        try:
            out.append(chr(c[0]))
        except (TypeError, ValueError):
            pass
    return "".join(out)


def _has_combining_mark(s: str) -> bool:
    return any(unicodedata.category(ch) in ("Mn", "Mc") for ch in s)


# Currency symbols and their base-14-font renderings, treated as separators so a
# monetary amount classifies as "numeric" regardless of an attached symbol. Note
# base-14 Helvetica renders € as U+00B7 (middle dot) and ¥ can render oddly, so
# both the real glyphs and their substitutes are stripped. (Same U+00B7 quirk the
# placeholder guard already accounts for.)
_CURRENCY_STRIP = "€$£¥·.,-"


def char_class(s: str) -> str:
    stripped = "".join(ch for ch in s if not ch.isspace() and ch not in _CURRENCY_STRIP)
    if _has_combining_mark(s) or s != unicodedata.normalize("NFC", s):
        return "excluded"
    if not stripped:
        return "excluded"
    if all(ch.isdigit() for ch in stripped):
        return "numeric"
    if all(ch.isalpha() for ch in stripped):
        return "alpha"
    return "mixed"


def norm(s: str) -> str:
    # Collapse ASCII whitespace via split(), then explicitly strip the no-break
    # spaces PDFs use as thousands separators: U+202F (narrow no-break space)
    # and U+00A0 (no-break space). Critical for French amounts, whose thousands
    # separator is typically U+202F, not a plain space -- matching "1 200,00"
    # against "1200,00" must not depend on split() incidentally handling
    # Unicode whitespace (a future refactor could silently break it).
    return "".join(s.split()).replace("\u202f", "").replace("\xa0", "").casefold()


# A monetary amount embedded anywhere in a span's text ("571,04" in
# "Gesamtbetrag: 571,04 \u20ac"). The (?<!\d) prefix and the (?![.,]?\d) suffix
# together stop a fragment of a longer dotted run from matching: the suffix
# rejects a 2-decimal group that is itself followed by "<sep><digit>", so a
# European date "10.01.2020" no longer yields the money-looking "10.01" (the
# decimals "01" are followed by ".2"). A real amount at a sentence end
# ("1.200,00.") is unaffected \u2014 its "." is followed by a space/end, not a digit.
# Used only to compare the amount portion of two spans in the embedded-span
# cover-and-retype paths (Signals 3 and 5) \u2014 never for classification.
_AMOUNT_RE = re.compile(r"(?<!\d)\d[\d\s.,]*[.,]\d{2}(?![.,]?\d)")


def amount_token(text: str) -> str | None:
    """The first monetary amount substring in `text`, or None.

    Lets the embedded-span path compare just the number when the original value
    is wrapped in a label span ("Gesamtbetrag: 571,04 \u20ac" -> "571,04").
    """
    m = _AMOUNT_RE.search(text or "")
    return m.group() if m else None


# =============================================================================
# AMOUNT PARSING  (for the human-readable delta in the cover-and-retype message)
# =============================================================================
# Money-like run: a digit then digits / whitespace / separators. \s already
# covers the no-break (U+00A0) and narrow no-break (U+202F) spaces PDFs use
# as thousands separators, which money_value() strips before parsing.
_MONEY_RE = re.compile(r"\d[\d\s.,]*")


def money_value(text: str) -> float | None:
    """Parse the numeric amount out of a span's text, or None if not money-like.

    Handles the French convention (space/U+202F thousands separator, comma
    decimal) as well as the anglo convention (comma thousands, dot decimal).
    Returns a float so callers can compute a delta between two amounts.
    """
    m = _MONEY_RE.search(text or "")
    if not m:
        return None
    raw = m.group(0)
    # Drop every kind of thousands whitespace separator.
    s = raw.replace(" ", "").replace(" ", "").replace("\xa0", "")
    if "," in s and "." in s:
        # Both present: the LAST separator is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # French: 1.200,00
        else:
            s = s.replace(",", "")                      # Anglo: 1,200.00
    elif "," in s:
        s = s.replace(",", ".")                         # French decimal comma
    try:
        return float(s)
    except ValueError:
        return None


def is_horizontal(span_dir, page_angle_deg: float) -> bool:
    import math
    dx, dy = span_dir
    ang = math.degrees(math.atan2(dy, dx))          # span text angle
    # normalize both angles to [0,360) and compare to the page baseline (page_angle_deg)
    diff = (ang - page_angle_deg) % 360.0
    diff = min(diff, 360.0 - diff)
    return diff <= HORIZONTAL_TOL_DEG


# =============================================================================
# PLACEHOLDER GUARD  (the single most FP-sensitive piece of Phase 2)
# =============================================================================
# A legitimate template / mail-merge engine produces the EXACT structural
# signature of the fraud: it paints a placeholder ("0,00", "XXX"), draws a white
# box over it, then stamps the real value on top — so the paint order
# (placeholder seqno < box seqno < value seqno) is identical to a cover-and-
# retype forgery. The ONLY reliable discriminator is WHAT gets covered:
#   - fraud   covers a REAL value  (e.g. "1 200,00") -> tampering
#   - template covers a PLACEHOLDER (e.g. "0,00")     -> benign scaffolding
# So Signals 1 and 3 must NOT fire when the covered/hidden text is a placeholder.

# Textual placeholder tokens (already lower-cased; compared against norm()).
_PLACEHOLDER_TOKENS = {
    "xxx", "x", "x.xx", "x,xx", "n/a", "na", "tbd", "-", "--", "---", "___",
}

# "Zeros and separators only" pattern, applied to the norm()'d string (which has
# already had ALL whitespace, U+202F and U+00A0 removed and been case-folded).
# The currency set includes the real glyphs (€ $ £ ¥) AND U+00B7 (MIDDLE DOT):
# PyMuPDF's base-14 Helvetica decodes an inserted "€" back to U+00B7 via
# get_texttrace, so a "0,00 €" placeholder surfaces as "0,00·". Including U+00B7
# lets the guard recognise that placeholder regardless of font encoding. This
# only ever ADDS currency-symbol tolerance — a real value like "1 200,00" still
# contains the non-zero digits 1 and 2, which are NOT in the class, so it can
# never be mistaken for a placeholder (the fraud/template discriminator holds).
_PLACEHOLDER_ZEROS_RE = re.compile(r"[0.,\s€$£¥·]*")


def is_placeholder(text: str) -> bool:
    """True if `text` is benign template scaffolding rather than a real value.

    A placeholder is one of: empty, a zeros-and-separators string ("0,00",
    "0.00 €", ""), or a known token ("XXX", "N/A", "-", ...). When the COVERED
    text of a cover-and-retype pattern is a placeholder, the pattern is honest
    template generation, not tampering — Signals 1 and 3 stay silent.

    Examples:
        >>> is_placeholder("0,00 €")     # template placeholder
        True
        >>> is_placeholder("XXX")
        True
        >>> is_placeholder("1 200,00 €") # a real value
        False
    """
    n = norm(text)
    if not n:
        return True
    if _PLACEHOLDER_ZEROS_RE.fullmatch(n):
        return True
    return n in _PLACEHOLDER_TOKENS


def page_angle(page) -> float:
    """Baseline text angle for the page, derived from page.rotation.

    get_texttrace() reports each span's `dir` in the page's *rotated* coordinate
    system, so horizontal body text on a 90°/270° page reports dir≈(0,1) — its
    baseline angle is 90°, not 0°. We keep the baseline at 0 for the common
    0/180 case and shift to 90 for the 90/270 landscape rotations, so
    is_horizontal() compares each span against the correct axis. (Getting this
    wrong can only silence the detector on a rotated page — a false negative,
    never a false positive.)
    """
    rot = int(getattr(page, "rotation", 0) or 0) % 360
    return 90.0 if rot in (90, 270) else 0.0


def pair_is_conflict(a: dict, b: dict, page_angle: float,
                     size_frac: float = SIZE_MATCH_FRAC) -> bool:
    """Do two spans hold conflicting versions of the SAME field?

    Shared predicate for Signals 2 and 3: different text, but the same *kind* of
    content (both numeric / both alpha, never punctuation-only), both horizontal,
    and matching size — i.e. one plausibly replaced the other. char_class returns
    "excluded" for accented / combining text so French prose never conflicts.

    `size_frac` is the largest relative font-size difference still read as
    "the same field". The default (SIZE_MATCH_FRAC) is for two texts that
    merely overlap; a caller with stronger geometric evidence tying the two
    texts together (Signal 3: a cover box painted between them) may widen it.
    """
    ta, tb = span_text(a), span_text(b)
    if norm(ta) == norm(tb):
        return False                       # identical text = faux-bold, not a conflict
    ca = char_class(ta)
    if ca == "excluded" or ca != char_class(tb):
        return False
    da, db = a.get("dir", (1.0, 0.0)), b.get("dir", (1.0, 0.0))
    if not (is_horizontal(da, page_angle) and is_horizontal(db, page_angle)):
        return False
    sa, sb = a.get("size", 0.0), b.get("size", 0.0)
    if max(sa, sb) <= 0 or abs(sa - sb) / max(sa, sb) > size_frac:
        return False                       # different font sizes = different fields
    return True
