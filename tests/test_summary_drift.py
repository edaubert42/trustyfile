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
